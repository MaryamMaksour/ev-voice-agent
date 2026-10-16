"""What the call record now keeps for the dashboard: time to first token, tokens, who ended the call,
the phone line result, recording, transfer to a person, and speech accuracy. No GPU, no LiveKit server."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import pytest
from livekit import api
from livekit.agents import metrics as m

from agent import telephony
from agent.config import Settings
from agent.events import CallEnded, PostCallProcessor, repeat_requests
from agent.main import outbound_number
from agent.metrics import CallMetrics
from agent.tools import LeadTools
from evals import run_speech
from evals.speech_score import key_terms_found, normalize, wer


# ---------- metrics: time to first token, tokens, every turn ----------
def _llm(sid, ttft, prompt, completion, duration=1.0):
    return m.LLMMetrics(label="llm", request_id="r", timestamp=1.0, duration=duration, ttft=ttft, cancelled=False,
                        completion_tokens=completion, prompt_tokens=prompt, prompt_cached_tokens=0,
                        total_tokens=prompt + completion, tokens_per_second=completion / duration, speech_id=sid)


def _eou(sid, eou):
    return m.EOUMetrics(timestamp=1.0, end_of_utterance_delay=eou, transcription_delay=0.1,
                        on_user_turn_completed_delay=0, speech_id=sid)


def _tts(sid, ttfb, chars=40):
    return m.TTSMetrics(label="tts", request_id="r", timestamp=1.0, ttfb=ttfb, duration=0.5, audio_duration=2.0,
                        cancelled=False, characters_count=chars, streamed=False, speech_id=sid)


def test_turn_details_tokens_and_slowest_stage():
    cm = CallMetrics()
    for sid, eou, ttft, ttfb in (("a", 0.3, 0.6, 0.2), ("b", 0.4, 2.5, 0.3)):
        cm.on_metrics(_eou(sid, eou))
        cm.on_metrics(_llm(sid, ttft, prompt=900, completion=30))
        cm.on_metrics(_tts(sid, ttfb))
    cm.on_metrics(_llm("b", 0.4, prompt=950, completion=10))  # a tool round trip in the same turn
    cm.on_metrics(_llm("text", 1.2, prompt=800, completion=20))  # text mode: no audio stages
    cm.on_metrics(m.STTMetrics(label="stt", request_id="r", timestamp=1.0, duration=0.2, audio_duration=3.0,
                               streamed=False))
    s = cm.summary()
    assert s["turns"] == 2 and s["llm_turns"] == 3 and s["llm_calls"] == 4
    assert s["llm_ttft"]["min"] == 0.6 and s["llm_ttft"]["max"] == 2.5
    assert s["end_to_end"]["max"] == pytest.approx(3.2)
    assert s["tokens"]["prompt"] == 900 + 900 + 950 + 800 and s["tokens"]["completion"] == 90
    assert s["stt"] == {"requests": 1, "audio_seconds": 3.0, "seconds_avg": 0.2}
    rows = {r["turn"]: r for r in s["turn_details"]}
    assert rows[2]["slowest_stage"] == "LLM first word" and rows[2]["llm_calls"] == 2
    assert rows[3]["end_to_end"] is None and rows[3]["llm_ttft"] == 1.2  # text-mode turn is still listed


# ---------- phone line ----------
def test_sip_codes():
    assert telephony.classify_sip(486) == "busy"
    assert telephony.classify_sip(480) == "no_answer" and telephony.classify_sip(408) == "no_answer"
    assert telephony.classify_sip(603) == "declined"
    assert telephony.classify_sip(404) == "invalid_number"
    assert telephony.classify_sip(503) == "line_error"
    assert telephony.mask_number("+971501234567") == "+97150*****67"


class FakeSip:
    def __init__(self, error=None):
        self.error, self.requests = error, []

    async def create_sip_participant(self, req):
        self.requests.append(req)
        if self.error:
            raise self.error

    async def transfer_sip_participant(self, req):
        self.requests.append(req)
        if self.error:
            raise self.error


async def test_dial_answered_and_busy():
    lk = SimpleNamespace(sip=FakeSip())
    line = await telephony.dial(lk, room="lead-L001-x", number="+971501234567", trunk_id="ST_1",
                                identity="sip-L001", ring_timeout=25)
    assert line["result"] == "answered" and line["number"] == "+97150*****67"
    req = lk.sip.requests[0]
    assert req.wait_until_answered and req.sip_call_to == "+971501234567" and req.ringing_timeout.seconds == 25

    busy = api.TwirpError("unavailable", "busy", status=503, metadata={"sip_status_code": "486", "sip_status": "Busy Here"})
    line = await telephony.dial(SimpleNamespace(sip=FakeSip(busy)), room="r", number="+9715", trunk_id="ST_1",
                                identity="sip-L001")
    assert (line["result"], line["sip_status_code"], line["label"]) == ("busy", 486, "Busy")


async def test_transfer_reports_failure():
    err = api.TwirpError("failed_precondition", "refer rejected", status=400, metadata={"sip_status_code": "403"})
    out = await telephony.transfer(SimpleNamespace(sip=FakeSip(err)), room="r", identity="sip-L001", to="sip:201@pbx")
    assert out["ok"] is False and out["error"] == "refer rejected"
    out = await telephony.transfer(SimpleNamespace(sip=FakeSip()), room="r", identity="sip-L001", to="tel:+97140000000")
    assert out["ok"] is True and out["to"] == "tel:+97140000000"


async def test_recording_start_and_failure():
    class Egress:
        def __init__(self, fail=False):
            self.fail, self.req = fail, None

        async def start_room_composite_egress(self, req):
            self.req = req
            if self.fail:
                raise RuntimeError("egress not running")
            return SimpleNamespace(egress_id="EG_1")

    ok = await telephony.start_recording(SimpleNamespace(egress=Egress()), room="r", filepath="recordings/r.ogg",
                                         s3={"bucket": "calls", "region": "me-central-1"})
    assert ok["ok"] and ok["egress_id"] == "EG_1" and ok["storage"] == "s3"
    bad = await telephony.start_recording(SimpleNamespace(egress=Egress(fail=True)), room="r", filepath="x.ogg")
    assert bad["ok"] is False and "egress not running" in bad["error"]


def test_outbound_number_from_metadata():
    lead = SimpleNamespace(phone="+971500000001")
    ctx = lambda job_meta, room_meta="": SimpleNamespace(job=SimpleNamespace(  # noqa: E731
        metadata=job_meta, room=SimpleNamespace(metadata=room_meta)))
    assert outbound_number(ctx(""), lead) is None
    assert outbound_number(ctx('{"dial": true}'), lead) == "+971500000001"
    assert outbound_number(ctx("", '{"lead_id": "L001", "phone_number": "+9715999"}'), lead) == "+9715999"


# ---------- transfer to a person ----------
async def test_transfer_without_phone_line_offers_callback(crm):
    tools = LeadTools(crm, "L001")
    reply = await tools.transfer_to_human("wants exact prices")
    assert "call them back" in reply
    assert tools.transfer["ok"] is False and tools.actions[-1]["action"] == "transfer_to_human"
    assert any("Asked to speak to a person" in n for n in (await crm.get_lead("L001")).notes)


async def test_transfer_with_phone_line(crm):
    async def transfer(reason):
        return {"ok": True, "to": "sip:201@pbx"}
    tools = LeadTools(crm, "L001", transfer=transfer)
    assert "transferred" in await tools.transfer_to_human("asked for a person")
    assert tools.transfer == {"reason": "asked for a person", "ok": True, "to": "sip:201@pbx"}


# ---------- call record ----------
async def test_unanswered_call_record(crm, tmp_path):
    class NoLLM:  # extraction must not run when nobody answered
        def __getattr__(self, name):
            raise AssertionError("LLM called for an unanswered call")
    proc = PostCallProcessor(crm, NoLLM(), "qwen3:8b", str(tmp_path))
    line = {"result": "busy", "label": "Busy", "sip_status_code": 486}
    record = await proc.handle(CallEnded(call_id="c1", lead_id="L001", transcript=[], actions=[], metrics={},
                                         started_at="2026-10-16T10:00:00+00:00", room="lead-L001-x",
                                         ended_by="not_reached", line=line))
    assert record["classification"] == "not_reached" and record["agent_closed"] is False
    saved = json.loads((tmp_path / "c1.json").read_text())
    assert saved["line"]["result"] == "busy" and saved["room"] == "lead-L001-x"
    assert any("Call attempt: Busy" in n for n in (await crm.get_lead("L001")).notes)


def test_repeat_requests():
    t = [{"role": "agent", "text": "Sorry, could you repeat that?"}, {"role": "user", "text": "sorry, again?"},
         {"role": "agent", "text": "Pardon, I didn't catch that."}, {"role": "agent", "text": "Great, thanks."}]
    assert repeat_requests(t) == 2


# ---------- speech accuracy ----------
def test_wer_normalises_numbers_and_punctuation():
    assert wer("My budget is around eight hundred thousand dollars.", "my budget is around 800,000 dollars") == 0
    assert wer("Call me at four pm.", "call me at 4pm") == 0
    assert wer("a b c d", "a x c") == 0.5  # one substitution + one deletion
    assert normalize("Around 1.2 million.") == ["around", "one", "point", "two", "million"]
    assert key_terms_found(["Dubai Marina", "villa"], "an apartment in dubai marina") == (["Dubai Marina"], ["villa"])


def test_round_trip_against_fake_speech_services(tmp_path):
    heard = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            if self.path.endswith("/audio/speech"):
                text = json.loads(body)["input"]
                heard["last"] = text.replace("Dubai Marina", "Dubai Marine")  # STT will "mishear" this one
                data, ctype = b"RIFF" + text.encode(), "audio/wav"
            else:
                data, ctype = json.dumps({"text": heard["last"]}).encode(), "application/json"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
        s = Settings(stt_base_url=url, tts_base_url=url)
        with httpx.Client(timeout=10) as client:
            items = run_speech.run_round_trip(client, s, voice="am_michael", language="en")
    finally:
        srv.shutdown()
        srv.server_close()
    summary = run_speech.summarize(items)
    bad = [i for i in items if i["errors"]]
    assert len(bad) == 1 and bad[0]["id"] == "s03_area_type" and bad[0]["missed_key_terms"] == ["Dubai Marina"]
    assert 0.98 < summary["accuracy"] < 1 and summary["key_term_accuracy"] < 1
    path = run_speech.save({"run_at": "2026-10-16T10:00:00+00:00", "mode": "round_trip", "summary": summary,
                            "items": items}, tmp_path)
    assert json.loads(path.read_text())["summary"]["items"] == 15
