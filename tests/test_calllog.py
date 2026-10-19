"""Per-call logs: timeline, every LLM request, CRM changes, service log lines, recording.

Runs the real AgentSession (text mode) against a scripted OpenAI-compatible server; no GPU, no LiveKit."""

import json
import logging
from types import SimpleNamespace

from livekit.agents import AgentSession
from livekit.agents import metrics as m
from livekit.plugins import openai

from agent.calllog import CallLogger, bundle_call_logs, recording_seconds
from agent.config import Settings, no_think_body, session_conn_options
from agent.logging_setup import JsonFormatter, bind_call, setup_logging
from agent.main import LeadQualifierAgent
from agent.tools import LeadTools
from crm.audited import AuditedCRM
from tests.fake_openai import FakeOpenAI


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def policy(messages):
    last = messages[-1]
    if last["role"] == "tool":
        return "<think>saved it</think>Thank you. Which area do you prefer?"
    if "800" in str(last.get("content", "")):
        return {"tool": "update_lead", "args": {"status": "interested", "budget": 800000, "preferred_area": None,
                                                "property_type": None, "timeline": None, "interest_level": None}}
    return "Hi, this is Alex from Horizon Realty."


async def test_call_folder_has_timeline_llm_requests_and_crm_changes(crm, tmp_path):
    log = CallLogger(tmp_path, "call-1")
    audited = AuditedCRM(crm, log.crm_change, source="call")
    tools = LeadTools(audited, "L001")
    lead = await audited.get_lead("L001")
    with FakeOpenAI(policy) as fake:
        llm = openai.LLM(base_url=fake.base_url, api_key="x", model="qwen3.5-9b",
                         extra_body=no_think_body("qwen3.5-9b"))
        session = AgentSession(llm=llm, conn_options=session_conn_options(Settings()))
        log.attach(session)
        async with session:
            await session.start(LeadQualifierAgent(lead, tools, Settings(), call_log=log))
            await session.run(user_input="My budget is about 800 thousand")
    log.close(ended_by="lead")

    folder = tmp_path / "call-1"
    # every LLM request: what was sent, what came back, tokens, timings
    llm_lines = _lines(folder / "llm.jsonl")
    assert len(llm_lines) >= 2
    tool_req = next(r for r in llm_lines if r["tool_calls"])
    assert tool_req["tool_calls"][0]["name"] == "update_lead"
    assert "tool_call_id" in tool_req["tool_calls"][0]
    assert tool_req["messages"] and tool_req["ttft"] is not None and tool_req["seconds"] is not None
    assert "update_lead" in tool_req["tools"] and tool_req["call_id"] == "call-1"
    assert tool_req["metrics"]["prompt_tokens"] == 120 and tool_req["metrics"]["completion_tokens"] == 12
    reply = next(r for r in llm_lines if "Which area" in r["text"])
    assert "<think>" not in reply["text"] and "saved it" not in reply["text"]  # thinking never logged as speech
    assert "reasoning_tokens" in reply["metrics"]

    # CRM write during the call, with the lead before and after
    changes = _lines(folder / "crm_changes.jsonl")
    assert changes[0]["source"] == "call" and changes[0]["action"] == "update_lead"
    assert changes[0]["changed"]["budget"]["after"] == 800000
    assert changes[0]["changed"]["status"] == {"before": "new", "after": "interested"}

    # timeline
    kinds = [e["event"] for e in _lines(folder / "events.jsonl")]
    for kind in ("message", "tool_call", "crm_change", "metrics.llm", "call_closed"):
        assert kind in kinds, kind
    tool_event = next(e for e in _lines(folder / "events.jsonl") if e["event"] == "tool_call")
    assert tool_event["call_id"] == "call-1" and tool_event["ok"] is True

    s = log.summary()
    assert s["llm_requests"]["total"] >= 2 and s["llm_requests"]["errors"] == 0
    assert s["counts"]["crm_writes"] == 1 and s["counts"]["tool_calls"] == 1
    assert set(log.files()) >= {"events", "llm", "crm_changes"}


def _llm_metrics(rid):
    return m.LLMMetrics(label="llm", request_id=rid, timestamp=1.0, duration=0.5, ttft=0.2, cancelled=False,
                        completion_tokens=10, prompt_tokens=90, prompt_cached_tokens=80, total_tokens=100,
                        tokens_per_second=20.0, speech_id="s1")


def _request(log, rid):
    rec = log.llm_started(chat_ctx=SimpleNamespace(to_dict=lambda **kw: {"items": [{"role": "user"}]}), tools=[])
    log.llm_chunk(rec, SimpleNamespace(id=rid, delta=SimpleNamespace(content="hello", tool_calls=[]), usage=None))
    return rec


def test_llm_metrics_join_in_either_order(tmp_path):
    log = CallLogger(tmp_path, "c")
    rec = _request(log, "r1")
    log.llm_finished(rec)                 # request first, metrics later
    log.on_metrics(_llm_metrics("r1"))
    log.on_metrics(_llm_metrics("r2"))    # metrics first, request later
    log.llm_finished(_request(log, "r2"))
    rec3 = _request(log, "r3")            # interrupted: metrics never come
    log.llm_finished(rec3, cancelled=True)
    log.close()
    lines = _lines(tmp_path / "c" / "llm.jsonl")
    assert [line["request_id"] for line in lines] == ["r1", "r2", "r3"]
    assert lines[0]["metrics"]["prompt_cached_tokens"] == 80 and lines[1]["metrics"]["tokens_per_second"] == 20.0
    assert lines[2]["cancelled"] is True and "metrics" not in lines[2]
    assert log.summary()["llm_requests"]["cancelled"] == 1


def test_json_logs_carry_call_fields(tmp_path):
    record = logging.LogRecord("voice-agent", logging.INFO, "x", 1, "hello %s", ("there",), None)
    record.call_id = "c9"
    line = json.loads(JsonFormatter().format(record))
    assert line["msg"] == "hello there" and line["call_id"] == "c9" and line["ts"].endswith("+00:00")

    folder = setup_logging("unit-test-service", logs_dir=str(tmp_path))
    bind_call(call_id="c42", lead_id="L001")
    logging.getLogger("voice-agent.test").warning("inside the call")
    for h in logging.getLogger().handlers:
        h.flush()
    lines = [json.loads(x) for f in folder.glob("*.jsonl") for x in f.read_text().splitlines()]
    mine = [x for x in lines if x["msg"] == "inside the call"]
    assert mine and mine[0]["call_id"] == "c42" and mine[0]["service"] == "unit-test-service"
    from agent import logging_setup
    logging_setup._call_fields.set({})
    for h in list(logging.getLogger().handlers):
        if getattr(h, "folder", None) == folder:
            logging.getLogger().removeHandler(h)
            h.close()
    logging_setup._configured.discard("unit-test-service")


def test_bundle_copies_call_lines_and_links_recording(tmp_path):
    logs = tmp_path / "logs" / "agent"
    logs.mkdir(parents=True)
    (logs / "2026-10-19.jsonl").write_text(
        '{"ts": "2026-10-19T10:00:01", "msg": "b", "call_id": "c1"}\n'
        '{"ts": "2026-10-19T10:00:00", "msg": "other", "call_id": "c2"}\n'
        '{"ts": "2026-10-19T10:00:00", "msg": "a", "call_id": "c1"}\n')
    rec_dir = tmp_path / "recordings"
    rec_dir.mkdir()
    # minimal OGG: last page granule position = 48000 * 12.5 samples
    page = b"OggS" + b"\0\0" + int(48000 * 12.5).to_bytes(8, "little") + b"\0" * 20
    (rec_dir / "lead-L001-x.ogg").write_bytes(page)

    files = bundle_call_logs("c1", calls_dir=tmp_path / "calls", logs_dir=tmp_path / "logs",
                             recordings_dir=rec_dir, recording_file="/out/lead-L001-x.ogg")
    agent_log = (tmp_path / "calls" / "c1" / "agent.log").read_text().splitlines()
    assert [json.loads(x)["msg"] for x in agent_log] == ["a", "b"]  # only this call, in time order
    assert files["agent_log"] == "c1/agent.log" and files["recording"] == "c1/recording.ogg"
    assert recording_seconds(tmp_path / "calls" / "c1" / "recording.ogg") == 12.5
