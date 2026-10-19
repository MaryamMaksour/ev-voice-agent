"""Post-call work in the agent process, the raw-record safety net (agent.postcall), and the CRM audit."""

import json
import os
import time
from datetime import datetime, timedelta, timezone

import pytest
from openai import AsyncOpenAI

from agent.events import CallEnded, InProcessBus, PostCallProcessor, process_pending, write_raw
from crm.audited import AuditedCRM
from crm.port import LeadUpdate
from tests.fake_openai import FakeOpenAI

EXTRACTION = {"outcome": "qualified", "summary": "Wants a villa in Dubai Hills, 2M, within 3 months.",
              "lead_update": {"status": "qualified", "budget": 2000000, "preferred_area": "Dubai Hills",
                              "property_type": "villa", "timeline": "within 3 months", "interest_level": "hot"}}


def _event(call_id="c1", **kw) -> CallEnded:
    ended = datetime.now(timezone.utc)
    started = ended - timedelta(seconds=210)
    return CallEnded(call_id=call_id, lead_id="L001", started_at=started.isoformat(),
                     ended_at=ended.isoformat(), room="lead-L001-x", ended_by="agent",
                     transcript=[{"role": "agent", "text": "Hi"}, {"role": "user", "text": "A villa, 2 million"}],
                     actions=[], metrics={"turns": 1}, **kw)


def _processor(crm, fake, tmp_path) -> PostCallProcessor:
    llm = AsyncOpenAI(base_url=fake.base_url, api_key="x")
    return PostCallProcessor(crm, llm, "qwen3.5-9b", str(tmp_path / "calls"), logs_dir=str(tmp_path / "logs"))


async def test_in_process_bus_runs_post_call_work(crm, tmp_path):
    calls = tmp_path / "calls"
    with FakeOpenAI(lambda messages: EXTRACTION) as fake:
        bus = InProcessBus(_processor(crm, fake, tmp_path))
        bus.publish(_event(versions={"code": "abc"}, errors=[]))
        assert (calls / "c1.raw.json").exists()  # saved before anything else can fail
        await bus.drain()

    record = json.loads((calls / "c1.json").read_text())
    assert record["classification"] == "qualified" and record["duration_seconds"] == 210.0
    assert record["versions"] == {"code": "abc"} and record["post_call_seconds"] >= 0
    assert record["filled_after_call"]["preferred_area"] == "Dubai Hills"
    assert not (calls / "c1.raw.json").exists()  # replaced by the final record
    lead = await crm.get_lead("L001")
    assert lead.status.value == "qualified" and lead.budget == 2000000
    # the post-call CRM writes are in the call's crm_changes.jsonl, after the call's own
    changes = [json.loads(x) for x in (calls / "c1" / "crm_changes.jsonl").read_text().splitlines()]
    assert {c["source"] for c in changes} == {"post_call"}
    assert {c["action"] for c in changes} == {"update_lead", "add_note"}
    assert changes[0]["t_ms"] > 200_000  # counted from the start of the call
    assert record["log_files"]["crm_changes"] == "c1/crm_changes.jsonl"


async def test_post_call_llm_request_is_timed_and_token_counted(crm, tmp_path):
    calls = tmp_path / "calls"
    with FakeOpenAI(lambda messages: EXTRACTION) as fake:
        await _processor(crm, fake, tmp_path).handle(_event())

    record = json.loads((calls / "c1.json").read_text())
    summary = record["post_call_llm"]
    assert summary["requests"] == 1 and summary["ok"] is True
    assert summary["prompt_tokens"] == 100 and summary["completion_tokens"] == 20  # from the fake server
    assert summary["seconds"] >= 0
    llm_lines = [json.loads(x) for x in (calls / "c1" / "llm.jsonl").read_text().splitlines()]
    assert [x["stage"] for x in llm_lines] == ["post_call"]
    assert llm_lines[0]["request_id"] and "Dubai Hills" in llm_lines[0]["reply"]
    events = [json.loads(x)["event"] for x in (calls / "c1" / "events.jsonl").read_text().splitlines()]
    assert "metrics.llm_post_call" in events


async def test_invalid_json_is_retried_and_both_requests_logged(crm, tmp_path):
    replies = iter(["not json at all", EXTRACTION])
    with FakeOpenAI(lambda messages: next(replies)) as fake:
        await _processor(crm, fake, tmp_path).handle(_event())

    record = json.loads((tmp_path / "calls" / "c1.json").read_text())
    assert record["classification"] == "qualified"
    assert record["post_call_llm"]["requests"] == 2 and record["post_call_llm"]["ok"] is True
    lines = [json.loads(x) for x in (tmp_path / "calls" / "c1" / "llm.jsonl").read_text().splitlines()]
    assert [x["ok"] for x in lines] == [False, True] and "invalid JSON" in lines[0]["error"]


async def test_llm_down_still_saves_the_record_with_the_error(crm, tmp_path):
    llm = AsyncOpenAI(base_url="http://127.0.0.1:9/v1", api_key="x", max_retries=0)
    processor = PostCallProcessor(crm, llm, "qwen3.5-9b", str(tmp_path / "calls"), logs_dir=str(tmp_path / "logs"))
    await processor.handle(_event())

    record = json.loads((tmp_path / "calls" / "c1.json").read_text())
    assert record["error"] and record["errors"][-1]["stage"] == "post_call"
    assert record["post_call_llm"]["requests"] == 1 and record["post_call_llm"]["ok"] is False


async def test_pending_raw_calls_are_finished_later(crm, tmp_path):
    """The agent died after the call: agent.postcall finishes the work from the raw file."""
    calls = tmp_path / "calls"
    old = write_raw(_event("crashed"), calls)
    fresh = write_raw(_event("running"), calls)
    an_hour_ago = time.time() - 3600
    os.utime(old, (an_hour_ago, an_hour_ago))

    with FakeOpenAI(lambda messages: EXTRACTION) as fake:
        processor = _processor(crm, fake, tmp_path)
        done = await process_pending(processor, min_age_seconds=300)
        assert done == ["crashed"]  # the fresh one may still be in progress: left alone
        assert await process_pending(processor, min_age_seconds=300) == []  # nothing left: safe to rerun
        assert await process_pending(processor, min_age_seconds=0) == ["running"]

    assert json.loads((calls / "crashed.json").read_text())["classification"] == "qualified"
    assert not old.exists() and not fresh.exists()


async def test_unreadable_raw_file_is_left_in_place(crm, tmp_path):
    calls = tmp_path / "calls"
    calls.mkdir()
    broken = calls / "bad.raw.json"
    broken.write_text("{not json")
    with FakeOpenAI(lambda messages: EXTRACTION) as fake:
        assert await process_pending(_processor(crm, fake, tmp_path), min_age_seconds=0) == []
    assert broken.exists()


async def test_audited_crm_reports_before_and_after(crm):
    seen = []
    audited = AuditedCRM(crm, lambda **kw: seen.append(kw), source="call")
    await audited.update_lead("L001", LeadUpdate(budget=900000, preferred_area="Marina"))
    await audited.schedule_callback("L001", datetime(2030, 1, 1, 16, tzinfo=timezone.utc))
    assert [s["action"] for s in seen] == ["update_lead", "schedule_callback"]
    assert seen[0]["before"]["budget"] != 900000 and seen[0]["after"]["budget"] == 900000
    assert seen[0]["args"] == {"budget": 900000, "preferred_area": "Marina"} and seen[0]["ok"] is True
    assert seen[1]["after"]["status"] == "callback"

    with pytest.raises(LookupError):  # a failed write is reported, then raised as before
        await audited.add_note("NOPE", "x")
    assert seen[-1]["ok"] is False and seen[-1]["before"] is None and "NOPE" in seen[-1]["error"]


async def test_audit_callback_errors_never_break_crm_writes(crm):
    def broken(**kw):
        raise RuntimeError("disk full")

    lead = await AuditedCRM(crm, broken).update_lead("L001", LeadUpdate(budget=1))
    assert lead.budget == 1
