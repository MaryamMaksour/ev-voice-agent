"""Post-call work on Celery (eager mode, no Redis), the raw-record safety net, and the CRM audit."""

import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

import pytest

from agent import tasks
from agent.events import CallEnded, CeleryBus, InProcessBus, PostCallProcessor, write_raw
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


@pytest.fixture
def eager(monkeypatch, tmp_path, crm):
    """Celery runs tasks in-process; the task builds its own CRM and LLM client from the environment."""
    monkeypatch.setenv("CRM_SQLITE_PATH", str(tmp_path / "crm.db"))  # the crm fixture's database
    monkeypatch.setenv("CALLS_DIR", str(tmp_path / "calls"))
    monkeypatch.setenv("LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("RECORDINGS_DIR", str(tmp_path / "recordings"))
    monkeypatch.setenv("RECORDING_WAIT_SECONDS", "0")
    tasks.app.conf.task_always_eager = True
    tasks.app.conf.task_eager_propagates = True
    yield tmp_path
    tasks.app.conf.task_always_eager = False


async def test_celery_task_runs_post_call_work(eager, crm, monkeypatch):
    with FakeOpenAI(lambda messages: EXTRACTION) as fake:
        monkeypatch.setenv("LLM_BASE_URL", fake.base_url)
        event = _event(versions={"code": "abc"}, errors=[])
        write_raw(event, eager / "calls")
        # Celery tasks are sync and call asyncio.run, so run it outside this test's event loop
        import asyncio
        result = await asyncio.to_thread(lambda: tasks.process_call.delay(asdict(event)).get())

    assert result == {"call_id": "c1", "classification": "qualified", "error": None}
    record = json.loads((eager / "calls" / "c1.json").read_text())
    assert record["classification"] == "qualified" and record["duration_seconds"] == 210.0
    assert record["versions"] == {"code": "abc"} and record["post_call_seconds"] >= 0
    assert record["filled_after_call"]["preferred_area"] == "Dubai Hills"
    assert not (eager / "calls" / "c1.raw.json").exists()  # replaced by the final record
    lead = await crm.get_lead("L001")
    assert lead.status.value == "qualified" and lead.budget == 2000000
    # the post-call CRM writes are in the call's crm_changes.jsonl, after the call's own
    changes = [json.loads(x) for x in (eager / "calls" / "c1" / "crm_changes.jsonl").read_text().splitlines()]
    assert {c["source"] for c in changes} == {"post_call"}
    assert {c["action"] for c in changes} == {"update_lead", "add_note"}
    assert changes[0]["t_ms"] > 200_000  # counted from the start of the call
    assert record["log_files"]["crm_changes"] == "c1/crm_changes.jsonl"


async def test_celery_bus_falls_back_to_in_process_when_redis_is_down(crm, tmp_path, monkeypatch):
    def broker_down(*a, **kw):
        raise ConnectionError("redis down")

    monkeypatch.setattr(tasks.process_call, "apply_async", broker_down)
    with FakeOpenAI(lambda messages: EXTRACTION) as fake:
        from openai import AsyncOpenAI

        llm = AsyncOpenAI(base_url=fake.base_url, api_key="x")
        processor = PostCallProcessor(crm, llm, "qwen3.5-9b", str(tmp_path), logs_dir=str(tmp_path / "logs"))
        bus = CeleryBus(str(tmp_path), fallback=InProcessBus(processor))
        bus.publish(_event("c2"))
        assert (tmp_path / "c2.raw.json").exists()  # saved before anything else can fail
        await bus.drain()
    assert json.loads((tmp_path / "c2.json").read_text())["classification"] == "qualified"
    assert not (tmp_path / "c2.raw.json").exists()


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
