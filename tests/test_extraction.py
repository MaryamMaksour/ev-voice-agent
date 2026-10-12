import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.events import CallEnded, PostCallProcessor
from agent.extraction import CallOutcome, Outcome, _parse_json, extract_outcome, reconcile
from crm.port import Lead, LeadUpdate
from evals.score import score_case

CASES = json.loads((Path(__file__).parent / "transcripts" / "cases.json").read_text())


class FakeLLM:
    """Stands in for AsyncOpenAI: returns canned completions in order."""

    def __init__(self, *contents: str) -> None:
        self._contents = list(contents)
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls += 1
        content = self._contents.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_cases_file_is_valid():
    assert len(CASES) == 15
    for case in CASES:
        CallOutcome.model_validate({**case["expected"], "summary": "x"})
        Lead(**case["lead"])


def test_parse_json_tolerates_fences_and_think():
    assert _parse_json('<think>hmm</think>\n```json\n{"a": 1}\n```') == {"a": 1}


async def test_no_user_speech_skips_llm():
    llm = FakeLLM()
    case = next(c for c in CASES if c["expected"]["outcome"] == "no_conversation")
    out = await extract_outcome(llm, "m", Lead(**case["lead"]), case["transcript"])
    assert out.outcome == Outcome.NO_CONVERSATION and llm.calls == 0


async def test_extraction_retries_on_bad_json():
    good = json.dumps({"outcome": "not_interested", "lead_update": {"status": "not_interested"}, "summary": "s"})
    llm = FakeLLM("not json at all", good)
    case = CASES[1]
    out = await extract_outcome(llm, "m", Lead(**case["lead"]), case["transcript"])
    assert out.outcome == Outcome.NOT_INTERESTED and llm.calls == 2


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_expected_answers_score_100(case):
    perfect = CallOutcome.model_validate({**case["expected"], "summary": "x"})
    assert all(r["ok"] for r in score_case(case["expected"], perfect))


def test_score_flags_hallucinated_budget():
    case = CASES[1]  # not interested, no budget given
    bad = CallOutcome(outcome="not_interested", lead_update=LeadUpdate(status="not_interested", budget=500000), summary="x")
    rows = score_case(case["expected"], bad)
    assert [r["field"] for r in rows if not r["ok"]] == ["budget"]


def test_reconcile_fills_missing_and_flags_conflicts():
    lead = Lead(id="L1", name="A", phone="1", budget=800000, status="interested")
    out = CallOutcome(
        outcome="qualified",
        lead_update=LeadUpdate(status="qualified", budget=900000, preferred_area="Marina"),
        summary="x",
    )
    fill, disagreements = reconcile(lead, out)
    assert fill.changes() == {"preferred_area": "Marina"}
    assert {d["field"] for d in disagreements} == {"budget", "status"}


def test_reconcile_treats_default_status_as_unset():
    lead = Lead(id="L1", name="A", phone="1")
    out = CallOutcome(outcome="not_interested", lead_update=LeadUpdate(status="not_interested"), summary="x")
    fill, disagreements = reconcile(lead, out)
    assert fill.changes() == {"status": "not_interested"} and disagreements == []


async def test_post_call_processor_writes_note_and_record(crm, tmp_path):
    case = CASES[0]
    llm = FakeLLM(json.dumps({**case["expected"], "summary": "Sarah wants a 2-bed in Marina."}))
    proc = PostCallProcessor(crm, llm, "m", str(tmp_path / "calls"))
    record = await proc.handle(
        CallEnded(call_id="c1", lead_id="L001", transcript=case["transcript"], actions=[], metrics={}, started_at="t")
    )
    lead = await crm.get_lead("L001")
    assert lead.status.value == "qualified" and lead.budget == 800000
    assert lead.notes == ["Sarah wants a 2-bed in Marina."]
    assert record["filled_after_call"]["budget"] == 800000
    assert (tmp_path / "calls" / "c1.json").exists()


async def test_post_call_processor_keeps_record_on_failure(crm, tmp_path):
    proc = PostCallProcessor(crm, FakeLLM("bad", "still bad"), "m", str(tmp_path))
    record = await proc.handle(
        CallEnded(call_id="c2", lead_id="L001", transcript=CASES[0]["transcript"], actions=[], metrics={}, started_at="t")
    )
    assert "error" in record and (tmp_path / "c2.json").exists()


async def test_post_call_enforces_do_not_call_missed_in_call(crm, tmp_path):
    case = next(c for c in CASES if c["expected"]["outcome"] == "do_not_call")
    llm = FakeLLM(json.dumps({**case["expected"], "summary": "Asked not to be called."}))
    proc = PostCallProcessor(crm, llm, "m", str(tmp_path))
    await proc.handle(CallEnded(call_id="c3", lead_id="L004", transcript=case["transcript"], actions=[],
                                metrics={}, started_at="t"))
    lead = await crm.get_lead("L004")
    assert lead.do_not_call and lead.status.value == "do_not_call"
