"""Agent wiring checks that need no models: tool schemas, prompt, lead resolution."""

from datetime import datetime
from types import SimpleNamespace

from livekit.agents.llm import utils

from agent.config import Settings, no_think_body
from agent.main import LeadQualifierAgent, resolve_lead_id
from agent.metrics import CallMetrics, percentile
from agent.prompts import build_system_prompt
from agent.tools import LeadTools
from crm.port import Lead

LEAD = Lead(id="L1", name="Sarah Johnson", phone="1", project_interest="Marina Heights", budget=800000)


def test_tool_schemas_are_strict_openai_compatible():
    agent = LeadQualifierAgent(LEAD, LeadTools(None, "L1"), Settings())
    names = set()
    for tool in agent.tools:
        if hasattr(tool, "tools"):  # toolsets like EndCallTool
            names |= {t.id for t in tool.tools}
            continue
        schema = utils.build_strict_openai_schema(tool)
        names.add(schema["function"]["name"])
    assert {"update_lead", "schedule_callback", "mark_do_not_call", "end_call"} <= names


def test_prompt_contains_lead_context():
    p = build_system_prompt(LEAD, company="Horizon", agent_name="Alex", tz="UTC", now=datetime(2026, 10, 12, 9))
    assert "Sarah" in p and "Marina Heights" in p and "800,000 USD" in p and "/no_think" not in p


def test_no_think_only_for_qwen3():
    off = {"chat_template_kwargs": {"enable_thinking": False}}
    assert no_think_body("qwen3.5-9b") == off and no_think_body("Qwen/Qwen3.5-9B") == off
    assert no_think_body("llama3.1") == {}


def _ctx(metadata: str, room: str):
    return SimpleNamespace(job=SimpleNamespace(metadata=metadata), room=SimpleNamespace(name=room))


def test_resolve_lead_id():
    assert resolve_lead_id(_ctx('{"lead_id": "L007"}', "x"), "L001") == "L007"
    assert resolve_lead_id(_ctx("", "lead-L003"), "L001") == "L003"
    assert resolve_lead_id(_ctx("", "lead-L003-sim7"), "L001") == "L003"
    assert resolve_lead_id(_ctx("", "playground-abc"), "L001") == "L001"


def test_latency_summary():
    from livekit.agents import metrics as m

    cm = CallMetrics()
    cm.on_metrics(m.EOUMetrics(timestamp=0, end_of_utterance_delay=0.4, transcription_delay=0.2,
                               on_user_turn_completed_delay=0, speech_id="s1"))
    cm.on_metrics(SimpleNamespace(speech_id=None))  # ignored
    assert cm.summary()["turns"] == 0
    assert percentile([1, 2, 3, 4], 50) == 2.5
