"""Per-call overrides for the demo website: only honoured in demo mode, only the allowed keys."""

import json
from datetime import datetime

from agent import demo_mode
from agent.prompts import SYSTEM_PROMPT, build_system_prompt, prompt_version
from crm import Lead

LEAD = Lead(id="L003", name="Emily Davis", phone="+15550100003", project_interest="Downtown Lofts")
NOW = datetime(2026, 10, 21, 10, 0)


def meta(**kw) -> str:
    return json.dumps({"source": "demo", **kw})


def test_default_prompt_is_valid():
    assert demo_mode.validate_template(SYSTEM_PROMPT) is None


def test_overrides_ignored_when_demo_mode_is_off(monkeypatch):
    monkeypatch.delenv("DEMO_MODE", raising=False)
    assert demo_mode.overrides_from_metadata([meta(tts_voice="am_adam")]) == {}


def test_overrides_ignored_without_source_demo():
    raw = json.dumps({"tts_voice": "am_adam", "prompt": "Hi {name}"})
    assert demo_mode.overrides_from_metadata([raw], on=True) == {}


def test_allowed_overrides_pass_and_others_are_dropped():
    raw = meta(tts_voice="am_adam", stt_model="small", stt_base_url="http://localhost:8002/v1",
               prompt="You are {agent_name}. Call {name}. Use update_lead.", llm_model="evil", lead_id="L001")
    got = demo_mode.overrides_from_metadata(["", raw], on=True)
    assert set(got) == {"tts_voice", "stt_model", "stt_base_url", "prompt"}
    assert got["tts_voice"] == "am_adam"


def test_bad_values_are_dropped_not_raised():
    raw = meta(prompt="Hello {nope}", stt_base_url="file:///etc/passwd", tts_voice="x" * 500)
    assert demo_mode.overrides_from_metadata([raw], on=True) == {}
    assert demo_mode.overrides_from_metadata(["not json", None], on=True) == {}


def test_enabled_reads_the_env(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "1")
    assert demo_mode.enabled()
    assert demo_mode.overrides_from_metadata([meta(tts_voice="am_adam")]) == {"tts_voice": "am_adam"}


def test_validate_template_messages():
    assert "empty" in demo_mode.validate_template("  ")
    assert "too long" in demo_mode.validate_template("a" * (demo_mode.MAX_PROMPT_CHARS + 1))
    assert "{nope}" in demo_mode.validate_template("Hi {nope}")
    assert "not closed" in demo_mode.validate_template("Hi {name")
    assert demo_mode.validate_template("Hi {}") is not None          # positional field
    assert demo_mode.validate_template("Hi {name.upper}") is not None  # no attribute access
    assert demo_mode.validate_template('Say {{"ok": true}} to {name}') is None  # escaped braces are fine


def test_custom_template_changes_the_prompt_and_version():
    template = "You are {agent_name} from {company}. Phone {first_name} about {project}. Use update_lead."
    text = build_system_prompt(LEAD, company="Evolutions", agent_name="Alex", tz="UTC", now=NOW, template=template)
    assert text == "You are Alex from Evolutions. Phone Emily about Downtown Lofts. Use update_lead."
    assert prompt_version(template) != prompt_version()
    default = build_system_prompt(LEAD, company="Evolutions", agent_name="Alex", tz="UTC", now=NOW)
    assert "## Your goal" in default


def test_summary_hides_the_prompt_text():
    assert demo_mode.summary({}) is None
    got = demo_mode.summary({"prompt": "secret text", "tts_voice": "am_adam"})
    assert got == {"tts_voice": "am_adam", "custom_prompt": True}
