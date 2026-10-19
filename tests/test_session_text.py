"""Text-mode run of the real AgentSession + OpenAI plugin + tools + CRM.

The model is a scripted fake server, so this checks the wiring, not model quality.
Model quality is checked by test_scenarios_llm.py and evals/run_extraction.py."""

from livekit.agents import AgentSession
from livekit.plugins import openai

import time

from agent.config import Settings, no_think_body, session_conn_options
from agent.main import LeadQualifierAgent
from agent.tools import LeadTools
from tests.fake_openai import FakeOpenAI


def policy(messages):
    last = messages[-1]
    if last["role"] == "tool":
        return "Got it, thank you. And which area do you prefer?"
    text = str(last.get("content", "")).lower()
    if "800" in text:
        return {"tool": "update_lead", "args": {"status": "interested", "budget": 800000,
                "preferred_area": None, "property_type": None, "timeline": None, "interest_level": None}}
    if "stop calling" in text:
        return {"tool": "mark_do_not_call", "args": {}}
    return "Hi, this is Alex from Horizon Realty. Am I speaking with Sarah?"


async def _session(fake, crm, lead_id="L001"):
    model = "qwen3.5-9b"
    llm = openai.LLM(base_url=fake.base_url, api_key="x", model=model, extra_body=no_think_body(model))
    lead = await crm.get_lead(lead_id)
    tools = LeadTools(crm, lead_id)
    session = AgentSession(llm=llm, conn_options=session_conn_options(Settings()))
    return session, LeadQualifierAgent(lead, tools, Settings()), tools


async def test_tool_call_reaches_crm(crm):
    with FakeOpenAI(policy) as fake:
        session, agent, tools = await _session(fake, crm)
        async with session:
            await session.start(agent)
            result = await session.run(user_input="Yes, my budget is about 800 thousand")
            result.expect.contains_function_call(name="update_lead")

    lead = await crm.get_lead("L001")
    assert lead.budget == 800000 and lead.status.value == "interested"
    assert tools.actions[0]["action"] == "update_lead"

    req = fake.requests[-1]
    assert req["chat_template_kwargs"] == {"enable_thinking": False}  # vLLM: thinking off
    tool_names = {t["function"]["name"] for t in req["tools"]}
    assert {"update_lead", "schedule_callback", "mark_do_not_call", "end_call"} <= tool_names
    assert "Sarah Johnson" in req["messages"][0]["content"]  # lead context in the system prompt


async def test_do_not_call(crm):
    with FakeOpenAI(policy) as fake:
        session, agent, _ = await _session(fake, crm, "L004")
        async with session:
            await session.start(agent)
            result = await session.run(user_input="Please stop calling me")
            result.expect.contains_function_call(name="mark_do_not_call")
    assert (await crm.get_lead("L004")).do_not_call


async def test_think_text_is_never_spoken(crm):
    def thinking_policy(messages):
        return "<think>The user said hi, I should greet.</think>Hello Sarah, this is Alex."

    with FakeOpenAI(thinking_policy) as fake:
        session, agent, _ = await _session(fake, crm)
        async with session:
            await session.start(agent)
            await session.run(user_input="Hi")
            said = [m.text_content for m in session.history.items
                    if getattr(m, "type", None) == "message" and m.role == "assistant"]
    assert said and all("think" not in t.lower() and "greet" not in t for t in said)
    assert said[-1] == "Hello Sarah, this is Alex."


async def test_slow_llm_does_not_time_out(crm, monkeypatch):
    """A first token after 11 s (slower than LiveKit's 10 s default) must still work."""
    def slow_policy(messages):
        if messages[-1]["role"] == "user" and "slow" in str(messages[-1].get("content")):
            time.sleep(11)
        return "Thanks for waiting."

    with FakeOpenAI(slow_policy) as fake:
        session, agent, _ = await _session(fake, crm)
        async with session:
            await session.start(agent)
            result = await session.run(user_input="this will be slow")
            result.expect.contains_message(role="assistant")
