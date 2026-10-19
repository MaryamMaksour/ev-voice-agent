"""Demo test #4 with the REAL model, in text mode (no audio, speed doesn't matter).

Skipped unless RUN_LLM_TESTS=1. Uses the vLLM server (Qwen3.5-9B on the company GPU
server) via LLM_BASE_URL / LLM_MODEL from .env:

    RUN_LLM_TESTS=1 pytest tests/test_scenarios_llm.py -v
"""

import os

import pytest
from dotenv import load_dotenv
from livekit.agents import AgentSession
from livekit.plugins import openai

from agent.config import Settings, no_think_body
from agent.main import LeadQualifierAgent
from agent.tools import LeadTools

load_dotenv()
pytestmark = pytest.mark.skipif(os.getenv("RUN_LLM_TESTS") != "1", reason="set RUN_LLM_TESTS=1 to use the real LLM")


async def converse(crm, lead_id: str, user_turns: list[str]):
    s = Settings()
    llm = openai.LLM(base_url=s.llm_base_url, api_key=s.llm_api_key, model=s.llm_model,
                     temperature=0, extra_body=no_think_body(s.llm_model), timeout=None)
    tools = LeadTools(crm, lead_id, tz=s.timezone)
    agent = LeadQualifierAgent(await crm.get_lead(lead_id), tools, s)
    async with AgentSession(llm=llm) as session:
        await session.start(agent)
        for turn in user_turns:
            await session.run(user_input=turn)
    return await crm.get_lead(lead_id), tools.actions


async def test_qualified_lead(crm):
    lead, _ = await converse(crm, "L001", [
        "Yes, this is Sarah.",
        "Yes, I'm still interested in Marina Heights.",
        "My budget is around eight hundred thousand dollars.",
        "I prefer the Marina area.",
        "A two bedroom apartment.",
        "Within the next two months.",
    ])
    assert lead.budget == 800000
    assert lead.property_type and lead.property_type.value == "apartment"
    assert lead.preferred_area and "marina" in lead.preferred_area.lower()
    assert lead.status.value in ("qualified", "interested")


async def test_callback(crm):
    lead, actions = await converse(crm, "L003", [
        "Yes, but I'm driving right now.",
        "Can you call me tomorrow at 4 pm?",
    ])
    assert lead.status.value == "callback" and lead.callback_at is not None
    assert any(a["action"] == "schedule_callback" for a in actions)


async def test_not_interested(crm):
    lead, _ = await converse(crm, "L002", [
        "Speaking.",
        "No, sorry, we already bought a house last month. Not interested.",
    ])
    assert lead.status.value == "not_interested"


async def test_do_not_call(crm):
    lead, _ = await converse(crm, "L004", ["Yes. Please stop calling me and take my number off your list."])
    assert lead.do_not_call


async def test_wrong_person(crm):
    lead, _ = await converse(crm, "L005", ["No, there's no Olivia here. You have the wrong number."])
    assert lead.status.value == "wrong_number"
    assert lead.budget is None
