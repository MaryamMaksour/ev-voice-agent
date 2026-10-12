"""Post-call structured extraction: transcript -> validated CallOutcome JSON.

Runs once after the call. It double-checks what the agent saved during the call,
fills fields the agent missed, and produces the summary note for the CRM."""

from __future__ import annotations

import json
import logging
from enum import Enum

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

from crm.port import Lead, LeadStatus, LeadUpdate

from .config import no_think_body

logger = logging.getLogger("voice-agent.extraction")


class Outcome(str, Enum):
    QUALIFIED = "qualified"
    INTERESTED = "interested"
    NOT_INTERESTED = "not_interested"
    CALLBACK = "callback"
    DO_NOT_CALL = "do_not_call"
    WRONG_NUMBER = "wrong_number"
    NO_CONVERSATION = "no_conversation"


class CallOutcome(BaseModel):
    outcome: Outcome
    lead_update: LeadUpdate = Field(default_factory=LeadUpdate)
    callback_requested: bool = False
    callback_time_text: str | None = Field(
        default=None, description="the callback time exactly as agreed, e.g. 'Thursday 4pm'"
    )
    summary: str = Field(description="2-3 sentence summary for the sales team")


EXTRACTION_PROMPT = """\
You read a phone call transcript between a real estate assistant (agent) and a lead (user) \
and return ONLY a JSON object matching this schema:

{schema}

Rules:
- Only use facts the USER actually said or clearly confirmed. Never guess.
- budget is a whole number in US dollars (e.g. "around 1.2 million" -> 1200000). If a range \
was given, use the upper bound.
- property_type: apartment, villa, townhouse, office or land.
- interest_level: hot (buying within 3 months), warm (3-12 months), cold (later or unsure).
- lead_update.status must match outcome when outcome is qualified, interested, \
not_interested, do_not_call or wrong_number; for callback use status "callback".
- outcome "qualified" only if budget, preferred_area, property_type and timeline are all known.
- Leave fields null when they were not discussed.
"""


def format_transcript(turns: list[dict]) -> str:
    return "\n".join(f"{t['role'].upper()}: {t['text']}" for t in turns if t.get("text"))


def _parse_json(text: str) -> dict:
    text = text.strip()
    # tolerate ```json fences and stray <think></think> blocks from Qwen
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start : end + 1])


async def extract_outcome(
    client: AsyncOpenAI, model: str, lead: Lead, turns: list[dict], retries: int = 1
) -> CallOutcome:
    if not any(t["role"] == "user" and t.get("text") for t in turns):
        return CallOutcome(outcome=Outcome.NO_CONVERSATION, summary="No conversation with the lead.")

    schema = CallOutcome.model_json_schema()
    messages = [
        {"role": "system", "content": EXTRACTION_PROMPT.format(schema=json.dumps(schema))},
        {
            "role": "user",
            "content": f"Lead name: {lead.name}\nProject: {lead.project_interest}\n\n"
            f"Transcript:\n{format_transcript(turns)}",
        },
    ]
    last_error: Exception | None = None
    for _ in range(retries + 1):
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "call_outcome", "schema": schema},
            },
            extra_body=no_think_body(model),
        )
        content = resp.choices[0].message.content or ""
        try:
            return CallOutcome.model_validate(_parse_json(content))
        except (ValueError, ValidationError) as e:
            last_error = e
            messages += [
                {"role": "assistant", "content": content},
                {"role": "user", "content": f"Invalid JSON for the schema: {e}. Return corrected JSON only."},
            ]
    raise ValueError(f"extraction failed: {last_error}")


def reconcile(lead_after_call: Lead, outcome: CallOutcome) -> tuple[LeadUpdate, list[dict]]:
    """Compare extraction with what the agent saved during the call.

    Returns (fill_in, disagreements):
    - fill_in: fields the agent never saved but the extraction found -> safe to write.
    - disagreements: fields where both have a value and they differ -> logged for review,
      the in-call value is kept (the agent confirmed it live with the caller).
    """
    fill: dict = {}
    disagreements: list[dict] = []
    for key, extracted in outcome.lead_update.changes().items():
        current = getattr(lead_after_call, key)
        current = getattr(current, "value", current)
        if key == "status" and current in (LeadStatus.NEW.value, LeadStatus.CONTACTED.value):
            current = None  # untouched default status, not a real in-call decision
        if current is None:
            fill[key] = extracted
        elif str(current).lower() != str(extracted).lower():
            disagreements.append({"field": key, "in_call": current, "extracted": extracted})
    return LeadUpdate(**fill), disagreements
