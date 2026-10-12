"""LiveKit agent worker: the cascaded voice pipeline.

    python -m agent.main download-files   # one-time: VAD + turn-detector weights
    python -m agent.main dev              # connect to LiveKit, wait for calls
    python -m agent.main console          # talk to it in the terminal (no LiveKit server)

Which lead is called: job metadata {"lead_id": "L003"} (explicit dispatch), else a room
named "lead-L003" (or "lead-L003-anything"), else DEFAULT_LEAD_ID.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Literal
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    ChatContext,
    ModelSettings,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    cli,
    function_tool,
)
from livekit.agents import llm as lk_llm
from livekit.agents.beta.tools import EndCallTool
from livekit.agents.llm.utils import ThinkingTokenFilter, strip_thinking_tokens
from livekit.plugins import openai, silero
# local turn-detector model (runs in-process). inference.TurnDetector is cloud-first,
# so we keep the plugin to stay fully self-hosted.
from livekit.plugins.turn_detector.english import EnglishModel
from openai import AsyncOpenAI

from crm import Lead, make_crm

from .config import Settings, no_think_body
from .events import CallEnded, InProcessBus, PostCallProcessor
from .metrics import CallMetrics
from .prompts import build_system_prompt, greeting_instructions
from .tools import LeadTools

load_dotenv()
logger = logging.getLogger("voice-agent")

Status = Literal["interested", "qualified", "not_interested", "wrong_number"]
PropertyKind = Literal["apartment", "villa", "townhouse", "office", "land"]
Interest = Literal["hot", "warm", "cold"]


class LeadQualifierAgent(Agent):
    def __init__(self, lead: Lead, tools: LeadTools, settings: Settings) -> None:
        super().__init__(
            instructions=build_system_prompt(
                lead,
                company=settings.company_name,
                agent_name=settings.agent_name,
                tz=settings.timezone,
                now=datetime.now(ZoneInfo(settings.timezone)),
            ),
            tools=[
                EndCallTool(
                    ignore_on_enter=True,
                    end_instructions="thank the caller briefly and say goodbye",
                )
            ],
        )
        self._lead = lead
        self._lead_tools = tools
        self._settings = settings

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=greeting_instructions(
                self._lead, company=self._settings.company_name, agent_name=self._settings.agent_name
            )
        )

    async def llm_node(self, chat_ctx: ChatContext, tools: list[lk_llm.Tool], model_settings: ModelSettings):
        """Default LLM step, minus any <think>...</think> text, so it is never spoken."""
        state = ThinkingTokenFilter()
        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            if isinstance(chunk, str):
                chunk = strip_thinking_tokens(chunk, state) or ""
                if not chunk:
                    continue
            elif isinstance(chunk, lk_llm.ChatChunk) and chunk.delta and chunk.delta.content:
                chunk.delta.content = strip_thinking_tokens(chunk.delta.content, state) or ""
            yield chunk
        tail = strip_thinking_tokens(None, state, final=True)
        if tail:
            yield tail

    @function_tool
    async def update_lead(
        self,
        status: Status | None = None,
        budget: int | None = None,
        preferred_area: str | None = None,
        property_type: PropertyKind | None = None,
        timeline: str | None = None,
        interest_level: Interest | None = None,
    ) -> str:
        """Save facts about the lead to the CRM as soon as the caller tells you.
        Only pass the fields you just learned; leave the others empty.

        Args:
            status: Lead status after what the caller said.
            budget: Budget in whole US dollars, e.g. 750000.
            preferred_area: Area or neighbourhood the caller prefers.
            property_type: Type of property the caller wants.
            timeline: When they plan to buy, in their words, e.g. "within 3 months".
            interest_level: hot (<3 months), warm (3-12 months), cold (later or unsure).
        """
        return await self._lead_tools.update_lead(
            status=status,
            budget=budget,
            preferred_area=preferred_area,
            property_type=property_type,
            timeline=timeline,
            interest_level=interest_level,
        )

    @function_tool
    async def schedule_callback(self, when: str) -> str:
        """Book a callback when the caller asks to be called at another time.

        Args:
            when: The agreed day and time as an ISO 8601 datetime, e.g. 2026-10-14T16:00.
        """
        return await self._lead_tools.schedule_callback(when)

    @function_tool
    async def mark_do_not_call(self) -> str:
        """Use when the caller asks not to be called again."""
        return await self._lead_tools.mark_do_not_call()


def resolve_lead_id(ctx: JobContext, default: str) -> str:
    if ctx.job.metadata:
        try:
            lead_id = json.loads(ctx.job.metadata).get("lead_id")
            if lead_id:
                return str(lead_id)
        except (json.JSONDecodeError, AttributeError):
            pass
    # "lead-L003" or "lead-L003-sim7" (load test) -> L003
    match = re.match(r"^lead-([A-Za-z0-9]+)", ctx.room.name)
    return match.group(1) if match else default


def transcript_from(session: AgentSession) -> list[dict]:
    turns = []
    for item in session.history.items:
        if getattr(item, "type", None) != "message" or item.role not in ("user", "assistant"):
            continue
        text = item.text_content
        if text:
            turns.append({"role": "agent" if item.role == "assistant" else "user", "text": text})
    return turns


server = AgentServer(
    # accept many calls per worker during load tests; tune per instance size
    load_threshold=float(os.getenv("AGENT_LOAD_THRESHOLD", "0.9")),
    num_idle_processes=int(os.getenv("AGENT_IDLE_PROCESSES", "3")),
)


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    settings = Settings()
    crm = make_crm()
    lead_id = resolve_lead_id(ctx, settings.default_lead_id)
    lead = await crm.get_lead(lead_id)  # "check the lead" before saying a word
    ctx.log_context_fields = {"lead_id": lead_id, "room": ctx.room.name}
    logger.info("calling lead %s (%s), status=%s", lead.id, lead.name, lead.status.value)
    if lead.do_not_call:
        logger.warning("lead %s is on the do-not-call list; not starting the call", lead_id)
        ctx.shutdown(reason="do_not_call")
        return

    tools = LeadTools(crm, lead_id, tz=settings.timezone)
    call_metrics = CallMetrics()
    started_at = datetime.now(timezone.utc).isoformat()

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=openai.STT(
            base_url=settings.stt_base_url,
            api_key="not-needed",
            model=settings.stt_model,
            language="en",
            use_realtime=False,
            # vocabulary hint for Whisper (names are the hardest words); Parakeet ignores it
            prompt=f"Real estate phone call with {lead.name} about {lead.project_interest or 'property'}.",
        ),
        llm=openai.LLM(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            temperature=0.3,
            extra_body=no_think_body(settings.llm_model),
        ),
        tts=openai.TTS(
            base_url=settings.tts_base_url,
            api_key="not-needed",
            model=settings.tts_model,
            voice=settings.tts_voice,
            response_format="mp3",
        ),
        turn_handling={
            # "model" = LiveKit's local English turn-detector; "vad" = silence only
            "turn_detection": EnglishModel() if settings.turn_detection == "model" else "vad",
            # the default "adaptive" interruption detector calls LiveKit Cloud;
            # "vad" keeps barge-in detection on this machine
            "interruption": {"mode": "vad"},
            # start the LLM and TTS while the caller may still be finishing the sentence
            "preemptive_generation": {"enabled": True, "preemptive_tts": True},
        },
    )

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent) -> None:
        call_metrics.on_metrics(ev.metrics)

    llm_client = AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    bus = InProcessBus(PostCallProcessor(crm, llm_client, settings.llm_model, settings.calls_dir))

    published = False

    def publish_call_ended() -> None:
        nonlocal published
        if published:
            return
        published = True
        bus.publish(
            CallEnded(
                call_id=f"{ctx.room.name}-{started_at[:19].replace(':', '')}",
                lead_id=lead_id,
                transcript=transcript_from(session),
                actions=tools.actions,
                metrics=call_metrics.summary(),
                started_at=started_at,
            )
        )
        logger.info("latency summary: %s", call_metrics.summary())

    # write the CRM note as soon as the caller hangs up, not when the room closes
    session.on("close", lambda _ev: publish_call_ended())

    async def on_shutdown() -> None:
        publish_call_ended()
        await bus.drain()  # keep the process alive until the CRM write is done

    ctx.add_shutdown_callback(on_shutdown)

    await session.start(agent=LeadQualifierAgent(lead, tools, settings), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
