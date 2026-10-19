"""LiveKit agent worker: the cascaded voice pipeline.

    python -m agent.main download-files   # one-time: VAD + turn-detector weights
    python -m agent.main dev              # connect to LiveKit, wait for calls
    python -m agent.main start            # same, production mode (docker: agent service)
    python -m agent.talk --lead L003      # talk to it by voice from a terminal, through LiveKit
    python -m agent.main console          # talk to it in the terminal without a LiveKit server
    python -m agent.dial --lead L003      # phone the lead through 3CX (SIP_OUTBOUND_TRUNK_ID)

Which lead is called: job metadata {"lead_id": "L003"} (explicit dispatch), else a room
named "lead-L003" (or "lead-L003-anything"), else DEFAULT_LEAD_ID.
Phone calls: job or room metadata {"dial": true} (the lead's CRM number) or {"phone_number": "+971..."}.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from typing import Literal
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    ChatContext,
    ModelSettings,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    FunctionToolsExecutedEvent,
    RunContext,
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
from crm.audited import AuditedCRM

from . import telephony
from .calllog import CallLogger
from .config import Settings, llm_server_info, no_think_body, session_conn_options, warm_up_llm
from .events import CallEnded, CeleryBus, InProcessBus, PostCallProcessor
from .logging_setup import bind_call, setup_logging
from .metrics import CallMetrics
from .prompts import build_system_prompt, greeting_instructions, prompt_version
from .tools import LeadTools

load_dotenv()
logger = logging.getLogger("voice-agent")

Status = Literal["interested", "qualified", "not_interested", "wrong_number"]
PropertyKind = Literal["apartment", "villa", "townhouse", "office", "land"]
Interest = Literal["hot", "warm", "cold"]


class LeadQualifierAgent(Agent):
    def __init__(self, lead: Lead, tools: LeadTools, settings: Settings, recorded: bool = False,
                 call_log: CallLogger | None = None) -> None:
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
        self._recorded = recorded
        self._call_log = call_log

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=greeting_instructions(
                self._lead, company=self._settings.company_name, agent_name=self._settings.agent_name,
                recorded=self._recorded,
            )
        )

    async def llm_node(self, chat_ctx: ChatContext, tools: list[lk_llm.Tool], model_settings: ModelSettings):
        """Default LLM step, minus any <think>...</think> text, so it is never spoken.

        Every request is also saved to data/calls/<call>/llm.jsonl: the messages sent, the reply,
        tool calls, time to first token, and (joined later) LiveKit's token counts."""
        state = ThinkingTokenFilter()
        log = self._call_log
        rec = log.llm_started(chat_ctx=chat_ctx, tools=tools) if log else None
        cancelled, error = True, None
        try:
            async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
                if isinstance(chunk, str):
                    raw = chunk
                    chunk = strip_thinking_tokens(chunk, state) or ""
                    if rec is not None:
                        rec["thinking_removed_chars"] += len(raw) - len(chunk)
                    if not chunk:
                        continue
                elif isinstance(chunk, lk_llm.ChatChunk) and chunk.delta and chunk.delta.content:
                    raw = chunk.delta.content
                    chunk.delta.content = strip_thinking_tokens(chunk.delta.content, state) or ""
                    if rec is not None:
                        rec["thinking_removed_chars"] += len(raw) - len(chunk.delta.content)
                if rec is not None:
                    log.llm_chunk(rec, chunk)
                yield chunk
            tail = strip_thinking_tokens(None, state, final=True)
            if tail:
                if rec is not None:
                    log.llm_chunk(rec, tail)
                yield tail
            cancelled = False
        except Exception as e:
            cancelled, error = False, e
            raise
        finally:  # also runs when the caller interrupts and LiveKit closes this stream
            if rec is not None:
                log.llm_finished(rec, cancelled=cancelled, error=error)

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

    @function_tool
    async def transfer_to_human(self, context: RunContext, reason: str) -> str:
        """Connect the caller to a person when they ask for one, or ask something only a
        consultant can answer (exact prices, contracts, payment plans).

        Args:
            reason: Why the caller needs a person, in a few words.
        """
        await context.wait_for_playout()  # let "I'll connect you" finish before the line changes
        return await self._lead_tools.transfer_to_human(reason)


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


def outbound_number(ctx: JobContext, lead: Lead) -> str | None:
    """The number to phone, if this job is a phone call: {"phone_number": ...} or {"dial": true}."""
    for raw in (ctx.job.metadata, getattr(ctx.job.room, "metadata", "")):
        try:
            meta = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(meta, dict) and meta.get("phone_number"):
            return str(meta["phone_number"])
        if isinstance(meta, dict) and meta.get("dial"):
            return lead.phone
    return None


# LiveKit's reason the session closed -> who ended the call (when the agent's end_call didn't)
ENDED_BY = {"participant_disconnected": "lead", "user_initiated": "lead", "error": "error",
            "job_shutdown": "shutdown", "task_completed": "agent"}


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
    # LiveKit is self-hosted on the LAN: talk to it directly. LiveKit would otherwise send its
    # connection through HTTPS_PROXY (it ignores NO_PROXY), which breaks behind a company proxy.
    http_proxy=os.getenv("LIVEKIT_HTTP_PROXY") or None,
)


def versions() -> dict:
    """Which code, prompt and libraries ran this call (saved in every call record)."""
    import importlib.metadata as md
    import subprocess

    sha = os.getenv("GIT_SHA", "")
    if not sha:
        try:
            sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                 timeout=2, cwd=os.path.dirname(__file__)).stdout.strip()
        except Exception:
            sha = ""

    def ver(pkg: str) -> str | None:
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return None

    return {"code": sha or None, "prompt": prompt_version(), "livekit_agents": ver("livekit-agents"),
            "livekit_plugins_openai": ver("livekit-plugins-openai"), "voice_agent": ver("voice-agent")}


def _result(task: asyncio.Future, default):
    """A background lookup's result, or `default` if it isn't done or failed."""
    if task.done() and not task.cancelled() and task.exception() is None:
        return task.result()
    return default


def call_settings(settings: Settings, vad) -> dict:
    """The knobs that change how a call behaves, saved with the call so runs can be compared."""
    vad_opts = getattr(vad, "_opts", None)
    return {
        "llm_temperature": settings.llm_temperature, "llm_timeout": settings.llm_timeout,
        "llm_max_retry": settings.llm_max_retry, "thinking": False,
        "turn_detection": settings.turn_detection, "interruption": "vad", "preemptive_generation": True,
        "vad": {k: getattr(vad_opts, k, None) for k in ("min_speech_duration", "min_silence_duration",
                                                         "activation_threshold", "prefix_padding_duration")}
        if vad_opts else None,
        "stt_language": "en", "tts_format": "mp3", "recording": settings.recording_enabled,
        "post_call": "celery" if settings.redis_url else "in_process",
    }


def make_bus(settings: Settings, crm, llm_client: AsyncOpenAI):
    """Celery (REDIS_URL set) or in-process post-call work; Celery falls back to in-process if Redis is down."""
    in_process = InProcessBus(PostCallProcessor(crm, llm_client, settings.llm_model, settings.calls_dir,
                                                logs_dir=settings.logs_dir, recordings_dir=settings.recordings_dir))
    if settings.redis_url:
        return CeleryBus(settings.calls_dir, fallback=in_process)
    return in_process


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()
    # load the LLM into memory in the background (can take 10-30 s on a laptop)
    threading.Thread(target=warm_up_llm, args=(Settings(),), daemon=True).start()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    settings = Settings()
    started_at = datetime.now(timezone.utc).isoformat()
    lead_id = resolve_lead_id(ctx, settings.default_lead_id)
    call_id = f"{ctx.room.name}-{started_at[:19].replace(':', '')}"
    # every log line of this call carries these fields (logs/agent/<date>.jsonl)
    ctx.log_context_fields = {"call_id": call_id, "lead_id": lead_id, "room": ctx.room.name}
    bind_call(call_id=call_id, lead_id=lead_id, room=ctx.room.name)
    call_log = CallLogger(settings.calls_dir, call_id, log_llm_messages=settings.log_llm_messages)
    # every CRM write during the call is saved with the lead before and after (crm_changes.jsonl)
    crm = AuditedCRM(make_crm(), call_log.crm_change, source="call")
    try:
        lead = await crm.get_lead(lead_id)  # "check the lead" before saying a word
    except Exception as e:
        call_log.event("error", stage="crm", error=repr(e), at_start=True)
        call_log.close(reason="lead_lookup_failed")
        raise
    call_log.event("call_started", lead_id=lead_id, lead_name=lead.name, lead_status=lead.status.value,
                   room=ctx.room.name, job_id=getattr(ctx.job, "id", None))
    logger.info("calling lead %s (%s), status=%s", lead.id, lead.name, lead.status.value)
    if lead.do_not_call:
        logger.warning("lead %s is on the do-not-call list; not starting the call", lead_id)
        call_log.close(reason="do_not_call")
        ctx.shutdown(reason="do_not_call")
        return
    # looked up in the background so the greeting isn't delayed; read when the call ends
    llm_info_task = asyncio.ensure_future(asyncio.to_thread(llm_server_info, settings))
    versions_task = asyncio.ensure_future(asyncio.to_thread(versions))

    call: dict = {"ended_by": "", "close_reason": "", "tool_calls": [], "line": None, "recording": None}

    async def transfer_call(reason: str) -> dict:
        if not settings.human_transfer_to:
            return {"ok": False, "error": "HUMAN_TRANSFER_TO is not set"}
        caller = next((p for p in ctx.room.remote_participants.values()
                       if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP), None)
        if caller is None:
            return {"ok": False, "error": "this call has no phone line (browser or console test)"}
        result = await telephony.transfer(ctx.api, room=ctx.room.name, identity=caller.identity,
                                          to=settings.human_transfer_to)
        if result.get("ok"):
            call["ended_by"] = "transferred"
        return result

    tools = LeadTools(crm, lead_id, tz=settings.timezone, transfer=transfer_call)
    call_metrics = CallMetrics()

    session = AgentSession(
        conn_options=session_conn_options(settings),
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
            temperature=settings.llm_temperature,
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

    call_log.attach(session)  # timeline, STT/LLM/TTS metrics, tool calls, errors -> data/calls/<call>/

    @session.on("function_tools_executed")
    def _on_tools(ev: FunctionToolsExecutedEvent) -> None:
        # every tool the LLM called, with what it returned (the CRM writes alone are in tools.actions)
        for fc, out in zip(ev.function_calls, ev.function_call_outputs):
            call["tool_calls"].append({
                "name": fc.name, "arguments": fc.arguments,
                "output": (out.output or "")[:500] if out else None,
                "is_error": bool(out and out.is_error),
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            if fc.name == "end_call":
                call["ended_by"] = "agent"

    llm_client = AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    bus = make_bus(settings, crm.inner, llm_client)
    vad = ctx.proc.userdata["vad"]

    published = False

    def publish_call_ended() -> None:
        nonlocal published
        if published:
            return
        published = True
        ended_by = call["ended_by"] or ENDED_BY.get(call["close_reason"], call["close_reason"])
        call_log.close(ended_by=ended_by, close_reason=call["close_reason"])
        llm_info = _result(llm_info_task, {"served_by": "vllm", "model": settings.llm_model})
        bus.publish(
            CallEnded(
                call_id=call_id,
                lead_id=lead_id,
                transcript=transcript_from(session),
                actions=tools.actions,
                metrics={**call_metrics.summary(), **call_log.summary()},
                started_at=started_at,
                room=ctx.room.name,
                models={"llm": settings.llm_model, "stt": settings.stt_model, "tts": settings.tts_model,
                        "tts_voice": settings.tts_voice, "llm_server": llm_info,
                        "served_by": llm_info.get("served_by")},
                ended_by=ended_by,
                close_reason=call["close_reason"],
                tool_calls=call["tool_calls"],
                line=call["line"],
                recording=call["recording"],
                transfer=tools.transfer,
                versions=_result(versions_task, {}),
                settings=call_settings(settings, vad),
                errors=call_log.errors,
                log_files=call_log.files(),
            )
        )
        logger.info("latency summary: %s", call_metrics.summary())

    def _on_close(ev) -> None:
        call["close_reason"] = getattr(ev.reason, "value", str(ev.reason))
        publish_call_ended()

    # write the CRM note as soon as the caller hangs up, not when the room closes
    session.on("close", _on_close)

    async def on_shutdown() -> None:
        publish_call_ended()
        await bus.drain()  # keep the process alive until the CRM write is done

    ctx.add_shutdown_callback(on_shutdown)

    number = outbound_number(ctx, lead)
    if number:
        # a phone call through 3CX: ring the lead and only start talking once they answer
        if settings.sip_outbound_trunk_id:
            await ctx.connect()
            call["line"] = await telephony.dial(
                ctx.api, room=ctx.room.name, number=number, trunk_id=settings.sip_outbound_trunk_id,
                identity=f"sip-{lead_id}", name=lead.name, ring_timeout=settings.ring_timeout)
        else:
            call["line"] = {"result": "failed", "label": telephony.LINE_RESULTS["failed"],
                            "number": telephony.mask_number(number),
                            "sip_status": "SIP_OUTBOUND_TRUNK_ID is not set"}
        if call["line"]["result"] != "answered":
            call["ended_by"] = "not_reached"
            publish_call_ended()
            ctx.shutdown(reason=f"line_{call['line']['result']}")
            return

    if settings.recording_enabled:
        call["recording"] = await telephony.start_recording(
            ctx.api, room=ctx.room.name, s3=settings.recording_s3(),
            filepath=settings.recording_filepath.format(room=ctx.room.name, lead=lead_id))

    recorded = bool(call["recording"] and call["recording"].get("ok"))
    call_log.event("recording", **(call["recording"] or {"ok": False, "error": "RECORDING_ENABLED=0"}))
    await session.start(agent=LeadQualifierAgent(lead, tools, settings, recorded=recorded, call_log=call_log),
                        room=ctx.room)


if __name__ == "__main__":
    # JSON log lines for the worker and every call: logs/agent/<date>.jsonl. Call processes forward
    # their lines here (tagged with call_id by ctx.log_context_fields), so only this process writes.
    setup_logging("agent")
    cli.run_app(server)
