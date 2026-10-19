"""Celery: post-call work runs here, off the call path.

    celery -A agent.tasks worker -c 4 --loglevel INFO     # the postcall worker (docker: postcall-worker)

Broker and result store: Redis (REDIS_URL). The agent sends one `process_call` task per call
(agent/events.py: CeleryBus). The task:
  1. checks the transcript with the LLM (structured extraction on vLLM),
  2. fills CRM fields the agent missed, enforces do-not-call, writes the summary note,
  3. links the recording and copies the call's log lines into data/calls/<call_id>/,
  4. saves the final data/calls/<call_id>.json (the dashboard reads it).

A task that fails (vLLM or CRM briefly down) is retried 3 times with growing waits.
`acks_late` means a worker that dies mid-task leaves the task in Redis for another worker.
"""

from __future__ import annotations

import asyncio
import logging
import os

from celery import Celery
from celery.signals import after_setup_logger, task_prerun
from dotenv import load_dotenv

load_dotenv()

app = Celery("voice_agent", broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
             backend=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_time_limit=300,
    task_soft_time_limit=240,
    result_expires=7 * 24 * 3600,
    task_serializer="json",
    accept_content=["json"],
    broker_connection_retry_on_startup=True,
    task_default_queue="postcall",
)

logger = logging.getLogger("voice-agent.tasks")


@after_setup_logger.connect
def _json_logs(logger=None, **_):  # noqa: ARG001 - Celery passes its own logger
    from .logging_setup import setup_logging

    setup_logging("postcall")


@task_prerun.connect
def _tag_call(task=None, args=None, **_):  # noqa: ARG001
    from .logging_setup import bind_call

    if args and isinstance(args[0], dict):
        payload = args[0]
        bind_call(call_id=payload.get("call_id"), lead_id=payload.get("lead_id"), room=payload.get("room"))


async def _process(payload: dict) -> dict:
    from openai import AsyncOpenAI

    from crm import make_crm

    from .config import Settings
    from .events import CallEnded, PostCallProcessor

    settings = Settings()
    llm = AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    processor = PostCallProcessor(make_crm(), llm, settings.llm_model, settings.calls_dir,
                                  logs_dir=settings.logs_dir, recordings_dir=settings.recordings_dir,
                                  wait_for_recording=float(os.getenv("RECORDING_WAIT_SECONDS", "20")))
    try:
        return await processor.handle(CallEnded(**payload))
    finally:
        await llm.close()


@app.task(name="voice_agent.process_call", bind=True, max_retries=3, acks_late=True)
def process_call(self, payload: dict) -> dict:
    """Post-call work for one call. Returns a short summary (the full record is on disk)."""
    try:
        record = asyncio.run(_process(payload))
    except Exception as exc:  # the processor saves what it can itself; this is for crashes before that
        logger.exception("post-call task failed for %s", payload.get("call_id"))
        raise self.retry(exc=exc, countdown=5 * 2 ** self.request.retries)
    return {"call_id": record.get("call_id"), "classification": record.get("classification"),
            "error": record.get("error")}
