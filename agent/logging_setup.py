"""JSON log lines for every service, one file per service per day.

    logs/<service>/<YYYY-MM-DD>.jsonl   e.g. logs/agent/2026-10-19.jsonl, logs/postcall/...

Every line carries the UTC time, level, logger, service, and - while a call is running -
its call_id, room and lead_id, so one call's lines can be pulled out of a busy day:

    grep '"call_id": "lead-L003-20261019T101500"' logs/agent/2026-10-19.jsonl

Inside the agent worker, LiveKit tags each call's lines itself (ctx.log_context_fields) and
forwards them from the call's process to the main worker process, which alone writes the file.
Elsewhere (post-call worker, console mode) the fields come from a contextvar set with `bind_call(...)`.
"""

from __future__ import annotations

import contextvars
import json
import logging
import logging.handlers
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

# read-only default: bind_call always sets a new dict, so contexts never share one
_call_fields: contextvars.ContextVar[Mapping] = contextvars.ContextVar("call_fields", default=MappingProxyType({}))
_configured: set[str] = set()

# standard LogRecord attributes: everything else passed in `extra=` is kept in the JSON line
_STANDARD = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime", "taskName"}


def bind_call(**fields) -> None:
    """Tag every log line from now on (in this context) with these call fields."""
    _call_fields.set({**_call_fields.get(), **{k: v for k, v in fields.items() if v is not None}})


def call_fields() -> dict:
    return dict(_call_fields.get())


class CallContextFilter(logging.Filter):
    def __init__(self, service: str) -> None:
        super().__init__()
        self.service = service

    def filter(self, record: logging.LogRecord) -> bool:
        record.service = self.service
        for key, value in _call_fields.get().items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD and key not in line and not key.startswith("_"):
                line[key] = value
        if record.exc_info:
            line["exc"] = "".join(traceback.format_exception(*record.exc_info)).strip()
        return json.dumps(line, default=str, ensure_ascii=False)


class DailyJsonlHandler(logging.handlers.TimedRotatingFileHandler):
    """logs/<service>/<date>.jsonl, a new file at UTC midnight, `backup_days` files kept."""

    def __init__(self, folder: Path, backup_days: int = 30) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.folder = folder
        super().__init__(folder / f"{datetime.now(timezone.utc):%Y-%m-%d}.jsonl", when="midnight",
                         utc=True, backupCount=backup_days, encoding="utf-8", delay=True)

    def doRollover(self) -> None:  # name the new file after the new day instead of renaming the old one
        if self.stream:
            self.stream.close()
            self.stream = None  # type: ignore[assignment]
        self.baseFilename = str(self.folder / f"{datetime.now(timezone.utc):%Y-%m-%d}.jsonl")
        if self.backupCount > 0:  # 0 = keep every day
            for old in sorted(self.folder.glob("*.jsonl"))[:-self.backupCount]:
                old.unlink(missing_ok=True)
        self.rolloverAt = self.computeRollover(int(datetime.now(timezone.utc).timestamp()))


def setup_logging(service: str, *, logs_dir: str | None = None, level: str | None = None,
                  console: bool = False) -> Path:
    """Send every logger's output to logs/<service>/<date>.jsonl (plus stderr if `console`).

    Safe to call more than once per process; the second call is a no-op for the same service."""
    folder = Path(logs_dir or os.getenv("LOGS_DIR", "logs")) / service
    if service in _configured:
        return folder
    _configured.add(service)
    root = logging.getLogger()
    root.setLevel(level or os.getenv("LOG_LEVEL", "INFO"))
    handler = DailyJsonlHandler(folder, backup_days=int(os.getenv("LOG_KEEP_DAYS", "30")))
    handler.setFormatter(JsonFormatter())
    handler.addFilter(CallContextFilter(service))
    root.addHandler(handler)
    if console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(stream)
    # chatty libraries: keep their warnings, drop their debug noise
    for noisy in ("httpx", "httpcore", "openai._base_client", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return folder
