"""Rules checked before the agent phones a lead. A blocked dial is not made; it is written to
data/blocked/<YYYY-MM-DD>.jsonl so the dashboard can show it.

Rules (all settings, see agent/config.py):
  - never call a lead marked do-not-call, or whose status is in BLOCKED_STATUSES
    (default: not_interested, do_not_call, wrong_number);
  - at most MAX_CALLS_PER_DAY phone calls per lead per calendar day (CALL_TIMEZONE), default 1;
  - at most MAX_CALLS_PER_WEEK phone calls per lead in the last 7 days, default 2;
  - CALL_HOURS like "09:00-18:00" in CALL_TIMEZONE (empty = any time, 24/7).

Only phone calls count: a call record with a phone line (answered, busy, no answer, ...).
Browser and console tests have no line and don't count. Blocked dials never count.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from crm.port import Lead

from .telephony import mask_number

# call ids end with the start time in UTC: lead-L003-20261019100000-2026-10-19T100000(.raw).json
_STARTED = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{6})(?:\.raw)?\.json$")


@dataclass(frozen=True)
class Decision:
    ok: bool
    rule: str = ""      # blocked_status, do_not_call, daily_limit, weekly_limit, outside_hours
    detail: str = ""


def parse_hours(value: str) -> tuple[time, time] | None:
    """ "09:00-18:00" -> (09:00, 18:00); empty -> None (24/7)."""
    value = (value or "").strip()
    if not value:
        return None
    m = re.fullmatch(r"(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})", value)
    if not m:
        raise ValueError(f"CALL_HOURS must look like 09:00-18:00, not {value!r}")
    h1, m1, h2, m2 = (int(x) for x in m.groups())
    return time(h1, m1), time(h2, m2)


def _record_started(path: str, name: str) -> datetime | None:
    m = _STARTED.search(name)
    if m:
        return datetime.strptime(m.group(1), "%Y-%m-%dT%H%M%S").replace(tzinfo=timezone.utc)
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f).get("started_at")
        dt = datetime.fromisoformat(raw)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def _is_phone_call(path: str) -> bool:
    try:
        with open(path, encoding="utf-8") as f:
            return bool(json.load(f).get("line"))
    except (OSError, ValueError, AttributeError):
        return False


def phone_calls(calls_dir: str, lead_id: str) -> list[datetime]:
    """Start times (UTC) of this lead's phone calls: finished records and ones still being processed."""
    prefix = f"lead-{lead_id}-"
    out = []
    try:
        names = os.listdir(calls_dir)
    except OSError:
        return out
    for name in names:
        if not name.startswith(prefix) or not name.endswith(".json"):
            continue
        path = os.path.join(calls_dir, name)
        if not os.path.isfile(path) or not _is_phone_call(path):
            continue
        started = _record_started(path, name)
        if started:
            out.append(started)
    return out


def check_dial(lead: Lead, *, calls_dir: str, now: datetime, tz: str, max_per_day: int, max_per_week: int,
               blocked_statuses: frozenset[str], hours: str = "") -> Decision:
    status = getattr(lead.status, "value", str(lead.status))
    if lead.do_not_call:
        return Decision(False, "do_not_call", "the lead is on the do-not-call list")
    if status in blocked_statuses:
        return Decision(False, "blocked_status", f"the lead's status is {status}")
    zone = ZoneInfo(tz)
    local_now = now.astimezone(zone)
    window = parse_hours(hours)
    if window:
        start, end = window
        t = local_now.time()
        inside = start <= t < end if start <= end else (t >= start or t < end)  # e.g. 20:00-02:00
        if not inside:
            return Decision(False, "outside_hours", f"calls are allowed {hours} ({tz}); it is {t:%H:%M}")
    calls = phone_calls(calls_dir, lead.id)
    today = sum(1 for c in calls if c.astimezone(zone).date() == local_now.date())
    if max_per_day and today >= max_per_day:
        return Decision(False, "daily_limit", f"{today} call(s) today; the limit is {max_per_day} per day")
    week = sum(1 for c in calls if now - timedelta(days=7) < c <= now)
    if max_per_week and week >= max_per_week:
        return Decision(False, "weekly_limit", f"{week} call(s) in the last 7 days; the limit is {max_per_week}")
    return Decision(True)


def check_dial_with(settings, lead: Lead, now: datetime | None = None) -> Decision:
    """check_dial with the values from Settings."""
    return check_dial(lead, calls_dir=settings.calls_dir, now=now or datetime.now(timezone.utc),
                      tz=settings.timezone, max_per_day=settings.max_calls_per_day,
                      max_per_week=settings.max_calls_per_week, blocked_statuses=settings.blocked_statuses,
                      hours=settings.call_hours)


def record_blocked(blocked_dir: str, *, lead_id: str, number: str | None, rule: str, detail: str,
                   source: str, now: datetime | None = None) -> str:
    """Append one blocked dial to <blocked_dir>/<YYYY-MM-DD>.jsonl (UTC date). Returns the file path."""
    now = now or datetime.now(timezone.utc)
    os.makedirs(blocked_dir, exist_ok=True)
    path = os.path.join(blocked_dir, f"{now:%Y-%m-%d}.jsonl")
    line = {"ts": now.isoformat(timespec="seconds"), "lead_id": lead_id,
            "number": mask_number(number) if number else None, "rule": rule, "detail": detail, "source": source}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    return path
