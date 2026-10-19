"""Calling rules checked before dialing (agent/compliance.py)."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from agent.compliance import check_dial, parse_hours, phone_calls, record_blocked
from crm.port import Lead, LeadStatus

UAE = "Asia/Dubai"
NOW = datetime(2026, 10, 19, 10, 0, tzinfo=timezone.utc)  # 14:00 in Dubai, a Monday
STATUSES = frozenset({"not_interested", "do_not_call", "wrong_number"})


def lead(status=LeadStatus.NEW, dnc=False, lead_id="L003"):
    return Lead(id=lead_id, name="Test Lead", phone="+971501234567", status=status, do_not_call=dnc)


def write_call(folder, lead_id, started: datetime, *, phone=True, raw=False):
    call_id = f"lead-{lead_id}-{started:%Y%m%d%H%M%S}-{started:%Y-%m-%dT%H%M%S}"
    rec = {"lead_id": lead_id, "started_at": started.isoformat(),
           "line": {"result": "busy"} if phone else None}
    (folder / f"{call_id}{'.raw' if raw else ''}.json").write_text(json.dumps(rec), encoding="utf-8")


def check(folder, the_lead, **kw):
    args = dict(calls_dir=str(folder), now=NOW, tz=UAE, max_per_day=1, max_per_week=2, blocked_statuses=STATUSES)
    args.update(kw)
    return check_dial(the_lead, **args)


def test_new_lead_may_be_called(tmp_path):
    assert check(tmp_path, lead()).ok


@pytest.mark.parametrize("status", [LeadStatus.NOT_INTERESTED, LeadStatus.DO_NOT_CALL, LeadStatus.WRONG_NUMBER])
def test_blocked_statuses(tmp_path, status):
    d = check(tmp_path, lead(status))
    assert (d.ok, d.rule) == (False, "blocked_status")


def test_do_not_call_flag(tmp_path):
    d = check(tmp_path, lead(dnc=True))
    assert (d.ok, d.rule) == (False, "do_not_call")


def test_one_call_per_day_in_uae_time(tmp_path):
    write_call(tmp_path, "L003", NOW - timedelta(hours=3))           # 11:00 Dubai, same day
    d = check(tmp_path, lead())
    assert (d.ok, d.rule) == (False, "daily_limit")


def test_yesterday_in_uae_time_is_another_day(tmp_path):
    # 19:30 UTC on the 18th = 23:30 on the 18th in Dubai: yesterday there, so today is free
    write_call(tmp_path, "L003", datetime(2026, 10, 18, 19, 30, tzinfo=timezone.utc))
    assert check(tmp_path, lead()).ok


def test_two_calls_per_week(tmp_path):
    write_call(tmp_path, "L003", NOW - timedelta(days=2))
    write_call(tmp_path, "L003", NOW - timedelta(days=5))
    d = check(tmp_path, lead())
    assert (d.ok, d.rule) == (False, "weekly_limit")
    write_call(tmp_path, "L003", NOW - timedelta(days=9))             # older than 7 days: not counted
    assert len(phone_calls(str(tmp_path), "L003")) == 3


def test_calls_older_than_a_week_dont_count(tmp_path):
    write_call(tmp_path, "L003", NOW - timedelta(days=8))
    write_call(tmp_path, "L003", NOW - timedelta(days=10))
    assert check(tmp_path, lead()).ok


def test_calls_still_being_processed_count(tmp_path):
    write_call(tmp_path, "L003", NOW - timedelta(hours=1), raw=True)
    assert check(tmp_path, lead()).rule == "daily_limit"


def test_browser_tests_and_other_leads_dont_count(tmp_path):
    write_call(tmp_path, "L003", NOW - timedelta(hours=1), phone=False)
    write_call(tmp_path, "L0031", NOW - timedelta(hours=1))           # another lead whose id starts the same
    assert check(tmp_path, lead()).ok


def test_limits_are_settings(tmp_path):
    write_call(tmp_path, "L003", NOW - timedelta(hours=1))
    assert check(tmp_path, lead(), max_per_day=2).ok
    assert check(tmp_path, lead(), max_per_day=0, max_per_week=0).ok   # 0 = no limit


def test_calling_hours(tmp_path):
    assert check(tmp_path, lead(), hours="").ok                         # 24/7
    assert check(tmp_path, lead(), hours="09:00-18:00").ok             # 14:00 in Dubai
    d = check(tmp_path, lead(), hours="15:00-18:00")
    assert (d.ok, d.rule) == (False, "outside_hours")
    assert check(tmp_path, lead(), hours="13:00-02:00").ok             # across midnight
    with pytest.raises(ValueError):
        parse_hours("9 to 5")


def test_blocked_dial_is_written_with_a_masked_number(tmp_path):
    path = record_blocked(str(tmp_path / "blocked"), lead_id="L003", number="+971501234567",
                          rule="daily_limit", detail="1 call(s) today", source="dial", now=NOW)
    line = json.loads(open(path, encoding="utf-8").read().strip())
    assert path.endswith("2026-10-19.jsonl")
    assert line["number"] == "+97150*****67"
    assert (line["lead_id"], line["rule"], line["source"]) == ("L003", "daily_limit", "dial")
