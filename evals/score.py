"""Field-level scoring of a CallOutcome against the expected JSON of a test case."""

from __future__ import annotations

from agent.extraction import CallOutcome

EXACT_FIELDS = ("status", "budget", "property_type", "interest_level")
FREE_TEXT_FIELDS = ("preferred_area", "timeline")


def score_case(expected: dict, got: CallOutcome) -> list[dict]:
    """One row per checked field: {field, expected, got, ok}."""
    rows = []
    got_update = got.lead_update.changes()
    exp_update = expected.get("lead_update", {})

    rows.append(_row("outcome", expected["outcome"], got.outcome.value))
    rows.append(_row("callback_requested", expected.get("callback_requested", False), got.callback_requested))

    for f in EXACT_FIELDS:
        exp, val = exp_update.get(f), got_update.get(f)
        if exp is None and val is None:
            continue
        # a value the caller never gave counts as a hallucination
        rows.append(_row(f, exp, val))

    for f in FREE_TEXT_FIELDS:
        exp, val = exp_update.get(f), got_update.get(f)
        if exp is None:
            continue  # free text: only score fields the caller actually gave
        if f == "preferred_area":
            ok = val is not None and exp.lower() in val.lower()
        else:  # timeline wording varies ("2 months" vs "two months"): presence is enough
            ok = val is not None
        rows.append({"field": f, "expected": exp, "got": val, "ok": ok})
    return rows


def _row(field: str, expected, got) -> dict:
    ok = (str(expected).lower() == str(got).lower()) if expected is not None else got is None
    return {"field": field, "expected": expected, "got": got, "ok": ok}
