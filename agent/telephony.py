"""Phone line (3CX via a LiveKit SIP outbound trunk), transfer to a person, and call recording.

Kept apart from the voice pipeline so each piece can be tested with a fake LiveKit API.

  dial()       calls the lead and waits until they answer; returns the line result
  transfer()   hands the caller to a person (SIP REFER to a 3CX extension or number)
  start_recording()  records the room's audio with LiveKit Egress
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from google.protobuf.duration_pb2 import Duration
from livekit import api

logger = logging.getLogger("voice-agent.telephony")

# SIP response code -> line result (RFC 3261 meanings)
LINE_RESULTS = {
    "answered": "Answered",
    "busy": "Busy",
    "no_answer": "No answer",
    "declined": "Declined",
    "invalid_number": "Invalid number",
    "not_allowed": "Trunk refused the call",
    "line_error": "Line or 3CX error",
    "failed": "Failed",
}


def classify_sip(code: int) -> str:
    if code in (486, 600):
        return "busy"
    if code in (408, 480, 487):
        return "no_answer"
    if code == 603:
        return "declined"
    if code in (404, 410, 484, 485, 604):
        return "invalid_number"
    if code in (401, 403, 407):
        return "not_allowed"
    if 500 <= code < 600:
        return "line_error"
    return "failed"


def mask_number(number: str) -> str:
    """+971501234567 -> +97150*****67: enough to recognise, not enough to dial."""
    n = (number or "").strip()
    if len(n) <= 6:
        return "*" * len(n)
    return n[:6] + "*" * (len(n) - 8) + n[-2:]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


async def dial(lk: api.LiveKitAPI, *, room: str, number: str, trunk_id: str, identity: str,
               name: str = "", ring_timeout: int = 30) -> dict:
    """Call `number` into `room` and wait for an answer. Never raises: the result says what happened."""
    started = time.monotonic()
    line = {"number": mask_number(number), "trunk_id": trunk_id, "dialed_at": _now(), "participant": identity}
    try:
        await lk.sip.create_sip_participant(api.CreateSIPParticipantRequest(
            sip_trunk_id=trunk_id, sip_call_to=number, room_name=room, participant_identity=identity,
            participant_name=name or identity, wait_until_answered=True, play_dialtone=False,
            ringing_timeout=Duration(seconds=ring_timeout),
        ))
        line.update(result="answered", sip_status_code=200, sip_status="OK")
    except api.TwirpError as e:
        meta = e.metadata or {}
        try:
            code = int(meta.get("sip_status_code") or 0)
        except ValueError:
            code = 0
        line.update(result=classify_sip(code) if code else "failed", sip_status_code=code or None,
                    sip_status=meta.get("sip_status") or e.message)
    except Exception as e:  # network, bad trunk id...: still a line result, not a crash
        line.update(result="failed", sip_status_code=None, sip_status=repr(e))
    line["ring_seconds"] = round(time.monotonic() - started, 1)
    line["label"] = LINE_RESULTS[line["result"]]
    logger.info("dial %s -> %s (%s)", line["number"], line["result"], line.get("sip_status"))
    return line


async def transfer(lk: api.LiveKitAPI, *, room: str, identity: str, to: str) -> dict:
    """Cold-transfer the caller to `to` ("tel:+9714..." or "sip:201@3cx.example.com")."""
    out = {"to": to, "at": _now()}
    try:
        await lk.sip.transfer_sip_participant(api.TransferSIPParticipantRequest(
            room_name=room, participant_identity=identity, transfer_to=to, play_dialtone=False))
        out.update(ok=True)
    except api.TwirpError as e:
        out.update(ok=False, error=e.message, sip_status_code=(e.metadata or {}).get("sip_status_code"))
    except Exception as e:
        out.update(ok=False, error=repr(e))
    return out


async def start_recording(lk: api.LiveKitAPI, *, room: str, filepath: str, s3: dict | None = None) -> dict:
    """Audio-only recording of the whole room (both sides). Needs the LiveKit Egress service."""
    output = api.EncodedFileOutput(file_type=api.EncodedFileType.OGG, filepath=filepath)
    if s3:
        output.s3.CopyFrom(api.S3Upload(**s3))
    try:
        info = await lk.egress.start_room_composite_egress(
            api.RoomCompositeEgressRequest(room_name=room, audio_only=True, file_outputs=[output]))
        return {"ok": True, "egress_id": info.egress_id, "file": filepath, "started_at": _now(),
                "storage": "s3" if s3 else "egress server disk"}
    except Exception as e:  # recording is never a reason to drop the call
        logger.warning("recording did not start: %r", e)
        return {"ok": False, "error": getattr(e, "message", repr(e)), "file": filepath}
