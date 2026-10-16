"""Speech accuracy: how well speech-to-text understands callers, and how clearly text-to-speech speaks.

    python -m evals.run_speech                         # round trip: TTS speaks each sentence, STT writes it back
    python -m evals.run_speech --voice am_michael      # a different caller voice
    python -m evals.run_speech --audio-dir my_calls/   # real recordings: a.wav + a.txt (what was really said)
    python -m evals.run_speech --audio-dir my_calls/ --language ar

Round trip measures both models together: if a word comes back wrong, either TTS said it unclearly or
STT misheard it. Real recordings (people's own voices, accents, phone-line audio) measure STT alone and
are the result to trust; record a few real calls (with consent) and type what was said.

Scores: word error rate (WER, lower is better), accuracy = 1 - WER, and key-term accuracy: names,
areas and amounts that must come through exactly, because they end up in the CRM.
Results are saved to data/speech_evals/ for the dashboard.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

from agent.config import Settings

from .speech_score import key_terms_found, word_errors

SENTENCES = Path(__file__).resolve().parent / "speech_sentences.json"
OUT_DIR = Path("data/speech_evals")
TARGET_ACCURACY = 0.90      # 1 - WER
TARGET_KEY_TERMS = 0.95     # names, areas, amounts


def synthesize(client: httpx.Client, s: Settings, text: str, voice: str) -> tuple[bytes, float]:
    t0 = time.perf_counter()
    r = client.post(f"{s.tts_base_url}/audio/speech",
                    json={"model": s.tts_model, "voice": voice, "response_format": "wav", "input": text})
    r.raise_for_status()
    return r.content, time.perf_counter() - t0


def transcribe(client: httpx.Client, s: Settings, audio: bytes, language: str, name: str = "a.wav") -> tuple[str, float]:
    t0 = time.perf_counter()
    r = client.post(f"{s.stt_base_url}/audio/transcriptions",
                    files={"file": (name, audio, "audio/wav")},
                    data={"model": s.stt_model, "language": language})
    r.raise_for_status()
    return (r.json().get("text") or "").strip(), time.perf_counter() - t0


def _wav_seconds(audio: bytes) -> float | None:
    import io
    import wave
    try:
        with wave.open(io.BytesIO(audio)) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return None


def score_item(item_id: str, reference: str, transcript: str, key: list[str]) -> dict:
    errors, n = word_errors(reference, transcript)
    found, missed = key_terms_found(key, transcript)
    return {"id": item_id, "reference": reference, "transcript": transcript, "errors": errors, "ref_words": n,
            "wer": round(errors / n, 3) if n else None, "key_terms": len(key), "key_terms_found": len(found),
            "missed_key_terms": missed}


def summarize(items: list[dict]) -> dict:
    errors = sum(i["errors"] for i in items)
    words = sum(i["ref_words"] for i in items)
    keys = sum(i["key_terms"] for i in items)
    keys_ok = sum(i["key_terms_found"] for i in items)
    w = errors / words if words else None
    acc = 1 - w if w is not None else None
    key_acc = keys_ok / keys if keys else None

    def avg(field):
        vals = [i[field] for i in items if i.get(field) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    return {
        "items": len(items), "words": words, "wer": round(w, 4) if w is not None else None,
        "accuracy": round(acc, 4) if acc is not None else None,
        "key_term_accuracy": round(key_acc, 4) if key_acc is not None else None,
        "stt_seconds_avg": avg("stt_seconds"), "tts_seconds_avg": avg("tts_seconds"),
        "tts_realtime_factor": avg("tts_realtime_factor"),
        "verdict": ("GOOD ENOUGH" if acc is not None and acc >= TARGET_ACCURACY
                    and (key_acc is None or key_acc >= TARGET_KEY_TERMS) else "NOT ENOUGH"),
        "targets": {"accuracy": TARGET_ACCURACY, "key_term_accuracy": TARGET_KEY_TERMS},
    }


def run_round_trip(client: httpx.Client, s: Settings, voice: str, language: str) -> list[dict]:
    items = []
    for sent in json.loads(SENTENCES.read_text()):
        audio, tts_s = synthesize(client, s, sent["text"], voice)
        text, stt_s = transcribe(client, s, audio, language)
        row = score_item(sent["id"], sent["text"], text, sent.get("key", []))
        secs = _wav_seconds(audio)
        row.update(tts_seconds=round(tts_s, 3), stt_seconds=round(stt_s, 3), audio_seconds=secs and round(secs, 2),
                   tts_realtime_factor=round(tts_s / secs, 3) if secs else None)
        items.append(row)
    return items


def run_recordings(client: httpx.Client, s: Settings, folder: Path, language: str) -> list[dict]:
    items = []
    for audio_path in sorted(folder.glob("*.wav")):
        ref_path = audio_path.with_suffix(".txt")
        if not ref_path.exists():
            print(f"skipped {audio_path.name}: no {ref_path.name} with what was really said")
            continue
        lines = ref_path.read_text(encoding="utf-8").splitlines()
        reference = lines[0].strip() if lines else ""
        key = [k.strip() for k in lines[1].removeprefix("key:").split(",")] if len(lines) > 1 and lines[1].startswith("key:") else []
        text, stt_s = transcribe(client, s, audio_path.read_bytes(), language, audio_path.name)
        row = score_item(audio_path.stem, reference, text, key)
        row.update(stt_seconds=round(stt_s, 3), audio_seconds=_wav_seconds(audio_path.read_bytes()))
        items.append(row)
    return items


def save(result: dict, out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result['run_at'][:19].replace(':', '').replace('-', '')}-{result['mode']}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return path


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--voice", default=None, help="caller voice for the round trip (default: SIM_CALLER_VOICE or am_michael)")
    p.add_argument("--audio-dir", type=Path, help="real recordings: name.wav + name.txt (first line: what was said; optional 'key: a, b')")
    p.add_argument("--language", default="en")
    args = p.parse_args()
    import os
    s = Settings()
    voice = args.voice or os.getenv("SIM_CALLER_VOICE", "am_michael")
    with httpx.Client(timeout=120) as client:
        if args.audio_dir:
            mode, items = "recordings", run_recordings(client, s, args.audio_dir, args.language)
        else:
            mode, items = "round_trip", run_round_trip(client, s, voice, args.language)
    result = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "mode": mode,
              "language": args.language, "models": {"stt": s.stt_model, "tts": s.tts_model if mode == "round_trip" else None,
                                                     "voice": voice if mode == "round_trip" else None},
              "summary": summarize(items), "items": items}
    for i in items:
        mark = "OK  " if i["errors"] == 0 else "MISS"
        print(f"{mark} {i['id']:<20} WER {i['wer']:.0%}  heard: {i['transcript']!r}"
              + (f"  missed: {', '.join(i['missed_key_terms'])}" if i["missed_key_terms"] else ""))
    sm = result["summary"]
    print(f"\naccuracy {sm['accuracy']:.1%} (WER {sm['wer']:.1%}) · key terms {sm['key_term_accuracy']:.1%} "
          f"· {sm['verdict']}")
    print(f"saved: {save(result)}")


if __name__ == "__main__":
    main()
