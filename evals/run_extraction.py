"""Demo test #3: CRM-update accuracy of the real LLM on the sample transcripts.

    python -m evals.run_extraction                     # uses LLM_BASE_URL / LLM_MODEL from .env
    python -m evals.run_extraction --model qwen3.5-9b  # compare models (vLLM served-model-name)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from agent.config import Settings
from agent.extraction import extract_outcome
from crm.port import Lead

from .score import score_case

CASES = Path(__file__).resolve().parent.parent / "tests" / "transcripts" / "cases.json"


async def run(model: str, base_url: str, api_key: str, only: str | None) -> float:
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    cases = json.loads(CASES.read_text())
    if only:
        cases = [c for c in cases if only in c["id"]]

    total = correct = 0
    for case in cases:
        t0 = time.perf_counter()
        try:
            got = await extract_outcome(client, model, Lead(**case["lead"]), case["transcript"])
            rows = score_case(case["expected"], got)
        except Exception as e:
            print(f"✘ {case['id']}: extraction failed: {e}")
            total += 1
            continue
        ok = sum(r["ok"] for r in rows)
        total += len(rows)
        correct += ok
        mark = "✔" if ok == len(rows) else "✘"
        print(f"{mark} {case['id']:<28} {ok}/{len(rows)} fields  ({time.perf_counter() - t0:.1f}s)")
        for r in rows:
            if not r["ok"]:
                print(f"     {r['field']}: expected={r['expected']!r} got={r['got']!r}")

    accuracy = correct / total if total else 0.0
    print(f"\nModel {model}: field accuracy {correct}/{total} = {accuracy:.1%} (target >= 95%)")
    return accuracy


def main() -> None:
    load_dotenv()
    s = Settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=s.llm_model)
    parser.add_argument("--base-url", default=s.llm_base_url)
    parser.add_argument("--only", help="run only cases whose id contains this text")
    args = parser.parse_args()
    asyncio.run(run(args.model, args.base_url, s.llm_api_key, args.only))


if __name__ == "__main__":
    main()
