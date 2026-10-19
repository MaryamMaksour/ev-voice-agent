"""Finish post-call work that never completed (the agent crashed or was restarted mid-call).

    python -m agent.postcall                 # raw call files older than 5 minutes
    python -m agent.postcall --min-age 0     # all of them (the agent container does this at start)

Each data/calls/<call_id>.raw.json is checked with the LLM, the CRM is filled in and the final
<call_id>.json written, exactly as if the call had just ended. Calls that are already done
have no raw file, so running this twice is safe.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from dotenv import load_dotenv
from openai import AsyncOpenAI

from crm import make_crm

from .config import Settings
from .events import PostCallProcessor, process_pending
from .logging_setup import setup_logging

logger = logging.getLogger("voice-agent.postcall")


async def run(min_age_seconds: float) -> list[str]:
    settings = Settings()
    llm = AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    processor = PostCallProcessor(make_crm(), llm, settings.llm_model, settings.calls_dir,
                                  logs_dir=settings.logs_dir, recordings_dir=settings.recordings_dir)
    try:
        return await process_pending(processor, min_age_seconds=min_age_seconds)
    finally:
        await llm.close()


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--min-age", type=float, default=300,
                   help="only raw files older than this many seconds (default 300)")
    args = p.parse_args()
    setup_logging("postcall", console=True)
    done = asyncio.run(run(args.min_age))
    print(f"post-call work finished for {len(done)} call(s)" + (f": {', '.join(done)}" if done else ""))


if __name__ == "__main__":
    main()
