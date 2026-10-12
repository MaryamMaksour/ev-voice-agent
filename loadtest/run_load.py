"""Demo test #6: how many calls at once can one GPU carry?

Ramps concurrent simulated calls (e.g. 1, 5, 10, 20, 30, 40) through the full pipeline
(LiveKit -> STT -> LLM -> TTS -> CRM) and prints p50/p95 reply latency per level.

    python -m loadtest.make_audio                       # once
    python -m loadtest.run_load --levels 1,5,10,20,30,40

Run it from a machine other than the GPU box (laptop, or a small CPU instance in the
same AWS region) so the callers don't steal the agent's CPU.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from agent.metrics import percentile

from .sim_caller import SimCaller

AUDIO_DIR = Path(__file__).parent / "audio"
LEADS = [f"L{i:03d}" for i in range(1, 21)]


class GpuSampler:
    """Samples nvidia-smi once a second (only when run on the GPU box itself)."""

    def __init__(self) -> None:
        self.enabled = shutil.which("nvidia-smi") is not None
        self.util: list[float] = []
        self.mem: list[float] = []
        self._task: asyncio.Task | None = None

    async def _loop(self) -> None:
        while True:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            ).stdout.strip().splitlines()
            if out:
                u, m = out[0].split(",")
                self.util.append(float(u))
                self.mem.append(float(m))
            await asyncio.sleep(1)

    def start(self) -> None:
        self.util, self.mem = [], []
        if self.enabled:
            self._task = asyncio.create_task(self._loop())

    def stop(self) -> dict:
        if self._task:
            self._task.cancel()
        if not self.util:
            return {}
        return {"gpu_util_avg": round(sum(self.util) / len(self.util)), "gpu_util_max": max(self.util),
                "gpu_mem_max_mb": max(self.mem)}


async def run_level(n: int, args, utterances: list[Path], gpu: GpuSampler) -> dict:
    url = os.environ["LIVEKIT_URL"]
    key, secret = os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"]
    stamp = int(time.time())
    callers = [
        SimCaller(url, key, secret, f"lead-{LEADS[i % len(LEADS)]}-sim{n}x{i}-{stamp}", utterances,
                  reply_timeout=args.reply_timeout)
        for i in range(n)
    ]
    gpu.start()
    t0 = time.perf_counter()
    results = await asyncio.gather(*(c.run(start_delay=i * args.stagger) for i, c in enumerate(callers)))
    elapsed = time.perf_counter() - t0
    gpu_stats = gpu.stop()

    lat = [x for r in results for x in r.latencies]
    row = {
        "concurrent_calls": n,
        "turns": len(lat),
        "p50_s": round(percentile(lat, 50), 2),
        "p95_s": round(percentile(lat, 95), 2),
        "max_s": round(max(lat), 2) if lat else None,
        "timeouts": sum(r.timeouts for r in results),
        "failed_calls": sum(1 for r in results if r.error),
        "elapsed_s": round(elapsed, 1),
        "load_avg_1m": round(os.getloadavg()[0], 2),
        **gpu_stats,
    }
    row["pass"] = row["p95_s"] <= args.p95_target and row["failed_calls"] == 0 and row["timeouts"] == 0
    return row


def print_table(rows: list[dict], target: float) -> None:
    print(f"\n| calls at once | turns | p50 (s) | p95 (s) | max (s) | timeouts | failed | GPU avg/max % | pass (p95<={target}s) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        gpu = f"{r.get('gpu_util_avg', '-')}/{r.get('gpu_util_max', '-')}"
        print(f"| {r['concurrent_calls']} | {r['turns']} | {r['p50_s']} | {r['p95_s']} | {r['max_s']} | "
              f"{r['timeouts']} | {r['failed_calls']} | {gpu} | {'yes' if r['pass'] else 'NO'} |")
    passing = [r["concurrent_calls"] for r in rows if r["pass"]]
    if passing:
        best = max(passing)
        at_least = best == max(r["concurrent_calls"] for r in rows)
        qualifier = "at least " if at_least else ""
        print(f"\nCapacity: {qualifier}{best} calls at once per GPU box with p95 <= {target}s.")
        if at_least:
            print("Every level passed - run higher levels to find the real limit.")
        print(f"For a 2,800 calls/day peak of ~40 calls at once -> {'at most ' if at_least else ''}"
              f"{-(-40 // best)} box(es).")
    else:
        print("\nNo level passed; check the agent logs and the slowest stage in data/calls/*.json.")


async def main_async(args) -> None:
    utterances = sorted(Path(args.audio_dir).glob("*.wav"))
    if not utterances:
        raise SystemExit("No caller audio. Run: python -m loadtest.make_audio")
    gpu = GpuSampler()
    rows = []
    for n in [int(x) for x in args.levels.split(",")]:
        print(f"running {n} concurrent call(s)...", flush=True)
        row = await run_level(n, args, utterances, gpu)
        print(f"  -> p50 {row['p50_s']}s  p95 {row['p95_s']}s  timeouts {row['timeouts']}  failed {row['failed_calls']}")
        rows.append(row)
        await asyncio.sleep(args.cooldown)
    print_table(rows, args.p95_target)

    out = Path("data/loadtest")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"results-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps({"model": os.getenv("LLM_MODEL"), "rows": rows}, indent=2))
    print(f"\nsaved {path}")


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--levels", default="1,5,10,20,30,40")
    p.add_argument("--stagger", type=float, default=0.5, help="seconds between call starts")
    p.add_argument("--reply-timeout", type=float, default=20.0)
    p.add_argument("--p95-target", type=float, default=1.5)
    p.add_argument("--cooldown", type=float, default=10.0)
    p.add_argument("--audio-dir", default=str(AUDIO_DIR))
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
