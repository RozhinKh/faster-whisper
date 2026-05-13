"""
Speed benchmark for faster-whisper.

Default mode mirrors the original benchmark (single config, timeit.repeat).
Pass --compare to run baseline vs candidate back-to-back and print a summary.

Usage:
    # Original usage (single config):
    python benchmark/speed_benchmark.py

    # Baseline vs candidate comparison (speed + VRAM):
    CUDA_VISIBLE_DEVICES=3 python benchmark/speed_benchmark.py --compare \
        --device-index 0 --language fr
"""

import argparse
import json
import statistics
import sys
import timeit
from pathlib import Path
from typing import Callable

from utils import inference, make_inference_fn

BENCHMARK_DIR = Path(__file__).parent

parser = argparse.ArgumentParser(description="Speed benchmark")
parser.add_argument("--repeat",       type=int, default=3,
                    help="Times an experiment will be run.")
parser.add_argument("--number",       type=int, default=10,
                    help="Transcriptions per repetition.")
parser.add_argument("--compare",      action="store_true",
                    help="Run baseline vs candidate comparison (speed + VRAM).")
parser.add_argument("--device",       default="cuda")
parser.add_argument("--device-index", type=int, default=0)
parser.add_argument("--language",     default="fr")
parser.add_argument("--output",       default=None,
                    help="Save comparison results to JSON.")
args = parser.parse_args()


def measure_speed(func: Callable[[], None]):
    # as written in https://docs.python.org/3/library/timeit.html#timeit.Timer.repeat,
    # min should be taken rather than the average
    runtimes = timeit.repeat(func, repeat=args.repeat, number=args.number)
    print(runtimes)
    print("Min execution time: %.3fs" % (min(runtimes) / args.number))
    return runtimes


def _vram_mib(device_index: int):
    try:
        import py3nvml.py3nvml as nvml
        nvml.nvmlInit()
        handle = nvml.nvmlDeviceGetHandleByIndex(device_index)
        used = nvml.nvmlDeviceGetMemoryInfo(handle).used >> 20
        nvml.nvmlShutdown()
        return used
    except Exception:
        return None


if args.compare:
    CONFIGS = [
        {
            "name": "baseline",
            "label": "Baseline  (float16,      beam=5, batch=16)",
            "compute_type": "float16",
            "beam_size": 5,
            "batch_size": 16,
        },
        {
            "name": "candidate",
            "label": "Candidate (int8_float16, beam=1, batch=32)",
            "compute_type": "int8_float16",
            "beam_size": 1,
            "batch_size": 32,
        },
    ]

    results = {}
    for config in CONFIGS:
        print(f"\n{'='*60}")
        print(f"  {config['label']}")
        print(f"{'='*60}")
        sys.stdout.flush()

        fn = make_inference_fn(
            compute_type=config["compute_type"],
            beam_size=config["beam_size"],
            batch_size=config["batch_size"],
            batched=True,
            device=args.device,
            device_index=args.device_index,
            language=args.language,
        )

        vram = _vram_mib(args.device_index)

        print("  Warmup...")
        sys.stdout.flush()
        fn()

        print(f"  Timing (repeat={args.repeat}, number={args.number})...")
        sys.stdout.flush()
        runtimes = timeit.repeat(fn, repeat=args.repeat, number=args.number)
        min_per_run = min(runtimes) / args.number

        print(f"  Raw totals : {[round(r, 3) for r in runtimes]}")
        print(f"  Min per run: {min_per_run:.3f}s")
        if vram is not None:
            print(f"  VRAM used  : {vram} MiB")

        results[config["name"]] = {
            "label": config["label"],
            "compute_type": config["compute_type"],
            "beam_size": config["beam_size"],
            "batch_size": config["batch_size"],
            "runtimes": [round(r, 4) for r in runtimes],
            "min_per_run_s": round(min_per_run, 4),
            "median_per_run_s": round(statistics.median(runtimes) / args.number, 4),
            "vram_mib": vram,
        }
        del fn

    b = results["baseline"]
    c = results["candidate"]
    speedup = b["min_per_run_s"] / c["min_per_run_s"]
    pct     = (b["min_per_run_s"] - c["min_per_run_s"]) / b["min_per_run_s"] * 100

    print(f"\n{'='*60}")
    print("  RESULTS")
    print(f"{'='*60}")
    print(f"  {b['label']}: {b['min_per_run_s']:.3f}s")
    print(f"  {c['label']}: {c['min_per_run_s']:.3f}s")
    print(f"\n  Speedup: {speedup:.2f}×  (−{pct:.1f}%)")

    if b["vram_mib"] and c["vram_mib"]:
        vram_pct = (b["vram_mib"] - c["vram_mib"]) / b["vram_mib"] * 100
        print(f"  VRAM:    {b['vram_mib']} MiB → {c['vram_mib']} MiB  (−{vram_pct:.1f}%)")

    print(f"{'='*60}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
            f.write("\n")
        print(f"\n  Saved → {args.output}")

else:
    measure_speed(inference)
