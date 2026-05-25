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
import os
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
parser.add_argument("--compute-type", default="float16",
                    help="CTranslate2 compute type.")
parser.add_argument("--beam-size",    type=int, default=5,
                    help="Beam size.")
parser.add_argument("--batch-size",   type=int, default=16,
                    help="Batch size (BatchedInferencePipeline).")
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


def _physical_gpu_index(device_index: int) -> int:
    """Resolve CUDA logical device index to physical nvml index.

    CUDA_VISIBLE_DEVICES=3 means logical index 0 maps to physical GPU 3.
    nvmlDeviceGetHandleByIndex always uses physical indices, so we must
    translate before calling it.
    """
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        try:
            physical = [int(x.strip()) for x in cuda_visible.split(",")]
            return physical[device_index]
        except (ValueError, IndexError):
            pass
    return device_index


def _vram_mib(device_index: int):
    try:
        import py3nvml.py3nvml as nvml
        nvml.nvmlInit()
        handle = nvml.nvmlDeviceGetHandleByIndex(_physical_gpu_index(device_index))
        used = nvml.nvmlDeviceGetMemoryInfo(handle).used >> 20
        nvml.nvmlShutdown()
        return used
    except Exception:
        return None


def _vram_delta(device_index: int, fn):
    """Call fn() and return (vram_before, vram_after, delta) in MiB."""
    before = _vram_mib(device_index)
    fn()
    after = _vram_mib(device_index)
    if before is None or after is None:
        return None, None, None
    return before, after, after - before


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

        vram_before = _vram_mib(args.device_index)
        fn = make_inference_fn(
            compute_type=config["compute_type"],
            beam_size=config["beam_size"],
            batch_size=config["batch_size"],
            batched=True,
            device=args.device,
            device_index=args.device_index,
            language=args.language,
        )
        vram_after = _vram_mib(args.device_index)
        vram_used = (vram_after - vram_before) if (vram_before and vram_after) else None

        print("  Warmup...")
        sys.stdout.flush()
        fn()

        print(f"  Timing (repeat={args.repeat}, number={args.number})...")
        sys.stdout.flush()
        runtimes = timeit.repeat(fn, repeat=args.repeat, number=args.number)
        min_per_run = min(runtimes) / args.number

        print(f"  Raw totals : {[round(r, 3) for r in runtimes]}")
        print(f"  Min per run: {min_per_run:.3f}s")
        if vram_used is not None:
            print(f"  VRAM used  : {vram_used} MiB  ({vram_before} → {vram_after} MiB)")

        results[config["name"]] = {
            "label": config["label"],
            "compute_type": config["compute_type"],
            "beam_size": config["beam_size"],
            "batch_size": config["batch_size"],
            "runtimes": [round(r, 4) for r in runtimes],
            "min_per_run_s": round(min_per_run, 4),
            "median_per_run_s": round(statistics.median(runtimes) / args.number, 4),
            "vram_used_mib": vram_used,
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

    if b["vram_used_mib"] and c["vram_used_mib"]:
        vram_pct = (b["vram_used_mib"] - c["vram_used_mib"]) / b["vram_used_mib"] * 100
        print(f"  VRAM:    {b['vram_used_mib']} MiB → {c['vram_used_mib']} MiB  (−{vram_pct:.1f}%)")

    print(f"{'='*60}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
            f.write("\n")
        print(f"\n  Saved → {args.output}")

else:
    print(f"\n{'='*60}")
    print(f"  faster-whisper large-v3  |  compute={args.compute_type}  beam={args.beam_size}  batch={args.batch_size}")
    print(f"  compute={args.compute_type}  beam={args.beam_size}  batch={args.batch_size}")
    print(f"{'='*60}")
    fn = make_inference_fn(
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        batch_size=args.batch_size,
        batched=True,
        device=args.device,
        device_index=args.device_index,
        language=args.language,
    )
    print("  Warmup...")
    fn()
    print(f"  Timing (repeat={args.repeat}, number={args.number})...")
    runtimes = timeit.repeat(fn, repeat=args.repeat, number=args.number)
    min_per_run = min(runtimes) / args.number
    print(f"  Raw totals : {[round(r, 3) for r in runtimes]}")
    print(f"  Min per run: {min_per_run:.3f}s")
