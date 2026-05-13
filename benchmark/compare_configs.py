"""
Baseline vs candidate config comparison using the official faster-whisper
speed_benchmark methodology (timeit.repeat, number=10).

Runs on benchmark.m4a — the same audio used by the faster-whisper maintainers
in their README benchmarks.

Usage:
    CUDA_VISIBLE_DEVICES=3 python benchmark/compare_configs.py
    CUDA_VISIBLE_DEVICES=3 python benchmark/compare_configs.py --repeat 5
"""

import argparse
import json
import statistics
import sys
import timeit
from pathlib import Path

BENCHMARK_DIR = Path(__file__).parent
BENCHMARK_AUDIO = str(BENCHMARK_DIR / "benchmark.m4a")

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


def make_fn(compute_type, beam_size, batch_size, device, device_index, language):
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    print(f"  Loading model ({compute_type})...")
    sys.stdout.flush()
    model = WhisperModel(
        "large-v3",
        device=device,
        device_index=device_index,
        compute_type=compute_type,
    )
    pipeline = BatchedInferencePipeline(model)
    print("  Model loaded.")
    sys.stdout.flush()

    def _run():
        segs, _ = pipeline.transcribe(
            BENCHMARK_AUDIO,
            language=language,
            beam_size=beam_size,
            batch_size=batch_size,
        )
        for _ in segs:
            pass

    return _run, model


def main():
    parser = argparse.ArgumentParser(
        description="Baseline vs candidate speed comparison — official benchmark methodology"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Number of repetitions (each runs --number transcriptions). Default: 3",
    )
    parser.add_argument(
        "--number",
        type=int,
        default=10,
        help="Transcriptions per repetition. Default: 10 (matches speed_benchmark.py)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    if not Path(BENCHMARK_AUDIO).exists():
        print(f"ERROR: benchmark audio not found: {BENCHMARK_AUDIO}")
        print("Place benchmark.m4a in the benchmark/ directory.")
        sys.exit(1)

    print("faster-whisper — baseline vs candidate comparison")
    print(f"Audio: {BENCHMARK_AUDIO}")
    print(f"Methodology: timeit.repeat(repeat={args.repeat}, number={args.number})")
    print(f"  → each reported time is min(repeat) / number  (same as speed_benchmark.py)")
    print()

    results = {}

    for config in CONFIGS:
        print(f"{'='*60}")
        print(f"  {config['label']}")
        print(f"{'='*60}")
        sys.stdout.flush()

        fn, model = make_fn(
            config["compute_type"],
            config["beam_size"],
            config["batch_size"],
            args.device,
            args.device_index,
            args.language,
        )

        # Warmup — not counted
        print("  Warmup run...")
        sys.stdout.flush()
        fn()

        print(f"  Timing ({args.repeat} × {args.number} runs)...")
        sys.stdout.flush()
        runtimes = timeit.repeat(fn, repeat=args.repeat, number=args.number)
        min_per_run = min(runtimes) / args.number

        print(f"  Raw totals: {[round(r, 3) for r in runtimes]}")
        print(f"  Min execution time: {min_per_run:.3f}s  "
              f"(median: {statistics.median(runtimes) / args.number:.3f}s)")
        sys.stdout.flush()

        results[config["name"]] = {
            "label": config["label"],
            "compute_type": config["compute_type"],
            "beam_size": config["beam_size"],
            "batch_size": config["batch_size"],
            "runtimes": [round(r, 4) for r in runtimes],
            "min_per_run_s": round(min_per_run, 4),
            "median_per_run_s": round(statistics.median(runtimes) / args.number, 4),
        }

        del fn, model
        print()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    b = results["baseline"]["min_per_run_s"]
    c = results["candidate"]["min_per_run_s"]
    speedup = b / c
    pct = (b - c) / b * 100

    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  {results['baseline']['label']}: {b:.3f}s")
    print(f"  {results['candidate']['label']}: {c:.3f}s")
    print()
    print(f"  Speedup:  {speedup:.2f}×")
    print(f"  Faster:   −{pct:.1f}%")
    print("=" * 60)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
            f.write("\n")
        print(f"\n  Saved → {args.output}")


if __name__ == "__main__":
    main()
