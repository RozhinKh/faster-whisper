"""
Direct model latency benchmark for Artemis ASR scenarios.

Runs baseline and candidate configs on scenario audio files using
BatchedInferencePipeline directly — no HTTP, no server overhead.
Gives a clean parameter-only comparison.

Usage:
    CUDA_VISIBLE_DEVICES=3 python benchmark/scenario_latency_benchmark.py \
        --model large-v3 \
        --datasets ~/rozhin/fw-optimised/artemis-bench/datasets
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


SCENARIOS = [
    {
        "id": "control_phrase_v1",
        "audio": "librispeech/test-clean/1089/134686/1089-134686-0000.flac",
        "duration_s": 4.21,
        "language": "en",
        "warmup": 5,
        "runs": 20,
    },
    {
        "id": "clean_short_v1",
        "audio": "librispeech/test-clean/1089/134686/1089-134686-0001.flac",
        "duration_s": 8.37,
        "language": "en",
        "warmup": 5,
        "runs": 20,
    },
    {
        "id": "noisy_v1",
        "audio": "librispeech/test-other/7902/96591/7902-96591-0014.flac",
        "duration_s": 9.710,
        "language": "en",
        "warmup": 5,
        "runs": 20,
    },
    {
        "id": "clean_long_v1",
        "audio": "librispeech/test-clean/1089/134686/1089-134686-0007.flac",
        "duration_s": 23.45,
        "language": "en",
        "warmup": 5,
        "runs": 20,
    },
    {
        "id": "long_form_v1",
        "audio": "librispeech/test-clean/1089/134686/1089-134686-chapter.flac",
        "duration_s": 156.470,
        "language": "en",
        "warmup": 1,
        "runs": 10,
    },
]

CONFIGS = [
    {
        "name": "baseline",
        "compute_type": "float16",
        "beam_size": 5,
        "batch_size": 16,
    },
    {
        "name": "candidate",
        "compute_type": "int8_float16",
        "beam_size": 1,
        "batch_size": 32,
    },
]


def bench_scenario(pipeline, audio_path: str, scenario: dict, beam_size: int, batch_size: int) -> dict:
    warmup = scenario["warmup"]
    runs = scenario["runs"]

    def transcribe():
        segs, _ = pipeline.transcribe(
            audio_path,
            language=scenario["language"],
            beam_size=beam_size,
            batch_size=batch_size,
        )
        # consume the generator
        list(segs)

    print(f"    warmup ({warmup} runs)...")
    sys.stdout.flush()
    for _ in range(warmup):
        transcribe()

    print(f"    timing ({runs} runs)...")
    sys.stdout.flush()
    timings = []
    for i in range(runs):
        t0 = time.perf_counter()
        transcribe()
        timings.append(time.perf_counter() - t0)
        print(f"      {i+1}/{runs}  {timings[-1]:.3f}s", end="\r")
    print()

    duration_s = scenario["duration_s"]
    median_s = statistics.median(timings)

    return {
        "median_s": round(median_s, 4),
        "mean_s": round(statistics.mean(timings), 4),
        "p95_s": round(sorted(timings)[int(len(timings) * 0.95)], 4),
        "stddev_ms": round(statistics.pstdev(timings) * 1000, 2),
        "rtf": round(median_s / duration_s, 5),
        "rt_multiple": round(duration_s / median_s, 1),
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(description="Scenario latency benchmark — direct model, no HTTP")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--datasets", required=True, help="Path to artemis-bench/datasets/")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--output", default="benchmark/artifacts/scenario_latency.json")
    args = parser.parse_args()

    datasets_path = Path(args.datasets).expanduser()

    from faster_whisper import BatchedInferencePipeline, WhisperModel

    results = {}

    for config in CONFIGS:
        cname = config["name"]
        print(f"\n{'='*62}")
        print(f"  {cname.upper()}  compute_type={config['compute_type']}  "
              f"beam_size={config['beam_size']}  batch_size={config['batch_size']}")
        print(f"{'='*62}")
        print("  Loading model...")
        sys.stdout.flush()

        model = WhisperModel(
            args.model,
            device=args.device,
            device_index=args.device_index,
            compute_type=config["compute_type"],
        )
        pipeline = BatchedInferencePipeline(model)
        print("  Model loaded.\n")
        sys.stdout.flush()

        config_results = {}
        for scenario in SCENARIOS:
            audio_path = datasets_path / scenario["audio"]
            if not audio_path.exists():
                print(f"  [{scenario['id']}] SKIP — not found: {audio_path}")
                continue

            print(f"  [{scenario['id']}]  {scenario['duration_s']}s")
            sys.stdout.flush()

            stats = bench_scenario(
                pipeline, str(audio_path), scenario,
                beam_size=config["beam_size"],
                batch_size=config["batch_size"],
            )
            config_results[scenario["id"]] = stats
            print(f"    → median {stats['median_s']:.3f}s  "
                  f"RTF {stats['rtf']:.5f}  ({stats['rt_multiple']}× RT)  "
                  f"p95 {stats['p95_s']:.3f}s")
            sys.stdout.flush()

        results[cname] = config_results

        del pipeline
        del model

    # -------------------------------------------------------------------------
    # Comparison table
    # -------------------------------------------------------------------------
    print(f"\n\n{'='*72}")
    print("  COMPARISON — direct model latency, same code, parameters only differ")
    print(f"{'='*72}")
    print(f"  {'Scenario':<22}  {'Dur':>7}  {'Base RTF':>10}  {'Cand RTF':>10}  {'Speedup':>8}  {'% faster':>9}")
    print(f"  {'-'*22}  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*9}")

    baseline = results.get("baseline", {})
    candidate = results.get("candidate", {})

    for scenario in SCENARIOS:
        sid = scenario["id"]
        b = baseline.get(sid)
        c = candidate.get(sid)
        if not b or not c:
            continue
        speedup = b["rtf"] / c["rtf"]
        pct = (b["rtf"] - c["rtf"]) / b["rtf"] * 100
        print(f"  {sid:<22}  {scenario['duration_s']:>6.2f}s  "
              f"{b['rtf']:>10.5f}  {c['rtf']:>10.5f}  "
              f"{speedup:>7.2f}×  {pct:>8.1f}%")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"\n  Saved → {output_path}")


if __name__ == "__main__":
    main()
