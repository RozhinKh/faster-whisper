"""
Minimal benchmark for GA optimisation loops.

Loads the model once, runs one warmup, one timed transcription,
and writes a flat JSON that the GA can read directly.

Target runtime: ~2 minutes on RTX 3090 + large-v3.

Usage:
    python benchmark/ga_benchmark.py \
        --model /path/to/faster-whisper-large-v3 \
        --output artemis_results.json
"""

import argparse
import json
import os
import sys
import time

import py3nvml.py3nvml as nvml

from faster_whisper import WhisperModel

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO = os.path.join(BENCHMARK_DIR, "benchmark.m4a")


def run(model_path: str, compute_type: str, device: str, device_index: int) -> dict:
    print("  loading model...")
    sys.stdout.flush()
    model = WhisperModel(
        model_path, device=device, device_index=device_index, compute_type=compute_type
    )
    print("  model loaded")
    sys.stdout.flush()

    def _transcribe():
        segs, _ = model.transcribe(AUDIO, language="fr")
        for _ in segs:
            pass

    # warmup — not timed
    print("  warming up...")
    sys.stdout.flush()
    _transcribe()

    # single timed run
    print("  timing...")
    sys.stdout.flush()
    t0 = time.perf_counter()
    _transcribe()
    elapsed = time.perf_counter() - t0

    # gpu memory snapshot
    vram_used_mib = None
    try:
        nvml.nvmlInit()
        handle = nvml.nvmlDeviceGetHandleByIndex(device_index)
        vram_used_mib = nvml.nvmlDeviceGetMemoryInfo(handle).used >> 20
        nvml.nvmlShutdown()
    except Exception:
        pass

    result = {
        "speed_min_s": round(elapsed, 3),
        "vram_used_mib": vram_used_mib,
        "model": model_path,
        "compute_type": compute_type,
        "device_index": device_index,
    }

    print(f"  transcription time : {elapsed:.3f}s")
    if vram_used_mib:
        print(f"  VRAM used          : {vram_used_mib} MiB")
    sys.stdout.flush()

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal GA benchmark")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--output", default="artemis_results.json")
    args = parser.parse_args()

    result = run(args.model, args.compute_type, args.device, args.device_index)

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  results → {args.output}")
