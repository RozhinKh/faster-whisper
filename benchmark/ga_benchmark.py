"""
Minimal benchmark for GA optimisation loops.

Loads the model once, runs one warmup, one timed transcription,
and writes a flat JSON that the GA can read directly.

Target runtime: ~2 minutes on RTX 3090 + large-v3 for the full
13-minute benchmark file. Use --clip-seconds for shorter smoke runs.

Usage:
    python benchmark/ga_benchmark.py \
        --model /path/to/faster-whisper-large-v3 \
        --output artemis_results.json
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from typing import Optional

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO = os.path.join(BENCHMARK_DIR, "benchmark.m4a")


def run(
    model_path: str,
    compute_type: str,
    device: str,
    device_index: int,
    language: str = "fr",
    beam_size: int = 1,
    batch_size: int = 16,
    clip_seconds: Optional[float] = None,
    timed_runs: int = 1,
) -> dict:
    import py3nvml.py3nvml as nvml

    from faster_whisper import BatchedInferencePipeline, WhisperModel

    print("  loading model...")
    sys.stdout.flush()
    model = WhisperModel(
        model_path, device=device, device_index=device_index, compute_type=compute_type
    )
    pipeline = BatchedInferencePipeline(model)
    print("  model loaded")
    sys.stdout.flush()

    transcribe_kwargs = {
        "language": language,
        "beam_size": beam_size,
        "batch_size": batch_size,
    }
    if clip_seconds is not None:
        transcribe_kwargs["clip_timestamps"] = f"0,{clip_seconds}"

    def _transcribe():
        segs, info = pipeline.transcribe(AUDIO, **transcribe_kwargs)
        segments = list(segs)
        transcript = "".join(segment.text for segment in segments)
        return {
            "info": info,
            "segments": segments,
            "transcript": transcript,
        }

    # Warmup run is not timed.
    print("  warming up...")
    sys.stdout.flush()
    warmup = _transcribe()

    print("  timing...")
    sys.stdout.flush()
    timings = []
    last_run = warmup
    for _ in range(timed_runs):
        t0 = time.perf_counter()
        last_run = _transcribe()
        timings.append(time.perf_counter() - t0)

    elapsed = statistics.median(timings)
    audio_duration_s = clip_seconds or last_run["info"].duration
    transcript = last_run["transcript"]
    throughput_x = audio_duration_s / elapsed if elapsed > 0 else None
    transcript_sha1 = hashlib.sha1(transcript.encode("utf-8")).hexdigest()

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
        "speed_p95_s": round(max(timings), 3),
        "speed_stddev_ms": round(statistics.pstdev(timings) * 1000, 3),
        "timed_runs": timed_runs,
        "vram_used_mib": vram_used_mib,
        "throughput_x": round(throughput_x, 3) if throughput_x is not None else None,
        "audio_duration_s": round(audio_duration_s, 3),
        "model": model_path,
        "compute_type": compute_type,
        "device_index": device_index,
        "beam_size": beam_size,
        "batch_size": batch_size,
        "language": language,
        "audio_seconds": clip_seconds,
        "num_segments": len(last_run["segments"]),
        "transcript_chars": len(transcript),
        "transcript_sha1": transcript_sha1,
        "benchmark_mode": "smoke" if clip_seconds is not None else "full",
    }

    print(f"  transcription time : {elapsed:.3f}s")
    print(f"  throughput         : {throughput_x:.3f}x")
    if vram_used_mib is not None:
        print(f"  VRAM used          : {vram_used_mib} MiB")
    sys.stdout.flush()

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal GA benchmark")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--compute-type", default="int8_float16")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=None,
        help="Only transcribe the first N seconds for a fast smoke benchmark.",
    )
    parser.add_argument(
        "--timed-runs",
        type=int,
        default=1,
        help="Number of timed transcription runs after warmup. Reported speed is the median.",
    )
    parser.add_argument("--output", default="artemis_results.json")
    args = parser.parse_args()

    result = run(
        args.model,
        args.compute_type,
        args.device,
        args.device_index,
        language=args.language,
        beam_size=args.beam_size,
        batch_size=args.batch_size,
        clip_seconds=args.clip_seconds,
        timed_runs=args.timed_runs,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump([result], f, indent=2)
        f.write("\n")
    print(f"  results -> {args.output}")
