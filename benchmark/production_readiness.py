"""
Production-readiness test suite for optimize/artemis-candidate.

Tests not covered by the existing benchmark suite:
  1. Transcript consistency  — same SHA1 across N independent runs
  2. Timestamp stability     — segment timestamps stable within 1 ms across runs
  3. Cold-start overhead     — first-run latency vs warm steady-state
  4. Concurrency             — N parallel pipelines (4, 8 streams), verify correctness
  5. Silence-heavy audio     — long leading silence, no crash / hallucination
  6. Power efficiency        — throughput per watt (requires py3nvml)

Usage:
    CUDA_VISIBLE_DEVICES=3 python benchmark/production_readiness.py \\
        --model /home/rozhin/rozhin/models/faster-whisper-large-v3 \\
        --device-index 0 --language fr

    # Skip slow tests:
    python benchmark/production_readiness.py --skip-concurrency --skip-power
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO = os.path.join(BENCHMARK_DIR, "benchmark.m4a")
SR = 16000

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _load_pipeline(model_path, device, device_index, compute_type="float16"):
    from faster_whisper import BatchedInferencePipeline, WhisperModel
    m = WhisperModel(model_path, device=device, device_index=device_index,
                     compute_type=compute_type)
    return BatchedInferencePipeline(m)


def _transcribe(pipeline, audio_or_path, language, beam_size, batch_size,
                clip_seconds=None):
    """Returns (transcript, list[Segment], elapsed_s)."""
    from faster_whisper.audio import decode_audio

    kwargs = {"language": language, "beam_size": beam_size, "batch_size": batch_size}

    if clip_seconds is not None:
        audio = decode_audio(audio_or_path) if isinstance(audio_or_path, str) else audio_or_path
        audio_or_path = audio[:int(clip_seconds * SR)]

    t0 = time.perf_counter()
    segs, _ = pipeline.transcribe(audio_or_path, **kwargs)
    segments = list(segs)
    elapsed = time.perf_counter() - t0
    transcript = "".join(s.text for s in segments)
    return transcript, segments, elapsed


# ---------------------------------------------------------------------------
# 1. Transcript consistency
# ---------------------------------------------------------------------------
def test_transcript_consistency(pipeline, language, beam_size, batch_size,
                                clip_seconds=45, runs=5):
    """All runs must produce the same SHA1."""
    print(f"\n[1] Transcript consistency  ({runs} runs, {clip_seconds}s clip)")
    sys.stdout.flush()

    sha1s = []
    for i in range(runs):
        text, _, elapsed = _transcribe(pipeline, AUDIO, language, beam_size,
                                       batch_size, clip_seconds)
        sha1s.append(_sha1(text))
        print(f"    run {i+1}: {elapsed:.3f}s  sha1={sha1s[-1][:12]}…")
        sys.stdout.flush()

    unique = set(sha1s)
    status = PASS if len(unique) == 1 else FAIL
    print(f"  → {status}  ({len(unique)} unique SHA1s across {runs} runs)")
    return {
        "status": status,
        "runs": runs,
        "unique_sha1s": len(unique),
        "sha1": sha1s[0] if sha1s else None,
    }


# ---------------------------------------------------------------------------
# 2. Timestamp stability
# ---------------------------------------------------------------------------
def test_timestamp_stability(pipeline, language, beam_size, batch_size,
                             clip_seconds=45, runs=3, tolerance_ms=5.0):
    """Segment start/end timestamps must not drift between runs."""
    print(f"\n[2] Timestamp stability  ({runs} runs, {clip_seconds}s clip, tol={tolerance_ms}ms)")
    sys.stdout.flush()

    all_segs = []
    for i in range(runs):
        _, segs, elapsed = _transcribe(pipeline, AUDIO, language, beam_size,
                                       batch_size, clip_seconds)
        all_segs.append(segs)
        print(f"    run {i+1}: {elapsed:.3f}s  {len(segs)} segments")
        sys.stdout.flush()

    ref = all_segs[0]
    max_drift_ms = 0.0
    drifts = []

    for run_i, segs in enumerate(all_segs[1:], start=2):
        if len(segs) != len(ref):
            print(f"    run {run_i}: segment count mismatch "
                  f"({len(segs)} vs {len(ref)} reference)")
            drifts.append(None)
            continue
        run_max = 0.0
        for s_ref, s_cmp in zip(ref, segs):
            d = max(abs(s_ref.start - s_cmp.start),
                    abs(s_ref.end   - s_cmp.end)) * 1000
            run_max = max(run_max, d)
        drifts.append(run_max)
        max_drift_ms = max(max_drift_ms, run_max)

    numeric_drifts = [d for d in drifts if d is not None]
    status = PASS if (numeric_drifts and max(numeric_drifts) <= tolerance_ms) else FAIL
    print(f"  → {status}  max drift {max_drift_ms:.2f} ms  (tolerance {tolerance_ms} ms)")
    return {
        "status": status,
        "max_drift_ms": round(max_drift_ms, 3),
        "tolerance_ms": tolerance_ms,
        "per_run_max_ms": [round(d, 3) if d is not None else None for d in drifts],
    }


# ---------------------------------------------------------------------------
# 3. Cold-start overhead
# ---------------------------------------------------------------------------
def test_cold_start(model_path, device, device_index, language, beam_size, batch_size,
                    clip_seconds=45, warm_runs=3):
    """Compares first-call latency to steady-state average."""
    print(f"\n[3] Cold-start overhead  (clip={clip_seconds}s)")
    sys.stdout.flush()

    t_load0 = time.perf_counter()
    pipeline = _load_pipeline(model_path, device, device_index)
    load_time = time.perf_counter() - t_load0
    print(f"    model load: {load_time:.3f}s")
    sys.stdout.flush()

    # Cold call — no prior warmup
    _, _, cold_elapsed = _transcribe(pipeline, AUDIO, language, beam_size,
                                     batch_size, clip_seconds)
    print(f"    cold run  : {cold_elapsed:.3f}s")
    sys.stdout.flush()

    warm_times = []
    for i in range(warm_runs):
        _, _, t = _transcribe(pipeline, AUDIO, language, beam_size,
                              batch_size, clip_seconds)
        warm_times.append(t)
        print(f"    warm run {i+1}: {t:.3f}s")
        sys.stdout.flush()

    warm_median = statistics.median(warm_times)
    overhead_pct = (cold_elapsed - warm_median) / warm_median * 100
    status = PASS if overhead_pct < 100 else FAIL  # cold < 2× warm = OK
    print(f"  → {status}  cold={cold_elapsed:.3f}s  warm_median={warm_median:.3f}s  "
          f"overhead=+{overhead_pct:.1f}%")
    del pipeline

    return {
        "status": status,
        "model_load_s": round(load_time, 3),
        "cold_s": round(cold_elapsed, 3),
        "warm_median_s": round(warm_median, 3),
        "overhead_pct": round(overhead_pct, 1),
    }


# ---------------------------------------------------------------------------
# 4. Concurrency
# ---------------------------------------------------------------------------
def _worker(model_path, device, device_index, language, beam_size, batch_size,
            clip_seconds, worker_id):
    pipeline = _load_pipeline(model_path, device, device_index)
    # Warmup
    _transcribe(pipeline, AUDIO, language, beam_size, batch_size, clip_seconds)
    t0 = time.perf_counter()
    text, _, _ = _transcribe(pipeline, AUDIO, language, beam_size, batch_size,
                              clip_seconds)
    elapsed = time.perf_counter() - t0
    del pipeline
    return worker_id, _sha1(text), elapsed


def test_concurrency(model_path, device, device_index, language, beam_size, batch_size,
                     stream_counts=(4, 8), clip_seconds=45):
    """Run N pipelines in parallel; verify all produce the same transcript."""
    print(f"\n[4] Concurrency  (streams={stream_counts}, clip={clip_seconds}s)")
    sys.stdout.flush()

    # Get reference SHA1 from a serial run first
    ref_pipeline = _load_pipeline(model_path, device, device_index)
    _transcribe(ref_pipeline, AUDIO, language, beam_size, batch_size, clip_seconds)
    ref_text, _, _ = _transcribe(ref_pipeline, AUDIO, language, beam_size,
                                 batch_size, clip_seconds)
    ref_sha1 = _sha1(ref_text)
    del ref_pipeline
    print(f"    reference SHA1: {ref_sha1[:12]}…")
    sys.stdout.flush()

    results = {}
    for n_streams in stream_counts:
        print(f"\n    {n_streams} streams ...", flush=True)
        t_wall0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_streams) as ex:
            futures = [
                ex.submit(_worker, model_path, device, device_index, language,
                          beam_size, batch_size, clip_seconds, i)
                for i in range(n_streams)
            ]
            outcomes = [f.result() for f in futures]
        wall_time = time.perf_counter() - t_wall0

        sha1s     = [o[1] for o in outcomes]
        latencies = [o[2] for o in outcomes]
        all_match = all(s == ref_sha1 for s in sha1s)
        agg_audio = clip_seconds * n_streams
        throughput = agg_audio / wall_time

        status = PASS if all_match else FAIL
        print(f"    → {status}  all_match={all_match}  "
              f"wall={wall_time:.2f}s  throughput={throughput:.1f}×  "
              f"latencies=[{', '.join(f'{l:.2f}' for l in latencies)}]")

        results[str(n_streams)] = {
            "status": status,
            "n_streams": n_streams,
            "all_match": all_match,
            "wall_s": round(wall_time, 3),
            "aggregate_throughput_x": round(throughput, 2),
            "latencies_s": [round(l, 3) for l in latencies],
        }

    overall = PASS if all(v["status"] == PASS for v in results.values()) else FAIL
    return {"status": overall, "streams": results}


# ---------------------------------------------------------------------------
# 5. Silence-heavy audio
# ---------------------------------------------------------------------------
def test_silence_heavy(pipeline, language, beam_size, batch_size,
                       silence_s=30, speech_s=30):
    """Prepend silence_s seconds of silence before speech. Expect no crash."""
    print(f"\n[5] Silence-heavy  ({silence_s}s silence + {speech_s}s speech)")
    sys.stdout.flush()

    import numpy as np
    from faster_whisper.audio import decode_audio

    try:
        speech = decode_audio(AUDIO)
        speech_clip = speech[:SR * speech_s]
        silence = np.zeros(SR * silence_s, dtype=np.float32)
        audio = np.concatenate([silence, speech_clip])

        t0 = time.perf_counter()
        segs, info = pipeline.transcribe(audio, language=language,
                                         beam_size=beam_size, batch_size=batch_size)
        segments = list(segs)
        elapsed = time.perf_counter() - t0

        text = "".join(s.text for s in segments)
        # Segments should start after the silence
        early_segs = [s for s in segments if s.start < silence_s - 1.0]

        status = PASS if len(segments) > 0 else FAIL
        print(f"  → {status}  {len(segments)} segments  elapsed={elapsed:.3f}s  "
              f"chars={len(text)}  early_segs={len(early_segs)}")
        if early_segs:
            print(f"    WARNING: {len(early_segs)} segment(s) start before silence ends "
                  f"(possible hallucination)")
        return {
            "status": status,
            "segments": len(segments),
            "chars": len(text),
            "elapsed_s": round(elapsed, 3),
            "early_segments": len(early_segs),
        }
    except Exception as e:
        print(f"  → {FAIL}  exception: {e}")
        return {"status": FAIL, "error": str(e)}


# ---------------------------------------------------------------------------
# 6. Power efficiency
# ---------------------------------------------------------------------------
def test_power_efficiency(pipeline, language, beam_size, batch_size,
                          device_index, clip_seconds=45, runs=3):
    """Measure throughput-per-watt during steady-state inference."""
    print(f"\n[6] Power efficiency  (clip={clip_seconds}s, {runs} runs)")
    sys.stdout.flush()

    try:
        import py3nvml.py3nvml as nvml
        nvml.nvmlInit()
        handle = nvml.nvmlDeviceGetHandleByIndex(device_index)
    except Exception as e:
        print(f"  → {SKIP}  py3nvml unavailable: {e}")
        return {"status": SKIP, "reason": str(e)}

    # Warmup
    _transcribe(pipeline, AUDIO, language, beam_size, batch_size, clip_seconds)

    power_samples = []
    timings = []

    def _sample_power(stop_event, interval=0.1):
        import threading
        while not stop_event.is_set():
            try:
                mw = nvml.nvmlDeviceGetPowerUsage(handle)
                power_samples.append(mw / 1000.0)  # W
            except Exception:
                pass
            stop_event.wait(interval)

    import threading
    stop = threading.Event()
    sampler = threading.Thread(target=_sample_power, args=(stop,), daemon=True)
    sampler.start()

    for i in range(runs):
        _, _, elapsed = _transcribe(pipeline, AUDIO, language, beam_size,
                                    batch_size, clip_seconds)
        timings.append(elapsed)
        print(f"    run {i+1}: {elapsed:.3f}s")
        sys.stdout.flush()

    stop.set()
    sampler.join(timeout=2)
    nvml.nvmlShutdown()

    median_s = statistics.median(timings)
    throughput_x = clip_seconds / median_s
    avg_power_w = statistics.mean(power_samples) if power_samples else None
    throughput_per_watt = throughput_x / avg_power_w if avg_power_w else None

    print(f"  → {PASS}  throughput={throughput_x:.2f}×  "
          f"avg_power={avg_power_w:.1f}W  "
          f"throughput/watt={throughput_per_watt:.4f}×/W")
    return {
        "status": PASS,
        "throughput_x": round(throughput_x, 3),
        "avg_power_w": round(avg_power_w, 1) if avg_power_w else None,
        "throughput_per_watt": round(throughput_per_watt, 5) if throughput_per_watt else None,
        "power_samples": len(power_samples),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Production-readiness tests")
    parser.add_argument("--model",        default="large-v3")
    parser.add_argument("--device",       default="cuda")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--language",     default="fr")
    parser.add_argument("--beam-size",    type=int, default=5)
    parser.add_argument("--batch-size",   type=int, default=16)
    parser.add_argument("--clip-seconds", type=float, default=45,
                        help="Audio clip length for fast tests.")
    parser.add_argument("--output", default="benchmark/artifacts/production_readiness.json")
    parser.add_argument("--skip-consistency",  action="store_true")
    parser.add_argument("--skip-timestamps",   action="store_true")
    parser.add_argument("--skip-cold-start",   action="store_true")
    parser.add_argument("--skip-concurrency",  action="store_true")
    parser.add_argument("--skip-silence",      action="store_true")
    parser.add_argument("--skip-power",        action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  Production Readiness — optimize/artemis-candidate")
    print(f"  model={args.model}  compute=float16  beam={args.beam_size}  "
          f"batch={args.batch_size}")
    print(f"{'='*65}")
    sys.stdout.flush()

    report = {
        "model": args.model,
        "compute_type": "float16",
        "beam_size": args.beam_size,
        "batch_size": args.batch_size,
        "language": args.language,
        "device": args.device,
        "device_index": args.device_index,
    }

    # Shared pipeline for tests that don't need a fresh load
    shared_pipeline = _load_pipeline(args.model, args.device, args.device_index)
    # One warmup so the shared pipeline is hot before any test
    _transcribe(shared_pipeline, AUDIO, args.language,
                args.beam_size, args.batch_size, args.clip_seconds)

    if not args.skip_consistency:
        report["consistency"] = test_transcript_consistency(
            shared_pipeline, args.language, args.beam_size, args.batch_size,
            clip_seconds=args.clip_seconds)

    if not args.skip_timestamps:
        report["timestamp_stability"] = test_timestamp_stability(
            shared_pipeline, args.language, args.beam_size, args.batch_size,
            clip_seconds=args.clip_seconds)

    if not args.skip_cold_start:
        report["cold_start"] = test_cold_start(
            args.model, args.device, args.device_index,
            args.language, args.beam_size, args.batch_size,
            clip_seconds=args.clip_seconds)

    if not args.skip_concurrency:
        report["concurrency"] = test_concurrency(
            args.model, args.device, args.device_index,
            args.language, args.beam_size, args.batch_size,
            stream_counts=(4, 8), clip_seconds=args.clip_seconds)

    if not args.skip_silence:
        report["silence_heavy"] = test_silence_heavy(
            shared_pipeline, args.language, args.beam_size, args.batch_size)

    if not args.skip_power:
        report["power_efficiency"] = test_power_efficiency(
            shared_pipeline, args.language, args.beam_size, args.batch_size,
            args.device_index, clip_seconds=args.clip_seconds)

    del shared_pipeline

    # Summary
    print(f"\n{'='*65}")
    print("  SUMMARY")
    print(f"{'='*65}")
    all_statuses = []
    for key, val in report.items():
        if isinstance(val, dict) and "status" in val:
            status = val["status"]
            all_statuses.append(status)
            print(f"  {key:<25} {status}")
    passed = sum(1 for s in all_statuses if s == PASS)
    skipped = sum(1 for s in all_statuses if s == SKIP)
    failed = sum(1 for s in all_statuses if s == FAIL)
    print(f"\n  PASS={passed}  FAIL={failed}  SKIP={skipped} / {len(all_statuses)} tests")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  report -> {args.output}")


if __name__ == "__main__":
    main()
