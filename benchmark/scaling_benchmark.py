"""
Two benchmarks in one script:

  PART 1 — Audio length scaling
    Measures transcription time vs audio duration.
    Shows how throughput changes with clip length.
    Tests: 30s, 5min, 13min (full), 60min (tiled).

  PART 2 — FFT thread scaling
    Measures feature extraction time as scipy.fft workers increases.
    Shows actual CPU parallelism gain on the Xeon Gold 6230.
    Tests: workers = 1, 2, 4, 8, 16, -1 (all cores).

Usage:
    # Generate variants first:
    python benchmark/generate_audio_variants.py

    python benchmark/scaling_benchmark.py \\
        --model /path/to/faster-whisper-large-v3 \\
        --device-index 1
"""

import argparse
import json
import os
import statistics
import time

import numpy as np

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
VARIANTS_DIR  = os.path.join(BENCHMARK_DIR, "audio_variants")
SR = 16000


# ---------------------------------------------------------------------------
# Part 1 — Audio length scaling
# ---------------------------------------------------------------------------

def run_length_scaling(model_path, device_index, language):
    from faster_whisper import WhisperModel, BatchedInferencePipeline
    from faster_whisper.audio import decode_audio

    clips = [
        ("30s",   os.path.join(VARIANTS_DIR, "clean_30s.wav"),   30),
        ("5min",  os.path.join(VARIANTS_DIR, "clean_5min.wav"),  300),
        ("13min", os.path.join(VARIANTS_DIR, "clean_full.wav"),  None),
        ("60min", os.path.join(VARIANTS_DIR, "extended_60min.wav"), None),
    ]

    missing = [c[1] for c in clips if not os.path.exists(c[1])]
    if missing:
        print("  Missing variants — run generate_audio_variants.py first")
        return []

    configs = [
        ("baseline",   "float16",      16, 5),
        ("optimised",  "int8_float16", 32, 1),
    ]

    results = []

    for cfg_label, compute_type, batch_size, beam_size in configs:
        print(f"\n  [{cfg_label}] loading model ...")
        model    = WhisperModel(model_path, device="cuda",
                                device_index=device_index, compute_type=compute_type)
        pipeline = BatchedInferencePipeline(model)

        # warmup
        audio_warm = decode_audio(os.path.join(VARIANTS_DIR, "clean_30s.wav"))
        _, _ = pipeline.transcribe(audio_warm, language=language, batch_size=batch_size, beam_size=beam_size)
        list(_)  # consume

        for label, path, nominal_s in clips:
            audio    = decode_audio(path)
            duration = len(audio) / SR

            timings = []
            for _ in range(3):
                t0   = time.perf_counter()
                segs, _ = pipeline.transcribe(audio, language=language,
                                              batch_size=batch_size, beam_size=beam_size)
                list(segs)
                timings.append(time.perf_counter() - t0)

            med      = statistics.median(timings)
            xrt      = duration / med
            print(f"    {label:>5}  {duration:7.1f}s audio → {med:.2f}s  ({xrt:.1f}× RT)")
            results.append({
                "config": cfg_label,
                "clip": label,
                "audio_duration_s": round(duration, 1),
                "median_s": round(med, 3),
                "throughput_x": round(xrt, 1),
                "stddev_ms": round(statistics.pstdev(timings) * 1000, 1),
            })

        del model, pipeline

    return results


def print_scaling_table(results):
    configs = list(dict.fromkeys(r["config"] for r in results))
    clips   = list(dict.fromkeys(r["clip"]   for r in results))

    lookup = {(r["config"], r["clip"]): r for r in results}

    print(f"\n{'Audio':>8}", end="")
    for cfg in configs:
        print(f"  {cfg:>12} (s)  {cfg:>12} (×RT)", end="")
    print()
    print("-" * (8 + len(configs) * 32))

    for clip in clips:
        dur = lookup.get((configs[0], clip), {}).get("audio_duration_s", "?")
        print(f"{clip:>8}", end="")
        for cfg in configs:
            r = lookup.get((cfg, clip))
            if r:
                print(f"  {r['median_s']:>16.3f}  {r['throughput_x']:>16.1f}", end="")
            else:
                print(f"  {'—':>16}  {'—':>16}", end="")
        print()


# ---------------------------------------------------------------------------
# Part 2 — FFT thread scaling
# ---------------------------------------------------------------------------

def run_thread_scaling():
    try:
        import scipy.fft as sfft
    except ImportError:
        print("  scipy not available — skipping thread scaling")
        return []

    from faster_whisper.audio import decode_audio
    from faster_whisper.feature_extractor import FeatureExtractor

    audio = decode_audio(os.path.join(VARIANTS_DIR, "clean_30s.wav"))
    if not os.path.exists(os.path.join(VARIANTS_DIR, "clean_30s.wav")):
        # fallback: use benchmark audio clip
        from benchmark_dir import BENCHMARK_DIR
        from faster_whisper.audio import decode_audio as da
        import os as _os
        bm = _os.path.join(BENCHMARK_DIR, "benchmark.m4a")
        audio = da(bm)[:SR * 30]

    fe = FeatureExtractor()
    # Pre-extract frames the way stft() would see them (batch=1, n_frames, n_fft)
    n_fft      = fe.n_fft
    hop_length = fe.hop_length
    chunk      = audio[:SR * 30].astype(np.float64)

    # Build frames manually (same as stft internals)
    pad = n_fft // 2
    padded = np.pad(chunk, pad)
    n_frames = 1 + (len(padded) - n_fft) // hop_length
    import numpy.lib.stride_tricks as stt
    frames = stt.as_strided(
        padded,
        (n_frames, n_fft),
        (hop_length * padded.strides[0], padded.strides[0]),
    ).copy()

    worker_counts = [1, 2, 4, 8, 16, -1]
    results = []

    print(f"\n  Benchmarking scipy.fft.rfft on {n_frames} × {n_fft}-pt frames ...")
    single_time = None

    for w in worker_counts:
        timings = []
        for _ in range(5):
            t0 = time.perf_counter()
            sfft.rfft(frames, n=n_fft, axis=-1, workers=w)
            timings.append(time.perf_counter() - t0)
        med = statistics.median(timings)
        if single_time is None:
            single_time = med
        speedup = single_time / med
        label = f"workers={w}" if w != -1 else "workers=-1 (all)"
        print(f"    {label:<22}  {med*1000:6.1f} ms  ({speedup:.2f}× vs single)")
        results.append({
            "workers": w,
            "median_ms": round(med * 1000, 2),
            "speedup_vs_1": round(speedup, 2),
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device-index", type=int, default=1)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--output",
                        default="benchmark/artifacts/scaling_report.json")
    parser.add_argument("--skip-scaling",      action="store_true")
    parser.add_argument("--skip-thread-scaling", action="store_true")
    args = parser.parse_args()

    report = {}

    if not args.skip_scaling:
        print("\n" + "=" * 70)
        print("PART 1 — Audio length scaling")
        print("=" * 70)
        scaling = run_length_scaling(args.model, args.device_index, args.language)
        report["length_scaling"] = scaling
        if scaling:
            print_scaling_table(scaling)

    if not args.skip_thread_scaling:
        print("\n" + "=" * 70)
        print("PART 2 — FFT thread scaling (scipy.fft workers)")
        print("=" * 70)
        thread = run_thread_scaling()
        report["thread_scaling"] = thread

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n  report -> {args.output}")


if __name__ == "__main__":
    main()
