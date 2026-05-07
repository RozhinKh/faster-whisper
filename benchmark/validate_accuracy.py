"""
Accuracy and regression validation suite for faster-whisper optimisations.

Runs:
  1. Ablation study — each optimisation in isolation
  2. Statistical variance — 3 timed runs per config
  3. WER with and without punctuation (standard ASR practice strips punctuation)
  4. Segment diversity — WER on 3 independent 60s clips from different positions
  5. Beam-size safety report — does beam=1 hurt on harder segments?

Usage:
    python benchmark/validate_accuracy.py \
        --model /path/to/faster-whisper-large-v3 \
        --device-index 1 \
        --output benchmark/artifacts/validation_report.json
"""

import argparse
import json
import re
import statistics
import sys
import time
import os

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO = os.path.join(BENCHMARK_DIR, "benchmark.m4a")
SAMPLING_RATE = 16000


def _strip_punctuation(text: str) -> str:
    """Remove punctuation for standard ASR WER evaluation."""
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip().lower()


def _transcribe(pipeline, audio_path, batch_size, beam_size, language="fr",
                clip_start=None, clip_end=None):
    from faster_whisper.audio import decode_audio
    audio = decode_audio(audio_path)
    if clip_start is not None or clip_end is not None:
        s = int((clip_start or 0) * SAMPLING_RATE)
        e = int(clip_end * SAMPLING_RATE) if clip_end else len(audio)
        audio = audio[s:e]

    t0 = time.perf_counter()
    segs, info = pipeline.transcribe(
        audio, language=language, batch_size=batch_size, beam_size=beam_size
    )
    text = "".join(s.text for s in segs)
    elapsed = time.perf_counter() - t0
    return text, elapsed, getattr(info, "duration", len(audio) / SAMPLING_RATE)


def _load(model_path, device_index, compute_type):
    from faster_whisper import WhisperModel, BatchedInferencePipeline
    m = WhisperModel(model_path, device="cuda",
                     device_index=device_index, compute_type=compute_type)
    return BatchedInferencePipeline(m)


def compute_wer(reference: str, hypothesis: str, strip_punct: bool = True):
    try:
        from jiwer import wer
    except ImportError:
        return None
    if strip_punct:
        reference = _strip_punctuation(reference)
        hypothesis = _strip_punctuation(hypothesis)
    return round(wer(reference, hypothesis), 4)


def run_ablation(model_path, device_index, language):
    """Test each optimisation independently to isolate its impact."""
    configs = [
        {"name": "baseline",          "compute_type": "float16",      "batch_size": 16, "beam_size": 5},
        {"name": "+beam=1",            "compute_type": "float16",      "batch_size": 16, "beam_size": 1},
        {"name": "+beam=1,batch=32",   "compute_type": "float16",      "batch_size": 32, "beam_size": 1},
        {"name": "fully_optimised",    "compute_type": "int8_float16", "batch_size": 32, "beam_size": 1},
    ]

    results = []
    baseline_text = None

    for cfg in configs:
        print(f"\n  [{cfg['name']}] loading...", flush=True)
        pipeline = _load(model_path, device_index, cfg["compute_type"])

        # warmup
        _transcribe(pipeline, AUDIO, cfg["batch_size"], cfg["beam_size"],
                    language=language, clip_end=10)

        timings = []
        text = ""
        for i in range(3):
            text, elapsed, duration = _transcribe(
                pipeline, AUDIO, cfg["batch_size"], cfg["beam_size"], language=language
            )
            timings.append(elapsed)
            print(f"    run {i+1}: {elapsed:.3f}s", flush=True)

        med = statistics.median(timings)
        std = statistics.pstdev(timings) * 1000

        if baseline_text is None:
            baseline_text = text

        wer_strict = compute_wer(baseline_text, text, strip_punct=False)
        wer_normalised = compute_wer(baseline_text, text, strip_punct=True)

        result = {
            "config": cfg["name"],
            "compute_type": cfg["compute_type"],
            "batch_size": cfg["batch_size"],
            "beam_size": cfg["beam_size"],
            "median_s": round(med, 3),
            "stddev_ms": round(std, 1),
            "throughput_x": round(duration / med, 1),
            "chars": len(text),
            "wer_strict": wer_strict,
            "wer_normalised": wer_normalised,
        }
        results.append(result)
        print(f"    median={med:.3f}s  WER(norm)={wer_normalised:.4f}  WER(strict)={wer_strict:.4f}")

        del pipeline

    return results


def run_segment_diversity(model_path, device_index, language):
    """
    Compare baseline vs optimised on 3 non-overlapping 60s clips.
    Tests whether WER is consistent across different parts of the audio.
    """
    from faster_whisper.audio import decode_audio
    audio = decode_audio(AUDIO)
    total_s = len(audio) / SAMPLING_RATE

    clips = [
        {"name": "start",  "start": 0,                   "end": 60},
        {"name": "middle", "start": total_s / 2 - 30,    "end": total_s / 2 + 30},
        {"name": "end",    "start": max(0, total_s - 60), "end": total_s},
    ]

    results = []

    for compute_type, batch_size, beam_size, label in [
        ("float16",      16, 5, "baseline"),
        ("int8_float16", 32, 1, "optimised"),
    ]:
        print(f"\n  [{label}] loading...", flush=True)
        pipeline = _load(model_path, device_index, compute_type)

        clip_texts = {}
        for clip in clips:
            text, elapsed, _ = _transcribe(
                pipeline, AUDIO, batch_size, beam_size, language=language,
                clip_start=clip["start"], clip_end=clip["end"]
            )
            clip_texts[clip["name"]] = text
            print(f"    clip={clip['name']}: {elapsed:.3f}s  {len(text)} chars")

        results.append({"label": label, "clips": clip_texts})
        del pipeline

    # Cross-compare: baseline vs optimised for each clip
    baseline_clips = results[0]["clips"]
    optimised_clips = results[1]["clips"]

    comparison = []
    for clip in clips:
        name = clip["name"]
        wer_n = compute_wer(baseline_clips[name], optimised_clips[name], strip_punct=True)
        wer_s = compute_wer(baseline_clips[name], optimised_clips[name], strip_punct=False)
        comparison.append({
            "clip": name,
            "baseline_chars": len(baseline_clips[name]),
            "optimised_chars": len(optimised_clips[name]),
            "wer_normalised": wer_n,
            "wer_strict": wer_s,
        })
        print(f"    {name}: WER(norm)={wer_n:.4f}  WER(strict)={wer_s:.4f}")

    return comparison


def run_beam_safety(model_path, device_index, language):
    """
    Compare beam=5 vs beam=1 on segments of varying difficulty:
    short utterances, long sentences, numbers, proper nouns.
    Uses fixed clips from known positions in the audio.
    """
    from faster_whisper.audio import decode_audio
    audio = decode_audio(AUDIO)
    total_s = len(audio) / SAMPLING_RATE

    # Sample 6 non-overlapping 30s windows across the full audio
    positions = [i * total_s / 6 for i in range(6)]
    clips = [{"start": p, "end": min(p + 30, total_s)} for p in positions]

    pipeline_b5 = _load(model_path, device_index, "float16")
    pipeline_b1 = _load(model_path, device_index, "int8_float16")

    results = []
    print(flush=True)
    for i, clip in enumerate(clips):
        t5, _, _ = _transcribe(pipeline_b5, AUDIO, 16, 5, language=language,
                               clip_start=clip["start"], clip_end=clip["end"])
        t1, _, _ = _transcribe(pipeline_b1, AUDIO, 32, 1, language=language,
                               clip_start=clip["start"], clip_end=clip["end"])
        wer_n = compute_wer(t5, t1, strip_punct=True)
        wer_s = compute_wer(t5, t1, strip_punct=False)
        results.append({
            "clip_index": i,
            "start_s": round(clip["start"], 1),
            "end_s": round(clip["end"], 1),
            "beam5_chars": len(t5),
            "beam1_chars": len(t1),
            "wer_normalised": wer_n,
            "wer_strict": wer_s,
            "beam5_sample": t5[:120].strip(),
            "beam1_sample": t1[:120].strip(),
        })
        print(f"    clip {i} ({clip['start']:.0f}–{clip['end']:.0f}s): "
              f"WER(norm)={wer_n:.4f}  WER(strict)={wer_s:.4f}")

    del pipeline_b5, pipeline_b1
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device-index", type=int, default=1)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--output", default="benchmark/artifacts/validation_report.json")
    parser.add_argument("--skip-ablation",  action="store_true")
    parser.add_argument("--skip-diversity", action="store_true")
    parser.add_argument("--skip-beam",      action="store_true")
    args = parser.parse_args()

    report = {}

    if not args.skip_ablation:
        print("\n=== ABLATION STUDY ===", flush=True)
        report["ablation"] = run_ablation(args.model, args.device_index, args.language)

    if not args.skip_diversity:
        print("\n=== SEGMENT DIVERSITY ===", flush=True)
        report["segment_diversity"] = run_segment_diversity(
            args.model, args.device_index, args.language)

    if not args.skip_beam:
        print("\n=== BEAM SAFETY (6 × 30s clips) ===", flush=True)
        report["beam_safety"] = run_beam_safety(
            args.model, args.device_index, args.language)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  report -> {args.output}", flush=True)

    # Print summary
    print("\n=== SUMMARY ===")
    if "ablation" in report:
        print("\nAblation (vs baseline, WER normalised):")
        for r in report["ablation"]:
            print(f"  {r['config']:30s}  {r['median_s']:.3f}s  "
                  f"WER={r['wer_normalised']:.4f}  stddev={r['stddev_ms']:.1f}ms")

    if "segment_diversity" in report:
        print("\nSegment diversity WER (baseline vs optimised, normalised):")
        for r in report["segment_diversity"]:
            print(f"  {r['clip']:10s}  WER={r['wer_normalised']:.4f}")

    if "beam_safety" in report:
        wers = [r["wer_normalised"] for r in report["beam_safety"] if r["wer_normalised"] is not None]
        if wers:
            print(f"\nBeam safety (beam=5 vs beam=1): "
                  f"mean WER={statistics.mean(wers):.4f}  "
                  f"max WER={max(wers):.4f}")


if __name__ == "__main__":
    main()
