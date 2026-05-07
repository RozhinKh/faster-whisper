"""
Code regression test suite — 40 test cases.

Split into two categories:

  PART A — Preprocessing hash tests (20 cases)
    CPU-only pipeline: audio decode → VAD → feature extraction.
    No GPU involved, so output must be bit-identical between
    baseline code and PR code.  Any hash mismatch = real code bug.

  PART B — Transcription WER tests (20 cases)
    Full GPU transcription. GPU float arithmetic is non-deterministic
    between model loads, so exact hashes cannot be compared.
    Instead: WER between baseline and PR must be < WER_THRESHOLD.
    Differences above the threshold indicate a real accuracy regression.

Usage:
    # Requires /tmp/fw-main (main branch clone):
    #   git clone -b main https://github.com/RozhinKh/faster-whisper.git /tmp/fw-main

    python benchmark/test_code_regression.py \\
        --model /home/rozhin/rozhin/models/faster-whisper-large-v3 \\
        --device-index 1
"""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import statistics
import sys

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO = os.path.join(BENCHMARK_DIR, "benchmark.m4a")
SAMPLING_RATE = 16000
WER_THRESHOLD = 0.05  # 5% — any regression above this is a real problem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:12]


def _strip_punct(text: str) -> str:
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip().lower()


def _wer(ref: str, hyp: str) -> float:
    try:
        from jiwer import wer
        return round(wer(_strip_punct(ref), _strip_punct(hyp)), 4)
    except ImportError:
        return None


def _load_audio(clip_start=None, clip_end=None):
    from faster_whisper.audio import decode_audio
    audio = decode_audio(AUDIO)
    if clip_start is not None:
        s = int(clip_start * SAMPLING_RATE)
        e = int(clip_end * SAMPLING_RATE) if clip_end else len(audio)
        audio = audio[s:e]
    return audio


# ---------------------------------------------------------------------------
# PART A — Preprocessing hash tests (CPU only)
# ---------------------------------------------------------------------------

def _preprocess_with_code(audio, fe_path, vad_path):
    """
    Run VAD + feature extraction using code from a specific branch.
    Returns (vad_timestamps_hash, features_hash).
    """
    import numpy as np

    fe_mod  = _load_module("_fe",  fe_path)
    vad_mod = _load_module("_vad", vad_path)

    # VAD
    timestamps = vad_mod.get_speech_timestamps(audio, sampling_rate=SAMPLING_RATE)
    ts_bytes = json.dumps(
        [{"start": t["start"], "end": t["end"]} for t in timestamps]
    ).encode()

    # Feature extraction (first 30s chunk, representative)
    chunk = audio[:SAMPLING_RATE * 30]
    fe = fe_mod.FeatureExtractor()
    features = fe(chunk)
    feat_bytes = features.tobytes()

    return sha1(ts_bytes), sha1(feat_bytes)


def run_part_a(baseline_dir):
    print("\n" + "=" * 80)
    print("PART A — Preprocessing hash tests (CPU only, 20 cases)")
    print("Expected: 20/20 PASS  (code changes must not change CPU output)")
    print("=" * 80)

    from faster_whisper.audio import decode_audio
    audio_full = decode_audio(AUDIO)
    total_s = len(audio_full) / SAMPLING_RATE

    baseline_fe  = os.path.join(baseline_dir, "faster_whisper", "feature_extractor.py")
    baseline_vad = os.path.join(baseline_dir, "faster_whisper", "vad.py")

    venv_fw = os.path.dirname(__import__("faster_whisper").__file__)
    pr_fe   = os.path.join(venv_fw, "feature_extractor.py")
    pr_vad  = os.path.join(venv_fw, "vad.py")

    for p in [baseline_fe, baseline_vad, pr_fe, pr_vad]:
        if not os.path.exists(p):
            print(f"ERROR: not found: {p}")
            sys.exit(1)

    # 20 clips: full audio + 19 evenly-spaced 60s windows
    clips = [(None, None, "full audio      ")]
    positions = [total_s * i / 19 for i in range(19)]
    for i, start in enumerate(positions):
        end = min(start + 60, total_s)
        clips.append((start, end, f"clip {i+1:02d} ({start:5.0f}s)"))

    passed = 0
    results = []

    print(f"\n{'Test case':<25} {'VAD hash':>12}  {'VAD match':>10}  {'FFT hash':>12}  {'FFT match':>10}")
    print("-" * 80)

    for clip_start, clip_end, label in clips:
        audio = _load_audio(clip_start, clip_end)

        vad_base, fft_base = _preprocess_with_code(audio, baseline_fe, baseline_vad)
        vad_pr,   fft_pr   = _preprocess_with_code(audio, pr_fe,       pr_vad)

        vad_ok = vad_base == vad_pr
        fft_ok = fft_base == fft_pr
        ok = vad_ok and fft_ok
        if ok:
            passed += 1

        status_v = "PASS ✓" if vad_ok else "FAIL ✗"
        status_f = "PASS ✓" if fft_ok else "FAIL ✗"
        print(f"  {label:<23} {vad_base}  {status_v:>10}  {fft_base}  {status_f:>10}")
        results.append({
            "test": label.strip(), "vad_match": vad_ok, "fft_match": fft_ok,
            "vad_base": vad_base, "vad_pr": vad_pr,
            "fft_base": fft_base, "fft_pr": fft_pr,
        })

    pct = 100 * passed / len(clips)
    print("-" * 80)
    print(f"\n  PASS RATE: {passed}/{len(clips)} ({pct:.0f}%)")
    return results, passed, len(clips)


# ---------------------------------------------------------------------------
# PART B — Transcription WER tests (GPU, WER threshold)
# ---------------------------------------------------------------------------

def _transcribe(model_path, device_index, compute_type, batch_size, beam_size,
                clip_start=None, clip_end=None):
    from faster_whisper import WhisperModel, BatchedInferencePipeline
    audio = _load_audio(clip_start, clip_end)
    model = WhisperModel(model_path, device="cuda",
                         device_index=device_index, compute_type=compute_type)
    pipeline = BatchedInferencePipeline(model)
    segs, _ = pipeline.transcribe(audio, language="fr",
                                  batch_size=batch_size, beam_size=beam_size)
    text = "".join(s.text for s in segs)
    del model, pipeline
    return text


def run_part_b(model_path, device_index):
    print("\n" + "=" * 80)
    print(f"PART B — Transcription WER tests (GPU, 20 cases, threshold={WER_THRESHOLD:.0%})")
    print("Baseline: float16 bs=16 beam=5    PR: int8_float16 bs=32 beam=1")
    print("GPU is non-deterministic between loads — WER measures real accuracy change")
    print("=" * 80)

    from faster_whisper.audio import decode_audio
    total_s = len(decode_audio(AUDIO)) / SAMPLING_RATE

    clips = [(None, None, "full audio      ")]
    positions = [total_s * i / 19 for i in range(19)]
    for i, start in enumerate(positions):
        end = min(start + 60, total_s)
        clips.append((start, end, f"clip {i+1:02d} ({start:5.0f}s)"))

    passed = 0
    results = []
    wer_values = []

    print(f"\n{'Test case':<25} {'WER':>8}  {'Chars diff':>12}  {'Result':>10}")
    print("-" * 65)

    for clip_start, clip_end, label in clips:
        base_text = _transcribe(model_path, device_index, "float16",      16, 5,
                                clip_start, clip_end)
        pr_text   = _transcribe(model_path, device_index, "int8_float16", 32, 1,
                                clip_start, clip_end)

        w = _wer(base_text, pr_text)
        chars_diff = len(pr_text) - len(base_text)
        ok = (w is not None and w <= WER_THRESHOLD)
        if ok:
            passed += 1
        if w is not None:
            wer_values.append(w)

        status = "PASS ✓" if ok else "FAIL ✗"
        print(f"  {label:<23} {w:>8.4f}  {chars_diff:>+12d}  {status:>10}")
        results.append({
            "test": label.strip(), "wer": w, "chars_diff": chars_diff, "pass": ok,
            "base_chars": len(base_text), "pr_chars": len(pr_text),
        })

    pct = 100 * passed / len(clips)
    mean_wer = statistics.mean(wer_values) if wer_values else None
    max_wer  = max(wer_values) if wer_values else None

    print("-" * 65)
    print(f"\n  PASS RATE : {passed}/{len(clips)} ({pct:.0f}%)")
    if mean_wer is not None:
        print(f"  Mean WER  : {mean_wer:.4f} ({mean_wer*100:.2f}%)")
        print(f"  Max WER   : {max_wer:.4f}  ({max_wer*100:.2f}%)")
    return results, passed, len(clips), mean_wer, max_wer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device-index", type=int, default=1)
    parser.add_argument("--baseline-dir", default="/tmp/fw-main")
    parser.add_argument("--output",
                        default="benchmark/artifacts/regression_report.json")
    parser.add_argument("--skip-part-a", action="store_true")
    parser.add_argument("--skip-part-b", action="store_true")
    args = parser.parse_args()

    report = {}

    if not args.skip_part_a:
        res_a, pass_a, total_a = run_part_a(args.baseline_dir)
        report["part_a_preprocessing"] = {
            "description": "CPU-only hash comparison (VAD + FFT)",
            "passed": pass_a, "total": total_a,
            "pass_rate_pct": round(100 * pass_a / total_a, 1),
            "tests": res_a,
        }

    if not args.skip_part_b:
        res_b, pass_b, total_b, mean_wer, max_wer = run_part_b(
            args.model, args.device_index)
        report["part_b_transcription"] = {
            "description": f"GPU transcription WER (threshold {WER_THRESHOLD:.0%})",
            "passed": pass_b, "total": total_b,
            "pass_rate_pct": round(100 * pass_b / total_b, 1),
            "mean_wer": round(mean_wer, 4) if mean_wer else None,
            "max_wer":  round(max_wer,  4) if max_wer  else None,
            "tests": res_b,
        }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    if "part_a_preprocessing" in report:
        pa = report["part_a_preprocessing"]
        status = "✅ PASS" if pa["passed"] == pa["total"] else "❌ FAIL"
        print(f"  Part A (preprocessing hash) : {pa['passed']}/{pa['total']} ({pa['pass_rate_pct']:.0f}%)  {status}")
    if "part_b_transcription" in report:
        pb = report["part_b_transcription"]
        status = "✅ PASS" if pb["passed"] == pb["total"] else "⚠️  CHECK"
        print(f"  Part B (transcription WER)  : {pb['passed']}/{pb['total']} ({pb['pass_rate_pct']:.0f}%)  "
              f"mean WER={pb['mean_wer']:.4f}  max WER={pb['max_wer']:.4f}  {status}")
    print(f"\n  report -> {args.output}")


if __name__ == "__main__":
    main()
