"""
Code regression test: baseline (main branch) vs PR code, same config each time.

For each test case:
  1. Run transcription using main-branch feature_extractor + vad (CPU numpy)
  2. Run transcription using PR feature_extractor + vad (scipy FFT + CUDA VAD)
  3. Compare SHA1 hashes — must be identical

A pass rate of 100% means the PR code is numerically transparent.
Any failure means the code changes affected output and must be investigated.

Usage:
    # Make sure /tmp/fw-main exists (main branch clone):
    #   git clone -b main https://github.com/RozhinKh/faster-whisper.git /tmp/fw-main

    python benchmark/test_code_regression.py \
        --model /home/rozhin/rozhin/models/faster-whisper-large-v3 \
        --device-index 1
"""

import argparse
import hashlib
import importlib.util
import os
import sys
import types

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO = os.path.join(BENCHMARK_DIR, "benchmark.m4a")
SAMPLING_RATE = 16000


def _load_module_from_path(module_name, file_path):
    """Import a single .py file as a module without affecting sys.modules."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _transcribe_with_code(
    model_path, device_index, compute_type,
    batch_size, beam_size, language,
    clip_start_s=None, clip_end_s=None,
    feature_extractor_path=None,
    vad_path=None,
):
    """
    Run one transcription, optionally overriding feature_extractor and vad
    with files from a specific branch (for baseline vs PR comparison).
    """
    from faster_whisper import WhisperModel, BatchedInferencePipeline
    from faster_whisper.audio import decode_audio
    import faster_whisper.transcribe as transcribe_mod

    # Temporarily patch feature_extractor and vad in the transcribe module
    # so it uses the code version we want without reinstalling anything.
    original_fe = None
    original_vad = None

    if feature_extractor_path:
        fe_mod = _load_module_from_path("_fe_override", feature_extractor_path)
        original_fe = transcribe_mod.FeatureExtractor
        transcribe_mod.FeatureExtractor = fe_mod.FeatureExtractor

    if vad_path:
        vad_mod = _load_module_from_path("_vad_override", vad_path)
        import faster_whisper.vad as vad_module
        original_get_speech = vad_module.get_speech_timestamps
        original_collect = vad_module.collect_chunks
        vad_module.get_speech_timestamps = vad_mod.get_speech_timestamps
        vad_module.collect_chunks = vad_mod.collect_chunks

    try:
        audio = decode_audio(AUDIO)
        if clip_start_s is not None:
            s = int(clip_start_s * SAMPLING_RATE)
            e = int(clip_end_s * SAMPLING_RATE) if clip_end_s else len(audio)
            audio = audio[s:e]

        model = WhisperModel(model_path, device="cuda",
                             device_index=device_index, compute_type=compute_type)
        pipeline = BatchedInferencePipeline(model)
        segs, _ = pipeline.transcribe(
            audio, language=language, batch_size=batch_size, beam_size=beam_size
        )
        text = "".join(seg.text for seg in segs)
        del model, pipeline
        return text

    finally:
        if original_fe is not None:
            transcribe_mod.FeatureExtractor = original_fe
        if vad_path:
            vad_module.get_speech_timestamps = original_get_speech
            vad_module.collect_chunks = original_collect


def sha1(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def run_regression(model_path, device_index, language, baseline_dir):
    baseline_fe  = os.path.join(baseline_dir, "faster_whisper", "feature_extractor.py")
    baseline_vad = os.path.join(baseline_dir, "faster_whisper", "vad.py")

    if not os.path.exists(baseline_fe):
        print(f"ERROR: baseline not found at {baseline_fe}")
        print("Run: git clone -b main https://github.com/RozhinKh/faster-whisper.git /tmp/fw-main")
        sys.exit(1)

    from faster_whisper.audio import decode_audio
    audio = decode_audio(AUDIO)
    total_s = len(audio) / SAMPLING_RATE

    # 20 test cases: full audio + 9 clips × 2 configs each
    # Clips are evenly spaced across the full audio at 10% intervals
    clip_positions = [total_s * i / 10 for i in range(9)]  # 0%, 10%, ..., 80%
    clip_duration = 60.0

    test_cases = [
        ("full audio  | float16 bs=16 beam=5", "float16",      16, 5, None, None),
        ("full audio  | int8    bs=32 beam=1", "int8_float16", 32, 1, None, None),
    ]
    for i, start in enumerate(clip_positions):
        end = min(start + clip_duration, total_s)
        label_f = f"clip {i+1:02d} ({start:4.0f}s) | float16 bs=16 beam=5"
        label_i = f"clip {i+1:02d} ({start:4.0f}s) | int8    bs=32 beam=1"
        test_cases.append((label_f, "float16",      16, 5, start, end))
        test_cases.append((label_i, "int8_float16", 32, 1, start, end))

    # Trim to exactly 20
    test_cases = test_cases[:20]

    results = []
    passed = 0

    print(f"\n{'Test case':<45} {'Baseline SHA1':>12} {'PR SHA1':>12}  {'Match':>6}  {'Chars diff':>10}")
    print("-" * 100)

    for label, compute_type, batch_size, beam_size, clip_start, clip_end in test_cases:
        # Baseline: main branch feature_extractor + vad, same config
        text_baseline = _transcribe_with_code(
            model_path, device_index, compute_type,
            batch_size, beam_size, language,
            clip_start, clip_end,
            feature_extractor_path=baseline_fe,
            vad_path=baseline_vad,
        )

        # PR: installed (venv) feature_extractor + vad, same config
        text_pr = _transcribe_with_code(
            model_path, device_index, compute_type,
            batch_size, beam_size, language,
            clip_start, clip_end,
        )

        h_base = sha1(text_baseline)[:12]
        h_pr   = sha1(text_pr)[:12]
        match  = h_base == h_pr
        chars_diff = len(text_pr) - len(text_baseline)

        if match:
            passed += 1
            status = "PASS ✓"
        else:
            status = "FAIL ✗"

        print(f"  {label:<43} {h_base}  {h_pr}  {status}  {chars_diff:+d}")
        results.append({
            "test": label,
            "compute_type": compute_type,
            "batch_size": batch_size,
            "beam_size": beam_size,
            "baseline_sha1": sha1(text_baseline),
            "pr_sha1": sha1(text_pr),
            "match": match,
            "baseline_chars": len(text_baseline),
            "pr_chars": len(text_pr),
        })

    total = len(test_cases)
    pct = 100 * passed / total
    print("-" * 100)
    print(f"\n  PASS RATE: {passed}/{total} ({pct:.0f}%)")

    if passed == total:
        print("  RESULT: PR code is numerically identical to baseline across all test cases.")
    else:
        failed = [r for r in results if not r["match"]]
        print(f"  RESULT: {len(failed)} test(s) failed — PR code changed output.")
        for f in failed:
            print(f"    - {f['test']}")

    return results, passed, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device-index", type=int, default=1)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--baseline-dir", default="/tmp/fw-main",
                        help="Path to main-branch clone for baseline code")
    parser.add_argument("--output",
                        default="benchmark/artifacts/regression_report.json")
    args = parser.parse_args()

    results, passed, total = run_regression(
        args.model, args.device_index, args.language, args.baseline_dir
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    import json
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "passed": passed,
            "total": total,
            "pass_rate_pct": round(100 * passed / total, 1),
            "tests": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  report -> {args.output}")


if __name__ == "__main__":
    main()
