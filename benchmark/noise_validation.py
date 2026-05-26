"""
Noisy-audio validation: compares baseline vs optimised config on degraded audio.

Tests whether beam=1 (optimised) regresses more than beam=5 (baseline) on:
  - Clean audio            (expected: low WER, both configs similar)
  - Slight noise SNR 20dB  (expected: small WER increase)
  - Moderate noise SNR 10dB
  - Heavy noise SNR  5dB   (expected: beam=5 advantage becomes visible)
  - Telephone quality      (bandlimited 300-3400 Hz)
  - Overlapping speech     (two speakers)

Since we have no ground-truth transcript for degraded audio, WER is computed
with the CLEAN baseline (float16/beam=5) as the reference — this measures
how much each degradation + config change drifts from the clean reference.

Usage:
    # First generate variants:
    python benchmark/generate_audio_variants.py

    python benchmark/noise_validation.py \\
        --model /path/to/faster-whisper-large-v3 \\
        --device-index 1
"""

import argparse
import json
import os
import re
import statistics
import sys

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
VARIANTS_DIR  = os.path.join(BENCHMARK_DIR, "audio_variants")
SR = 16000


def _strip_punct(text):
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip().lower()


def _wer(ref, hyp):
    from jiwer import wer
    return round(wer(_strip_punct(ref), _strip_punct(hyp)), 4)


def _transcribe(model, pipeline, audio_path, batch_size, beam_size, language="fr"):
    from faster_whisper.audio import decode_audio
    audio = decode_audio(audio_path)
    segs, _ = pipeline.transcribe(audio, language=language,
                                  batch_size=batch_size, beam_size=beam_size)
    return "".join(s.text for s in segs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device-index", type=int, default=1)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--beam-size", type=int, default=1,
                        help="Beam size for optimised config (default 1).")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for optimised config (default 32).")
    parser.add_argument("--compute-type", default="int8_float16",
                        help="Compute type for optimised config (default int8_float16).")
    parser.add_argument("--output",
                        default="benchmark/artifacts/noise_validation.json")
    args = parser.parse_args()

    variants = [
        ("clean",        os.path.join(VARIANTS_DIR, "clean_full.wav"),   "Clean (reference)"),
        ("noisy_snr20",  os.path.join(VARIANTS_DIR, "noisy_snr20.wav"),  "Noise SNR 20 dB (slight)"),
        ("noisy_snr10",  os.path.join(VARIANTS_DIR, "noisy_snr10.wav"),  "Noise SNR 10 dB (moderate)"),
        ("noisy_snr5",   os.path.join(VARIANTS_DIR, "noisy_snr5.wav"),   "Noise SNR  5 dB (heavy)"),
        ("telephone",    os.path.join(VARIANTS_DIR, "telephone.wav"),    "Telephone quality (300-3400 Hz)"),
        ("overlapping",  os.path.join(VARIANTS_DIR, "overlapping.wav"),  "Overlapping speech (−6 dB)"),
    ]

    missing = [v[1] for v in variants if not os.path.exists(v[1])]
    if missing:
        print("Missing audio variants. Run first:")
        print("  python benchmark/generate_audio_variants.py")
        sys.exit(1)

    from faster_whisper import WhisperModel, BatchedInferencePipeline

    print("\nLoading baseline model (float16) ...")
    model_b = WhisperModel(args.model, device="cuda",
                           device_index=args.device_index, compute_type="float16")
    pipe_b  = BatchedInferencePipeline(model_b)

    print(f"Loading optimised model ({args.compute_type}, beam={args.beam_size}, batch={args.batch_size}) ...")
    model_o = WhisperModel(args.model, device="cuda",
                           device_index=args.device_index, compute_type=args.compute_type)
    pipe_o  = BatchedInferencePipeline(model_o)

    # Get clean reference transcript (baseline config on clean audio)
    print("\nTranscribing clean reference ...")
    ref_text = _transcribe(model_b, pipe_b,
                           os.path.join(VARIANTS_DIR, "clean_full.wav"),
                           batch_size=16, beam_size=5, language=args.language)
    print(f"  reference: {len(ref_text)} chars")

    results = []
    print(f"\n{'Condition':<35} {'Baseline WER':>13} {'Optimised WER':>14} {'Delta':>8} {'Verdict':>10}")
    print("-" * 88)

    for key, path, label in variants:
        base_text = _transcribe(model_b, pipe_b, path, 16, 5,  args.language)
        opt_text  = _transcribe(model_o, pipe_o, path, args.batch_size, args.beam_size, args.language)

        wer_base = _wer(ref_text, base_text)
        wer_opt  = _wer(ref_text, opt_text)
        delta    = round(wer_opt - wer_base, 4)

        # Verdict: optimised is acceptable if it doesn't regress more than 3% over baseline
        verdict = "OK ✓" if delta <= 0.03 else "REGRESS ✗"

        print(f"  {label:<33} {wer_base:>13.4f} {wer_opt:>14.4f} {delta:>+8.4f} {verdict:>10}")
        results.append({
            "condition": key, "label": label,
            "baseline_wer": wer_base, "optimised_wer": wer_opt,
            "delta": delta, "pass": delta <= 0.03,
        })

    print("-" * 88)
    passed = sum(r["pass"] for r in results)
    print(f"\n  PASS RATE: {passed}/{len(results)}")
    print(f"  Note: WER is relative to clean baseline (float16/beam=5) transcript.")
    print(f"  Optimised config: {args.compute_type}, beam={args.beam_size}, batch={args.batch_size}.")
    print(f"  Delta = optimised WER − baseline WER on same degraded audio.")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"reference_chars": len(ref_text), "results": results}, f,
                  indent=2, ensure_ascii=False)
    print(f"\n  report -> {args.output}")


if __name__ == "__main__":
    main()
