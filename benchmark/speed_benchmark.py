"""
Speed benchmark for faster-whisper.

Default mode mirrors the original benchmark (single config, timeit.repeat).
Use flags to run a full comparison suite.

Usage:
    # Original single-config benchmark:
    python benchmark/speed_benchmark.py

    # Speed + VRAM comparison:
    CUDA_VISIBLE_DEVICES=3 python benchmark/speed_benchmark.py --compare

    # Full suite (speed + VRAM + preprocessing + noise robustness):
    CUDA_VISIBLE_DEVICES=3 python benchmark/speed_benchmark.py --all

    # Individual flags:
    CUDA_VISIBLE_DEVICES=3 python benchmark/speed_benchmark.py --compare --noise --preprocessing
"""

import argparse
import json
import os
import re
import statistics
import sys
import timeit
from pathlib import Path
from typing import Callable

from utils import inference, make_inference_fn

BENCHMARK_DIR   = Path(__file__).parent
BENCHMARK_AUDIO = str(BENCHMARK_DIR / "benchmark.m4a")
VARIANTS_DIR    = str(BENCHMARK_DIR / "audio_variants")

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

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="faster-whisper speed benchmark")
parser.add_argument("--repeat",       type=int, default=3,
                    help="timeit repetitions (default 3)")
parser.add_argument("--number",       type=int, default=10,
                    help="transcriptions per repetition (default 10)")
parser.add_argument("--compare",      action="store_true",
                    help="baseline vs candidate: speed + VRAM")
parser.add_argument("--noise",        action="store_true",
                    help="noise robustness across 6 degradation conditions")
parser.add_argument("--preprocessing", action="store_true",
                    help="VAD + FFT preprocessing timing breakdown")
parser.add_argument("--all",          action="store_true",
                    help="run all sections: --compare --noise --preprocessing")
parser.add_argument("--device",       default="cuda")
parser.add_argument("--device-index", type=int, default=0)
parser.add_argument("--language",     default="fr")
parser.add_argument("--output",       default=None,
                    help="save all results to JSON")
args = parser.parse_args()

if args.all:
    args.compare      = True
    args.noise        = True
    args.preprocessing = True

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _vram_mib(device_index: int) -> int | None:
    try:
        import py3nvml.py3nvml as nvml
        nvml.nvmlInit()
        handle = nvml.nvmlDeviceGetHandleByIndex(device_index)
        used = nvml.nvmlDeviceGetMemoryInfo(handle).used >> 20
        nvml.nvmlShutdown()
        return used
    except Exception:
        return None


def _strip_punct(text: str) -> str:
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip().lower()


def _wer(ref: str, hyp: str) -> float:
    from jiwer import wer
    return round(wer(_strip_punct(ref), _strip_punct(hyp)), 4)


def _transcribe(pipeline, audio_path: str, beam_size: int,
                batch_size: int, language: str) -> str:
    segs, _ = pipeline.transcribe(
        audio_path, language=language,
        beam_size=beam_size, batch_size=batch_size,
    )
    return "".join(s.text for s in segs)


def _load_model(compute_type: str):
    from faster_whisper import BatchedInferencePipeline, WhisperModel
    model = WhisperModel(
        "large-v3",
        device=args.device,
        device_index=args.device_index,
        compute_type=compute_type,
    )
    return model, BatchedInferencePipeline(model)


def _section(title: str):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")


# ─────────────────────────────────────────────────────────────────────────────
# Original single-config mode
# ─────────────────────────────────────────────────────────────────────────────

def measure_speed(func: Callable[[], None]):
    runtimes = timeit.repeat(func, repeat=args.repeat, number=args.number)
    print(runtimes)
    print("Min execution time: %.3fs" % (min(runtimes) / args.number))
    return runtimes


if not (args.compare or args.noise or args.preprocessing):
    measure_speed(inference)
    sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# Full suite
# ─────────────────────────────────────────────────────────────────────────────

all_results = {}

# ── SECTION 1: Speed + VRAM ──────────────────────────────────────────────────

if args.compare:
    _section("SPEED + VRAM  (timeit.repeat — official speed_benchmark methodology)")
    speed_results = {}

    for config in CONFIGS:
        print(f"\n  {config['label']}")
        print(f"  {'-'*56}")
        sys.stdout.flush()

        fn = make_inference_fn(
            compute_type=config["compute_type"],
            beam_size=config["beam_size"],
            batch_size=config["batch_size"],
            batched=True,
            device=args.device,
            device_index=args.device_index,
            language=args.language,
        )

        vram = _vram_mib(args.device_index)

        print("  Warmup...")
        sys.stdout.flush()
        fn()

        print(f"  Timing (repeat={args.repeat}, number={args.number})...")
        sys.stdout.flush()
        runtimes = timeit.repeat(fn, repeat=args.repeat, number=args.number)
        min_per_run = min(runtimes) / args.number

        print(f"  Raw totals : {[round(r, 3) for r in runtimes]}")
        print(f"  Min per run: {min_per_run:.3f}s")
        if vram is not None:
            print(f"  VRAM used  : {vram} MiB")

        speed_results[config["name"]] = {
            "label": config["label"],
            "compute_type": config["compute_type"],
            "beam_size": config["beam_size"],
            "batch_size": config["batch_size"],
            "runtimes": [round(r, 4) for r in runtimes],
            "min_per_run_s": round(min_per_run, 4),
            "median_per_run_s": round(statistics.median(runtimes) / args.number, 4),
            "vram_mib": vram,
        }
        del fn

    b = speed_results["baseline"]
    c = speed_results["candidate"]
    speedup = b["min_per_run_s"] / c["min_per_run_s"]
    pct     = (b["min_per_run_s"] - c["min_per_run_s"]) / b["min_per_run_s"] * 100

    vram_delta = ""
    if b["vram_mib"] and c["vram_mib"]:
        vram_pct = (b["vram_mib"] - c["vram_mib"]) / b["vram_mib"] * 100
        vram_delta = f"  VRAM: {b['vram_mib']} MiB → {c['vram_mib']} MiB  (−{vram_pct:.1f}%)"

    print(f"\n  {'─'*56}")
    print(f"  {b['label']}: {b['min_per_run_s']:.3f}s")
    print(f"  {c['label']}: {c['min_per_run_s']:.3f}s")
    print(f"  Speed:  {speedup:.2f}×  (−{pct:.1f}%)")
    if vram_delta:
        print(vram_delta)

    all_results["speed"] = speed_results

# ── SECTION 2: Preprocessing timing ─────────────────────────────────────────

if args.preprocessing:
    _section("PREPROCESSING  (VAD + FFT, 5-run median, CPU only)")
    sys.stdout.flush()

    try:
        from faster_whisper.audio import decode_audio
        from faster_whisper.feature_extractor import FeatureExtractor
        from faster_whisper.vad import get_speech_timestamps
        import time

        print(f"  Loading audio: {BENCHMARK_AUDIO}")
        audio = decode_audio(BENCHMARK_AUDIO)
        chunk = audio[: 16000 * 30]

        pre_runs = []
        for i in range(5):
            t0 = time.perf_counter()
            _ = get_speech_timestamps(audio, sampling_rate=16000)
            vad_t = time.perf_counter() - t0

            fe = FeatureExtractor()
            t0 = time.perf_counter()
            _ = fe(chunk)
            fft_t = time.perf_counter() - t0

            pre_runs.append((vad_t, fft_t))
            print(f"  run {i+1}/5  VAD {vad_t:.3f}s  FFT {fft_t:.4f}s", end="\r")

        print()
        vad_med = statistics.median(r[0] for r in pre_runs)
        fft_med = statistics.median(r[1] for r in pre_runs)
        total   = vad_med + fft_med

        print(f"\n  VAD (median):  {vad_med:.3f}s")
        print(f"  FFT (median):  {fft_med:.4f}s")
        print(f"  Total:         {total:.3f}s")
        print(f"  Note: preprocessing is config-independent (CPU only).")

        all_results["preprocessing"] = {
            "vad_median_s": round(vad_med, 4),
            "fft_median_s": round(fft_med, 5),
            "total_median_s": round(total, 4),
            "runs": 5,
        }
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  (preprocessing benchmark requires faster_whisper in Python path)")

# ── SECTION 3: Noise robustness ──────────────────────────────────────────────

if args.noise:
    _section("NOISE ROBUSTNESS  (6 conditions, WER vs clean baseline)")

    NOISE_VARIANTS = [
        ("clean",       "clean_full.wav",    "Clean (reference)"),
        ("noisy_snr20", "noisy_snr20.wav",   "Noise SNR 20 dB (slight)"),
        ("noisy_snr10", "noisy_snr10.wav",   "Noise SNR 10 dB (moderate)"),
        ("noisy_snr5",  "noisy_snr5.wav",    "Noise SNR  5 dB (heavy)"),
        ("telephone",   "telephone.wav",     "Telephone quality (300–3400 Hz)"),
        ("overlapping", "overlapping.wav",   "Overlapping speech (−6 dB)"),
    ]
    PASS_THRESHOLD = 0.03

    missing = [
        v[1] for v in NOISE_VARIANTS
        if not os.path.exists(os.path.join(VARIANTS_DIR, v[1]))
    ]
    if missing:
        print(f"\n  Audio variants not found — generating now...")
        sys.stdout.flush()
        import subprocess
        result = subprocess.run(
            [sys.executable, str(BENCHMARK_DIR / "generate_audio_variants.py")],
            check=True,
        )

    print("\n  Loading models...")
    sys.stdout.flush()
    model_b, pipe_b = _load_model("float16")
    model_o, pipe_o = _load_model("int8_float16")

    print("\n  Getting clean reference transcript (baseline)...")
    sys.stdout.flush()
    ref_text = _transcribe(
        pipe_b,
        os.path.join(VARIANTS_DIR, "clean_full.wav"),
        beam_size=5, batch_size=16, language=args.language,
    )
    print(f"  Reference: {len(ref_text)} chars\n")

    fmt = "  {:<35} {:>12}  {:>13}  {:>8}  {:>8}"
    print(fmt.format("Condition", "Baseline WER", "Candidate WER", "Delta", "Result"))
    print("  " + "─" * 78)

    noise_results = []
    for key, filename, label in NOISE_VARIANTS:
        path = os.path.join(VARIANTS_DIR, filename)
        sys.stdout.flush()

        base_text = _transcribe(pipe_b, path, 5,  16, args.language)
        opt_text  = _transcribe(pipe_o, path, 1,  32, args.language)

        wer_base = _wer(ref_text, base_text)
        wer_opt  = _wer(ref_text, opt_text)
        delta    = round(wer_opt - wer_base, 4)
        passed   = delta <= PASS_THRESHOLD

        print(fmt.format(
            label, f"{wer_base:.4f}", f"{wer_opt:.4f}",
            f"{delta:+.4f}", "PASS ✓" if passed else "FAIL ✗",
        ))

        noise_results.append({
            "condition": key, "label": label,
            "baseline_wer": wer_base, "candidate_wer": wer_opt,
            "delta": delta, "passed": passed,
        })

    print("  " + "─" * 78)
    n_pass = sum(r["passed"] for r in noise_results)
    print(f"\n  Pass rate: {n_pass}/{len(noise_results)}")
    print(f"  WER relative to clean baseline (float16/beam=5) transcript.")
    print(f"  Delta = candidate WER − baseline WER on the same degraded audio.")

    all_results["noise"] = {
        "pass_threshold": PASS_THRESHOLD,
        "pass_rate": f"{n_pass}/{len(noise_results)}",
        "results": noise_results,
    }

# ── Final JSON output ─────────────────────────────────────────────────────────

if args.output and all_results:
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"\n\n  Results saved → {args.output}")
