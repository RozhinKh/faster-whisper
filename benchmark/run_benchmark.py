"""
Benchmark runner for faster-whisper.

Suites:
  speed       – one warmup + one timed transcription of benchmark.m4a
  memory      – peak GPU VRAM and power during transcription
  wer         – Word Error Rate on LibriSpeech clean (default: 50 samples)
  regression  – detect output changes vs a saved pre-optimisation reference
  all         – speed + memory + wer + regression

Usage:
    # GA loop (fast, ~2 min)
    python benchmark/ga_benchmark.py --model <path> --output artemis_results.json

    # Full evaluation before/after optimisation (~10 min)
    python benchmark/run_benchmark.py --model <path> --suite all

    # Save regression reference once before optimising
    python benchmark/run_benchmark.py --model <path> --suite regression --regression-save

    # Quick accuracy spot-check
    python benchmark/run_benchmark.py --model <path> --suite wer --audio-numb 50
"""

import argparse
import json
import os
import sys
import time
import timeit
from datetime import datetime
from typing import Optional

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BENCHMARK_DIR)
for _p in (BENCHMARK_DIR, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import py3nvml.py3nvml as nvml
from jiwer import wer as compute_wer
from memory_profiler import memory_usage
from tqdm import tqdm
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from faster_whisper import WhisperModel
from regression import compare_against_reference, save_reference
from utils import MyThread

AUDIO = os.path.join(BENCHMARK_DIR, "benchmark.m4a")
NORMALIZER_JSON = os.path.join(BENCHMARK_DIR, "normalizer.json")
RESULTS_DIR = os.path.join(BENCHMARK_DIR, "results")


def _normalizer():
    with open(NORMALIZER_JSON, encoding="utf-8") as f:
        import json as _json
        return EnglishTextNormalizer(_json.load(f))


def _header(title: str):
    print(f"\n{'=' * 56}\nSUITE: {title}\n{'=' * 56}")


# ── Speed ─────────────────────────────────────────────────────

def suite_speed(model_size, compute_type, device, device_index) -> dict:
    _header("SPEED")
    model = WhisperModel(model_size, device=device, device_index=device_index,
                         compute_type=compute_type)

    def _run():
        for _ in model.transcribe(AUDIO, language="fr")[0]:
            pass

    _run()  # warmup
    t0 = time.perf_counter()
    _run()
    elapsed = round(time.perf_counter() - t0, 3)
    print(f"  transcription time : {elapsed}s")
    return {"transcription_time_s": elapsed}


# ── Memory ────────────────────────────────────────────────────

def suite_memory(model_size, compute_type, device, device_index,
                 interval: float = 0.5) -> dict:
    _header("MEMORY")
    model = WhisperModel(model_size, device=device, device_index=device_index,
                         compute_type=compute_type)

    def _run():
        for _ in model.transcribe(AUDIO, language="fr")[0]:
            pass

    if device != "cuda":
        increase = memory_usage(_run, max_usage=True, interval=interval)
        print(f"  RAM increase : {increase:.0f} MiB")
        return {"ram_increase_mib": round(increase, 1)}

    idx = device_index if isinstance(device_index, int) else device_index[0]
    nvml.nvmlInit()
    handle = nvml.nvmlDeviceGetHandleByIndex(idx)
    mem_limit = nvml.nvmlDeviceGetMemoryInfo(handle).total >> 20
    pwr_limit = nvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
    samples = {"mem": [], "pwr": []}
    stop = [False]

    def _poll():
        while not stop[0]:
            samples["mem"].append(nvml.nvmlDeviceGetMemoryInfo(handle).used >> 20)
            samples["pwr"].append(nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
            time.sleep(interval)

    t = MyThread(_poll, params=())
    t.start()
    _run()
    stop[0] = True
    t.join()
    nvml.nvmlShutdown()

    max_mem = max(samples["mem"])
    max_pwr = max(samples["pwr"])
    print(f"  VRAM : {max_mem} / {mem_limit} MiB  ({max_mem/mem_limit*100:.1f}%)")
    print(f"  power: {max_pwr:.0f} / {pwr_limit:.0f} W  ({max_pwr/pwr_limit*100:.1f}%)")
    return {
        "max_vram_mib": max_mem, "vram_limit_mib": mem_limit,
        "vram_pct": round(max_mem / mem_limit * 100, 2),
        "max_power_w": round(max_pwr, 1), "power_limit_w": round(pwr_limit, 1),
    }


# ── WER ───────────────────────────────────────────────────────

def suite_wer(model_size, compute_type, device, device_index,
              audio_numb: Optional[int] = 50) -> dict:
    _header(f"WER  (LibriSpeech clean, {audio_numb or 'all'} samples)")
    from datasets import load_dataset
    model = WhisperModel(model_size, device=device, device_index=device_index,
                         compute_type=compute_type)
    norm = _normalizer()
    ds = load_dataset("librispeech_asr", "clean", split="validation", streaming=True)

    def _infer(batch):
        batch["transcription"] = [
            "".join(s.text for s in model.transcribe(a["array"], language="en")[0])
            for a in batch["audio"]
        ]
        batch["reference"] = batch["text"]
        return batch

    ds = ds.map(_infer, batched=True, batch_size=16)
    hyps, refs = [], []
    for i, row in tqdm(enumerate(ds), desc="WER"):
        hyps.append(row["transcription"])
        refs.append(row["reference"])
        if audio_numb and i >= audio_numb - 1:
            break

    wer_pct = round(100 * compute_wer(
        hypothesis=[norm(h) for h in hyps],
        reference=[norm(r) for r in refs]
    ), 3)
    print(f"  WER : {wer_pct}%  ({len(hyps)} samples)")
    return {"wer_pct": wer_pct, "num_samples": len(hyps)}


# ── Regression ────────────────────────────────────────────────

def suite_regression(model_size, compute_type, device, device_index,
                     do_save: bool) -> dict:
    _header("REGRESSION")
    audio_files = [AUDIO]
    if do_save:
        save_reference(model_size, compute_type, device, device_index, audio_files)
        return {"action": "saved"}
    return compare_against_reference(model_size, compute_type, device, device_index)


# ── Orchestrator ──────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="faster-whisper benchmark runner")
    p.add_argument("--suite", nargs="+",
                   choices=["speed", "memory", "wer", "regression", "all"],
                   default=["all"])
    p.add_argument("--model", default="large-v3")
    p.add_argument("--compute-type", default="float16")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--device-index", type=int, nargs="+", default=[0])
    p.add_argument("--audio-numb", type=int, default=50,
                   help="WER sample count (default: 50)")
    p.add_argument("--regression-save", action="store_true",
                   help="Save reference outputs before optimising")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    suites = {"speed", "memory", "wer", "regression"} if "all" in args.suite \
        else set(args.suite)
    device_index = args.device_index[0] if len(args.device_index) == 1 \
        else args.device_index

    results = {"run_at": datetime.now().isoformat(),
               "model": args.model, "compute_type": args.compute_type,
               "suites": {}}

    if "speed" in suites:
        results["suites"]["speed"] = suite_speed(
            args.model, args.compute_type, args.device, device_index)

    if "memory" in suites:
        results["suites"]["memory"] = suite_memory(
            args.model, args.compute_type, args.device, device_index)

    if "wer" in suites:
        results["suites"]["wer"] = suite_wer(
            args.model, args.compute_type, args.device, device_index, args.audio_numb)

    if "regression" in suites:
        results["suites"]["regression"] = suite_regression(
            args.model, args.compute_type, args.device, device_index,
            do_save=args.regression_save)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = args.output or os.path.join(
        RESULTS_DIR, f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults → {out}")


if __name__ == "__main__":
    main()
