"""
Unified benchmark runner for faster-whisper.

Integrates all existing benchmarks and new ones into a single CLI:

  speed       – timeit-based transcription speed on benchmark.m4a
  memory      – GPU VRAM, GPU power draw, and RAM usage
  wer         – Word Error Rate on LibriSpeech clean validation
  yt_commons  – Word Error Rate on YouTube Commons ASR dataset
  throughput  – Multi-GPU, RTF, batch-size sweep, num_workers sweep, concurrency
  regression  – Output change detection vs a saved pre-optimisation reference
  all         – Run every suite above

Usage examples:

    # Full run
    python benchmark/run_benchmark.py --suite all

    # Only speed + memory
    python benchmark/run_benchmark.py --suite speed memory

    # Save regression reference before optimisation
    python benchmark/run_benchmark.py --suite regression --regression-save

    # Compare after optimisation (reference must exist)
    python benchmark/run_benchmark.py --suite regression

    # Multi-GPU throughput sweep on all 4 RTX 3090s
    python benchmark/run_benchmark.py --suite throughput --gpu-indices 0 1 2 3

    # Limit WER / YouTube samples to run faster
    python benchmark/run_benchmark.py --suite wer yt_commons --audio-numb 100
"""

import argparse
import json
import os
import sys
import time
import timeit
from datetime import datetime
from typing import Optional

# Allow imports from both the benchmark/ folder and the repo root.
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

from faster_whisper import BatchedInferencePipeline, WhisperModel, decode_audio
from regression import compare_against_reference, save_reference
from throughput_benchmark import run_throughput_benchmark
from utils import MyThread

BENCHMARK_M4A = os.path.join(BENCHMARK_DIR, "benchmark.m4a")
NORMALIZER_JSON = os.path.join(BENCHMARK_DIR, "normalizer.json")
RESULTS_DIR = os.path.join(BENCHMARK_DIR, "results")


def _load_normalizer() -> EnglishTextNormalizer:
    with open(NORMALIZER_JSON, encoding="utf-8") as f:
        return EnglishTextNormalizer(json.load(f))


def _section(title: str):
    bar = "=" * 64
    print(f"\n{bar}\nSUITE: {title}\n{bar}")


# ── Suite: Speed ──────────────────────────────────────────────────────────────

def suite_speed(
    model_size: str,
    compute_type: str,
    device: str,
    device_index,
    repeat: int,
    number: int,
) -> dict:
    _section("SPEED  (benchmark.m4a, timeit)")

    model = WhisperModel(
        model_size, device=device, device_index=device_index, compute_type=compute_type
    )

    def _run():
        segments, _ = model.transcribe(BENCHMARK_M4A, language="fr")
        for _ in segments:
            pass

    # warmup — excluded from timing
    _run()

    runtimes = timeit.repeat(_run, repeat=repeat, number=number)
    per_run = [round(r / number, 3) for r in runtimes]
    min_s = min(per_run)

    print(f"  Per-run times (s) : {per_run}")
    print(f"  Min time          : {min_s:.3f}s")

    return {"per_run_times_s": per_run, "min_s": min_s}


# ── Suite: Memory ─────────────────────────────────────────────────────────────

def suite_memory(
    model_size: str,
    compute_type: str,
    device: str,
    device_index,
    measure_gpu: bool,
    interval: float,
) -> dict:
    _section("MEMORY")

    model = WhisperModel(
        model_size, device=device, device_index=device_index, compute_type=compute_type
    )

    def _run():
        segments, _ = model.transcribe(BENCHMARK_M4A, language="fr")
        for _ in segments:
            pass

    results = {}

    if measure_gpu and device == "cuda":
        idx = device_index if isinstance(device_index, int) else device_index[0]
        nvml.nvmlInit()
        handle = nvml.nvmlDeviceGetHandleByIndex(idx)
        gpu_name = nvml.nvmlDeviceGetName(handle)
        mem_limit = nvml.nvmlDeviceGetMemoryInfo(handle).total >> 20
        pwr_limit = nvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
        collected = {"mem": [], "pwr": []}
        stop_flag = [False]

        def _poll():
            while not stop_flag[0]:
                collected["mem"].append(nvml.nvmlDeviceGetMemoryInfo(handle).used >> 20)
                collected["pwr"].append(nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
                time.sleep(interval)

        thread = MyThread(_poll, params=())
        thread.start()
        _run()
        stop_flag[0] = True
        thread.join()
        nvml.nvmlShutdown()

        max_mem = max(collected["mem"])
        max_pwr = max(collected["pwr"])

        print(f"  GPU               : {gpu_name} (device {idx})")
        print(
            f"  Max VRAM          : {max_mem} MiB / {mem_limit} MiB "
            f"({max_mem / mem_limit * 100:.1f}%)"
        )
        print(
            f"  Max power draw    : {max_pwr:.0f} W / {pwr_limit:.0f} W "
            f"({max_pwr / pwr_limit * 100:.1f}%)"
        )

        results["gpu"] = {
            "name": gpu_name,
            "device_index": idx,
            "max_vram_mib": max_mem,
            "vram_limit_mib": mem_limit,
            "vram_pct": round(max_mem / mem_limit * 100, 2),
            "max_power_w": round(max_pwr, 1),
            "power_limit_w": round(pwr_limit, 1),
            "power_pct": round(max_pwr / pwr_limit * 100, 2),
        }
    else:
        max_ram = memory_usage(_run, max_usage=True, interval=interval)
        print(f"  Max RAM increase  : {max_ram:.0f} MiB")
        results["ram_increase_mib"] = round(max_ram, 1)

    return results


# ── Suite: WER – LibriSpeech ──────────────────────────────────────────────────

def suite_wer(
    model_size: str,
    compute_type: str,
    device: str,
    device_index,
    audio_numb: Optional[int],
) -> dict:
    _section("WER  (LibriSpeech clean validation)")

    from datasets import load_dataset

    model = WhisperModel(
        model_size, device=device, device_index=device_index, compute_type=compute_type
    )
    normalizer = _load_normalizer()
    dataset = load_dataset("librispeech_asr", "clean", split="validation", streaming=True)

    def _infer(batch):
        batch["transcription"] = []
        for sample in batch["audio"]:
            segs, _ = model.transcribe(sample["array"], language="en")
            batch["transcription"].append("".join(s.text for s in segs))
        batch["reference"] = batch["text"]
        return batch

    dataset = dataset.map(function=_infer, batched=True, batch_size=16)

    transcriptions, references = [], []
    for i, row in tqdm(enumerate(dataset), desc="LibriSpeech"):
        transcriptions.append(row["transcription"])
        references.append(row["reference"])
        if audio_numb and i >= audio_numb - 1:
            break

    transcriptions = [normalizer(t) for t in transcriptions]
    references = [normalizer(r) for r in references]
    wer_pct = 100 * compute_wer(hypothesis=transcriptions, reference=references)

    print(f"  WER : {wer_pct:.3f}%  ({len(transcriptions)} samples)")
    return {"wer_pct": round(wer_pct, 3), "num_samples": len(transcriptions)}


# ── Suite: WER – YouTube Commons ─────────────────────────────────────────────

def suite_yt_commons(
    model_size: str,
    compute_type: str,
    device: str,
    device_index,
    audio_numb: Optional[int],
) -> dict:
    _section("WER  (YouTube Commons ASR)")

    from io import BytesIO

    from datasets import load_dataset
    from pytubefix import YouTube
    from pytubefix.exceptions import VideoUnavailable

    normalizer = _load_normalizer()
    model = WhisperModel(
        model_size, device=device, device_index=device_index, compute_type=compute_type
    )
    pipeline = BatchedInferencePipeline(model)
    dataset = load_dataset(
        "mobiuslabsgmbh/youtube-commons-asr-eval", streaming=True
    )

    def _download(row):
        buf = BytesIO()
        try:
            yt = YouTube(row["link"])
            vid = (
                yt.streams.filter(only_audio=True, mime_type="audio/mp4")
                .order_by("bitrate")
                .desc()
                .last()
            )
            vid.stream_to_buffer(buf)
            buf.seek(0)
            row["audio"] = decode_audio(buf)
        except VideoUnavailable:
            row["audio"] = []
        return row

    dataset = dataset.map(_download)
    transcriptions, references = [], []

    for i, row in tqdm(enumerate(dataset["test"]), desc="YouTube Commons"):
        if row["audio"] is None or row["audio"].get("array") is None:
            continue
        result, _ = pipeline.transcribe(
            row["audio"]["array"], batch_size=8, without_timestamps=True
        )
        transcriptions.append("".join(s.text for s in result))
        references.append(row["text"][0])
        if audio_numb and i >= audio_numb - 1:
            break

    transcriptions = [normalizer(t) for t in transcriptions]
    references = [normalizer(r) for r in references]
    wer_pct = 100 * compute_wer(hypothesis=transcriptions, reference=references)

    print(f"  WER : {wer_pct:.3f}%  ({len(transcriptions)} samples)")
    return {"wer_pct": round(wer_pct, 3), "num_samples": len(transcriptions)}


# ── Suite: Throughput ─────────────────────────────────────────────────────────

def suite_throughput(
    model_size: str,
    compute_type: str,
    device: str,
    gpu_indices: list,
    batch_sizes: list,
    worker_counts: list,
    concurrency_levels: list,
) -> dict:
    _section(
        f"THROUGHPUT  (RTF · multi-GPU · batch sweep · worker sweep · concurrency)"
    )
    return run_throughput_benchmark(
        model_size=model_size,
        compute_type=compute_type,
        device=device,
        gpu_indices=gpu_indices,
        batch_sizes=batch_sizes,
        worker_counts=worker_counts,
        concurrency_levels=concurrency_levels,
    )


# ── Suite: Regression ─────────────────────────────────────────────────────────

def suite_regression(
    model_size: str,
    compute_type: str,
    device: str,
    device_index,
    do_save: bool,
    extra_audio: list,
) -> dict:
    _section("OUTPUT REGRESSION  (benchmark.m4a + optional extras)")

    audio_files = [BENCHMARK_M4A] + [os.path.abspath(p) for p in extra_audio]
    seen: set = set()
    audio_files = [
        p for p in audio_files
        if os.path.exists(p) and not (p in seen or seen.add(p))
    ]

    if do_save:
        save_reference(model_size, compute_type, device, device_index, audio_files)
        return {
            "action": "saved",
            "files": [os.path.basename(p) for p in audio_files],
        }

    return compare_against_reference(model_size, compute_type, device, device_index)


# ── Result persistence ────────────────────────────────────────────────────────

def _save_results(payload: dict, override_path: Optional[str] = None) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = override_path or os.path.join(RESULTS_DIR, f"benchmark_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {path}")
    return path


def _write_artemis_metrics(payload: dict) -> None:
    """Flatten key metrics into artemis_results.json at the repo root."""
    metrics: dict = {}
    suites = payload.get("suites", {})

    speed = suites.get("speed", {})
    if "min_s" in speed:
        metrics["speed_min_s"] = speed["min_s"]

    mem = suites.get("memory", {})
    gpu = mem.get("gpu", {})
    if "max_vram_mib" in gpu:
        metrics["memory_max_vram_mib"] = gpu["max_vram_mib"]
    if "max_power_w" in gpu:
        metrics["memory_max_power_w"] = gpu["max_power_w"]
    if "ram_increase_mib" in mem:
        metrics["memory_ram_increase_mib"] = mem["ram_increase_mib"]

    wer = suites.get("wer", {})
    if "wer_pct" in wer:
        metrics["wer_pct"] = wer["wer_pct"]

    yt = suites.get("yt_commons", {})
    if "wer_pct" in yt:
        metrics["yt_commons_wer_pct"] = yt["wer_pct"]

    tp = suites.get("throughput", {})
    baseline = tp.get("single_gpu_baseline", {})
    if "xrt" in baseline:
        metrics["throughput_baseline_xrt"] = baseline["xrt"]
    if "rtf" in baseline:
        metrics["throughput_baseline_rtf"] = baseline["rtf"]
    multigpu = tp.get("multi_gpu", {})
    if "xrt" in multigpu:
        metrics["throughput_multi_gpu_xrt"] = multigpu["xrt"]

    reg = suites.get("regression", {})
    if "drift_wer_pct" in reg:
        metrics["regression_drift_wer_pct"] = reg["drift_wer_pct"]
    if "drift_cer_pct" in reg:
        metrics["regression_drift_cer_pct"] = reg["drift_cer_pct"]

    if not metrics:
        return

    artemis_path = os.path.join(REPO_ROOT, "artemis_results.json")
    with open(artemis_path, "w", encoding="utf-8") as f:
        json.dump([metrics], f, indent=2)
    print(f"Artemis metrics saved → {artemis_path}")


def _print_summary(payload: dict):
    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    for suite_name, result in payload.get("suites", {}).items():
        print(f"  {suite_name}:")
        if not isinstance(result, dict):
            print(f"    {result}")
            continue
        for k, v in result.items():
            if isinstance(v, (int, float, str)):
                print(f"    {k}: {v}")
            elif isinstance(v, list) and v and not isinstance(v[0], dict):
                print(f"    {k}: {v}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified faster-whisper benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--suite",
        nargs="+",
        choices=["speed", "memory", "wer", "yt_commons", "throughput", "regression", "all"],
        default=["all"],
        metavar="SUITE",
        help=(
            "Suites to run: speed memory wer yt_commons throughput regression all. "
            "Default: all"
        ),
    )

    # ── model config ──
    parser.add_argument("--model", default="large-v3", help="Model size (default: large-v3)")
    parser.add_argument(
        "--compute-type", default="float16",
        help="Compute type: float16 int8 float32 (default: float16)"
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--device-index", type=int, nargs="+", default=[0],
        help="GPU device index(es) for speed/memory/wer/regression suites (default: 0)"
    )

    # ── speed ──
    parser.add_argument(
        "--repeat", type=int, default=3,
        help="[speed] timeit repeat count (default: 3)"
    )
    parser.add_argument(
        "--number", type=int, default=1,
        help="[speed] timeit number of runs per repeat (default: 1 for GA loops, use 10 for stable baselines)"
    )

    # ── memory ──
    parser.add_argument(
        "--no-gpu-memory", dest="gpu_memory", action="store_false", default=True,
        help="[memory] measure RAM instead of GPU VRAM"
    )
    parser.add_argument(
        "--interval", type=float, default=0.5,
        help="[memory] polling interval in seconds (default: 0.5)"
    )

    # ── wer / yt_commons ──
    parser.add_argument(
        "--audio-numb", type=int, default=None,
        help="[wer/yt_commons] max number of samples to evaluate (default: all)"
    )

    # ── throughput ──
    parser.add_argument(
        "--gpu-indices", type=int, nargs="+", default=[0, 1, 2, 3],
        help="[throughput] GPU indices to use (default: 0 1 2 3)"
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32],
        help="[throughput] batch sizes to sweep (default: 1 4 8 16 32)"
    )
    parser.add_argument(
        "--worker-counts", type=int, nargs="+", default=[1, 2, 4],
        help="[throughput] num_workers values to sweep (default: 1 2 4)"
    )
    parser.add_argument(
        "--concurrency-levels", type=int, nargs="+", default=[1, 2, 4],
        help="[throughput] concurrent request counts to test (default: 1 2 4)"
    )

    # ── regression ──
    parser.add_argument(
        "--regression-save", action="store_true",
        help="[regression] save reference outputs (run BEFORE optimisation)"
    )
    parser.add_argument(
        "--regression-audio", nargs="+", default=[],
        help="[regression] extra audio files beyond benchmark.m4a"
    )

    # ── output ──
    parser.add_argument(
        "--output", default=None,
        help="Override output JSON path (default: benchmark/results/benchmark_<ts>.json)"
    )

    args = parser.parse_args()

    suites = set(args.suite)
    if "all" in suites:
        suites = {"speed", "memory", "wer", "yt_commons", "throughput", "regression"}

    device_index = (
        args.device_index[0] if len(args.device_index) == 1 else args.device_index
    )

    payload = {
        "run_at": datetime.now().isoformat(),
        "config": {
            "model": args.model,
            "compute_type": args.compute_type,
            "device": args.device,
            "device_index": str(device_index),
        },
        "suites": {},
    }

    if "speed" in suites:
        payload["suites"]["speed"] = suite_speed(
            args.model, args.compute_type, args.device, device_index, args.repeat, args.number
        )

    if "memory" in suites:
        payload["suites"]["memory"] = suite_memory(
            args.model, args.compute_type, args.device, device_index,
            measure_gpu=args.gpu_memory,
            interval=args.interval,
        )

    if "wer" in suites:
        payload["suites"]["wer"] = suite_wer(
            args.model, args.compute_type, args.device, device_index, args.audio_numb
        )

    if "yt_commons" in suites:
        payload["suites"]["yt_commons"] = suite_yt_commons(
            args.model, args.compute_type, args.device, device_index, args.audio_numb
        )

    if "throughput" in suites:
        payload["suites"]["throughput"] = suite_throughput(
            args.model, args.compute_type, args.device,
            args.gpu_indices, args.batch_sizes, args.worker_counts,
            args.concurrency_levels,
        )

    if "regression" in suites:
        payload["suites"]["regression"] = suite_regression(
            args.model, args.compute_type, args.device, device_index,
            do_save=args.regression_save,
            extra_audio=args.regression_audio,
        )

    _save_results(payload, override_path=args.output)
    _write_artemis_metrics(payload)
    _print_summary(payload)


if __name__ == "__main__":
    main()
