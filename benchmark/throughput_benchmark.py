"""
Throughput benchmark for faster-whisper.

Tests (in order):
  1. Single-GPU baseline        – device_index=0, batch_size=1
  2. Multi-GPU                  – all supplied GPU indices, num_workers=N
  3. Batch-size sweep           – single GPU, batch_size in [1,4,8,16,32]
  4. num_workers sweep          – single GPU, batch_size=8, num_workers in [1,2,4]
  5. Concurrent-requests sweep  – single GPU, N parallel transcription threads

Each configuration reports:
  - transcription_time_s : wall-clock time for one transcription
  - RTF                  : transcription_time / audio_duration (< 1 = faster than real-time)
  - xRT                  : audio_duration / transcription_time (e.g. 15 = 15× real-time)
  - gpu_memory_mib       : VRAM used at the end of the run (per device)

Usage:
    python benchmark/throughput_benchmark.py
    python benchmark/throughput_benchmark.py --gpu-indices 0 1 2 3 --batch-sizes 1 4 8 16 32
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import py3nvml.py3nvml as nvml

from faster_whisper import BatchedInferencePipeline, WhisperModel, decode_audio

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_AUDIO = os.path.join(BENCHMARK_DIR, "benchmark.m4a")


# ── helpers ────────────────────────────────────────────────────────────────────

def _gpu_memory_snapshot(gpu_indices: list) -> dict:
    snapshot = {}
    try:
        nvml.nvmlInit()
        for idx in gpu_indices:
            handle = nvml.nvmlDeviceGetHandleByIndex(idx)
            info = nvml.nvmlDeviceGetMemoryInfo(handle)
            snapshot[str(idx)] = {
                "name": nvml.nvmlDeviceGetName(handle),
                "used_mib": info.used >> 20,
                "total_mib": info.total >> 20,
                "used_pct": round((info.used / info.total) * 100, 1),
            }
        nvml.nvmlShutdown()
    except Exception:
        pass
    return snapshot


def _load_audio():
    audio = decode_audio(BENCHMARK_AUDIO)
    duration_s = len(audio) / 16000.0
    return audio, duration_s


def _timed_run(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def _make_result(label: str, elapsed: float, duration_s: float, **extra) -> dict:
    rtf = elapsed / duration_s
    xrt = duration_s / elapsed
    r = {
        "label": label,
        "audio_duration_s": round(duration_s, 2),
        "transcription_time_s": round(elapsed, 3),
        "rtf": round(rtf, 4),
        "xrt": round(xrt, 2),
    }
    r.update(extra)
    print(
        f"  {label:55s}  "
        f"time={elapsed:.2f}s  RTF={rtf:.4f}  {xrt:.1f}× real-time"
    )
    return r


# ── individual benchmark functions ────────────────────────────────────────────

def bench_single_gpu(
    audio,
    duration_s: float,
    model_size: str,
    compute_type: str,
    device: str,
    device_index: int,
) -> dict:
    model = WhisperModel(
        model_size, device=device, device_index=device_index, compute_type=compute_type
    )
    # warmup
    list(model.transcribe(audio)[0])

    elapsed = _timed_run(lambda: list(model.transcribe(audio)[0]))
    return _make_result(
        f"single GPU (device={device_index}, batch=1)",
        elapsed, duration_s,
        device_index=device_index, batch_size=1, num_workers=1,
    )


def bench_multi_gpu(
    audio,
    duration_s: float,
    model_size: str,
    compute_type: str,
    device: str,
    gpu_indices: list,
) -> dict:
    n = len(gpu_indices)
    model = WhisperModel(
        model_size,
        device=device,
        device_index=gpu_indices,
        compute_type=compute_type,
        num_workers=n,
    )
    list(model.transcribe(audio)[0])

    elapsed = _timed_run(lambda: list(model.transcribe(audio)[0]))
    return _make_result(
        f"multi-GPU {gpu_indices}  (num_workers={n})",
        elapsed, duration_s,
        device_index=gpu_indices, batch_size=1, num_workers=n,
    )


def bench_batch_sweep(
    audio,
    duration_s: float,
    model_size: str,
    compute_type: str,
    device: str,
    device_index: int,
    batch_sizes: list,
) -> list:
    model = WhisperModel(
        model_size, device=device, device_index=device_index, compute_type=compute_type
    )
    pipeline = BatchedInferencePipeline(model)

    # warmup at first batch size
    list(pipeline.transcribe(audio, batch_size=batch_sizes[0])[0])

    results = []
    for bs in batch_sizes:
        elapsed = _timed_run(
            lambda bs=bs: list(pipeline.transcribe(audio, batch_size=bs)[0])
        )
        results.append(
            _make_result(
                f"batch_size={bs:2d}  (device={device_index})",
                elapsed, duration_s,
                device_index=device_index, batch_size=bs, num_workers=1,
            )
        )
    return results


def bench_worker_sweep(
    audio,
    duration_s: float,
    model_size: str,
    compute_type: str,
    device: str,
    device_index: int,
    worker_counts: list,
    batch_size: int = 8,
) -> list:
    results = []
    for nw in worker_counts:
        model = WhisperModel(
            model_size,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            num_workers=nw,
        )
        pipeline = BatchedInferencePipeline(model)
        # warmup
        list(pipeline.transcribe(audio, batch_size=batch_size)[0])

        elapsed = _timed_run(
            lambda: list(pipeline.transcribe(audio, batch_size=batch_size)[0])
        )
        results.append(
            _make_result(
                f"num_workers={nw}  batch={batch_size}  (device={device_index})",
                elapsed, duration_s,
                device_index=device_index, batch_size=batch_size, num_workers=nw,
            )
        )
    return results


def bench_concurrent_requests(
    audio,
    duration_s: float,
    model_size: str,
    compute_type: str,
    device: str,
    device_index: int,
    concurrency_levels: list,
    batch_size: int = 8,
) -> list:
    """
    Simulates N simultaneous transcription requests on a single model to test
    how throughput scales with num_workers under concurrent load.
    """
    results = []
    for n_req in concurrency_levels:
        model = WhisperModel(
            model_size,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            num_workers=n_req,
        )
        pipeline = BatchedInferencePipeline(model)

        def _one_request():
            list(pipeline.transcribe(audio, batch_size=batch_size)[0])

        # warmup
        _one_request()

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_req) as pool:
            futs = [pool.submit(_one_request) for _ in range(n_req)]
            for f in as_completed(futs):
                f.result()
        wall = time.perf_counter() - t0

        total_audio = duration_s * n_req
        throughput_xrt = total_audio / wall
        print(
            f"  concurrent={n_req:2d} requests  "
            f"wall={wall:.2f}s  "
            f"total_audio={total_audio:.1f}s  "
            f"throughput={throughput_xrt:.1f}× real-time"
        )
        results.append(
            {
                "label": f"concurrent={n_req} requests",
                "n_concurrent_requests": n_req,
                "wall_time_s": round(wall, 3),
                "total_audio_s": round(total_audio, 2),
                "throughput_xrt": round(throughput_xrt, 2),
                "device_index": device_index,
                "batch_size": batch_size,
                "num_workers": n_req,
            }
        )
    return results


# ── orchestrator ──────────────────────────────────────────────────────────────

def run_throughput_benchmark(
    model_size: str,
    compute_type: str,
    device: str,
    gpu_indices: list,
    batch_sizes: list,
    worker_counts: list,
    concurrency_levels: list,
) -> dict:
    audio, duration_s = _load_audio()
    results = {}

    print(f"\n--- 1. Single-GPU Baseline (device={gpu_indices[0]}) ---")
    results["single_gpu"] = bench_single_gpu(
        audio, duration_s, model_size, compute_type, device, gpu_indices[0]
    )

    if len(gpu_indices) > 1:
        print(f"\n--- 2. Multi-GPU  {gpu_indices} ---")
        results["multi_gpu"] = bench_multi_gpu(
            audio, duration_s, model_size, compute_type, device, gpu_indices
        )

    print(f"\n--- 3. Batch-Size Sweep (device={gpu_indices[0]}) ---")
    results["batch_sweep"] = bench_batch_sweep(
        audio, duration_s, model_size, compute_type, device, gpu_indices[0], batch_sizes
    )

    print(f"\n--- 4. num_workers Sweep (device={gpu_indices[0]}, batch=8) ---")
    results["worker_sweep"] = bench_worker_sweep(
        audio, duration_s, model_size, compute_type, device, gpu_indices[0], worker_counts
    )

    print(f"\n--- 5. Concurrent-Requests Sweep (device={gpu_indices[0]}) ---")
    results["concurrent_sweep"] = bench_concurrent_requests(
        audio, duration_s, model_size, compute_type, device,
        gpu_indices[0], concurrency_levels
    )

    if device == "cuda":
        print("\n--- GPU Memory Snapshot ---")
        results["gpu_memory_snapshot"] = _gpu_memory_snapshot(gpu_indices)
        for idx, info in results["gpu_memory_snapshot"].items():
            print(
                f"  GPU {idx} ({info['name']}): "
                f"{info['used_mib']} / {info['total_mib']} MiB ({info['used_pct']}%)"
            )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Throughput and RTF benchmark")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--gpu-indices", type=int, nargs="+", default=[0, 1, 2, 3],
        help="GPU device indices to test (default: 0 1 2 3)"
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32],
        help="Batch sizes to sweep (default: 1 4 8 16 32)"
    )
    parser.add_argument(
        "--worker-counts", type=int, nargs="+", default=[1, 2, 4],
        help="num_workers values to sweep (default: 1 2 4)"
    )
    parser.add_argument(
        "--concurrency-levels", type=int, nargs="+", default=[1, 2, 4],
        help="Number of concurrent requests to test (default: 1 2 4)"
    )
    args = parser.parse_args()

    out = run_throughput_benchmark(
        model_size=args.model,
        compute_type=args.compute_type,
        device=args.device,
        gpu_indices=args.gpu_indices,
        batch_sizes=args.batch_sizes,
        worker_counts=args.worker_counts,
        concurrency_levels=args.concurrency_levels,
    )
    print("\n" + json.dumps(out, indent=2))
