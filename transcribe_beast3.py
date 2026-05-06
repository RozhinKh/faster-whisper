"""
Beast3 optimised transcription — single-GPU and multi-GPU pipelines.

Hardware: 4× RTX 3090 24GB, Intel Xeon Gold 6230, 251 GB RAM
Best config: int8_float16, batch_size=32, beam_size=1, BatchedInferencePipeline

IMPORTANT — model loading vs inference:
  Model loading takes ~6s per GPU (disk I/O + VRAM allocation).
  For one-off calls this dominates. For production (many files), load
  models once with Beast3Server and reuse — then per-file time is:
    1 GPU  → ~7.5s  (107× real-time)
    3 GPUs → ~2.5s  (300× real-time)

Usage:
    # One-off single GPU
    python transcribe_beast3.py audio.m4a --gpu 1

    # One-off multi-GPU
    python transcribe_beast3.py audio.m4a --multi-gpu

    # Benchmark (loads once, times inference only — shows true speedup)
    python transcribe_beast3.py audio.m4a --benchmark

    # Python API for production (load once, call many times)
    server = Beast3Server(gpu_indices=[1, 2, 3])
    server.warmup("any_audio.m4a")
    text = server.transcribe("file1.m4a")   # ~2.5s
    text = server.transcribe("file2.m4a")   # ~2.5s
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np

MODEL_PATH = "/home/rozhin/rozhin/models/faster-whisper-large-v3"
COMPUTE_TYPE = "int8_float16"
BATCH_SIZE = 32
BEAM_SIZE = 1
SAMPLING_RATE = 16000


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------

def _load_pipeline(gpu_index: int):
    from faster_whisper import BatchedInferencePipeline, WhisperModel
    model = WhisperModel(
        MODEL_PATH,
        device="cuda",
        device_index=gpu_index,
        compute_type=COMPUTE_TYPE,
    )
    return BatchedInferencePipeline(model)


def _load_pipelines_parallel(gpu_indices: List[int]) -> dict:
    with ThreadPoolExecutor(max_workers=len(gpu_indices)) as pool:
        futures = {pool.submit(_load_pipeline, idx): idx for idx in gpu_indices}
        return {futures[f]: f.result() for f in futures}


# ---------------------------------------------------------------------------
# Audio splitting at VAD silence boundaries
# ---------------------------------------------------------------------------

def _split_at_silence(audio: np.ndarray, n: int) -> List[tuple]:
    """
    Split audio into n chunks at silence boundaries.
    Returns list of (chunk_array, time_offset_seconds).
    """
    from faster_whisper.vad import get_speech_timestamps

    total = len(audio)
    if n == 1:
        return [(audio, 0.0)]

    timestamps = get_speech_timestamps(audio, sampling_rate=SAMPLING_RATE)

    split_samples = [0]
    for i in range(1, n):
        target = i * (total // n)
        best, best_dist = target, float("inf")
        for j in range(len(timestamps) - 1):
            mid = (timestamps[j]["end"] + timestamps[j + 1]["start"]) // 2
            dist = abs(mid - target)
            if dist < best_dist:
                best_dist, best = dist, mid
        split_samples.append(best)
    split_samples.append(total)

    return [
        (audio[split_samples[i] : split_samples[i + 1]], split_samples[i] / SAMPLING_RATE)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Core inference (no model loading)
# ---------------------------------------------------------------------------

def _transcribe_chunk(pipeline, audio: np.ndarray, time_offset: float, language: str):
    segs, info = pipeline.transcribe(
        audio, language=language, batch_size=BATCH_SIZE, beam_size=BEAM_SIZE,
    )
    return (
        [{"start": s.start + time_offset, "end": s.end + time_offset, "text": s.text} for s in segs],
        info,
    )


def _infer(pipelines: dict, audio: np.ndarray, language: str) -> tuple:
    """Run inference on pre-loaded pipelines. Returns (text, duration, elapsed)."""
    gpu_indices = list(pipelines.keys())
    n = len(gpu_indices)
    duration = len(audio) / SAMPLING_RATE

    chunks = _split_at_silence(audio, n)

    t0 = time.perf_counter()

    def _run(i):
        chunk_audio, offset = chunks[i]
        return _transcribe_chunk(pipelines[gpu_indices[i]], chunk_audio, offset, language)

    with ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(_run, range(n)))

    elapsed = time.perf_counter() - t0

    all_segs = []
    for segs, _ in results:
        all_segs.extend(segs)
    text = "".join(s["text"] for s in all_segs)
    return text, duration, elapsed


# ---------------------------------------------------------------------------
# Beast3Server — load once, transcribe many
# ---------------------------------------------------------------------------

class Beast3Server:
    """
    Pre-loads models on specified GPUs. Each call to transcribe() skips
    model loading and goes straight to inference (~2.5s for 3 GPUs).
    """

    def __init__(self, gpu_indices: List[int] = None):
        if gpu_indices is None:
            gpu_indices = [1, 2, 3]
        print(f"  loading models on GPUs {gpu_indices}...", file=sys.stderr, flush=True)
        t0 = time.perf_counter()
        self.pipelines = _load_pipelines_parallel(gpu_indices)
        self.gpu_indices = gpu_indices
        print(f"  models ready in {time.perf_counter() - t0:.1f}s", file=sys.stderr)

    def warmup(self, audio_path: str, language: str = "fr"):
        """Run one silent warmup pass to prime CUDA kernels."""
        from faster_whisper.audio import decode_audio
        audio = decode_audio(audio_path)
        _infer(self.pipelines, audio, language)
        print("  warmup done", file=sys.stderr, flush=True)

    def transcribe(self, audio_path: str, language: str = "fr") -> str:
        from faster_whisper.audio import decode_audio
        audio = decode_audio(audio_path)
        text, duration, elapsed = _infer(self.pipelines, audio, language)
        print(
            f"  {len(self.gpu_indices)} GPUs: {duration:.1f}s audio → {elapsed:.2f}s "
            f"({duration / elapsed:.1f}× real-time)",
            file=sys.stderr,
        )
        return text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio on beast3")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--gpu", type=int, default=1, help="GPU for single-GPU mode")
    parser.add_argument("--multi-gpu", action="store_true", help="Use GPUs 1, 2, 3 in parallel")
    parser.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--language", default="fr")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Load models once, warmup, then time inference only (shows true per-file speed)",
    )
    args = parser.parse_args()

    if args.benchmark:
        gpu_indices = args.gpus if args.multi_gpu else [args.gpu]
        server = Beast3Server(gpu_indices=gpu_indices)
        print("  warming up...", file=sys.stderr, flush=True)
        server.warmup(args.audio, args.language)
        print("  timing...", file=sys.stderr, flush=True)
        from faster_whisper.audio import decode_audio
        audio = decode_audio(args.audio)
        results = []
        for _ in range(3):
            text, duration, elapsed = _infer(server.pipelines, audio, args.language)
            results.append(elapsed)
            print(f"    {elapsed:.3f}s ({duration/elapsed:.1f}×)", file=sys.stderr)
        import statistics
        med = statistics.median(results)
        print(f"  median: {med:.3f}s  ({duration/med:.1f}× real-time)  GPUs={gpu_indices}", file=sys.stderr)

    elif args.multi_gpu:
        server = Beast3Server(gpu_indices=args.gpus)
        server.warmup(args.audio, args.language)
        text = server.transcribe(args.audio, args.language)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"  saved → {args.output}", file=sys.stderr)
        else:
            print(text)

    else:
        pipeline = _load_pipeline(args.gpu)
        from faster_whisper.audio import decode_audio
        audio = decode_audio(args.audio)
        # warmup
        _transcribe_chunk(pipeline, audio, 0.0, args.language)
        # timed
        t0 = time.perf_counter()
        segs, info = pipeline.transcribe(args.audio, language=args.language, batch_size=BATCH_SIZE, beam_size=BEAM_SIZE)
        text = "".join(s.text for s in segs)
        elapsed = time.perf_counter() - t0
        print(f"  GPU {args.gpu}: {info.duration:.1f}s audio → {elapsed:.2f}s ({info.duration/elapsed:.1f}×)", file=sys.stderr)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(text)
        else:
            print(text)
