"""
Beast3 optimised transcription — single-GPU and multi-GPU pipelines.

Hardware: 4× RTX 3090 24GB, Intel Xeon Gold 6230, 251 GB RAM
Best config: int8_float16, batch_size=32, beam_size=1, BatchedInferencePipeline

Benchmarked throughput:
  1 GPU  → ~107× real-time  (~7.5s for 13 min audio)
  3 GPUs → ~300× real-time  (~2.5s for 13 min audio, theoretical)

Usage:
    # Single GPU
    python transcribe_beast3.py audio.m4a --gpu 1

    # Multi-GPU (splits audio across GPUs 1, 2, 3)
    python transcribe_beast3.py audio.m4a --multi-gpu

    # Save transcript
    python transcribe_beast3.py audio.m4a --multi-gpu --output transcript.txt
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

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
        return {idx: f.result() for f, idx in [(f, futures[f]) for f in futures]}


# ---------------------------------------------------------------------------
# Audio splitting at VAD boundaries
# ---------------------------------------------------------------------------

def _split_at_silence(audio: np.ndarray, n: int) -> List[tuple]:
    """
    Split audio into n chunks, cutting at the silence point closest to each
    equal-time boundary. Returns list of (chunk_array, time_offset_seconds).
    """
    from faster_whisper.vad import get_speech_timestamps

    total = len(audio)
    if n == 1:
        return [(audio, 0.0)]

    timestamps = get_speech_timestamps(audio, sampling_rate=SAMPLING_RATE)

    # Build split points: for each of the n-1 interior boundaries, find the
    # silence midpoint closest to the ideal equal-time split.
    split_samples = [0]
    for i in range(1, n):
        target = i * (total // n)
        best = target
        best_dist = float("inf")
        for j in range(len(timestamps) - 1):
            mid = (timestamps[j]["end"] + timestamps[j + 1]["start"]) // 2
            dist = abs(mid - target)
            if dist < best_dist:
                best_dist = dist
                best = mid
        split_samples.append(best)
    split_samples.append(total)

    return [
        (audio[split_samples[i] : split_samples[i + 1]], split_samples[i] / SAMPLING_RATE)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Core transcription helpers
# ---------------------------------------------------------------------------

def _transcribe_chunk(pipeline, audio: np.ndarray, time_offset: float, language: str):
    segs, info = pipeline.transcribe(
        audio,
        language=language,
        batch_size=BATCH_SIZE,
        beam_size=BEAM_SIZE,
    )
    segments = [
        {"start": seg.start + time_offset, "end": seg.end + time_offset, "text": seg.text}
        for seg in segs
    ]
    return segments, info


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transcribe_single(audio_path: str, gpu_index: int = 1, language: str = "fr") -> str:
    """Single-GPU path — loads model, transcribes, returns transcript."""
    print(f"  loading model on GPU {gpu_index}...", file=sys.stderr, flush=True)
    pipeline = _load_pipeline(gpu_index)

    t0 = time.perf_counter()
    segs, info = pipeline.transcribe(
        audio_path, language=language, batch_size=BATCH_SIZE, beam_size=BEAM_SIZE
    )
    text = "".join(s.text for s in segs)
    elapsed = time.perf_counter() - t0

    print(
        f"  GPU {gpu_index}: {info.duration:.1f}s audio → {elapsed:.2f}s "
        f"({info.duration / elapsed:.1f}× real-time)",
        file=sys.stderr,
    )
    return text


def transcribe_multi(
    audio_path: str,
    gpu_indices: List[int] = None,
    language: str = "fr",
) -> str:
    """
    Multi-GPU path: loads one model per GPU in parallel, splits audio at VAD
    silence boundaries, transcribes each chunk concurrently, merges in order.
    """
    if gpu_indices is None:
        gpu_indices = [1, 2, 3]

    from faster_whisper.audio import decode_audio

    n = len(gpu_indices)
    t_start = time.perf_counter()

    # Load all GPU pipelines and decode audio simultaneously.
    print(f"  loading {n} pipelines on GPUs {gpu_indices} + decoding audio...", file=sys.stderr, flush=True)
    with ThreadPoolExecutor(max_workers=n + 1) as pool:
        pipeline_futures = {pool.submit(_load_pipeline, idx): idx for idx in gpu_indices}
        audio_future = pool.submit(decode_audio, audio_path)
        audio = audio_future.result()
        pipelines = {futures[f]: f.result() for f in pipeline_futures for futures in [pipeline_futures]}

    duration = len(audio) / SAMPLING_RATE
    print(f"  splitting {duration:.1f}s audio across {n} GPUs...", file=sys.stderr, flush=True)
    chunks = _split_at_silence(audio, n)

    # Transcribe all chunks in parallel — CTranslate2 releases the GIL for CUDA ops.
    print(f"  transcribing in parallel...", file=sys.stderr, flush=True)

    def _run(gpu_idx, chunk_audio, offset):
        return _transcribe_chunk(pipelines[gpu_idx], chunk_audio, offset, language)

    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [
            pool.submit(_run, gpu_indices[i], chunk, offset)
            for i, (chunk, offset) in enumerate(chunks)
        ]
        results = [f.result() for f in futures]

    elapsed = time.perf_counter() - t_start
    print(
        f"  {n} GPUs: {duration:.1f}s audio → {elapsed:.2f}s "
        f"({duration / elapsed:.1f}× real-time)",
        file=sys.stderr,
    )

    # Merge in chunk order (each chunk is internally sorted by time).
    all_segs = []
    for segs, _ in results:
        all_segs.extend(segs)
    return "".join(s["text"] for s in all_segs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio on beast3")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--gpu", type=int, default=1, help="GPU index for single-GPU mode (default: 1)")
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Use GPUs 1, 2, 3 in parallel (fastest for long audio)",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="GPU indices to use in multi-GPU mode (default: 1 2 3)",
    )
    parser.add_argument("--language", default="fr", help="Language code (default: fr)")
    parser.add_argument("--output", default=None, help="Write transcript to file instead of stdout")
    args = parser.parse_args()

    t0 = time.perf_counter()
    if args.multi_gpu:
        text = transcribe_multi(args.audio, gpu_indices=args.gpus, language=args.language)
    else:
        text = transcribe_single(args.audio, gpu_index=args.gpu, language=args.language)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  saved → {args.output}", file=sys.stderr)
    else:
        print(text)
