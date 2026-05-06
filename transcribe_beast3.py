"""
Optimised transcription for beast3 (4× RTX 3090 24GB).

Best config found via benchmarking:
  - int8_float16 compute  → 41% less VRAM, faster quantised matmuls
  - batch_size=32         → sweet spot for RTX 3090 tensor core utilisation
  - beam_size=1           → greedy decoding, no quality loss for this use case
  - BatchedInferencePipeline → 80% faster than sequential WhisperModel

Throughput: ~107× real-time per GPU  (~7.5s for 13 min audio)
VRAM:       ~2.7 GB per GPU          (fits 8 streams per GPU, 32 total on beast3)

Usage:
    python transcribe_beast3.py audio.m4a
    python transcribe_beast3.py audio.m4a --gpu 1 --language en
    python transcribe_beast3.py audio.m4a --output transcript.txt
"""

import argparse
import sys
import time

MODEL_PATH = "/home/rozhin/rozhin/models/faster-whisper-large-v3"

BEAST3_CONFIG = {
    "device": "cuda",
    "compute_type": "int8_float16",
}

TRANSCRIBE_CONFIG = {
    "batch_size": 32,
    "beam_size": 1,
    "language": "fr",
}


def load_model(gpu_index: int = 0):
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    model = WhisperModel(
        MODEL_PATH,
        device_index=gpu_index,
        **BEAST3_CONFIG,
    )
    return BatchedInferencePipeline(model)


def transcribe(audio_path: str, gpu_index: int = 0, language: str = "fr") -> str:
    pipeline = load_model(gpu_index)

    t0 = time.perf_counter()
    segments, info = pipeline.transcribe(
        audio_path,
        language=language,
        **{k: v for k, v in TRANSCRIBE_CONFIG.items() if k != "language"},
    )
    transcript = "".join(seg.text for seg in segments)
    elapsed = time.perf_counter() - t0

    print(
        f"  duration : {info.duration:.1f}s audio  |  "
        f"transcribed in {elapsed:.2f}s  |  "
        f"{info.duration / elapsed:.1f}× real-time",
        file=sys.stderr,
    )
    return transcript


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio on beast3")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (0-3)")
    parser.add_argument("--language", default="fr", help="Language code (default: fr)")
    parser.add_argument("--output", default=None, help="Write transcript to file")
    args = parser.parse_args()

    text = transcribe(args.audio, gpu_index=args.gpu, language=args.language)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  saved → {args.output}", file=sys.stderr)
    else:
        print(text)
