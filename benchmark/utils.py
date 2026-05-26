import logging
import os
import time

from threading import Thread
from typing import Callable, Optional

from faster_whisper import BatchedInferencePipeline, WhisperModel
from faster_whisper.audio import decode_audio

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_AUDIO = os.path.join(BENCHMARK_DIR, "benchmark.m4a")

# Prefer a pre-downloaded local copy; fall back to HF download.
_LOCAL_MODEL = os.path.join(os.path.dirname(os.path.dirname(BENCHMARK_DIR)), "models", "faster-whisper-large-v3")
model_path = _LOCAL_MODEL if os.path.isdir(_LOCAL_MODEL) else "large-v3"
_model = None
_pipeline = None


def _get_model():
    global _model, _pipeline
    if _model is None:
        _model = WhisperModel(model_path, device="cuda", compute_type="float16")
        _pipeline = BatchedInferencePipeline(_model)
    return _model, _pipeline


def inference():
    model, _ = _get_model()
    segments, info = model.transcribe(BENCHMARK_AUDIO, language="fr")
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))


def batched_inference(batch_size: int = 8, beam_size: int = 5):
    _, pipeline = _get_model()
    segments, info = pipeline.transcribe(
        BENCHMARK_AUDIO,
        language="fr",
        batch_size=batch_size,
        beam_size=beam_size,
    )
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))


def make_inference_fn(
    compute_type: str = "float16",
    beam_size: int = 5,
    batched: bool = False,
    batch_size: int = 8,
    device: str = "cuda",
    device_index: int = 0,
    language: str = "fr",
) -> Callable[[], None]:
    """
    Factory that returns a no-arg callable suitable for timeit sweeps.

    Use this when you need to benchmark a specific (compute_type, beam_size,
    batch_size) combination without mutating the module-level model.

    Example — sweep compute_type:
        for ct in ("float16", "int8_float16"):
            fn = make_inference_fn(compute_type=ct)
            runtimes = timeit.repeat(fn, repeat=3, number=10)
    """
    m = WhisperModel(model_path, device=device, device_index=device_index, compute_type=compute_type)
    if batched:
        p = BatchedInferencePipeline(m)

        # Pre-decode audio once — each timeit call would otherwise re-decode
        # a 21-min file from disk, which is I/O unrelated to model inference.
        # A real server decodes bytes once per request; this matches that.
        sr = m.feature_extractor.sampling_rate
        t0 = time.perf_counter()
        _audio = decode_audio(BENCHMARK_AUDIO, sampling_rate=sr)
        print(f"  [pre-load] decode={( time.perf_counter()-t0)*1000:.0f}ms  "
              f"samples={len(_audio):,}")

        def _batched():
            segs, _ = p.transcribe(
                _audio,
                language=language,
                batch_size=batch_size,
                beam_size=beam_size,
            )
            for _ in segs:
                pass

        return _batched
    else:
        def _plain():
            segs, _ = m.transcribe(
                BENCHMARK_AUDIO, language=language, beam_size=beam_size
            )
            for _ in segs:
                pass

        return _plain


def get_logger(name: Optional[str] = None) -> logging.Logger:
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class MyThread(Thread):
    def __init__(self, func, params):
        super(MyThread, self).__init__()
        self.func = func
        self.params = params
        self.result = None

    def run(self):
        self.result = self.func(*self.params)

    def get_result(self):
        return self.result
