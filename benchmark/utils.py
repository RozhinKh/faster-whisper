import logging
import os

from threading import Thread
from typing import Callable, Optional

from faster_whisper import BatchedInferencePipeline, WhisperModel

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_AUDIO = os.path.join(BENCHMARK_DIR, "benchmark.m4a")

model_path = "large-v3"
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

        def _batched():
            segs, _ = p.transcribe(
                BENCHMARK_AUDIO,
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
