"""Smoke-test: verify new SileroVADModel.__call__ padding logic is identical
to the old np.pad approach for a variety of audio lengths.

Run without GPU (uses a mock session) to verify the buffer construction only.
"""
import numpy as np

NUM_SAMPLES = 512
CONTEXT = 64


def old_make_buf(audio):
    """Reproduce original pre-ONNX buffer construction."""
    pad_n = NUM_SAMPLES - audio.shape[0] % NUM_SAMPLES
    padded = np.pad(audio, (0, pad_n))
    batched = padded.reshape(-1, NUM_SAMPLES)
    num_seg = batched.shape[0]
    buf = np.empty((num_seg, CONTEXT + NUM_SAMPLES), dtype=np.float32)
    buf[0, :CONTEXT] = 0.0
    if num_seg > 1:
        buf[1:, :CONTEXT] = batched[:-1, -CONTEXT:]
    buf[:, CONTEXT:] = batched
    return buf


def new_make_buf(audio):
    """New single-allocation approach."""
    n = len(audio)
    pad_n = NUM_SAMPLES - n % NUM_SAMPLES
    num_segments = (n + pad_n) // NUM_SAMPLES
    full_segs = n // NUM_SAMPLES

    buf = np.empty((num_segments, CONTEXT + NUM_SAMPLES), dtype=np.float32)
    buf[0, :CONTEXT] = 0.0

    remainder = n % NUM_SAMPLES
    if full_segs > 0:
        src = audio[:full_segs * NUM_SAMPLES].reshape(full_segs, NUM_SAMPLES)
        buf[:full_segs, CONTEXT:] = src
        buf[1:num_segments, :CONTEXT] = src[:, -CONTEXT:]
    if remainder:
        buf[full_segs, CONTEXT:CONTEXT + remainder] = audio[full_segs * NUM_SAMPLES:]
        buf[full_segs, CONTEXT + remainder:] = 0.0
    else:
        buf[full_segs, CONTEXT:] = 0.0

    return buf


def check(n, label):
    rng = np.random.default_rng(42)
    audio = rng.random(n, dtype=np.float32)
    b_old = old_make_buf(audio)
    b_new = new_make_buf(audio)
    assert b_old.shape == b_new.shape, f"{label}: shape mismatch {b_old.shape} vs {b_new.shape}"
    assert np.allclose(b_old, b_new, atol=1e-7), f"{label}: values differ"
    print(f"  PASS  n={n:>9,}  shape={b_old.shape}  ({label})")


if __name__ == "__main__":
    print("Testing new VAD buffer construction against original np.pad approach…")
    for n, label in [
        (0,        "empty"),
        (1,        "1 sample"),
        (100,      "< 1 window"),
        (511,      "1 window - 1"),
        (512,      "exactly 1 window"),
        (513,      "1 window + 1"),
        (1024,     "exactly 2 windows"),
        (4800000,  "5-min clean_short (aligned)"),
        (4800100,  "5-min clean_short (unaligned)"),
        (20160000, "21-min telephone (aligned)"),
        (20160001, "21-min telephone (unaligned)"),
    ]:
        check(n, label)
    print("\nAll checks passed.")
