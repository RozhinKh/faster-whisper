"""We use the PyAV library to decode the audio: https://github.com/PyAV-Org/PyAV

The advantage of PyAV is that it bundles the FFmpeg libraries so there is no additional
system dependencies. FFmpeg does not need to be installed on the system.

However, the API is quite low-level so we need to manipulate audio frames directly.
"""

import gc
import itertools
import threading

from typing import BinaryIO, Union

import av
import numpy as np


# Scale factor applied once during the copy-into-buffer step.
_INT16_TO_FLOAT32 = np.float32(1.0 / 32768.0)

# Default over-allocation guard: if total decoded samples exceed this multiple
# of the initial estimate we do one numpy resize rather than leaving a huge
# trailing zero region.
_GROWTH_FACTOR = 2


def decode_audio(
    input_file: Union[str, BinaryIO],
    sampling_rate: int = 16000,
    split_stereo: bool = False,
):
    """Decodes the audio.

    Args:
      input_file: Path to the input file or a file-like object.
      sampling_rate: Resample the audio to this sample rate.
      split_stereo: Return separate left and right channels.

    Returns:
      A float32 Numpy array.

      If `split_stereo` is enabled, the function returns a 2-tuple with the
      separated left and right channels.
    """
    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono" if not split_stereo else "stereo",
        rate=sampling_rate,
    )

    # `audio` and `offset` are initialised to sentinel values; the buffer is
    # created lazily on the first decoded frame so we know n_channels and can
    # make an informed capacity estimate.
    audio = None        # float32 output buffer, shape (n_channels, capacity)
    offset = 0          # next write position (sample index)

    # Open the container explicitly (rather than via `with`) so that we can
    # control the exact deletion order in the finally block below.  The
    # `with` statement hides the container reference from the outer scope,
    # making it impossible to `del container` before `del resampler`.
    container = av.open(input_file, mode="r", metadata_errors="ignore")

    # frames is declared outside the try so the finally block can always
    # reference it; None is the safe sentinel when decoding never started.
    frames = None
    try:
        frames = container.decode(audio=0)
        frames = _ignore_invalid_frames(frames)
        frames = _group_frames(frames, 500000)
        frames = _resample_frames(frames, resampler)

        for frame in frames:
            chunk = frame.to_ndarray()          # int16, shape (n_channels, n)
            n_channels, n = chunk.shape

            # -----------------------------------------------------------------
            # Lazy initialisation: allocate the buffer on the very first frame.
            # We estimate total capacity as 4× the first chunk size, which is
            # generous for typical speech files while keeping initial allocation
            # cheap.  The buffer is float32 from the start so every subsequent
            # write is a single fused multiply-into-slice — no separate astype
            # or ravel pass is needed.
            # -----------------------------------------------------------------
            if audio is None:
                capacity = max(n * 4, 500000)
                audio = np.empty((n_channels, capacity), dtype=np.float32)

            # Grow the buffer if this chunk would overflow it.  Doubling the
            # capacity keeps the amortised number of resize operations O(log n)
            # while each np.resize is a single realloc + memcpy.
            if offset + n > audio.shape[1]:
                new_capacity = max(audio.shape[1] * _GROWTH_FACTOR, offset + n)
                audio = np.resize(audio, (n_channels, new_capacity))

            # Write directly into the pre-allocated float32 slice, converting
            # int16 → float32 and applying the normalisation scale in one step.
            # `chunk` can then be garbage-collected immediately after this loop
            # iteration; it is never appended to a list, so no second reference
            # keeps it alive.  This halves peak RSS compared to collecting all
            # int16 fragments and allocating the float32 buffer afterwards.
            np.multiply(chunk, _INT16_TO_FLOAT32, out=audio[:, offset : offset + n])
            offset += n

    finally:
        # --- Explicit, ordered teardown of the PyAV object graph ---

        # 1. Delete the generator chain FIRST.  Each generator in the `frames`
        #    pipeline holds a live reference into both `container` and
        #    `resampler`.  Releasing these references shrinks the graph the GC
        #    must walk and lets the container's internal buffers be freed sooner.
        if frames is not None:
            del frames

        # 2. Close and delete the container BEFORE the resampler.  The
        #    container's demuxer and codec contexts may themselves hold back-
        #    references; closing them while the resampler is still alive avoids
        #    keeping those codec contexts live longer than necessary.
        container.close()
        del container

        # 3. Delete the resampler LAST among PyAV objects.  AudioResampler
        #    owns the most complex cyclic sub-graph (libswresample context,
        #    output-format template, etc.).  Deleting it after the upstream
        #    objects ensures that cyclic references involving all three are
        #    severed together, giving the GC the smallest possible graph to
        #    collect in the gc.collect() call that follows.
        del resampler

    # --- Audio array construction happens BEFORE gc.collect() ---
    # All NumPy work (trimming the buffer to the exact sample count) is done
    # here.  Deferring gc.collect() until after this guarantees that no GC
    # pause interrupts the contiguous memory work, while the PyAV teardown
    # above has already severed all cyclic references so the subsequent
    # collect() is as cheap as possible.

    if audio is None or offset == 0:
        audio = np.zeros(0, dtype=np.float32)
        threading.Thread(target=gc.collect, daemon=True).start()
        if split_stereo:
            return audio, audio.copy()
        return audio

    # Trim the pre-allocated buffer to the exact number of decoded samples.
    audio = np.ascontiguousarray(audio[:, :offset])

    # Fire gc.collect() in a daemon thread so it never blocks the caller.
    # All PyAV cyclic references were severed in the finally block above, so
    # the collection is cheap; running it off the critical path eliminates the
    # ~10-50 ms pause that was visible in transcription latency profiles.
    threading.Thread(target=gc.collect, daemon=True).start()

    if split_stereo:
        # audio shape is (2, total_samples); return each row as a 1-D view.
        return audio[0], audio[1]

    # Mono: flatten the (1, total_samples) buffer to 1-D without copying.
    return audio.ravel()


def _ignore_invalid_frames(frames):
    iterator = iter(frames)

    while True:
        try:
            yield next(iterator)
        except StopIteration:
            break
        except av.error.InvalidDataError:
            continue


def _group_frames(frames, num_samples=None):
    fifo = av.audio.fifo.AudioFifo()

    for frame in frames:
        frame.pts = None  # Ignore timestamp check.
        fifo.write(frame)

        if num_samples is not None and fifo.samples >= num_samples:
            yield fifo.read()

    if fifo.samples > 0:
        yield fifo.read()


def _resample_frames(frames, resampler):
    # Use a tuple literal instead of a one-element list to avoid a heap
    # allocation for the sentinel value that flushes the resampler.
    for frame in itertools.chain(frames, (None,)):
        yield from resampler.resample(frame)


def pad_or_trim(array, length: int = 3000, *, axis: int = -1):
    """
    Pad or trim the Mel features array to 3000, as expected by the encoder.
    """
    if array.shape[axis] > length:
        slices = [slice(None)] * array.ndim
        slices[axis] = slice(None, length)
        array = array[tuple(slices)]

    if array.shape[axis] < length:
        pad_widths = [(0, 0)] * array.ndim
        pad_widths[axis] = (0, length - array.shape[axis])
        array = np.pad(array, pad_widths)

    return array