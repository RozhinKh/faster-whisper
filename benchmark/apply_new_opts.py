"""
Apply new optimizations to Beast3:
  1. vad.py      — eliminate np.pad copy; single-allocation __call__
  2. transcribe.py — pre-allocate feature result buffer; single-batch executor skip

Run from the repo root:
    python benchmark/apply_new_opts.py
"""
import re, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── vad.py ──────────────────────────────────────────────────────────────────

vad_path = ROOT / "faster_whisper" / "vad.py"
vad_src = vad_path.read_text(encoding="utf-8")

# 1a. Remove np.pad in get_speech_timestamps
old_pad = (
    "    padded_audio = np.pad(\n"
    "        audio, (0, window_size_samples - audio.shape[0] % window_size_samples)\n"
    "    )\n"
    "    speech_probs = model(padded_audio)"
)
new_pad = "    speech_probs = model(audio)"
if old_pad in vad_src:
    vad_src = vad_src.replace(old_pad, new_pad)
    print("vad.py: removed np.pad in get_speech_timestamps")
else:
    print("vad.py: np.pad already removed (skipping)")

# 1b. Replace SileroVADModel.__call__ body
old_call = '''\
    def __call__(
        self, audio: np.ndarray, num_samples: int = 512, context_size_samples: int = 64
    ):
        assert audio.ndim == 1, "Input should be a 1D array"
        assert (
            audio.shape[0] % num_samples == 0
        ), "Input size should be a multiple of num_samples"

        h = np.zeros((1, 1, 128), dtype="float32")
        c = np.zeros((1, 1, 128), dtype="float32")
        batched_audio = audio.reshape(-1, num_samples)
        num_segments = batched_audio.shape[0]

        # Pre-allocate the full output buffer (context + audio) in one shot.
        # This avoids the roll + concatenate sequence which creates three
        # short-lived intermediate arrays on every call.
        buf = np.empty(
            (num_segments, context_size_samples + num_samples), dtype=np.float32
        )
        # Segment 0 gets zero context; all others inherit the tail of the previous segment.
        buf[0, :context_size_samples] = 0.0
        if num_segments > 1:
            buf[1:, :context_size_samples] = batched_audio[:-1, -context_size_samples:]
        buf[:, context_size_samples:] = batched_audio
        batched_audio = buf

        output, h, c = self.session.run(
            None,
            {"input": batched_audio, "h": h, "c": c},
        )

        return output'''

new_call = '''\
    def __call__(
        self, audio: np.ndarray, num_samples: int = 512, context_size_samples: int = 64
    ):
        assert audio.ndim == 1, "Input should be a 1D array"

        n = len(audio)
        # Preserve original padding semantics: always add (num_samples - n % num_samples)
        # zeros at the end, which appends a full extra window when audio is already aligned.
        pad_n = num_samples - n % num_samples
        num_segments = (n + pad_n) // num_samples
        full_segs = n // num_samples  # segments fully covered by audio (no padding needed)

        h = np.zeros((1, 1, 128), dtype="float32")
        c = np.zeros((1, 1, 128), dtype="float32")

        # Single allocation — eliminates the separate np.pad copy the caller previously
        # performed.  We use np.empty and zero only the parts that must be zero (segment 0
        # context + last-segment tail padding) to avoid a 90 MB memset for long audio.
        buf = np.empty((num_segments, context_size_samples + num_samples), dtype=np.float32)
        # Segment 0: zero context; audio filled below (or stays uninitialized then overwritten).
        buf[0, :context_size_samples] = 0.0

        remainder = n % num_samples
        if full_segs > 0:
            # View over the aligned portion of audio (no copy).
            src = audio[:full_segs * num_samples].reshape(full_segs, num_samples)
            # Copy audio into the audio portion of each full segment.
            buf[:full_segs, context_size_samples:] = src
            # Context for segments 1..num_segments-1: last context_size_samples of previous audio.
            buf[1:num_segments, :context_size_samples] = src[:, -context_size_samples:]
        if remainder:
            buf[full_segs, context_size_samples:context_size_samples + remainder] = audio[full_segs * num_samples:]
            # Zero-pad the tail of the last partial segment.
            buf[full_segs, context_size_samples + remainder:] = 0.0
        else:
            # Audio is aligned; the extra window is all zeros.
            buf[full_segs, context_size_samples:] = 0.0

        output, h, c = self.session.run(
            None,
            {"input": buf, "h": h, "c": c},
        )

        return output'''

if old_call in vad_src:
    vad_src = vad_src.replace(old_call, new_call)
    print("vad.py: rewrote __call__ (single-allocation, no np.pad)")
else:
    print("vad.py: __call__ already updated (skipping)")

vad_path.write_text(vad_src, encoding="utf-8")

# ── transcribe.py ────────────────────────────────────────────────────────────

tr_path = ROOT / "faster_whisper" / "transcribe.py"
tr_src = tr_path.read_text(encoding="utf-8")

# 2a. Pre-allocate feature result buffer in _extract_and_cache
old_feats = (
    "                batch = audio_chunks[start : start + batch_size]\n"
    "                feats = [self.model.feature_extractor(chunk)[..., :-1] for chunk in batch]\n"
    "                result = np.stack([pad_or_trim(f) for f in feats])"
)
new_feats = (
    "                batch = audio_chunks[start : start + batch_size]\n"
    "                result = np.zeros((len(batch), _n_mels, 3000), dtype=np.float32)\n"
    "                for j, chunk in enumerate(batch):\n"
    "                    f = self.model.feature_extractor(chunk)\n"
    "                    # f.shape = (n_mels, n_frames+1); exclude the last frame to match\n"
    "                    # the original [..,-1] slice, then copy up to 3000 frames.\n"
    "                    n = min(f.shape[-1] - 1, 3000)\n"
    "                    result[j, :, :n] = f[:, :n]"
)
if old_feats in tr_src:
    tr_src = tr_src.replace(old_feats, new_feats)
    print("transcribe.py: pre-allocated feature result buffer")
else:
    print("transcribe.py: feature buffer already updated (skipping)")

# 2b. Add _n_mels outside the closure if not present
old_audio_chunks = (
    "            audio_chunks = features_or_chunks\n"
    "\n"
    "            def _extract_and_cache(start):"
)
new_audio_chunks = (
    "            audio_chunks = features_or_chunks\n"
    "            _n_mels = self.model.feature_extractor.mel_filters.shape[0]\n"
    "\n"
    "            def _extract_and_cache(start):"
)
if old_audio_chunks in tr_src:
    tr_src = tr_src.replace(old_audio_chunks, new_audio_chunks)
    print("transcribe.py: added _n_mels outside closure")
elif "_n_mels = self.model.feature_extractor.mel_filters.shape[0]" in tr_src:
    print("transcribe.py: _n_mels already outside closure (skipping)")
else:
    print("WARNING: could not find anchor for _n_mels insertion")

# 2c. Single-batch executor skip
old_executor = (
    "            else:\n"
    "                with ThreadPoolExecutor(max_workers=1) as executor:\n"
    "                    # Submit all extraction jobs up front.  The single worker processes\n"
    "                    # them sequentially, but each job begins as soon as the GPU starts\n"
    "                    # the previous batch, so extraction and inference overlap.\n"
    "                    # Cache hits return instantly without blocking the executor thread.\n"
    "                    futures = [\n"
    "                        executor.submit(_extract_and_cache, i) for i in batch_starts\n"
    "                    ]\n"
    "\n"
    "                    for i, future in zip(batch_starts, futures):\n"
    "                        features = future.result()\n"
)
new_executor = (
    "            else:\n"
    "                # For multi-batch audio use a background thread so feature extraction\n"
    "                # for batch N+1 overlaps with GPU inference on batch N.  For single-batch\n"
    "                # audio the executor adds thread-creation overhead with no pipeline benefit.\n"
    "                if len(batch_starts) > 1:\n"
    "                    _pool = ThreadPoolExecutor(max_workers=1)\n"
    "                    _futures = [_pool.submit(_extract_and_cache, i) for i in batch_starts]\n"
    "                    _get_features = lambda fut: fut.result()\n"
    "                else:\n"
    "                    _pool = None\n"
    "                    _futures = [_extract_and_cache(batch_starts[0])]\n"
    "                    _get_features = lambda feat: feat\n"
    "\n"
    "                try:\n"
    "                    for i, future in zip(batch_starts, _futures):\n"
    "                        features = _get_features(future)\n"
)
if old_executor in tr_src:
    tr_src = tr_src.replace(old_executor, new_executor)
    print("transcribe.py: applied single-batch executor skip")
    # Also need to close the try block — find pbar.update(1) at end of the loop and add finally
    old_close = (
    "                            pbar.update(1)\n"
    "\n"
    "        pbar.close()"
    )
    new_close = (
    "                            pbar.update(1)\n"
    "                finally:\n"
    "                    if _pool is not None:\n"
    "                        _pool.shutdown(wait=True)\n"
    "\n"
    "        pbar.close()"
    )
    if old_close in tr_src:
        tr_src = tr_src.replace(old_close, new_close)
        print("transcribe.py: added finally block for pool shutdown")
    else:
        print("WARNING: could not add finally block for pool shutdown")
elif "single-batch" in tr_src or "_pool = None" in tr_src:
    print("transcribe.py: single-batch skip already applied (skipping)")
else:
    print("WARNING: executor pattern not recognised — skipping single-batch skip")

tr_path.write_text(tr_src, encoding="utf-8")
print("\nDone. Run the cold-pass benchmark to measure improvement.")
