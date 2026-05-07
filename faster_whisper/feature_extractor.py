import numpy as np

try:
    import scipy.fft as _scipy_fft
    _FFT_WORKERS = -1
except ImportError:
    _scipy_fft = None
    _FFT_WORKERS = None


class FeatureExtractor:
    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_samples = chunk_length * sampling_rate
        self.nb_max_frames = self.n_samples // hop_length
        self.time_per_frame = hop_length / sampling_rate
        self.sampling_rate = sampling_rate
        self.mel_filters = self.get_mel_filters(
            sampling_rate, n_fft, n_mels=feature_size
        ).astype("float32")
        self.hann_window = np.hanning(n_fft + 1)[:-1].astype("float32")

    @staticmethod
    def get_mel_filters(sr, n_fft, n_mels=128):
        # Initialize the weights
        n_mels = int(n_mels)

        # Center freqs of each FFT bin
        fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)

        # 'Center freqs' of mel bands - uniformly spaced between limits
        min_mel = 0.0
        max_mel = 45.245640471924965

        mels = np.linspace(min_mel, max_mel, n_mels + 2)

        # Fill in the linear scale
        f_min = 0.0
        f_sp = 200.0 / 3
        freqs = f_min + f_sp * mels

        # And now the nonlinear scale
        min_log_hz = 1000.0  # beginning of log region (Hz)
        min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
        logstep = np.log(6.4) / 27.0  # step size for log region

        # If we have vector data, vectorize
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))

        fdiff = np.diff(freqs)
        ramps = freqs.reshape(-1, 1) - fftfreqs.reshape(1, -1)

        lower = -ramps[:-2] / np.expand_dims(fdiff[:-1], axis=1)
        upper = ramps[2:] / np.expand_dims(fdiff[1:], axis=1)

        # Reuse `lower` in-place to avoid two extra heap allocations:
        #   np.minimum writes into lower (replaces the separate minimum result),
        #   np.maximum then clamps to zero in the same buffer.
        # Original: np.maximum(np.zeros_like(lower), np.minimum(lower, upper))
        # created zeros_like + minimum output + maximum output = 3 allocations.
        np.minimum(lower, upper, out=lower)
        np.maximum(lower, 0.0, out=lower)
        weights = lower

        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (freqs[2 : n_mels + 2] - freqs[:n_mels])
        weights *= np.expand_dims(enorm, axis=1)

        return weights

    @staticmethod
    def stft(
        input_array: np.ndarray,
        n_fft: int,
        hop_length: int = None,
        win_length: int = None,
        window: np.ndarray = None,
        center: bool = True,
        mode: str = "reflect",
        normalized: bool = False,
        onesided: bool = None,
        return_complex: bool = None,
    ):
        # Default initialization for hop_length and win_length
        hop_length = hop_length if hop_length is not None else n_fft // 4
        win_length = win_length if win_length is not None else n_fft
        input_is_complex = np.iscomplexobj(input_array)

        # Determine if the output should be complex
        return_complex = (
            return_complex
            if return_complex is not None
            else (input_is_complex or (window is not None and np.iscomplexobj(window)))
        )

        if not return_complex and return_complex is None:
            raise ValueError(
                "stft requires the return_complex parameter for real inputs."
            )

        # Input checks
        if not np.issubdtype(input_array.dtype, np.floating) and not input_is_complex:
            raise ValueError(
                "stft: expected an array of floating point or complex values,"
                f" got {input_array.dtype}"
            )

        if input_array.ndim > 2 or input_array.ndim < 1:
            raise ValueError(
                f"stft: expected a 1D or 2D array, but got {input_array.ndim}D array"
            )

        # Handle 1D input
        if input_array.ndim == 1:
            input_array = np.expand_dims(input_array, axis=0)
            input_array_1d = True
        else:
            input_array_1d = False

        # Center padding if required — single allocation, explicit dtype preserved
        if center:
            pad_amount = n_fft // 2
            input_array = np.pad(
                input_array, ((0, 0), (pad_amount, pad_amount)), mode=mode
            )

        batch, length = input_array.shape

        # Additional input checks
        if n_fft <= 0 or n_fft > length:
            raise ValueError(
                f"stft: expected 0 < n_fft <= {length}, but got n_fft={n_fft}"
            )

        if hop_length <= 0:
            raise ValueError(
                f"stft: expected hop_length > 0, but got hop_length={hop_length}"
            )

        if win_length <= 0 or win_length > n_fft:
            raise ValueError(
                f"stft: expected 0 < win_length <= n_fft, but got win_length={win_length}"
            )

        if window is not None:
            if window.ndim != 1 or window.shape[0] != win_length:
                raise ValueError(
                    f"stft: expected a 1D window array of size equal to win_length={win_length}, "
                    f"but got window with size {window.shape}"
                )

        # Handle padding of the window if necessary
        if win_length < n_fft:
            left = (n_fft - win_length) // 2
            window_ = np.zeros(n_fft, dtype=window.dtype)
            window_[left : left + win_length] = window
        else:
            window_ = window

        # Calculate the number of frames
        n_frames = 1 + (length - n_fft) // hop_length

        # Build a strided view over the (possibly padded) signal — no copy yet
        strided = np.lib.stride_tricks.as_strided(
            input_array,
            (batch, n_frames, n_fft),
            (
                input_array.strides[0],
                hop_length * input_array.strides[1],
                input_array.strides[1],
            ),
        )

        # Apply window into a pre-allocated contiguous buffer to avoid a silent
        # temporary: `strided * window_` would allocate batch×n_frames×n_fft
        # elements implicitly; np.multiply with out= reuses that one buffer.
        if window_ is not None:
            frames = np.empty((batch, n_frames, n_fft), dtype=input_array.dtype)
            np.multiply(strided, window_, out=frames)
        else:
            # np.fft requires a contiguous array; force a single copy here
            frames = np.ascontiguousarray(strided)

        # FFT and transpose
        complex_fft = input_is_complex
        onesided = onesided if onesided is not None else not complex_fft

        norm = "ortho" if normalized else None

        if complex_fft:
            if onesided:
                raise ValueError(
                    "Cannot have onesided output if window or input is complex"
                )
            if _scipy_fft is not None:
                # Cast to float64 to match numpy.fft internal precision (pocketfft
                # upcasts float32→float64 internally; scipy respects input dtype).
                # This preserves multi-threaded speedup while keeping bit-identical output.
                output = _scipy_fft.fft(frames.astype(np.float64), n=n_fft, axis=-1, norm=norm, workers=_FFT_WORKERS)
            else:
                output = np.fft.fft(frames, n=n_fft, axis=-1, norm=norm)
        else:
            if _scipy_fft is not None:
                output = _scipy_fft.rfft(frames.astype(np.float64), n=n_fft, axis=-1, norm=norm, workers=_FFT_WORKERS)
            else:
                output = np.fft.rfft(frames, n=n_fft, axis=-1, norm=norm)

        output = output.transpose((0, 2, 1))

        if input_array_1d:
            output = output.squeeze(0)

        return output if return_complex else np.real(output)

    def __call__(self, waveform: np.ndarray, padding=160, chunk_length=None):
        """
        Compute the log-Mel spectrogram of the provided audio.
        """

        if chunk_length is not None:
            self.n_samples = chunk_length * self.sampling_rate
            self.nb_max_frames = self.n_samples // self.hop_length

        # Combine dtype conversion + zero-padding into a single allocation.
        # Previously: astype (copy, N×4 B) then np.pad (copy, (N+pad)×4 B).
        # Now: one buffer of the final size; waveform is copied in once.
        padded_length = len(waveform) + (padding if padding else 0)
        if waveform.dtype == np.float32 and not padding:
            # No dtype change, no padding — use as-is (zero extra allocations)
            processed = waveform
        else:
            processed = np.zeros(padded_length, dtype=np.float32)
            if waveform.dtype == np.float32:
                processed[: len(waveform)] = waveform
            else:
                # casting='unsafe' avoids a hidden astype temporary
                np.copyto(processed[: len(waveform)], waveform, casting="unsafe")
            # tail is already zeroed by np.zeros — matches np.pad constant mode

        stft = self.stft(
            processed,
            self.n_fft,
            self.hop_length,
            window=self.hann_window,
            return_complex=True,
        )
        stft_trimmed = stft[..., :-1]  # view — no copy

        # Compute the power spectrum directly into a float32 buffer.
        #
        # Why not `stft_trimmed.real ** 2 + stft_trimmed.imag ** 2`?
        #   • `** 2` always allocates a new array (it is never in-place).
        #   • The implicit temporary for `imag ** 2` inside `+=` is a second
        #     silent allocation of the same shape.
        #   • If rfft returned complex128 (NumPy default when input is float32),
        #     both squares land in float64, then a third copy is made by astype.
        #
        # Instead:
        #   1. Pre-allocate `magnitudes` as float32 once.
        #   2. np.multiply with out= + casting='unsafe' squares the real part
        #      *and* casts to float32 in a single write pass — no intermediate.
        #   3. Reuse one scratch buffer for imag², then accumulate in-place.
        #      Total allocations: 2 × (n_bins × n_frames × 4 B) instead of
        #      up to 4 × (n_bins × n_frames × 8 B) for the float64 path.
        n_bins, n_frames = stft_trimmed.shape
        magnitudes = np.empty((n_bins, n_frames), dtype=np.float32)
        np.multiply(stft_trimmed.real, stft_trimmed.real, out=magnitudes, casting="unsafe")
        _imag_sq = np.empty((n_bins, n_frames), dtype=np.float32)
        np.multiply(stft_trimmed.imag, stft_trimmed.imag, out=_imag_sq, casting="unsafe")
        magnitudes += _imag_sq
        del _imag_sq  # release scratch before the matmul allocation below

        # Pre-allocate the output buffer and use np.dot's `out` parameter to
        # avoid the hidden allocation that `@` always incurs.  magnitudes is
        # already float32 (guaranteed above), so the matmul stays in float32
        # precision and BLAS dispatches to sgemm without an upcast copy.
        mel_spec = np.empty(
            (self.mel_filters.shape[0], magnitudes.shape[-1]), dtype=np.float32
        )
        np.dot(self.mel_filters, magnitudes, out=mel_spec)

        # All subsequent operations are in-place (no extra allocations).
        np.maximum(mel_spec, 1e-10, out=mel_spec)
        log_spec = np.log10(mel_spec, out=mel_spec)
        np.maximum(log_spec, log_spec.max() - 8.0, out=log_spec)
        log_spec += 4.0
        # Multiply by the reciprocal instead of dividing: the multiply ufunc
        # path is cheaper than the divide ufunc path for the same in-place op.
        log_spec *= 0.25

        return log_spec