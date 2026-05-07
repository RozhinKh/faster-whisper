"""
Generate audio variants for robustness validation.

Produces from benchmark.m4a:
  - clean_30s.wav       : first 30s, clean
  - clean_5min.wav      : first 5 min, clean
  - noisy_snr20.wav     : full audio + Gaussian noise at SNR 20 dB (slight)
  - noisy_snr10.wav     : full audio + Gaussian noise at SNR 10 dB (moderate)
  - noisy_snr5.wav      : full audio + Gaussian noise at SNR  5 dB (heavy)
  - telephone.wav       : 300–3400 Hz bandpass + 8 kHz resample (phone quality)
  - overlapping.wav     : original + offset copy at −6 dB (overlapping speech)
  - extended_60min.wav  : benchmark.m4a tiled to ~60 min (scaling test)

Usage:
    python benchmark/generate_audio_variants.py
"""

import os
import sys

import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as sig

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_IN  = os.path.join(BENCHMARK_DIR, "benchmark.m4a")
OUT_DIR   = os.path.join(BENCHMARK_DIR, "audio_variants")
SR        = 16000


def save_wav(path, audio, sr=SR):
    pcm = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype(np.int16)
    wav.write(path, sr, pcm16)
    print(f"  saved {os.path.basename(path)}  ({len(audio)/sr:.1f}s)")


def add_noise(audio, snr_db):
    signal_power = np.mean(audio ** 2) + 1e-12
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise = np.random.default_rng(42).normal(0, np.sqrt(noise_power), len(audio))
    return (audio + noise).astype(np.float32)


def telephone_quality(audio, sr=SR):
    # Downsample to 8 kHz, bandpass 300–3400 Hz, upsample back
    audio_8k   = sig.resample_poly(audio, 1, 2)
    b, a       = sig.butter(4, [300 / 4000, 3400 / 4000], btype="band")
    filtered   = sig.filtfilt(b, a, audio_8k).astype(np.float32)
    audio_16k  = sig.resample_poly(filtered, 2, 1).astype(np.float32)
    return audio_16k[: len(audio)]


def add_overlap(audio, offset_s=3.0, gain_db=-6):
    gain    = 10 ** (gain_db / 20)
    offset  = int(offset_s * SR)
    result  = audio.copy()
    end     = min(len(audio), offset + len(audio))
    result[offset:end] += audio[: end - offset] * gain
    return np.clip(result, -1.0, 1.0).astype(np.float32)


def main():
    from faster_whisper.audio import decode_audio

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Loading {AUDIO_IN} ...")
    audio = decode_audio(AUDIO_IN)  # float32, 16 kHz
    total_s = len(audio) / SR
    print(f"  duration: {total_s:.1f}s\n")

    # Clean clips for scaling benchmark
    save_wav(os.path.join(OUT_DIR, "clean_30s.wav"),   audio[:SR * 30])
    save_wav(os.path.join(OUT_DIR, "clean_5min.wav"),  audio[:SR * 300])
    save_wav(os.path.join(OUT_DIR, "clean_full.wav"),  audio)

    # Noise variants (full audio)
    save_wav(os.path.join(OUT_DIR, "noisy_snr20.wav"), add_noise(audio, 20))
    save_wav(os.path.join(OUT_DIR, "noisy_snr10.wav"), add_noise(audio, 10))
    save_wav(os.path.join(OUT_DIR, "noisy_snr5.wav"),  add_noise(audio,  5))

    # Telephone quality
    print("  generating telephone variant (may take ~30s) ...")
    save_wav(os.path.join(OUT_DIR, "telephone.wav"),   telephone_quality(audio))

    # Overlapping speech
    save_wav(os.path.join(OUT_DIR, "overlapping.wav"), add_overlap(audio))

    # Extended audio for 60-min scaling test (tile benchmark ~5×)
    print("  tiling audio for 60-min scaling test ...")
    tiles  = int(np.ceil(3600 / total_s)) + 1
    long   = np.tile(audio, tiles)[: SR * 3600]
    save_wav(os.path.join(OUT_DIR, "extended_60min.wav"), long)

    print(f"\nAll variants written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
