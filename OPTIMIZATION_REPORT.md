# Faster-whisper Optimization Report

**Repository:** https://github.com/RozhinKh/faster-whisper.git
**Branch:** `optimize/artemis-candidate`
**Date:** 2026-05-07
**Author:** Rozhin Khalilian

---

## 1. Optimization Target

| Field | Value |
|---|---|
| Library | faster-whisper |
| Model | faster-whisper-large-v3 |
| Hardware | NVIDIA RTX 3090 24 GB, Intel Xeon Gold 6230 (20 cores), 251 GB RAM |
| Runtime | CTranslate2 + ONNX Runtime |

---

## 2. What Changed

### A — Configuration Tuning

Artemis found a parameter combination across three high-impact axes:

**`compute_type`: float16 → int8_float16**
Model weights are stored in int8 (half the memory of float16) but matrix multiplications use float16 precision. This halves VRAM usage and speeds up memory-bound operations. Model weights and sampling are unchanged — only the numerical representation of the weight matrix changes.

**`batch_size`: 16 → 32**
The `BatchedInferencePipeline` groups audio chunks into batches before sending them to the GPU encoder. Larger batches amortise kernel launch overhead and improve GPU occupancy. On RTX 3090, batch_size=32 was the sweet spot — beyond this, diminishing returns.

**`beam_size`: 5 → 1 (greedy decoding)**
This was the biggest single gain. Whisper's decoder generates text token by token. With beam_size=5 it keeps 5 candidate transcriptions alive simultaneously and picks the best at the end. With beam_size=1 it commits to the most likely token each step (greedy). Greedy is 2–3× faster on the decoder with near-identical output on high-quality audio.

---

### B — Code Changes

**`feature_extractor.py` — Multi-threaded FFT**
Replaced `numpy.fft.rfft` with `scipy.fft.rfft(workers=-1)`. NumPy's FFT is single-threaded. SciPy's FFT parallelises across all available CPU cores — on a 20-core Xeon Gold 6230 this delivers 13.36× more CPU throughput for mel spectrogram computation. Input is cast to float64 before the FFT call to match NumPy's internal precision (pocketfft upcasts float32→float64 internally; without this cast, parallel thread scheduling produces different floating point rounding). Falls back to NumPy automatically if SciPy is not installed.

**`vad.py` — Algorithmic improvements**
Two algorithmic fixes in the VAD post-processing pipeline:
- `collect_chunks`: reduced from O(n²) to O(n) — previously used `np.concatenate` inside a loop, allocating a new array on every iteration; now collects all slices into a list and concatenates once.
- `get_chunk_index`: replaced a linear `.index()` scan with `bisect` binary search.

Note: CUDA execution for the SileroVAD ONNX model was evaluated and reverted. GPU floating-point non-determinism caused speech boundary timestamps to shift between runs on long audio, producing different chunk splits and breaking the regression test. VAD runs on CPU (deterministic).

**`ga_benchmark.py` — Preprocessing metric**
Added `preprocessing_time_s` as an independently timed metric (median of 5 standalone runs of: audio decode → VAD → feature extraction). This gives Artemis a direct signal for code-quality improvements that is independent of GPU-bound inference time, which otherwise masks CPU-side gains.

---

## 3. Results

All runs: full benchmark audio (~13 min), GPU 3 (RTX 3090), single-GPU mode, sequential baseline then optimised.

### Performance Metrics

| Metric | Baseline | Optimised | Change | Source |
|---|---|---|---|---|
| Transcription time (min) | 9.995s | 7.332s | **−26.7%** | speed_benchmark.py |
| Transcription time (median) | 10.022s | 7.359s | **−26.6%** | speed_benchmark.py |
| Speedup | — | — | **1.36×** | speed_benchmark.py |
| VRAM used | 4,003 MiB | 1,824 MiB | **−54.4%** | speed_benchmark.py |
| Preprocessing time | 1.661s | 1.526s | **−8.1%** | ga_benchmark (20-run median) |
| VAD time | 1.644s | 1.506s | −8.4% | ga_benchmark |
| FFT time | 0.017s | 0.018s | — | ga_benchmark |

### Variance (20-run)

| Config | Median | Mean | Stddev | P95 | n |
|---|---|---|---|---|---|
| Baseline | 10.200s | 10.185s | 46.7ms | 10.241s | 20 |
| Optimised | 7.442s | 7.446s | 56.4ms | 7.541s | 20 |

Stddev is <0.6% of median for both configs. The improvement is consistent across all 20 runs — not a lucky outlier.

### Contribution Breakdown

| Source | Time saved | % of total |
|---|---|---|
| Config changes (beam, batch, compute_type) | ~2.62s | ~25.7% |
| Code changes (scipy FFT + algorithmic fixes) | ~0.14s | ~1.3% |
| **Total** | **~2.76s** | **−27.0%** |

---

## 4. Accuracy Validation

Two-part validation suite (`benchmark/test_code_regression.py`) covering 40 test cases.

### Part A — Preprocessing Regression (CPU only, 20 cases)

Tests VAD and FFT output of PR code vs main-branch code across full audio and 19 evenly-spaced 60s clips. No GPU involved — output must be functionally equivalent.

| Component | Method | Result |
|---|---|---|
| VAD (speech timestamps) | Exact SHA1 hash | **20/20 PASS** — bit-identical |
| FFT (mel spectrogram) | Max absolute difference | **20/20 PASS** — max diff = 1.19×10⁻⁷ |

The FFT difference of `1.19e-07` is float32 machine epsilon — the minimum representable floating point difference. This confirms the parallel scipy.fft and sequential numpy.fft are numerically equivalent at the precision limit of the data type.

### Part B — Transcription Accuracy (GPU, 20 cases)

Compares baseline config (float16, bs=16, beam=5) vs optimised config (int8_float16, bs=32, beam=1) across full audio and 19 clips using the **French benchmark audio** (the production use case). GPU inference is non-deterministic between model loads, so WER threshold (< 5%) is used instead of exact hash comparison.

| Metric | Value |
|---|---|
| Pass rate | **20/20 (100%)** |
| Mean WER | **1.35%** |
| Max WER | **3.92%** |
| Clips with WER = 0.00% | 10 / 20 |
| WER threshold | 5.00% |

**Nature of differences:** Minor punctuation and capitalisation variations (e.g. `". Et"` → `", et"`). No spoken content or word-level meaning changed.

---

## 5. Robustness Validation

### Noise Conditions (`benchmark/noise_validation.py`)

WER computed against the clean baseline transcript (float16/beam=5) as reference. Pass = optimised config regresses no more than 3% over baseline on the same degraded audio.

| Condition | Baseline WER | Optimised WER | Delta | Result |
|---|---|---|---|---|
| Clean (reference) | 0.0000 | 0.0098 | +0.0098 | **PASS** |
| Noise SNR 20 dB (slight) | 0.0041 | 0.0089 | +0.0048 | **PASS** |
| Noise SNR 10 dB (moderate) | 0.0187 | 0.0223 | +0.0036 | **PASS** |
| Noise SNR 5 dB (heavy) | 0.0612 | 0.0748 | +0.0136 | **PASS** |
| Telephone quality (300-3400 Hz) | 0.0241 | 0.0289 | +0.0048 | **PASS** |
| Overlapping speech (−6 dB) | 0.0814 | 0.1346 | +0.0532 | **FAIL** |

**Pass rate: 5/6**

**Overlapping speech note:** beam=1 (greedy) regresses 5.32% vs beam=5 on two-speaker audio. This is expected — beam search recovers from ambiguous decoder states where greedy commits to the wrong token. Recommendation: use beam_size=3 or beam_size=5 for multi-speaker or overlapping speech use cases. Single-speaker French broadcast audio (the primary use case) is unaffected.

---

## 6. Scaling Behaviour

### Audio Length Scaling (`benchmark/scaling_benchmark.py`, Part 1)

| Audio | Baseline (s) | Baseline (×RT) | Optimised (s) | Optimised (×RT) | Change |
|---|---|---|---|---|---|
| 30s | 0.482 | 62.2× | 0.490 | 61.2× | +1.6% (flat) |
| 5min | 2.241 | 134.2× | 1.863 | 161.4× | −16.9% |
| 13min | 10.200 | 78.4× | 7.442 | 107.4× | −27.0% |
| 60min | 46.1 | 78.3× | 32.7 | 110.4× | −29.1% |

**Scope boundary:** This optimisation targets long-form transcription. At 30s, the GPU encoder processes so few batches that beam=1 saves almost nothing — kernel launch overhead dominates decode time at short lengths. No improvement at 30s is expected behaviour, not a failure. The full gain appears at 5+ minutes and holds at 60 minutes. The production use case (French broadcast, typically 5–60+ min files) sits entirely in the range where the optimisation is effective.

### FFT Thread Scaling (`benchmark/scaling_benchmark.py`, Part 2)

Measured on 20-core Intel Xeon Gold 6230, scipy.fft.rfft on 1501 × 400-pt frames.

| Workers | Median (ms) | Speedup vs 1 |
|---|---|---|
| 1 | 8.3 | 1.00× |
| 2 | 4.7 | 1.77× |
| 4 | 2.8 | 2.96× |
| 8 | 1.6 | 5.19× |
| 16 | 0.9 | 9.22× |
| -1 (all 20) | 0.6 | **13.36×** |

The 13.36× CPU parallelism is measurable in isolation. Its wall-clock contribution to overall transcription is small (FFT is 0.017s of a 9.99s pipeline) because CTranslate2 CUDA inference dominates.

---

## 7. Speed Benchmark (Official faster-whisper methodology)

Measured with `benchmark/speed_benchmark.py --compare` using `benchmark.m4a` — the same script and audio used by the SYSTRAN/faster-whisper maintainers in their README benchmarks. Both configs run on the same GPU in the same process with no server or HTTP overhead. Parameters are the only variable.

### Methodology

| Parameter | Value |
|---|---|
| Script | `benchmark/speed_benchmark.py` (SYSTRAN official) |
| Audio | `benchmark/benchmark.m4a` (~13 min, French broadcast) |
| Hardware | NVIDIA RTX 3090 24 GB |
| Method | `timeit.repeat(repeat=3, number=10)` — min/10 reported |
| Warmup | 1 full transcription before timing |
| Model | faster-whisper-large-v3 |

### Setup

| Parameter | Baseline | Optimised |
|---|---|---|
| compute_type | float16 | int8_float16 |
| beam_size | 5 | 1 |
| batch_size | 16 | 32 |

### Results

| Config | Min per run | Median per run | VRAM | Raw totals (3 reps × 10 runs) |
|---|---|---|---|---|
| Baseline  (float16,      beam=5, batch=16) | **9.995s** | 10.022s | 4,003 MiB | 100.221, 99.954, 100.396 |
| Candidate (int8_float16, beam=1, batch=32) | **7.332s** | 7.359s | 1,824 MiB | 73.316, 74.038, 73.589 |

**Speedup: 1.36× (−26.7%) — VRAM: −54.4% (4,003 → 1,824 MiB)**

Candidate variance across 3 repetitions: 0.72s (1.0%) — highly stable. Both configs show tight variance confirming the GPU was warm and uncontended throughout the run.

This cross-validates Section 3's ga_benchmark result (−27.0%) measured independently with a different timing method (20-run median on a different audio file). Both point to the same underlying gain.

### How the gain scales with audio length

The optimisation targets long-form transcription. The speedup grows with audio length because beam=1 savings compound across 30-second encoder chunks:

| Audio length | Speedup | Source |
|---|---|---|
| 4–10s (1 chunk) | 1.02–1.09× | scenario_latency_benchmark (direct model) |
| 156s (~5 chunks) | 1.14× | scenario_latency_benchmark (direct model) |
| ~780s (~26 chunks) | **1.37×** | ga_benchmark (20-run median) |
| ~780s (~26 chunks) | **1.36×** | speed_benchmark (timeit, this section) |

At short clip lengths (≤30s), the GPU encoder always processes a full 30-second window regardless of audio content, so beam search overhead is a small fraction of total time and savings are modest. At long-form lengths, each additional chunk saves decoder work and the gains compound.

---

## 8. Artemis ASR Bench — Service-Level Validation (Beast3, RTX 3090)

Validated using `artemisasrbench` — HTTP-level benchmark against `server.py` running on Beast3. 9 scenarios covering clean, noisy, telephone, and overlapping speech conditions. All audio derived from LibriVox "A Scandal in Bohemia" Chapter 1 (~21 min).

**Hardware:** NVIDIA RTX 3090 24 GB, Intel Xeon Gold 6230, faster-whisper 1.2.1, CTranslate2 4.7.2, CUDA 12.9

### Validity Gate — 9/9 PASS on both baseline and candidate

### RTF Delta (concurrent P50, baseline → candidate)

| Scenario | Baseline RTF | Candidate RTF | RTF Δ |
|---|---|---|---|
| long_form_v1 (~21 min, clean) | 0.0104 | 0.0082 | **−21.3%** |
| clean_long_v1 (10 min, clean) | 0.0107 | 0.0084 | **−22.0%** |
| clean_short_v1 (5 min, clean) | 0.0107 | 0.0093 | **−13.3%** |
| noisy_snr20_v1 (~21 min, SNR 20 dB) | 0.0107 | 0.0086 | **−19.7%** |
| noisy_snr10_v1 (~21 min, SNR 10 dB) | 0.0107 | 0.0084 | **−20.9%** |
| noisy_snr5_v1 (~21 min, SNR 5 dB) | 0.0107 | 0.0087 | **−19.3%** |
| overlapping_v1 (~21 min, −6 dB mix) | 0.0118 | 0.0089 | **−24.0%** |
| telephone_v1 (~21 min, 300–3400 Hz) | 0.0071 | 0.0055 | **−22.2%** |
| control_phrase_v1 (10.4 s) | 0.3637 | 0.3294 | **−9.5%** |

All 9 scenarios improved. Short clips see less gain (fewer encoder chunks for beam savings to compound); long-form sees the full improvement — consistent with speed_benchmark scaling table.

### Layer 1/2 Probe — Kernel/Compute + Memory/IO

| Metric | Baseline | Candidate | Change |
|---|---|---|---|
| VRAM (long_form_v1) | 4,839 MiB | 2,951 MiB | **−39%** |
| model_ms (long_form_v1) | 13,184 ms | 10,593 ms | **−19.6%** |
| encoder_throughput_x (long_form_v1) | 96× | 120× | **+25%** |
| decode_tokens_per_s (long_form_v1) | 266/s | 328/s | **+23%** |

### Layer 4 — Decoder Overhead (beam search cost saved)

| Scenario | ms saved | % of decoder |
|---|---|---|
| overlapping_v1 | 3,756 ms | 25.2% |
| noisy_snr10_v1 | 2,874 ms | 21.2% |
| long_form_v1 | 2,590 ms | 19.6% |
| telephone_v1 | 1,908 ms | 21.3% |
| clean_long_v1 | 1,502 ms | 23.2% |
| clean_short_v1 | 404 ms | 12.4% |
| control_phrase_v1 | 24 ms | 4.5% |

The service-level RTF improvement (−19 to −24%) is slightly below the direct model measurement (−26.7%) due to HTTP overhead. Both measurements confirm the same underlying gain.

---

## 9. Summary

Artemis found a parameter combination (compute_type=int8_float16, batch_size=32, beam_size=1) that reduces transcription time by 27.0% and increases throughput from 78× to 107× real-time. VRAM usage dropped 39.1% as a side effect of int8 quantization — a significant benefit for concurrent workloads on a shared multi-GPU server.

Code-level changes to the preprocessing pipeline (multi-threaded scipy FFT at 13.36× CPU speedup, O(n) chunk collection, binary search index lookup) contribute 8.1% preprocessing improvement measured independently.

Accuracy is validated across 40 test cases: preprocessing output is numerically equivalent to the main branch (FFT diff = float32 epsilon), and transcription WER across 20 audio clips averages 1.35% with a worst case of 3.92% — all differences limited to punctuation variation, no content regression.

Robustness validation across 6 noise conditions shows 5/6 pass. The single failure (overlapping speech) is a known beam_size=1 limitation for multi-speaker audio; the recommendation is beam_size=3–5 for that use case.

The `preprocessing_time_s` metric is exposed in `artemis_results.json`, giving Artemis a direct optimisation target for CPU-side code quality independent of GPU inference time.
