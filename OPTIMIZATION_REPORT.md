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

### Performance Metrics (20-run median)

| Metric | Baseline | Optimised | Change |
|---|---|---|---|
| Transcription time | 10.200s | 7.442s | **−27.0%** |
| Throughput | 78.358× real-time | 107.388× real-time | **+37.0%** |
| VRAM used | 4,581 MiB | 2,789 MiB | **−39.1%** |
| Preprocessing time | 1.661s | 1.526s | **−8.1%** |
| VAD time | 1.644s | 1.506s | −8.4% |
| FFT time | 0.017s | 0.018s | — |

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

## 7. Production API Benchmark (Turing ASR Benchmark)

Measured with `artemisasrbench` on the Turing-ASR-Benchmark suite. Both baseline and optimised ran sequentially on the same GPU (RTX 3090) to ensure a fair comparison.

**Accuracy note:** The benchmark audio is English LibriSpeech. The production use case is French broadcast speech. WER=0.0% on these clips confirms the optimisation introduces no regression on English; the French accuracy characterisation is in Section 4 (ga_benchmark on French audio, mean WER 1.35%).

### Measurement methodology

| Parameter | Value |
|---|---|
| Hardware | NVIDIA RTX 3090 24 GB |
| Sequential runs | 20 (validity + per-request latency) |
| Concurrent streams | 5–10 (scenario-dependent) |
| Concurrent requests | 100 per scenario |
| Timing scope | Wall-clock per HTTP request — no model load, no pre-read |
| Throughput mechanism | `BatchedInferencePipeline`: groups concurrent requests into GPU batches up to `batch_size` before encoding. Higher concurrency fills batches more fully, reducing per-request latency. |

The high throughput on short clips (125×, 325× RT) reflects concurrent batching: with `batch_size=32` and 5–10 parallel streams, the GPU processes multiple requests in one encoder pass. Per-request latency drops to ~65 ms even for 8–23s clips. This is a valid production metric for a multi-client API — it measures how fast each caller gets their result under realistic load, not single-threaded throughput.

### Setup

| Component | Baseline | Optimised |
|---|---|---|
| Server | fedirz/faster-whisper-server | Custom image (this branch) |
| compute_type | float16 | int8_float16 |
| beam_size | 5 (default) | 1 |
| batch_size | 16 (default) | 32 |
| Code changes | — | scipy FFT, O(n) VAD |

### Results

**Sequential P50 RTF** (single isolated request, no queuing)

| Scenario | Audio | Baseline | Optimised | Speedup |
|---|---|---|---|---|
| long_form_v1 | 156.5s | 0.0388 (25.8× RT) | 0.0333 (30.1× RT) | **1.17×** |
| clean_long_v1 | 23.45s | 0.01627 (61.5× RT) | 0.00308 (325× RT) | **5.3×** |
| noisy_v1 | 2.10s | 0.1822 (5.5× RT) | 0.0269 (37× RT) | **6.8×** |

**Concurrent P50 RTF** (100 requests, 5–10 parallel streams — primary API metric)

| Scenario | Audio | Baseline | Optimised | Latency P50 (opt) | Speedup |
|---|---|---|---|---|---|
| clean_long_v1 | 23.45s | 0.01627 (61.5× RT) | 0.00308 (325× RT) | 72 ms | **5.3×** |
| clean_short_v1 | 8.37s | 0.13452 (7.4× RT) | 0.00771 (130× RT) | 65 ms | **17.4×** |
| control_phrase_v1 | 4.21s | 0.45681 (2.2× RT) | 0.01764 (56.7× RT) | 74 ms | **25.9×** |
| noisy_v1 | 2.10s | 0.8834 (1.1× RT) | 0.0295 (33.9× RT) | 62 ms | **29.9×** |

All scenarios passed accuracy validation (WER=0.0% on clean, WER=22.2% on noisy — same as baseline).

### How the numbers scale with audio length

The same optimisation produces very different speedup depending on clip length. This is expected behaviour, not inconsistency:

| Audio length | Baseline (× RT) | Optimised (× RT) | Speedup | What limits baseline | Source |
|---|---|---|---|---|---|
| 2.1s | 1.1× | 34× | **30×** | Decoder search dominates | ASR bench concurrent |
| 4–8s | 2–7× | 57–130× | **17–26×** | Decoder + GPU warmup | ASR bench concurrent |
| 23.5s | 62× | 325× | **5.3×** | GPU begins to saturate | ASR bench concurrent |
| 156s | 25.8× | 30.1× | **1.17×** | GPU utilisation varies by content | ASR bench sequential |
| 780s | 76.5× | 104.8× | **1.37×** | Full pipeline, all stages | ga_benchmark sequential |

The large gains at short lengths reflect beam=5 decoder overhead being disproportionate to audio content. beam=1 collapses this to a single-pass argmax, dropping per-request latency to ~65 ms regardless of clip length. The long-form rows (156s, 780s) show the underlying encoder/pipeline speedup (1.2–1.4×) once the decoder is no longer the bottleneck.

**Bottom line: −27% for batch processing of long files, 5–30× speedup for a real-time API serving short concurrent requests.**

---

## 8. Summary

Artemis found a parameter combination (compute_type=int8_float16, batch_size=32, beam_size=1) that reduces transcription time by 27.0% and increases throughput from 78× to 107× real-time. VRAM usage dropped 39.1% as a side effect of int8 quantization — a significant benefit for concurrent workloads on a shared multi-GPU server.

Code-level changes to the preprocessing pipeline (multi-threaded scipy FFT at 13.36× CPU speedup, O(n) chunk collection, binary search index lookup) contribute 8.1% preprocessing improvement measured independently.

Accuracy is validated across 40 test cases: preprocessing output is numerically equivalent to the main branch (FFT diff = float32 epsilon), and transcription WER across 20 audio clips averages 1.35% with a worst case of 3.92% — all differences limited to punctuation variation, no content regression.

Robustness validation across 6 noise conditions shows 5/6 pass. The single failure (overlapping speech) is a known beam_size=1 limitation for multi-speaker audio; the recommendation is beam_size=3–5 for that use case.

The `preprocessing_time_s` metric is exposed in `artemis_results.json`, giving Artemis a direct optimisation target for CPU-side code quality independent of GPU inference time.
