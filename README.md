# faster-whisper — Optimized Inference Pipeline

[![CI](https://github.com/SYSTRAN/faster-whisper/workflows/CI/badge.svg)](https://github.com/SYSTRAN/faster-whisper/actions?query=workflow%3ACI)

This is an optimized fork of [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) with end-to-end inference pipeline improvements for `BatchedInferencePipeline`. No model weights, decoder parameters, or quantization settings were changed — config is held constant between baseline and optimized on each machine.

**Author:** Rozhin Khalilian — [Full Optimization Report →](OPTIMIZATION_REPORT.md)

---

## Results

### NVIDIA RTX 3090 · int8_float16 / beam=5 / batch=32

| Metric | Baseline | Optimized | Change |
|---|---|---|---|
| Transcription time (13 min audio) | 8.991 s | **7.049 s** | **−21.6%** |
| Throughput | 88.9× | **113.4×** | **+27.6%** |
| Speed benchmark (SYSTRAN, 30 runs) | 7.994 s | **6.206 s** | **−22.4%** |
| Artemis ASR — 5 min clean (cold-pass) | 3,038 ms | **2,561 ms** | **−15.7%** |
| Artemis ASR — 21 min long-form (cold-pass) | 12,250 ms | **10,157 ms** | **−17.1%** |
| VRAM | 2,789 MiB | **2,789 MiB** | — |
| Concurrent overhead (4 streams) | — | **+12 ms** | negligible |

*Hardware: Beast3 · Intel Xeon Gold 6230 (20 cores) · 251 GB RAM · CTranslate2 4.7.2*

---

### NVIDIA A100-SXM4-80GB · bfloat16 / beam=5 / batch=32

| Scenario | Baseline | Optimized | Change |
|---|---|---|---|
| 5 min clean audio (cold-pass) | 2,676 ms | **2,189 ms** | **−18.2%** |
| 21 min long-form (cold-pass) | 10,322 ms | **8,153 ms** | **−21.0%** |
| 21 min telephone-quality (cold-pass) | 10,380 ms | **8,059 ms** | **−22.4%** |
| 21 min noisy SNR 10 dB (cold-pass) | 10,717 ms | **8,258 ms** | **−22.9%** |
| All 9 scenarios pass rate | — | **9 / 9** | all pass |

*Hardware: Golden Beast2 · AMD EPYC 7742 (128 cores) · 2 TB RAM · A100-SXM4-80GB GPU 5*

The A100 shows larger gains because higher GPU throughput makes CPU preprocessing proportionally more expensive — exactly where the pipeline optimizations apply.

---

## What Changed

All changes are in `faster_whisper/transcribe.py` and `faster_whisper/vad.py`.

**VAD on GPU** — `SileroVADModel` previously hardcoded `CPUExecutionProvider`. Profiling revealed the 1 MB VAD model was consuming ~1.5 s per request due to sequential CPU execution across hundreds of 512-sample windows. The session now prefers `CUDAExecutionProvider` (falling back to CPU when unavailable), reducing VAD time to ~0.05 s.

**Feature extraction pipelining** — `BatchedInferencePipeline` previously blocked GPU inference waiting for CPU feature extraction. A `ThreadPoolExecutor(max_workers=1)` now submits all batch extraction futures upfront. A background thread extracts batch N+1 features while CTranslate2 processes batch N on the GPU. CTranslate2 releases the GIL during CUDA operations, so the background thread runs freely with no synchronization overhead.

**Single-allocation VAD buffer** — eliminated the full-audio `np.pad` copy (~80 MB for 21-min audio) by constructing the segmented buffer directly in one `np.empty` allocation.

**Feature buffer pre-allocation** — eliminated `np.stack` + `pad_or_trim` intermediates by pre-allocating a single result buffer per batch.

**Single-batch executor skip** — for short audio (single batch), bypasses the executor entirely since there is nothing to pipeline.

**Eliminated duplicate tokenizer decode** — `tokenizer.decode(tokens)` was called twice per subsegment; walrus operator reduces to one call.

---

## Accuracy

Outputs are SHA1-identical across 5 consecutive runs. All 6 noise robustness conditions (clean, SNR 20/10/5 dB, telephone, overlapping speech) pass the 3% WER regression threshold.

→ [Full Optimization Report](OPTIMIZATION_REPORT.md)

---

---

# faster-whisper

**faster-whisper** is a reimplementation of OpenAI's Whisper model using [CTranslate2](https://github.com/OpenNMT/CTranslate2/), which is a fast inference engine for Transformer models.

This implementation is up to 4 times faster than [openai/whisper](https://github.com/openai/whisper) for the same accuracy while using less memory. The efficiency can be further improved with 8-bit quantization on both CPU and GPU.

## Benchmark

### Whisper

For reference, here's the time and memory usage that are required to transcribe [**13 minutes**](https://www.youtube.com/watch?v=0u7tTptBo9I) of audio using different implementations:

* [openai/whisper](https://github.com/openai/whisper)@[v20240930](https://github.com/openai/whisper/tree/v20240930)
* [whisper.cpp](https://github.com/ggerganov/whisper.cpp)@[v1.7.2](https://github.com/ggerganov/whisper.cpp/tree/v1.7.2)
* [transformers](https://github.com/huggingface/transformers)@[v4.46.3](https://github.com/huggingface/transformers/tree/v4.46.3)
* [faster-whisper](https://github.com/SYSTRAN/faster-whisper)@[v1.1.0](https://github.com/SYSTRAN/faster-whisper/tree/v1.1.0)

### Large-v2 model on GPU

| Implementation | Precision | Beam size | Time | VRAM Usage |
| --- | --- | --- | --- | --- |
| openai/whisper | fp16 | 5 | 2m23s | 4708MB |
| whisper.cpp (Flash Attention) | fp16 | 5 | 1m05s | 4127MB |
| transformers (SDPA)[^1] | fp16 | 5 | 1m52s | 4960MB |
| faster-whisper | fp16 | 5 | 1m03s | 4525MB |
| faster-whisper (`batch_size=8`) | fp16 | 5 | 17s | 6090MB |
| faster-whisper | int8 | 5 | 59s | 2926MB |
| faster-whisper (`batch_size=8`) | int8 | 5 | 16s | 4500MB |

### distil-whisper-large-v3 model on GPU

| Implementation | Precision | Beam size | Time | YT Commons WER |
| --- | --- | --- | --- | --- |
| transformers (SDPA) (`batch_size=16`) | fp16 | 5 | 46m12s | 14.801 |
| faster-whisper (`batch_size=16`) | fp16 | 5 | 25m50s | 13.527 |

*GPU Benchmarks are Executed with CUDA 12.4 on a NVIDIA RTX 3070 Ti 8GB.*
[^1]: transformers OOM for any batch size > 1

### Small model on CPU

| Implementation | Precision | Beam size | Time | RAM Usage |
| --- | --- | --- | --- | --- |
| openai/whisper | fp32 | 5 | 6m58s | 2335MB |
| whisper.cpp | fp32 | 5 | 2m05s | 1049MB |
| whisper.cpp (OpenVINO) | fp32 | 5 | 1m45s | 1642MB |
| faster-whisper | fp32 | 5 | 2m37s | 2257MB |
| faster-whisper (`batch_size=8`) | fp32 | 5 | 1m06s | 4230MB |
| faster-whisper | int8 | 5 | 1m42s | 1477MB |
| faster-whisper (`batch_size=8`) | int8 | 5 | 51s | 3608MB |

*Executed with 8 threads on an Intel Core i7-12700K.*

## Requirements

* Python 3.9 or greater

Unlike openai-whisper, FFmpeg does **not** need to be installed on the system. The audio is decoded with the Python library [PyAV](https://github.com/PyAV-Org/PyAV) which bundles the FFmpeg libraries in its package.

### GPU

GPU execution requires the following NVIDIA libraries to be installed:

* [cuBLAS for CUDA 12](https://developer.nvidia.com/cublas)
* [cuDNN 9 for CUDA 12](https://developer.nvidia.com/cudnn)

**Note**: The latest versions of `ctranslate2` only support CUDA 12 and cuDNN 9. For CUDA 11 and cuDNN 8, the current workaround is downgrading to the `3.24.0` version of `ctranslate2`, for CUDA 12 and cuDNN 8, downgrade to the `4.4.0` version of `ctranslate2`.

<details>
<summary>Other installation methods (click to expand)</summary>

#### Use Docker

The libraries (cuBLAS, cuDNN) are installed in this official NVIDIA CUDA Docker images: `nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04`.

#### Install with `pip` (Linux only)

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*

export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'`
```

#### Download the libraries from Purfview's repository (Windows & Linux)

Purfview's [whisper-standalone-win](https://github.com/Purfview/whisper-standalone-win) provides the required NVIDIA libraries for Windows & Linux in a [single archive](https://github.com/Purfview/whisper-standalone-win/releases/tag/libs).

</details>

## Installation

```bash
pip install faster-whisper
```

<details>
<summary>Other installation methods (click to expand)</summary>

### Install the master branch

```bash
pip install --force-reinstall "faster-whisper @ https://github.com/SYSTRAN/faster-whisper/archive/refs/heads/master.tar.gz"
```

### Install a specific commit

```bash
pip install --force-reinstall "faster-whisper @ https://github.com/SYSTRAN/faster-whisper/archive/a4f1cc8f11433e454c3934442b5e1a4ed5e865c3.tar.gz"
```

</details>

## Usage

### Faster-whisper

```python
from faster_whisper import WhisperModel

model_size = "large-v3"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe("audio.mp3", beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
```

**Warning:** `segments` is a *generator* so the transcription only starts when you iterate over it. The transcription can be run to completion by gathering the segments in a list or a `for` loop:

```python
segments, _ = model.transcribe("audio.mp3")
segments = list(segments)  # The transcription will actually run here.
```

### Batched Transcription

```python
from faster_whisper import WhisperModel, BatchedInferencePipeline

model = WhisperModel("turbo", device="cuda", compute_type="float16")
batched_model = BatchedInferencePipeline(model=model)
segments, info = batched_model.transcribe("audio.mp3", batch_size=16)

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
```

### Faster Distil-Whisper

```python
from faster_whisper import WhisperModel

model_size = "distil-large-v3"

model = WhisperModel(model_size, device="cuda", compute_type="float16")
segments, info = model.transcribe("audio.mp3", beam_size=5, language="en", condition_on_previous_text=False)

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
```

### Word-level timestamps

```python
segments, _ = model.transcribe("audio.mp3", word_timestamps=True)

for segment in segments:
    for word in segment.words:
        print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))
```

### VAD filter

```python
segments, _ = model.transcribe("audio.mp3", vad_filter=True)
```

```python
segments, _ = model.transcribe(
    "audio.mp3",
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500),
)
```

### Logging

```python
import logging

logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
```

## Community integrations

* [speaches](https://github.com/speaches-ai/speaches) — OpenAI compatible server using faster-whisper
* [WhisperX](https://github.com/m-bain/whisperX) — speaker diarization and word-level timestamps
* [whisper-ctranslate2](https://github.com/Softcatala/whisper-ctranslate2) — command line client compatible with openai/whisper
* [whisper-diarize](https://github.com/MahmoudAshraf97/whisper-diarization) — speaker diarization with NVIDIA NeMo
* [whisper-standalone-win](https://github.com/Purfview/whisper-standalone-win) — standalone CLI executables for Windows, Linux & macOS
* [WhisperLive](https://github.com/collabora/WhisperLive) — near-live transcription
* [Whisper-Streaming](https://github.com/ufal/whisper_streaming) — real-time streaming with adaptive latency
