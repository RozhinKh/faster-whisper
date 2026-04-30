"""
Output regression detection for faster-whisper.

Saves reference transcriptions before an optimisation and compares them
afterwards to detect any drift introduced by the change.

Usage:
    # Step 1 – before optimisation, save reference outputs:
    python benchmark/regression.py --save

    # Step 2 – after optimisation, compare against the reference:
    python benchmark/regression.py --compare

    # Include extra audio files alongside benchmark.m4a:
    python benchmark/regression.py --save --audio path/a.wav path/b.mp3
"""

import argparse
import json
import os
from datetime import datetime

import jiwer
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from faster_whisper import WhisperModel

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_AUDIO = os.path.join(BENCHMARK_DIR, "benchmark.m4a")
REFERENCE_FILE = os.path.join(BENCHMARK_DIR, "reference_outputs.json")
NORMALIZER_JSON = os.path.join(BENCHMARK_DIR, "normalizer.json")


def _load_normalizer() -> EnglishTextNormalizer:
    with open(NORMALIZER_JSON, encoding="utf-8") as f:
        return EnglishTextNormalizer(json.load(f))


def _transcribe_files(model: WhisperModel, audio_paths: list) -> dict:
    outputs = {}
    for path in audio_paths:
        segments, info = model.transcribe(path)
        outputs[os.path.abspath(path)] = {
            "transcript": "".join(seg.text for seg in segments),
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "duration_s": round(info.duration, 2),
        }
    return outputs


def save_reference(
    model_size: str,
    compute_type: str,
    device: str,
    device_index,
    audio_paths: list,
) -> dict:
    model = WhisperModel(
        model_size, device=device, device_index=device_index, compute_type=compute_type
    )
    outputs = _transcribe_files(model, audio_paths)

    reference = {
        "saved_at": datetime.now().isoformat(),
        "model_size": model_size,
        "compute_type": compute_type,
        "device": device,
        "outputs": outputs,
    }
    with open(REFERENCE_FILE, "w", encoding="utf-8") as f:
        json.dump(reference, f, indent=2, ensure_ascii=False)

    print(f"Reference saved → {REFERENCE_FILE}")
    for path, data in outputs.items():
        print(
            f"  {os.path.basename(path):30s}  "
            f"{data['duration_s']:.1f}s audio  "
            f"{len(data['transcript'])} chars  "
            f"lang={data['language']} ({data['language_probability']:.2f})"
        )
    return reference


def compare_against_reference(
    model_size: str,
    compute_type: str,
    device: str,
    device_index,
) -> dict:
    if not os.path.exists(REFERENCE_FILE):
        raise FileNotFoundError(
            f"No reference file found at {REFERENCE_FILE}. Run --save first."
        )

    with open(REFERENCE_FILE, encoding="utf-8") as f:
        reference = json.load(f)

    audio_paths = list(reference["outputs"].keys())
    missing = [p for p in audio_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Reference audio files missing from disk: {missing}")

    normalizer = _load_normalizer()
    model = WhisperModel(
        model_size, device=device, device_index=device_index, compute_type=compute_type
    )
    current = _transcribe_files(model, audio_paths)

    ref_texts, cur_texts = [], []
    changed = []

    print("\n" + "=" * 64)
    print("OUTPUT REGRESSION REPORT")
    print("=" * 64)
    print(f"  Reference : {reference['saved_at']}  "
          f"[{reference['model_size']} / {reference['compute_type']}]")
    print(f"  Current   : {datetime.now().isoformat()}  [{model_size} / {compute_type}]")
    print()

    for path in audio_paths:
        ref_norm = normalizer(reference["outputs"][path]["transcript"])
        cur_norm = normalizer(current[path]["transcript"])
        ref_texts.append(ref_norm)
        cur_texts.append(cur_norm)
        name = os.path.basename(path)

        if ref_norm == cur_norm:
            print(f"  [OK      ]  {name}")
        else:
            drift = 100 * jiwer.wer(hypothesis=cur_norm, reference=ref_norm)
            cer = 100 * jiwer.cer(hypothesis=cur_norm, reference=ref_norm)
            changed.append(
                {"file": name, "drift_wer_pct": round(drift, 3), "drift_cer_pct": round(cer, 3)}
            )
            print(f"  [CHANGED ]  {name}   drift WER={drift:.3f}%  CER={cer:.3f}%")

            # show first diverging word in context
            ref_words = ref_norm.split()
            cur_words = cur_norm.split()
            for i, (rw, cw) in enumerate(zip(ref_words, cur_words)):
                if rw != cw:
                    lo = max(0, i - 3)
                    print(
                        f"              first diff @ word {i}: "
                        f"before=«{' '.join(ref_words[lo:i + 4])}»  "
                        f"after=«{' '.join(cur_words[lo:i + 4])}»"
                    )
                    break

    overall_wer = (
        100 * jiwer.wer(hypothesis=cur_texts, reference=ref_texts) if ref_texts else 0.0
    )
    overall_cer = (
        100 * jiwer.cer(hypothesis=cur_texts, reference=ref_texts) if ref_texts else 0.0
    )

    print()
    print(f"  Overall drift WER (pre→post) : {overall_wer:.3f}%")
    print(f"  Overall drift CER (pre→post) : {overall_cer:.3f}%")
    print(f"  Changed files                : {len(changed)} / {len(audio_paths)}")

    return {
        "overall_drift_wer_pct": round(overall_wer, 3),
        "overall_drift_cer_pct": round(overall_cer, 3),
        "changed_files": len(changed),
        "total_files": len(audio_paths),
        "details": changed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Output regression detection")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--save", action="store_true",
        help="Save reference outputs (run this BEFORE optimisation)"
    )
    group.add_argument(
        "--compare", action="store_true",
        help="Compare current outputs against saved reference (run AFTER optimisation)"
    )
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--device-index", type=int, nargs="+", default=[0],
        help="GPU device index(es), e.g. --device-index 0 1 2 3"
    )
    parser.add_argument(
        "--audio", nargs="+", default=[],
        help="Extra audio files to include (benchmark.m4a is always included)"
    )
    args = parser.parse_args()

    device_index = args.device_index[0] if len(args.device_index) == 1 else args.device_index

    audio_files = [DEFAULT_AUDIO] + [os.path.abspath(p) for p in args.audio]
    # deduplicate while preserving order, skip files that don't exist
    seen = set()
    audio_files = [
        p for p in audio_files
        if os.path.exists(p) and not (p in seen or seen.add(p))
    ]

    if args.save:
        save_reference(args.model, args.compute_type, args.device, device_index, audio_files)
    else:
        compare_against_reference(args.model, args.compute_type, args.device, device_index)
