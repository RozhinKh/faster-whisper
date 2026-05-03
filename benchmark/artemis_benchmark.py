"""
Artemis-facing benchmark entrypoint.

This wrapper keeps the GA benchmark logic but writes results in the
repo root as artemis_results.json, which Artemis automatically ingests
as custom metrics after each benchmark run.
"""

import argparse
import json
import os
from datetime import datetime, timezone

from ga_benchmark import run

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BENCHMARK_DIR)
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "artemis_results.json")
DEFAULT_PROFILE_DIR = os.path.join(BENCHMARK_DIR, "artifacts")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _write_summary(profile_dir: str, result: dict, output_path: str) -> None:
    os.makedirs(profile_dir, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artemis_results_path": output_path,
        "metrics": result,
    }

    json_path = os.path.join(profile_dir, "artemis_benchmark_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    md_path = os.path.join(profile_dir, "artemis_benchmark_summary.md")
    lines = [
        "# Artemis Benchmark Summary",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Results file: `{output_path}`",
        f"- Model: `{result['model']}`",
        f"- Compute type: `{result['compute_type']}`",
        f"- Device index: `{result['device_index']}`",
        f"- Benchmark mode: `{result['benchmark_mode']}`",
        f"- Audio seconds: `{result['audio_seconds']}`",
        f"- Speed (s): `{result['speed_min_s']}`",
        f"- VRAM used (MiB): `{result['vram_used_mib']}`",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Artemis benchmark wrapper")
    parser.add_argument("--model", default=_env("BENCHMARK_MODEL", "large-v3"))
    parser.add_argument(
        "--compute-type", default=_env("BENCHMARK_COMPUTE_TYPE", "float16")
    )
    parser.add_argument("--device", default=_env("BENCHMARK_DEVICE", "cuda"))
    parser.add_argument(
        "--device-index", type=int, default=int(_env("BENCHMARK_DEVICE_INDEX", "0"))
    )
    parser.add_argument("--language", default=_env("BENCHMARK_LANGUAGE", "fr"))
    parser.add_argument(
        "--beam-size", type=int, default=int(_env("BENCHMARK_BEAM_SIZE", "5"))
    )
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=float(_env("BENCHMARK_CLIP_SECONDS", "45")),
        help="First N seconds to transcribe for quick Artemis smoke tests.",
    )
    parser.add_argument(
        "--full-audio",
        action="store_true",
        help="Run on the full benchmark.m4a instead of a clipped smoke test.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--profile-dir", default=DEFAULT_PROFILE_DIR)
    args = parser.parse_args()

    clip_seconds = None if args.full_audio else args.clip_seconds
    result = run(
        args.model,
        args.compute_type,
        args.device,
        args.device_index,
        language=args.language,
        beam_size=args.beam_size,
        clip_seconds=clip_seconds,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump([result], f, indent=2)
        f.write("\n")

    _write_summary(args.profile_dir, result, args.output)
    print(f"  results -> {args.output}")


if __name__ == "__main__":
    main()
