import argparse
import timeit

from utils import make_inference_fn

parser = argparse.ArgumentParser(description="Speed benchmark")
parser.add_argument("--repeat",       type=int,   default=3,            help="Number of repetitions.")
parser.add_argument("--number",       type=int,   default=10,           help="Runs per repetition.")
parser.add_argument("--compute-type", default="float16",                help="CTranslate2 compute type.")
parser.add_argument("--beam-size",    type=int,   default=5,            help="Beam size.")
parser.add_argument("--batch-size",   type=int,   default=16,           help="Batch size (BatchedInferencePipeline).")
args = parser.parse_args()


def measure_speed(label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  compute={args.compute_type}  beam={args.beam_size}  batch={args.batch_size}")
    print(f"{'='*60}")
    fn = make_inference_fn(
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        batched=True,
        batch_size=args.batch_size,
    )
    print("  Warmup...")
    fn()
    print(f"  Timing (repeat={args.repeat}, number={args.number})...")
    runtimes = timeit.repeat(fn, repeat=args.repeat, number=args.number)
    min_per_run = min(runtimes) / args.number
    print(f"  Raw totals : {[round(r, 3) for r in runtimes]}")
    print(f"  Min per run: {min_per_run:.3f}s")
    return min_per_run


if __name__ == "__main__":
    measure_speed(
        f"faster-whisper large-v3  |  "
        f"compute={args.compute_type}  beam={args.beam_size}  batch={args.batch_size}"
    )
