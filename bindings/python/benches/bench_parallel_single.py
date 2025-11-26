"""Time `Tokenizer.encode` against `Tokenizer.encode_parallel` on one long input.

Usage:
    python bench_parallel_single.py TOKENIZER [TOKENIZER ...] [--text big.txt] [--sizes 0.1,1,4]

TOKENIZER is a `tokenizer.json` path or a Hugging Face Hub id. Each run also checks that both
methods return the same encoding.
"""

import argparse
import os
import statistics
import time

from tokenizers import Tokenizer


def load(name):
    return Tokenizer.from_file(name) if os.path.exists(name) else Tokenizer.from_pretrained(name)


def timed(fn, runs):
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times), result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizers", nargs="+")
    parser.add_argument("--text", default="../../../tokenizers/data/big.txt")
    parser.add_argument("--sizes", default="0.1,0.5,1,4", help="input sizes in MB")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    base = open(args.text, encoding="utf-8").read()
    print(f"{'tokenizer':<40} {'MB':>5} {'encode':>10} {'parallel':>10} {'speedup':>8}")
    for name in args.tokenizers:
        tokenizer = load(name)
        for size in (float(s) for s in args.sizes.split(",")):
            n = int(size * 1_000_000)
            text = (base * (n // len(base) + 1))[:n]
            serial_time, expected = timed(lambda: tokenizer.encode(text, add_special_tokens=False), args.runs)
            parallel_time, actual = timed(lambda: tokenizer.encode_parallel(text, add_special_tokens=False), args.runs)
            assert actual.ids == expected.ids and actual.offsets == expected.offsets, f"{name}: mismatch"
            print(
                f"{os.path.basename(name):<40} {size:>5} {serial_time * 1e3:>8.1f}ms "
                f"{parallel_time * 1e3:>8.1f}ms {serial_time / parallel_time:>7.2f}x"
            )


if __name__ == "__main__":
    main()
