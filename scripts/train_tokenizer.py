"""Train a byte-level BPE tokenizer and save it to disk (Section 2.5 / needed for 2.7).

Example:
    uv run python scripts/train_tokenizer.py data/TinyStoriesV2-GPT4-train.txt 10000 data/tokenizers/tinystories
"""

from __future__ import annotations

import argparse
import os
import resource
import time

from cs336_basics.bpe import train_bpe
from cs336_basics.tokenizer import save_tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_path")
    ap.add_argument("vocab_size", type=int)
    ap.add_argument("out_prefix", help="writes <prefix>-vocab.json and <prefix>-merges.txt")
    ap.add_argument("--special", nargs="*", default=["<|endoftext|>"])
    ap.add_argument("--num-processes", type=int, default=None)
    args = ap.parse_args()

    start = time.time()
    vocab, merges = train_bpe(args.input_path, args.vocab_size, args.special, num_processes=args.num_processes)
    elapsed = time.time() - start

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    save_tokenizer(vocab, merges, f"{args.out_prefix}-vocab.json", f"{args.out_prefix}-merges.txt")

    # ru_maxrss is in KB on Linux. Include worker processes (RUSAGE_CHILDREN) as well.
    peak_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    peak_child = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    longest = max(vocab.values(), key=len)
    print(f"time: {elapsed:.1f}s  peak RSS main: {peak_self:.0f} MB  largest worker: {peak_child:.0f} MB")
    print(f"vocab size: {len(vocab)}  merges: {len(merges)}")
    print(f"longest token ({len(longest)} bytes): {longest!r}")


if __name__ == "__main__":
    main()
