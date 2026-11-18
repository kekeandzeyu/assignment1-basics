"""Section 2.7: tokenizer experiments.

    # (a), (b), (c): compression ratios, cross-domain behaviour and throughput
    uv run python scripts/tokenizer_experiments.py stats

    # (d): encode a text file into a uint16 NumPy array of token IDs
    uv run python scripts/tokenizer_experiments.py encode data/TinyStoriesV2-GPT4-valid.txt \
        data/tokenizers/tinystories data/tokens/tinystories_valid.npy
"""

from __future__ import annotations

import argparse
import os
import random
import time
from array import array
from multiprocessing import Pool

import numpy as np

from cs336_basics.bpe import find_chunk_boundaries
from cs336_basics.tokenizer import Tokenizer

EOT = "<|endoftext|>"
TS_VALID = "data/TinyStoriesV2-GPT4-valid.txt"
OWT_VALID = "data/owt_valid.txt"
TS_TOK = "data/tokenizers/tinystories"
OWT_TOK = "data/tokenizers/owt"


def load(prefix: str) -> Tokenizer:
    return Tokenizer.from_files(f"{prefix}-vocab.json", f"{prefix}-merges.txt", [EOT])


def sample_docs(path: str, k: int = 10, seed: int = 0, max_bytes: int = 50_000_000) -> list[str]:
    """Sample k documents (separated by <|endoftext|>) from the first max_bytes of a file."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        docs = [d.strip() for d in f.read(max_bytes).split(EOT)]
    docs = [d for d in docs[:-1] if d]  # the last one may be cut off by max_bytes
    return random.Random(seed).sample(docs, k)


def compression(tok: Tokenizer, docs: list[str]) -> float:
    n_bytes = sum(len(d.encode("utf-8")) for d in docs)
    n_tokens = sum(len(tok.encode(d)) for d in docs)
    return n_bytes / n_tokens


def stats() -> None:
    ts, owt = load(TS_TOK), load(OWT_TOK)
    ts_docs, owt_docs = sample_docs(TS_VALID), sample_docs(OWT_VALID)

    print("(a) compression ratio (bytes/token), each tokenizer on its own data")
    print(f"    TinyStories tokenizer (10K) on TinyStories docs: {compression(ts, ts_docs):.2f}")
    print(f"    OpenWebText tokenizer (32K) on OWT docs:         {compression(owt, owt_docs):.2f}")

    print("(b) cross-domain")
    print(f"    TinyStories tokenizer on OWT docs:  {compression(ts, owt_docs):.2f}")
    print(f"    OWT tokenizer on TinyStories docs:  {compression(owt, ts_docs):.2f}")
    sentence = owt_docs[0][:200]
    for name, tok in [("TinyStories", ts), ("OWT", owt)]:
        pieces = [tok.decode([i]) for i in tok.encode(sentence)]
        print(f"    {name} tokenizer, {len(pieces)} tokens: {pieces[:40]}")

    print("(c) throughput (single process, fresh tokenizer so the cache starts empty)")
    with open(OWT_VALID, encoding="utf-8", errors="ignore") as f:
        text = f.read(20_000_000)
    tok = load(OWT_TOK)
    t = time.time()
    tok.encode(text)
    dt = time.time() - t
    mb_s = len(text.encode("utf-8")) / dt / 1e6
    pile_hours = 825e9 / (mb_s * 1e6) / 3600
    print(f"    {mb_s:.2f} MB/s  ->  The Pile (825 GB): {pile_hours:,.0f} hours (~{pile_hours / 24:.0f} days) on 1 core")


# ---------------------------------------------------------------- (d) encoding
_tok: Tokenizer | None = None


def _init(prefix: str) -> None:
    global _tok
    _tok = load(prefix)


def _encode_chunk(args: tuple[str, int, int, str]) -> tuple[str, int]:
    """Encode bytes [start, end) of the file and write the IDs (uint16) to a part file."""
    path, start, end, part_path = args
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    ids = array("H")  # 2 bytes per token instead of ~36 for a Python list of ints
    ids.extend(_tok.encode_iterable(text.splitlines(keepends=True)))
    with open(part_path, "wb") as f:
        ids.tofile(f)
    return part_path, len(ids)


def encode_file(input_path: str, tok_prefix: str, out_path: str, num_processes: int = 4) -> None:
    assert len(load(tok_prefix).vocab) <= 2**16, "uint16 needs vocab size <= 65536"
    size = os.path.getsize(input_path)
    with open(input_path, "rb") as f:
        # Chunks start at <|endoftext|>, a hard boundary, so encoding them independently
        # gives exactly the same IDs as encoding the whole file at once.
        bounds = find_chunk_boundaries(f, max(num_processes, size // 32_000_000), EOT.encode())
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tasks = [(input_path, s, e, f"{out_path}.part{i}") for i, (s, e) in enumerate(zip(bounds[:-1], bounds[1:]))]

    t = time.time()
    with Pool(num_processes, initializer=_init, initargs=(tok_prefix,)) as pool:
        results = pool.map(_encode_chunk, tasks)  # keeps chunk order
    total = sum(n for _, n in results)

    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint16, shape=(total,))
    pos = 0
    for part, n in results:
        out[pos : pos + n] = np.fromfile(part, dtype=np.uint16)
        pos += n
        os.remove(part)
    out.flush()
    dt = time.time() - t
    print(f"{input_path}: {size / 1e6:.0f} MB -> {total:,} tokens in {dt:.0f}s "
          f"({size / dt / 1e6:.1f} MB/s with {num_processes} processes), ratio {size / total:.2f} bytes/token")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats")
    enc = sub.add_parser("encode")
    enc.add_argument("input_path")
    enc.add_argument("tokenizer_prefix")
    enc.add_argument("out_path")
    enc.add_argument("--num-processes", type=int, default=4)
    args = ap.parse_args()
    if args.cmd == "stats":
        stats()
    else:
        encode_file(args.input_path, args.tokenizer_prefix, args.out_path, args.num_processes)


if __name__ == "__main__":
    main()
