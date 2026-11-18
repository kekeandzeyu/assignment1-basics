from __future__ import annotations

import heapq
import os
from collections import Counter, defaultdict
from multiprocessing import Pool
from typing import BinaryIO

import regex as re

# GPT-2 pre-tokenization regex pattern
GPT2_PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if desired_num_chunks <= 1 or file_size == 0:
        return [0, file_size] if file_size > 0 else [0]

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def _pretokenize_text(text: str, special_tokens: list[str]) -> Counter[tuple[int, ...]]:
    """Split text on special tokens, pre-tokenize with GPT2_PAT, and count UTF-8 byte tuples."""
    counts: Counter[tuple[int, ...]] = Counter()
    if special_tokens:
        # Sort longest-first so overlapping special tokens match greedily
        escaped = [re.escape(st) for st in sorted(special_tokens, key=len, reverse=True)]
        segments = re.split("|".join(escaped), text)
    else:
        segments = [text]

    for seg in segments:
        if not seg:
            continue
        for match in GPT2_PAT.finditer(seg):
            token_bytes = match.group(0).encode("utf-8")
            counts[tuple(token_bytes)] += 1

    return counts


def _pretokenize_chunk(args: tuple[str, int, int, list[str]]) -> Counter[tuple[int, ...]]:
    """Worker function to read a byte range [start, end) and pre-tokenize it."""
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    return _pretokenize_text(text, special_tokens)


def _count_pretokens(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int | None = None,
) -> Counter[tuple[int, ...]]:
    """Pre-tokenize the input corpus, using multiprocessing for larger files."""
    file_size = os.path.getsize(input_path)
    if num_processes is None:
        num_processes = min(os.cpu_count() or 1, 8)

    # For small files (< 1 MB) or when no special token is available for chunk alignment,
    # single-process execution avoids multiprocessing startup overhead.
    if file_size < 1_000_000 or num_processes <= 1 or not special_tokens:
        with open(input_path, encoding="utf-8", errors="ignore") as f:
            return _pretokenize_text(f.read(), special_tokens)

    # Use many more chunks than processes (~64 MB each) so that each worker only ever
    # holds a small piece of the file in memory, instead of 1/num_processes of it.
    num_chunks = max(num_processes, file_size // (64 * 1024 * 1024))
    split_token = special_tokens[0].encode("utf-8")
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, split_token)

    tasks = [
        (os.fspath(input_path), start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if end > start
    ]

    total_counts: Counter[tuple[int, ...]] = Counter()
    with Pool(processes=min(num_processes, len(tasks))) as pool:
        for chunk_counts in pool.imap_unordered(_pretokenize_chunk, tasks):
            total_counts.update(chunk_counts)
    return total_counts


class _Desc:
    """Wraps a (bytes, bytes) pair so that a min-heap pops the lexicographically GREATEST pair first."""

    __slots__ = ("key",)

    def __init__(self, key: tuple[bytes, bytes]):
        self.key = key

    def __lt__(self, other: _Desc) -> bool:
        return self.key > other.key

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Desc) and self.key == other.key


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer on the given corpus.

    Args:
        input_path: Path to the training text file.
        vocab_size: Maximum final vocabulary size (including initial 256 bytes and special tokens).
        special_tokens: Strings treated as indivisible hard boundaries during training.

    Returns:
        (vocab, merges):
            vocab: Mapping from integer token ID to token bytes.
            merges: Ordered list of merged (token1_bytes, token2_bytes) pairs.
    """
    # 1. Vocabulary initialization: 256 single-byte tokens + special tokens
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    existing_tokens = set(vocab.values())
    for st in special_tokens:
        st_bytes = st.encode("utf-8")
        if st_bytes not in existing_tokens:
            vocab[len(vocab)] = st_bytes
            existing_tokens.add(st_bytes)

    num_merges = vocab_size - len(vocab)
    if num_merges <= 0:
        return vocab, []

    # 2. Pre-tokenization: map each unique pre-token (as byte IDs) to its corpus frequency
    pretoken_counts = _count_pretokens(
        input_path=input_path,
        special_tokens=special_tokens,
        num_processes=kwargs.get("num_processes"),
    )

    words: list[list[int]] = []
    freqs: list[int] = []
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    pair_to_words: dict[tuple[int, int], set[int]] = defaultdict(set)

    for w_idx, (token_tuple, freq) in enumerate(pretoken_counts.items()):
        word = list(token_tuple)
        words.append(word)
        freqs.append(freq)
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(w_idx)

    # 3. Compute BPE merges iteratively with incremental pair-count updates.
    #
    # Finding the best pair with max() over all pairs costs O(#pairs) per merge. Instead we keep
    # a heap ordered by (highest count, then lexicographically greatest bytes) -- the same rule
    # as max(). Entries are never updated in place: when a count changes we push a fresh entry,
    # and when popping we skip "stale" entries whose count no longer matches pair_counts.
    def heap_entry(p: tuple[int, int]) -> tuple[int, _Desc, tuple[int, int]]:
        return (-pair_counts[p], _Desc((vocab[p[0]], vocab[p[1]])), p)

    heap = [heap_entry(p) for p in pair_counts]
    heapq.heapify(heap)

    merges: list[tuple[bytes, bytes]] = []
    for _ in range(num_merges):
        best_pair = None
        while heap:
            neg_count, _, p = heapq.heappop(heap)
            if pair_counts.get(p) == -neg_count:  # entry is up to date
                best_pair = p
                break
        if best_pair is None:
            break

        p0, p1 = best_pair
        new_id = len(vocab)
        vocab[new_id] = vocab[p0] + vocab[p1]
        merges.append((vocab[p0], vocab[p1]))

        affected_words = list(pair_to_words.pop(best_pair, ()))
        changed: set[tuple[int, int]] = set()
        for w_idx in affected_words:
            word = words[w_idx]
            freq = freqs[w_idx]

            # Subtract old pair counts for this word
            old_pairs: set[tuple[int, int]] = set()
            for i in range(len(word) - 1):
                p = (word[i], word[i + 1])
                old_pairs.add(p)
                pair_counts[p] -= freq
                if pair_counts[p] <= 0:
                    del pair_counts[p]

            # Merge all non-overlapping occurrences of best_pair left-to-right
            new_word: list[int] = []
            i = 0
            n = len(word)
            while i < n:
                if i + 1 < n and word[i] == p0 and word[i + 1] == p1:
                    new_word.append(new_id)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            words[w_idx] = new_word

            # Add new pair counts for the updated word
            new_pairs: set[tuple[int, int]] = set()
            for i in range(len(new_word) - 1):
                p = (new_word[i], new_word[i + 1])
                new_pairs.add(p)
                pair_counts[p] += freq

            for p in old_pairs - new_pairs:
                if p in pair_to_words:
                    pair_to_words[p].discard(w_idx)
                    if not pair_to_words[p]:
                        del pair_to_words[p]
            for p in new_pairs - old_pairs:
                pair_to_words[p].add(w_idx)
            changed |= old_pairs
            changed |= new_pairs

        # Push the new counts of every pair touched by this merge (old heap entries become stale).
        for p in changed:
            if p in pair_counts:
                heapq.heappush(heap, heap_entry(p))

    return vocab, merges
