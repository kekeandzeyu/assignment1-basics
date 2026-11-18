from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator

import regex as re

from cs336_basics.bpe import GPT2_PAT


# ---------------------------------------------------------------------------
# Serialization helpers (GPT-2 style vocab.json + merges.txt)
# ---------------------------------------------------------------------------
def bytes_to_unicode() -> dict[int, str]:
    """
    GPT-2's reversible byte -> printable-character mapping.

    Raw bytes such as b"\\x00" or b" " are awkward to store in JSON/text files,
    so each of the 256 bytes is mapped to a visible Unicode character
    (e.g. the space byte 32 becomes "Ġ"). This is only for writing files to disk;
    the tokenizer itself works on real bytes.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def save_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
) -> None:
    """Write vocab as {token_string: id} JSON and merges as one 'a b' pair per line."""
    enc = bytes_to_unicode()
    to_str = lambda b: "".join(enc[x] for x in b)  # noqa: E731
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({to_str(tok): idx for idx, tok in vocab.items()}, f, ensure_ascii=False, indent=0)
    with open(merges_path, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{to_str(a)} {to_str(b)}\n")


def load_tokenizer_files(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Inverse of save_tokenizer. Also reads GPT-2's original vocab.json / merges.txt."""
    dec = {c: b for b, c in bytes_to_unicode().items()}
    to_bytes = lambda s: bytes(dec[c] for c in s)  # noqa: E731
    with open(vocab_path, encoding="utf-8") as f:
        vocab = {idx: to_bytes(tok) for tok, idx in json.load(f).items()}
    merges: list[tuple[bytes, bytes]] = []
    with open(merges_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(" ")
            if len(parts) == 2 and not line.startswith("#version"):
                merges.append((to_bytes(parts[0]), to_bytes(parts[1])))
    return vocab, merges


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab: dict[int, bytes] = dict(vocab)
        self.special_tokens: list[str] = list(special_tokens or [])

        # Append any special token that is not already in the vocabulary.
        existing = set(self.vocab.values())
        for st in self.special_tokens:
            b = st.encode("utf-8")
            if b not in existing:
                self.vocab[len(self.vocab)] = b
                existing.add(b)

        # Reverse lookup: bytes -> id (needed to turn bytes into integer IDs).
        self.byte_to_id: dict[bytes, int] = {b: i for i, b in self.vocab.items()}

        # Merge rank: (left_id, right_id) -> (rank, merged_id).
        # A lower rank means the merge was learned earlier and must be applied first.
        self.merge_ranks: dict[tuple[int, int], tuple[int, int]] = {}
        for rank, (a, b) in enumerate(merges):
            self.merge_ranks[(self.byte_to_id[a], self.byte_to_id[b])] = (rank, self.byte_to_id[a + b])

        # Regex that finds special tokens. Longest first, so that
        # "<|endoftext|><|endoftext|>" wins over "<|endoftext|>" when both exist.
        # The capture group (...) makes re.split keep the special tokens in its output.
        if self.special_tokens:
            alts = "|".join(re.escape(s) for s in sorted(self.special_tokens, key=len, reverse=True))
            self._special_re = re.compile(f"({alts})")
        else:
            self._special_re = None

        # Memo: pre-token bytes -> token IDs. Words like " the" repeat constantly.
        self._cache: dict[bytes, list[int]] = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        vocab, merges = load_tokenizer_files(vocab_filepath, merges_filepath)
        return cls(vocab, merges, special_tokens)

    # ------------------------------------------------------------------ encode
    def _encode_pretoken(self, token_bytes: bytes) -> list[int]:
        """Apply the BPE merges to ONE pre-token (e.g. b' cat') and return its IDs."""
        cached = self._cache.get(token_bytes)
        if cached is not None:
            return cached

        ids = [self.byte_to_id[bytes([b])] for b in token_bytes]
        while len(ids) >= 2:
            # Among all adjacent pairs, find the one whose merge was learned earliest.
            best = None
            for pair in zip(ids, ids[1:]):
                info = self.merge_ranks.get(pair)
                if info is not None and (best is None or info[0] < best[0]):
                    best = (info[0], pair, info[1])
            if best is None:
                break  # no merge applies any more
            _, (p0, p1), new_id = best
            # Merge every occurrence of that pair, left to right.
            merged: list[int] = []
            i = 0
            while i < len(ids):
                if i + 1 < len(ids) and ids[i] == p0 and ids[i + 1] == p1:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(ids[i])
                    i += 1
            ids = merged

        if len(self._cache) < 100_000:
            self._cache[token_bytes] = ids
        return ids

    def _encode_ordinary(self, text: str) -> list[int]:
        """Encode text that contains no special tokens: regex pre-tokenize, then BPE each piece."""
        out: list[int] = []
        for m in GPT2_PAT.finditer(text):
            out.extend(self._encode_pretoken(m.group(0).encode("utf-8")))
        return out

    def encode(self, text: str) -> list[int]:
        if self._special_re is None:
            return self._encode_ordinary(text)
        out: list[int] = []
        # Because of the capture group, parts alternate: [text, special, text, special, ...]
        for i, part in enumerate(self._special_re.split(text)):
            if i % 2 == 1:
                out.append(self.byte_to_id[part.encode("utf-8")])
            elif part:
                out.extend(self._encode_ordinary(part))
        return out

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Lazily encode a stream of strings (e.g. a file handle, read line by line).

        The tricky part is that tokenizing chunk-by-chunk must give EXACTLY the same IDs as
        tokenizing the whole text at once. A token could straddle a chunk boundary, e.g.
        "  \\n" + "Hello" pre-tokenizes differently from "  \\n" and "Hello" separately, or a
        special token could be split as "<|endo" + "ftext|>". So we keep a small buffer and
        only emit the parts that more text cannot change.
        """
        buffer = ""
        for chunk in iterable:
            buffer += chunk

            # 1) Everything up to the last complete special token is final.
            if self._special_re is not None:
                parts = self._special_re.split(buffer)
                if len(parts) > 1:
                    yield from self.encode("".join(parts[:-1]))
                    buffer = parts[-1]

            # 2) Hold back a tail that might be the beginning of a special token.
            hold = self._special_prefix_len(buffer)
            safe, held = buffer[: len(buffer) - hold], buffer[len(buffer) - hold :]

            # 3) In the safe region, all pre-tokens except the last are final; the last one
            #    might still grow (e.g. " hel" + "lo"), so it stays in the buffer.
            last_start = None
            for m in GPT2_PAT.finditer(safe):
                if last_start is not None:
                    yield from self._encode_pretoken(safe[last_start : m.start()].encode("utf-8"))
                last_start = m.start()
            if last_start is not None:
                buffer = safe[last_start:] + held
            # else: safe is empty, and buffer is just the held tail

        if buffer:
            yield from self.encode(buffer)

    def _special_prefix_len(self, text: str) -> int:
        """Length of the longest suffix of text that is a proper prefix of some special token."""
        best = 0
        for st in self.special_tokens:
            for k in range(min(len(st) - 1, len(text)), best, -1):
                if text.endswith(st[:k]):
                    best = k
                    break
        return best

    # ------------------------------------------------------------------ decode
    def decode(self, ids: list[int]) -> str:
        # Concatenate bytes FIRST, then decode. One character such as "🙃" (4 bytes)
        # may be spread over several tokens, so decoding token-by-token would break it.
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")
