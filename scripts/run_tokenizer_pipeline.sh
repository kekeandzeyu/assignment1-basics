#!/usr/bin/env bash
# Runs the remaining Section 2.5-2.7 work unattended. Log: data/pipeline.log
set -euo pipefail
cd "$(dirname "$0")/.."

step() { echo; echo "=== $(date '+%F %T') $* ==="; }

step "download TinyStories train (resumes a partial file)"
wget -c -q -O data/TinyStoriesV2-GPT4-train.txt \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
ls -la data/TinyStoriesV2-GPT4-train.txt

step "2.5 train TinyStories tokenizer (10K)"
uv run python scripts/train_tokenizer.py data/TinyStoriesV2-GPT4-train.txt 10000 data/tokenizers/tinystories

step "2.5 OWT tokenizer (32K, trained earlier on owt_valid)"
[ -f data/tokenizers/owt-vocab.json ] || uv run python scripts/train_tokenizer.py data/owt_valid.txt 32000 data/tokenizers/owt

step "2.7 (a)(b)(c) stats"
uv run python scripts/tokenizer_experiments.py stats

step "2.7 (d) encode datasets to uint16"
uv run python scripts/tokenizer_experiments.py encode data/TinyStoriesV2-GPT4-valid.txt data/tokenizers/tinystories data/tokens/tinystories_valid.npy
uv run python scripts/tokenizer_experiments.py encode data/owt_valid.txt data/tokenizers/owt data/tokens/owt_valid.npy
uv run python scripts/tokenizer_experiments.py encode data/TinyStoriesV2-GPT4-train.txt data/tokenizers/tinystories data/tokens/tinystories_train.npy

step "ALL DONE"
ls -la data/tokenizers data/tokens
