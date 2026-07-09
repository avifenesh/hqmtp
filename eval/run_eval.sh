#!/usr/bin/env bash
# hqmtp before/after eval runner (see EVAL.md). Run ON the g7e box from ~/bw24.
#
# Usage: run_eval.sh <model_dir> <tag>
#   model_dir  st-ct checkpoint (BEFORE) or patched copy (AFTER)
#   tag        results label, e.g. before-stct / after-r1
#
# Expects: eval prompts at ~/hqmtp-eval/prompts/pe*.txt, corpora at
# ~/hqmtp-eval/corpora/eval-*.txt (synced from hqmtp/eval/), bw24 built.
set -euo pipefail

MODEL="${1:?usage: run_eval.sh <model_dir> <tag> [FULL_PREC=1]}"
TAG="${2:?missing tag}"
FULL_PREC="${3:-0}"
FP_ENV=()
RP_ENV=()
if [ "$FULL_PREC" = "1" ]; then FP_ENV=(FULL_PREC=1); RP_ENV=(BW24_FULL_PREC=1); fi
ROOT=~/bw24
EV=~/hqmtp-eval
OUT="$EV/results/$TAG"
mkdir -p "$OUT"
cd "$ROOT"

# stage held-out prompts where the battery's prompt_file resolver finds them
for f in "$EV"/prompts/pe*.txt; do cp "$f" research/e2e/prompts/; done

# 1. live battery on held-out prompts (+ 8-turn agent loop seeded from pe2)
env "${FP_ENV[@]}" ARM="$TAG" KS="1 2 3 4" N=3 NGEN=128 PROMPTS="pe1 pe2 pe3 pe4 pe5 pe6" \
  RUN_AGENTLOOP=1 SEED_PROMPT="$(ls research/e2e/prompts/pe2-*.txt | head -1)" \
  CORPUS_DIR="$OUT/corpus-capture" TIMEOUT=3600 \
  tools/acceptance_battery.sh "$MODEL" "$OUT/battery.jsonl" 2>&1 | tee "$OUT/battery.log"

# 2. teacher-forced replay on held-out corpora (fixed chunk — cross-run comparable)
for c in eval-code-64k eval-prose-64k eval-agentic-64k; do
  env "${RP_ENV[@]}" BW24_REPLAY_T=65536 BW24_REPLAY_K=4 BW24_REPLAY_STRIDE=16 BW24_REPLAY_CHUNK=512 \
    BW24_REPLAY_DUMP="$OUT/replay-$c.jsonl" BW24_PROMPT_FILE="$EV/corpora/$c.txt" \
    ./target/release/replay-acceptance "$MODEL" 2>&1 | tee "$OUT/replay-$c.log"
done

echo "EVAL DONE tag=$TAG -> $OUT"
