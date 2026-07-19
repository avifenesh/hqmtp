#!/usr/bin/env bash
# round-2 review response chain: per-domain audit + hot coverage + mixed full grid
set -x
cd ~/hqmtp-distill
mark() { echo "$1 $(date -u +%H:%M)" >> r2-status.txt; }

~/.venvs/distill/bin/python perdomain_audit.py > perdomain.log 2>&1 || true
mark PERDOMAIN-DONE
~/.venvs/distill/bin/python hotcov.py > hotcov.log 2>&1 || true
mark HOTCOV-DONE

# mixed arm: complete 9B sampled aggregate (seeds 17, 27; s7 done in mixed-arm.log)
cd ~/bw24-sv
M=/data/ai-ml/hf-models/qwen35-9b-hf
{
for SUITE in mtbench longset; do
  case $SUITE in
    mtbench) export BW24_PROMPT_DIR=/tmp/mtbench-prompts BW24_NGEN=256 ;;
    longset) export BW24_PROMPT_DIR=/tmp/longset-prompts BW24_NGEN=512 ;;
  esac
  for S in 17 27; do
    echo "##### 9b $SUITE mixed t0.7s$S"
    BW24_MTP_DRAFT=$HOME/hqmtp-distill/frtrim-mixed-draft.gguf BW24_SPEC_TEMP=0.7 BW24_SEED=$S \
      target/release/run-spec $M 2>/dev/null | grep -a SWEEP
  done
done
} > ~/hqmtp-distill/mixed9b-seeds.log 2>&1
mark MIXED9B-DONE

# 27B mixed export + full grid
M27=/data/ai-ml/hf-models/qwen36-27b-hf
cd ~/hqmtp-distill
~/.venvs/distill/bin/python export_frtrim.py --model $M27 --hot /opt/dlami/nvme/hqmtp-data2/hot32k-27b.pt \
  --mixed --out frtrim27-mixed-draft.gguf > mixed27-export.log 2>&1
cd ~/bw24-sv
{
for SUITE in mtbench longset; do
  case $SUITE in
    mtbench) export BW24_PROMPT_DIR=/tmp/mtbench-prompts BW24_NGEN=256 ;;
    longset) export BW24_PROMPT_DIR=/tmp/longset-prompts BW24_NGEN=512 ;;
  esac
  echo "##### 27b $SUITE mixed greedy"
  BW24_MTP_DRAFT=$HOME/hqmtp-distill/frtrim27-mixed-draft.gguf target/release/run-spec $M27 2>/dev/null | grep -a SWEEP
  for S in 7 17 27; do
    echo "##### 27b $SUITE mixed t0.7s$S"
    BW24_MTP_DRAFT=$HOME/hqmtp-distill/frtrim27-mixed-draft.gguf BW24_SPEC_TEMP=0.7 BW24_SEED=$S \
      target/release/run-spec $M27 2>/dev/null | grep -a SWEEP
  done
done
} > ~/hqmtp-distill/mixed27.log 2>&1
mark R2CHAIN-DONE
