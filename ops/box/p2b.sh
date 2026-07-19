#!/usr/bin/env bash
# P2 retry: 27B quantized-base composition, correct gguf path, exclusive GPU
cd ~/hqmtp-distill
mark() { echo "$1 $(date -u +%H:%M)" >> ~/hqmtp-distill/probes-status.txt; }
while ! grep -q "PROBES-DONE" ~/bw24-sv/probes-status.txt 2>/dev/null; do sleep 60; done
B=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
FR=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/mtp-Qwen3.6-27B-Q4_K_M-frspec32768.gguf
cd ~/bw24-sv
export BW24_CHAT=1 BW24_SPEC_K=4
{
for SUITE in longset mtbench; do
  case $SUITE in
    longset) export BW24_PROMPT_DIR=/tmp/longset-prompts BW24_NGEN=512 ;;
    mtbench) export BW24_PROMPT_DIR=/tmp/mtbench-prompts BW24_NGEN=256 ;;
  esac
  echo "##### 27bQ $SUITE builtin-mtp greedy"
  target/release/run-spec $B 2>/dev/null | grep -a SWEEP
  echo "##### 27bQ $SUITE frspec32768 greedy"
  BW24_MTP_DRAFT=$FR target/release/run-spec $B 2>/dev/null | grep -a SWEEP
  echo "##### 27bQ $SUITE frtrim27-nvfp4 greedy"
  BW24_MTP_DRAFT=$HOME/hqmtp-distill/frtrim27-nvfp4-draft.gguf target/release/run-spec $B 2>/dev/null | grep -a SWEEP
  echo "##### 27bQ $SUITE frtrim27-mixed greedy"
  BW24_MTP_DRAFT=$HOME/hqmtp-distill/frtrim27-mixed-draft.gguf target/release/run-spec $B 2>/dev/null | grep -a SWEEP
done
echo "##### 27bQ longset frtrim27-mixed K=5 greedy"
export BW24_PROMPT_DIR=/tmp/longset-prompts BW24_NGEN=512
BW24_MTP_DRAFT=$HOME/hqmtp-distill/frtrim27-mixed-draft.gguf BW24_SPEC_K=5 target/release/run-spec $B 2>/dev/null | grep -a SWEEP
} > ~/hqmtp-distill/p2b-compose27.log 2>&1
mark P2B-DONE
