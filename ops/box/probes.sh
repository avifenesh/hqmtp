#!/usr/bin/env bash
# Direction probes P1-P4: K-depth, 27B compose, zh re-rank, base-model audit
cd ~/hqmtp-distill
mark() { echo "$1 $(date -u +%H:%M)" >> probes-status.txt; }
M27=/data/ai-ml/hf-models/qwen36-27b-hf
M27Q=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp
M9=/data/ai-ml/hf-models/qwen35-9b-hf

# P1: speculation-depth sweep on cheap heads, 27B long (biggest-gain regime)
cd ~/bw24-sv
export BW24_CHAT=1 BW24_PROMPT_DIR=/tmp/longset-prompts BW24_NGEN=512
{
for K in 5 6 8; do
  for ARM in frtrim27-mixed-draft frtrim27-nvfp4-draft; do
    echo "##### 27b longset $ARM K=$K greedy"
    BW24_MTP_DRAFT=$HOME/hqmtp-distill/$ARM.gguf BW24_SPEC_K=$K target/release/run-spec $M27 2>/dev/null | grep -a SWEEP
  done
done
echo "##### 27b longset natural K=6 greedy (control)"
BW24_SPEC_K=6 target/release/run-spec $M27 2>/dev/null | grep -a SWEEP
} > ~/hqmtp-distill/p1-ksweep.log 2>&1
mark P1-DONE

# P2: 27B quantized-base composition
{
for SUITE in longset mtbench; do
  case $SUITE in
    longset) export BW24_PROMPT_DIR=/tmp/longset-prompts BW24_NGEN=512 ;;
    mtbench) export BW24_PROMPT_DIR=/tmp/mtbench-prompts BW24_NGEN=256 ;;
  esac
  echo "##### 27bQ $SUITE natural greedy"
  BW24_SPEC_K=4 target/release/run-spec $M27Q 2>/dev/null | grep -a SWEEP
  for ARM in frtrim27-draft frtrim27-nvfp4-draft frtrim27-mixed-draft; do
    echo "##### 27bQ $SUITE $ARM greedy"
    BW24_MTP_DRAFT=$HOME/hqmtp-distill/$ARM.gguf BW24_SPEC_K=4 target/release/run-spec $M27Q 2>/dev/null | grep -a SWEEP
  done
done
} > ~/hqmtp-distill/p2-compose27.log 2>&1
mark P2-DONE

# P3: zh vocabulary re-rank probe
cd ~/hqmtp-distill
~/.venvs/distill/bin/python p3_zh_rank.py > p3-zhrank.log 2>&1 || true
~/.venvs/distill/bin/python export_frtrim.py --model $M9 --hot /opt/dlami/nvme/hqmtp-data2/hot32k-zh.pt \
  --nvfp4 --out frtrim-zh-draft.gguf >> p3-zhrank.log 2>&1 || true
cd ~/bw24-sv
{
ZDOCS=$(ls /opt/dlami/nvme/hqmtp-trueood-corpus/wiki-zh-*.txt | head -4)
for ARM in "" "frtrim-nvfp4-draft" "frtrim-zh-draft"; do
  NAME=${ARM:-natural}
  for D in $ZDOCS; do
    echo "##### 9b zh-replay $NAME $(basename $D)"
    BW24_REPLAY_K=4 BW24_PROMPT_FILE=$D BW24_MTP_DRAFT=${ARM:+$HOME/hqmtp-distill/$ARM.gguf} \
      target/release/replay-acceptance $M9 2>/dev/null | grep -a -iE "accept|chain|slot"
  done
done
} > ~/hqmtp-distill/p3-zhreplay.log 2>&1
mark P3-DONE

# P4: base-model audit generality probe
cd ~/hqmtp-distill
~/.venvs/distill/bin/python p4_base_audit.py > p4-baseaudit.log 2>&1 || true
mark P4-DONE
mark PROBES-DONE
