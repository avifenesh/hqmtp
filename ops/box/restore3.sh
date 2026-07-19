#!/usr/bin/env bash
# Resumable restore: every step skips if its output already exists (EBS-backed via
# nvme-links.sh, so spot-stops lose nothing). Safe to relaunch on every boot.
ST=~/hqmtp-distill/restore-status.txt
mark() { echo "$1 $(date -u +%H:%M)" >> $ST; }
cd ~/hqmtp-distill || exit 1
bash ~/hqmtp-distill/nvme-links.sh || exit 1
P=~/.venvs/distill/bin/python
SRV=~/bw24/target/release/bw24-server
M9=/data/ai-ml/hf-models/qwen35-9b-hf
M27=/data/ai-ml/hf-models/qwen36-27b-hf
D2=/opt/dlami/nvme/hqmtp-data2

start_server() {
  BW24_MODELS="hqmtp-9b=$1" BW24_ADDR=127.0.0.1:8014 nohup $SRV > ~/hqmtp-distill/server.log 2>&1 &
  echo $! > /tmp/bw24-server.pid
  for i in $(seq 1 90); do curl -s http://127.0.0.1:8014/health > /dev/null && return 0; sleep 5; done
  return 1
}
stop_server() { kill $(cat /tmp/bw24-server.pid) 2>/dev/null; sleep 5; }

need_docs() { [ "$(ls $1 2>/dev/null | wc -l)" -lt "$2" ]; }

if need_docs /opt/dlami/nvme/hqmtp-corpus 100; then
  $P fetch_corpus.py --out /opt/dlami/nvme/hqmtp-corpus --files-per-source 40 > restore-fetch.log 2>&1
  mark FETCH-DONE
fi

if need_docs /opt/dlami/nvme/hqmtp-gen2-corpus 1150; then
  start_server $M9 || { mark SERVER9-FAIL; exit 1; }
  mark SERVER9-UP
  $P gen_corpus2.py --out /opt/dlami/nvme/hqmtp-gen2-corpus --docs 1200 > restore-gen9.log 2>&1
  stop_server
  mark "GEN9-DONE n=$(ls /opt/dlami/nvme/hqmtp-gen2-corpus | wc -l)"
fi

if [ ! -f $D2/hot32k.pt ]; then
  $P build_hot_vocab.py --corpus-dir /opt/dlami/nvme/hqmtp-gen2-corpus --size 32768 \
    --out $D2/hot32k.pt > restore-hot.log 2>&1
  mark HOT32K-DONE
fi
[ -f frtrim-draft.gguf ]       || $P export_frtrim.py --model $M9 --hot $D2/hot32k.pt --out frtrim-draft.gguf > restore-exp.log 2>&1
[ -f frtrim-nvfp4-draft.gguf ] || $P export_frtrim.py --model $M9 --hot $D2/hot32k.pt --nvfp4 --out frtrim-nvfp4-draft.gguf >> restore-exp.log 2>&1
[ -f frtrim-mixed-draft.gguf ] || $P export_frtrim.py --model $M9 --hot $D2/hot32k.pt --mixed --out frtrim-mixed-draft.gguf >> restore-exp.log 2>&1
mark EXPORTS9-DONE
[ -d /tmp/longset-prompts ] && [ "$(ls /tmp/longset-prompts | wc -l)" -ge 20 ] || $P gen_longset.py > restore-longset.log 2>&1
if [ ! -d /tmp/mtbench-prompts ] || [ "$(ls /tmp/mtbench-prompts | wc -l)" -lt 80 ]; then
$P - << 'PYEOF' > restore-mtbench.log 2>&1
from datasets import load_dataset
import os
os.makedirs("/tmp/mtbench-prompts", exist_ok=True)
for n, ex in enumerate(load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")):
    q = ex["prompt"][0] if isinstance(ex["prompt"], list) else ex["prompt"]
    open(f"/tmp/mtbench-prompts/q{n:02d}.txt", "w").write(q)
PYEOF
fi
# archive prompt suites on EBS (identity-critical: suites must not silently change)
mkdir -p /data/hqmtp-live/prompt-suites
[ -f /data/hqmtp-live/prompt-suites/prompts.tgz ] || tar -czf /data/hqmtp-live/prompt-suites/prompts.tgz -C /tmp mtbench-prompts longset-prompts 2>/dev/null
mark PROMPTS-DONE
need_docs /opt/dlami/nvme/hqmtp-trueood-corpus 20 && $P build_trueood.py > restore-trueood.log 2>&1
mark TRUEOOD-DONE

if need_docs /opt/dlami/nvme/hqmtp-27b-corpus 550; then
  start_server $M27 || { mark SERVER27-FAIL; exit 1; }
  mark SERVER27-UP
  $P gen_corpus2.py --out /opt/dlami/nvme/hqmtp-27b-corpus --docs 600 > restore-gen27.log 2>&1
  stop_server
  mark "GEN27-DONE n=$(ls /opt/dlami/nvme/hqmtp-27b-corpus | wc -l)"
fi
if [ ! -f $D2/hot32k-27b.pt ]; then
  $P build_hot_vocab.py --corpus-dir /opt/dlami/nvme/hqmtp-27b-corpus --size 32768 \
    --out $D2/hot32k-27b.pt > restore-hot27.log 2>&1
fi
[ -f frtrim27-draft.gguf ]       || $P export_frtrim.py --model $M27 --hot $D2/hot32k-27b.pt --out frtrim27-draft.gguf >> restore-exp.log 2>&1
[ -f frtrim27-nvfp4-draft.gguf ] || $P export_frtrim.py --model $M27 --hot $D2/hot32k-27b.pt --nvfp4 --out frtrim27-nvfp4-draft.gguf >> restore-exp.log 2>&1
[ -f frtrim27-mixed-draft.gguf ] || $P export_frtrim.py --model $M27 --hot $D2/hot32k-27b.pt --mixed --out frtrim27-mixed-draft.gguf >> restore-exp.log 2>&1
mark EXPORTS27-DONE
cp ~/hqmtp-distill/*.gguf /data/hqmtp-live/ 2>/dev/null
mark RESTORE3-DONE
