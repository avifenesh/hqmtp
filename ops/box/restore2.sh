#!/usr/bin/env bash
# Restore chain v2: rebuilt server, EBS persistence of everything regenerated
ST=~/hqmtp-distill/restore-status.txt
mark() { echo "$1 $(date -u +%H:%M)" >> $ST; }
cd ~/hqmtp-distill
P=~/.venvs/distill/bin/python
SRV=~/bw24/target/release/bw24-server
M9=/data/ai-ml/hf-models/qwen35-9b-hf
M27=/data/ai-ml/hf-models/qwen36-27b-hf

start_server() {
  BW24_MODELS="hqmtp-9b=$1" BW24_ADDR=127.0.0.1:8014 nohup $SRV > ~/hqmtp-distill/server.log 2>&1 &
  echo $! > /tmp/bw24-server.pid
  for i in $(seq 1 90); do curl -s http://127.0.0.1:8014/health > /dev/null && return 0; sleep 5; done
  return 1
}
stop_server() { kill $(cat /tmp/bw24-server.pid) 2>/dev/null; sleep 5; }

start_server $M9 || { mark SERVER9-FAIL-V2; exit 1; }
mark SERVER9-UP
$P gen_corpus2.py --out /opt/dlami/nvme/hqmtp-gen2-corpus --docs 1200 > restore-gen9.log 2>&1
stop_server
N=$(ls /opt/dlami/nvme/hqmtp-gen2-corpus | wc -l); mark "GEN9-DONE n=$N"
[ "$N" -lt 50 ] && { mark GEN9-TOO-SMALL; exit 1; }
$P build_hot_vocab.py --corpus-dir /opt/dlami/nvme/hqmtp-gen2-corpus --size 32768 \
  --out /opt/dlami/nvme/hqmtp-data2/hot32k.pt > restore-hot.log 2>&1
mark HOT32K-DONE
$P export_frtrim.py --model $M9 --hot /opt/dlami/nvme/hqmtp-data2/hot32k.pt --out frtrim-draft.gguf > restore-exp.log 2>&1
$P export_frtrim.py --model $M9 --hot /opt/dlami/nvme/hqmtp-data2/hot32k.pt --nvfp4 --out frtrim-nvfp4-draft.gguf >> restore-exp.log 2>&1
$P export_frtrim.py --model $M9 --hot /opt/dlami/nvme/hqmtp-data2/hot32k.pt --mixed --out frtrim-mixed-draft.gguf >> restore-exp.log 2>&1
mark EXPORTS9-DONE
$P gen_longset.py > restore-longset.log 2>&1
mark LONGSET-DONE

start_server $M27 || { mark SERVER27-FAIL-V2; exit 1; }
mark SERVER27-UP
$P gen_corpus2.py --out /opt/dlami/nvme/hqmtp-27b-corpus --docs 600 > restore-gen27.log 2>&1
stop_server
mark GEN27-DONE
$P build_hot_vocab.py --corpus-dir /opt/dlami/nvme/hqmtp-27b-corpus --size 32768 \
  --out /opt/dlami/nvme/hqmtp-data2/hot32k-27b.pt > restore-hot27.log 2>&1
$P export_frtrim.py --model $M27 --hot /opt/dlami/nvme/hqmtp-data2/hot32k-27b.pt --out frtrim27-draft.gguf >> restore-exp.log 2>&1
$P export_frtrim.py --model $M27 --hot /opt/dlami/nvme/hqmtp-data2/hot32k-27b.pt --nvfp4 --out frtrim27-nvfp4-draft.gguf >> restore-exp.log 2>&1
$P export_frtrim.py --model $M27 --hot /opt/dlami/nvme/hqmtp-data2/hot32k-27b.pt --mixed --out frtrim27-mixed-draft.gguf >> restore-exp.log 2>&1
mark EXPORTS27-DONE

# persist everything regenerated to EBS (survives spot-stop; NVMe does not)
B=/data/hqmtp-backup/$(date -u +%Y%m%dT%H%M)
sudo mkdir -p $B && sudo chown ubuntu $B
tar -C /opt/dlami/nvme -czf $B/corpora.tgz hqmtp-corpus hqmtp-gen2-corpus hqmtp-27b-corpus hqmtp-trueood-corpus hqmtp-data2 2>/dev/null
cp ~/hqmtp-distill/*.gguf $B/ 2>/dev/null
tar -czf $B/prompts.tgz -C /tmp mtbench-prompts longset-prompts 2>/dev/null
cp ~/hqmtp-distill/*.json ~/hqmtp-distill/restore-*.log $B/ 2>/dev/null
mark "PERSIST-DONE $B"
mark RESTORE2-DONE
