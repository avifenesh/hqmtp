#!/usr/bin/env bash
# Idempotent: back all hqmtp working dirs with EBS (/data) and expose them at the
# legacy NVMe paths via symlinks. NVMe is wiped on every spot-stop; EBS survives.
set -e
sudo mkdir -p /data/hqmtp-live
sudo chown ubuntu /data/hqmtp-live
sudo mkdir -p /opt/dlami/nvme
sudo chown ubuntu /opt/dlami/nvme 2>/dev/null || true
for d in hqmtp-corpus hqmtp-gen2-corpus hqmtp-27b-corpus hqmtp-trueood-corpus \
         hqmtp-data2 hqmtp-gen2-data hqmtp-gen2-eval hqmtp-ood-data hqmtp-ood-stream \
         hqmtp-trueood-data hqmtp-27b-corpus hqmtp-27b-data hqmtp-27b-evaldata; do
  mkdir -p /data/hqmtp-live/$d
  if [ -e /opt/dlami/nvme/$d ] && [ ! -L /opt/dlami/nvme/$d ]; then
    # real dir with data (fresh NVMe regen) — merge into EBS then replace
    rsync -a /opt/dlami/nvme/$d/ /data/hqmtp-live/$d/ && rm -rf /opt/dlami/nvme/$d
  fi
  ln -sfn /data/hqmtp-live/$d /opt/dlami/nvme/$d
done
echo "nvme-links: all hqmtp dirs EBS-backed"
