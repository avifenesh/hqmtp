#!/usr/bin/env python3
"""Hot-32k coverage per domain of the true-OOD suite, plus in-dist baseline."""
import glob
import json
import os

import torch
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("/data/ai-ml/hf-models/qwen35-9b-hf")
hot = set(torch.load("/opt/dlami/nvme/hqmtp-data2/hot32k.pt",
                     weights_only=False)["hot_ids"].tolist())

out = {}
groups = {}
for f in sorted(glob.glob("/opt/dlami/nvme/hqmtp-trueood-corpus/*.txt")):
    dom = os.path.basename(f).rsplit("-", 1)[0]
    groups.setdefault(dom, []).append(f)
groups["INDIST"] = sorted(glob.glob("/opt/dlami/nvme/hqmtp-gen2-corpus/*.txt"))[:8]

for dom, files in sorted(groups.items()):
    tot = hit = 0
    for f in files:
        for i in tok(open(f).read()).input_ids:
            tot += 1
            hit += i in hot
    out[dom] = {"coverage": hit / max(tot, 1), "n_tokens": tot}
    print(f"{dom:10s} coverage {hit / max(tot, 1):.4f}  ({tot} tokens)", flush=True)

json.dump(out, open(os.path.expanduser("~/hqmtp-distill/hotcov.json"), "w"))
