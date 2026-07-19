#!/usr/bin/env python3
"""P3: build a zh-ranked hot32k from fresh zh wiki docs (disjoint from the eval docs),
for the dynamic-vocabulary-re-rank probe."""
import json
import os

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("/data/ai-ml/hf-models/qwen35-9b-hf")
eval_titles = set()
for f in os.listdir("/opt/dlami/nvme/hqmtp-trueood-corpus"):
    if f.startswith("wiki-zh") and f.endswith(".txt"):
        eval_titles.add(open(f"/opt/dlami/nvme/hqmtp-trueood-corpus/{f}").read()[:80])

cnt = torch.zeros(248320, dtype=torch.long)
ds = load_dataset("wikimedia/wikipedia", "20231101.zh", split="train", streaming=True)
used = 0
for ex in ds:
    text = ex["text"][:20000]
    if len(text) < 3000 or text[:80] in eval_titles:
        continue
    for i in tok(text).input_ids:
        cnt[i] += 1
    used += 1
    if used >= 40:
        break

old = torch.load("/opt/dlami/nvme/hqmtp-data2/hot32k.pt", weights_only=False)
specials = old["hot_ids"][-2000:]  # keep tail (specials live in original ranking set)
top = torch.argsort(cnt, descending=True)[:32768]
# ensure special tokens present: union then trim back to 32768 by zh rank
hot = torch.unique(torch.cat([top[:31000], old["hot_ids"][:1768].long()]))[:32768]
hot = hot.to(old["hot_ids"].dtype)
torch.save({"hot_ids": hot}, "/opt/dlami/nvme/hqmtp-data2/hot32k-zh.pt")
cov = cnt[hot.long()].sum().item() / max(cnt.sum().item(), 1)
print(f"zh hot32k built from {used} docs; self-coverage {cov:.4f}")
