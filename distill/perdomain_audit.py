#!/usr/bin/env python3
"""Per-domain audit tallies on the true-OOD suite, for leave-one-domain-out
robustness of the Spearman ranking (reviewer round-2). Reuses audit_lab arms."""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/hqmtp-distill"))
from audit_lab import make_arms, forward_arm, load_docs
from qwen_mtp import TeacherHead
import utilization_probe as up
from transformers import AutoTokenizer

OOD_DIR = "/opt/dlami/nvme/hqmtp-trueood-data"
OUT = os.path.expanduser("~/hqmtp-distill/audit-perdomain.json")


def main():
    tok = AutoTokenizer.from_pretrained("/data/ai-ml/hf-models/qwen35-9b-hf")
    T = TeacherHead(interleaved_rope=True)
    dev = "cuda"
    hot = torch.load("/opt/dlami/nvme/hqmtp-data2/hot32k.pt", weights_only=False)["hot_ids"]
    t2d = torch.full((T.lm_head.shape[0],), -1, dtype=torch.long, device=dev)
    t2d[hot.long().to(dev)] = torch.arange(len(hot), device=dev)

    cal = load_docs("/opt/dlami/nvme/hqmtp-gen2-data/*.hiddens.meta.json",
                    "/opt/dlami/nvme/hqmtp-gen2-corpus", 12, tok)
    arms = make_arms(T, hot, cal)

    metas = sorted(glob.glob(f"{OOD_DIR}/*.hiddens.meta.json"))
    domains = {}
    for mp in metas:
        dom = os.path.basename(mp).rsplit("-", 1)[0]
        domains.setdefault(dom, []).append(mp)

    results = {}
    for dom, mps in sorted(domains.items()):
        docs = []
        for mp in mps:
            base = mp[: -len(".hiddens.meta.json")]
            from hdata import Hiddens
            h = Hiddens(base)
            txt = base + ".txt"
            if not os.path.exists(txt):
                txt = os.path.join("/opt/dlami/nvme/hqmtp-trueood-corpus",
                                   os.path.basename(base) + ".txt")
            meta = json.load(open(mp))
            ids = tok(open(txt).read()).input_ids[: h.n]
            docs.append((ids, h, meta["bg"]))
        per_arm = {a: {"acc": 0, "n": 0} for a in arms}
        for embq, hq, pq, ck, cv, mask, tgt in up.batches(docs, 800, 24, dev, T):
            tgt_d = t2d[tgt]
            valid = tgt_d >= 0
            for a, arm in arms.items():
                lg = forward_arm(T, arm, embq, hq, pq, ck, cv, mask)
                hit = ((lg.argmax(-1) == tgt_d) & valid).sum().item()
                per_arm[a]["acc"] += hit
                per_arm[a]["n"] += tgt_d.shape[0]
        results[dom] = per_arm
        print(dom, {a: round(v["acc"] / max(v["n"], 1), 4) for a, v in per_arm.items()},
              flush=True)
    json.dump(results, open(OUT, "w"))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
