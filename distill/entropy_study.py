#!/usr/bin/env python3
"""NVFP4 sampled-stabilization mechanism study (paper appendix).
Measures draft-distribution statistics of Q8 vs NVFP4 vs FP heads at identical
states: softmax entropy, top-1 margin, effective temperature proxy.
Hypothesis: 4-bit weight noise flattens sharp wrong modes / raises entropy
slightly, producing more-typical sampled trajectories (observed: 6x tighter
seed variance)."""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/hqmtp-distill"))
from qwen_mtp import TeacherHead
from audit_lab import make_arms, forward_arm, load_docs
import utilization_probe as up


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/data/ai-ml/hf-models/qwen35-9b-hf")
    load_docs.tok = tok
    T = TeacherHead(interleaved_rope=True)
    dev = "cuda"
    hot = torch.load("/opt/dlami/nvme/hqmtp-data2/hot32k.pt", weights_only=False)["hot_ids"]
    cal = load_docs("/opt/dlami/nvme/hqmtp-gen2-data/*.hiddens.meta.json",
                    "/opt/dlami/nvme/hqmtp-gen2-corpus", 12, tok)
    docs = load_docs("/opt/dlami/nvme/hqmtp-gen2-eval/*.hiddens.meta.json", "", 12, tok)
    arms = make_arms(T, hot, cal)
    keep = {k: arms[k] for k in ["reference", "q8", "nvfp4"]}

    stats = {a: {"H": [], "margin": [], "kl_ref": []} for a in keep}
    for embq, hq, pq, ck, cv, m, tgt in up.batches(docs, 3000, 24, dev, T):
        ref_lp = None
        for a, arm in keep.items():
            lg = forward_arm(T, arm, embq, hq, pq, ck, cv, m)
            lp = F.log_softmax(lg, dim=-1)
            p = lp.exp()
            H = -(p * lp).sum(-1)
            top2 = lg.topk(2, dim=-1).values
            margin = (top2[:, 0] - top2[:, 1])
            stats[a]["H"] += H.cpu().tolist()
            stats[a]["margin"] += margin.cpu().tolist()
            if a == "reference":
                ref_lp = lp
            else:
                kl = (ref_lp.exp() * (ref_lp - lp)).sum(-1)
                stats[a]["kl_ref"] += kl.cpu().tolist()
    for a, d in stats.items():
        H = np.array(d["H"]); mg = np.array(d["margin"])
        line = (f"{a:10} entropy mean {H.mean():.4f} (p90 {np.percentile(H,90):.3f}) | "
                f"top1-margin mean {mg.mean():.3f} median {np.median(mg):.3f}")
        if d["kl_ref"]:
            kl = np.array(d["kl_ref"])
            line += f" | KL(ref||arm) mean {kl.mean():.5f} p99 {np.percentile(kl,99):.4f}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
