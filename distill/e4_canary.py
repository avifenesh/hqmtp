#!/usr/bin/env python3
"""E4 — divergence-tail (canary) pruning calibration: causal test + practical fix.

1. Find canary positions: in-dist tokens where prune25 diverges from the full head.
2. Score PRUNED channels by their activation mass AT CANARY POSITIONS ONLY.
3. Restore a fixed budget (+5% of d_ff) of pruned channels two ways:
     (a) canary-conditioned mass rank   (b) global mass rank (control)
4. Evaluate both restored masks in-dist and OOD.
Prediction: canary restoration recovers disproportionately more OOD acceptance.
"""
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/hqmtp-distill"))
from qwen_mtp import TeacherHead
from train import rms
from hdata import Hiddens
import utilization_probe as up
from audit_lab import forward_arm, load_docs

KEEP = 3072
RESTORE = 614          # ~5% of 12288


@torch.no_grad()
def channel_mass_at(T, docs, npos, positions_filter, dev="cuda"):
    """Accumulate per-channel |act| over positions where filter(batch_hits) True.
    filter receives (pred_full, pred_pruned) and returns bool mask."""
    acc = torch.zeros(T.w_gate.shape[0])
    seen = 0
    mask25 = channel_mass_at.mask25
    lm = channel_mass_at.lm
    for embq, hq, pq, ck, cv, m, tgt in up.batches(docs, npos, 24, dev, T):
        ref_arm = {"lm": lm, "ov": {}, "mask": None}
        pr_arm = {"lm": lm, "ov": {}, "mask": mask25}
        lg_f = forward_arm(T, ref_arm, embq, hq, pq, ck, cv, m)
        lg_p = forward_arm(T, pr_arm, embq, hq, pq, ck, cv, m)
        sel = positions_filter(lg_f.argmax(-1), lg_p.argmax(-1))
        if sel.sum() == 0:
            continue
        # recompute act at selected positions with FULL head (stats)
        x = torch.cat((rms(embq[sel].to(T.dtype_), T.enorm, T.eps),
                       rms(hq[sel].to(T.dtype_), T.hnorm, T.eps)), -1) @ T.fc.T
        z = rms(x, T.post_ln, T.eps)          # attention-free proxy consistent w/ ranking use
        act = (F.silu(z @ T.w_gate.T) * (z @ T.w_up.T)).float().abs()
        acc += act.sum(0).cpu()
        seen += int(sel.sum())
    return acc, seen


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/data/ai-ml/hf-models/qwen35-9b-hf")
    T = TeacherHead(interleaved_rope=True)
    dev = "cuda"
    hot = torch.load("/opt/dlami/nvme/hqmtp-data2/hot32k.pt",
                     weights_only=False)["hot_ids"]
    lm = T.lm_head[hot.to(dev).long()].clone()
    load_docs.tok = tok

    cal = load_docs("/opt/dlami/nvme/hqmtp-gen2-data/*.hiddens.meta.json",
                    "/opt/dlami/nvme/hqmtp-gen2-corpus", 12, tok)
    indist = load_docs("/opt/dlami/nvme/hqmtp-gen2-eval/*.hiddens.meta.json", "", 12, tok)
    ood = load_docs("/opt/dlami/nvme/hqmtp-ood-data/*.hiddens.meta.json", "", 24, tok)

    st = {"ffn": torch.zeros(T.w_gate.shape[0]), "head": torch.zeros(T.n_q)}
    up.evaluate(T, cal, 3000, dev, stats=st)
    order = st["ffn"].argsort(descending=True)
    keep25 = order[:KEEP]
    mask25 = torch.zeros(T.w_gate.shape[0], dtype=torch.bool, device=dev)
    mask25[keep25.to(dev)] = True
    channel_mass_at.mask25 = mask25
    channel_mass_at.lm = lm

    canary_mass, n_canary = channel_mass_at(
        T, cal, 4000, lambda f, p: (f != p))
    print(f"[canary] {n_canary} divergence positions on calibration stream")

    pruned = order[KEEP:]                      # channels currently cut
    global_rank_restore = pruned[:RESTORE]     # next-best by global mass (control)
    cm = canary_mass.clone()
    keep_set = torch.zeros(T.w_gate.shape[0], dtype=torch.bool)
    keep_set[keep25] = True
    cm[keep_set] = -1
    canary_restore = cm.argsort(descending=True)[:RESTORE]

    ov = len(set(canary_restore.tolist()) & set(global_rank_restore.tolist()))
    print(f"[restore] overlap canary-vs-global restore sets: {ov}/{RESTORE}")

    def make_mask(extra):
        m = mask25.clone()
        m[extra.to(dev)] = True
        return m

    arms = {
        "full": {"lm": lm, "ov": {}, "mask": None},
        "prune25": {"lm": lm, "ov": {}, "mask": mask25},
        "restore-global": {"lm": lm, "ov": {}, "mask": make_mask(global_rank_restore)},
        "restore-canary": {"lm": lm, "ov": {}, "mask": make_mask(canary_restore)},
    }
    for cname, docs in [("indist", indist), ("ood", ood)]:
        accs = {a: [0, 0] for a in arms}
        t2d = torch.full((T.lm_head.shape[0],), -1, dtype=torch.long, device=dev)
        t2d[hot.to(dev).long()] = torch.arange(len(hot), device=dev)
        for embq, hq, pq, ck, cv, m, tgt in up.batches(docs, 2500, 24, dev, T):
            tgt_d = t2d[tgt]
            valid = tgt_d >= 0
            for a, arm in arms.items():
                lg = forward_arm(T, arm, embq, hq, pq, ck, cv, m)
                accs[a][0] += int(((lg.argmax(-1) == tgt_d) & valid).sum())
                accs[a][1] += len(tgt_d)
        line = " | ".join(f"{a} {h/n:.4f}" for a, (h, n) in accs.items())
        print(f"[{cname}] {line}", flush=True)


if __name__ == "__main__":
    main()
