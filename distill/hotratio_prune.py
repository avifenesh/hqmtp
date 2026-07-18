#!/usr/bin/env python3
"""Trim-aligned pruning: rank FFN channels by their HOT-logit energy ratio
r_i = ||lm_head[hot] @ w_down[:,i]||^2 / ||lm_head @ w_down[:,i]||^2  (weights only,
data-free). Channels with low r_i mostly moved logits the vocab trim already
deleted -> prunable BY CONSTRUCTION for the trimmed draft. Compare vs mass-ranked
pruning on in-dist eval AND the OOD live stream (slot0), plus a combined score.
"""
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.expanduser("~/hqmtp-distill"))
from qwen_mtp import TeacherHead
from hdata import Hiddens
from transformers import AutoTokenizer
import utilization_probe as up

KEEP = 3072


def load_docs(pat, n):
    tok = load_docs.tok
    out = []
    for mp in sorted(glob.glob(pat))[:n]:
        base = mp[: -len(".hiddens.meta.json")]
        meta = json.load(open(mp))
        h = Hiddens(base)
        txt = base + ".txt"
        if not os.path.exists(txt):
            import os.path as _p
            txt = "/opt/dlami/nvme/hqmtp-gen2-corpus/" + _p.basename(base) + ".txt"
        ids = tok(open(txt).read()).input_ids[: h.n]
        out.append((ids, h, meta["bg"]))
    return out


def main():
    tok = AutoTokenizer.from_pretrained("/data/ai-ml/hf-models/qwen35-9b-hf")
    load_docs.tok = tok
    T = TeacherHead(interleaved_rope=True)
    dev = "cuda"
    hot = torch.load("/opt/dlami/nvme/hqmtp-data2/hot32k.pt",
                     weights_only=False)["hot_ids"].to(dev)

    # hot-energy ratio per channel (weights only). Chunk over channels for memory.
    L = T.lm_head.float()                      # [V, d]
    Wd = T.w_down.float()                      # [d, d_ff]
    hot_mask = torch.zeros(L.shape[0], dtype=torch.bool, device=dev)
    hot_mask[hot.long()] = True
    r = torch.empty(Wd.shape[1])
    for s in range(0, Wd.shape[1], 1024):
        e = min(s + 1024, Wd.shape[1])
        z = L @ Wd[:, s:e]                     # [V, chunk]
        tot = z.pow(2).sum(0)
        hotE = z[hot_mask].pow(2).sum(0)
        r[s:e] = (hotE / tot.clamp(min=1e-12)).cpu()
    print(f"hot-ratio: median {r.median():.3f}  p10 {r.quantile(0.1):.3f}  "
          f"p90 {r.quantile(0.9):.3f}")

    indist = load_docs("/opt/dlami/nvme/hqmtp-gen2-eval/*.hiddens.meta.json", 12)
    ood = load_docs("/opt/dlami/nvme/hqmtp-ood-data/*.hiddens.meta.json", 24)

    # mass stats from corpus calibration (as before)
    cal = load_docs("/opt/dlami/nvme/hqmtp-gen2-data/*.hiddens.meta.json", 12)
    st = {"ffn": torch.zeros(T.w_gate.shape[0]), "head": torch.zeros(T.n_q)}
    up.evaluate(T, cal, 3000, dev, stats=st)
    mass = st["ffn"] / st["ffn"].sum()

    ranks = {
        "mass": mass.argsort(descending=True)[:KEEP],
        "hot-ratio": r.argsort(descending=True)[:KEEP],
        "mass*ratio": (mass * r).argsort(descending=True)[:KEEP],
    }

    def ev(docs, keep):
        return up.evaluate(T, docs, 2500, dev,
                           ffn_keep=keep.to(dev) if keep is not None else None)

    print(f"{'variant':12} {'in-dist':>8} {'OOD':>8}")
    fi, fo = ev(indist, None), ev(ood, None)
    print(f"{'full':12} {fi:8.4f} {fo:8.4f}")
    for name, keep in ranks.items():
        i, o = ev(indist, keep), ev(ood, keep)
        print(f"{name:12} {i:8.4f} ({i/fi*100:.1f}%) {o:8.4f} ({o/fo*100:.1f}%)",
              flush=True)


if __name__ == "__main__":
    main()
