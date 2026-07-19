#!/usr/bin/env python3
"""P4: does the harm/disagreement audit generalize from draft heads to the BASE model?
Compress the base LM's FFNs (q4 roundtrip vs prune25 vs svd) and measure teacher-forced
next-token accuracy + top-1 agreement vs the uncompressed base, on in-dist and true-shift text.
Cheap probe: last 12 layers only (keeps 3 model variants in memory as weight deltas)."""
import glob
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "/data/ai-ml/hf-models/qwen35-9b-hf"
IND = sorted(glob.glob("/opt/dlami/nvme/hqmtp-gen2-corpus/*.txt"))[:6]
OOD = sorted(glob.glob("/opt/dlami/nvme/hqmtp-trueood-corpus/*.txt"))


def q4_roundtrip(w):
    # Q4_0-style: blocks of 32, symmetric scale
    x = w.float().view(-1, 32)
    s = x.abs().amax(1, keepdim=True) / 7.0
    q = torch.clamp(torch.round(x / s.clamp(min=1e-12)), -8, 7)
    return (q * s).view(w.shape).to(w.dtype)


def main():
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()
    layers = model.model.layers if hasattr(model, "model") else model.transformer.h
    tail = layers[-12:]

    # collect calibration activation mass for prune ranking (cal docs, down_proj input)
    mass = {}
    hooks = []
    for i, l in enumerate(tail):
        hooks.append(l.mlp.down_proj.register_forward_pre_hook(
            lambda mod, inp, i=i: mass.__setitem__(i, mass.get(i, 0) + inp[0].float().abs().mean(dim=(0, 1)).cpu())))
    cal = sorted(glob.glob("/opt/dlami/nvme/hqmtp-gen-corpus/*.txt"))[:12] or IND[:4]
    with torch.no_grad():
        for f in cal[:8]:
            ids = tok(open(f).read(), return_tensors="pt", truncation=True,
                      max_length=1024).input_ids.cuda()
            model(ids)
    for h in hooks:
        h.remove()

    orig = {}
    for i, l in enumerate(tail):
        orig[i] = {n: getattr(l.mlp, n).weight.data.clone()
                   for n in ("gate_proj", "up_proj", "down_proj")}

    def restore():
        for i, l in enumerate(tail):
            for n, w in orig[i].items():
                getattr(l.mlp, n).weight.data.copy_(w)

    def apply_q4():
        for i, l in enumerate(tail):
            for n in ("gate_proj", "up_proj", "down_proj"):
                getattr(l.mlp, n).weight.data.copy_(q4_roundtrip(orig[i][n]))

    def apply_prune25():
        for i, l in enumerate(tail):
            m = mass[i]
            keep = torch.argsort(m, descending=True)[: int(len(m) * 0.25)]
            mask = torch.zeros(len(m), dtype=torch.bool)
            mask[keep] = True
            mask = mask.cuda()
            l.mlp.down_proj.weight.data[:, ~mask] = 0
            l.mlp.gate_proj.weight.data[~mask, :] = 0
            l.mlp.up_proj.weight.data[~mask, :] = 0

    def apply_svd():
        for i, l in enumerate(tail):
            w = orig[i]["down_proj"].float()
            k = int(min(w.shape) * 0.35)
            u, s, v = torch.svd_lowrank(w, q=k)
            l.mlp.down_proj.weight.data.copy_((u @ torch.diag(s) @ v.T).to(w.dtype if w.dtype != torch.float32 else torch.bfloat16))

    def evaluate(files, ref_preds=None):
        hits = agr = tot = 0
        preds_all = []
        with torch.no_grad():
            for fi, f in enumerate(files):
                ids = tok(open(f).read(), return_tensors="pt", truncation=True,
                          max_length=1024).input_ids.cuda()
                lg = model(ids).logits[0, :-1]
                pred = lg.argmax(-1)
                tgt = ids[0, 1:]
                hits += (pred == tgt).sum().item()
                tot += len(tgt)
                if ref_preds is not None:
                    agr += (pred == ref_preds[fi]).sum().item()
                preds_all.append(pred)
        return hits, tot, agr, preds_all

    results = {}
    arms = [("reference", restore), ("q4", apply_q4), ("prune25", apply_prune25),
            ("svd35", apply_svd)]
    ref = {}
    for name, fn in arms:
        restore()
        fn() if name != "reference" else None
        r = {}
        for cname, files in (("indist", IND), ("ood", OOD)):
            rp = ref.get(cname)
            hits, tot, agr, preds = evaluate(files, rp)
            if name == "reference":
                ref[cname] = preds
            r[cname] = {"acc": hits / tot, "agree": (agr / tot) if rp else 1.0, "n": tot}
        results[name] = r
        ra = results["reference"]
        for cname in ("indist", "ood"):
            d = r[cname]
            ret = d["acc"] / ra[cname]["acc"]
            print(f"{name:10s} {cname:6s} acc {d['acc']:.4f} agree {d['agree']:.4f} "
                  f"retention {ret:.4f}", flush=True)
        # harm per disagreement on indist
        if name != "reference":
            dis = 1 - r["indist"]["agree"]
            harm_n = ra["indist"]["acc"] - r["indist"]["acc"]
            print(f"{name:10s} INDIST disagreement {dis:.4f} harm/dis "
                  f"{harm_n / max(dis, 1e-9):.3f}", flush=True)
    restore()
    json.dump(results, open(os.path.expanduser("~/hqmtp-distill/base-audit.json"), "w"))
    print("wrote base-audit.json")


if __name__ == "__main__":
    main()
