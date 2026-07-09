#!/usr/bin/env python3
"""Slot-0 replay acceptance eval for distilled heads (torch-side, windowed).

Scores, on held-out eval corpora (engine-extracted hiddens + bg):
  engine   — the engine's own chain drafts (from the replay dump, full context)
  teacher  — torch TeacherHead, windowed batched step-0 (same window as students)
  <ckpts>  — student checkpoints

Usage:
  eval_replay.py --data /opt/dlami/nvme/hqmtp-distill-data/eval-gen-dist-9b \
      --corpus ~/hqmtp-eval/corpora/gen-dist-9b.txt --window 2048 \
      --ckpt checkpoints/s2-kd.pt checkpoints/s2-b0.pt
"""
import argparse
import json
import os

import numpy as np
import torch

from qwen_mtp import TeacherHead
from heads import StudentS1, StudentS2, StudentSV
from train import batched_teacher_step0, teacher_ctx_rows


def load(base):
    meta = json.load(open(base + ".hiddens.meta.json"))
    from hdata import Hiddens
    hid = Hiddens(base)
    rows = [json.loads(l) for l in open(base + ".drafts.jsonl")]
    return meta, hid, rows


@torch.no_grad()
def eval_model(kind, model, T, ids, hid, bg, ps, W, B=48):
    """kind: teacher|s1|s2. Returns slot-0 hit rate vs bg at positions ps."""
    dev = "cuda"
    hits = 0
    embed = T.embed
    for s in range(0, len(ps), B):
        batch = ps[s: s + B]
        bsz = len(batch)
        pt = torch.tensor(batch)
        lo = (pt - W).clamp(min=0)
        idx = lo[:, None] + torch.arange(W)[None, :]
        valid = (idx < pt[:, None]).to(dev)
        idx = idx.clamp(max=len(ids) - 1)
        ctx_tok = torch.tensor(ids)[idx].to(dev)
        ctx_hprev = torch.from_numpy(hid[(idx - 1).clamp(min=0).numpy()]).float().to(dev)
        ctx_hprev[(idx == 0).to(dev)] = 0.0
        ctx_pos = (idx + 1).to(dev)
        tok_q = torch.tensor([ids[p] for p in batch]).to(dev)
        hprev_q = torch.from_numpy(hid[[p - 1 for p in batch]]).float().to(dev)
        pos_q = (pt + 1).to(dev)
        tgt = torch.tensor([bg[p + 1] for p in batch]).to(dev)
        emb_q = embed[tok_q].float()
        ctx_emb = embed[ctx_tok].float()
        if kind == "teacher":
            tk, tv = teacher_ctx_rows(T, ctx_emb.to(T.dtype_), ctx_hprev.to(T.dtype_),
                                      ctx_pos)
            logits, _ = batched_teacher_step0(T, emb_q.to(T.dtype_),
                                              hprev_q.to(T.dtype_), pos_q, tk, tv, valid)
        elif kind == "s1":
            pre, _ = model(emb_q, hprev_q, None)
            logits = (pre.to(T.lm_head.dtype) @ T.lm_head.T).float()
        else:
            sk, sv = model.ctx_rows(ctx_emb, ctx_hprev, ctx_pos)
            pre, _ = model(emb_q, hprev_q, (sk, sv), pos_q, valid)
            if kind == "sv":
                logits = model.head_logits(pre).float()
            else:
                logits = (pre.to(T.lm_head.dtype) @ T.lm_head.T).float()
        pred = logits.argmax(-1)
        if kind == "sv":
            pred = model.to_target_ids(pred)          # OOV targets = automatic miss
        hits += int((pred == tgt).sum())
    return hits / len(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--ckpt", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=1500)
    args = ap.parse_args()

    meta, hid, rows = load(args.data)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/data/ai-ml/hf-models/qwen35-9b-hf")
    ids = tok(open(os.path.expanduser(args.corpus)).read()).input_ids[: meta["n_tokens"]]
    bg = meta["bg"]
    ps = [r["pos"] for r in rows if r["pos"] >= 1][: args.n]
    eng = sum(r["drafts"][0] == r["targets"][0] for r in rows[: args.n]) / len(rows[: args.n])
    print(f"{os.path.basename(args.data)}: {len(ps)} positions")
    print(f"{'engine (full ctx)':24} slot0 {eng:.4f}")

    T = TeacherHead(interleaved_rope=True)
    acc = eval_model("teacher", None, T, ids, hid, bg, ps, args.window)
    print(f"{'teacher (windowed)':24} slot0 {acc:.4f}")
    for c in args.ckpt:
        ck = torch.load(c, map_location="cuda", weights_only=False)
        if ck["student"] == "sv":
            hot = torch.load("/opt/dlami/nvme/hqmtp-data2/hot32k.pt",
                             weights_only=False)["hot_ids"]
            S = StudentSV(hot).cuda()
        else:
            S = (StudentS1() if ck["student"] == "s1" else StudentS2()).cuda()
        S.load_state_dict(ck["state"])
        S.eval()
        acc = eval_model(ck["student"], S, T, ids, hid, bg, ps, args.window)
        print(f"{os.path.basename(c):24} slot0 {acc:.4f}")


if __name__ == "__main__":
    main()
