#!/usr/bin/env python3
"""Per-slot SELF-DRAFTING chain eval — the engine-equivalent acceptance measure.

At each held-out position the student drafts K slots feeding its OWN picks + carrier
back (exactly the engine chain regime). Per-slot hit vs the engine-recorded targets
(bg[p+1+j] in the eval dumps). The engine row = the TEACHER's own chain from the same
dumps — the direct closeness comparison.

Usage:
  eval_chain.py --dir /opt/dlami/nvme/hqmtp-gen-eval \
      --ckpt checkpoints/sv-chain.pt checkpoints/sv-own2.pt
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from qwen_mtp import TeacherHead
from heads import StudentSV
from hdata import Hiddens

SLOTS = 4


@torch.no_grad()
def chain_eval(S, T, docs, W, dev="cuda", B=48):
    hits = np.zeros(SLOTS)
    chain_len = 0
    n = 0
    embed = T.embed
    for ids, hid, rows in docs:
        ps = [r["pos"] for r in rows if r["pos"] >= 1 and r["pos"] + SLOTS < len(ids)]
        tgts = {r["pos"]: r["targets"] for r in rows}
        for s0 in range(0, len(ps), B):
            batch = ps[s0: s0 + B]
            b = len(batch)
            pt = torch.tensor(batch)
            lo = (pt - W).clamp(min=0)
            idx = lo[:, None] + torch.arange(W)[None, :]
            valid = (idx < pt[:, None]).to(dev)
            idx = idx.clamp(max=len(ids) - 1)
            ctx_tok = torch.tensor(ids)[idx].to(dev)
            ctx_hprev = torch.from_numpy(hid[(idx - 1).clamp(min=0).numpy()]).float().to(dev)
            ctx_hprev[(idx == 0).to(dev)] = 0.0
            ctx_pos = (idx + 1).to(dev)
            sk, sv = S.ctx_rows(embed[ctx_tok].float(), ctx_hprev, ctx_pos)
            cur_k, cur_v, cur_mask = sk, sv, valid
            tok = torch.tensor([ids[p] for p in batch]).to(dev)
            carrier = torch.from_numpy(hid[[p - 1 for p in batch]]).float().to(dev)
            pos_q = (pt + 1).to(dev)
            alive = torch.ones(b, dtype=torch.bool)
            for j in range(SLOTS):
                pre, h, (ko, vo) = S(embed[tok].float(), carrier, (cur_k, cur_v),
                                     pos_q + j, cur_mask, return_kv=True)
                pred = S.to_target_ids(S.head_logits(pre).float().argmax(-1)).cpu()
                tgt_j = torch.tensor([tgts[p][j] for p in batch])
                hit = (pred == tgt_j).numpy()
                hits[j] += hit.sum()
                # chain acceptance: count while all previous slots hit
                alive &= torch.from_numpy(hit)
                chain_len += int(alive.sum())
                tok = pred.to(dev)                     # SELF-drafted token feeds back
                carrier = h
                cur_k = torch.cat((cur_k, ko), dim=1)
                cur_v = torch.cat((cur_v, vo), dim=1)
                cur_mask = F.pad(cur_mask, (0, 1), value=True)
            n += b
    return hits / n, chain_len / (n * SLOTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--ckpt", nargs="*", default=[])
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        os.environ.get("HQMTP_TOK", "/data/ai-ml/hf-models/qwen35-9b-hf"))
    T = TeacherHead(interleaved_rope=True)

    docs = []
    eng = np.zeros(SLOTS)
    eng_chain = eng_n = 0
    for mp in sorted(glob.glob(os.path.join(args.dir, "*.hiddens.meta.json"))):
        base = mp[: -len(".hiddens.meta.json")]
        h = Hiddens(base)
        rows = [json.loads(l) for l in open(base + ".drafts.jsonl")]
        ids = tok(open(base + ".txt").read()).input_ids[: h.n]
        rows = [r for r in rows if r["pos"] >= 1 and r["pos"] + SLOTS < len(ids)]
        docs.append((ids, h, rows))
        for r in rows:
            alive = True
            for j in range(SLOTS):
                hit = r["drafts"][j] == r["targets"][j]
                eng[j] += hit
                alive = alive and hit
                eng_chain += alive
            eng_n += 1
    print(f"{len(docs)} docs, {eng_n} positions")
    e = eng / eng_n
    print(f"{'engine teacher chain':22} slots {e[0]:.3f} {e[1]:.3f} {e[2]:.3f} {e[3]:.3f}"
          f"  chain {eng_chain/(eng_n*SLOTS):.3f}")
    for c in args.ckpt:
        ck = torch.load(c, map_location="cuda", weights_only=False)
        hot = torch.load(os.environ.get("HQMTP_HOT",
                         "/opt/dlami/nvme/hqmtp-data2/hot32k.pt"),
                         weights_only=False)["hot_ids"]
        S = StudentSV(hot, d=T.embed.shape[1]).cuda()
        S.load_state_dict(ck["state"])
        S.eval()
        r, cl = chain_eval(S, T, docs, args.window)
        print(f"{os.path.basename(c):22} slots {r[0]:.3f} {r[1]:.3f} {r[2]:.3f} {r[3]:.3f}"
              f"  chain {cl:.3f}")


if __name__ == "__main__":
    main()
