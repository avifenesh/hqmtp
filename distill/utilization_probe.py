#!/usr/bin/env python3
"""Is the natural MTP head actually ALL in use? Zero-training structured-prune probe.

Measures, over serving-regime eval docs (slot-0 teacher-forced, windowed):
  - per-FFN-channel activation mass  E|silu(z@gate.T) * (z@up.T)|  (d_ff channels)
  - per-q-head attention contribution  E||o_h||  (pre-wo, per head)
then re-evaluates slot-0 acceptance (top-1 vs engine bg) with the head PHYSICALLY
pruned: FFN channels cut to top-{75,50,25}% by mass, weakest {2,4,8} q-heads zeroed.

If acceptance holds under heavy pruning, "trim++" (vocab trim + block prune, no
training) is a real zero-cost baseline; if it collapses, the block is genuinely
dense-in-use and shrinking it requires distillation.

Usage (box):
  utilization_probe.py --dir /opt/dlami/nvme/hqmtp-gen2-eval --docs 12 --positions 4000
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from qwen_mtp import TeacherHead
from train import rms, rope_batch, teacher_ctx_rows
from hdata import Hiddens

W = 2048


@torch.no_grad()
def forward_probe(T, emb_q, hprev_q, pos_q, ctx_k, ctx_v, ctx_mask,
                  ffn_keep=None, head_zero=None, stats=None):
    """batched_teacher_step0 with instrumentation + physical ablation.
    ffn_keep: LongTensor channel idx to KEEP (None = all). head_zero: set of q-head
    idx to zero. stats: dict accumulating channel/head mass."""
    x = torch.cat((rms(emb_q, T.enorm, T.eps), rms(hprev_q, T.hnorm, T.eps)),
                  dim=-1) @ T.fc.T
    xin = rms(x, T.in_ln, T.eps)
    B = x.shape[0]
    qg = (xin @ T.wq.T).view(B, T.n_q, 2 * T.hd)
    q, gate = qg.chunk(2, dim=-1)
    q = rms(q, T.q_norm, T.eps)
    q = rope_batch(T, q.unsqueeze(1), pos_q[:, None]).squeeze(1)
    k_own = rms((xin @ T.wk.T).view(B, 1, T.n_kv, T.hd), T.k_norm, T.eps)
    k_own = rope_batch(T, k_own, pos_q[:, None])
    v_own = (xin @ T.wv.T).view(B, 1, T.n_kv, T.hd)
    K = torch.cat((ctx_k, k_own), dim=1)
    V = torch.cat((ctx_v, v_own), dim=1)
    rep = T.n_q // T.n_kv
    Ke = K.repeat_interleave(rep, dim=2)
    Ve = V.repeat_interleave(rep, dim=2)
    att = torch.einsum("bhd,bnhd->bhn", q.float(), Ke.float()) / (T.hd ** 0.5)
    pad = F.pad(ctx_mask, (0, 1), value=True)
    att = att.masked_fill(~pad[:, None, :], float("-inf"))
    w = att.softmax(-1)
    o = torch.einsum("bhn,bnhd->bhd", w, Ve.float())            # [B, n_q, hd]
    if stats is not None:
        stats["head"] += o.norm(dim=-1).sum(0).cpu()            # [n_q]
    if head_zero:
        o[:, list(head_zero)] = 0.0
    o = o.to(x.dtype)
    gated = o.reshape(B, -1) * torch.sigmoid(gate.reshape(B, -1).float()).to(x.dtype)
    h1 = x + gated @ T.wo.T
    z = rms(h1, T.post_ln, T.eps)
    act = F.silu(z @ T.w_gate.T) * (z @ T.w_up.T)               # [B, d_ff]
    if stats is not None:
        stats["ffn"] += act.float().abs().sum(0).cpu()          # [d_ff]
    if ffn_keep is not None:
        mask = torch.zeros(act.shape[-1], dtype=torch.bool, device=act.device)
        mask[ffn_keep] = True
        act = act * mask                                        # physical cut equivalent
    h_nextn = h1 + act @ T.w_down.T
    logits = rms(h_nextn, T.final_norm, T.eps) @ T.lm_head.T
    return logits.float()


def batches(docs, n_pos, B, dev, T):
    import random
    random.seed(3)
    pool = [(d, p) for d, (ids, hid, bg) in enumerate(docs)
            for p in range(1, len(ids) - 2)]
    random.shuffle(pool)
    pool = pool[:n_pos]
    for s in range(0, len(pool), B):
        chunk = pool[s: s + B]
        embq, hq, pq, tgt, cks, cvs, masks = [], [], [], [], [], [], []
        for d, p in chunk:
            ids, hid, bg = docs[d]
            lo = max(0, p - W)
            idx = torch.arange(lo, p)
            ctx_tok = torch.tensor(ids)[idx].unsqueeze(0).to(dev)
            hp = torch.from_numpy(hid[(idx - 1).clamp(min=0).numpy()]).float().unsqueeze(0).to(dev)
            hp[:, idx == 0] = 0
            ck, cv = teacher_ctx_rows(T, T.embed[ctx_tok].to(T.dtype_), hp.to(T.dtype_),
                                      (idx + 1).unsqueeze(0).to(dev))
            cks.append(ck); cvs.append(cv)
            masks.append(torch.ones(1, p - lo, dtype=torch.bool, device=dev))
            embq.append(T.embed[ids[p]]); hq.append(torch.from_numpy(hid[[p - 1]])[0])
            pq.append(p + 1); tgt.append(bg[p + 1])
        Wm = max(c.shape[1] for c in cks)
        ck = torch.zeros(len(chunk), Wm, T.n_kv, T.hd, device=dev, dtype=cks[0].dtype)
        cv = torch.zeros_like(ck)
        mask = torch.zeros(len(chunk), Wm, dtype=torch.bool, device=dev)
        for i, (k, v, m) in enumerate(zip(cks, cvs, masks)):
            ck[i, -k.shape[1]:] = k[0]; cv[i, -v.shape[1]:] = v[0]
            mask[i, -m.shape[1]:] = m[0]
        yield (torch.stack(embq).to(dev).float(), torch.stack(hq).to(dev).float(),
               torch.tensor(pq, device=dev), ck, cv, mask,
               torch.tensor(tgt, device=dev))


def evaluate(T, docs, n_pos, dev, ffn_keep=None, head_zero=None, stats=None):
    hit = n = 0
    for embq, hq, pq, ck, cv, mask, tgt in batches(docs, n_pos, 24, dev, T):
        lg = forward_probe(T, embq.to(T.dtype_), hq.to(T.dtype_), pq, ck, cv, mask,
                           ffn_keep, head_zero, stats)
        hit += (lg.argmax(-1) == tgt).sum().item()
        n += len(tgt)
    return hit / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--docs", type=int, default=12)
    ap.add_argument("--positions", type=int, default=4000)
    args = ap.parse_args()
    dev = "cuda"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        os.environ.get("HQMTP_TOK", "/data/ai-ml/hf-models/qwen35-9b-hf"))
    T = TeacherHead(interleaved_rope=True)

    docs = []
    for mp in sorted(glob.glob(os.path.join(args.dir, "*.hiddens.meta.json")))[: args.docs]:
        base = mp[: -len(".hiddens.meta.json")]
        meta = json.load(open(mp))
        h = Hiddens(base)
        ids = tok(open(base + ".txt").read()).input_ids[: h.n]
        docs.append((ids, h, meta["bg"]))
    print(f"{len(docs)} docs")

    stats = {"ffn": torch.zeros(T.w_gate.shape[0]), "head": torch.zeros(T.n_q)}
    base = evaluate(T, docs, args.positions, dev, stats=stats)
    print(f"baseline slot0: {base:.4f}")
    print("head mass (rel): " +
          " ".join(f"{v:.2f}" for v in (stats['head'] / stats['head'].max()).tolist()))

    order = stats["ffn"].argsort(descending=True)
    d_ff = len(order)
    for frac in (0.75, 0.5, 0.25):
        keep = order[: int(d_ff * frac)].to(dev)
        acc = evaluate(T, docs, args.positions, dev, ffn_keep=keep)
        print(f"ffn keep {int(frac*100)}% ({int(d_ff*frac)}ch): slot0 {acc:.4f} "
              f"({acc/base*100:.1f}% of base)")
    horder = stats["head"].argsort()          # weakest first
    for nz in (2, 4, 8):
        hz = set(horder[:nz].tolist())
        acc = evaluate(T, docs, args.positions, dev, head_zero=hz)
        print(f"heads zeroed {sorted(hz)}: slot0 {acc:.4f} ({acc/base*100:.1f}% of base)")


if __name__ == "__main__":
    main()
