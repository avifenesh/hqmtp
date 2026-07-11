#!/usr/bin/env python3
"""Precompute the TEACHER head's logits (top-k) for all 4 chain slots over the training
corpus — the distillation targets ("copy the logits, train on top").

Per doc: fill the teacher's draft-KV once (rows = (x_i, h_{i-1}), exact), then for each
slot j run batched queries. Teacher-forced chain inputs (x_{p+j}) mean the chain's
appended rows EQUAL the true fill rows, so slot-j targets are exact per position:
  slot j query at position p: input (x_{p+j}, carrier_j), rope p+1+j, KV rows [0..p+j)
  carrier_0 = trunk h_{p-1}; carrier_{j+1} = teacher h_nextn from slot j.

Output per doc <stem>.tt.npz: idx [N, S, K] int32, prob [N, S, K] float16 (top-K softmax
probs, unnormalized tail dropped), aligned to positions 0..N-1 (position p row = slot
targets for drafts at p; rows with p+S > N-1 are valid only up to the doc end).

Usage (box):
  teacher_targets.py --data-dir /opt/dlami/nvme/hqmtp-gen-data \
      --corpus-dir /opt/dlami/nvme/hqmtp-gen-corpus --slots 4 --topk 64
"""
import argparse
import glob
import os

import numpy as np
import torch

from qwen_mtp import TeacherHead
from train import rms, rope_batch
from hdata import Hiddens


@torch.no_grad()
def doc_kv_fill(T, ids, hiddens, dev):
    """Teacher K/V for every doc row (exact fill): row i = (x_i, h_{i-1}), rope i+1."""
    N = len(ids)
    pred = torch.zeros(N, T.embed.shape[1], device=dev)
    pred[1:] = hiddens[:-1]
    emb = T.embed[ids].float()
    x = torch.cat((rms(emb.to(T.dtype_), T.enorm, T.eps),
                   rms(pred.to(T.dtype_), T.hnorm, T.eps)), dim=-1) @ T.fc.T
    xin = rms(x, T.in_ln, T.eps)
    k = rms((xin @ T.wk.T).view(N, T.n_kv, T.hd), T.k_norm, T.eps)
    v = (xin @ T.wv.T).view(N, T.n_kv, T.hd)
    pos = torch.arange(1, N + 1, device=dev)
    k = rope_batch(T, k.unsqueeze(0), pos.unsqueeze(0)).squeeze(0)
    return k, v   # [N, n_kv, hd]


@torch.no_grad()
def slot_query(T, tok_ids, carrier, q_pos, K, V, kv_len, dev, B=256):
    """Batched teacher single-row queries. tok_ids [M], carrier [M,d], q_pos [M],
    kv_len [M] (#rows visible). Returns (logits [M,vocab] fp32-chunked topk outside,
    h_nextn [M,d])."""
    M = len(tok_ids)
    outs_i, outs_p, outs_h = [], [], []
    N = K.shape[0]
    ar = torch.arange(N, device=dev)
    for s in range(0, M, B):
        e = min(s + B, M)
        emb = T.embed[tok_ids[s:e]].float()
        x = torch.cat((rms(emb.to(T.dtype_), T.enorm, T.eps),
                       rms(carrier[s:e].to(T.dtype_), T.hnorm, T.eps)),
                      dim=-1) @ T.fc.T
        xin = rms(x, T.in_ln, T.eps)
        b = e - s
        qg = (xin @ T.wq.T).view(b, T.n_q, 2 * T.hd)
        q, gate = qg.chunk(2, dim=-1)
        q = rms(q, T.q_norm, T.eps)
        q = rope_batch(T, q.unsqueeze(1), q_pos[s:e, None]).squeeze(1)
        k_own = rms((xin @ T.wk.T).view(b, 1, T.n_kv, T.hd), T.k_norm, T.eps)
        k_own = rope_batch(T, k_own, q_pos[s:e, None])
        v_own = (xin @ T.wv.T).view(b, 1, T.n_kv, T.hd)
        # attention over SHARED doc rows [0..kv_len) + own row — GQA einsum without
        # materializing K per batch row (b x N x n_q x hd blew 24GB at N=3.5k, b=256)
        rep = T.n_q // T.n_kv
        qg = q.view(b, T.n_kv, rep, T.hd).float()
        att = torch.einsum("bkrd,nkd->bkrn", qg, K.float()) / (T.hd ** 0.5)  # [b,kv,rep,N]
        mask = ar[None, :] < kv_len[s:e, None]                   # [b, N]
        att = att.masked_fill(~mask[:, None, None, :], float("-inf"))
        att_own = (qg * k_own.squeeze(1).float()[:, :, None, :]).sum(-1, keepdim=True) \
                  / (T.hd ** 0.5)                                # [b,kv,rep,1]
        w = torch.cat((att, att_own), dim=-1).softmax(-1)
        o = torch.einsum("bkrn,nkd->bkrd", w[..., :-1], V.float()) \
            + w[..., -1:] * v_own.squeeze(1).float()[:, :, None, :]
        o = o.reshape(b, T.n_q, T.hd).to(T.dtype_)
        gated = o.reshape(b, -1) * torch.sigmoid(gate.reshape(b, -1).float()).to(T.dtype_)
        h1 = x + gated @ T.wo.T
        z = rms(h1, T.post_ln, T.eps)
        mlp = (torch.nn.functional.silu(z @ T.w_gate.T) * (z @ T.w_up.T)) @ T.w_down.T
        h_nextn = h1 + mlp
        logits = rms(h_nextn, T.final_norm, T.eps) @ T.lm_head.T
        p, i = logits.float().softmax(-1).topk(args_topk, dim=-1)
        outs_i.append(i.int().cpu())
        outs_p.append(p.half().cpu())
        outs_h.append(h_nextn.float())
    return torch.cat(outs_i), torch.cat(outs_p), torch.cat(outs_h)


def main():
    global args_topk
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--topk", type=int, default=64)
    args = ap.parse_args()
    args_topk = args.topk
    dev = "cuda"
    T = TeacherHead(interleaved_rope=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/data/ai-ml/hf-models/qwen35-9b-hf")

    metas = sorted(glob.glob(f"{args.data_dir}/*.hiddens.meta.json"))
    done = 0
    for mp in metas:
        base = mp[: -len(".hiddens.meta.json")]
        out = base + ".tt.npz"
        if os.path.exists(out):
            done += 1
            continue
        h = Hiddens(base)
        stem = os.path.basename(base)
        cands = glob.glob(os.path.join(args.corpus_dir, stem + "*"))
        if not cands:
            continue
        ids = tok(open(cands[0], errors="replace").read()).input_ids[: h.n]
        N = len(ids)
        if N < args.slots + 4:
            continue
        ids_t = torch.tensor(ids, device=dev)
        hiddens = torch.from_numpy(h[np.arange(N)]).float().to(dev)
        K, V = doc_kv_fill(T, ids_t, hiddens, dev)
        S = args.slots
        M = N - S                                  # positions with full slot coverage
        pos_idx = torch.arange(1, M + 1, device=dev)  # position p = 1..M
        all_i = np.zeros((N, S, args.topk), dtype=np.int32)
        all_p = np.zeros((N, S, args.topk), dtype=np.float16)
        carrier = hiddens[pos_idx - 1]             # h_{p-1}
        for j in range(S):
            tok_j = ids_t[pos_idx + j]             # x_{p+j}
            q_pos = pos_idx + 1 + j
            kv_len = pos_idx + j                   # rows [0..p+j)
            ti, tp, th = slot_query(T, tok_j, carrier, q_pos, K, V, kv_len, dev)
            all_i[1: M + 1, j] = ti.numpy()
            all_p[1: M + 1, j] = tp.numpy()
            carrier = th                           # teacher's own carrier recursion
        np.savez(out, idx=all_i, prob=all_p)
        done += 1
        if done % 50 == 0:
            print(f"[tt] {done}/{len(metas)}", flush=True)
    print(f"TT-DONE {done}/{len(metas)}")


if __name__ == "__main__":
    main()
