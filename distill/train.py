#!/usr/bin/env python3
"""Train student draft heads by distilling the co-trained MTP head (slot-0, v1).

Data: engine-extracted (BW24_REPLAY_HDUMP) trunk hiddens + bg per corpus, tokens
re-derived with the HF tokenizer (count-verified vs engine). Teacher targets are
computed on the fly (frozen TeacherHead, batched, windowed attention W rows —
students see the same window, so distillation pairs are consistent).

Losses: alpha*KL(topk teacher||student) + beta*CE(student, bg) + gamma*(1-cos(h)).
B0 control = --alpha 0 --gamma 0 (pure CE, same architecture, same data).

Usage (box):
  ~/.venvs/distill/bin/python train.py --student s2 --steps 3000 --out checkpoints/s2-kd
  ~/.venvs/distill/bin/python train.py --student s2 --alpha 0 --gamma 0 --out checkpoints/s2-b0
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from qwen_mtp import TeacherHead
from heads import StudentS1, StudentS2, StudentSV, param_count

DATA_DIR = "/opt/dlami/nvme/hqmtp-distill-data"


class Corpus:
    def __init__(self, base, tok, corpus_dir=None):
        meta = json.load(open(base + ".hiddens.meta.json"))
        self.n, self.d = meta["n_tokens"], meta["n_embd"]
        self.bg = torch.tensor(meta["bg"], dtype=torch.long)
        from hdata import Hiddens
        self.hid = Hiddens(base)
        name = os.path.basename(base).replace("train-", "")
        self.name = name
        pats = ([f"{corpus_dir}/{name}*"] if corpus_dir else []) + [f"~/corpora/{name}*"]
        cands = sum((glob.glob(os.path.expanduser(p)) for p in pats), [])
        assert cands, f"corpus file for {name}?"
        ids = tok(open(cands[0]).read()).input_ids[: self.n]
        assert len(ids) == self.n, f"{name}: token count {len(ids)} != {self.n}"
        self.ids = torch.tensor(ids, dtype=torch.long)
        tt = base + ".tt.npz"
        self.tt = np.load(tt, mmap_mode="r") if os.path.exists(tt) else None


def batched_teacher_step0(T, emb_q, hprev_q, pos_q, ctx_k, ctx_v, ctx_mask):
    """Frozen teacher slot-0, batched + windowed. emb_q/hprev_q [B,d]; pos_q [B];
    ctx_k/ctx_v [B,W,n_kv,hd]. Returns (logits [B,vocab], h_nextn [B,d])."""
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
    o = torch.einsum("bhn,bnhd->bhd", w, Ve.float()).to(x.dtype)
    gated = o.reshape(B, -1) * torch.sigmoid(gate.reshape(B, -1).float()).to(x.dtype)
    h1 = x + gated @ T.wo.T
    z = rms(h1, T.post_ln, T.eps)
    mlp = (F.silu(z @ T.w_gate.T) * (z @ T.w_up.T)) @ T.w_down.T
    h_nextn = h1 + mlp
    logits = rms(h_nextn, T.final_norm, T.eps) @ T.lm_head.T
    return logits.float(), h_nextn


def teacher_ctx_rows(T, emb, hprev, pos):
    """Teacher K/V for window rows. emb/hprev [B,W,d], pos [B,W]."""
    x = torch.cat((rms(emb, T.enorm, T.eps), rms(hprev, T.hnorm, T.eps)),
                  dim=-1) @ T.fc.T
    xin = rms(x, T.in_ln, T.eps)
    B, W, _ = x.shape
    k = rms((xin @ T.wk.T).view(B, W, T.n_kv, T.hd), T.k_norm, T.eps)
    v = (xin @ T.wv.T).view(B, W, T.n_kv, T.hd)
    k = rope_batch(T, k, pos)
    return k, v


def rms(x, w, eps):
    v = x.float()
    v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    return (v * w.float()).to(x.dtype)


def rope_batch(T, x, pos):
    """x [B, N, heads, hd]; pos [B, N]. Interleaved pairing, partial rotary."""
    rot = T.rot
    xr, xp = x[..., :rot], x[..., rot:]
    ang = pos.float()[..., None] * T.inv_freq[None, None, :]
    cos, sin = ang.cos().unsqueeze(-2), ang.sin().unsqueeze(-2)
    x1, x2 = xr[..., 0::2], xr[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((torch.stack((o1, o2), -1).flatten(-2).to(x.dtype), xp), -1)


class Prefetcher:
    """Background-thread batch producer — the memmap gathers (~1GB/batch at B=64,W=2048)
    starved the GPU (measured 135W, bursty util). Overlaps CPU gather + H2D with compute."""

    def __init__(self, corpora, B, W, device, weights, depth=3, slots=1):
        import queue
        import threading
        self.q = queue.Queue(maxsize=depth)
        self.stop = False

        def worker():
            while not self.stop:
                self.q.put(sample_batch(corpora, B, W, device, weights, slots))
        self.ts = [threading.Thread(target=worker, daemon=True) for _ in range(2)]
        for t in self.ts:
            t.start()

    def next(self):
        return self.q.get()


def sample_batch(corpora, B, W, device, weights=None, slots=1):
    """Random positions; returns query inputs + window context (tokens, hprev, pos, mask)
    + chain inputs (x_{p+j}), per-slot bg targets, and teacher topk targets if present."""
    if weights is not None:
        c = corpora[int(torch.multinomial(weights, 1))]
    else:
        c = corpora[torch.randint(len(corpora), (1,)).item()]
    ps = torch.randint(1, c.n - slots - 2, (B,))
    lo = (ps - W).clamp(min=0)
    tok_q = c.ids[ps]
    tgt = c.bg[ps + 1]
    idx = lo[:, None] + torch.arange(W)[None, :]          # [B, W] (may overshoot p)
    valid = idx < ps[:, None]
    idx = idx.clamp(max=c.n - 1)
    ctx_tok = c.ids[idx]
    # gather hidden rows: h_prev of row i = hid[i-1] (zeros at 0)
    prev_idx = (idx - 1).clamp(min=0)
    ctx_hprev = torch.from_numpy(c.hid[prev_idx.numpy()]).float()
    ctx_hprev[idx == 0] = 0.0
    hprev_q = torch.from_numpy(c.hid[(ps - 1).numpy()]).float()
    # chain: inputs x_{p+j}, targets bg[p+1+j] for j in 0..slots-1
    off = torch.arange(slots)
    chain_toks = c.ids[ps[:, None] + off[None, :]]              # [B, S]
    slot_tgts = c.bg[ps[:, None] + 1 + off[None, :]]            # [B, S]
    if c.tt is not None:
        pn = ps.numpy()
        tt_i = torch.from_numpy(np.ascontiguousarray(c.tt["idx"][pn])).long()
        tt_p = torch.from_numpy(np.ascontiguousarray(c.tt["prob"][pn])).float()
    else:
        tt_i = tt_p = None
    return (tok_q.to(device), hprev_q.to(device), (ps + 1).to(device),
            ctx_tok.to(device), ctx_hprev.to(device), (idx + 1).to(device),
            valid.to(device), tgt.to(device), chain_toks.to(device),
            slot_tgts.to(device),
            tt_i.to(device) if tt_i is not None else None,
            tt_p.to(device) if tt_p is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", choices=["s1", "s2", "sv"], required=True)
    ap.add_argument("--hot", default="/opt/dlami/nvme/hqmtp-data2/hot32k.pt")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--window", type=int, default=768)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--alpha", type=float, default=1.0)   # KL weight
    ap.add_argument("--beta", type=float, default=0.5)    # CE weight
    ap.add_argument("--gamma", type=float, default=0.3)   # h-carrier cosine weight
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--corpus-dir")
    ap.add_argument("--glob", default="train-*.hiddens")
    ap.add_argument("--init-from", help="checkpoint to continue from")
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--chat-boost", type=float, default=1.0)
    ap.add_argument("--chain-slots", type=int, default=1)
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(7)

    T = TeacherHead(interleaved_rope=True)
    for p in T.parameters():
        p.requires_grad_(False)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        os.environ.get("HQMTP_TOK", "/data/ai-ml/hf-models/qwen35-9b-hf"))
    # glob COMPLETED extractions only (meta written last — in-flight .hiddens files race)
    metas = sorted(glob.glob(f"{args.data_dir}/{args.glob}.meta.json")) or \
            sorted(glob.glob(f"{args.data_dir}/{args.glob}"))
    corpora = [Corpus(b.replace(".hiddens.meta.json", ".hiddens")[: -len(".hiddens")],
                      tok, args.corpus_dir) for b in metas]
    print(f"[data] {len(corpora)} corpora, {sum(c.n for c in corpora)} tokens")

    if args.student == "sv":
        hot = torch.load(args.hot, weights_only=False)["hot_ids"]
        S = StudentSV(hot, big_lm_head=T.lm_head, d=T.embed.shape[1]).to(dev)
        # full-id -> draft-idx map (-1 = out of hot set; CE masks those positions)
        t2d = torch.full((T.lm_head.shape[0],), -1, dtype=torch.long, device=dev)
        t2d[hot.to(dev)] = torch.arange(len(hot), device=dev)
    else:
        S = (StudentS1() if args.student == "s1" else StudentS2()).to(dev)
        t2d = None
    print(f"[student] {args.student} params {param_count(S)/1e6:.1f}M "
          f"(teacher block ~243M)")
    if args.init_from:
        ck = torch.load(args.init_from, map_location=dev, weights_only=False)
        S.load_state_dict(ck["state"])
        print(f"[init] continued from {args.init_from}")
    weights = torch.tensor([args.chat_boost if c.name.startswith("chat") else 1.0
                            for c in corpora])
    if args.chat_boost != 1.0:
        print(f"[data] chat-boost {args.chat_boost}: chat mass "
              f"{weights[weights > 1].sum() / weights.sum():.3f}")
    opt = torch.optim.AdamW(S.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    if args.selftest:
        selftest(T, corpora[0], args.window, dev)
        return

    embed = T.embed          # frozen, shared
    lm_head = T.lm_head
    pf = Prefetcher(corpora, args.batch, args.window, dev, weights, slots=args.chain_slots)
    for step in range(1, args.steps + 1):
        (tok_q, hprev_q, pos_q, ctx_tok, ctx_hprev, ctx_pos, valid, tgt,
         chain_toks, slot_tgts, tt_i, tt_p) = pf.next()
        emb_q = embed[tok_q].float()
        ctx_emb = embed[ctx_tok].float()
        # KL reads precomputed teacher targets (.tt.npz); live teacher only for gamma
        need_teacher = args.gamma > 0
        if need_teacher:
            with torch.no_grad():
                tk, tv = teacher_ctx_rows(T, ctx_emb.to(T.dtype_), ctx_hprev.to(T.dtype_),
                                          ctx_pos)
                t_logits, t_h = batched_teacher_step0(
                    T, emb_q.to(T.dtype_), hprev_q.to(T.dtype_), pos_q, tk, tv, valid)
                t_top_p, t_top_i = t_logits.softmax(-1).topk(args.topk, dim=-1)
                t_top_p = t_top_p / t_top_p.sum(-1, keepdim=True)

        sk, sv = S.ctx_rows(ctx_emb, ctx_hprev, ctx_pos)
        cur_k, cur_v, cur_mask = sk, sv, valid
        h_carrier = hprev_q
        slot_logits = []
        for j in range(args.chain_slots):
            emb_j = embed[chain_toks[:, j]].float()
            pre_head, s_h, (ko, vo) = S(emb_j, h_carrier, (cur_k, cur_v),
                                        pos_q + j, cur_mask, return_kv=True)
            if isinstance(S, StudentSV):
                slot_logits.append(S.head_logits(pre_head).float())
            else:
                slot_logits.append((pre_head.to(lm_head.dtype) @ lm_head.T).float())
            if j + 1 < args.chain_slots:
                cur_k = torch.cat((cur_k, ko), dim=1)
                cur_v = torch.cat((cur_v, vo), dim=1)
                cur_mask = F.pad(cur_mask, (0, 1), value=True)
                h_carrier = s_h                      # SELF-recursive: trains the carrier
        s_logits = slot_logits[0]

        loss = torch.zeros((), device=dev)
        for j, lg in enumerate(slot_logits):
            if args.alpha > 0 and tt_i is not None and isinstance(S, StudentSV):
                ti = t2d[tt_i[:, j]]                  # teacher topk -> draft idx
                ok = ti >= 0
                tp = (tt_p[:, j] * ok).float()
                tp = tp / tp.sum(-1, keepdim=True).clamp(min=1e-9)
                # soft-CE over the FULL draft vocab: subset-only log_softmax leaves the
                # other 32k logits unconstrained (pure-KD argmax collapsed to junk)
                s_logp = lg.log_softmax(-1).gather(-1, ti.clamp(min=0))
                loss = loss + args.alpha * -(tp * s_logp).sum(-1).mean()
            if args.beta > 0:
                sj = slot_tgts[:, j]
                ce_tgt = t2d[sj] if t2d is not None else sj
                keep = ce_tgt >= 0
                if keep.any():
                    loss = loss + args.beta * F.cross_entropy(lg[keep], ce_tgt[keep])
        loss = loss / max(len(slot_logits), 1)
        if args.gamma > 0:
            loss = loss + args.gamma * (1 - F.cosine_similarity(
                s_h.float(), t_h.float(), dim=-1).mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(S.parameters(), 1.0)
        opt.step()
        sched.step()

        if args.save_every and step % args.save_every == 0 and step < args.steps:
            torch.save({"student": args.student, "state": S.state_dict(),
                        "args": vars(args)}, f"{args.out}-step{step}.pt")
        if step % 100 == 0 or step == 1:
            with torch.no_grad():
                pred = s_logits.argmax(-1)
                if isinstance(S, StudentSV):
                    pred = S.to_target_ids(pred)
                acc_bg = (pred == tgt).float().mean().item()
                if not need_teacher:
                    print(f"[{step:5d}] loss {loss.item():.4f} | student-vs-bg {acc_bg:.3f}", flush=True)
                    continue
                acc_t = (pred == t_logits.argmax(-1)).float().mean().item()
                tacc = (t_logits.argmax(-1) == tgt).float().mean().item()
            print(f"[{step:5d}] loss {loss.item():.4f} | student-vs-bg {acc_bg:.3f} "
                  f"student-vs-teacher {acc_t:.3f} teacher-vs-bg {tacc:.3f}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"student": args.student, "state": S.state_dict(),
                "args": vars(args)}, args.out + ".pt")
    print(f"saved {args.out}.pt")


@torch.no_grad()
def selftest(T, c, W, dev):
    """Batched-teacher internal consistency: windowed batched step-0 vs the sequential
    full-context chain (qwen_mtp.draft) on 50 positions. Window truncation causes some
    divergence — report, expect >0.9 agreement."""
    import numpy as np
    ps = list(range(W + 16, min(c.n - 8, W + 16 + 50 * 97), 97))[:50]
    hview = torch.from_numpy(np.ascontiguousarray(c.hid[: max(ps) + 4])).to(dev)
    predfull = torch.zeros_like(hview)
    predfull[1:] = hview[:-1]
    Kf, Vf = [], []
    for s in range(0, max(ps) + 1, 4096):
        e = min(s + 4096, max(ps) + 1)
        k, v = T.kv_fill(c.ids[s:e].to(dev), predfull[s:e], s)
        Kf.append(k); Vf.append(v)
    Kf = torch.cat(Kf); Vf = torch.cat(Vf)
    agree = 0
    for p in ps:
        seq = T.draft(c.ids.tolist(), hview, p, 1, (Kf[:p].clone(), Vf[:p].clone()))[0]
        lo = p - W
        ctx_tok = c.ids[lo:p].unsqueeze(0).to(dev)
        ctx_hprev = hview[lo - 1: p - 1].unsqueeze(0).float()
        ctx_pos = torch.arange(lo + 1, p + 1, device=dev).unsqueeze(0)
        tk, tv = teacher_ctx_rows(T, T.embed[ctx_tok].to(T.dtype_),
                                  ctx_hprev.to(T.dtype_), ctx_pos)
        lg, _ = batched_teacher_step0(
            T, T.embed[c.ids[p]].unsqueeze(0).to(T.dtype_),
            hview[p - 1].unsqueeze(0).to(T.dtype_),
            torch.tensor([p + 1], device=dev), tk, tv,
            torch.ones(1, W, dtype=torch.bool, device=dev))
        agree += int(lg.argmax(-1).item() == seq)
    print(f"[selftest] windowed-batched vs sequential-full: {agree}/50 = {agree/50:.3f}")


if __name__ == "__main__":
    main()
