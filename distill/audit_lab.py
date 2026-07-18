#!/usr/bin/env python3
"""AUDIT LAB — validates the pre-deployment robustness audit across compression arms.

For each head variant (all trim-32k, differing in block/head treatment):
  reference   : FP block + FP hot lm_head
  q8/q4/nvfp4 : weight roundtrip quantization (uniform fidelity cuts)
  prune25/50  : activation-mass FFN channel masks (selective cuts)
  svd768/svdw768 : low-rank FFN, data-free vs activation-whitened (the 2x2 cell)
measure per token over IN-DIST (gen2-eval) and OOD (ood-stream) corpora:
  slot-0 accept vs bg, top-1 agreement with reference, target-token frequency bin.
Audit statistic (in-dist only): concentration of disagreement in rare-token bins.
Validation: audit score vs measured OOD tax across arms.
Zero training. Run on box.
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
from train import rms, rope_batch, teacher_ctx_rows
from hdata import Hiddens
from nvfp4_pack import quantize_nvfp4, dequantize_nvfp4
import gguf

W = 2048
BLOCK_TENSORS = ["fc", "wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down"]


def rt_gguf(x, qt):
    a = np.ascontiguousarray(x.float().cpu().numpy())
    q = gguf.quants.quantize(a, qt)
    d = gguf.quants.dequantize(q, qt)
    return torch.from_numpy(np.ascontiguousarray(d)).to(x.device, x.dtype)


def rt_nvfp4(x):
    a = np.ascontiguousarray(x.float().cpu().numpy())
    q = quantize_nvfp4(a)
    d = dequantize_nvfp4(q, a.shape[0], a.shape[1])
    return torch.from_numpy(d).to(x.device, x.dtype)


def make_arms(T, hot, cal_docs):
    """Return {arm: dict of tensor overrides} + masks. lm_head trimmed to hot for all."""
    dev = T.embed.device
    hot_l = hot.to(dev).long()
    lm_hot = T.lm_head[hot_l].clone()
    arms = {}
    arms["reference"] = {"lm": lm_hot, "ov": {}, "mask": None}
    for name, qt in [("q8", gguf.GGMLQuantizationType.Q8_0),
                     ("q4", gguf.GGMLQuantizationType.Q4_0)]:
        ov = {t: rt_gguf(getattr(T, t), qt) for t in BLOCK_TENSORS}
        arms[name] = {"lm": rt_gguf(lm_hot, qt), "ov": ov, "mask": None}
    ov = {t: rt_nvfp4(getattr(T, t)) for t in BLOCK_TENSORS}
    arms["nvfp4"] = {"lm": rt_nvfp4(lm_hot), "ov": ov, "mask": None}

    # activation-mass ranking from calibration docs
    import utilization_probe as up
    st = {"ffn": torch.zeros(T.w_gate.shape[0]), "head": torch.zeros(T.n_q)}
    up.evaluate(T, cal_docs, 3000, "cuda", stats=st)
    order = st["ffn"].argsort(descending=True)
    for frac, name in [(0.25, "prune25"), (0.5, "prune50")]:
        m = torch.zeros(T.w_gate.shape[0], dtype=torch.bool, device=dev)
        m[order[: int(len(order) * frac)].to(dev)] = True
        arms[name] = {"lm": lm_hot, "ov": {}, "mask": m}

    # low-rank FFN at rank ~byte-matched to prune25 (r*(d+dff)=0.25*d*dff)
    d = T.w_gate.shape[1]; dff = T.w_gate.shape[0]
    r = int(0.25 * d * dff / (d + dff) // 32 * 32)
    def svd_pair(w, rank, row_scale=None):
        wf = w.float()
        if row_scale is not None:
            wf = wf * row_scale[None, :]
        U, S, Vh = torch.linalg.svd(wf, full_matrices=False)
        lo = (U[:, :rank] * S[:rank]) @ Vh[:rank]
        if row_scale is not None:
            lo = lo / row_scale[None, :].clamp(min=1e-8)
        return lo.to(w.dtype)
    arms["svd768"] = {"lm": lm_hot, "mask": None, "ov": {
        "w_gate": svd_pair(T.w_gate, r), "w_up": svd_pair(T.w_up, r),
        "w_down": svd_pair(T.w_down, r)}}
    # whitened: scale input dims by activation RMS measured on calibration
    zs = collect_z_rms(T, cal_docs)                      # [d] rms of pre-FFN z
    arms["svdw768"] = {"lm": lm_hot, "mask": None, "ov": {
        "w_gate": svd_pair(T.w_gate, r, zs), "w_up": svd_pair(T.w_up, r, zs),
        "w_down": svd_pair(T.w_down, r)}}
    print(f"[arms] svd rank {r}, arms: {list(arms)}", flush=True)
    return arms


@torch.no_grad()
def collect_z_rms(T, docs, npos=800):
    import utilization_probe as up
    acc = torch.zeros(T.w_gate.shape[1], device="cuda")
    n = 0
    for embq, hq, pq, ck, cv, mask, tgt in up.batches(docs, npos, 24, "cuda", T):
        x = torch.cat((rms(embq.to(T.dtype_), T.enorm, T.eps),
                       rms(hq.to(T.dtype_), T.hnorm, T.eps)), -1) @ T.fc.T
        # approximate: use post-attn-free z proxy = rms(x) (attention omitted for speed)
        z = rms(x, T.post_ln, T.eps).float()
        acc += (z ** 2).sum(0)
        n += z.shape[0]
    return (acc / n).sqrt()


@torch.no_grad()
def forward_arm(T, arm, embq, hq, pq, ck, cv, mask):
    """slot-0 logits over HOT vocab with arm overrides. ck/cv computed with REFERENCE
    weights (persistent draft KV comes from the fill — engine uses the same head for
    fill; approximation: shared reference ctx isolates the QUERY-side function change,
    matching the audit's purpose of ranking arms cheaply)."""
    g = lambda n: arm["ov"].get(n, getattr(T, n))
    x = torch.cat((rms(embq.to(T.dtype_), T.enorm, T.eps),
                   rms(hq.to(T.dtype_), T.hnorm, T.eps)), -1) @ g("fc").T
    xin = rms(x, T.in_ln, T.eps)
    B = x.shape[0]
    qg = (xin @ g("wq").T).view(B, T.n_q, 2 * T.hd)
    q, gate = qg.chunk(2, -1)
    q = rms(q, T.q_norm, T.eps)
    q = rope_batch(T, q.unsqueeze(1), pq[:, None]).squeeze(1)
    k_own = rms((xin @ g("wk").T).view(B, 1, T.n_kv, T.hd), T.k_norm, T.eps)
    k_own = rope_batch(T, k_own, pq[:, None])
    v_own = (xin @ g("wv").T).view(B, 1, T.n_kv, T.hd)
    K = torch.cat((ck, k_own), 1); V = torch.cat((cv, v_own), 1)
    rep = T.n_q // T.n_kv
    att = torch.einsum("bhd,bnhd->bhn", q.float(),
                       K.repeat_interleave(rep, 2).float()) / (T.hd ** 0.5)
    pad = F.pad(mask, (0, 1), value=True)
    att = att.masked_fill(~pad[:, None, :], float("-inf"))
    o = torch.einsum("bhn,bnhd->bhd", att.softmax(-1),
                     V.repeat_interleave(rep, 2).float()).to(x.dtype)
    gated = o.reshape(B, -1) * torch.sigmoid(gate.reshape(B, -1).float()).to(x.dtype)
    h1 = x + gated @ g("wo").T
    z = rms(h1, T.post_ln, T.eps)
    act = F.silu(z @ g("w_gate").T) * (z @ g("w_up").T)
    if arm["mask"] is not None:
        act = act * arm["mask"]
    hn = h1 + act @ g("w_down").T
    return (rms(hn, T.final_norm, T.eps) @ arm["lm"].T).float()


def load_docs(pat, corpus_dir, n, tok):
    out = []
    for mp in sorted(glob.glob(pat))[:n]:
        base = mp[: -len(".hiddens.meta.json")]
        stem = os.path.basename(base)
        txt = base + ".txt"
        if not os.path.exists(txt):
            txt = os.path.join(corpus_dir, stem + ".txt")
        meta = json.load(open(mp))
        h = Hiddens(base)
        ids = tok(open(txt).read()).input_ids[: h.n]
        out.append((ids, h, meta["bg"]))
    return out


def main():
    from transformers import AutoTokenizer
    import utilization_probe as up
    tok = AutoTokenizer.from_pretrained("/data/ai-ml/hf-models/qwen35-9b-hf")
    T = TeacherHead(interleaved_rope=True)
    dev = "cuda"
    hot = torch.load("/opt/dlami/nvme/hqmtp-data2/hot32k.pt",
                     weights_only=False)["hot_ids"]
    hot_l = hot.long()
    t2d = torch.full((T.lm_head.shape[0],), -1, dtype=torch.long, device=dev)
    t2d[hot_l.to(dev)] = torch.arange(len(hot), device=dev)

    cal = load_docs("/opt/dlami/nvme/hqmtp-gen2-data/*.hiddens.meta.json",
                    "/opt/dlami/nvme/hqmtp-gen2-corpus", 12, tok)
    indist = load_docs("/opt/dlami/nvme/hqmtp-gen2-eval/*.hiddens.meta.json",
                       "", 12, tok)
    ood = load_docs("/opt/dlami/nvme/hqmtp-ood-data/*.hiddens.meta.json", "", 24, tok)

    # unigram freq bins from calibration corpus
    cnt = np.zeros(T.lm_head.shape[0], dtype=np.int64)
    for f in sorted(glob.glob("/opt/dlami/nvme/hqmtp-gen2-corpus/*.txt"))[::4]:
        for i in tok(open(f).read()).input_ids:
            cnt[i] += 1
    arms = make_arms(T, hot, cal)

    results = {}
    for cname, docs, npos in [("indist", indist, 2500), ("ood", ood, 2500)]:
        per_arm = {a: {"acc": np.zeros(5), "agree": np.zeros(5), "n": np.zeros(5)}
                   for a in arms}
        for embq, hq, pq, ck, cv, mask, tgt in up.batches(docs, npos, 24, dev, T):
            tgt_d = t2d[tgt]
            valid = tgt_d >= 0
            fr = cnt[tgt.cpu().numpy()]
            # frequency quintile bins over LOG count (0 = rarest)
            bins = np.clip(np.digitize(np.log1p(fr), [2, 4, 6, 8]), 0, 4)
            ref_pred = None
            for a, arm in arms.items():
                lg = forward_arm(T, arm, embq, hq, pq, ck, cv, mask)
                pred = lg.argmax(-1)
                if a == "reference":
                    ref_pred = pred
                hit = ((pred == tgt_d) & valid).cpu().numpy()
                agr = (pred == ref_pred).cpu().numpy()
                for b in range(5):
                    sel = bins == b
                    per_arm[a]["acc"][b] += hit[sel].sum()
                    per_arm[a]["agree"][b] += agr[sel].sum()
                    per_arm[a]["n"][b] += sel.sum()
        results[cname] = {a: {k: v.tolist() for k, v in d.items()}
                          for a, d in per_arm.items()}
        for a in arms:
            d = per_arm[a]
            acc = d["acc"].sum() / d["n"].sum()
            print(f"[{cname}] {a:10} acc {acc:.4f} | per-bin acc "
                  + " ".join(f"{x:.3f}" for x in (d["acc"] / np.maximum(d["n"], 1)))
                  + " | agree " + " ".join(f"{x:.3f}"
                  for x in (d["agree"] / np.maximum(d["n"], 1))), flush=True)

    json.dump(results, open(os.path.expanduser("~/hqmtp-distill/audit-lab.json"), "w"))
    # audit score (in-dist only): disagreement concentration in rarest 2 bins
    print("\n=== AUDIT vs OOD TAX ===")
    ref_i = results["indist"]["reference"]
    ref_o = results["ood"]["reference"]
    ref_acc_i = sum(ref_i["acc"]) / sum(ref_i["n"])
    ref_acc_o = sum(ref_o["acc"]) / sum(ref_o["n"])
    for a in arms:
        if a == "reference":
            continue
        di = results["indist"][a]
        do = results["ood"][a]
        dis = [di["n"][b] - di["agree"][b] for b in range(5)]
        rare_share = sum(dis[:2]) / max(sum(dis), 1)
        acc_i = sum(di["acc"]) / sum(di["n"])
        acc_o = sum(do["acc"]) / sum(do["n"])
        tax = (acc_i / ref_acc_i) - (acc_o / ref_acc_o)
        print(f"{a:10} audit(rare-share of disagreement) {rare_share:.3f} | "
              f"in-dist retention {acc_i/ref_acc_i:.3f} | OOD retention "
              f"{acc_o/ref_acc_o:.3f} | TAX {tax:+.3f}", flush=True)


if __name__ == "__main__":
    main()
