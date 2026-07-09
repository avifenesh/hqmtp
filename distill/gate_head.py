#!/usr/bin/env python3
"""Engine-parity gate for the torch TeacherHead (mandatory before any training).

Feeds the ENGINE's own hiddens (BW24_REPLAY_HDUMP) into the torch head and compares
its K=4 drafts against the engine's replay drafts at the same positions. Both rope
pairing conventions are tried; report per-slot match rates.

Usage:
  python3 gate_head.py --data /opt/dlami/nvme/hqmtp-distill-data/train-ggml --n 200
  (expects <data>.hiddens + <data>.hiddens.meta.json + <data>.drafts.jsonl)
"""
import argparse
import json

import numpy as np
import torch

from qwen_mtp import TeacherHead


def load_data(base):
    meta = json.load(open(base + ".hiddens.meta.json"))
    n, d = meta["n_tokens"], meta["n_embd"]
    from hdata import Hiddens
    hid = Hiddens(base)
    rows = [json.loads(l) for l in open(base + ".drafts.jsonl")]
    return meta, hid, rows


def run_gate(base, n_eval, interleaved, corpus=None, fp32=False):
    global _CORPUS
    _CORPUS = corpus
    meta, hid, rows = load_data(base)
    head = TeacherHead(interleaved_rope=interleaved,
                       dtype=torch.float32 if fp32 else torch.bfloat16)
    # token ids: recover from bg? No — drafts.jsonl has positions; tokens come from the
    # corpus meta? The engine tokenized the corpus; we replicate with the HF tokenizer
    # (proven identical: 17794 == 17794 on gen-dist).
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/data/ai-ml/hf-models/qwen35-9b-hf")
    text = open(meta_corpus_path(base)).read()
    ids = tok(text).input_ids[: meta["n_tokens"]]
    ids_t = torch.tensor(ids)
    rows = [r for r in rows if r["pos"] >= 1][:n_eval]
    max_p = max(r["pos"] for r in rows)
    # prefill context KV once up to max_p (rows 0..max_p), slice per position
    hview = torch.from_numpy(np.ascontiguousarray(hid[: max_p + 4])).cuda()
    pred = torch.zeros_like(hview)
    pred[1:] = hview[:-1]           # row i pairs h_{i-1}; row 0 zeros
    K, V = [], []
    B = 4096
    with torch.no_grad():
        for s in range(0, max_p + 1, B):
            e = min(s + B, max_p + 1)
            k, v = head.kv_fill(ids_t[s:e].cuda(), pred[s:e], s)
            K.append(k)
            V.append(v)
    K = torch.cat(K); V = torch.cat(V)
    slot_hit = [0] * 4
    n = 0
    for r in rows:
        p = r["pos"]
        drafts = head.draft(ids, hview, p, 4, (K[:p].clone(), V[:p].clone()))
        n += 1
        for j in range(4):
            if drafts[j] == r["drafts"][j]:
                slot_hit[j] += 1
    rates = [h / n for h in slot_hit]
    print(f"[gate-head] interleaved={interleaved} n={n} slot match: "
          + " ".join(f"{x:.4f}" for x in rates))
    return rates


_CORPUS = None


def meta_corpus_path(base):
    # explicit --corpus wins; else train-<name>.hiddens -> ~/corpora/<name>.<ext>
    import glob
    import os
    if _CORPUS:
        return os.path.expanduser(_CORPUS)
    name = os.path.basename(base).replace("train-", "")
    cands = glob.glob(os.path.expanduser(f"~/corpora/{name}*"))
    assert cands, f"no corpus file for {name}"
    return cands[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--corpus")
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()
    for il in (False, True):
        run_gate(args.data, args.n, il, args.corpus, args.fp32)
