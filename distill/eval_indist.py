#!/usr/bin/env python3
"""In-distribution eval: aggregate slot-0 acceptance over per-doc extraction dumps
(fresh model-generated docs the checkpoints never saw).

Usage: eval_indist.py --dir /opt/dlami/nvme/hqmtp-indist-eval --ckpt a.pt b.pt
"""
import argparse
import glob
import json
import os

import torch

from qwen_mtp import TeacherHead
from heads import StudentS1, StudentS2, StudentSV
from eval_replay import eval_model
from hdata import Hiddens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--ckpt", nargs="*", default=[])
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/data/ai-ml/hf-models/qwen35-9b-hf")
    T = TeacherHead(interleaved_rope=True)

    docs = []
    eng_hit = eng_n = 0
    for meta_path in sorted(glob.glob(os.path.join(args.dir, "*.hiddens.meta.json"))):
        base = meta_path[: -len(".hiddens.meta.json")]
        h = Hiddens(base)
        rows = [json.loads(l) for l in open(base + ".drafts.jsonl")]
        text = open(base + ".txt").read()
        ids = tok(text).input_ids[: h.n]
        ps = [r["pos"] for r in rows if r["pos"] >= 1]
        docs.append((ids, h, h.meta["bg"], ps))
        eng_hit += sum(r["drafts"][0] == r["targets"][0] for r in rows if r["pos"] >= 1)
        eng_n += len(ps)
    total = sum(len(d[3]) for d in docs)
    print(f"{len(docs)} docs, {total} eval positions")
    print(f"{'engine (full ctx)':24} slot0 {eng_hit/max(eng_n,1):.4f}")

    def score(kind, model):
        hit = 0
        for ids, h, bg, ps in docs:
            hit += eval_model(kind, model, T, ids, h, bg, ps, args.window) * len(ps)
        return hit / total

    print(f"{'teacher (windowed)':24} slot0 {score('teacher', None):.4f}")
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
        print(f"{os.path.basename(c):24} slot0 {score(ck['student'], S):.4f}")


if __name__ == "__main__":
    main()
