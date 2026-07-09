#!/usr/bin/env python3
"""Build the frequency-ranked hot vocab for the small-vocab draft head (issue #25187
follow-through: not trim-at-load — a natively-trained small head).

Ranks token ids by frequency over the training corpus (the extraction token streams),
force-includes every special/markup token seen anywhere plus the tokenizer's special
ids, takes top --size. Saves hot_ids (d2t: draft idx -> target id) and reports the
coverage ceiling (fraction of bg targets inside the hot set) on the eval corpora —
the acceptance ceiling any trimmed/small-vocab head can reach.

Usage (box):
  build_hot_vocab.py --corpus-dir /opt/dlami/nvme/hqmtp-corpus --size 32768 \
      --out /opt/dlami/nvme/hqmtp-data2/hot32k.pt
"""
import argparse
import collections
import glob
import json
import os

import torch
from transformers import AutoTokenizer

BASE = "/data/ai-ml/hf-models/qwen35-9b-hf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--size", type=int, default=32768)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-metas", nargs="*", default=[
        "/opt/dlami/nvme/hqmtp-distill-data/eval-eval-code-64k.hiddens.meta.json",
        "/opt/dlami/nvme/hqmtp-distill-data/eval-gen-dist-9b.hiddens.meta.json"])
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    freq = collections.Counter()
    files = sorted(glob.glob(os.path.join(args.corpus_dir, "*.txt")))
    for f in files:
        freq.update(tok(open(f, errors="replace").read()).input_ids)
    print(f"[freq] {len(files)} files, {sum(freq.values())} tokens, "
          f"{len(freq)} distinct ids")

    forced = set(tok.all_special_ids)
    # everything the chat template emits + any high-id (>=248000) token seen in data
    forced |= set(tok("<|im_start|>system user assistant<|im_end|>\n<think></think>"
                      "<tool_call></tool_call><tool_response></tool_response>").input_ids)
    forced |= {i for i in freq if i >= 248000}
    hot = list(dict.fromkeys(
        list(forced) + [t for t, _ in freq.most_common() if t not in forced]))[: args.size]
    hot_ids = torch.tensor(sorted(hot), dtype=torch.long)
    torch.save({"hot_ids": hot_ids, "size": len(hot_ids),
                "corpus_tokens": sum(freq.values())}, args.out)
    print(f"[hot] {len(hot_ids)} ids -> {args.out}")

    hot_set = set(hot_ids.tolist())
    for m in args.eval_metas:
        meta = json.load(open(m))
        bg = meta["bg"][1:]
        cov = sum(1 for t in bg if t in hot_set) / len(bg)
        print(f"[coverage] {os.path.basename(m)}: {cov:.4f} "
              f"(acceptance ceiling for the small-vocab head)")


if __name__ == "__main__":
    main()
