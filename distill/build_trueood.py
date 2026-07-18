#!/usr/bin/env python3
"""Build the TRUE domain-shift suite (reviewer C1): text genuinely outside the
calibration register — non-English Wikipedia, biomedical abstracts, C4-style web.
~30 docs, 800-2500 tokens each, plain text for dir-mode extraction."""
import os

from datasets import load_dataset
from transformers import AutoTokenizer

OUT = "/opt/dlami/nvme/hqmtp-trueood-corpus"
tok = AutoTokenizer.from_pretrained("/data/ai-ml/hf-models/qwen35-9b-hf")


def take(text, lo=800, hi=2500):
    ids = tok(text).input_ids
    if len(ids) < lo:
        return None
    return tok.decode(ids[:hi])


def main():
    os.makedirs(OUT, exist_ok=True)
    n = 0

    def emit(name, text):
        nonlocal n
        t = take(text)
        if t is None:
            return False
        open(f"{OUT}/{name}.txt", "w").write(t)
        n += 1
        return True

    # 1) non-English Wikipedia: German, Chinese, Hebrew, Russian (3 each)
    for lang in ["de", "zh", "he", "ru"]:
        ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}",
                          split="train", streaming=True)
        got = 0
        for ex in ds:
            if got >= 3:
                break
            if emit(f"wiki-{lang}-{got}", ex["text"]):
                got += 1
        print(f"[wiki-{lang}] {got}", flush=True)

    # 2) biomedical abstracts (concatenate several per doc)
    ds = load_dataset("ccdv/pubmed-summarization", split="train", streaming=True)
    buf, got = [], 0
    for ex in ds:
        if got >= 6:
            break
        buf.append(ex["article"][:6000])
        if len(buf) == 1:
            if emit(f"pubmed-{got}", buf[0]):
                got += 1
            buf = []
    print(f"[pubmed] {got}", flush=True)

    # 3) C4-style web text
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    buf, got = [], 0
    for ex in ds:
        if got >= 6:
            break
        buf.append(ex["text"])
        joined = "\n\n".join(buf)
        if len(tok(joined).input_ids) >= 900:
            if emit(f"c4web-{got}", joined):
                got += 1
            buf = []
    print(f"[c4] {got}", flush=True)

    # 4) legal text (pile-of-law subset small stream)
    try:
        ds = load_dataset("pile-of-law/pile-of-law", "courtListener_opinions",
                          split="train", streaming=True)
        got = 0
        for ex in ds:
            if got >= 4:
                break
            if emit(f"legal-{got}", ex["text"]):
                got += 1
        print(f"[legal] {got}", flush=True)
    except Exception as e:
        print(f"[legal] skipped: {type(e).__name__}", flush=True)

    print(f"TRUEOOD-READY {n} docs", flush=True)


if __name__ == "__main__":
    main()
