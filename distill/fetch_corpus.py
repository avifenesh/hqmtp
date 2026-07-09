#!/usr/bin/env python3
"""Fetch + pack the distillation training corpus (streamed, no full downloads).

Mix (DISTILL.md scaling round): ~40% code (the-stack-smol), ~40% web text
(fineweb sample), ~20% chat (ultrachat rendered with qwen im_start markup — the
gen-dist eval showed a chat-markup domain gap). Packs documents into ~250KB text
files (~64k tokens each, the engine extraction cap) under --out.

Usage: fetch_corpus.py --out /opt/dlami/nvme/hqmtp-corpus --files-per-source 40
"""
import argparse
import os

from datasets import load_dataset


def pack(stream, out_dir, prefix, n_files, target_bytes=250_000, render=None):
    os.makedirs(out_dir, exist_ok=True)
    buf, size, fi = [], 0, 0
    written = 0
    for ex in stream:
        text = render(ex) if render else ex.get("text") or ex.get("content") or ""
        if not text or len(text) < 200:
            continue
        buf.append(text)
        size += len(text)
        if size >= target_bytes:
            path = os.path.join(out_dir, f"{prefix}-{fi:04d}.txt")
            open(path, "w").write("\n\n".join(buf))
            written += 1
            buf, size = [], 0
            fi += 1
            if fi >= n_files:
                break
    print(f"{prefix}: {written} files")


def render_chat(ex):
    """ultrachat messages -> qwen chat markup (thinking off), matching serve format."""
    out = []
    for m in ex.get("messages", []):
        out.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--files-per-source", type=int, default=40)
    args = ap.parse_args()
    n = args.files_per_source

    stack = load_dataset("bigcode/the-stack-smol", data_dir="data/python",
                         split="train", streaming=True)
    pack(stack, args.out, "code-py", n)
    stack_c = load_dataset("bigcode/the-stack-smol", data_dir="data/c",
                           split="train", streaming=True)
    pack(stack_c, args.out, "code-c", n // 2)
    stack_rs = load_dataset("bigcode/the-stack-smol", data_dir="data/rust",
                            split="train", streaming=True)
    pack(stack_rs, args.out, "code-rs", n // 2)
    web = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                       split="train", streaming=True)
    pack(web, args.out, "web", 2 * n)
    chat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft",
                        streaming=True)
    pack(chat, args.out, "chat", n, render=render_chat)
    print("FETCH-DONE")


if __name__ == "__main__":
    main()
