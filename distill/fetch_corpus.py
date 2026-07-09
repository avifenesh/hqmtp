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

    # Code: shallow-clone real repos (the-stack is gated). Mixed languages.
    import glob as g
    import subprocess
    repos = [("https://github.com/python/cpython", ["Lib/**/*.py"]),
             ("https://github.com/ggml-org/llama.cpp", ["src/**/*.cpp", "src/**/*.c", "ggml/**/*.c", "ggml/**/*.cpp", "ggml/**/*.cu"]),
             ("https://github.com/tokio-rs/tokio", ["tokio/src/**/*.rs"]),
             ("https://github.com/redis/redis", ["src/**/*.c"])]
    tmp = os.path.join(args.out, "_repos")
    files = []
    for url, pats in repos:
        name = url.rsplit("/", 1)[1]
        dst = os.path.join(tmp, name)
        if not os.path.isdir(dst):
            subprocess.run(["git", "clone", "--depth", "1", url, dst], check=True,
                           capture_output=True)
        for p in pats:
            files += sorted(g.glob(os.path.join(dst, p), recursive=True))
    def code_stream():
        for f in files:
            try:
                yield {"text": open(f, errors="replace").read()}
            except OSError:
                continue
    pack(code_stream(), args.out, "code", 2 * n)
    web = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                       split="train", streaming=True)
    pack(web, args.out, "web", 2 * n)
    chat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft",
                        streaming=True)
    pack(chat, args.out, "chat", n, render=render_chat)
    print("FETCH-DONE")


if __name__ == "__main__":
    main()
