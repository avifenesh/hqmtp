#!/usr/bin/env python3
"""Build the LONG-PROMPT benchmark set: 20 diverse 2-6k-token prompts (question text
only; the battery wraps them with BW24_CHAT). Sources: serving-regime eval docs
(cleaned) + code files, varied instructions — long-context coverage to match what
MT-Bench gives short prompts."""
import glob
import os
import re

from transformers import AutoTokenizer

OUT = "/tmp/longset-prompts"
tok = AutoTokenizer.from_pretrained("/data/ai-ml/hf-models/qwen35-9b-hf")

DOC_TASKS = [
    "Summarize the key points of this document, then critique its weakest argument in detail.",
    "Write an executive summary of the following material, then list five follow-up questions.",
    "Identify every factual claim in this document and rate your confidence in each.",
    "Rewrite the following content as a structured tutorial with numbered steps.",
    "Compare the perspectives in this document and propose a synthesis.",
]
CODE_TASKS = [
    "Review this code thoroughly: correctness, performance, and style. Show fixes.",
    "Write comprehensive unit tests for the following code.",
    "Refactor this code for clarity and explain each change.",
    "Find every potential bug or edge case in this code and rank by severity.",
    "Document this code: docstrings, a module overview, and usage examples.",
]


def clean(t):
    return re.sub(r"<\|im_start\|>|<\|im_end\|>|<think>|</think>", "", t)


def take_tokens(text, n):
    ids = tok(text).input_ids[:n]
    return tok.decode(ids)


def main():
    os.makedirs(OUT, exist_ok=True)
    docs = sorted(glob.glob("/opt/dlami/nvme/hqmtp-gen2-eval/*.txt"))
    codes = sorted(glob.glob("/opt/dlami/nvme/hqmtp-corpus/code-*.txt"))
    n = 0
    sizes = [2000, 3000, 4000, 5000, 6000]
    for i in range(10):                                   # 10 doc prompts
        body = clean(open(docs[i * 3 % len(docs)]).read())
        body = take_tokens(body, sizes[i % 5] - 100)
        q = f"Here is a document:\n\n{body}\n\n{DOC_TASKS[i % 5]}"
        open(f"{OUT}/long-{n:02d}-doc.txt", "w").write(q)
        n += 1
    for i in range(10):                                   # 10 code prompts
        body = open(codes[i * 7 % len(codes)], errors="replace").read()
        body = take_tokens(body, sizes[i % 5] - 100)
        q = f"{CODE_TASKS[i % 5]}\n```\n{body}\n```"
        open(f"{OUT}/long-{n:02d}-code.txt", "w").write(q)
        n += 1
    lens = [len(tok(open(f).read()).input_ids) for f in sorted(glob.glob(f"{OUT}/*.txt"))]
    print(f"{n} prompts, token lens min {min(lens)} max {max(lens)} mean {sum(lens)//n}")


if __name__ == "__main__":
    main()
