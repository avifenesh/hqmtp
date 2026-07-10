#!/usr/bin/env python3
"""Generate the SV training corpus from THE MODEL ITSELF (user rule: draft heads train
on model-exact output only — anything else downgrades).

Seeds: ultrachat user questions (prompts only — outputs come from our model) + code
review/refactor tasks built from local code files. Each doc = chat-template-rendered
prompt + the st-ct model's OWN sampled continuation (temp 0.7 via bw24-server
/v1/completions; non-greedy = tokenwise decode path). A fraction of docs run 3 turns
(self-conditioning depth). Docs go to --out as plain text, ready for the existing
dir-mode hidden extraction.

Usage (box):
  gen_corpus.py --out /opt/dlami/nvme/hqmtp-gen-corpus --docs 300 --max-tokens 1536
"""
import argparse
import concurrent.futures as cf
import glob
import json
import os
import random
import urllib.request

from chat_proxy import build_template

TEMPLATE = "/data/ai-ml/hf-models/qwen35-9b-hf/chat_template.jinja"
SERVER = "http://127.0.0.1:8014"

FOLLOWUPS = [
    "Now improve your answer: fix any mistake and make it more concrete.",
    "Continue: add error handling / edge cases to what you wrote.",
    "Summarize what you produced and list two weaknesses.",
]


def complete(prompt, max_tokens):
    body = {"model": "hqmtp-9b", "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0.7, "top_p": 0.95, "stop": ["<|im_end|>"]}
    r = urllib.request.urlopen(urllib.request.Request(
        f"{SERVER}/v1/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}), timeout=1800)
    return json.loads(r.read()).get("text", "")


def build_seeds(n, code_dir):
    seeds = []
    from datasets import load_dataset
    chat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft",
                        streaming=True)
    it = iter(chat)
    for _ in range(int(n * 0.6)):
        ex = next(it)
        msgs = ex.get("messages") or []
        if msgs and msgs[0]["role"] == "user":
            seeds.append(msgs[0]["content"][:2000])
    files = sorted(glob.glob(os.path.join(code_dir, "code-*.txt")))
    random.seed(11)
    tasks = ["Review this code and explain its three biggest risks:",
             "Refactor this code for clarity, show the full result:",
             "Write unit tests for the following code:",
             "Explain what this code does and where it could fail:"]
    while len(seeds) < n and files:
        f = random.choice(files)
        text = open(f, errors="replace").read()
        start = random.randint(0, max(0, len(text) - 4000))
        seeds.append(f"{random.choice(tasks)}\n```\n{text[start:start+3500]}\n```")
    random.shuffle(seeds)
    return seeds[:n]


def gen_doc(args_tuple):
    i, seed, out_dir, max_tokens, template, multiturn = args_tuple
    path = os.path.join(out_dir, f"gen-{i:04d}.txt")
    if os.path.exists(path):
        return f"skip {i}"
    msgs = [{"role": "user", "content": seed}]
    doc = ""
    turns = 3 if multiturn else 1
    for t in range(turns):
        prompt = template.render(messages=msgs, tools=None,
                                 add_generation_prompt=True, enable_thinking=False)
        reply = complete(prompt, max_tokens)
        doc = prompt + reply + "<|im_end|>"
        msgs.append({"role": "assistant", "content": reply.strip()})
        if t + 1 < turns:
            msgs.append({"role": "user", "content": random.choice(FOLLOWUPS)})
    open(path, "w").write(doc)
    return f"done {i} ({len(doc)} chars)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--docs", type=int, default=300)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--code-dir", default="/opt/dlami/nvme/hqmtp-corpus")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    template = build_template(TEMPLATE)
    seeds = build_seeds(args.docs, args.code_dir)
    print(f"[seeds] {len(seeds)}")
    jobs = [(i, s, args.out, args.max_tokens, template, i % 3 == 0)
            for i, s in enumerate(seeds)]
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for msg in ex.map(gen_doc, jobs):
            done += 1
            if done % 20 == 0:
                print(f"[gen] {done}/{len(jobs)}", flush=True)
    print("GEN-DONE")


if __name__ == "__main__":
    main()
