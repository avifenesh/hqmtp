#!/usr/bin/env python3
"""Corpus v2: own-output training data generated through the SERVING regime.

v1 gap (found via engine e2e, 2026-07-12): v1 docs were sampled-only (t0.7),
thinking DISABLED, long multi-turn — live serving is thinking-ON, includes greedy,
and short generations dominate early positions. Student acceptance collapsed live
(13-18%) despite 0.70 slot0 on v1-style text.

v2: thinking ENABLED, 50% greedy / 50% t0.7 top_p 0.95, 2/3 single-shot + 1/3
multi-turn, fresh seeds (ultrachat offset past v1's range + code tasks), 1024-token
replies. Same output format as v1 (plain text, dir-mode extraction ready).

Usage (box):
  gen_corpus2.py --out /opt/dlami/nvme/hqmtp-gen2-corpus --docs 2400
"""
import argparse
import concurrent.futures as cf
import glob
import json
import os
import random
import urllib.request

from chat_proxy import build_template

TEMPLATE = os.environ.get("HQMTP_TEMPLATE",
                          "/data/ai-ml/hf-models/qwen35-9b-hf/chat_template.jinja")
SERVER = os.environ.get("HQMTP_SERVER", "http://127.0.0.1:8014")

FOLLOWUPS = [
    "Now improve your answer: fix any mistake and make it more concrete.",
    "Continue: add error handling / edge cases to what you wrote.",
    "Summarize what you produced and list two weaknesses.",
    "Give a shorter, sharper version of your answer.",
]


def complete(prompt, max_tokens, temperature):
    body = {"model": "hqmtp-9b", "prompt": prompt, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": 0.95 if temperature > 0 else 1.0,
            "stop": ["<|im_end|>"]}
    # retry through server startup (model load ~2min) and transient hiccups
    import time
    for attempt in range(8):
        try:
            r = urllib.request.urlopen(urllib.request.Request(
                f"{SERVER}/v1/completions", data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"}), timeout=1800)
            return json.loads(r.read()).get("text", "")
        except Exception:
            if attempt == 7:
                raise
            time.sleep(30)


def build_seeds(n, code_dir):
    seeds = []
    from datasets import load_dataset
    chat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft",
                        streaming=True)
    it = iter(chat)
    for _ in range(2000):          # skip v1's seed range — fresh questions only
        next(it)
    for _ in range(int(n * 0.6)):
        ex = next(it)
        msgs = ex.get("messages") or []
        if msgs and msgs[0]["role"] == "user":
            seeds.append(msgs[0]["content"][:2000])
    files = sorted(glob.glob(os.path.join(code_dir, "code-*.txt")))
    random.seed(23)                # different slices than v1 (seed 11)
    tasks = ["Review this code and explain its three biggest risks:",
             "Refactor this code for clarity, show the full result:",
             "Write unit tests for the following code:",
             "Explain what this code does and where it could fail:",
             "Find the bugs in this code and show minimal fixes:"]
    while len(seeds) < n and files:
        f = random.choice(files)
        text = open(f, errors="replace").read()
        start = random.randint(0, max(0, len(text) - 4000))
        seeds.append(f"{random.choice(tasks)}\n```\n{text[start:start+3500]}\n```")
    random.shuffle(seeds)
    return seeds[:n]


def gen_doc(args_tuple):
    i, seed, out_dir, max_tokens, template = args_tuple
    path = os.path.join(out_dir, f"gen2-{i:04d}.txt")
    if os.path.exists(path):
        return f"skip {i}"
    temperature = 0.0 if i % 2 == 0 else 0.7     # half greedy, half sampled
    multiturn = i % 3 == 0
    msgs = [{"role": "user", "content": seed}]
    doc = ""
    for t in range(3 if multiturn else 1):
        prompt = template.render(messages=msgs, tools=None,
                                 add_generation_prompt=True, enable_thinking=True)
        reply = complete(prompt, max_tokens, temperature)
        doc = prompt + reply + "<|im_end|>"
        msgs.append({"role": "assistant", "content": reply.strip()})
        if t < 2 and multiturn:
            msgs.append({"role": "user", "content": random.choice(FOLLOWUPS)})
    open(path, "w").write(doc)
    return f"done {i} ({len(doc)} chars)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--docs", type=int, default=2400)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--code-dir", default="/opt/dlami/nvme/hqmtp-corpus")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    template = build_template(TEMPLATE)
    seeds = build_seeds(args.docs, args.code_dir)
    print(f"[seeds] {len(seeds)}")
    jobs = [(i, s, args.out, args.max_tokens, template) for i, s in enumerate(seeds)]
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for msg in ex.map(gen_doc, jobs):
            done += 1
            if done % 50 == 0:
                print(f"[gen] {done}/{len(jobs)}", flush=True)
    print("GEN2-DONE")


if __name__ == "__main__":
    main()
