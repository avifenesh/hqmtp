#!/usr/bin/env python3
"""Generate a fresh-topic 'live traffic' stream through the serving path — the OOD
test set for adaptive-vs-static draft pruning. Run on the box next to chat_proxy."""
import json
import os
import time
import urllib.request

from chat_proxy import build_template

QS = [
    "Explain how a jet engine achieves thrust.",
    "Write a sonnet about deep-sea creatures.",
    "Describe the economics of coffee farming in Ethiopia.",
    "How does CRISPR gene editing work?",
    "Plan a 7-day cycling trip through the Alps.",
    "Explain the rules of cricket to an American.",
    "What caused the 2008 financial crisis?",
    "Write a recipe for Georgian khachapuri.",
    "How do noise-cancelling headphones work?",
    "Describe the history of the Silk Road.",
    "Explain quantum entanglement without math.",
    "Write a product review for a fictional smartwatch.",
    "How are violins made?",
    "Explain the offside rule in football.",
    "What is the water cycle and why does it matter?",
    "Write a cover letter for a marine biologist job.",
    "How do vaccines create immunity?",
    "Describe daily life in ancient Rome.",
    "Explain how GPS positioning works.",
    "Write a short ghost story set in a lighthouse.",
    "How does a nuclear reactor stay controlled?",
    "Explain inflation to a teenager.",
    "Describe the process of making sourdough bread.",
    "How do octopuses camouflage?",
]


def complete(prompt, temperature):
    body = {"model": "hqmtp-9b", "prompt": prompt, "max_tokens": 700,
            "temperature": temperature, "top_p": 0.95, "stop": ["<|im_end|>"]}
    for attempt in range(10):
        try:
            r = urllib.request.urlopen(urllib.request.Request(
                "http://127.0.0.1:8014/v1/completions", data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"}), timeout=1800)
            return json.loads(r.read()).get("text", "")
        except Exception:
            if attempt == 9:
                raise
            time.sleep(30)


def main():
    tpl = build_template("/data/ai-ml/hf-models/qwen35-9b-hf/chat_template.jinja")
    out = "/opt/dlami/nvme/hqmtp-ood-stream"
    os.makedirs(out, exist_ok=True)
    for i, q in enumerate(QS):
        path = f"{out}/ood-{i:02d}.txt"
        if os.path.exists(path):
            continue
        p = tpl.render(messages=[{"role": "user", "content": q}], tools=None,
                       add_generation_prompt=True, enable_thinking=True)
        txt = complete(p, 0.0 if i % 2 == 0 else 0.7)
        open(path, "w").write(p + txt + "<|im_end|>")
        print(f"gen {i}", flush=True)
    print("OOD-GEN-DONE")


if __name__ == "__main__":
    main()
