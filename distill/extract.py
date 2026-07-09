#!/usr/bin/env python3
"""Extraction + gates for the MTP-head distillation pipeline (see DISTILL.md).

Stages (each gated before the next is trusted):
  A. --gate-base: torch base forward greedy track vs an engine replay dump's targets.
     Arms: --arm bf16 (plain checkpoint) or --arm nvfp4 (dequantized st-ct weights
     swapped into the same model). bf16 must match near-exactly; nvfp4 is expected
     slightly lower (engine runs W4A8 activation-quantized kernels, torch runs
     weight-only dequant at bf16 compute) — the number is recorded, not asserted.
  B. --extract: teacher-forced pass over a corpus -> tokens, greedy track, pre-norm
     hidden stack saved to .pt for training (teacher head runs on-the-fly in train.py).

Run on the box:
  ~/.venvs/distill/bin/python extract.py --gate-base --arm bf16 \
      --corpus ~/hqmtp-eval/corpora/gen-dist-9b.txt \
      --dump ~/bw24/research/mtp-heal/g7e/gendist/replay-bf16-gendist.jsonl
"""
import argparse
import json
import os

import torch


BASE_DIR = "/data/ai-ml/hf-models/qwen35-9b-hf"
STCT_DIR = "/data/ai-ml/hf-models/qwen35-9b-nvfp4-st-ct"


def load_base(arm: str, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_DIR, dtype=torch.bfloat16, device_map=device)
    model.eval()
    if arm == "nvfp4":
        swap_nvfp4_weights(model)
    return tok, model


def swap_nvfp4_weights(model):
    """Overwrite the bf16 model's MLP weights with weight-exact dequantized NVFP4
    tensors from the st-ct checkpoint (96 stems, MLP-only per its recipe)."""
    from safetensors import safe_open
    from nvfp4 import dequant_nvfp4

    idx = json.load(open(os.path.join(STCT_DIR, "model.safetensors.index.json")))
    wmap = idx["weight_map"]
    stems = sorted({n[: -len(".weight_packed")] for n in wmap
                    if n.endswith(".weight_packed")})
    params = dict(model.named_parameters())
    handles = {}

    def get(name):
        shard = wmap[name]
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(STCT_DIR, shard),
                                       framework="pt", device="cuda")
        return handles[shard].get_tensor(name)

    swapped = 0
    for stem in stems:
        w = dequant_nvfp4(get(stem + ".weight_packed"),
                          get(stem + ".weight_scale"),
                          get(stem + ".weight_global_scale"))
        # st-ct namespace: model.language_model.layers.N...; CausalLM param namespace
        # may drop the language_model segment — try both.
        for cand in (stem + ".weight",
                     stem.replace("model.language_model.", "model.") + ".weight"):
            if cand in params:
                assert params[cand].shape == w.shape, f"{cand}: {params[cand].shape} vs {w.shape}"
                with torch.no_grad():
                    params[cand].copy_(w)
                swapped += 1
                break
        else:
            raise KeyError(f"no param for {stem} (tried both namespaces)")
    print(f"[swap] dequantized + swapped {swapped}/{len(stems)} stems")


@torch.no_grad()
def greedy_track(model, ids, chunk=2048):
    """bg[i] = argmax prediction for position i (i>=1), teacher-forced, chunked with
    a KV cache so memory stays bounded."""
    bg = torch.zeros(len(ids) + 1, dtype=torch.long)
    cache = None   # first forward builds the model's own hybrid cache class
    x = torch.tensor(ids, device="cuda").unsqueeze(0)
    for s in range(0, len(ids), chunk):
        e = min(s + chunk, len(ids))
        out = model(input_ids=x[:, s:e], past_key_values=cache, use_cache=True)
        bg[s + 1: e + 1] = out.logits[0].argmax(-1).cpu()
        cache = out.past_key_values
    return bg


def gate_base(args):
    tok, model = load_base(args.arm)
    text = open(args.corpus).read()
    ids = tok(text, add_special_tokens=True).input_ids
    rows = [json.loads(l) for l in open(os.path.expanduser(args.dump))]
    max_pos = max(r["pos"] for r in rows) + 8
    ids = ids[: min(len(ids), max_pos + 8, args.max_t)]
    print(f"[gate-base] arm={args.arm} corpus tokens={len(ids)} dump rows={len(rows)}")
    bg = greedy_track(model, ids)
    n = hit = 0
    mism = []
    for r in rows:
        p = r["pos"]
        if p + 1 >= len(bg):
            continue
        n += 1
        if int(bg[p + 1]) == r["targets"][0]:
            hit += 1
        elif len(mism) < 5:
            mism.append((p, int(bg[p + 1]), r["targets"][0]))
    print(f"[gate-base] slot0 target match: {hit}/{n} = {hit/max(n,1):.4f}")
    for p, a, b in mism:
        print(f"  mismatch pos={p} torch={a} engine={b}")
    return hit / max(n, 1)


def extract(args):
    tok, model = load_base(args.arm)
    text = open(args.corpus).read()
    ids = tok(text, add_special_tokens=True).input_ids[: args.max_t]
    print(f"[extract] arm={args.arm} {len(ids)} tokens -> {args.out}")
    cache = None   # first forward builds the model's own hybrid cache class
    x = torch.tensor(ids, device="cuda").unsqueeze(0)
    hiddens = torch.empty(len(ids), model.config.hidden_size, dtype=torch.bfloat16)
    bg = torch.zeros(len(ids) + 1, dtype=torch.long)
    chunk = args.chunk
    with torch.no_grad():
        for s in range(0, len(ids), chunk):
            e = min(s + chunk, len(ids))
            out = model(input_ids=x[:, s:e], past_key_values=cache,
                        use_cache=True, output_hidden_states=True)
            # hidden_states[-1] = last decoder layer output, PRE final norm — the trunk
            # hidden the MTP head pairs with (bw24 convention).
            hiddens[s:e] = out.hidden_states[-1][0].to(torch.bfloat16).cpu()
            bg[s + 1: e + 1] = out.logits[0].argmax(-1).cpu()
            cache = out.past_key_values
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"ids": torch.tensor(ids), "bg": bg, "hiddens": hiddens,
                "arm": args.arm, "corpus": args.corpus}, args.out)
    print(f"[extract] saved ids/bg/hiddens {tuple(hiddens.shape)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["bf16", "nvfp4"], required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--gate-base", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--dump")
    ap.add_argument("--out")
    ap.add_argument("--max-t", type=int, default=200_000)
    ap.add_argument("--chunk", type=int, default=2048)
    args = ap.parse_args()
    if args.gate_base:
        gate_base(args)
    if args.extract:
        extract(args)


if __name__ == "__main__":
    main()
