#!/usr/bin/env python3
"""Export the NATURAL MTP head with a TRIMMED hot-vocab lm_head as a standalone
draft GGUF — the clean FR-Spec baseline arm, loaded through the same
BW24_MTP_DRAFT path as the student (avoids the in-engine bf16 row-gather path
whose runs showed a corrupted plain-generate baseline).

Tensors come from the HF checkpoint, so all RMSNorm weights get the Qwen3.5
zero-centered +1 fold at export (engine GGUF loading applies no NormPlusOne).
No rope permutation: the natural head was trained in the engine's own layout.

Usage (box):
  export_frtrim.py --model /data/ai-ml/hf-models/qwen35-9b-hf \
      --hot /opt/dlami/nvme/hqmtp-data2/hot32k.pt --out frtrim-draft.gguf
"""
import argparse
import json
import os

import numpy as np
import torch
import gguf


def load_tensors(model_dir, prefixes):
    from safetensors import safe_open
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
    out, handles = {}, {}
    for name, shard in idx["weight_map"].items():
        if not any(name.startswith(p) for p in prefixes):
            continue
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(model_dir, shard),
                                       framework="pt", device="cpu")
        out[name] = handles[shard].get_tensor(name)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--hot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ffn-keep", help="pt file with LongTensor channel idx to keep "
                    "(activation-ranked) — the trim++ arm: physical FFN prune, no training")
    ap.add_argument("--q4", action="store_true", help="Q4_0 all matrices")
    ap.add_argument("--nvfp4", action="store_true", help="NVFP4 all matrices")
    ap.add_argument("--mixed", action="store_true",
                    help="NVFP4 lm_head + Q4_0 block (engine-lane winning combo)")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.model, "config.json")))
    tc = cfg.get("text_config", cfg)
    d = tc["hidden_size"]
    n_q, n_kv, hd = tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
    rp = tc.get("rope_parameters") or tc
    rot = int(hd * rp.get("partial_rotary_factor", tc.get("partial_rotary_factor", 1.0)))
    theta = float(rp.get("rope_theta") or 1e7)

    t = load_tensors(args.model, ("mtp.", "lm_head."))
    hot = torch.load(args.hot, weights_only=False)["hot_ids"].to(torch.int32)
    g = lambda n: t[n].float()
    gn = lambda n: t[n].float() + 1.0          # zero-centered norm fold
    L = "mtp.layers.0."

    w = gguf.GGUFWriter(args.out, "qwen35")
    w.add_uint32("qwen35.block_count", 1)
    w.add_uint32("qwen35.nextn_predict_layers", 1)
    w.add_uint32("qwen35.embedding_length", d)
    w.add_uint32("qwen35.attention.head_count", n_q)
    w.add_uint32("qwen35.attention.head_count_kv", n_kv)
    w.add_uint32("qwen35.attention.key_length", hd)
    w.add_uint32("qwen35.attention.value_length", hd)
    w.add_uint32("qwen35.feed_forward_length", t[L + "mlp.gate_proj.weight"].shape[0])  # dims from tensors at load; kv only informational
    w.add_uint32("qwen35.vocab_size", t["lm_head.weight"].shape[0])
    w.add_uint32("qwen35.context_length", 262144)
    w.add_float32("qwen35.attention.layer_norm_rms_epsilon", tc["rms_norm_eps"])
    w.add_float32("qwen35.rope.freq_base", theta)
    w.add_uint32("qwen35.rope.dimension_count", rot)

    QT = gguf.GGMLQuantizationType.Q4_0 if (args.q4 or args.mixed) \
        else gguf.GGMLQuantizationType.Q8_0

    def q8(name, x):
        a = np.ascontiguousarray(x.numpy().astype(np.float32))
        if args.nvfp4 or (args.mixed and name == "output.weight"):
            from nvfp4_pack import quantize_nvfp4
            qb = quantize_nvfp4(a)
            w.add_tensor(name, qb, raw_shape=qb.shape,
                         raw_dtype=gguf.GGMLQuantizationType.NVFP4)
            return
        w.add_tensor(name, gguf.quants.quantize(a, QT),
                     raw_shape=gguf.quant_shape_to_byte_shape(a.shape, QT),
                     raw_dtype=QT)

    def f32(name, x):
        w.add_tensor(name, np.ascontiguousarray(x.numpy().astype(np.float32)))

    f32("blk.0.nextn.enorm.weight", gn("mtp.pre_fc_norm_embedding.weight"))
    f32("blk.0.nextn.hnorm.weight", gn("mtp.pre_fc_norm_hidden.weight"))
    q8("blk.0.nextn.eh_proj.weight", g("mtp.fc.weight"))
    f32("blk.0.attn_norm.weight", gn(L + "input_layernorm.weight"))
    q8("blk.0.attn_q.weight", g(L + "self_attn.q_proj.weight"))
    q8("blk.0.attn_k.weight", g(L + "self_attn.k_proj.weight"))
    q8("blk.0.attn_v.weight", g(L + "self_attn.v_proj.weight"))
    q8("blk.0.attn_output.weight", g(L + "self_attn.o_proj.weight"))
    f32("blk.0.attn_q_norm.weight", gn(L + "self_attn.q_norm.weight"))
    f32("blk.0.attn_k_norm.weight", gn(L + "self_attn.k_norm.weight"))
    f32("blk.0.post_attention_norm.weight", gn(L + "post_attention_layernorm.weight"))
    wg, wu, wd = g(L + "mlp.gate_proj.weight"), g(L + "mlp.up_proj.weight"), \
        g(L + "mlp.down_proj.weight")
    if args.ffn_keep:
        keep = torch.load(args.ffn_keep, weights_only=False)
        keep = keep[: (len(keep) // 32) * 32].sort().values      # Q8_0 %32 alignment
        wg, wu, wd = wg[keep], wu[keep], wd[:, keep]
        print(f"ffn pruned to {len(keep)} channels")
    q8("blk.0.ffn_gate.weight", wg)
    q8("blk.0.ffn_up.weight", wu)
    q8("blk.0.ffn_down.weight", wd)
    f32("blk.0.nextn.shared_head_norm.weight", gn("mtp.norm.weight"))
    q8("output.weight", g("lm_head.weight")[hot.long()])
    w.add_tensor("d2t", hot.numpy())

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out}: natural block + {len(hot)}-row trimmed head (FR-trim arm)")


if __name__ == "__main__":
    main()
