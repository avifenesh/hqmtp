#!/usr/bin/env python3
"""Export StudentSV -> standalone draft GGUF for bw24 (BW24_MTP_DRAFT).

Tensor map (engine MtpHead::load_draft naming, blk.0.* with block_count=1,
nextn_predict_layers=1):
  enorm/hnorm            -> blk.0.nextn.{enorm,hnorm}.weight        [4096]
  fc                     -> blk.0.nextn.eh_proj.weight              [8192 -> 2048]
  in_ln                  -> blk.0.attn_norm.weight                  [2048]
  wq (per-head [q;gate]) -> blk.0.attn_q.weight                     [2048 -> 4096]
  wk / wv                -> blk.0.attn_{k,v}.weight                 [2048 -> 512]
  wo                     -> blk.0.attn_output.weight                [2048 -> 2048]
  q_norm / k_norm        -> blk.0.attn_{q,k}_norm.weight            [256]
  post_ln                -> blk.0.post_attention_norm.weight        [2048]
  gate_p/up_p/down_p     -> blk.0.ffn_{gate,up,down}.weight
  out_up                 -> blk.0.nextn.out_up.weight               [2048 -> 4096] (NEW)
  final                  -> blk.0.nextn.shared_head_norm.weight     [4096]
  lm_head                -> output.weight                           [4096 -> 32768]
  d2t                    -> d2t (I32 [32768], draft row -> target token id)

ROPE LAYOUT: the student trains with INTERLEAVED pairing (pair j = dims 2j,2j+1);
the engine kernel is NEOX half-split (pair j = dims j, j+rot/2). Exported wq/wk rows
(and q_norm/k_norm entries) are permuted inside the roped block [0..rot) so that
neox-on-permuted == interleaved-on-trained: new[j]=old[2j], new[rot/2+j]=old[2j+1].
Attention dots are invariant (q and k permuted identically); v/gate untouched.

2D matmul weights are quantized Q8_0 (the format the teacher head runs in-engine via
the loader law -> bandwidth-comparable e2e numbers); norms/d2t stay F32/I32.

Usage (box):
  python export_sv.py --ckpt checkpoints/sv-kd24-ce.pt \
      --hot /opt/dlami/nvme/hqmtp-data2/hot32k.pt --out sv-draft.gguf
"""
import argparse

import numpy as np
import torch
import gguf

ROT = 64
HD = 256
N_Q = 8
N_KV = 2
D_S = 2048
D = 4096          # n_embd of the TARGET model (--d overrides: 27B = 5120)
D_FF = 6144


def neox_permute_rows(w, n_heads, block, q_gate=False):
    """Permute the first ROT rows of each head's query block: new[j]=old[2j] (j<ROT/2),
    new[ROT/2+j]=old[2j+1]. w [n_heads*block, in]. q_gate: block = [q(HD); gate(HD)],
    only the q part's roped rows move."""
    w = w.clone()
    perm = torch.cat((torch.arange(0, ROT, 2), torch.arange(1, ROT, 2)))
    for h in range(n_heads):
        base = h * block
        w[base: base + ROT] = w[base + perm]
    return w


def neox_permute_vec(v):
    perm = torch.cat((torch.arange(0, ROT, 2), torch.arange(1, ROT, 2)))
    out = v.clone()
    out[:ROT] = v[perm]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-permute", action="store_true", help="debug: skip neox permute")
    ap.add_argument("--f16", action="store_true", help="debug: F16 instead of Q8_0")
    ap.add_argument("--d", type=int, default=D, help="target n_embd (27B: 5120)")
    args = ap.parse_args()
    global D
    D = args.d

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state"]
    hot = torch.load(args.hot, weights_only=False)["hot_ids"].to(torch.int32)
    g = lambda n: sd[n].float()

    wq = g("wq.weight")                       # [N_Q*2*HD, D_S] per-head [q;gate]
    wk = g("wk.weight")                       # [N_KV*HD, D_S]
    qn = g("q_norm.w")
    kn = g("k_norm.w")
    if not args.no_permute:
        wq = neox_permute_rows(wq, N_Q, 2 * HD, q_gate=True)
        wk = neox_permute_rows(wk, N_KV, HD)
        qn = neox_permute_vec(qn)
        kn = neox_permute_vec(kn)

    w = gguf.GGUFWriter(args.out, "qwen35")
    w.add_uint32("qwen35.block_count", 1)
    w.add_uint32("qwen35.nextn_predict_layers", 1)
    w.add_uint32("qwen35.embedding_length", D)
    w.add_uint32("qwen35.attention.head_count", N_Q)
    w.add_uint32("qwen35.attention.head_count_kv", N_KV)
    w.add_uint32("qwen35.attention.key_length", HD)
    w.add_uint32("qwen35.attention.value_length", HD)
    w.add_uint32("qwen35.feed_forward_length", D_FF)
    w.add_uint32("qwen35.vocab_size", 248320)
    w.add_uint32("qwen35.context_length", 262144)
    w.add_float32("qwen35.attention.layer_norm_rms_epsilon", 1e-6)
    w.add_float32("qwen35.rope.freq_base", 1e7)
    w.add_uint32("qwen35.rope.dimension_count", ROT)

    def q8(name, t):
        d = np.ascontiguousarray(t.numpy().astype(np.float32))
        if args.f16:
            w.add_tensor(name, d.astype(np.float16))
        else:
            w.add_tensor(name, gguf.quants.quantize(d, gguf.GGMLQuantizationType.Q8_0),
                         raw_shape=gguf.quant_shape_to_byte_shape(
                             d.shape, gguf.GGMLQuantizationType.Q8_0),
                         raw_dtype=gguf.GGMLQuantizationType.Q8_0)

    def f32(name, t):
        w.add_tensor(name, np.ascontiguousarray(t.numpy().astype(np.float32)))

    f32("blk.0.nextn.enorm.weight", g("enorm.w"))
    f32("blk.0.nextn.hnorm.weight", g("hnorm.w"))
    q8("blk.0.nextn.eh_proj.weight", g("fc.weight"))
    f32("blk.0.attn_norm.weight", g("in_ln.w"))
    q8("blk.0.attn_q.weight", wq)
    q8("blk.0.attn_k.weight", wk)
    q8("blk.0.attn_v.weight", g("wv.weight"))
    q8("blk.0.attn_output.weight", g("wo.weight"))
    f32("blk.0.attn_q_norm.weight", qn)
    f32("blk.0.attn_k_norm.weight", kn)
    f32("blk.0.post_attention_norm.weight", g("post_ln.w"))
    q8("blk.0.ffn_gate.weight", g("gate_p.weight"))
    q8("blk.0.ffn_up.weight", g("up_p.weight"))
    q8("blk.0.ffn_down.weight", g("down_p.weight"))
    q8("blk.0.nextn.out_up.weight", g("out_up.weight"))
    f32("blk.0.nextn.shared_head_norm.weight", g("final.w"))
    q8("output.weight", g("lm_head.weight"))
    w.add_tensor("d2t", hot.numpy())

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out}: 32k-vocab StudentSV draft"
          f" ({'f16' if args.f16 else 'Q8_0'}, permute={'off' if args.no_permute else 'on'})")


if __name__ == "__main__":
    main()
