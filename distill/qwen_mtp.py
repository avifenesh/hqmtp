"""Qwen3.5 MTP head in torch — teacher for distillation (bw24 spec.rs conventions).

Chain semantics (predecessor pairing, bw24 2026-07-04 convention):
  row i of the draft KV carries (token x_i, trunk hidden h_{i-1}), rope position i+1.
  chain step j: input (tok, h_seed) -> x = fc([enorm(embed(tok)); hnorm(h_seed)]) ->
  one full-attn transformer layer over the draft KV (append row, rope pos p+1+j) ->
  h_nextn; logits = lm_head(norm(h_nextn)). Greedy pick feeds the next step.

Rope: partial rotary (factor 0.25 -> 64 of head_dim 256), theta 1e7. mrope sections
collapse to standard rope for text-only (equal positions). Pairing convention is a
flag (HF rotate_half vs interleaved) — the engine-parity gate picks the right one.
"""
import json
import os

import torch
import torch.nn.functional as F

# model-generic: point at another checkpoint (e.g. the 27B) via env
STCT_DIR = os.environ.get("HQMTP_TEACHER_DIR",
                          "/data/ai-ml/hf-models/qwen35-9b-nvfp4-st-ct")
BASE_CFG = os.environ.get("HQMTP_BASE_CFG",
                          "/data/ai-ml/hf-models/qwen35-9b-hf/config.json")


def _load_tensors(prefixes, model_dir=STCT_DIR):
    from safetensors import safe_open
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
    wmap = idx["weight_map"]
    out, handles = {}, {}
    for name, shard in wmap.items():
        if not any(name.startswith(p) for p in prefixes):
            continue
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(model_dir, shard),
                                       framework="pt", device="cpu")
        out[name] = handles[shard].get_tensor(name)
    return out


def rms_norm(x, w, eps):
    v = x.float()
    v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    return (v * w.float()).to(x.dtype)


class TeacherHead(torch.nn.Module):
    """The co-trained mtp.* block, loaded verbatim from the st-ct checkpoint."""

    def __init__(self, device="cuda", dtype=torch.bfloat16, interleaved_rope=False):
        super().__init__()
        top = json.load(open(BASE_CFG))
        cfg = top.get("text_config", top)
        self.eps = cfg["rms_norm_eps"]
        self.n_q = cfg["num_attention_heads"]        # 9B: 16, 27B: 24
        self.n_kv = cfg["num_key_value_heads"]       # 4
        self.hd = cfg["head_dim"]                    # 256
        rp = cfg.get("rope_parameters") or cfg
        self.rot = int(self.hd * rp.get("partial_rotary_factor",
                                        cfg.get("partial_rotary_factor", 1.0)))
        self.theta = float(rp.get("rope_theta") or 1e7)
        self.interleaved_rope = interleaved_rope
        t = _load_tensors(("mtp.", "lm_head.", "model.language_model.embed_tokens."))
        g = lambda n: t[n].to(device=device, dtype=dtype)
        # Qwen3.5 RMSNorm convention: checkpoint stores ZERO-CENTERED norm weights,
        # forward multiplies by (1 + w) — HF modeling_qwen3_5:753, engine hf_mapping.rs
        # TransformKind::NormPlusOne. Fold the +1 at load.
        gn = lambda n: (t[n].float() + 1.0).to(device=device, dtype=dtype)
        self.embed = g("model.language_model.embed_tokens.weight")
        self.lm_head = g("lm_head.weight")
        self.enorm = gn("mtp.pre_fc_norm_embedding.weight")
        self.hnorm = gn("mtp.pre_fc_norm_hidden.weight")
        self.fc = g("mtp.fc.weight")
        self.final_norm = gn("mtp.norm.weight")
        L = "mtp.layers.0."
        self.in_ln = gn(L + "input_layernorm.weight")
        self.post_ln = gn(L + "post_attention_layernorm.weight")
        self.wq = g(L + "self_attn.q_proj.weight")
        self.wk = g(L + "self_attn.k_proj.weight")
        self.wv = g(L + "self_attn.v_proj.weight")
        self.wo = g(L + "self_attn.o_proj.weight")
        self.q_norm = gn(L + "self_attn.q_norm.weight")
        self.k_norm = gn(L + "self_attn.k_norm.weight")
        self.w_gate = g(L + "mlp.gate_proj.weight")
        self.w_up = g(L + "mlp.up_proj.weight")
        self.w_down = g(L + "mlp.down_proj.weight")
        inv = 1.0 / (self.theta ** (torch.arange(0, self.rot, 2).float() / self.rot))
        self.register_buffer("inv_freq", inv.to(device))
        self.device_ = device
        self.dtype_ = dtype

    def _rope(self, x, pos):
        """x [T, n_heads, hd]; pos [T] absolute rope positions. Partial rotary."""
        rot = self.rot
        xr, xp = x[..., :rot], x[..., rot:]
        ang = pos.float()[:, None] * self.inv_freq[None, :]          # [T, rot/2]
        cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]      # [T, 1, rot/2]
        if self.interleaved_rope:
            x1, x2 = xr[..., 0::2], xr[..., 1::2]
            o1 = x1 * cos - x2 * sin
            o2 = x2 * cos + x1 * sin
            out = torch.stack((o1, o2), dim=-1).flatten(-2)
        else:  # HF rotate_half: pairs (i, i+rot/2)
            x1, x2 = xr[..., : rot // 2], xr[..., rot // 2:]
            o1 = x1 * cos - x2 * sin
            o2 = x2 * cos + x1 * sin
            out = torch.cat((o1, o2), dim=-1)
        return torch.cat((out.to(x.dtype), xp), dim=-1)

    def _mix_input(self, tokens, pred_hiddens):
        """tokens [T] long, pred_hiddens [T, d] -> block input x [T, d]."""
        e = self.embed[tokens]
        e = rms_norm(e, self.enorm, self.eps)
        h = rms_norm(pred_hiddens.to(self.dtype_), self.hnorm, self.eps)
        return torch.cat((e, h), dim=-1) @ self.fc.T

    def _qkv(self, x, pos):
        """attn_output_gate arch: q_proj packs per-head [query(hd); gate(hd)] pairs —
        view [T, n_q, 2*hd], chunk last dim (HF modeling_qwen3_5:687)."""
        T = x.shape[0]
        xin = rms_norm(x, self.in_ln, self.eps)
        qg = (xin @ self.wq.T).view(T, self.n_q, 2 * self.hd)
        q, gate = qg.chunk(2, dim=-1)
        k = (xin @ self.wk.T).view(T, self.n_kv, self.hd)
        v = (xin @ self.wv.T).view(T, self.n_kv, self.hd)
        q = rms_norm(q, self.q_norm, self.eps)
        k = rms_norm(k, self.k_norm, self.eps)
        q = self._rope(q, pos)
        k = self._rope(k, pos)
        return xin, q, k, v, gate.reshape(T, -1)

    def kv_fill(self, tokens, pred_hiddens, pos0):
        """Build draft-KV rows for context tokens (K/V only). Rope pos = pos0+i+1."""
        x = self._mix_input(tokens, pred_hiddens)
        pos = torch.arange(pos0 + 1, pos0 + 1 + len(tokens), device=x.device)
        _, _, k, v, _ = self._qkv(x, pos)
        return k, v   # [T, n_kv, hd] each

    def chain_step(self, tok, h_seed, rope_pos, kv):
        """One draft step. kv = (K [N, n_kv, hd], V [N, n_kv, hd]) — appended in place.
        Returns (logits [vocab], h_nextn [d], new_kv)."""
        x = self._mix_input(torch.tensor([tok], device=self.device_),
                            h_seed.unsqueeze(0))
        pos = torch.tensor([rope_pos], device=self.device_)
        xin, q, k, v, gate = self._qkv(x, pos)
        K = torch.cat((kv[0], k), dim=0) if kv is not None else k
        V = torch.cat((kv[1], v), dim=0) if kv is not None else v
        # GQA attention over all rows (query = the single new row, causal trivially)
        rep = self.n_q // self.n_kv
        Ke = K.repeat_interleave(rep, dim=1)                     # [N, n_q, hd]
        Ve = V.repeat_interleave(rep, dim=1)
        att = torch.einsum("qhd,nhd->hn", q.float(), Ke.float()) / (self.hd ** 0.5)
        w = att.softmax(-1)
        o = torch.einsum("hn,nhd->hd", w, Ve.float()).to(self.dtype_)
        # attn_output_gate: sigmoid gate (from q_proj's per-head second half) before o_proj
        gated = o.reshape(1, -1) * torch.sigmoid(gate.float()).to(self.dtype_)
        attn_out = gated @ self.wo.T
        h1 = x + attn_out
        hin = rms_norm(h1, self.post_ln, self.eps)
        mlp = (F.silu(hin @ self.w_gate.T) * (hin @ self.w_up.T)) @ self.w_down.T
        h_nextn = (h1 + mlp)[0]
        logits = rms_norm(h_nextn.unsqueeze(0), self.final_norm, self.eps) @ self.lm_head.T
        return logits[0].float(), h_nextn, (K, V)

    @torch.no_grad()
    def draft(self, tokens, hiddens, p, k_draft, kv_ctx):
        """Draft k tokens at eval position p (context tokens[0..=p]), given the
        prefilled context kv (rows [0..p)). Mirrors the engine chain exactly."""
        tok = int(tokens[p])
        h_seed = hiddens[p - 1]
        kv = kv_ctx
        drafts = []
        for j in range(k_draft):
            logits, h_nextn, kv = self.chain_step(tok, h_seed, p + 1 + j, kv)
            tok = int(logits.argmax())
            drafts.append(tok)
            h_seed = h_nextn
        return drafts
