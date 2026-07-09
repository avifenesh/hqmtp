"""Student draft heads for MTP-head distillation (DISTILL.md).

Interface (mirrors TeacherHead chain semantics, WINDOWED attention):
  context rows i carry (token x_i, trunk hidden h_{i-1}); a draft query at position p
  reads rows [max(0, p-W)..p) plus its own appended row. Window W is a training/eval
  tractability compromise (teacher targets are computed under the SAME window, so the
  distillation pairs are consistent; full-context engine integration comes later).

v1 scope: slot-0 (single draft step) — the acceptance lever. Chain training later.

Students:
  S1 "mlp-only":  fc -> norm -> MLP -> h (no attention; tests whether draft attention
                  matters at all for slot-0).
  S2 "half-width": teacher block shape at d_s=2048 (gated attention, 8/2 heads) with
                  an output up-projection back to n_embd for the shared lm_head.
  B0: same as S2 but trained from scratch on CE only (control — built in train.py by
      loss flags, not a separate class).
All students share the FROZEN base embed + lm_head (the deployment constraint).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from qwen_mtp import rms_norm


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return rms_norm(x, self.w, self.eps)


class StudentS1(nn.Module):
    """fc -> norm -> MLP. No attention."""

    def __init__(self, d=4096, d_ff=8192, eps=1e-6):
        super().__init__()
        self.enorm = RMSNorm(d, eps)
        self.hnorm = RMSNorm(d, eps)
        self.fc = nn.Linear(2 * d, d, bias=False)
        self.pre_ff = RMSNorm(d, eps)
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)
        self.final = RMSNorm(d, eps)

    def forward(self, emb, h_prev, ctx=None):
        """emb [B,d] token embedding, h_prev [B,d] predecessor hidden.
        Returns (pre_head [B,d] for lm_head, h_out [B,d] chain carrier)."""
        x = self.fc(torch.cat((self.enorm(emb), self.hnorm(h_prev)), dim=-1))
        z = self.pre_ff(x)
        h = x + self.down(F.silu(self.gate(z)) * self.up(z))
        return self.final(h), h

    def ctx_rows(self, emb, h_prev):   # no attention -> no context state
        return None


class StudentS2(nn.Module):
    """Half-width teacher clone: gated attention + MLP at d_s, up-proj out."""

    def __init__(self, d=4096, d_s=2048, n_q=8, n_kv=2, hd=256, d_ff=6144,
                 rot=64, theta=1e7, eps=1e-6):
        super().__init__()
        self.d_s, self.n_q, self.n_kv, self.hd, self.rot = d_s, n_q, n_kv, hd, rot
        self.eps = eps
        self.enorm = RMSNorm(d, eps)
        self.hnorm = RMSNorm(d, eps)
        self.fc = nn.Linear(2 * d, d_s, bias=False)
        self.in_ln = RMSNorm(d_s, eps)
        self.wq = nn.Linear(d_s, n_q * hd * 2, bias=False)   # per-head [query; gate]
        self.wk = nn.Linear(d_s, n_kv * hd, bias=False)
        self.wv = nn.Linear(d_s, n_kv * hd, bias=False)
        self.wo = nn.Linear(n_q * hd, d_s, bias=False)
        self.q_norm = RMSNorm(hd, eps)
        self.k_norm = RMSNorm(hd, eps)
        self.post_ln = RMSNorm(d_s, eps)
        self.gate_p = nn.Linear(d_s, d_ff, bias=False)
        self.up_p = nn.Linear(d_s, d_ff, bias=False)
        self.down_p = nn.Linear(d_ff, d_s, bias=False)
        self.out_up = nn.Linear(d_s, d, bias=False)
        self.final = RMSNorm(d, eps)
        inv = 1.0 / (theta ** (torch.arange(0, rot, 2).float() / rot))
        self.register_buffer("inv_freq", inv)

    def _rope(self, x, pos):
        """x [B, N, heads, hd]; pos [B, N]. Interleaved pairing, partial rotary —
        identical math to train.rope_batch / teacher convention."""
        rot = self.rot
        xr, xp = x[..., :rot], x[..., rot:]
        ang = pos.float()[..., None] * self.inv_freq.to(x.device)     # [B, N, rot/2]
        cos, sin = ang.cos()[:, :, None, :], ang.sin()[:, :, None, :]  # [B, N, 1, rot/2]
        x1, x2 = xr[..., 0::2], xr[..., 1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat((torch.stack((o1, o2), -1).flatten(-2).to(x.dtype), xp), -1)

    def _mix(self, emb, h_prev):
        return self.fc(torch.cat((self.enorm(emb), self.hnorm(h_prev)), dim=-1))

    def ctx_rows(self, emb, h_prev, pos):
        """Context K/V for rows: emb [B,W,d], h_prev [B,W,d], pos [B,W] rope positions.
        Returns (K, V) [B, W, n_kv, hd]."""
        x = self._mix(emb, h_prev)
        xin = self.in_ln(x)
        B, W, _ = x.shape
        k = self.k_norm(self.wk(xin).view(B, W, self.n_kv, self.hd))
        v = self.wv(xin).view(B, W, self.n_kv, self.hd)
        k = self._rope(k, pos)                            # [B, W, n_kv, hd]
        return k, v

    def forward(self, emb, h_prev, ctx, q_pos, ctx_mask=None):
        """One draft query per batch row. emb/h_prev [B,d]; ctx=(K,V) [B,W,n_kv,hd];
        q_pos [B] rope positions. Own-row attention included (row appended)."""
        x = self._mix(emb, h_prev)                        # [B, d_s]
        xin = self.in_ln(x)
        B = x.shape[0]
        qg = self.wq(xin).view(B, self.n_q, 2 * self.hd)
        q, gate = qg.chunk(2, dim=-1)
        q = self.q_norm(q)
        q = self._rope(q.unsqueeze(1), q_pos[:, None]).squeeze(1)        # [B,n_q,hd]
        k_own = self.k_norm(self.wk(xin).view(B, 1, self.n_kv, self.hd))
        k_own = self._rope(k_own, q_pos[:, None])
        v_own = self.wv(xin).view(B, 1, self.n_kv, self.hd)
        K = torch.cat((ctx[0], k_own), dim=1)             # [B, W+1, n_kv, hd]
        V = torch.cat((ctx[1], v_own), dim=1)
        rep = self.n_q // self.n_kv
        Ke = K.repeat_interleave(rep, dim=2)              # [B, W+1, n_q, hd]
        Ve = V.repeat_interleave(rep, dim=2)
        att = torch.einsum("bhd,bnhd->bhn", q.float(), Ke.float()) / (self.hd ** 0.5)
        if ctx_mask is not None:                          # mask short-context pad rows
            pad = F.pad(ctx_mask, (0, 1), value=True)     # own row always visible
            att = att.masked_fill(~pad[:, None, :], float("-inf"))
        w = att.softmax(-1)
        o = torch.einsum("bhn,bnhd->bhd", w, Ve.float()).to(x.dtype)
        gated = o.reshape(B, -1) * torch.sigmoid(gate.reshape(B, -1).float()).to(x.dtype)
        h1 = x + self.wo(gated)
        z = self.post_ln(h1)
        h_s = h1 + self.down_p(F.silu(self.gate_p(z)) * self.up_p(z))
        h = self.out_up(h_s)                              # back to n_embd
        return self.final(h), h


class StudentSV(StudentS2):
    """Small-vocab small-block draft head (llama.cpp #25187 follow-through): S2 block
    + OWN trainable lm_head over a frequency-ranked hot vocab (warm-started from the
    big lm_head's hot rows) + d2t map. Verify stays full-vocab (lossless); the head
    simply cannot draft cold tokens — ceiling = hot-set coverage (~0.95-0.99)."""

    def __init__(self, hot_ids, big_lm_head=None, d=4096, **kw):
        super().__init__(d=d, **kw)
        self.register_buffer("d2t", hot_ids.clone())
        self.lm_head = nn.Linear(d, len(hot_ids), bias=False)
        if big_lm_head is not None:
            with torch.no_grad():
                self.lm_head.weight.copy_(big_lm_head[hot_ids].float())

    def head_logits(self, pre_head):
        return self.lm_head(pre_head)          # [B, V_hot]

    def to_target_ids(self, draft_idx):
        return self.d2t[draft_idx]


def param_count(m):
    return sum(p.numel() for p in m.parameters())
