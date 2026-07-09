"""NVFP4 (compressed-tensors) dequantization — weight-exact torch replica.

Layout per quantized linear `<stem>` (llm-compressor, nvfp4-pack-quantized):
  <stem>.weight_packed        u8  [out, in/2]   e2m1 codes, 2/byte, LOW nibble first
  <stem>.weight_scale         f8e4m3 [out, in/16] per-16 block scales
  <stem>.weight_global_scale  f32 scalar        DIVISOR (bw24 mapping doc)

  dequant[o, i] = e2m1(code) * scale[o, i // 16] / global_scale
"""
import torch

# e2m1 magnitude table, index = low 3 bits; bit 3 = sign
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def dequant_nvfp4(packed: torch.Tensor, scale: torch.Tensor,
                  global_scale: torch.Tensor) -> torch.Tensor:
    """packed [out, in/2] u8; scale [out, in/16] float8_e4m3fn; -> [out, in] bf16."""
    out_f, half = packed.shape
    lo = packed & 0xF
    hi = packed >> 4
    codes = torch.stack((lo, hi), dim=-1).reshape(out_f, half * 2)  # low nibble first
    mag = _E2M1.to(packed.device)[(codes & 0x7).long()]
    sign = torch.where(codes & 0x8 != 0, -1.0, 1.0)
    vals = mag * sign
    s = scale.to(torch.float32).repeat_interleave(16, dim=1)
    gs = float(global_scale.reshape(-1)[0])
    return (vals * s / gs).to(torch.bfloat16)
