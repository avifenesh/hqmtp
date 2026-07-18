"""NVFP4 GGUF encoder (ggml type 40) — gguf-py knows the type but has no quantizer.

Block layout (engine bw24-gguf/src/dequant.rs, verbatim ggml): QK=64 elems ->
36 bytes = 4 UE4M3 sub-scale bytes (one per 16 elems) + 32 nibble bytes.
Nibble order within sub-block s: byte j -> low nibble = elem j, high = elem j+8.
Value table = kvalues_mxfp4 (e2m1 DOUBLED); UE4M3 decode carries a 0.5 factor.
"""
import numpy as np

KV = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
              dtype=np.float32)
# signed lookup order for encoding: value -> code
_POS = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (6, 5), (8, 6), (12, 7)]


def _ue4m3_table():
    t = np.zeros(256, dtype=np.float32)
    for b in range(256):
        if b == 0 or b == 0x7F or b > 0x7F:
            t[b] = 0.0
            continue
        exp = (b >> 3) & 0xF
        man = b & 7
        raw = man * 2.0 ** -9 if exp == 0 else (1 + man / 8.0) * 2.0 ** (exp - 7)
        t[b] = raw * 0.5
    return t


UE4M3 = _ue4m3_table()          # code -> scale value (codes 1..0x7E usable)


def _encode_scale(d):
    """nearest UE4M3 code for target scale d (>0)."""
    codes = np.arange(1, 0x7F)
    vals = UE4M3[1:0x7F]
    return codes[np.abs(vals - d).argmin()]


def quantize_nvfp4(x):
    """x [rows, cols] f32, cols % 64 == 0 -> uint8 [rows, cols//64*36]."""
    rows, cols = x.shape
    assert cols % 64 == 0, cols
    sub = x.reshape(rows, cols // 16, 16)                     # 16-elem sub-blocks
    amax = np.abs(sub).max(axis=-1)                           # [rows, nsub]
    d_target = amax / 12.0
    # encode scales via nearest-UE4M3 (vectorized over unique-ish values)
    vals = UE4M3[1:0x7F]
    idx = np.abs(d_target[..., None] - vals[None, None, :]).argmin(-1)
    d_codes = (idx + 1).astype(np.uint8)
    d_enc = UE4M3[d_codes]
    d_enc_safe = np.where(d_enc > 0, d_enc, 1.0)
    q = sub / d_enc_safe[..., None]                           # scaled values
    # nearest signed table entry
    mags = np.array([v for v, _ in _POS], dtype=np.float32)
    pos_code = np.array([c for _, c in _POS], dtype=np.uint8)
    mag_idx = np.abs(np.abs(q)[..., None] - mags[None, None, None, :]).argmin(-1)
    codes = pos_code[mag_idx]
    codes = np.where((q < 0) & (codes > 0), codes + 8, codes).astype(np.uint8)
    codes = np.where(d_enc[..., None] > 0, codes, 0)
    # pack: sub-block s byte j = elem j | elem j+8 << 4
    lo = codes[..., :8]
    hi = codes[..., 8:]
    packed = (lo | (hi << 4)).astype(np.uint8)                # [rows, nsub, 8]
    nsubs = cols // 16
    nblk = cols // 64
    d_codes_b = d_codes.reshape(rows, nblk, 4)
    packed_b = packed.reshape(rows, nblk, 32)
    out = np.concatenate([d_codes_b, packed_b], axis=-1)      # [rows, nblk, 36]
    return np.ascontiguousarray(out.reshape(rows, nblk * 36))


def dequantize_nvfp4(raw, rows, cols):
    """reference decode for roundtrip tests (mirrors engine dequant_nvfp4)."""
    raw = raw.reshape(rows, cols // 64, 36)
    d = UE4M3[raw[..., :4]]                                   # [rows, nblk, 4]
    qs = raw[..., 4:]                                         # [rows, nblk, 32]
    qs = qs.reshape(rows, -1, 4, 8)
    lo = KV[(qs & 0x0F).astype(np.int64)]
    hi = KV[(qs >> 4).astype(np.int64)]
    subs = np.concatenate([lo, hi], axis=-1) * d[..., None]   # [rows, nblk, 4, 16]
    return subs.reshape(rows, cols).astype(np.float32)


if __name__ == "__main__":
    x = (np.random.RandomState(3).randn(8, 256) * 0.05).astype(np.float32)
    q = quantize_nvfp4(x)
    y = dequantize_nvfp4(q, 8, 256)
    err = y - x
    print(f"roundtrip rel-rms {np.sqrt((err**2).mean())/x.std():.4f} "
          f"max {np.abs(err).max():.4f} bytes/elem {q.size/x.size:.3f}")
