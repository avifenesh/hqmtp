# SV head — production round 2 (2026-07-10)

Batch-64 teacher-free CE run, 15k steps (960k samples), 113-file corpus (~7M tokens),
window 2048. Eval: torch replay slot-0, window 2048.

| head | eval-code | gen-dist | draft FLOPs vs teacher |
|---|---|---|---|
| engine (full ctx) | 0.678 | 0.348 | 1x |
| teacher (windowed) | 0.675 | 0.371 | 1x |
| sv @ 960k samples | 0.527 | 0.166 | ~0.13x |
| sv @ 60k samples | 0.337 | 0.105 | ~0.13x |
| s2-b0 full-vocab @ 60k | 0.373 | — | ~0.35x |

- 78% of teacher acceptance at ~1/8 draft compute; curve unsaturated (loss ~1.9 falling).
- Vocab ceiling not binding (0.989 code coverage). Gap = budget + capacity.
- Chat domain weak (0.166): 20% mix + 0.946 coverage — chat-weighted data/vocab lever.
- GPU lesson: CE-only runs were paying a useless teacher pass per step + batch 12 left
  2/3 of VRAM idle; fixing both tripled effective throughput (34->56GB, 487->550W).
