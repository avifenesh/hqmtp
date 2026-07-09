# Final verdict — NVFP4 does not degrade the co-trained MTP head (Qwen3.5-9B)

Date: 2026-07-10 · closes the original hqmtp question for NVFP4.

## Evidence matrix (all clean metrics, all arms)

| metric | st-ct (MLP-only NVFP4) | modelopt (attention-quantized NVFP4) |
|---|---|---|
| teacher-forced replay, 3 generic corpora, depths to 64k | ±0.03 vs bf16 | — |
| teacher-forced replay, gen-dist (own generations) | +0.008 vs bf16 | ≤+0.055 shallow, ≈0.000 deep vs bf16 |
| sampled live battery (temp 0.7, 3 seeds, 5 prompts × K1/K3) | gaps scatter ± within seed variance | same |

**Conclusion: NVFP4 weight quantization — gentle or aggressive recipe — causes no
measurable MTP acceptance loss for this model, in any regime, at any context depth.**
The premise "quantization breaks the co-trained head, worse with length" is
comprehensively unsupported for NVFP4.

## Artifact ledger (why it ever looked otherwise)

1. Greedy self-generation loops inflate acceptance (bf16 fake 0.97 on p3).
2. GGUF pipeline mixing (Q4/Q5 attention + Q8_0 head + converter) ≠ NVFP4 compression.
3. Single-seed greedy comparisons (st-ct 50.0 vs bf16 55.6 smoke) = seed noise;
   sampled 3-seed medians erase it.
4. Small-N smokes (modelopt "29.6%" = 8/27 drafts) — N=5-median rule applies.
5. Wild reports of drafter collapse on quantized targets (DFlash −44.6%) involve Q4_K
   GGUF-class quants, not NVFP4.

## What survives as future work

- Crude-quant arms (Q4_K-class) as the subject where a heal target may genuinely
  exist (the GGUF observation direction, needs a clean single-variable setup).
- Head distillation (task #9) — independent speed/size thread, active.
- 27B replication for scale confirmation (#8, staged).
