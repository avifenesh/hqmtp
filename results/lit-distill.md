# Literature map — distilling smaller draft/MTP heads (searched 2026-07-09)

Verdict: **our combination is novel** — (smaller student ← co-trained MTP-head teacher) ×
(quantized-base hidden states) × (acceptance-evaluated) is not in the literature.

## Adjacent work

| work | what it does | overlap |
|---|---|---|
| MTP-D / Self-Distillation for MTP (arXiv 2603.23911) | self-distill BETWEEN same-size MTP heads, +7.5% acceptance; looped head extension | closest objective, no size reduction, no quant |
| FastMTP (arXiv 2509.18362) | consolidates multiple MTP heads into one shared-weight head via self-distill on model-generated data | consolidation ≠ shrinking |
| SlimSpec (arXiv 2605.10453) | low-rank LM-head for EAGLE-3 drafter, 4-5x lm_head speedup, acceptance held | compresses lm_head only, no distillation, composable with ours |
| AdaSPEC (arXiv 2510.19779) | selective-token KD for draft training, +15% acceptance over DistillSpec | KD recipe steal (mask hard tokens); teacher = base model not head |
| EAGLE-3 practice (Baseten 2026-04) | from-scratch 1-layer heads, 1-5% of target size, 70-80% accuracy | the mandatory from-scratch baseline B0 |
| QSpec (EMNLP 2025) | two quant schemes of same model draft/verify | no heads, no adaptation |
| Qwen3.6-35B GGUF discussion (HF #14) | BF16-trained DFlash drafter loses 44.6% acceptance on Q4_K target; hypothesis = quantized hidden-state shift; **no fix attempted** | external proof the quant×draft problem is real + open |
| MLX EAGLE-3 (#890) | 4-bit RUNTIME quant of draft head = free (identical acceptance) | matches our Q8-head finding; format ≠ function |
| MTP via self-distillation (arXiv 2602.06019) | converts AR model into standalone MTP, no aux heads | orthogonal |
| DFlash / P-EAGLE (2602.06036 / 2602.01469) | diffusion / parallel drafters | alternative drafter archs, no distill-from-head |

## What this buys the writeup

- Novelty claims: (1) first shrink-distillation from a co-trained MTP head; (2) first
  training-time adaptation of any draft head to quantized hidden states.
- The HF DFlash-on-Q4 report is the in-the-wild motivation anecdote.
- Design debts: AdaSPEC masking (ablation flag), SlimSpec low-rank head (second
  compression axis), EAGLE-3 from-scratch baseline (control).

## Quantized-draft prior art (searched 2026-07-18, post Q4/NVFP4 results)

| work | what | overlap with ours |
|---|---|---|
| ML-SpecQD (arXiv 2503.13565, Intel, 2025-03) | MXFP4 direct-cast of the TARGET as plug-and-play draft + multi-level drafting; CPU; up to 2x | closest — "4-bit drafts viable, high acceptance" claim is THEIRS; draft = whole-model copy (GBs), no trim, no head, no mechanism |
| QSpec (arXiv 2410.11305, 2024-10) | W4A4 drafts / W4A16 verify, same model | quant-scheme self-spec; already in map |
| SpecDec meets Quantization (arXiv 2505.22179, 2025-05) | EAGLE-2 x quantized targets compatibility + hierarchical framework | evaluates spec ON quant models, not quantized drafters per se |

POSITIONING CONSEQUENCE: cannot headline "quantize the draft" as new. Ours =
(1) the OBJECT: co-trained MTP heads, trim x quant composed, 212MB w/ lossless
d2t verify (their drafts are whole-model copies); (2) the MECHANISM: robustness
law with prune/distill counterfactuals + OOD evidence — why precision is the only
free knife; (3) NVFP4 + sampled-stabilization finding; (4) GPU production engine +
scale law (2.67x @27B vs their CPU 2x). Title reverts toward the frontier study;
ML-SpecQD/QSpec cited as precedent we extend and explain.
