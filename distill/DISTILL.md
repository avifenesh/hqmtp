# Distilling a smaller MTP head from the co-trained one (on the quant base)

Goal: student draft head, smaller than the original co-trained `mtp.*` block, distilled
FROM that block (teacher), with the quantized (st-ct NVFP4) base supplying hidden states —
so the student is simultaneously cheaper AND adapted to the quant deployment. Evaluated by
teacher-forced replay acceptance (protocol v2 metric).

Literature check (2026-07-09): combination is novel — self-distill between same-size MTP
heads (MTP-D), multi-head consolidation (FastMTP), low-rank draft lm_head (SlimSpec), and
selective-KD draft training (AdaSPEC) all exist; nobody distills smaller-from-co-trained,
nobody trains a head against quantized hiddens (the failure is documented unsolved in the
wild: BF16 DFlash drafter −44.6% acceptance on Q4 target). Full map: results/lit-distill.md.

## Teacher / inputs (Qwen3.5-9B)

MTP block interface (bw24 spec.rs conventions, predecessor pairing):
- inputs per position p: token embedding e(x_p), predecessor trunk hidden h_{p-1},
  rope position p+1; the block runs enorm/hnorm -> eh_proj([e;h]) -> 1 transformer
  layer (full attn over draft KV) -> shared lm_head (output.weight).
- teacher outputs: draft hidden h_nextn (chain carrier) + draft logits.

## Student candidates (phase-1 ablation, smallest first)

- S1 "mlp-only": drop the attention mixer entirely — enorm/hnorm + eh_proj + MLP.
  Tests whether draft-KV attention matters at K<=4 chain depths.
- S2 "half-width": full block shape at d=2048 (down-proj in, up-proj out at the seams).
- Baseline B0: EAGLE-style from-scratch single-layer head, same size as S2, trained on
  CE only — the "does distillation even help" control.

## Data

- Extraction: teacher-forced pass over corpus on the DEQUANTIZED st-ct base in torch
  (weight-exact NVFP4 dequant: e2m1 * ue4m3_scale / global_scale; bf16 compute).
  Per position: (x_p, h_{p-1}, base greedy target, teacher-head logits topk + h_nextn).
- Corpus: fresh public text/code (NOT the eval corpora), spike = ~2M tokens, scale later.
- GATE (mandatory before training): torch teacher must reproduce bw24 replay drafts on
  a shared corpus prefix (we have engine dumps with per-position drafts — compare
  slot-0 draft match rate; >99% required, FP-order tolerance for the rest).

## Losses

KL(student logits || teacher logits, topk-renormalized) + CE(student, base greedy)
+ cosine/MSE on h_nextn (chain carrier — slot>=2 quality rides on it).
AdaSPEC-style selective masking (drop positions where teacher itself is wrong) as an
ablation flag.

## Eval

Torch-side replay acceptance (same metric as engine replay-acceptance, computed in the
extraction harness): student drafts K=4 at strided positions vs base greedy track, on
gen-dist-9b + eval-code. Compare: teacher head / S1 / S2 / B0. Engine integration only
after a student wins on the torch metric (needs engine-side head-arch support —
coordinate with the engine agent, not ours to build).

## Success

Student at <=25% teacher params holding >=95% of teacher slot-0 acceptance on both
corpora. Distillation must beat B0 from-scratch — otherwise report that finding.

## Layout

- distill/extract.py — data extraction + torch teacher gate
- distill/train.py — student training
- distill/eval_replay.py — torch replay acceptance
- distill/checkpoints/, distill/data/ (gitignored)
