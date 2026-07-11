# Naked KD ablation — recipe-order answer

Question (user): does KD-from-scratch match the staged CE→KD pipeline? ("we might
discover that we get the same results from kd naked")

**Answer: yes, exactly.** Chain 0.449 both — naked KD at HALF the total steps.

## Setup

- Fresh StudentSV (hot-row lm_head warm start only, zero prior training)
- Pure teacher-logit loss: alpha 1.0, beta 0, gamma 0 — no CE anywhere
- 12k steps, batch 64, W 2048, 4-slot chain, cosine LR 3e-4, own-output corpus
  (1,380 docs) + precomputed teacher top-64 targets (.tt.npz)
- Eval: self-drafting chain (eval_chain.py), 45 held-out fresh-generation docs

## Loss bug found (fixed in 4e14d3e)

Original KD term gathered teacher top-64 logits and log_softmax'd the SUBSET —
constrains ordering inside 64, leaves the other 32k logits unconstrained. Pure KD
collapsed (loss ↓ 3.8→1.8, argmax acc → 0.000, junk token dominated full vocab).
CE anchored the full vocab in the staged run and masked the flaw. Fix = soft-CE:
log_softmax over full 32k draft vocab, sparse teacher-prob targets.

## Result

| ckpt | slot0 | slot1 | slot2 | slot3 | chain |
|---|---|---|---|---|---|
| teacher (engine chain) | 0.740 | 0.598 | 0.520 | 0.470 | 0.593 |
| staged CE→KD (sv-chain, ~24k steps) | 0.679 | 0.563 | 0.468 | 0.405 | 0.449 |
| **naked KD 12k (fixed loss)** | 0.656 | 0.563 | 0.488 | 0.425 | **0.449** |

Chain curve per 1k steps: .075 .129 .199 .270 .316 .339 .379 .405 .434 .444 .448 .449
— smooth, flattens only as cosine LR → 0, not capacity saturation.

## Reading

- Naked KD trades slot0 (−.023) for slots 2–3 (+.020 each) vs staged. Deep-slot
  gains are what extend accepted chain length; pure distribution matching teaches
  the chain regime better than CE warmup did.
- CE pretraining stage adds nothing to chain acceptance. Canonical recipe collapses
  to: generate own output → copy teacher logits → distill. One stage.
- Slot3 0.425 vs teacher 0.470 — student within .045 of teacher at depth.

## Next

CE-reinforcement stage on top (sv-kd-ce: init sv-kd-naked, beta 1.0 alpha 0,
lr 1e-4, 4k steps) — tests "kd then re-inforce with ce". Results appended below.
