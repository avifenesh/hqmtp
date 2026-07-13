# Replacing the Co-Trained MTP Head: Distilling a Small-Vocabulary Draft Head that Beats its Teacher End-to-End

*Working draft skeleton — numbers sourced from results/. TODO slots marked.*

## Abstract (draft)

Models shipping co-trained multi-token-prediction (MTP) heads pay a fixed draft cost:
the head is a full trunk-width transformer block with a full-vocabulary output
projection (243M+ parameters at Qwen3.5-9B scale, 248k-row lm_head). Vocabulary
trim-at-load (FR-Spec) shrinks only the lm_head read and leaves end-to-end decode
nearly unchanged. We distill the co-trained head into a natively-trained replacement
at ~13% of the draft FLOPs — half-width block (2048 vs 4096), a third of the
attention heads, and an own trainable 32k-row hot-vocabulary head with a
draft-to-target index map; verification stays full-vocabulary and lossless. The
recipe is two stages: (1) chain knowledge distillation on precomputed teacher
logits over the model's OWN output generated under the SERVING regime, with a
self-recursive hidden carrier; (2) brief cross-entropy reinforcement. In a
production CUDA engine the student reaches 98% of the teacher's chain acceptance on
serving-regime text, exceeds the teacher on deep chain slots, and at long context
BEATS the natural head end-to-end: 1.99x vs 1.60x over plain decode (269.6 vs 181.5
tok/s) — lower acceptance, much cheaper rounds. We report two negative results with
practical weight: subset-topk KL silently collapses pure-KD training (the unshaped
tail of the vocabulary hijacks argmax), and register mismatch between training
corpus and live serving (sampling policy, thinking mode) silently halves live
acceptance while replay metrics look perfect.

## 1. Introduction

- MTP heads co-trained with the base (Qwen3.5/DeepSeek lineage): free spec-decode
  drafts, but heavy (full-width block + full-vocab head).
- Motivation thread: llama.cpp issue #25187 — FR-Spec-style trim-at-load cuts the
  draft lm_head kernel ~85% but e2e gain is 1-3%: the block dominates and
  acceptance is capped by hot-vocab coverage anyway.
- Claim: train the small head instead of trimming the big one. Both faster decode
  AND VRAM back; acceptance-per-FLOP is what matters, not acceptance alone.
- Headline: e2e win at long context (1.99x vs 1.60x) despite acceptance deficit.

## 2. Related work

(from results/lit-distill.md — the niche is empty)
- FR-Spec: vocab trim at load; lm_head-only; our matched-vocab baseline arm.
- EAGLE-1/2/3: external from-scratch drafters, feature-level autoregression; no one
  distills FROM a co-trained MTP head into a smaller replacement.
- MTP-D / FastMTP: self-distill between same-size heads / consolidate multiple
  heads; no size reduction.
- SlimSpec: low-rank lm_head for EAGLE — composable, vocab axis only.
- AdaSPEC / DistillSpec: KD recipes teacher=base-model; our teacher = the head.
- DFlash-on-Q4 HF report: in-the-wild evidence draft heads are distribution-fragile.

## 3. Method

### 3.1 Student architecture (StudentSV)
- Half-width clone of the head block: d_s 2048 (teacher 4096), 8q/2kv heads
  (teacher 16/4), head_dim 256 kept (shared rope), gated attention + partial rotary
  matched to teacher conventions; out_up projection 2048->4096 restores the
  interface; own lm_head over frequency-ranked top-32768 hot vocab warm-started
  from the teacher's hot rows; d2t index map; drafts can only propose hot tokens,
  verify is full-vocab => exactness preserved, ceiling = hot coverage (98.9% code /
  94.6% chat).
- 212M params ≈ 13% of teacher draft FLOPs. [TODO: exact FLOPs table]

### 3.2 Training data: own-output under the serving regime
- Rule 1 (validated +17pts): train ONLY on the model's own generated text — generic
  corpora actively hurt (sv-own 0.548 @1.3M own tokens vs 0.376 @13M generic).
- Rule 2 (this work): generation must match the SERVING configuration — sampling
  policy mix (greedy + t0.7), thinking mode ON, chat template. v1 corpus
  (thinking-off, t0.7-only) scored 0.70 slot0 on replay yet 13-18% live; v2
  regime-matched corpus lifted serving-regime chain 0.359 -> 0.630.

### 3.3 Chain distillation on copied teacher logits
- Precompute teacher top-64 logits x 4 chain slots over the whole corpus (one
  exact draft-KV fill per doc + batched slot queries; teacher-forced chain rows
  equal true fill rows so per-position slot targets are exact). 7 min for 5M tokens.
- Loss: full-vocab log-softmax soft-CE against sparse teacher targets, per slot,
  self-recursive carrier (student's own h feeds slot j+1 — the engine chain regime).
- NEGATIVE RESULT: subset-KL over the gathered top-64 leaves the other 32k logits
  unconstrained; pure KD collapses (loss falls, argmax accuracy -> 0.000). CE in a
  mixed loss masks the defect. Full-vocab normalization is mandatory.
- Chain training resurrects deep slots: pre-chain slots 1-3 = 0.17/0.05/0.02;
  post = 0.61/0.55/0.48 — above teacher (0.598/0.520/0.470 engine chain).

### 3.4 Recipe order ablation (results/kd-naked.md)
| recipe | steps | chain |
|---|---|---|
| CE pretrain -> KD | 24k | 0.449 |
| naked KD (clean init) | 12k | 0.449 (exact tie) |
| naked KD 24k | 24k | 0.490 |
| naked KD 24k -> CE 4k | 28k | 0.507 |
CE-first buys nothing; KD-then-CE is the recipe. CE reinforcement: brief dip then
+0.017.

### 3.5 Engine integration
- bw24 (production CUDA engine): DraftGeom on the MTP head — narrow inner width +
  out_up, carrier at n_embd, existing d2t verify path; scratch KV halves.
- Export: Q8_0 GGUF; interleaved-trained rope -> neox kernel via row permutation of
  wq/wk + q/k norms inside the roped block (exact equivalence).

## 4. Experiments

### 4.1 Chain acceptance (teacher-forced self-draft, held-out own-output docs)
| head | v1 set | serving-regime set |
|---|---|---|
| teacher (engine chain) | 0.593 | 0.641 |
| sv2 (ours) | 0.521 (88%) | 0.630 (98%) |
Per-slot: student EXCEEDS teacher at slots 1-3 on both sets (e.g. v2 set slot3
0.603 vs 0.440); slot0 is the remaining gap (structural: less context, less width).

### 4.2 End-to-end decode (bw24, 9B bf16, K=4, 512-token generations)
| prompt | natural greedy | ours greedy | natural t0.7 | ours t0.7 |
|---|---|---|---|---|
| chat short | 42.0% 1.35x | 21.6% 1.23x | 49.7% 1.38x | 31.0% 1.35x |
| code short | 41.8% 1.36x | 16.7% 1.03x | 40.2% 1.07x | 11.7% 0.91x |
| longchat 3.5k | 60.5% 1.60x | 53.4% **1.99x** | 63.3% 1.56x | 23.6% 1.35x |
[TODO: replace ad-hoc trio with Spec-Bench/MT-Bench sweep, multi-seed CIs — battery running]
[TODO: FR-Spec trim arm (same 32k vocab, natural block) — battery running]

### 4.3 27B replication
[TODO: full pipeline queued on box — Qwen3.6-27B, hidden 5120, 24q/4kv, same recipe]

### 4.4 Serving-regime study (the measurement trap)
- Same model, same question: sampled corpus doc replay 0.70 slot0; greedy live
  generation of the same question 0.048 chain. Replay on live text == live
  acceptance (0.177 vs 0.134) => the metric gap is TEXT REGISTER, not mechanics.
- Regime-matched v2 data closes it: live chat t0.7 18% -> 31%; longchat greedy
  42% -> 53.4%.

### 4.5 Ablations & negatives
- own-output vs generic (+17pts at 1/3 steps)
- naked-KD tie + subset-KL collapse
- data scaling: 1.3M -> 5.2M own tokens = +8.6pts slot0
- window robustness: W=2048-trained student flat from W=2048 to full ctx in torch
- [TODO: sampled-tail calibration — draft-temp knob or tail-aware KD]

## 5. Analysis
- Why deep slots beat teacher: distribution-matching under the self-recursive
  carrier trains exactly the regime the chain runs; the teacher was never trained
  on its own chain rollouts.
- Acceptance-per-FLOP: at 53.4% vs 60.5% acceptance the student wins e2e by +49%
  at long ctx — round cost dominates once verify is memory-bound.
- Short-ctx weakness: small head + little context; per-position buckets 0.194 /
  0.555 / 0.678 vs teacher 0.292 / 0.651 / 0.748.

## 6. Limitations
- Sampled drafting collapses without tail calibration (13-24% at t0.7 long).
- Short-context cells below natural head; K adaptation / hybrid fallback possible.
- Single model family (Qwen3.5/3.6); [27B pending].
- Greedy long-ctx self-consistency FAIL is an engine bug on BOTH arms (bw24 #10),
  excluded from claims.

## 7. Future work
- EAGLE-3-style from-scratch baseline at matched budget (B0 control exists at 5k
  steps: 0.373 vs sv 0.527 full-vocab — needs a full-budget rerun).
- Draft-side temperature knob in-engine (exactness-preserving).
- Quantized-base variant (the NVFP4 thread: base quantization does NOT degrade the
  head — VERDICT-nvfp4-final.md — so the student ports).
- SlimSpec-style low-rank on top of hot-vocab head.
