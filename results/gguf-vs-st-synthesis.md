# Why the GGUF's MTP "recovers better" than plain ST — synthesis (3 research agents + measurements)

Date: 2026-07-09. Question: live MTP acceptance p1/K1 — GGUF 0.868 > bf16 0.841 > st-ct 0.745.
Both quant arms are "NVFP4", so why does the GGUF beat the cleaner ST checkpoint?

## Fact base

**Quant coverage (tensor-inventory + provenance agents):**

| tensor class | GGUF | st-ct (llm-compressor) |
|---|---|---|
| MLP gate/up/down | NVFP4 | NVFP4 (the ONLY quantized class) |
| attention q/k/v/o | Q4_K/Q5_K | bf16 (excluded) |
| linear-attn projections | Q4_K/Q8_0 | bf16 (excluded) |
| embeddings | NVFP4 | bf16 |
| lm_head / output | Q6_K | bf16 |
| MTP block (full transformer, blk.32 / mtp.*) | Q8_0 (protected) | bf16 (excluded) |

GGUF = aggressively quantized everywhere. st-ct = NVFP4 on MLP only; gentlest possible.
(ct's "dynamic activation quant" config field is vLLM-side; bw24 runs its own W4A8 — n/a here.)

**Engine path (code-audit agent):** both sources land in the same QT_NVFP4 split-plane
resident format and the same kernels; global-scale divisor→multiplier inversion verified on
the hot path; no Float-poison loader warnings in st-ct run logs; K=1..8 exactness gates PASS
on both. No engine-path smoking gun found.

**Measurements:**
- Teacher-forced replay (identical fixed contexts): st-ct ≈ bf16 within ±0.03 at all depths;
  GGUF *above* bf16 by up to +0.14 at depth.
- Live spec chain (self-generated): GGUF 0.868 > bf16 0.841 > st-ct 0.745 (p1/K1).

## Answer (ranked)

1. **The premise of the question is inverted. The GGUF does not "recover better" — it is
   damaged harder, and heavier damage inflates self-acceptance.** Q4/Q5 attention + NVFP4
   embeddings collapse the base's output entropy toward more predictable continuations; the
   (also-quantized, Q8_0) head co-drifts with it. Self-acceptance measures head↔base
   agreement, not quality — a more degraded, more repetitive base is easier to predict.
   Replay v1 showed exactly this signature (GGUF self-agreement above the bf16 ceiling).

2. **st-ct's low LIVE number is the real signal for hqmtp.** Forced replay says its weights
   are fine (≈ bf16 agreement); live chain drops ~10pts below bf16. The bf16-trained head
   sitting on slightly-shifted NVFP4 hiddens loses exactly at the close-margin argmax calls
   that live self-conditioned generation keeps revisiting — and/or its generated text drifts
   into regions the head predicts worse. This is the "quant hurts MTP in sessions" phenomenon,
   isolated: it lives in generation dynamics, not in forced-context weight mismatch.

3. Secondary contributors, unproven: GGUF Q8_0 head slightly "matched" to its degraded
   trunk (both quantized in one recipe); st-ct 512-sample Nemotron calibration possibly
   mismatched to code prompts. Both subordinate to (1)/(2).

## Probes queued

- st-ct full live battery p1–p6 (running): does the live gap grow with prompt length?
- Cross-text replay: replay each arm over the OTHER arm's live-generated text — separates
  "harder own text" from "chain dynamics".
- pi-harness serve turns (axis D): the regime that matters.

## Implication for the heal direction

If the live-only gap holds, LoRA-retraining the head should target the quantized base's
OWN GENERATION distribution (self-conditioned contexts — the captured corpus), not forced
public text; eval metric = live-chain acceptance, with teacher-forced replay as the
control that must stay flat.
