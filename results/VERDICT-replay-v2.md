# Replay v2 verdict — NVFP4 compression effect on MTP head agreement (Qwen3.5-9B)

Date: 2026-07-09 · Rig: bw24-g7e (RTX PRO 6000 Blackwell 96GB) · Engine: bw24 @ replay-acceptance

## Setup (clean ST-vs-ST)

- Arms share the **identical bf16 `mtp.*` head weights**; the only difference is the base:
  - `qwen35-9b-hf` — bf16 safetensors, full-prec path (`BW24_FULL_PREC=1`)
  - `qwen35-9b-nvfp4-st-ct` — compressed-tensors NVFP4 (apolo13x, llm-compressor)
- Teacher-forced replay: fixed 64k-token corpus, head drafts K=4 at every 16th position,
  scored against the arm's own teacher-forced greedy picks. Identical contexts both arms.
- Corpora (topic axis): agentic-64k (agent-session text), code-64k (bw24 Rust/CUDA),
  techdoc-64k (technical markdown). Depth buckets = length axis on shared content.
- Gate: bg/pos exact both arms; chunk-FP-order draft noise floor 0.6–2.8% (higher on the
  quant arm — tighter head margins under NVFP4 hiddens).

## Result

Slot-0 delta (bf16 − nvfp4) stays within **±0.03 at every depth bucket on every corpus**,
with no depth trend up to 65536 tokens. Chain scores match equally closely. Bases agree on
87–92% of greedy picks. Full table: `delta-replay-v2.txt`.

**NVFP4 weight compression alone does not measurably degrade MTP head↔base agreement — at
any context length, on any tested topic.** The original premise (quant breaks the head, worse
with length) is not supported on this metric.

## Why earlier numbers said otherwise (artifact ledger)

1. **Self-generation battery**: bf16 arm degenerates into repetition loops on long raw
   prompts (verified p3: "right before it" loop, acceptance 0.969) → the apparent
   16–43% "quant loss" on p4–p6 was loop-inflated bf16 self-agreement.
2. **Replay v1 with the NVFP4 GGUF**: quant arm looked *better* than bf16 — the GGUF is a
   different pipeline (Q8_0 eh_proj head, Q6_K output head, converter path), not a
   compression-only comparison. GGUF arm is out of the game.

## Open thread — where the real-world drop may live

Live spec-decode still shows a gap the forced metric doesn't: st-ct 50.0% vs bf16 55.6%
(p1/K3 smoke, live chain). Live dynamics differ: self-conditioned generated text,
draft-KV built from generated hiddens, session accumulation. Running: full live battery
for the st-ct arm (p1–p6). Next: pi-harness serve turns (axis D) — same task, real agent
loop, acceptance per turn. If the drop reproduces there, the heal target is
generation-regime drift, not weight mismatch — which changes what a LoRA retrain
should train on (self-generated contexts, not forced corpus).
