# hqmtp — healing quantized-model MTP heads

## Problem

MTP draft heads are trained jointly with the full-precision base model. When the base
model is quantized (target here: NVFP4) the head is usually kept at 8-bit or better, so
the head itself is not damaged — but it now predicts against hidden states produced by a
*different* (quantized) model than the one it was trained with. Observed effect: MTP
acceptance drops, and the drop grows with prompt length and with session depth
(accumulating agent-loop context).

## Subject model

Qwen3.5-9B (dense, hybrid linear-attention/full-attention, natural `mtp.*` head trained
with the model).

- bf16 ceiling arm: `Qwen/Qwen3.5-9B` HF checkpoint, run with `BW24_FULL_PREC=1`
  (bf16-resident weights, f32 oracle path).
- quant arm: `Qwen3.5-9B-NVFP4-MTP-GGUF.gguf` (NVFP4 base + embedded nextn head).

## Engine / rig

- Engine: [bw24](../bw24) `run-spec` (spec-decode with per-run acceptance stats).
- Rig: `bw24-g7e` EC2 box — RTX PRO 6000 Blackwell Server Edition, 96GB.
  All experiments run remote; local GPU is not used.

## Protocol (first bench)

`bw24/tools/acceptance_battery.sh` per arm:

- Prompts p1–p6 (prompt-length axis): 146B code-short, 6.3KB code-medium,
  23KB agentic-long, 79KB (~16k tok), 135KB (~32k tok), 247KB (~64k tok).
- K ∈ {1,2,3,4}, N=3 runs per cell, NGEN=128.
- 8-turn accumulative agent loop (session-depth axis), K=3, NGEN=256.
- `CORPUS_DIR` set for both arms — every prompt payload sent to the model is saved;
  the NVFP4-arm payloads are the retraining corpus for the head-heal work.
- Delta: `tools/acceptance_delta.py out-bf16.jsonl out-nvfp4.jsonl`
  (positive delta = quant loses acceptance vs bf16 ceiling).

## Prior data point (rig5090, 2026-07-09 morning, K 1–4, p1–p3 only)

From `bw24/research/mtp-heal/delta-9b-bf16-vs-nvfp4.md`:

- p3 (agentic-long) K=1/K=2: quant loses **14.4% / 18.8%** relative acceptance — the
  long-prompt hypothesis holds.
- Agent loop: quant loss grows with turn depth, +3.2% (t1) → +7.5% (t8).
- Short prompts (p1) roughly break even; some K=3/4 cells show quant *ahead* —
  needs the wider prompt sweep to interpret.

## Direction after first bench

LoRA-retrain (or otherwise adjust) the MTP head against the *quantized* base model's
hidden states, using the saved prompt corpus; show acceptance recovery vs this
baseline, or a clean negative.

Note on head trimming: bw24 production usually runs a vocab-trimmed head
(frspec-style). This project starts from the **natural, untrimmed** head — trimming is
already proven elsewhere and would confound the quant-mismatch signal. Later milestone:
show that trimming the *retrained* head beats (or matches) the trimmed-vocab head built
from the non-retrained one.

## Layout

- `results/` — per-run JSONL + delta tables from the g7e box.
- `corpus/` — prompt payloads captured during NVFP4 runs.
