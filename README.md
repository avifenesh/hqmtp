# hqmtp — from "heal the quantized MTP head" to a 212M drafter at 85% of teacher chain acceptance

Research harness for multi-token-prediction (MTP) draft heads on Qwen3.5-9B:
extraction, training, replay evaluation, corpus controls, and the negative-result
ledgers behind the working paper
[Small-vocabulary MTP heads for memory-bound speculative decoding](https://avifenesh.ai/research/small-vocabulary-mtp/).

The measurement instrument is [bw24](https://github.com/avifenesh/bw24), a
from-scratch Rust + CUDA inference engine with per-run acceptance stats and
exactness gates. Experiments run on an RTX PRO 6000 Blackwell (96 GB) EC2 box.

## How the question changed

This project is a worked example of letting the evidence rewrite the plan.

1. **Original premise** — quantize the trunk (NVFP4), keep the co-trained MTP head
   precise, and draft agreement should fall with context depth; then "heal" the head
   against quantized hidden states. Early same-rig numbers supported it
   (−14…−19% relative acceptance on long prompts).
2. **Refuted** — the early signal was measurement artifact: greedy self-generation
   loops, GGUF pipeline mixing, single-seed noise. Clean forced replay to 64k,
   generation-distribution replay, and a 3-seed sampled battery across two
   quantization recipes found **no NVFP4 degradation** (±0.03 at every depth).
   Full forensics: [`results/VERDICT-nvfp4-final.md`](results/VERDICT-nvfp4-final.md),
   [`results/VERDICT-replay-v2.md`](results/VERDICT-replay-v2.md).
3. **Pivot** — if the head doesn't need healing, the interesting question is whether
   it needs to be that *big*. StudentSV: a 212M half-width draft block with its own
   32,768-row hot-token head, replacing the model-width block and 248,320-row
   projection (~0.13× the draft FLOPs).
4. **Training-distribution rule** — training on the model's *own generations* beats a
   generic corpus by 17 points at a third of the steps
   ([`results/sv-round2.md`](results/sv-round2.md)).
5. **Chain distillation** — soft cross-entropy over the full draft vocabulary against
   precomputed teacher logits, then a short CE reinforcement. A subset-KL loss bug
   (top-64 only) was what made early KD look worse than CE; fixed, distillation-first
   wins on every slot ([`results/kd-naked.md`](results/kd-naked.md)).

## Current best

Self-drafting chain acceptance, 45 held-out own-generation docs, 4 recursive slots:

| checkpoint | slot 0 | slot 1 | slot 2 | slot 3 | chain |
|---|---|---|---|---|---|
| teacher (co-trained, engine chain) | 0.740 | 0.598 | 0.520 | 0.470 | 0.593 |
| naked KD 24k | 0.706 | 0.598 | 0.527 | 0.448 | 0.490 |
| **+ CE reinforce 4k** | **0.708** | **0.611** | **0.548** | **0.477** | **0.507** |

**0.507 = 85.5% of the teacher's chain acceptance at ~13% of the draft compute.
Slots 1–3 all exceed the teacher.** The remaining gap is slot 0 only; the KD curve
is near-saturated at 24k steps, so the next levers are more own-output data and a
wider student block. Wall time and memory are not yet measured — this is
acceptance-side evidence, not a speedup claim.

## Layout

- `distill/` — extraction (`extract.py`, `teacher_targets.py`), hot-vocab builder,
  StudentSV model (`heads.py`, `qwen_mtp.py`), training (`train.py`), chain/replay
  eval (`eval_chain.py`, `eval_replay.py`, `eval_indist.py`). See
  [`distill/DISTILL.md`](distill/DISTILL.md).
- `eval/` — fixed corpora (code / prose / agentic to 64k), prompt battery,
  head patcher, replay comparison. See [`eval/EVAL.md`](eval/EVAL.md).
- `results/` — the ledgers: verdicts, round notes, raw per-run JSONL and the
  saved generation corpora from the g7e box. Negative results stay.
- `serve/` — chat proxy used to capture own-output training data.

## Reproducing

The harness assumes bw24 as a sibling checkout (`../bw24`) for the engine-side
replay and acceptance batteries; the PyTorch side (`distill/`) is standalone.
Checkpoints and extracted `.npz`/`.tt.npz` training data are not in the repo
(see `.gitignore`); `distill/DISTILL.md` documents the exact extraction and
training commands that produced every number above.

Related public threads: FR-Spec-style trimming for native MTP —
[llama.cpp #25187](https://github.com/ggml-org/llama.cpp/issues/25187).

## Status

Active. Working-paper numbers are labeled preliminary until wall-time, memory,
and multi-seed replication land. MIT license.
