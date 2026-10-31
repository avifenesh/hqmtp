# hqmtp — from "heal the quantized MTP head" to a 212M drafter at 85% of teacher chain acceptance

Research harness for multi-token-prediction (MTP) draft heads on Qwen3.5-9B:
extraction, training, replay evaluation, corpus controls, and the negative-result
ledgers behind the working paper
[Small-vocabulary MTP heads for memory-bound speculative decoding](https://avifenesh.ai/research/small-vocabulary-mtp/).

The measurement instrument is [memra](https://github.com/avifenesh/memra), a
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
6. **Fidelity cuts vs function cuts.** Teacher-forced audits on Qwen3.5-9B and
   Qwen3.6-27B split all head compression into two families. Function cuts (pruning,
   low-rank, distillation) pay 10 to 19 retention points on shifted text; fidelity cuts
   (precision) cost at most 1.9. The zero-training recipe wins: trim the head to the
   model's own 32k hottest tokens, quantize to NVFP4, verify full-vocabulary. In live engine
   batteries it runs 1.8 to 2.7x end-to-end over plain decode, distribution-exact. Paper: `paper/`
   ([`paper/latex/main.pdf`](paper/latex/main.pdf)); tools: [`distill/README.md`](distill/README.md);
   ledger: [`results/engine-e2e-v1.md`](results/engine-e2e-v1.md).

## Best trained student

Self-drafting chain acceptance, 45 held-out own-generation docs, 4 recursive slots:

| checkpoint | slot 0 | slot 1 | slot 2 | slot 3 | chain |
|---|---|---|---|---|---|
| teacher (co-trained, engine chain) | 0.740 | 0.598 | 0.520 | 0.470 | 0.593 |
| naked KD 24k | 0.706 | 0.598 | 0.527 | 0.448 | 0.490 |
| **+ CE reinforce 4k** | **0.708** | **0.611** | **0.548** | **0.477** | **0.507** |

**0.507 = 85.5% of the teacher's chain acceptance at ~13% of the draft compute.
Slots 1–3 all exceed the teacher.** The remaining gap is slot 0 only; the KD curve
is near-saturated at 24k steps. Live in the engine, the student ties the natural
head at long-context greedy (1.70x vs 1.71x), but the trimmed head matches or beats
it in every reliably measured cell, so the deployment answer is: trim, no training
([`results/engine-e2e-v1.md`](results/engine-e2e-v1.md)).

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
- `paper/`: the Fidelity Cuts vs Function Cuts paper. Section drafts, LaTeX build,
  figures (`paper/figs/make_figs.py` reads `results/*.json`).
- `ops/box/`: GPU box runbooks. Resumable environment restore, probe chains.

## Reproducing

The engine-side runners (`eval/run_eval.sh`, `ops/box/`) run on the GPU box against
the engine under its earlier name, bw24 (`~/bw24`, `bw24-server`, `BW24_*` env vars).
Current memra reads `MEMRA_*` (the draft file is `MEMRA_MTP_DRAFT`). The PyTorch
side (`distill/`) is standalone.
Checkpoints and extracted `.npz`/`.tt.npz` training data are not in the repo
(see `.gitignore`); `distill/DISTILL.md` documents the exact extraction and
training commands that produced every number above.

Related public threads: FR-Spec-style trimming for native MTP —
[llama.cpp #25187](https://github.com/ggml-org/llama.cpp/issues/25187).

## Status

The measurement program is closed. Wall-time batteries, the 27B replication and
the multi-seed sampled protocol are in `results/engine-e2e-v1.md`. MIT license.
