# SV head — production round 2 (2026-07-10)

Batch-64 teacher-free CE run, 15k steps (960k samples), 113-file corpus (~7M tokens),
window 2048. Eval: torch replay slot-0, window 2048.

| head | eval-code | gen-dist | draft FLOPs vs teacher |
|---|---|---|---|
| engine (full ctx) | 0.678 | 0.348 | 1x |
| teacher (windowed) | 0.675 | 0.371 | 1x |
| sv @ 960k samples | 0.527 | 0.166 | ~0.13x |
| sv @ 60k samples | 0.337 | 0.105 | ~0.13x |
| s2-b0 full-vocab @ 60k | 0.373 | — | ~0.35x |

- 78% of teacher acceptance at ~1/8 draft compute; curve unsaturated (loss ~1.9 falling).
- Vocab ceiling not binding (0.989 code coverage). Gap = budget + capacity.
- Chat domain weak (0.166): 20% mix + 0.946 coverage — chat-weighted data/vocab lever.
- GPU lesson: CE-only runs were paying a useless teacher pass per step + batch 12 left
  2/3 of VRAM idle; fixing both tripled effective throughput (34->56GB, 487->550W).

## Own-output rounds (2026-07-11, user rule: model-exact data only)

In-distribution eval = fresh unseen model-generated docs (45 docs, 14,111 positions):

| head | slot0 |
|---|---|
| engine (full ctx) | 0.872 |
| teacher (windowed) | 0.879 |
| sv-own (12k steps, 1.3M own tokens) | 0.548 |
| sv-30k (40k+ steps, 13M generic tokens) | 0.376 |

Own-output training beats generic by +17pts at 1/3 the steps — the training-distribution
rule validated cleanly. Earlier "memorization" read was an eval artifact (gen-dist = old
greedy degenerate transcripts ≠ the temp-0.7 training distribution; retired as primary).
Teacher ceiling on own text 0.879: serving regime is the easy regime. Round 2 running:
2000 docs (~6.5M own tokens), batch 128, held-out split as the standing eval.

## Own-output round 2 (2026-07-11): scaling confirmed

1,380 docs / 5.2M own-tokens (4x round 1), batch 128, 10k steps, clean warm-start.
Held-out split = 45 unseen own-output docs (12,117 positions):

| head | slot0 | % teacher |
|---|---|---|
| engine (full ctx) | 0.853 | — |
| teacher (windowed) | 0.863 | 100% |
| sv-own2 (10k, 5.2M tok) | 0.636 | 74% |
| sv-own2 @7.5k | 0.624 | |
| sv-own (round 1, 1.3M tok) | 0.550 | 64% |

- 4x data => +8.6pts at identical architecture; curve decelerating but still rising
  (+1.2pts over the last 2.5k steps).
- 74% of teacher acceptance at ~13% draft compute, 32k vocab, 212M params.
- Code control 0.384: own-output specialization trades generic-code acceptance —
  correct trade for the serving regime.
- Next-lever menu: (a) 4x data again (~20M tokens, ~6h gen at current serve rate,
  projected ~0.69-0.70), (b) bigger student (d_s 2560/3072 — capacity may bind
  before data), (c) chain training slots 1-3 (multiplies accepted length, not slot0),
  (d) engine integration of current best (d2t GGUF export -> real e2e tok/s).

## Chain distillation on copied teacher logits (2026-07-11)

Init sv-own2, 12k steps batch 64, per-slot top-64 KL (precomputed teacher targets,
teacher_targets.py) + CE anchor, self-recursive carrier. Held-out 45 docs, 12,112
positions, SELF-drafting chain eval (engine regime):

| head | slot0 | slot1 | slot2 | slot3 | chain |
|---|---|---|---|---|---|
| engine teacher | 0.853 | 0.694 | 0.570 | 0.378 | 0.593 |
| sv-chain | 0.679 | 0.563 | 0.468 | 0.405 | 0.449 |
| sv-own2 (pre-chain) | 0.636 | 0.167 | 0.051 | 0.021 | 0.196 |

- Slots 1-3 from dead to strong; slot-3 ABOVE teacher (student trained for the
  approximate-carrier regime the teacher only encounters at inference).
- Chain acceptance 76% of teacher at ~13% draft compute; slot0 80%.
- Distillation-from-the-head vindicated: same data, same arch, the copied logits are
  what moved slot0 (+4.3pts) and unlocked the chain.
