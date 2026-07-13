# Engine integration + first e2e — sv-kd24-ce as live draft head

Engine side (bw24 PR #7, merged as f1395e90): `DraftGeom` on `MtpHead` — narrow
student block (d_inner 2048, 8q/2kv) + `out_up` back to n_embd, selected by
`blk.N.nextn.out_up.weight` in a `BW24_MTP_DRAFT` GGUF. d2t infra reused. Export:
`distill/export_sv.py` (Q8_0, **neox row permutation** of wq/wk + q/k_norm — student
trains interleaved rope, engine kernel is half-split; permuting rows inside the roped
block makes them exactly equivalent).

## Fidelity gate

Engine replay on training-distribution text ≈ torch (chain within doc-noise;
slot0 draft agreement 0.78–0.87). Ruled out empirically: attention kernels (scalar
FA identical), weight quant (F16 export identical), activation quant, scratch-KV
quant (torch q8_0/q5_1 simulation identical), thinking-template mismatch, rope
precision. One phantom chase: single-doc long-bucket "collapse" was doc variance —
torch same doc same bucket 0.399 vs engine 0.373.

## E2E (512-token gens, 9B bf16 ST, K=4, tok/s vs 136 plain decode)

| regime | natural head | student sv-kd24-ce |
|---|---|---|
| short-prompt greedy chat | 42.0% acc, 186.7 (1.37x) | 13.4% acc, 138.6 (1.01x) |
| short-prompt greedy code | 41.8% acc, 185.5 (1.36x) | 16.6% acc, 148.8 (1.09x) |
| short t0.7 chat | 49.7% acc, 191.2 (1.40x) | 18.0% acc, 144.3 (1.05x) |
| **long-prompt (3.5k) greedy** | 60.5% acc, 231.9 (**1.71x**) | 41.9% acc, 230.8 (**1.70x**) |
| long-prompt t0.7 | 63.3% acc, 221.6 (1.63x) | 13.5% acc, 126.9 (0.94x) |

**Headline: at long-ctx greedy the student MATCHES the natural head e2e (1.70x vs
1.71x) with only 2/3 its acceptance — the cheap-round advantage is real, measured.**
(Long greedy runs printed self-consistency FAIL on BOTH arms — engine-level long-ctx
exactness quirk, natural head included; flagged to the engine lane.)

## Two gaps, both diagnosed

1. **Short-context acceptance.** Not live-specific: torch position-buckets show the
   student was always weak early (chain 0.194 @ pos<512 vs 0.555 mid, 0.678 @2048+;
   teacher 0.292/0.651/0.748). Every short e2e run lives entirely in that bucket.
   v1 corpus (gen_corpus.py) was also thinking-DISABLED (`enable_thinking=False` —
   empty think blocks) and t0.7-only; serving is thinking-ON with greedy in the mix.
2. **Sampled-draft collapse (13.5% at t0.7 even long-ctx).** KD shapes only the
   teacher top-64; the remaining 32k tail is unshaped, so gumbel draws at t0.7 hit
   junk mass. Greedy drafting unaffected (argmax). Candidate fixes: engine draft-side
   temperature knob (scale draft logits before gumbel AND before retained q —
   rejection sampling stays exact), and/or tail-aware KD.

## Round in flight (v2 pipeline, box)

gen_corpus2.py: thinking ON, 50% greedy / 50% t0.7, 2/3 single-shot, fresh seeds,
2400 docs -> extract -> teacher targets -> KD 24k on v1+v2 mix -> CE 4k ->
eval both sets -> export -> e2e battery. Stages marked in v2-status.txt.

## v2 result (2026-07-13, sv2-kd-ce = KD 24k on v1+v2 mix + CE 4k)

Torch: v1 set 0.521 (new best, 88% of teacher), v2 serving-regime set **0.630 vs
teacher 0.641 = 98%** (sv1 scored 0.359 there — the regime gap, closed by data).

Clean e2e battery (512 gens, K=4, tok/s vs 136 plain):

| prompt | natural greedy | sv2 greedy | natural t0.7 | sv2 t0.7 |
|---|---|---|---|---|
| chat short | 42.0% 1.35x | 21.6% 1.23x | 49.7% 1.38x | 31.0% 1.35x |
| code short | 41.8% 1.36x | 16.7% 1.03x | 40.2% 1.07x | 11.7% 0.91x |
| longchat 3.5k | 60.5% 1.60x | **53.4% 1.99x** | 63.3% 1.56x | 23.6% 1.35x |

**HEADLINE: at long context the student now BEATS the natural head e2e by +49%
(269.6 vs 181.5 tok/s) with lower acceptance — the cheap-round advantage compounds
with depth, where spec matters most.** Remaining acceptance-side gaps: short-ctx
(structural), code register, long+sampled tail. Longchat-greedy self-consistency
FAIL reproduces on BOTH arms — engine-lane issue, arm-independent.

## MT-Bench 80-question sweep (2026-07-13, K=4, 256-token gens) — RATIO-ONLY

GPU contention artifact (27B corpus generation shared the card — see repo log):
absolutes invalid, within-run gen/spec ratios approximately fair. Clean pass queued.

| arm | greedy acc | greedy x | t0.7 acc | t0.7 x |
|---|---|---|---|---|
| natural | 50.1% | 1.59x | 45.5% | 0.92x |
| FR-trim 32k (in-engine gather) | 46.9% | 1.51x | 45.8% | 1.00x |
| student sv2 | 20.8% | 0.98x | 19.0% | 0.66x |

Readings: (a) FR-trim is a STRONG baseline in bw24 (llama.cpp's +1-3% does not
transfer — bw24's draft loop is lm_head-bound enough that trim pays); (b) MT-Bench
= short prompts, the student's weak regime — matches the short-ctx bucket story;
(c) sampled spec underperforms on short generations for ALL arms here (0.66-1.00x)
— overhead + acceptance; needs the clean pass before conclusions.
