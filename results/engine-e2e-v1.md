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
