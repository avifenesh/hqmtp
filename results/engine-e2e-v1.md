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

## Utilization probe (2026-07-13): the natural block is NOT all in use

User question ("how do we know all pieces are in use?") -> activation-mass probe +
physical ablation, zero training, serving-regime eval (slot0, windowed torch):

| cut | slot0 | % of natural |
|---|---|---|
| none | 0.886 | 100% |
| FFN keep 50% | 0.872 | 98.4% |
| FFN keep 25% | 0.860 | 97.1% |
| zero 2 weakest heads | 0.885 | 99.9% |
| zero 8 of 16 heads | 0.852 | 96.2% |
| FFN 25% + 2 heads | 0.860 | 97.1% |

Head-mass profile: heads 8-11 carry ~0.3 relative mass. => "trim++" arm: vocab trim
+ activation-ranked FFN prune, exported physically (trimplus25 281MB / trimplus50
321MB vs frtrim 401MB vs student 225MB draft files). No engine change needed (n_ff
from tensor dims). 5-arm final battery queued: natural / frtrim / trimplus25 /
trimplus50 / student. Paper frame shifts to a cost/acceptance FRONTIER: trim ->
prune -> distill; distillation buys the last ~2x block cut + deep-slot chain gains.

## 27B replication (2026-07-14, sv27-kd-ce: KD 24k* + CE 4k, serving-regime corpus)

*KD restarted from step 12k after a disk-full crash (12k + 12k @ lr 2e-4).

Torch chain (27B serving-regime eval): teacher 0.662, student 0.605 = 91%.
E2E (plain 47.0 tok/s, K=4):

| prompt | nat greedy | nat t0.7 | student greedy | student t0.7 |
|---|---|---|---|---|
| chat | 45.2% 1.82x | 40.5% 1.65x | 13.9% 1.17x | 13.0% 1.12x |
| code | 59.7% 2.18x | 52.1% 1.92x | 23.3% 1.45x | 16.7% 1.22x |
| longchat | 77.6% 2.62x | 79.1% 2.56x | 41.9% 1.98x | **69.0% 2.68x** |

**Student BEATS natural at long-ctx SAMPLED — the serving mode — 2.68x vs 2.56x
(125.6 vs 120.3 tok/s) at 69% acceptance.** Replicates: short-ctx tax, long-ctx
strength. Does NOT replicate: the 9B sampled collapse (9B's KD mixed v1-regime
data; 27B trained on serving-regime corpus only) and the 9B longchat-greedy
self-consistency FAIL (all 27B cells PASS).

## FINAL frontier tables (2026-07-15, clean baselines 137.7 tok/s, K=4)

MT-Bench 80 prompts (short regime):
| arm | greedy acc -> x | t0.7 acc -> x (3 seeds) |
|---|---|---|
| natural | 50.1% -> 1.59x | 45.5-49.3% -> 1.38-1.45x |
| frtrim | 47.0% -> **1.86x** | 43.8-45.9% -> 1.68-1.72x |
| trimplus50 | 40.1% -> 1.72x | 37.3-38.9% -> 1.54-1.58x |
| trimplus25 | 34.5% -> 1.58x | 32.5-33.1% -> 1.43-1.45x |
| student sv2 | 20.8% -> 1.25x | 18.4-19.0% -> 1.11-1.13x |

Longchat 3.5k greedy K=4: frtrim 63.4% 292.9 (2.15x) > student 53.4% 273.9 (2.01x)
> trimplus50 45.3% 240.5 (1.76x) > natural 59.9% 234.4 (1.71x) > trimplus25 41.3%
230.2 (1.69x). K=8 counterproductive on all arms (0.69-1.01x; student degrades least).

27B extras: probe replicates over-provisioning HARDER (FFN 25% = 98.5% slot0);
e2e longchat greedy: frtrim27 66.1% 2.57x, trimplus25-27b 57.6% 2.38x (vs natural
2.62x, student sampled 2.68x = best 27B cell).

KEY MECHANISM FINDING: pruning's teacher-forced slot0 (97-98.5%) massively
overstates its LIVE chain acceptance (41-45% vs frtrim 63%) — FFN pruning distorts
the h_nextn carrier and errors COMPOUND down the chain. Distillation trains the
carrier (self-recursive chain KD) — the only method that holds deep slots, hence
the student's serving-mode long-ctx wins despite lower slot0.

Verdicts: trim = production default; prune = VRAM lever with a chain tax (report
per-slot, not slot0); distill = long-ctx/sampled specialist + only path below
trim's cost floor. Seeds: +-2pts acc, +-0.04x.

## Root cause of the prune live-tax (2026-07-15): the tail IS the generalization

Carrier-compounding theory REFUTED: torch 4-slot self-draft chain on pruned teacher
= 0.710-0.715 vs full 0.735 (97%); engine FORCED replay agrees (tp25 = 96.7% of
frtrim). Export + engine exonerated. Robust 100-doc max-over-docs calibration:
identical rankings (88% overlap), identical live acceptance (41.7% vs 41.3%).

Decisive cell — forced replay ON LIVE-GENERATED text:
| arm | corpus text chain | live text chain |
|---|---|---|
| frtrim | 0.945 | 0.427 |
| trimplus25 | 0.914 (97% of frtrim) | 0.290 (68% of frtrim) |

Pruning's tax is DISTRIBUTION-DEPENDENT: ~3% in-calibration-distribution, ~35%
relative on fresh live content. The low-mass channels are not dead — they are the
long tail that fires on unfamiliar content. **Over-provisioning is the
generalization margin.** Same mechanism explains the distilled student's fresh-
topic/short-ctx weakness: every FUNCTION cut (prune or distill) is brittle exactly
off-distribution; vocab trim is robust because it removes no computation.

Paper conclusion (final): draft-cost cuts are free until you touch function;
function cuts pay in the tokens you haven't seen.

## Adaptive (online) pruning — tested, negative (2026-07-15)

User hypothesis: learn the prune mask from live traffic instead of offline calib.
Simulated deployment on a 24-doc fresh-topic stream (served in order, EMA mask
rebuilt from previously-served docs only): adaptive == static within noise
(0.8456 vs 0.8442 slot0); mask calibrated ON the stream itself == corpus mask
(0.8542 both, converged half). Mechanism: activation-mass ranking is dominated by
common channels regardless of calibration source; the OOD-critical tail channels
are rare-firing BY DEFINITION and never enter a top-25% cut. Learning them needs
per-channel acceptance gradients — i.e. training. The zero-training frontier is
closed: trim robust; prune carries an irreducible OOD chain tax.

## Vocab-conditioned pruning ("mark functions orphaned by the trim") — tested, negative (2026-07-15)

User hypothesis: channels serving TRIMMED (cold) tokens are dead by construction
post-trim — rank by hot-logit energy ratio ||lm_head[hot] w_down[:,i]||^2 /
||lm_head w_down[:,i]||^2 (weights-only). Measured: the ratio is UNIFORM across
all 12288 channels (median 0.135, p10-p90 0.116-0.159 ~= the hot set's 13% vocab
share). No channel is vocabulary-specialized — FFN features live in superposition
and every channel's output moves hot and cold logits alike. Pruning by the ratio
is WORSE than mass (93.8% vs 97.1% in-dist); combined score adds noise-level gains.
Conclusion: vocab trim orphans OUTPUTS, never COMPUTATION — the same fact that
makes trim robust makes it create zero pruning headroom. The OOD-critical tail
channels are rare-CONTENT specialists whose outputs still land on hot tokens.

## Sampled-cell variance correction (2026-07-17) — headline revision

Multi-seed audit of single-prompt SAMPLED cells (27B longchat, student): seeds
{7,17,27,37} -> 20.0/43.1/26.1/44.7% (mean ~33%, sd ~12pts). The 69% / 2.68x
"best 27B cell" was a single favorable seed — RETRACTED as a point estimate.
Eager==graph bit-identical; FA-kernel-flavor toggle alone moves the trajectory
(37.6%) — content variance, the robustness law expressed per-trajectory. Greedy
cells (deterministic) and MT-Bench sweeps (80-prompt x 3-seed aggregates) stand.

Deployment verdict FINAL: frtrim >= student in every reliably-measured live cell
for this model family — **if the model ships a co-trained MTP head: trim it and
stop; no training**. The distillation recipe's value: models WITHOUT a co-trained
head, heads broken by base quantization, and the science (deep-slot trainability,
regime laws). bw24: dispatch PR #13 closed unmerged (owner call), #12 closed
not-a-bug (multi-seed protocol note), DraftGeom (#7) stays.

## QUANTIZE THE DRAFT (2026-07-17) — the positive result

frtrim 27B sampled spread completes the correction set: seeds {7,17,27,37} ->
60.8/60.7/41.5/70.5 (mean ~58% +-12; variance arm-independent, trajectory-driven;
frtrim mean 58 vs student mean 33).

Q4_0 draft (trim + 4-bit, 212MB vs Q8's 401MB, zero training), 9B:

| cell | frtrim Q8 | frtrim Q4_0 |
|---|---|---|
| longchat greedy | 55.5% -> 283.4 (1.98x) | **57.2% -> 303.5 (2.12x)** |
| chat greedy | 38.4% -> 227.5 (1.58x) | 37.9% -> 238.9 (1.65x) |
| OOD live-text chain | 0.427 | 0.419 (98%) |

**4-bit draft = free: acceptance within noise everywhere INCLUDING off-distribution
(-2% rel, vs prune's -32%), +7-11% tok/s, 47% smaller.** The robustness law's
constructive corollary: UNIFORM cuts (precision) keep the generalization tail;
SELECTIVE cuts (channels, distill) lose it. Draft-step reads: natural 1.33GB ->
trim-Q8 0.40GB -> trim-Q4 0.21GB (~6x), zero training, error-tolerant by
construction. Note: prior "-15pts Q4_K external draft" engine datum was a
cross-converter artifact — clean same-source Q4_0 shows none of it.

Positive-paper direction (owner gate passed: faster AND smaller): "Quantize the
Draft". Ladder: NVFP4 head (~120MB), 27B replication, long-prompt SET battery
(coverage fix), MT-Bench Q4 arm, multi-seed protocol from day one.

## STRONG BATTERY (2026-07-17) — Quantize the Draft, final 9B tables

Protocol: aggregates only — MT-Bench 80 prompts (short, NGEN 256) + longset 20
prompts 2-6k tokens (NGEN 512), greedy + 3 sampled seeds, K=4, resident-model
sweeps, clean baselines (gen 143-144.5 all arms; one contended cell rerun clean).

MT-Bench: natural 50.1% 1.52x | trim-Q8 47.0% 1.78x | trim-NVFP4 45.6% **1.82x**
  sampled: natural 1.31-1.38x | Q8 1.60-1.64x | NVFP4 1.63-1.64x
Longset: natural 61.6% 1.72x | trim-Q8 60.0% **2.04x** | trim-NVFP4 54.7% 1.98x
  sampled: natural 1.51-1.52x | Q8 1.49-1.81x (acc ±7pts!) | NVFP4 **1.77-1.83x (acc ±1.2pts)**

Verdict: **trim+NVFP4 (212MB, zero training, verify-lossless) = 1.63-1.98x across
every regime**: best short arm, best sampled-long arm, within 3% of Q8's single
greedy-long win. 4-bit acceptance cost 1.4pts short / 5.3pts long-greedy, bought
back by halved reads. UNEXPECTED FINDING: NVFP4 stabilizes sampled trajectories
(seed sd 1.2pts vs Q8's 7pts) — smoother quantization grid -> less trajectory
sensitivity. Aggregate protocol validated: single-prompt sampled sd ±12pts ->
20-prompt aggregate ±1.2pts (NVFP4) — reproduces across builds.

## 27B strong battery (2026-07-17) — replication complete

MT-Bench: natural 53.2% 2.02x | trim-Q8 50.4% 2.15x | trim-NVFP4 48.6% 2.15x
  sampled: natural 1.81-1.82x | Q8 1.89-2.03x | NVFP4 1.96-2.02x
Longset: natural 69.7% 2.42x | trim-Q8 69.5% 2.67x | **trim-NVFP4 67.5% 2.67x
  (127.5 tok/s = fastest cell of the research; NVFP4 > Q8 outright at 27B)**
  sampled: natural 2.15-2.31x | Q8 1.60-2.41x | NVFP4 2.01-2.44x (stabler again)

Replicates 9B on every axis: trim free (-0.2pts long), 4-bit ~free, NVFP4
preferred (speed at scale + sampled stability). Cross-model law: spec gains GROW
with model size — NVFP4-trim 1.6-2.0x @9B -> 2.0-2.7x @27B, zero training,
212MB/259MB draft files, verify-lossless. Independent convergence: engine lane's
own-trim + NVFP4-head + q4_K-block arm beat its prior champion on BOTH axes
(100.7 tok/s, 85.2% acc, K=3 interleaved) on a third config.

EXPERIMENTAL PROGRAM COMPLETE for "Quantize the Draft": two models x two suites x
multi-seed + third-config independent replication + mechanism theory (robustness
law) + methodology (aggregate protocol). Remaining: prose.

## AUDIT LAB (2026-07-18) — the pre-deployment robustness audit, validated

7 compression arms on the trimmed 9B head (torch slot0, in-dist gen2-eval 2500
pos + OOD fresh-live 2500 pos, shared reference ctx): uniform (q8/q4/nvfp4
weight roundtrips) vs selective (prune25/50 act-mass masks; SVD-768 data-free +
activation-whitened low-rank FFN, byte-matched to prune25).

| arm | in-dist disagree | harm/disagree | OOD tax |
|---|---|---|---|
| q8/q4/nvfp4 | 0.003-0.044 | ~0.14 | -0.008..-0.002 (ZERO) |
| prune50/25 | 0.050/0.083 | 0.28/0.30 | +0.005/+0.022 |
| svdw/svd768 | 0.155/0.166 | ~0.54 | +0.035/+0.048 |

1. LAW CONFIRMED, STRONG FORM: all dimension cuts pay OOD tax incl. LOW-RANK;
   data-free SVD pays MORE than calibrated SVD -> dimensionality is the axis,
   not calibration-dependence (the 2x2 disconfound nobody ran).
2. AUDIT: in-dist disagreement-with-reference predicts OOD tax at Spearman 0.89
   (7 arms); harm-per-disagreement separates families (quant disagreements are
   accuracy-neutral tie-flips ~0.14; dimension-cut disagreements are function
   loss 0.28-0.54). Practical rule: forward passes on in-dist docs only ->
   low harm-rate = OOD-safe; high harm-rate = expect OOD tax ~0.5-1x in-dist
   damage. NO OOD DATA REQUIRED.
3. v1 statistic (rare-share of disagreement) FALSIFIED — kept for honesty;
   rarity-concentration is prune-specific (SVD damages broadly), harm-rate is
   the universal signal.
Tooling: distill/audit_lab.py. Next: divergence-tail (canary) pruning calibration.

## E4 — divergence-tail (canary) pruning calibration: negative, completes the set (2026-07-18)

Restore +5% budget to prune25 chosen by (a) canary-conditioned mass (channels
firing at the 417 in-dist divergence positions) vs (b) global mass (control):
canary in-dist 0.8544 / OOD 0.8052; global 0.8596 / 0.8068 (full 0.8784/0.8404).
Canary restoration recovers LESS than the naive control (sets 102/614 overlap —
real comparison). SIX selection schemes now fail identically: activation-mass,
Wanda act*weight, wide calibration, online EMA, vocab-conditioned energy,
divergence-conditioned mass. Practical playbook line hardens: there is no known
safe pruning calibration for draft heads — quantize instead; the audit
(harm-rate) is the tool that tells you your cut is in the unsafe family.

PRACTICAL-PAPER MEASUREMENT PROGRAM COMPLETE:
1. Method: trim+NVFP4 one-command compression — 1.6-2.7x, two models, tooling done.
2. Instrument: in-dist harm-rate audit, rho 0.89 over 7 arms, no OOD data needed.
3. Playbook: quantize don't prune (6-scheme impossibility); distill only when no
   head exists; serving-regime data rules; aggregate benchmark protocol.
Remaining: prose + tooling README.

## TRUE domain-shift validation (2026-07-18, reviewer C1) — taxonomy AMPLIFIES

New suite: 24 docs of de/zh/he/ru Wikipedia + PubMed abstracts + C4 web text
(reference slot0 drops 0.878 -> 0.362 — genuinely shifted). 7-arm audit rerun:

| arm | in-dist ret | true-OOD ret | tax |
|---|---|---|---|
| q8 | 0.998 | 0.999 | -0.001 |
| q4 / nvfp4 | 0.994/0.993 | 0.977/0.975 | +0.017/+0.019 |
| prune50 | 0.984 | 0.882 | +0.102 |
| prune25 | 0.972 | 0.804 | +0.168 |
| svdw / svd | 0.905/0.898 | 0.737/0.711 | +0.168/+0.187 |

Family gap widens ~10x under genuine shift (fidelity <=2pts; function 10-19pts).
8-bit exactly free; 4-bit shows first visible tax under extreme shift (honest
refinement: fidelity cuts are CHEAP, 8-bit is FREE). In-dist disagreement ranking
predicts true-OOD tax monotonically (rho ~0.96) — audit strengthens on the harder
test. Reviewer delta's reject concern answered: same-register OOD UNDERSTATED the
law. Suite: distill/build_trueood.py -> /opt/dlami/nvme/hqmtp-trueood-*.

## Statistical tests (2026-07-18)

Binomial 95% CIs (delta method) + exact permutation tests:
- TRUE shift: Spearman(in-dist disagreement, tax) = 1.000, exact p = 0.0002
  (5040 permutations); family tax gap +0.145, permutation p = 0.0286 (35
  labelings = the n=7 minimum). All four function-cut taxes exclude 0
  individually; both 4-bit taxes are within CI of 0.
- Same-register: rho = 0.893, p = 0.0062; family gap +0.031, p = 0.0286.

## C5: base-quantization cross battery (2026-07-18)

Question (reviewer C5): does the recipe survive when the BASE model is quantized? 2x3x2 battery on 9B: {bf16 base, NVFP4-quantized base} x {natural head, frtrim+Q8 head, frtrim+NVFP4 head} x {greedy, t0.7s7}. MT-Bench 80 prompts, K=4, aggregate protocol.

| base | arm | greedy acc / x | t0.7s7 acc / x |
|---|---|---|---|
| bf16 (144.5 tok/s plain) | natural | 50.1% / 1.52x | 45.5% / 1.31x |
| bf16 | frtrimq8 | 47.0% / 1.77x | 45.9% / 1.64x |
| bf16 | nvfp4 | 45.6% / 1.82x | 43.1% / 1.64x |
| nvfp4-st-ct (190.5 tok/s plain) | natural | 51.9% / 1.27x | 47.0% / 1.09x |
| nvfp4-st-ct | frtrimq8 | 48.0% / 1.50x | 45.3% / 1.35x |
| nvfp4-st-ct | nvfp4 | 45.9% / 1.54x | 42.7% / 1.36x |

Findings:
1. Head-compression acceptance cost is base-independent: natural->nvfp4 head drops 4.5pts on bf16 base, 6.0pts on quantized base (greedy); compressed-head absolute acceptance within 1pt across bases (47.0 vs 48.0, 45.6 vs 45.9). Fidelity cuts to head and base do not compound.
2. Recipe composes with base quantization: best absolute cell = quantized base + NVFP4 head, 292.7 tok/s = 2.03x over bf16 plain decode (144.5). Relative speedup per-base shrinks (1.82x -> 1.54x) because the quantized base's plain decode is already 32% faster; absolute throughput still rises.
3. Natural head accepts slightly MORE against the quantized verifier (51.9 vs 50.1 greedy) - verifier smoothing, within seed noise; not a claim.

## C6: 27B audit replication (2026-07-18, true-OOD suite re-extracted with 27B trunk)

audit_lab full grid on Qwen3.6-27B heads (same 7 arms, svd rank 960 byte-matched), in-dist 2,500 positions, OOD = true domain-shift suite (de/zh/he/ru wiki + PubMed + C4). Reference in-dist acc 0.805 -> OOD 0.324 (2.5x drop, mirrors 9B's 2.4x).

| arm | in-dist disagree | shift tax (retention pts) |
|---|---|---|
| q8 | 0.60% | -0.1 |
| q4 | 3.88% | +0.2 |
| nvfp4 | 4.00% | +1.5 |
| prune25 | 8.16% | +11.6 |
| prune50 | 5.28% | +7.6 |
| svd960 | 14.36% | +12.7 |
| svdw960 | 11.88% | +12.5 |

Family separation replicates: fidelity <=1.5pts, function 7.6-12.7pts. **Disagreement ranks tax at Spearman 1.0 again** (exact ranking on both models). Taxonomy + audit both model-independent within the family.

## Entropy study (9B): why NVFP4 stabilizes sampled acceptance

KL(ref||q8) mean 0.0003 (p99 0.0024) — Q8 is the reference. KL(ref||nvfp4) mean 0.020 (p99 0.19).
NVFP4 raises draft entropy 0.686 -> 0.721 (+5%) and lowers top1 margin 5.91 -> 5.70: 4-bit rounding acts as mild smoothing on the draft distribution. Candidate mechanism for the observed sampled-variance stabilization (NVFP4 seed spread ±1.2pts vs Q8 ±7pts on 9B longset): a slightly flatter draft is less trajectory-sensitive under temperature sampling. 20-seed study (running) quantifies the variance side.

## 20-seed sampled variance + mixed-precision arm (2026-07-18, megachain tail)

**Seed20** (9B, single longchat prompt, 512 gens, t0.7, seeds 7..197):
- frtrim-Q8 head: mean 59.0%, sd 13.2, range 22.3-79.7 (spread 57.4pts)
- frtrim-NVFP4 head: mean 50.1%, sd 14.0, range 13.1-63.6 (spread 50.5pts)

Single-prompt sampled cells are far worse than the 4-seed estimate suggested (12pts): the 20-seed tails span ~50-57pts. **Correction to 814cdb2's "NVFP4 stabilizes sampled variance": dead.** At 20 seeds the two formats' spreads are indistinguishable (sd 13.2 vs 14.0). The earlier ±1.2 vs ±7pts contrast was 3-seed suite-level luck. Aggregation is the only stabilizer. Entropy study's smoothing (KL 0.020) is real but has no measurable variance effect.

**Mixed arm** (--mixed: NVFP4 lm_head + Q4_0 block, trim 32k; artifact 212,553,280 B = byte-same as all-NVFP4):
| suite | greedy | t0.7s7 |
|---|---|---|
| MT-Bench | 45.1% / 1.81x | 41.8% / 1.61x |
| longset | **58.3% / 2.09x** | 50.6% / 1.75x |

2.09x = new best 9B long cell (vs Q8 2.04, all-NVFP4 1.98). vs all-NVFP4: same size, +3.6 acceptance pts long-greedy -> the 9B long 4-bit acceptance cost lives in the BLOCK codec, not the projection. Projection NVFP4 free everywhere; block prefers Q4_0 at 9B long. Matches engine-lane q4_K-block winner.

MEGACHAIN COMPLETE: C5, C6, entropy, seed20, mixed all landed.

## Round-2 review response experiments (2026-07-18)

**Leave-one-domain-out Spearman** (per-domain audit re-run, 800 pos/domain): full rho=1.000; dropping any single domain keeps rho in {0.964, 1.000} (worst = one adjacent-pair swap). Family gap survives every drop (fidelity <=2.5pts, function >=5.0pts, worst case drop=wiki-de). Ranking is not a lynchpin-document artifact. results/audit-perdomain.json.

**Hot-32k coverage per domain** (hotcov.json): INDIST 94.0%, c4web 94.9%, pubmed 92.8%, wiki-de 61.7%, wiki-ru 21.1%, wiki-he 9.5%, wiki-zh 5.4%. Confirms the honest scoping: trim's coverage is register-bound (collapses on non-Latin scripts) while the retained function is not (4-bit arms keep >=97.5% within-hot retention on the same corpora). Trim tax = coverage gap, measurable pre-deployment; function tax = invisible without the audit.

**Rare-share audit v1 concrete numbers** (9B): rare-share of disagreement q4 0.316, nvfp4 0.327 vs prune25 0.377, prune50 0.357, svd768 0.228, svdw768 0.265 — quantized arms sit inside the function arms' range; statistic separates nothing. (27B: same overlap.)

**Wanda scheme cell** (recovered to record): wanda25 slots 0.859/0.792/0.698/0.637 chain 0.712 vs mass25 chain 0.712; ranking overlap 88%; identical live acceptance. Wanda == mass for this head.

## Mixed arm full grid (2026-07-18, r2chain tail)

27B mixed export: 333MB (NVFP4 proj + Q4_0 block, hot32k-27b).
| cell | acc / x |
|---|---|
| 27B mtbench greedy | 44.7% / 2.04x |
| 27B mtbench t0.7 s7/s17/s27 | 42.3/36.6/42.6% -> 1.92/1.75/1.93x |
| 27B longset greedy | **68.2% / 2.70x (128.6 tok/s) = NEW BEST CELL** (beats all-NVFP4 2.67x/127.5) |
| 27B longset t0.7 s7/s17/s27 | 58.9/37.6/51.8% -> 2.37/1.77/2.17x |
| 9B mtbench t0.7 s17/s27 | CONTENDED (gen 107-121 vs 144.5 clean) - rerun in flight |
| 9B longset t0.7 s7/s17/s27 | 50.6/48.2/66.4% -> 1.75/1.72/2.16x |

Block-codec law refined: Q4_0 block wins LONG suites at both scales (2.09x@9B, 2.70x@27B); NVFP4 block wins SHORT at 27B (2.15 vs 2.04). Projection NVFP4 free everywhere. Deployment picks block codec per regime.

Longset sampled seed spread: mixed 9B 1.72-2.16 (0.44x), 27B 1.77-2.37 (0.60x) — 20-prompt suite at t0.7 does NOT pin speedup to 0.04x; that tightness is MT-Bench-only (80 prompts). Protocol note scoped accordingly in paper §4.0.

Rerun of contended 9B mtbench mixed sampled cells (clean gen 144.3): s17 43.4%/1.65x, s27 43.0%/1.64x (s7 41.8%/1.61x). Mean 1.63 == Q8/NVFP4 sampled 1.64; MT-Bench tie-threshold (0.04x) holds for mixed too. Contended cells (1.48/1.51) superseded.

R2CHAIN COMPLETE. Paper measurement program closed: all review-response experiments landed.

## Direction probes (2026-07-19)

**P1 — compression unlocks speculation depth (27B longset greedy):**
| arm | K=4 | K=5 | K=6 | K=8 |
|---|---|---|---|---|
| natural | 2.42x (115.3) | - | 2.44x (116.4) | - |
| trim-NVFP4 | 2.67x | - | 2.74x (130.6) | 1.24x |
| trim-mixed | 2.70x | **2.77x (132.1)** | 2.75x | 1.22x |

Natural head is depth-neutral (2.42 -> 2.44 at K=6). Compressed heads gain +0.05-0.07x moving K 4->5/6. Compression shifts the optimal speculation depth because the drafting round is cheaper; K=8 hits an engine cliff for all arms. New best cell 132.1 tok/s (2.77x).

**P4 — audit generality on the BASE model (9B, last-12-layer FFN arms, torch):**
reference acc indist 0.703 / trueood 0.544.
| arm | ret indist | ret ood | shift tax | disagree | harm/dis |
|---|---|---|---|---|---|
| q4 | 0.998 | 0.995 | 0.3pt | 3.8% | 0.034 |
| prune25 | 0.940 | 0.603 | 33.7pts | 23.9% | 0.177 |
| svd35 | 0.853 | 0.715 | 13.8pts | 27.1% | 0.382 |

Taxonomy replicates AND amplifies on the base model (function tax up to 34pts). Harm separates families (0.03 vs 0.18-0.38). Caveats: within-function disagreement->tax ranking flipped on n=3 (arms not byte-matched); head-fit harm thresholds (0.15/0.25) put base prune25 in the gray zone -> thresholds object-specific, separation general. distill/p4 script: box ~/hqmtp-distill/p4_base_audit.py, base-audit.json.

P2 (quantized-base 27B compose) rerun in flight w/ correct gguf path incl. engine-lane frspec32768 arm. P3 (zh re-rank) rerun in flight (first pass grep-filtered the table rows).
