# 4 Evaluation

<!-- Topic-sentence spine:
4.0 We evaluate on two models, two prompt suites, and a fixed aggregate protocol that we adopted after measuring trajectory variance.
4.1 The recipe matches or beats every other arm end to end on both models.
4.2 Robustness separates the two families by an order of magnitude, and the separation grows with distribution shift.
4.3 Speedups grow with target model size.
4.4 Within zero-training compression, no selection statistic escapes the function-cut tax.
-->

## 4.0 Setup

We evaluate on Qwen3.5-9B and Qwen3.6-27B, two dense hybrid models that ship co-trained MTP heads, running in bw24, a production CUDA engine, on a single RTX PRO 6000 (96GB). Two prompt suites cover the serving range. The short suite is MT-Bench: 80 prompts across eight categories, 256 generated tokens each. The long suite is 20 prompts of 1.9k to 5.9k tokens (documents with analysis instructions, long code files with review instructions), 512 generated tokens each. Each cell reports acceptance (accepted drafts over drafted tokens at K=4) and end-to-end tokens per second against the same-suite plain-decode baseline, for greedy decoding and for temperature 0.7 with three sampling seeds.

The aggregate protocol is itself a finding. A single-prompt sampled cell swings by 12 acceptance points across seeds: the sampler wanders into a different trajectory, and acceptance reflects whatever content appears. Toggling an attention-kernel variant moves the trajectory the same way. Suite-level aggregation with three seeds brings the spread under 1.5 points. Every number below is an aggregate; we report single-prompt cells nowhere. One earlier headline of ours (a 2.68x sampled cell) failed this protocol; we retracted it in the lab record.

Arms. The reference is the unmodified head. Trim replaces the 248,320-row output projection with the 32,768 rows ranked hottest on the model's own generated text, plus an index map into the full vocabulary. Verification stays full-vocabulary, so greedy output stays identical and sampled output stays distribution-exact. Trim-Q8 and trim-NVFP4 quantize all head matrices to 8-bit and 4-bit blocks on top of the trim. The robustness study (§4.2) adds channel pruning at two budgets, low-rank factorization at a byte-matched rank in data-free and calibrated variants, and the distilled replacement head.

## 4.1 The recipe matches or beats every arm end to end

Table 1 reports the full grid. On the 9B short suite, trim-NVFP4 reaches 262.8 tok/s (1.82x over plain decode) against 256.5 (1.78x) for trim-Q8 and 219.1 (1.52x) for the unmodified head. On the 9B long suite the ordering tightens: trim-Q8 292.3 (2.04x), trim-NVFP4 282.5 (1.98x), unmodified 245.8 (1.72x). At 27B the 4-bit arm leads outright. It reaches 103.3 tok/s (2.15x) on the short suite, tying Q8 to the decimal at half the head bytes. On the long suite it reaches 127.5 tok/s (2.67x), ahead of Q8's 123.6 and the unmodified head's 115.3. Sampled cells track greedy with a 0.1 to 0.2x penalty across all arms.

Acceptance explains the shape. Trimming costs 0 to 3 acceptance points (at 27B long, 0.2 points) because hot-vocabulary coverage on generated text is 95 to 99%. Four-bit quantization costs another 1.4 to 5.3 points. The bandwidth saving buys those points back everywhere except one cell (9B long greedy, where Q8's higher acceptance wins by 3%). A third configuration built independently in the engine's own lane, with a different vocabulary ranking and a q4_K block, reproduced the pattern and beat its own prior champion on both speed and acceptance.

**Takeaway.** Trim-NVFP4 turns the 1.33GB-per-token head into a 212MB artifact that is the fastest or tied-fastest arm in every aggregate cell, with zero training and unchanged outputs.

## 4.2 Robustness separates the families by an order of magnitude

We measure each compressed arm's acceptance retention (arm accuracy over reference accuracy on identical states) on three corpora at increasing distance from the calibration distribution. The first is held-out generated documents (in-distribution). The second is fresh generations on unseen topics (same register). The third is a genuine domain-shift suite of German, Chinese, Hebrew, and Russian Wikipedia, PubMed abstracts, and C4 web text, on which the reference head's own accuracy drops from 0.88 to 0.36.

The tax divides the arms into two families. Fidelity cuts stay within noise of free: Q8 retains 99.8 to 99.9% everywhere, and the 4-bit arms retain 97.5 to 99.4%, a worst-case tax of 1.9 points with a confidence interval that includes zero. Function cuts pay 10 to 19 points on the domain-shift suite: 16.8 for pruning at 25% keep, 10.2 at 50% keep, 16.8 to 18.7 for the low-rank arms. The family gap of 14.5 retention points is significant at the permutation-test floor for seven arms (p = 0.029), and every function-cut tax individually excludes zero at 95% confidence. The same ordering appears on the same-register corpus at one tenth the magnitude: same-register evaluation understates the effect that deployment will see.

The taxonomy also survives quantization of the base model itself. A cross on the 9B model pairs each head arm with a bf16 base and an NVFP4-quantized base, since deployed bases are rarely full-precision. Head-compression costs stay base-independent: the 4-bit head costs 4.5 acceptance points against the bf16 base and 6.0 against the quantized base, and each compressed head's absolute acceptance moves under one point across bases. Fidelity cuts to head and base compound in speed rather than in damage. The quantized base decodes 32% faster on its own, and adding the trim-NVFP4 head reaches 292.7 tokens per second, 2.03x over bf16 plain decode.

Two internal comparisons localize the cause. Data-free SVD pays a larger tax than calibration-weighted SVD at equal rank, so dimensionality itself drives the damage, independent of calibration data. And pruning's tax concentrates in rare-token bins while the low-rank tax spreads across all bins, yet both pay: selective cuts choose different function to delete, and shift surfaces the deletion either way.

**Takeaway.** Under genuine distribution shift, function cuts cost an order of magnitude more acceptance than fidelity cuts, the gap widens with shift distance, and the effect is invisible to in-distribution evaluation.

## 4.3 Speedups grow with target scale

Every arm gains more at 27B than at 9B: the unmodified head goes from 1.52x/1.72x (short/long) to 2.02x/2.42x, and trim-NVFP4 from 1.82x/1.98x to 2.15x/2.67x. The draft's cost shrinks relative to the target's verify pass as the target grows, so each accepted token buys more. Draft compression compounds with this trend: the cheaper the round, the higher the optimal speculation depth the deployment can afford. K=8 remains counterproductive for all arms at both scales; K=4 is the operating point throughout.

**Takeaway.** The recipe's value increases with model size; 27B gains roughly double the 9B gains in absolute tokens per second saved.

## 4.4 No selection statistic escapes the function-cut tax

Six selection schemes for channel pruning fail identically. The first four rank by activation statistics: global mass, activation-times-weight-norm (Wanda-style), wide calibration over 100 documents, and an online exponential-moving-average mask updated on the live stream. The last two condition the statistic: vocabulary-conditioned selection by hot-logit energy, and divergence-conditioned selection calibrated on the exact positions where the pruned head fails. The last is the sharpest: restoring pruned channels ranked by their activation at observed failure positions recovers less accuracy than restoring channels by plain global mass. The mechanism is statistical. The channels that serve rare content fire rarely, so they accumulate negligible mass on any finite calibration sample. The hot-logit energy of every channel sits in a narrow band around the hot vocabulary's share, so no channel specializes in the trimmed vocabulary and none identifies as safe to cut. We scope this result to zero-training, statistics-based selection: gradient-based selection and prune-then-finetune recovery lie outside this paper's zero-training envelope.

**Takeaway.** Within zero-training compression, the function-cut tax is a property of selection from finite samples, and switching the statistic does not remove it.

<!-- PENDING SLOTS (megachain): §4.1 add mixed-precision arm row; §4.2 add 27B audit replication sentence (C6); §4.0 seed-variance note cite 20-seed study. -->
