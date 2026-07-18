# 6 Why selection fails: the tail is statistically invisible

<!-- Topic-sentence spine:
6.0 The function-cut tax has a statistical mechanism, and it predicts the six-scheme failure.
6.1 Rare-firing channels carry the shifted-content function and accumulate no calibration mass.
6.2 No channel is vocabulary-specialized, so the trim orphans nothing that pruning could harvest.
6.3 Two theories we falsified along the way.
-->

## 6.1 Rare function, rare evidence

An activation statistic assigns each channel the evidence its firing history provides. A channel that serves common syntax fires at every position and dominates any ranking. A channel that serves rare content, an uncommon language, a technical register, an unusual construction, fires on the rare positions that need it. Over a finite calibration sample its accumulated mass is indistinguishable from a useless channel's. The selection problem is therefore ill-posed for exactly the parameters that carry shifted-content function: the calibration data that would reveal them is by definition the data the deployment has never seen. The two function-cut geometries delete differently under this one mechanism. Pruning removes whole channels, so its damage lands where those channels fired, in the rare-token bins; low-rank factorization removes small spectral directions, whose loss spreads over every position. Mass and spectrum are both statistics of typical data, and both go blind at the same place. Widening the sample does not help at any width we tested (12 to 100 documents, and an online mask over the live stream itself), because the tail scales with the sample.

The divergence-conditioned scheme sharpens the point. We restored pruned channels ranked by their activation at the exact positions where the pruned head fails, the most failure-targeted statistic available without gradients. It recovered less accuracy than restoring channels by plain global mass. Even conditioning on observed failures does not expose the tail, because at those positions the missing channels are missing: they cannot fire in the pruned network that produced the failures.

## 6.2 Superposition blocks the shortcut

Vocabulary trim suggests a shortcut: after deleting 87% of the output rows, some block machinery should serve only the deleted rows and die harmlessly. We measured it. For every FFN channel we computed the fraction of its output-projection energy landing on hot rows. The distribution is a narrow band (0.116 to 0.159 across all 12,288 channels) around the hot vocabulary's share of total energy. Every channel spreads its influence across the whole vocabulary in near-uniform proportion, the signature of features stored in superposition. Trim therefore orphans no computation, which explains both of its properties at once: it is shift-proof because it removes no function, and it creates no pruning headroom because no function became dead.

## 6.3 Falsified explanations

Two mechanisms we believed earlier failed their tests, and both failures tightened the result. First, carrier compounding: we hypothesized that pruning corrupts the hidden state each draft step passes to the next, compounding down the chain. Teacher-forced chain measurements refuted it: the pruned head holds 97% of the full head's chain acceptance on calibration-register text, in torch and in the engine. The damage is content-conditional and needs no error accumulation. Second, rarity-concentration as the audit signal (§5.4): where disagreements sit does not separate the families; what they cost does. The surviving mechanism is the statistical one above, and it makes one falsifiable prediction the data confirms: the tax must grow with shift distance, because shift raises the frequency of positions whose machinery was invisible at selection time. Retention falls monotonically from held-out (97 to 99%) through same-register (95 to 97%) to domain shift (71 to 88%) for every function cut, while every fidelity cut stays within two points across the whole range.
