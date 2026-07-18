# 5 The audit: classify a compressed draft before deploying it

<!-- Topic-sentence spine:
5.0 Deployment needs the robustness answer before live traffic arrives, and OOD test sets rarely exist for a specific deployment.
5.1 The audit runs the compressed head beside its reference on in-distribution documents and scores the disagreements.
5.2 Harm per disagreement separates the families; the disagreement rate ranks the tax.
5.3 The audit's prediction holds on genuine domain shift, where its ranking is exact.
5.4 Honest failure: our first audit statistic misclassified the quantized arms, and the reason is instructive.
-->

## 5.0 Why an audit

Section 4.2's taxes are measurable only with a domain-shift test set, which a deployment rarely has: the traffic that breaks a compressed draft is the traffic nobody collected. The practical question is whether in-distribution measurements, which any deployment can produce in minutes, predict the off-distribution behavior. They do.

## 5.1 Protocol

Run the compressed head and its uncompressed reference on the same in-distribution documents, teacher-forced over identical states, and record two numbers per arm. The **disagreement rate** is the fraction of positions where the two heads' top-1 predictions differ. The **harm rate** is the accuracy the compressed head loses per disagreement: how often a disagreement replaces a correct prediction with a wrong one instead of trading predictions of equal value. Both numbers come from forward passes only; the whole audit on 2,500 positions takes minutes on one GPU.

## 5.2 What the two numbers say

Harm rate identifies the family. Fidelity cuts disagree by tie-flipping: their rounding noise swaps tokens of near-equal probability, so a disagreement costs almost nothing (harm 0.14 across our 4-bit arms; Q8 disagrees 8 times in 2,500 positions, too few to score and too few to matter). Function cuts disagree by losing function: their disagreements land where machinery vanished, so each one costs accuracy (harm 0.28 to 0.54). Disagreement rate then ranks severity within and across families: on the same-register corpus it predicts the off-distribution tax at Spearman 0.89 (exact p = 0.006) over seven arms.

The decision rule for a deployment has two branches. Harm rate near 0.15 means the compression is noise-type, and its in-distribution cost is the full cost. Harm rate above 0.25 means the compression is function-type, and live traffic will multiply the in-distribution damage. In our measurements the multiplier under genuine shift is 6 to 10x.

## 5.3 Validation under genuine shift

On the domain-shift suite (§4.2), where the taxes are largest, the audit's in-distribution disagreement ranking orders the seven arms' taxes exactly: Spearman 1.0, exact p = 0.0002. The audit gets stronger where it matters more, because larger true taxes are easier to rank. The instrument's limits are its inputs: it compares against the uncompressed reference (which must exist), and it certifies the compression alone; model quality is out of its scope.

## 5.4 The statistic that failed

Our first audit scored the concentration of disagreements in rare-token bins, on the mechanism's logic that function cuts damage rare continuations. It failed on both families. Quantized arms also disagree disproportionately on rare tokens, harmlessly, by tie-flip, since rare-token logit margins are thin. The low-rank arms damage all bins. Rarity-concentration measures where disagreements sit; harm rate measures what they cost, and only cost separates the families. We report this because instrument design is the contribution: the obvious statistic reads plausibly and misclassifies every quantized arm.
