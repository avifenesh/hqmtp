Good. Got Medusa [2401.10774], EAGLE-2 [2406.16858], Hydra [2402.05109], QSpec [2410.11305], and hierarchical framework [2505.22179]. Now generating seed and writing review.

---

```
SEED: 7f3c91a8e4d2b607f1c5938a74e0d26f

LENS: Related work and novelty attribution (int(7f, 16) mod 6 = 127 mod 6 = 4)
STANCE: Adversarial (int(3c, 16) mod 3 = 60 mod 3 = 0 → wait, recalc: 60 mod 3 = 0)
```

Wait, mapping error. Stance 0 = skeptical, stance 1 = adversarial, stance 2 = steelman. Recompute:

```
int("3c", 16) = 60
60 mod 3 = 0
```

Stance = 0 = Skeptical-but-fair.

Actually reviewing now.

---

# Review for "Fidelity Cuts vs Function Cuts: What Compressing Speculative-Decoding Draft Heads Destroys, and How to Detect It Before Deployment"

**Reviewer: echo**

```
SEED: 7f3c91a8e4d2b607f1c5938a74e0d26f
LENS: Related work and novelty attribution
STANCE: Skeptical-but-fair
```

---

## Summary

Paper compresses co-trained multi-token-prediction draft heads on Qwen3.5-9B/Qwen3.6-27B via vocabulary trimming, quantization (fidelity cuts), channel pruning, low-rank factorization, distillation (function cuts). Claims fidelity cuts cost ≤1.9 acceptance-retention points under domain shift while function cuts cost 10-19 points. Proposes recipe: trim to 32k hot tokens + NVFP4 quantization → 212MB artifact → 1.8-2.7x speedup. Audit protocol: in-distribution disagreement rate + harm-per-disagreement classify family, predict off-distribution tax (Spearman 1.0, p=0.0002).

---

## Major concerns

### 1. Missing prior art: QSpec, hierarchical framework, FR-Spec, SlimSpec

**Where:** §7 Related Work, introduction

**Why it matters:** Paper positions itself as first rigorous compression study of co-trained draft heads, but:
- **QSpec [2410.11305]** already established 4-bit drafting preserves acceptance via complementary quantization schemes (low-precision joint quant for drafting, high-precision for verify). Paper cites this but claims "neither analyzes which compressions are safe" — QSpec's entire contribution *is* analyzing which quantization scheme is safe for drafting.
- **Hierarchical framework [2505.22179]** evaluated speculative decoding on quantized targets and found fragility in memory dimension — directly relevant to paper's claim that 4-bit compression is safe.
- **FR-Spec** (cited but dismissed): paper reports 1.8-2.7x gains from trim alone and claims FR-Spec "reports 1-3% end-to-end gain" in llama.cpp. This comparison is apples-to-oranges: FR-Spec runs in llama.cpp (CPU inference), this paper runs bw24 (production CUDA engine). Engine architecture determines whether bandwidth-bound regime holds. Dismissing FR-Spec's modest gain without acknowledging the engine difference understates the contribution's dependency on deployment context.
- **SlimSpec** mentioned only as "low-rank drafter head" for models without co-trained heads. If SlimSpec uses low-rank factorization on draft heads, its robustness profile directly validates or refutes this paper's function-cut taxonomy. No engagement with whether SlimSpec reports domain-shift behavior.

**What would address it:** 
- Table comparing this work to QSpec/hierarchical-framework on: compression method, domain-shift evaluation (yes/no), artifact size, acceptance retention under shift.
- Explicit statement whether FR-Spec's trim method differs from this paper's (e.g., ranking statistic, vocabulary size), and whether the gain difference is engine architecture or method.
- Locate SlimSpec's robustness claims (if any) and position function-cut taxonomy relative to them.

### 2. Medusa/EAGLE/Hydra co-trained head prior art unacknowledged

**Where:** §1, §2.1, §7

**Why it matters:** Paper frames co-trained multi-token-prediction heads as Qwen/DeepSeek-lineage innovation. But:
- **Medusa [2401.10774]** introduced lightweight draft heads operating on base model hidden states in 2024.
- **Hydra [2402.05109]** extended Medusa to sequentially-dependent draft heads.
- **EAGLE-2 [2406.16858]** uses context-dependent draft trees, also operating on base hidden states.

All three are co-trained heads that draft from trunk state. Qwen/DeepSeek may ship them as first-class artifacts, but the architecture is not novel to those lineages. Paper must clarify: is novelty in (a) compressing pre-shipped heads vs. training new ones, (b) vocabulary trimming as compression axis, (c) fidelity/function taxonomy, or (d) audit protocol? As written, framing implies Qwen/DeepSeek invented co-trained draft heads, which is false.

**What would address it:** Rewrite §2.1 intro acknowledging Medusa/EAGLE/Hydra as prior co-trained head architectures. State explicitly: "Qwen/DeepSeek ship these as frozen artifacts; we compress them post-training. Medusa/EAGLE require training; our recipe does not."

### 3. "Distribution-exact by construction" claim for sampled decoding needs justification

**Where:** §3.4, §4.0, Abstract

**Why it matters:** Paper claims trim keeps "sampled output distribution-exact by construction" because verification runs full-vocabulary. But:
- Draft proposes only hot tokens.
- If true next token is cold, draft never proposes it → verify rejects all drafts → target samples once from full distribution.
- This is distribution-exact in expectation over many rounds, but single-trajectory samples may systematically avoid rare tokens because the draft never proposes them, reducing the sampler's effective exploration.

Is "distribution-exact" proven formally, or is it "exact in expectation"? If the latter, claim overstates. Clarify whether "by construction" means "mathematically proven identical token-distribution over infinite samples" or "correct in expectation per round." If former, where's the proof?

**What would address it:** Appendix proof that trim + full-vocabulary verify yields identical token distribution to plain sampling, or downgrade claim to "distribution-correct in expectation per verify round; single-trajectory deviations are sampler-noise, not compression artifact."

### 4. Domain-shift suite is 24 documents; "severe shift" claim lacks calibration

**Where:** §4.2, §8.2 limitations

**Why it matters:** "Severe domain shift" is supported by reference head accuracy dropping 0.88 → 0.36 (2.4x drop). But:
- Suite is German/Chinese/Hebrew/Russian Wikipedia + PubMed + C4 web, 24 documents total.
- 24 documents cannot represent the tail: if mechanism is "rare content underrepresented in calibration," then 24-document eval underrepresents that tail too.
- "Severe" is relative: accuracy 0.36 is still better than random (1/248320). Is 0.36 typical of actual production shift, or is this suite artificially hard?
- Confidence intervals in §4.2 mention "the taxes it measures carry the confidence intervals of §4.2" — but those CIs are not reported in the text. Table 1 has no error bars.

**What would address it:** 
- Report per-document acceptance variance on domain-shift suite to show whether 24-sample uncertainty overlaps function-cut CIs.
- Compare 0.36 accuracy to published speculative-decoding acceptance on genuinely out-of-domain tasks (code → prose, English → multilingual). Is 2.4x drop typical or extreme?
- Move exact CIs into Table or Appendix; "carries confidence intervals" without showing them is not verifiable.

### 5. Six selection schemes "fail identically" — but magnitudes differ

**Where:** §4.4, §6.1

**Why it matters:** "Fail identically" implies equal magnitude. But text says schemes pay 10.2 to 18.7 retention points (§4.2 lists pruning 50% = 10.2pts, pruning 25% = 16.8pts, low-rank = 16.8-18.7pts). That's 1.8x range. "Identically" is wrong if magnitudes differ.

**What would address it:** Either:
- Table reporting each scheme's domain-shift tax with CIs, showing overlapping intervals → "statistically indistinguishable" justified.
- Or replace "identically" with "comparably" and report range explicitly.

### 6. Audit's Spearman 1.0 ranking: 7 arms, exact p=0.0002 — but permutation test has floor p=1/7!

**Where:** §5.3

**Why it matters:** Exact permutation test on ranking 7 items has minimum achievable p = 1/7! = 1/5040 ≈ 0.0002. Observing p = 0.0002 means Spearman correlation is perfect (1.0) and no other permutation achieves equal or better correlation. This is correct, but presenting it as "exact p = 0.0002" without explaining it's the floor may read as lucky p-hacking. Did you run this test once (as preregistered by protocol), or did you test multiple audit statistics and report the one that worked?

**What would address it:** Footnote: "Exact p = 1/7! = 0.0002 is the permutation-test floor for perfect rank correlation; observed Spearman = 1.0 achieves this minimum." State explicitly: "Harm rate was the first audit statistic we tested; rarity-concentration (§5.4) was a failed second attempt."

### 7. "Zero-training" scope excludes gradient-based selection; conclusion overgeneralizes

**Where:** §4.4, §6.1, §8.2

**Why it matters:** Concluding mechanism is "selection from finite samples is impossible" when scope explicitly excludes prune-then-finetune is a category error. Finetune *trains on* the pruned head's failures, which is gradient-based evidence unavailable to zero-training statistics. Paper's impossibility claim is scoped to zero-training but reads as universal.

**What would address it:** Every statement "selection fails" must be scoped: "zero-training, statistics-based selection fails; gradient-based selection and prune-then-finetune recovery remain open questions (§8.2)."

### 8. "Mixed arm" (NVFP4 projection + Q4_0 block) introduced mid-results without protocol

**Where:** §4.1, "The 4-bit cost sits in the block, and a mixed export recovers it."

**Why it matters:** Protocol in §4.0 defines arms: reference, trim, trim-Q8, trim-NVFP4, pruning, low-rank, distill. "Mixed arm" (NVFP4 proj + Q4_0 block) appears only in §4.1 discussion, not in evaluation grid. Was this:
- Post-hoc based on §4.1 findings (bad: looks like p-hacking)
- Planned but unreported in protocol (bad: protocol incomplete)
- A separate experiment presented as main result (bad: claims "2.09x takes that suite outright" when it's not in Table 1)

**What would address it:** Either:
- Label mixed arm as post-hoc ablation: "After observing block sensitivity, we tested a mixed configuration; this is an exploratory result, not part of the preregistered protocol."
- Or add mixed arm to Table 1 with full grid.

---

## Minor concerns

- **Figure 2 referenced but not present** (§2.1 "Figure 2" — no figures in text).
- **"Natural head" terminology inconsistent:** sometimes "unmodified head," sometimes "reference," sometimes "natural head" (§3.4). Pick one.
- **§4.0 "three sampling seeds"** but §4.0 also says "twenty seeds on one long prompt span 57 points" — were both 3-seed and 20-seed experiments run? Clarify which cells use which.
- **"bf16 base" vs "full-precision base":** bf16 is not full precision (it's 16-bit). Say "bf16" or "unquantized" consistently.
- **"Carrier compounding" falsified (§6.3):** good that failed mechanism appears, but "carrier compounding" is jargon never defined before §6.3. Define on first use.
- **§8.1 playbook table:** "Run the audit" row says "Spearman 1.0 tax ranking on our hardest suite" — this is a claim about this paper's data, not a promise about user's data. Rewrite: "Minutes of forward passes; on held-out domain-shift data, audit disagreement ranked compression tax at Spearman 1.0 in our study."

---

## Verdict

**Major revision**

Paper makes three contributions: (1) zero-training recipe (trim + NVFP4), (2) fidelity/function taxonomy, (3) harm-rate audit. All three have merit, but related-work positioning overstates novelty (Medusa/EAGLE/QSpec prior art), domain-shift evidence rests on small sample (24 docs), and "selection fails identically" conflates qualitative mechanism with quantitative variation. Audit's Spearman 1.0 result is strong but p=0.0002 on 7 items is the permutation floor, not evidence of extra robustness. Revision should: (a) reposition against Medusa/EAGLE/QSpec explicitly, (b) report per-document variance + CIs on domain-shift suite, (c) clarify "distribution-exact by construction" with proof or weaker claim, (d) scope "selection impossible" to zero-training, (e) move mixed-arm result to exploratory section or full grid.
