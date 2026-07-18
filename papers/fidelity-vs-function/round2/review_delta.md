Good - got prior art on vocab trimming, self-spec, quantized drafting. Now produce review.

---

```
SEED: 7a3c9d2f14b8e6501f729a84cd35b0e2
```

```
AXIS: Cherry-picked qualitative examples
STANCE: Sharp but balanced
```

---

# Summary

Paper claims co-trained draft heads compress into two families with order-of-magnitude difference in robustness: "fidelity cuts" (quantization) cost ≤1.9 acceptance points under domain shift while "function cuts" (pruning/low-rank/distillation) cost 10–19 points. Authors offer zero-training recipe (vocab-trim + NVFP4 quantize → 212MB artifact, 1.8–2.7× speedup), audit protocol (disagreement rate + harm-per-disagreement predicts shift tax at Spearman 1.0), and mechanistic story (finite calibration can't see rare-firing channels). Major concerns center on: (1) impossibly clean ranking result (exact p=0.0002, Spearman 1.0) from 7 arms on 24-document suite; (2) single-model-family evidence (Qwen lineage only); (3) missing reproducibility details despite author emphasis on reproducibility; (4) cherry-picking risk in suite construction (domain-shift corpus designed *after* measuring initial failures). Core contribution (trim+quantize beats selective cuts) likely holds — trim is structural/safe by construction, quantization robustness is known — but strength-of-evidence claims (Spearman 1.0, taxonomy "independent of architecture") overreach the data.

---

# Major concerns

**1. [Evidence — statistical rigor] Spearman 1.0 ranking is suspiciously clean for a 7-arm × 24-document measurement**

*Where:* §5.3, "Spearman 1.0, exact p = 0.0002" for in-distribution disagreement predicting domain-shift tax over 7 arms.

*Why it matters:* Exact-rank agreement across 7 methods on 24 documents (German/Chinese/Hebrew/Russian Wiki + PubMed + C4) screams either (a) cherry-picked suite, (b) missing confounds (all function cuts share implementation detail that correlates with shift), or (c) real-but-fragile result that won't survive method variants. The 27B replication "reproduces exact ranking" but uses *same suite extracted through different trunk* — not independent validation. Authors correctly note "wider suites would tighten [confidence intervals]" in §8.2 but then claim the ranking "transfers across model scale unchanged" as if 2-model, 1-suite replication generalizes. A result this clean on this few documents, with explicit knowledge that single-prompt sampled cells "swing 12 acceptance points across seeds," should trigger skepticism. Even the best-designed suites have noise; perfect rank concordance suggests the suite was tuned until it matched the hypothesis.

*What would address it:* (1) Third-party suite with different languages/domains, measured on both models, with reported confidence intervals on the Spearman itself (bootstrap or permutation); (2) ablate suite composition — does removing any single domain break the 1.0? If yes, you found the lynchpin document. (3) Report raw arm × document tax matrix so readers can see where variance sits.

---

**2. [Novelty — overclaimed generality] "Independent of architecture" claim rests on one model family**

*Where:* §1 "mechanism is statistical, independent of architecture"; §4.2 "family gap survives the model change"; §6.1 mechanism section treats result as universal.

*Why it matters:* Evidence is Qwen3.5-9B + Qwen3.6-27B (same lineage, same co-trained MTP head design, same training recipe likely) plus one unnamed "independently built third configuration" (§4.1) that "reproduced the pattern" but gets no detail. The mechanism (rare channels accumulate no calibration mass) is architecture-agnostic *in principle*, but whether it dominates in practice depends on superposition degree, head capacity, vocabulary distribution — all of which vary across model families. The "third configuration" hand-wave does not substitute for measurement on a non-Qwen model. Authors scope to "one model family" in §8.2 but then spend three pages (§6) explaining why the mechanism is universal. Pick one: either this is a Qwen-family finding pending replication, or measure Llama/Mistral/DeepSeek and own the generality.

*What would address it:* Measure the recipe + one function cut on one non-Qwen model with co-trained head (DeepSeek-V3 ships MTP head). If family gap survives, claim stands. If not, rewrite §6 as "Qwen-specific observation + candidate mechanism" instead of settled law.

---

**3. [Evidence — distinguishing test] Domain-shift suite construction timeline creates cherry-pick risk**

*Where:* §4.2 domain-shift suite (German/Chinese/Hebrew/Russian Wiki + PubMed + C4); §4.0 "aggregate protocol is itself a finding."

*Why it matters:* Authors discovered that single-prompt sampled cells swing 57 points (20-seed span on one long prompt, sd=13) *during* this project, then defined the 24-document suite *after* that discovery. Paper does not state whether suite was locked before measuring all arms or iteratively refined. The "retracted headline" (§4.0, a 2.68× sampled cell) shows authors found and killed one cherry-pick, which raises prior that suite itself was tuned. The effect (10–19pt function-cut tax) is large enough it probably survives suite choices, but the *exact ranking* (Spearman 1.0) could easily be artifact of which 24 documents made the cut. Authors correctly fixed the prompts-are-noisy problem with aggregation; they did not address the suite-selection problem.

*What would address it:* (1) State explicitly: was suite composition locked before running all arms? (2) If not pre-registered, measure on second suite drawn by different sampling rule (e.g., random 24 from each domain instead of however these 24 were chosen). (3) If Spearman drops to 0.7–0.9, the result is still useful but the "exact ranking" claim dies.

---

**4. [Reproducibility] Recipe in §3.4 says "one command exports the artifact" but command is absent**

*Where:* §3.4 "One command exports the artifact from a HuggingFace checkpoint: rank the vocabulary on the model's own output, gather the hot rows, quantize head and block to NVFP4, and emit a standalone draft file with the index map."

*Why it matters:* Authors frame reproducibility as priority ("measurement protocol that makes sampled-decoding comparisons reproducible" in abstract, §4.0 full protocol, §8.2 limitations enumerate what's missing) yet omit the literal command. "Rank the vocabulary on the model's own output" — how many tokens? which sampling settings? what if model hasn't generated anything yet? "Gather the hot rows" — does tie-breaking matter? "Quantize […] to NVFP4" — which library/kernel? NVFP4 is not a public standard (it's bw24-engine-internal per §4.0). The HF checkpoint for a Qwen model does not contain the draft head in separate-artifact form; the head is part of model weights. A competent grad student cannot replicate "the recipe" from §3.4 alone.

*What would address it:* (1) Provide the export script or exact command sequence in appendix. (2) If NVFP4 tooling is proprietary, state clearly "recipe requires bw24 engine" instead of framing as general method. (3) Open-source the 212MB artifact itself so practitioners can verify against it.

---

**5. [Evidence — baselines] Pruning/low-rank arms get one budget each; no sensitivity to hyperparameters**

*Where:* §4.2 "channel pruning at two budgets" (25% keep, 50% keep); "low-rank factorization at a byte-matched rank in data-free and calibrated variants."

*Why it matters:* The taxonomy hinges on function cuts paying 10–19 points. Authors test 2 prune budgets, 1 rank, and 1 distilled head — total 4 function-cut configurations against 3 fidelity-cut configurations. Pruning and low-rank have known sensitivity to: calibration set size (authors vary 12–100 docs in §6.1 but don't report whether tax changes), ranking metric (authors test 6 in §4.4 but bury them in one paragraph), and budget (2 points is not a sweep). The byte-matched rank for SVD is unspecified — at 9B scale with 248k vocab, what rank matched 212MB? If it's rank 64 and the head is 4096-dim, that's catastrophic compression; if it's rank 2048, it's mild. "Distilled replacement" gets one sentence (§3.3) with no architecture or training detail. You can't claim "function cuts" as a category if you tested 1 rank, 2 budgets, and 1 mystery distillation.

*What would address it:* (1) Sweep low-rank budget (say 3 ranks from 256 to 2048) and report tax vs bytes. (2) Report the actual rank used. (3) For distillation, state: student architecture, dataset size, training steps, final loss. (4) If tax is monotonic in budget and insensitive to statistic (which §4.4 claims), state that and show the monotonicity plot.

---

**6. [Evidence — alternative explanation] Trim's shift-invariance could be coverage-ceiling artifact, not mechanism**

*Where:* §4.2 "trim's ceiling is hot coverage: 95 to 99%"; §3.1 "structural reason trim carries no distribution-shift tax"; §6.2 "trim orphans no computation."

*Why it matters:* Authors argue trim is safe because it's "structural" (removes outputs, keeps all channels). Alternative explanation: trim costs 1–5% acceptance on *every* domain (5–1% = cold tokens that would have been correct), and the "shift-invariance" is just the fact that 5% is 5% everywhere. The 95–99% coverage number *is itself measured on the deployment distribution* (§4.0 "hot-vocabulary coverage on generated text"). If authors measure trim on a different language where cold coverage is 80%, does tax jump to 20%? The superposition measurement (§6.2, all channels spread energy uniformly over vocab) is interesting but proves wrong thing: it shows trim creates no *pruning* headroom; it does not prove trim's shift tax is zero rather than coverage-capped. Authors even note in §8.2 "A deployment dominated by content outside its hot set needs its vocabulary re-ranked" — which admits trim's safety is distribution-conditional.

*What would address it:* (1) Measure trim on a language/domain where hot-32k coverage is <90% (e.g., code-switch text, rare language). If tax jumps to 10%, the mechanism is coverage, not structure. (2) Separate "trim costs nothing beyond coverage ceiling" (probably true) from "trim is shift-proof" (only true if coverage is shift-invariant, which is empirical). (3) Rewrite §3.1 to say "trim's tax is its coverage gap, which is small on our suites" instead of "trim carries no tax."

---

**7. [Evidence — missing ablation] Mixed-codec result (§4.1, Q4_0 block + NVFP4 proj = 2.09×) is the actual winner but appears in one sentence**

*Where:* §4.1 last paragraph, "A mixed export recovers it. Pairing the NVFP4 projection with a Q4_0 block, at a byte-identical 212MB, keeps 58.3% acceptance […] and its 2.09× takes that suite outright."

*Why it matters:* This is the *best result in the paper* — beats all-NVFP4 by 3.6 acceptance points and 0.11× speed — yet it gets one paragraph, no table entry, no cross-suite measurement, no Figure 1 callout. The recipe in §3.4 says "quantize head and block to NVFP4"; the playbook in §8.1 says "Trim + NVFP4." The 2.09× mixed config is not "Trim + NVFP4"; it's "Trim + NVFP4 proj + Q4_0 block." If this is deployment-ready (which the claim "takes that suite outright" implies), why isn't it the recipe? If it's not deployment-ready (maybe Q4_0 isn't always available?), why is it in the results? Authors bury their own best number.

*What would address it:* (1) Measure mixed config on all suites (short/long, 9B/27B, greedy/sampled). (2) Either promote it to the recipe or explain why it's not general (e.g., "Q4_0 support varies by engine"). (3) Add it to Table 1.

---

# Minor concerns

- **Figure 2** (cost breakdown): shows "1.06GB projection, 0.26GB block" but these are 8-bit numbers; 4-bit would be ~0.53 + 0.13 = 0.66GB, yet recipe delivers 0.21GB per-token reads. Where's the gap? Likely vocabulary trim (248k → 32k = 7.75× compression on projection alone), but Figure 2 + text never connects the dots explicitly.

- **§4.0 "separately built third configuration"**: no detail. Which model? Which suite? What was "independently built" — the quantization scheme, the trim, the engine? One sentence of context (e.g., "Llama3-8B head, same trim recipe, Q4_K instead of NVFP4") would make this citation credible instead of hand-wave.

- **§5.4 "rarity-concentration statistic failed"**: great that authors report the failed instrument, but where's the actual number? "Quantized arms also disagree disproportionately on rare tokens, harmlessly" — by what factor? Without the false-positive rate, readers can't tell if the statistic was *slightly* worse or *completely* uninformative.

- **Table 1**: 9B long greedy, trim-Q8 beats trim-NVFP4 despite 4-bit being "fastest or tied-fastest" (§4.1 takeaway). Takeaway says "within the family, block codec picks the winner" — but codec isn't a family, it's a format choice. Rephrase or add qualifier ("fastest on 5 of 6 cells").

- **§6.1 "widening the sample does not help at any width we tested (12 to 100 documents)"**: 12 to 100 is one order of magnitude. Calibration-data scaling laws suggest the tail might need 10k documents. Did authors try it, or stop at 100 because mechanism was already clear? If stopped early, say so.

- **§7 Related Work**: FR-Spec [missing arXiv ID] "trims draft vocabulary at load time and reports 1 to 3% end-to-end gain in llama.cpp." Authors should have arXiv-searched "FR-Spec speculative decoding vocabulary" to get the ID or confirm it's unpublished. [2603.05210] (Balancing Coverage and Draft Latency in Vocabulary Trimming, Ben Shoham 2026) directly addresses trim's coverage-vs-latency tradeoff and is conspicuously uncited despite being the *only prior arXiv paper on draft vocab trimming*. Authors claim §3.1 as novel ("structural reason trim carries no tax") but [2603.05210] likely scooped it.

- **§8.1 Playbook, row 2**: "Memory pressure demands more than 4-bit trim delivers" → "Stop. Pruning and low-rank buy at most 90MB more and pay 10 to 19 acceptance points." The 90MB is unsubstantiated (no calculation or cite). If it's 212MB artifact minus pruned-head size, show the arithmetic. If it's from unpublished measurements, remove or caveat.

- **Appendix is absent.** Authors promise "negative results appear in full" (abstract, §1) but six failed selection schemes (§4.4) get one paragraph total, and the divergence-conditioned scheme (coolest failure) gets two sentences. Where's the table? Where are the per-scheme tax numbers?

---

# Belief update

**What I updated on:**

1. **Trim + quantization is probably safe and sufficient for Qwen-family co-trained draft heads**, and the recipe (modulo reproducibility gaps) is a real contribution. The mechanism (trim removes outputs not compute, quantization perturbs uniformly) is sound. The 1.8–2.7× speedups at 212MB are deployment-relevant.

2. **Harm rate (§5.2) is an elegant instrument.** Fidelity cuts disagree by tie-flip (harm ~0.14), function cuts disagree by losing function (harm ~0.5). That's a clean separation, and the instrument only needs forward passes. I'm 70% confident it transfers to other compression settings (e.g., base-model quantization).

3. **Single-prompt sampled benchmarking is broken** (57-point spread, sd=13, §4.0). This is valuable negative result. I already believed sampled LM evals are noisy, but 57 points from seed alone is worse than I thought. Authors' fix (suite-level aggregation, 3 seeds) is correct.

**What I did not update on:**

1. **"Function cuts cost 10–19 points under shift, fidelity cuts cost ≤1.9" as a *universal law***. I believe it for this model family on this suite. I do not believe "the family gap survives the model change" (§4.2) generalizes from 9B→27B Qwen to, say, Qwen→Llama, because gap size depends on superposition degree and head capacity. Authors provide candidate mechanism (§6) but zero evidence the mechanism *quantitatively* predicts the 10–19pt magnitude.

2. **Spearman 1.0 ranking** (§5.3). This is either measurement on a cherry-picked suite or a result that won't survive method variants. The p=0.0002 is exact-test; it says "assuming suite is fixed, arms are distinguishable," not "this ranking is universal." I need third-party replication on different suite.

3. **"Trim is shift-proof"** (§3.1, §4.2). I believe trim's tax equals its coverage gap, which *happens* to be small (1–5%) on the suites authors tested. I do not believe the tax is structurally zero — coverage is distribution-dependent, and authors admit it (§8.2).

**Bottom line:** Paper makes *one* strong claim I believe (trim+quantize compresses Qwen draft heads effectively) and *one* weaker claim I'm 50/50 on (function cuts are systematically less robust than fidelity cuts, with mechanistic explanation). The evidence for the strong claim is solid modulo reproducibility gaps. The evidence for the weak claim overshoots: Spearman 1.0 is too clean, single-family measurement + architecture-independent framing is mismatch, and "shift-proof trim" conflates coverage empirics with structural guarantee.

---

# Verdict

**Major revision**. The core contribution (zero-training recipe for Qwen draft heads: trim + NVFP4, 212MB, 1.8–2.7×) is novel and useful. The fidelity/function taxonomy is interesting and *probably* right, but strength-of-evidence claims (Spearman 1.0 from 7 arms on 24 documents; "independent of architecture" from 1 model family; "shift-proof trim" when coverage is distribution-conditional) overreach the data. Major concerns #1 (implausibly clean ranking), #2 (overclaimed generality), #3 (suite cherry-pick risk), and #4 (missing reproducibility details despite reproducibility emphasis) must be addressed before acceptance. The taxonomy likely survives those fixes — the mechanism is sound and the effect size is large — but the current draft overclaims certainty from narrow evidence. Fix: third-party suite replication (breaks or confirms Spearman 1.0), one non-Qwen model measurement (scopes or validates generality), and release the export script (makes recipe actually reproducible). With those, this is a strong Accept.
