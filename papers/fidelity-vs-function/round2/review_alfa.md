PIE not found — need to check if it's non-arXiv. Got enough context. Now write review.

---

```
SEED: 7a4e2f8b93c15d60e1279f4a5b8c6d03
```

**Primary lens** = `int(7a, 16) mod 6` = `122 mod 6` = **4** → Related work and novelty attribution  
**Stance intensity** = `int(4e, 16) mod 3` = `78 mod 3` = **0** → Skeptical-but-fair

```
LENS: Related work and novelty attribution
STANCE: Skeptical-but-fair
```

---

# Summary

Paper evaluates compression methods for co-trained draft heads in speculative decoding (Qwen/DeepSeek lineage). Authors measure acceptance retention under domain shift, find uniform quantization (fidelity cuts) costs ≤1.9 points while pruning/low-rank/distillation (function cuts) costs 10-19 points on shifted corpora. Propose vocabulary trimming + NVFP4 quantization (212MB artifact, 1.8-2.7x speedup, no training). Pre-deployment audit uses disagreement rate + harm-per-disagreement to classify cut type. Validates taxonomy on 9B/27B models with 80/20 prompt suites, 3 seeds per cell.

---

# Major concerns

### 1. Missing prior art: Hooker et al. "Pruning Identified Exemplars" uncitable

**Issue:** §7 cites "Hooker et al." as establishing pruned classifiers sacrifice underrepresented examples first, but provides no year, venue, or paper title.

**Where:** §7 Related Work, paragraph 2, first sentence.

**Why it matters:** This is the paper's **primary external precedent** for the tail-damage claim. Without a retrievable citation (arXiv search for "pruning identified exemplars" returns zero matches; the phrase appears nowhere in indexed literature), reviewers and readers cannot verify the claim, assess novelty overlap, or distinguish this work's contribution from prior art. If PIE established statistical invisibility of rare examples under pruning, the mechanism in §6 is incremental; if PIE only measured task-level harm without mechanism, §6's contribution is larger.

**What would address it:** Provide full citation (authors, year, title, venue/preprint ID) or retract the claim. If PIE is unpublished/inaccessible, state that explicitly and cite next-closest verifiable work.

---

### 2. Related work omits QSpec and hierarchical-framework study mechanisms

**Issue:** §7 cites QSpec [2410.11305] and [2505.22179] for "4-bit drafting preserves acceptance" and "combination fragile in memory dimension," but neither paper's arXiv record mentions pruning, low-rank, or distribution shift — the taxonomy's core. QSpec abstracts report "complementary quantization schemes" (high-precision verify, low-precision draft); [2505.22179] studies "memory-bound inference" compatibility. Paper does not explain what those works measured, how their findings relate to fidelity-vs-function distinction, or why they didn't discover the taxonomy.

**Where:** §7, paragraph 1, sentences 3-5.

**Why it matters:** If QSpec already separated quantization from structural cuts on distribution-shifted corpora, the taxonomy is not novel. If those papers evaluated only in-distribution or never tested pruning/low-rank, that absence is the contribution's boundary and must be stated. Currently readers cannot tell whether authors replicated, extended, or independently discovered.

**What would address it:** Two sentences per cited paper: what they measured (methods × test conditions), what they found (results on shift if any), why they didn't produce this taxonomy. If they evaluated quantization only, say so. If they never tested shift, say so.

---

### 3. Novelty claim for harm-rate audit ungrounded

**Issue:** §5 intro claims harm-per-disagreement metric "has no counterpart in this line" (compression robustness literature), but provides zero citations to what prior audits measured. SCOP [2010.10732] proposes "scientific control" for pruning; recent robustness works ([2206.03669], neural-symbolic verification under distribution shift) exist. Paper does not compare harm-rate to those methods' metrics or explain why theirs didn't classify cut types.

**Where:** §7 paragraph 2, final sentence; entire §5 lacks comparison to prior audit/pre-deployment methods.

**Why it matters:** If prior work used disagreement-only metrics and failed to separate quantization from pruning, harm-rate is genuinely novel. If prior work already measured accuracy-loss-per-disagreement under different names, the contribution is application to draft heads, not the metric. Unsupported novelty claims undermine credibility.

**What would address it:** Survey prior pre-deployment compression audits (SCOP, certified robustness methods, pruning diagnostics). State explicitly: "Prior audits measured X; ours differs by Y." If no prior work exists for draft-head compression audit, state that scoping ("first audit for draft heads") rather than broad claim.

---

### 4. LLM-KICK [2310.01382] cited but not retrieved

**Issue:** §7 cites LLM-KICK as establishing "broad capability loss from pruning that perplexity hides," but arXiv search for "2310.01382" returns zero results. Neither does "LLM-KICK" keyword search. Citation format matches arXiv convention, but paper is unverifiable.

**Where:** §7, paragraph 2, sentence 2.

**Why it matters:** If LLM-KICK already showed pruning damages tail tasks invisible to aggregate metrics, this paper's novelty narrows to (1) token-level measurement, (2) draft-head application, (3) predictive audit. If LLM-KICK doesn't exist or measures something else, the citation is fabricated or mis-numbered. Either outcome is a Major concern.

**What would address it:** Verify arXiv ID or retract. If preprint under different ID/venue, correct it. If paper is not public, state "(unpublished)" and provide authors/year.

---

### 5. Contribution list omits what trim *removes*

**Issue:** Trim deletes 215k vocabulary rows (87% of output projection) but §3.1 claims "trim carries no distribution-shift tax" and "deletes no computation." These statements contradict: if trim removes 1.06GB → 0.21GB of projection weights, it removed parameters serving cold tokens. Paper does not explain why losing those parameters costs zero on German/Chinese/Hebrew/Russian/PubMed (Table in §4.2 shows trim retention 97.5-99.4%, indistinguishable from Q8's 99.8-99.9%).

**Where:** §3.1, sentences 3-5; §4.2 results.

**Why it matters:** If trim removes 87% of parameters and loses <2% acceptance on severe domain shift, one of three things is true: (1) cold vocabulary genuinely never fires on shifted corpora (implausible — German/Russian use different Unicode ranges), (2) verification's full-vocab oracle recovers trimmed-token proposals as rejects (correct mechanism but paper doesn't state it), or (3) measurement error. Without explanation, the zero-tax claim for an 87% parameter cut is not credible.

**What would address it:** Measure trim's recall on shift suite: what fraction of reference-accepted tokens lie outside hot-32k? If verification rejects trimmed-token drafts harmlessly (because reference uses full vocab), state that mechanism explicitly in §3.1. If shift corpora genuinely stay within hot-32k, show the vocabulary overlap table.

---

### 6. Six-scheme study (§4.4) reports failure but no false-discovery correction

**Issue:** Paper tests six pruning selection schemes, finds all fail identically, claims "the liability follows from selection itself" (§4.4 conclusion). But with six schemes tested on same data, multiple-comparison probability of at least one false negative (failing to detect a real difference) is 1 - 0.95^6 = 26%. Paper reports no Bonferroni, Holm, or FDR correction, no effect-size thresholds, no Bayesian analysis.

**Where:** §4.4, all results; no correction mentioned anywhere.

**Why it matters:** The null result (no scheme escapes the tax) is the paper's **central mechanistic claim** (§6 builds on it). If uncorrected, the probability that at least one scheme (e.g., divergence-conditioned) actually works but failed due to noise is >25%. A false negative here would falsely indict selection statistics and incorrectly validate fidelity-only compression.

**What would address it:** Apply family-wise error correction or report effect sizes + confidence intervals for each scheme's tax. If all six taxes exclude zero at Bonferroni-corrected alpha, the null result stands. If any CI includes zero post-correction, weaken claim to "no tested scheme reliably escapes."

---

# Minor concerns

- **§2.1:** "1.06GB of 8-bit weights read per drafted token" — but Qwen3.5 vocab is 248,320 × 4096 × 1 byte = 1.016GB, not 1.06GB. Check arithmetic.
- **§4.0:** "standard deviation 13" on 20-seed single-prompt cell — was this one prompt or average across suite? If one prompt, that's anecdotal. If suite, should appear in a table.
- **§4.2:** "permutation-test floor for seven arms (p = 0.029)" — what null hypothesis? Permuting arms' labels? Needs one clarifying sentence.
- **§6.3:** "Teacher-forced chain measurements refuted it" — no table/figure showing chain acceptance by step position. Add table or remove "measurements" (unshown measurements don't count).
- **Figure 2:** No Figure 2 appears in manuscript text; §2.1 references it.
- **Table 1:** Column headers unclear — is "acceptance" = accepted/drafted, or accepted/generated? Define once in caption.
- **Typo §1:** "calibration-register text" should be "calibration-distribution text" for clarity (register implies linguistic register; you mean distribution).

---

# Verdict

**Major revision**

Five citations (Hooker PIE, QSpec/[2505.22179] mechanisms, LLM-KICK, prior audit baselines) are unverifiable, mis-stated, or omitted, violating research integrity standards for a venue claiming novelty over prior work. Trim's zero-tax claim lacks mechanism explanation (full-vocab verification likely the answer, but paper doesn't say it). Six-scheme null result drives central claim but carries no multiple-comparison correction. All six issues are fixable: add citations, verify arXiv IDs, measure trim's recall on shift corpora, explain verification recovery, apply Bonferroni/effect-size analysis to §4.4. Experimental design (2x2 cut types, 3-corpus shift distance, forward-pass audit, full negative-result disclosure) is excellent; execution on novelty attribution and statistical rigor needs one revision round.
