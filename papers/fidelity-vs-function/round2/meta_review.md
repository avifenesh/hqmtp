# Meta-review: round 2 (post-revision draft, 2026-07-18)

Five reviewers (alfa, bravo, charlie, delta, echo). All five: **Major revision**. All five endorse the same core: the zero-training recipe is real and useful, the harm-rate audit is an elegant instrument, the negative-result disclosure is exemplary. Every reviewer's blocking concerns target evidence strength and claim scope, not the method.

## Consensus concerns (3+ reviewers), ranked

**C1. Spearman 1.0 too clean for 7 arms on 24 documents** (delta, charlie, echo, bravo). The ranking claim needed a fragility test: suite ablation, CI on the correlation, and honesty that p=0.0002 is the n=7 permutation floor.
*Response:* leave-one-domain-out ablation run: rho stays 0.964 to 1.000 for every single-domain drop (Appendix B); permutation-floor phrasing added in §4.2 and §5.3; validation circularity scoped in §5.3 (the seven arms both motivated and test the instrument).

**C2. Six-scheme study reported without numbers** (charlie, echo, alfa, delta). "All fail identically" is a claim, not a table.
*Response:* Appendix A added with each scheme's decisive cell (Wanda ranking overlap 88%, chain 0.712 vs 0.712; wide-calib 41.7% vs 41.3%; online EMA 0.846 vs 0.844; vocab-conditioned 93.8% vs 97.1%; canary restoration 0.854/0.805 vs control 0.860/0.807).

**C3. Trim's "shift-proof" claim conflates coverage with structure** (delta, alfa, charlie). Coverage is distribution-conditional; the claim read as structural guarantee.
*Response:* §3.1 rewritten around the explicit damage model (tax = coverage gap; within-hot function identical). Measured per-register coverage added to §8.2: 93-95% English shift, 62% German, 21% Russian, 9% Hebrew, 5% Chinese. Retention's hot-conditioning stated in §4.2. This is the largest honest-scoping change of the round.

**C4. Architecture-independence overclaim** (delta, bravo, echo). One lineage, two scales.
*Response:* §1 rephrased: mechanism "carries no model-specific ingredient, though our evidence spans two scales of one model lineage." Non-Qwen replication remains future work (DeepSeek-V3's MTP head is the natural target; out of scope this round).

**C5. Prior-art verification** (alfa, echo). Several citations "unverifiable" via reviewers' arXiv searches; Medusa unacknowledged; delta surfaced 2603.05210 (Ben Shoham, vocab-trimming coverage/latency optimization).
*Response:* every citation in the bib was verified against the arXiv API before submission (Hooker PIE 1911.05248 and LLM-KICK 2310.01382 exist; reviewer searches hit throttling). Medusa (2401.10774) and Ben Shoham (2603.05210) now cited. The Ben Shoham paper optimizes vocab size for separate draft models and does not touch co-trained heads, robustness, or auditing; positioned in §7.

**C6. Mixed-precision arm under-measured** (delta, echo, charlie). Best 9B-long cell introduced mid-results without full protocol.
*Response:* full mixed grid (27B export + both suites, greedy + 3 sampled seeds; 9B sampled seeds 17/27) running now; cells land in Table 1 when complete. Speedup tie-threshold (0.04x seed noise) added to §4.0 so ordering claims respect measurement precision.

## Reviewer-specific notes

- **bravo#1 (standard-framework replication)**: out of scope by locked decision (one engine + independent third configuration); §8.2 now states speedups are engine-specific while acceptance/retention are engine-independent (torch = engine agreement).
- **charlie#3 (mechanism fails for low-rank)**: §6.1 now separates the two deletion geometries (channels → rare bins; spectral directions → uniform) under the one selection mechanism.
- **alfa#6 (multiple comparisons)**: six schemes are attempts to beat a baseline, all lost or tied; per-scheme cells in Appendix A let readers judge; permutation phrasing corrected.
- **alfa minor (1.06GB arithmetic)**: real error, fixed to 1.08GB projection / 1.34GB total everywhere including the figure.
- **echo#3 (distribution-exact)**: now cited to the standard speculative guarantee (Leviathan; Chen).

## Verdict distribution

alfa: Major revision. bravo: Major revision. charlie: Major revision. delta: Major revision. echo: Major revision.

**Meta-verdict: Major revision, with every consensus concern either answered by new measurement (C1, C2, C3, C6-partial) or honestly scoped (C4, C5) in the same revision cycle. Remaining open: mixed-arm full grid (running), non-Qwen replication (future work).**
