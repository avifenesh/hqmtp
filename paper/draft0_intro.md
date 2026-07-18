# Draft 0 Introduction (disposable framing scaffold)

<!-- Move 1: Stakes -->
Models in the Qwen3.5 and DeepSeek lineage ship with a co-trained multi-token-prediction (MTP) head: a spare transformer block that drafts tokens for speculative decoding. The draft is free in one sense: no external model to train, align, or load. It is expensive in another. The head is a full-width block with a full-vocabulary output projection, and on a 9B model it reads 1.3GB of weights for every drafted token. Decode is bandwidth-bound, so those reads come straight out of generation speed. Local deployments pay twice: llama.cpp-style stacks load the head as a separate 1 to 2GB file on cards where every gigabyte is context budget.

<!-- Move 2: Problem gap -->
Practitioners have one published lever for this cost. FR-Spec trims the draft vocabulary to the most frequent tokens and reports 1 to 3% end-to-end gain. The rest of the compression toolbox (pruning, low-rank factorization, quantization, distillation) comes from full-model compression, and standard practice evaluates it in distribution, on data that resembles the calibration set. That evaluation hides the failure that matters. We measure a pruned head that keeps 97% of its acceptance on calibration-style text and loses a third of it, relative, on fresh live traffic. The gap is structural. The parameters that serve rare content fire rarely, so they accumulate no mass on any finite calibration set, and every calibration-based selection statistic is blind to them. We tried six, including online adaptation on the live stream itself. All six failed the same way.

<!-- Move 3: Key abstraction -->
The results sort every compression method into two families. **Fidelity cuts** lower the numeric precision of every weight uniformly; 4-bit quantization is the extreme case. **Function cuts** remove parameters selectively (channel pruning, low-rank factorization) or refit the function on chosen data (distillation). The family, and only the family, determines robustness: fidelity cuts cost nothing on any distribution we tested, and function cuts pay an acceptance tax exactly on the content their calibration never saw.

<!-- Move 4: Design intuition -->
Draft heads are the ideal place to exploit this distinction, for two reasons. First, a draft error costs latency and never correctness, because the target model verifies every token; the head tolerates fidelity loss that no other component in the stack tolerates. Second, verification makes every drafted token a labeled measurement of the compressed function, which turns speculative decoding into a per-token oracle for compression damage. We use the first property to build the recipe and the second to build the audit.

<!-- Move 5: Contributions -->
This paper makes three contributions.

First, a zero-training compression recipe: trim the draft vocabulary to the model's own 32k hottest tokens, quantize the head to NVFP4, and keep full-vocabulary verification, which leaves the output distribution untouched. The recipe cuts the head's per-token reads from 1.33GB to 0.21GB.

Second, the fidelity/function taxonomy with the evidence that makes it usable. A 2x2 design separates dimensionality from data-dependence: data-free SVD pays a larger off-distribution tax than calibrated SVD, so dimensionality itself is the axis. A six-scheme selection study shows the tax survives every calibration choice.

Third, a pre-deployment audit that needs no out-of-distribution data: run the compressed head beside the reference on in-distribution documents and measure how often they disagree and how harmful the disagreements are. Harm per disagreement separates the families (0.14 for fidelity cuts, tie-flips; 0.28 to 0.54 for function cuts, function loss), and the in-distribution statistic predicts the off-distribution tax at Spearman 0.89 across seven compression arms.

<!-- Move 6: Results preview -->
On Qwen3.5-9B and Qwen3.6-27B in a production CUDA engine, the recipe delivers 1.6 to 2.0x and 2.0 to 2.7x end-to-end speedup over plain decoding. The unmodified head delivers 1.5x and 2.4x. All cells aggregate MT-Bench plus a 20-prompt long-context suite, with three sampling seeds each. The 27B long-context configuration reaches 127.5 tokens per second, 2.67x over plain decode, from a 259MB draft file. A third configuration, built independently in the engine's own lane, reproduces the pattern. Every negative result that shaped the taxonomy appears in the paper, with the measurement traps we hit (single-seed sampled cells swing by 12 points; aggregate protocol brings seed spread under 1.5) documented so others avoid them.
