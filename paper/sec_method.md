# 3 The cut taxonomy and the recipe

<!-- Topic-sentence spine:
3.0 The head's cost decomposes into an output projection and a block; each compression method attacks one or both, and every method belongs to one of two families.
3.1 Trim removes output rows, never computation, which is why shift leaves it intact.
3.2 Quantization reduces the precision of every weight uniformly, which spreads its damage evenly.
3.3 Function cuts remove or refit specific computation, which ties their damage to the data that selected the cut.
3.4 The recipe composes the two safe moves.
-->

## 3.0 Where the bytes go

The co-trained head is one transformer block plus an output projection over the full vocabulary. On Qwen3.5-9B the projection is 248,320 x 4096 (1.08GB at 8-bit) and the block is roughly 243M parameters (0.26GB), so a single drafted token reads 1.34GB. Speculative decoding drafts K tokens per round, so at K=4 the head reads 5.3GB per verify pass. Every compression method attacks this budget somewhere: the projection's rows, the block's precision, the block's channels or rank, or the whole function. We group the methods by what they remove. **Fidelity cuts** lower the numeric precision of every weight and leave the function's structure intact. **Function cuts** remove structure: channels, ranks, or the function itself in favor of a retrained substitute.

## 3.1 Trim: remove outputs, keep computation

Trim ranks vocabulary rows by frequency over text the model itself generated and keeps the top 32,768 plus every special token. A draft-to-target index map converts the trimmed head's argmax back to full-vocabulary token ids, and verification runs against the full vocabulary. Trim's damage model is therefore explicit. A draft whose correct continuation falls outside the hot set proposes a wrong token, verification rejects it, and the round loses speed, never correctness. The tax equals the coverage gap and nothing else: within the hot vocabulary the trimmed head computes the identical function. Coverage is 95 to 99% of generated tokens on our suites, and unlike a function cut's damage, the gap is measurable on any traffic sample before deployment. Trim deletes no computation. Every channel, head, and rank still executes; only output rows the draft would rarely propose disappear. It is also the reason trim creates no further pruning headroom. We measured the hot-logit energy of every FFN channel: each channel spreads its output over hot and cold rows in proportion to their count. Deleting cold rows orphans no channel.

## 3.2 Fidelity cuts: uniform precision loss

Quantizing the head's matrices to 8-bit or 4-bit blocks perturbs every weight by a small rounding error. The perturbation is uniform by construction: the rounding selects no weight, so it singles out no sub-function for damage. The draft setting makes this nearly free twice over. A draft error costs one round of latency and never correctness, because the target verifies every token; and the head's job, ranking likely continuations, survives logit noise that would matter in a lossless setting. Our 4-bit format is NVFP4 (16-element blocks, e2m1 values, fp8 sub-scales), the project's deployment standard; Q4_0 behaves the same within noise. Section 4.2 quantifies the family's worst case at 1.9 retention points under severe domain shift, with 8-bit indistinguishable from the reference.

## 3.3 Function cuts: selection is the liability

Channel pruning and low-rank factorization remove specific structure, chosen either by a statistic over calibration data or, for data-free SVD, by the weight matrix's own spectrum. Distillation replaces the head with a smaller one trained on chosen data. All three tie the surviving function to a finite sample: the calibration set, the spectrum's large directions, or the training distribution. Content that the sample under-represents loses its machinery, and §4.2 shows the loss arrives exactly there, at 10 to 19 retention points under domain shift. The six-scheme study in §4.4 shows the liability follows from selection itself: statistics computed on finite samples cannot see rare-firing function, whatever the statistic.

Distillation deserves one scoping sentence. A distilled replacement reaches 91 to 98% of the teacher's chain acceptance on its training register at 13% of the compute. It is the only option when a model ships no co-trained head, or when a quantized base breaks the head it has. When a strong co-trained head exists, the trained substitute never beat the trimmed original in our aggregate cells, and it inherits the function-cut robustness profile.

## 3.4 The recipe

The safe moves compose: trim the projection (structural, coverage-capped, shift-proof in §4.2) and quantize what remains (uniform, 1.9 points worst case). One command exports the artifact from a HuggingFace checkpoint: rank the vocabulary on the model's own output, gather the hot rows, quantize head and block to NVFP4, and emit a standalone draft file with the index map. The 9B artifact is 212MB against the natural head's 1.34GB of per-token reads; the 27B artifact is 259MB. No training step exists anywhere in the pipeline, so the recipe runs on the deployment machine in minutes.
