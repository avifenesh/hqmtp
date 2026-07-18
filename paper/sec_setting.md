# 2 Setting: draft heads, and verification as an oracle

<!-- Topic-sentence spine:
2.1 Co-trained MTP heads make speculative decoding a built-in feature with a fixed cost.
2.2 Verification makes draft errors free and draft measurements exact.
-->

## 2.1 Co-trained draft heads

Speculative decoding accelerates autoregressive generation by drafting K candidate tokens cheaply and verifying them with one batched pass of the target model. Classical pipelines draft with a separate small model, which must exist, fit in memory, and match the target's distribution. Co-trained multi-token-prediction heads remove those requirements: the model ships with one extra transformer block, trained jointly with the trunk, that predicts ahead from the trunk's hidden state. The head inherits the trunk's distribution by construction and reaches acceptance rates external drafters struggle to match.

The built-in convenience carries a fixed cost with two parts (Figure 2). The output projection spans the full vocabulary: 248,320 rows at Qwen3.5 scale, 1.06GB of 8-bit weights read per drafted token. The block itself adds 0.26GB. Decode-time inference is bandwidth-bound, so at K=4 the head's 5.3GB of reads per round competes directly with the target's own weights for the same memory bus. Deployments that load the head as a separate artifact pay in capacity as well: 1 to 2GB of a consumer GPU's memory that could hold context.

## 2.2 The verifier as a measurement instrument

Two properties make draft heads the right laboratory for studying compression. First, drafting is error-tolerant by construction. The target verifies every drafted token; a wrong draft costs one round of wasted speculation and never changes the output. Compression errors that would corrupt a base model only slow a draft head down. Second, verification labels every drafted token against ground truth. Each position yields an exact, binary, task-grounded measurement of whether compression changed the function's behavior there, at a scale of millions of labeled positions per GPU-hour. Classifier benchmarks measure compression damage through task metrics with their own noise; the verifier measures it directly. We use the first property to justify aggressive compression and the second to measure precisely what each kind of compression destroys.
