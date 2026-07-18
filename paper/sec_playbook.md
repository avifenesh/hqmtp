# 8 Playbook and limitations

<!-- Topic-sentence spine:
8.1 One table maps deployment constraints to the safe cut.
8.2 The study's boundaries are explicit.
-->

## 8.1 Constraint to cut

| Your situation | Do this | Expect |
|---|---|---|
| Model ships a co-trained head; decode is bandwidth-bound | Trim + NVFP4 (the recipe, §3.4) | 1.8 to 2.7x over plain decode; 212 to 259MB artifact; no training |
| Memory pressure demands more than 4-bit trim delivers | Stop. Pruning and low-rank buy at most 90MB more and pay 10 to 19 acceptance points on shifted traffic | Run the audit (§5) before believing any further cut |
| Compressed draft from any source, provenance unknown | Audit it: harm rate under 0.15 is noise-type, above 0.25 is function-type | Minutes of forward passes; Spearman 1.0 tax ranking on our hardest suite |
| No co-trained head exists, or a quantized base broke it | Distill a replacement on the model's own serving-regime output (chain KD, then brief CE) | 91 to 98% of a co-trained teacher's chain acceptance; function-cut robustness profile |
| Benchmarking any sampled-decoding change | Aggregate suites with 3+ seeds; never single prompts | Single-prompt sampled cells swing 12 points across seeds |

## 8.2 Limitations

The evidence covers one model family (Qwen3.5-9B, Qwen3.6-27B) and one engine, with an independent third configuration reproducing the recipe's pattern. Speedup magnitudes are engine-specific; acceptance and retention are not, since teacher-forced torch measurements and live engine chains agree (§6.3). Other architectures' heads may sit differently on the cost curve. The taxonomy's mechanism (§6) has no model-specific ingredient. The impossibility result covers zero-training, statistics-based selection; gradient-informed selection and prune-then-finetune recovery sit outside the zero-training envelope this paper commits to, and could in principle re-earn their cuts. The audit requires the uncompressed reference head for its comparison passes. The domain-shift suite, though genuinely shifted (the reference's own accuracy drops 2.4x), is 24 documents; the taxes it measures carry the confidence intervals of §4.2, and wider suites would tighten them. Hot-vocabulary coverage caps trim at 95 to 99% on the registers we serve, and coverage is register-bound where retention is not: on our domain-shift suite the hot set covers 93 to 95% of English scientific and web text but 62% of German, 21% of Russian, 9% of Hebrew, and 5% of Chinese Wikipedia tokens, while the 4-bit arms keep at least 97.5% within-hot retention on those same corpora. A deployment serving registers outside its hot set needs its vocabulary re-ranked, and the check is cheap: coverage is a token count on any traffic sample, and re-export takes minutes.
