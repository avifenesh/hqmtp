# Draft-head compression toolkit

Code for the paper *Fidelity Cuts vs Function Cuts: What Compressing Speculative-Decoding
Draft Heads Destroys, and How to Detect It Before Deployment* (`paper/`). The recipe, the
audit and the study scripts run forward passes only; only the distillation lane trains.

## The two tools practitioners need

| Tool | Script | Paper section |
|---|---|---|
| **The recipe** — export a trimmed + NVFP4 draft artifact from a HuggingFace checkpoint | `export_frtrim.py` (vocabulary ranking from `build_hot_vocab.py`, NVFP4 packing from `nvfp4_pack.py`) | §3.4 |
| **The audit** — classify a compressed head before deployment (disagreement rate + harm per disagreement) | `audit_lab.py` | §5 |

`audit_lab.py` builds all seven arms (Q8/Q4/NVFP4 roundtrips, prune 25/50, data-free and
calibrated SVD) against the reference head, runs them teacher-forced over identical states,
and reports per-arm accept/agree rates with log-frequency-quintile bins. `--indist` and
`--ood` point at extracted hidden-state dumps; `--out` writes the JSON the paper's figures
read. It reproduced identically on Qwen3.5-9B and Qwen3.6-27B (Spearman 1.0 both).

## Study scripts (paper evidence, in section order)

- `hotratio_prune.py`, and the hot-ratio dump used by Figure 7 — superposition measurement (§6.2)
- `build_trueood.py` — the domain-shift suite: de/zh/he/ru Wikipedia + PubMed + C4 (§4.2)
- `gen_ood_stream.py`, `gen_longset.py` — same-register OOD stream and the long-context suite (§4.0)
- `e4_canary.py` — divergence-conditioned selection, the sixth failed scheme (§4.4)
- `entropy_study.py` — draft-distribution entropy/KL under quantization (appendix)
- `eval_replay.py`, `eval_chain.py`, `eval_indist.py` — teacher-forced acceptance evaluators
- `extract.py`, `hdata.py` — hidden-state extraction and storage
- `heads.py`, `qwen_mtp.py`, `nvfp4.py` — head module, MTP wrapper, NVFP4 reference codec

Distillation lane (taxonomy arm, §3.3): `train.py`, `teacher_targets.py`, `export_sv.py`,
`gate_head.py`, `DISTILL.md`.

Engine-side (speedups, §4.1): the [memra](https://github.com/avifenesh/memra) CUDA engine
(bw24 at the time of the paper) loads the exported artifact via
`MEMRA_MTP_DRAFT=<file>.gguf` (`BW24_MTP_DRAFT` then); sweep batteries live in the engine repo.

Figures: `paper/figs/make_figs.py` regenerates every data figure from `results/*.json`.
