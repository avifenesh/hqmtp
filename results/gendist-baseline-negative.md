# Gen-dist replay baseline — NEGATIVE (no replay-visible gap to heal)

Date: 2026-07-09 · protocol v2 (teacher-forced replay only, 9B)

## Setup

Corpus `gen-dist-9b.txt` = the st-ct quant model's OWN generation texts (3 pi-session
transcripts incl. degenerate tails + captured agent-loop transcript), replayed as fixed
text through both arms (identical contexts, K=4, stride 8, chunk 512, 2223 eval positions).

## Result

| depth | n | bf16 slot0 | st-ct slot0 | delta |
|---|---|---|---|---|
| 0–2048 | 255 | 0.431 | 0.400 | +0.031 |
| 2048–6000 | 494 | 0.287 | 0.279 | +0.008 |
| 6000–12000 | 750 | 0.627 | 0.635 | −0.008 |
| 12000–50000 | 724 | 0.728 | 0.713 | +0.015 |
| **all** | 2223 | **0.562** | **0.554** | **+0.008** |

All deltas at/below the quant-arm chunk-FP-order noise floor (2–3%). Chain scores favor
st-ct slightly at depth.

## Conclusion

Under teacher forcing, the bf16 MTP head predicts the NVFP4 base as well as the bf16
pairing does — on generic corpora (replay v2) AND on the quant model's own generation
distribution (this run). Combined with the live gaps that DO exist (st-ct −5.6pts vs
bf16 in the p1/K3 smoke; p1/K2-K3 battery cells), the phenomenon is fully localized:

**NVFP4 weight compression does not break head↔hidden-state compatibility. The live
acceptance drop comes from generation dynamics — divergent self-generated text and/or
live draft-KV/session mechanics — which the replay metric cannot see by construction.**

## Status of the core proof (task #10)

Per EVAL.md v2 criteria: no replay-visible gap → a head retrain has no target on the
locked metric. The retrain-recovery proof is BLOCKED until live-regime measurement
returns (engine sampled-spec fix), unless the protocol adds a dynamics-sensitive
deterministic metric. Candidate for later discussion: replay each arm over the OTHER
arm's generated text (cross-text replay) — deterministic, and sensitive to the
text-divergence component of the live gap.
