# hqmtp before/after eval — NVFP4 (st-ct) MTP head retrain

Judges task #10: retrain the `mtp.*` head against the quantized base's own distribution,
prove live-acceptance recovery. "Before" = shipped bf16 head (loader-law Q8_0 in-engine).
"After" = retrained head patched into the same checkpoint.

## Arms and artifacts

- Base (fixed, both runs): `qwen35-9b-nvfp4-st-ct` on bw24-g7e, bw24 `run-spec`/`replay-acceptance`.
- BEFORE: checkpoint as-is.
- AFTER: copy of the checkpoint with retrained `mtp.*` tensors patched in
  (`eval/patch_head.py`), everything else byte-identical.
- Reference ceiling: `qwen35-9b-hf` bf16 numbers (already measured).

## Contamination rule (hard)

Training data = captured corpus (`results/g7e/corpus/`: p1–p6 payloads, agent-loop
transcripts, generated continuations) and anything else EXCEPT the eval set below.
The eval prompts/corpora live in `eval/prompts/` + `eval/corpora/` and MUST NEVER enter
training. p1–p6 are training material → they are NOT eval prompts for the proof
(they remain a secondary tracked set, reported separately as "seen" numbers).

## Eval set (held out)

Live battery prompts (same classes as p1–p6, fresh content):
- `pe1-code-short.txt` ~150B, `pe2-code-medium.txt` ~6KB, `pe3-agentic-long.txt` ~23KB,
- `pe4-16k.txt`, `pe5-32k.txt`, `pe6-64k.txt` (fresh source material: CPython stdlib code,
  different public-domain prose, freshly composed agent-session doc).

Replay corpora (held out): `eval-code-64k.txt` (CPython concat), `eval-prose-64k.txt`
(different Gutenberg text), `eval-agentic-64k.txt` (fresh session doc).

## Protocol per arm (BEFORE and AFTER identical)

1. Live battery: `acceptance_battery.sh` — PROMPTS="pe1..pe6", KS="1 2 3 4", N=3,
   NGEN=128, RUN_AGENTLOOP=1 (8 turns, seed pe2). Self-consistency must PASS on every row.
2. Teacher-forced replay: `replay-acceptance` over the 3 eval corpora, K=4, stride=16,
   chunk=512 (fixed), 65536 tokens, dumps kept.
3. Speed: spec tok/s + speedup-vs-generate come free in the battery rows.

`eval/run_eval.sh <model_dir> <tag>` runs all of it → `eval/results/<tag>/`.
`eval/compare.py <before_tag> <after_tag>` → delta report.

## Metrics and success criteria

Primary: live-chain acceptance per (prompt,K), median of N=3.
- **Success**: AFTER closes ≥50% of the live (bf16 − st-ct) gap on the held-out battery,
  consistently across prompt classes (not one cell), at K=1..4.
- **Controls**: teacher-forced replay slot-0 within the arm's chunk-FP-order noise floor
  (~2-3%) of BEFORE — the head must not overfit into forced-context regression;
  self-consistency PASS everywhere (exactness is non-negotiable).
- Secondary: per-slot decay, chain len, spec tok/s / speedup (accepted-length × speed is
  the end-to-end win), seen-set (p1–p6) numbers reported for reference only.
- Clean negative = AFTER within noise of BEFORE on held-out live battery with healthy
  training curves — publishable as "PTQ mismatch is not head-repairable at 9B".

## Order of operations

1. BEFORE eval runs after the current p1–p6 battery finishes (GPU serial).
2. Training (task #10) touches only training data.
3. AFTER eval = patch head → same two commands, tag `after-<run-id>`.
