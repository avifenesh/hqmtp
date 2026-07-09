# hqmtp before/after eval — teacher-forced replay only (protocol v2, locked 2026-07-09)

Judges task #10: retrain the 9B `mtp.*` head against the quantized base, prove recovery.
Metric = **teacher-forced replay only** (deterministic, loop-immune, arm-comparable).
Live serve sessions are PARKED until the engine ships sampled spec decode; the serve rig
(`serve/chat_proxy.py` + bw24-server) stays ready.

## Arms

- Base fixed: `qwen35-9b-nvfp4-st-ct` (compressed-tensors NVFP4, bf16 mtp.* head).
- BEFORE = shipped head. AFTER = retrained head via `patch_head.py` (head-only diff).
- Ceiling reference: `qwen35-9b-hf` bf16 full-prec.

## Corpora (two, no more)

1. **`gen-dist-9b.txt`** — the quant model's OWN generation texts (captured pi-session
   transcripts + agent-loop transcript), replayed as fixed text. This is where the live
   gap should reproduce deterministically; it is the corpus the retrain must improve.
   Successive retrain rounds must use FRESH captured generations for training and keep
   this file eval-only (or regenerate an eval split per round — never train on it).
2. **`eval-code-64k.txt`** — held-out generic corpus (CPython). CONTROL: must stay flat.

## Protocol per arm

`replay-acceptance` K=4 stride=8 chunk=512 (fixed), dumps kept:
- gen-dist corpus (~20k tokens) + eval-code (65536 tokens).

## Success criteria

- BEFORE must first show a real gap on gen-dist (st-ct vs bf16 beyond the ~2-3%
  quant-arm noise floor). If it doesn't, the head has nothing measurable to heal on this
  metric and the result is reported as such.
- AFTER closes ≥50% of that gap on gen-dist, while eval-code slot-0 stays within the
  noise floor of BEFORE (no forced-context regression).
- Bonus (optional, cheap): battery p1/K2-K3 recheck as a secondary live-flavored signal.

## Runbook

BEFORE/AFTER, on the box, per arm:
```
BW24_REPLAY_K=4 BW24_REPLAY_STRIDE=8 BW24_REPLAY_CHUNK=512 \
  BW24_REPLAY_DUMP=<out>.jsonl BW24_PROMPT_FILE=~/hqmtp-eval/corpora/<corpus>.txt \
  ./target/release/replay-acceptance <model_dir>       # + BW24_FULL_PREC=1 for the ceiling
```
Compare: `eval/compare.py` replay section (battery section idle until live resumes).
