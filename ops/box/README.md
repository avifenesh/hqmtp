# Box operation chains (bw24-g7e)

Runbooks for the GPU box. Everything here regenerates box-side state after loss;
see `results/engine-e2e-v1.md` for the numbers each chain produced and
`~/.claude` memory `box-bw24-g7e-migration` for the 2026-07-19 termination lesson.

| script | purpose |
|---|---|
| `restore2.sh` | full environment rebuild: bw24-server lifecycle (`BW24_MODELS="hqmtp-9b=<path>" BW24_ADDR=127.0.0.1:8014`), corpus v2 regen (1200 docs 9B / 600 docs 27B through the serving regime), hot32k build, all six draft exports, prompt suites, then EBS persistence tarballs to `/data/hqmtp-backup/<ts>/` |
| `probes.sh` | direction probes P1 (K-depth sweep), P2 (quantized-base compose), P3 (zh vocab re-rank), P4 (base-model audit) |
| `p2b.sh` | P2 retry against the Q4_K_M+NVFP4 27B gguf (direct file path, not the symlink dir) |
| `r2chain.sh` | round-2 review response: per-domain audit (LODO), hot coverage, mixed-arm grid |

Python experiment scripts live in `distill/` (synced to box `~/hqmtp-distill/`).

Rules learned the hard way:
1. `mark()` with ABSOLUTE paths only — cwd-relative markers scatter and break waiters.
2. Record numbers into `results/` immediately; box-only state is one spot-stop from gone.
3. NVMe (`/opt/dlami/nvme`) dies on spot-stop; EBS (`/data`) survives. Persist regenerated
   corpora/artifacts to `/data/hqmtp-backup/` at chain end (restore2.sh does).
4. Speed cells need the GPU exclusive; acceptance-only cells tolerate sharing.
5. Rebuilt corpora/hot-sets/suites are not bit-identical to the lost ones: compare new
   cells within-suite only.

6. `chat_proxy.py` was box-only and died with the box; reconstructed into `distill/`.
   Rule: every module a chain imports lives in the repo, no exceptions. Venv deps:
   torch transformers datasets accelerate gguf numpy safetensors sentencepiece jinja2.
