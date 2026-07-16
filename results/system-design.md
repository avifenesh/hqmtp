# From the frontier study to an efficient serving system

Principles (each measured, results/engine-e2e-v1.md):
1. Optimize BYTES-READ PER ACCEPTED TOKEN, not acceptance — cheap rounds beat
   high acceptance (student wins cells at -12pts acceptance; frtrim > natural
   everywhere on the same mechanism).
2. Output cuts are free and robust (vocab trim); function cuts (prune/distill)
   pay ~35% relative off-distribution, irreducibly (mass/Wanda/online/vocab-
   conditioned rankings all tested negative).
3. Train components in the regime of their COMPOSITION: chain-rollout training
   put the student above its teacher at depth; serving-regime data was +27pts.
   Live gates mandatory — replay metrics lie three different ways (degenerate
   loops, register mismatch, doc variance).
4. Over-provision open-domain traffic, compress narrow-domain traffic — spare
   capacity IS the robustness margin.
5. Spec pays more as target grows (9B 1.71x -> 27B 2.62x natural) and as
   context deepens — cheap drafts compound exactly where decode is slowest.

## Design: per-request draft dispatch

- Resident heads: frtrim (401MB) + distilled student (225MB); natural as fallback.
- Route on (context length, sampling mode):
    short           -> frtrim   (1.86x)
    long + greedy   -> frtrim   (2.15x)
    long + sampled  -> student  (2.68x @27B — the serving mode)
- Adaptive K from rolling acceptance: default 4, degrade to 2; never 8 (all arms
  regress at K=8; student degrades least).
- Draft-side temperature knob for sampled tails (scale draft logits before gumbel
  AND retained q — rejection sampling stays exact). Small engine change.
- Continuous live gate: per-session acceptance telemetry vs routing table;
  underperforming cell falls back to natural. Draft errors cost latency, never
  correctness — routing can be aggressive by construction.
- Retrain cadence: student refreshed from the deployment's own serving traffic
  (the pipeline is built: generate -> extract -> tt -> KD 24k -> CE 4k, ~12h).

Expected envelope from measured cells: ~1.9-2.7x over plain decode across the
request mix, vs 1.6-2.6x natural-only and 1.9-2.2x frtrim-only.

## v1 implemented (2026-07-16, bw24 PR #13)

BW24_MTP_DRAFT2 second resident head + spec::pick_draft routing on (ctx len,
sampled); scratch sized to the dispatched head; session geometry guard;
BW24_DISPATCH{,_CTX,_LOG}. Validated on 27B: routing correct all cells,
dispatched == standalone bit-for-bit, greedy-long via dispatch = 2.62x.

Found in the process: engine regression (bw24 #12) — sampled x DraftGeom heads
collapsed 69% -> 20% between f1395e90 and e1ef5ab (b-tier int4 window); frtrim
sampled and student greedy unaffected. The dispatch payoff cell resumes when the
engine lane fixes it; policy meanwhile: frtrim everywhere (still 1.86-2.6x).
