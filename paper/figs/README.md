# Figure plan

| id | file | placement | caption core |
|---|---|---|---|
| Fig 1 | f1_headline | §4.1 | End-to-end speedup over plain decode, aggregate cells; trim-NVFP4 fastest or tied in every cell |
| Fig 2 | f0_cost_anatomy | §2.1 | Weights read per drafted token: 1.32GB reference vs 0.21GB recipe |
| Fig 3 | f6_taxonomy | §3 | The two-family taxonomy with its measured signatures |
| Fig 4 | f2_tax_by_family | §4.2 | Acceptance retention vs shift distance; family alone separates the trajectories |
| Fig 5 | f3_per_bin | §4.2 | Per-bin retention under domain shift: pruning concentrates in rare bins, SVD damages all bins, NVFP4 flat |
| Fig 6 | f4_audit_scatter | §5.3 | In-dist disagreement ranks domain-shift tax exactly (Spearman 1.0) |
| Fig 7 | f5_hot_ratio | §6 | Hot-logit energy ratio of all 12,288 FFN channels peaks at the uniform share: no channel specializes |

Rebuild: `python3 make_figs.py && python3 make_diagrams.py` (data figures read ../../results/*.json).
