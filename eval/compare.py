#!/usr/bin/env python3
"""Before/after report for the hqmtp head-retrain eval (see EVAL.md).

Usage: compare.py <results_dir_before> <results_dir_after> [--bf16 <battery.jsonl>]

Reads battery.jsonl + replay-*.jsonl from each dir. Prints:
  1. live battery: per-(prompt,K) median acceptance before/after (+recovered fraction of
     the bf16 gap when --bf16 given)
  2. replay control: slot-0 by depth bucket, before vs after (must sit within the
     ~2-3% chunk-FP-order noise floor)
  3. self-consistency violations (any FAIL = eval invalid)
"""
import json
import statistics as st
import sys

BUCKETS = [(0, 2048), (2048, 8192), (8192, 16384), (16384, 32768), (32768, 65536)]


def battery(path):
    cells, fails = {}, 0
    for line in open(path):
        r = json.loads(line)
        cells.setdefault((r["prompt"], r["k"]), []).append(r["acc_rate"])
        if r.get("self_consistency") not in (None, "PASS"):
            fails += 1
    return {k: st.median(v) for k, v in cells.items()}, fails


def replay(path):
    rows = [json.loads(l) for l in open(path)]
    out = {}
    for lo, hi in BUCKETS:
        sel = [r for r in rows if lo <= r["pos"] < hi]
        if sel:
            out[(lo, hi)] = sum(r["hits"][0] for r in sel) / len(sel)
    return out


def main():
    before_dir, after_dir = sys.argv[1], sys.argv[2]
    bf16 = None
    if "--bf16" in sys.argv:
        bf16, _ = battery(sys.argv[sys.argv.index("--bf16") + 1])

    b, bfail = battery(f"{before_dir}/battery.jsonl")
    a, afail = battery(f"{after_dir}/battery.jsonl")
    print("== live battery (median acceptance)")
    hdr = f"{'cell':14} {'before':>7} {'after':>7} {'delta':>7}"
    print(hdr + ("  bf16   recovered%" if bf16 else ""))
    for cell in sorted(set(b) & set(a)):
        line = f"{cell[0]+'/K'+str(cell[1]):14} {b[cell]:7.3f} {a[cell]:7.3f} {a[cell]-b[cell]:+7.3f}"
        if bf16 and cell in bf16:
            gap = bf16[cell] - b[cell]
            rec = (a[cell] - b[cell]) / gap * 100 if abs(gap) > 1e-9 else float("nan")
            line += f" {bf16[cell]:6.3f} {rec:10.1f}"
        print(line)
    if bfail or afail:
        print(f"!! SELF-CONSISTENCY FAILURES: before={bfail} after={afail} — EVAL INVALID")

    print("\n== replay control (slot-0 by depth; must stay within ~0.03 of before)")
    for corpus in ["eval-code-64k", "eval-prose-64k", "eval-agentic-64k"]:
        try:
            rb = replay(f"{before_dir}/replay-{corpus}.jsonl")
            ra = replay(f"{after_dir}/replay-{corpus}.jsonl")
        except FileNotFoundError:
            print(f"{corpus}: missing dump, skipped")
            continue
        print(f"-- {corpus}")
        for k in sorted(set(rb) & set(ra)):
            flag = "  <-- CHECK" if abs(ra[k] - rb[k]) > 0.03 else ""
            print(f"  {k[0]:>6}-{k[1]:<6} {rb[k]:6.3f} -> {ra[k]:6.3f} ({ra[k]-rb[k]:+.3f}){flag}")


if __name__ == "__main__":
    main()
