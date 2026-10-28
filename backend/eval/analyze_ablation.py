"""Summarise ablation_visual.jsonl: accuracy per condition, overall and per Video-MME
task_type, with paired comparison against the text_only baseline.

Paired because every condition answers the SAME questions:
  - delta accuracy with a 95% paired-bootstrap CI,
  - exact McNemar test on the discordant pairs (baseline-wrong/cond-right vs reverse).
Stdlib only.

Usage: python eval/analyze_ablation.py [--input results/ablation_visual.jsonl] [--baseline text_only]
"""
import argparse
import json
import random
from collections import defaultdict
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar (binomial) p-value on discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * p)


def bootstrap_ci(diffs, n_boot=5000, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=HERE / "results" / "ablation_visual.jsonl")
    ap.add_argument("--baseline", default="text_only")
    ap.add_argument("--min-n", type=int, default=5, help="hide task types with fewer paired questions")
    ap.add_argument("--output", type=Path, default=HERE / "results" / "ablation_report.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.input.open(encoding="utf-8") if l.strip()]
    by = defaultdict(dict)                       # (video, qid) -> condition -> record
    for r in rows:
        by[(r["videoID"], r["question_id"])][r["condition"]] = r
    conds = sorted({r["condition"] for r in rows}, key=lambda c: (c != args.baseline, c))
    if args.baseline not in conds:
        raise SystemExit(f"baseline '{args.baseline}' not in results")

    report = {"n_questions": len(by), "conditions": {}, "by_task_type": {}}

    def paired(cond, subset):
        pairs = [(v[args.baseline]["correct"], v[cond]["correct"]) for v in subset
                 if args.baseline in v and cond in v]
        return pairs

    print(f"{len(by)} questions, conditions: {conds}\n")
    print(f"{'condition':18s} {'n':>5s} {'acc':>7s} {'delta':>8s} {'95% CI':>17s} {'McNemar p':>10s}")
    for cond in conds:
        pairs = paired(cond, by.values())
        n = len(pairs)
        if not n:
            continue
        acc = sum(c for _, c in pairs) / n
        base = sum(b for b, _ in pairs) / n
        d = [int(c) - int(b) for b, c in pairs]
        lo, hi = bootstrap_ci(d) if cond != args.baseline else (0.0, 0.0)
        b01 = sum(1 for b, c in pairs if not b and c)
        b10 = sum(1 for b, c in pairs if b and not c)
        p = mcnemar_exact(b01, b10) if cond != args.baseline else float("nan")
        print(f"{cond:18s} {n:5d} {acc:7.3f} {acc - base:+8.3f} [{lo:+.3f},{hi:+.3f}] {p:10.4f}")
        report["conditions"][cond] = {"n": n, "accuracy": acc, "delta": acc - base,
                                      "ci95": [lo, hi], "mcnemar_p": p, "gained": b01, "lost": b10}

    print("\nBy task type (accuracy; delta vs baseline):")
    types = sorted({r["task_type"] for r in rows})
    header = f"{'task_type':24s} {'n':>4s} " + " ".join(f"{c[:14]:>14s}" for c in conds)
    print(header)
    for tt in types:
        subset = [v for v in by.values() if any(r["task_type"] == tt for r in v.values())]
        n = len(paired(args.baseline, subset))
        if n < args.min_n:
            continue
        cells, rec = [], {"n": n}
        for cond in conds:
            pairs = paired(cond, subset)
            if not pairs:
                cells.append(f"{'-':>14s}"); continue
            acc = sum(c for _, c in pairs) / len(pairs)
            base = sum(b for b, _ in pairs) / len(pairs)
            cells.append(f"{acc:6.3f}({acc - base:+.2f})".rjust(14))
            rec[cond] = {"accuracy": acc, "delta": acc - base}
        print(f"{tt:24s} {n:4d} " + " ".join(cells))
        report["by_task_type"][tt] = rec

    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
