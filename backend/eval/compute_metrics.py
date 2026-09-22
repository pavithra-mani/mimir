"""
Score generated general-summaries against reference summaries: WER,
ROUGE-1/2/L (precision/recall/F1), mean/std/95% bootstrap CI per metric, and
(only when a second --generated file is given) a paired significance test
between two conditions.

Setup:
    pip install rouge-score jiwer scipy

Input formats (JSONL, one record per line):
    --generated   {"video_id": "...", "summary": "..."}          (from generate_summaries.py)
    --references  {"video_id": "...", "reference_summary": "..."} (your Claude-generated references)

A plain JSON dict {"video_id": "summary text", ...} is also accepted for
either file.

Usage:
    python eval/compute_metrics.py \
        --generated eval/results/gemini_summaries.jsonl \
        --references eval/results/claude_references.jsonl \
        --report eval/results/metrics_report.json

    # Comparing two conditions (e.g. before/after a prompt change) instead of
    # just reporting descriptive stats against one reference set:
    python eval/compute_metrics.py \
        --generated eval/results/gemini_summaries_v1.jsonl \
        --compare-generated eval/results/gemini_summaries_v2.jsonl \
        --references eval/results/claude_references.jsonl \
        --report eval/results/metrics_report.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

import jiwer
from rouge_score import rouge_scorer
from scipy import stats

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_REFERENCES = RESULTS_DIR / "claude_references.jsonl"
DEFAULT_GENERATED = RESULTS_DIR / "gemini_summaries.jsonl"


def load_text_map(path: Path, text_key_candidates) -> dict:
    """Load {video_id: text} from a JSONL file of records or a flat JSON dict."""
    raw = path.read_text(encoding="utf-8")
    result = {}

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return {k: v for k, v in data.items()}
    except json.JSONDecodeError:
        pass

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        vid = rec["video_id"]
        text = next((rec[k] for k in text_key_candidates if k in rec), None)
        if text is None:
            raise KeyError(f"Record for {vid} has none of {text_key_candidates}: {rec.keys()}")
        result[vid] = text

    return result


def bootstrap_ci(values, n_resamples=10000, ci=0.95, seed=0):
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int((1 - ci) / 2 * n_resamples)
    hi_idx = int((1 + ci) / 2 * n_resamples)
    return (means[lo_idx], means[hi_idx])


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def std(values):
    if len(values) < 2:
        return float("nan")
    m = mean(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def score_pair(reference: str, hypothesis: str, scorer: rouge_scorer.RougeScorer) -> dict:
    rouge = scorer.score(reference, hypothesis)
    wer = jiwer.wer(reference, hypothesis) if reference.strip() and hypothesis.strip() else float("nan")
    return {
        "wer": wer,
        "rouge1_p": rouge["rouge1"].precision, "rouge1_r": rouge["rouge1"].recall, "rouge1_f": rouge["rouge1"].fmeasure,
        "rouge2_p": rouge["rouge2"].precision, "rouge2_r": rouge["rouge2"].recall, "rouge2_f": rouge["rouge2"].fmeasure,
        "rougeL_p": rouge["rougeL"].precision, "rougeL_r": rouge["rougeL"].recall, "rougeL_f": rouge["rougeL"].fmeasure,
    }


def aggregate(per_video_scores: list, metric_names) -> dict:
    agg = {}
    for m in metric_names:
        values = [s[m] for s in per_video_scores if s[m] == s[m]]  # drop NaN
        lo, hi = bootstrap_ci(values)
        agg[m] = {
            "mean": mean(values), "std": std(values),
            "ci95_lo": lo, "ci95_hi": hi, "n": len(values),
        }
    return agg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generated", type=Path, default=DEFAULT_GENERATED, help=f"default: {DEFAULT_GENERATED}")
    ap.add_argument("--compare-generated", type=Path, default=None, help="Second condition, for a paired significance test")
    ap.add_argument("--references", type=Path, default=DEFAULT_REFERENCES, help=f"default: {DEFAULT_REFERENCES}")
    ap.add_argument("--report", type=Path, default=RESULTS_DIR / "metrics_report.json", help="Write full JSON report here")
    args = ap.parse_args()

    if not args.references.exists():
        sys.exit(f"--references not found at {args.references} (run generate_references.py first, or pass --references explicitly)")
    if not args.generated.exists():
        sys.exit(f"--generated not found at {args.generated} (run generate_summaries.py first, or pass --generated explicitly)")

    generated = load_text_map(args.generated, ["summary"])
    references = load_text_map(args.references, ["reference_summary", "summary"])

    common_ids = sorted(set(generated) & set(references))
    missing_gen = set(references) - set(generated)
    missing_ref = set(generated) - set(references)
    if missing_gen:
        print(f"Warning: {len(missing_gen)} videos have a reference but no generated summary (skipped)")
    if missing_ref:
        print(f"Warning: {len(missing_ref)} videos have a generated summary but no reference (skipped)")
    if not common_ids:
        sys.exit("No overlapping video_ids between --generated and --references.")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    metric_names = [
        "wer", "rouge1_p", "rouge1_r", "rouge1_f",
        "rouge2_p", "rouge2_r", "rouge2_f",
        "rougeL_p", "rougeL_r", "rougeL_f",
    ]

    per_video = {}
    for vid in common_ids:
        per_video[vid] = score_pair(references[vid], generated[vid], scorer)

    agg = aggregate(list(per_video.values()), metric_names)

    print(f"\n--- Scored {len(common_ids)} videos ---")
    for m in metric_names:
        a = agg[m]
        print(f"  {m:10s}  mean={a['mean']:.4f}  std={a['std']:.4f}  95% CI=[{a['ci95_lo']:.4f}, {a['ci95_hi']:.4f}]  n={a['n']}")

    report = {
        "n_videos": len(common_ids),
        "missing_generated": sorted(missing_gen),
        "missing_reference": sorted(missing_ref),
        "aggregate": agg,
        "per_video": per_video,
    }

    if args.compare_generated:
        generated_b = load_text_map(args.compare_generated, ["summary"])
        common_ids_b = sorted(set(generated_b) & set(references) & set(generated))
        if len(common_ids_b) < 2:
            print("\nNot enough overlapping videos with --compare-generated to run a significance test.")
        else:
            per_video_b = {vid: score_pair(references[vid], generated_b[vid], scorer) for vid in common_ids_b}
            print(f"\n--- Paired significance test vs --compare-generated ({len(common_ids_b)} videos) ---")
            sig = {}
            for m in metric_names:
                a_vals = [per_video[vid][m] for vid in common_ids_b if per_video[vid][m] == per_video[vid][m]]
                b_vals = [per_video_b[vid][m] for vid in common_ids_b if per_video_b[vid][m] == per_video_b[vid][m]]
                if len(a_vals) != len(b_vals) or len(a_vals) < 2:
                    continue
                # Wilcoxon signed-rank: paired, no normality assumption - appropriate
                # for bounded per-video scores like ROUGE/WER that are rarely normal.
                try:
                    w_stat, w_p = stats.wilcoxon(a_vals, b_vals)
                except ValueError:
                    w_stat, w_p = float("nan"), float("nan")
                t_stat, t_p = stats.ttest_rel(a_vals, b_vals)
                sig[m] = {"wilcoxon_stat": w_stat, "wilcoxon_p": w_p, "paired_t_stat": t_stat, "paired_t_p": t_p}
                print(f"  {m:10s}  wilcoxon p={w_p:.4f}  paired-t p={t_p:.4f}")
            report["significance_vs_compare_generated"] = sig
    else:
        print("\nNo --compare-generated given - only one condition present, so no significance test is meaningful "
              "(a p-value needs something to compare against). Reporting descriptive stats + CI only.")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        print(f"\nFull report written to {args.report}")


if __name__ == "__main__":
    main()
