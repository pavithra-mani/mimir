"""
Token-count sanity check for the Video-MME reference-summary job.

Replaces the earlier word-count guess (300/300/300 duration split x assumed
words-per-minute) with real token counts from Anthropic's count_tokens
endpoint, run against a sample of actually-extracted subtitle files.

Setup:
    pip install anthropic          # not in requirements.txt - eval-only tool
    export ANTHROPIC_API_KEY=...   # or `ant auth login`

Usage:
    python check_transcript_tokens.py --subs-dir /path/to/extracted_srts

    # Weight the extrapolation by Video-MME's real short/medium/long split
    # (needs the videomme/test-00000-of-00001.parquet metadata file):
    python check_transcript_tokens.py --subs-dir ./srts --metadata ./meta.parquet

    # Also measure real output-token length by actually generating N summaries
    # (costs a few cents - this is the only mode that spends money):
    python check_transcript_tokens.py --subs-dir ./srts --metadata ./meta.parquet --sample-generate 6

IMPORTANT: SUMMARY_SYSTEM_PROMPT below is a placeholder. Token counts (and
the --sample-generate cost/output estimate) are only meaningful if this
matches the actual prompt you'll use to generate the 900 reference summaries
- edit it before trusting the numbers.
"""

import argparse
import re
import sys
from pathlib import Path

from anthropic import Anthropic

# Per-1M-token pricing, USD. Batch API is 50% off both columns.
PRICING = {
    "claude-opus-5": {"input": 5.00, "output": 25.00},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}

DURATION_BUCKETS = ["short", "medium", "long"]
FULL_DATASET_COUNTS = {"short": 300, "medium": 300, "long": 300}  # Video-MME's real split

# --- EDIT THIS to match your actual reference-generation prompt ---
SUMMARY_SYSTEM_PROMPT = (
    "You are given the transcript of a video. Write a concise, faithful "
    "summary of what happens in the video, in 3-6 sentences."
)


def extract_subtitle_text(path: Path) -> str:
    """Strip SRT/VTT timing codes and cue numbers down to plain transcript text."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    lines = raw.splitlines()

    timing_re = re.compile(r"\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[.,]\d{3}")
    index_re = re.compile(r"^\d+$")
    tag_re = re.compile(r"<[^>]+>")

    kept = []
    prev = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.upper().startswith("WEBVTT") or line.upper().startswith("NOTE"):
            continue
        if index_re.match(line) or timing_re.search(line):
            continue
        line = tag_re.sub("", line).strip()
        if not line or line == prev:  # dedupe rolling auto-caption repeats
            continue
        kept.append(line)
        prev = line

    return " ".join(kept)


def load_duration_lookup(metadata_path: Path) -> dict:
    """videoID (YouTube id) -> duration bucket, from Video-MME's metadata parquet."""
    import pandas as pd

    df = pd.read_parquet(metadata_path).drop_duplicates("video_id")
    return dict(zip(df["videoID"], df["duration"]))


def count_input_tokens(client: Anthropic, model: str, transcript: str) -> int:
    resp = client.messages.count_tokens(
        model=model,
        system=SUMMARY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": transcript}],
    )
    return resp.input_tokens


def measure_output_tokens(client: Anthropic, model: str, transcript: str) -> int:
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SUMMARY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": transcript}],
    )
    return resp.usage.output_tokens


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subs-dir", required=True, type=Path, help="Directory of extracted .srt/.vtt files")
    ap.add_argument("--metadata", type=Path, default=None, help="Video-MME metadata parquet, for duration-bucket weighting")
    ap.add_argument("--model", default="claude-sonnet-5", choices=sorted(PRICING))
    ap.add_argument("--sample-generate", type=int, default=0, help="Also generate N real summaries to measure output tokens (spends money)")
    args = ap.parse_args()

    sub_files = sorted(list(args.subs_dir.glob("*.srt")) + list(args.subs_dir.glob("*.vtt")))
    if not sub_files:
        sys.exit(f"No .srt/.vtt files found in {args.subs_dir}")

    duration_lookup = load_duration_lookup(args.metadata) if args.metadata else {}

    client = Anthropic()
    per_bucket = {b: [] for b in DURATION_BUCKETS}
    unmatched = []

    print(f"Counting input tokens for {len(sub_files)} transcripts against {args.model} ...")
    for path in sub_files:
        text = extract_subtitle_text(path)
        if not text:
            print(f"  [skip] {path.name}: no extractable text")
            continue
        tokens = count_input_tokens(client, args.model, text)
        bucket = duration_lookup.get(path.stem)
        if bucket in per_bucket:
            per_bucket[bucket].append((path.stem, tokens))
        else:
            unmatched.append((path.stem, tokens))
        print(f"  {path.name}: {tokens} input tokens" + (f"  [{bucket}]" if bucket else ""))

    all_samples = [t for lst in per_bucket.values() for _, t in lst] + [t for _, t in unmatched]
    if not all_samples:
        sys.exit("No samples produced a token count - nothing to report.")

    print("\n--- Per-bucket averages ---")
    have_buckets = any(per_bucket[b] for b in DURATION_BUCKETS)
    for b in DURATION_BUCKETS:
        if per_bucket[b]:
            avg = sum(t for _, t in per_bucket[b]) / len(per_bucket[b])
            print(f"  {b:7s}: n={len(per_bucket[b]):3d}  avg_input_tokens={avg:,.0f}")
    if unmatched:
        avg = sum(t for _, t in unmatched) / len(unmatched)
        print(f"  (unmatched to a bucket, no --metadata match): n={len(unmatched)}  avg_input_tokens={avg:,.0f}")

    # Extrapolate total input tokens for the full 900-video dataset.
    if have_buckets and all(per_bucket[b] for b in DURATION_BUCKETS):
        total_input_tokens = sum(
            (sum(t for _, t in per_bucket[b]) / len(per_bucket[b])) * FULL_DATASET_COUNTS[b]
            for b in DURATION_BUCKETS
        )
        method = "weighted by measured per-bucket average x Video-MME's real 300/300/300 split"
    else:
        flat_avg = sum(all_samples) / len(all_samples)
        total_input_tokens = flat_avg * 900
        method = "flat average across all samples x 900 (no --metadata bucket breakdown available)"

    # Output tokens: measure a few real generations if asked, else assume a flat estimate.
    if args.sample_generate > 0:
        gen_samples = sub_files[: args.sample_generate]
        out_tokens = []
        print(f"\nGenerating {len(gen_samples)} real summaries to measure output length ...")
        for path in gen_samples:
            text = extract_subtitle_text(path)
            if not text:
                continue
            out_tokens.append(measure_output_tokens(client, args.model, text))
        avg_output = sum(out_tokens) / len(out_tokens) if out_tokens else 300
        output_note = f"measured from {len(out_tokens)} real generations"
    else:
        avg_output = 300
        output_note = "assumed (pass --sample-generate N to measure real output length)"

    total_output_tokens = avg_output * 900

    price = PRICING[args.model]
    standard_cost = (total_input_tokens / 1e6) * price["input"] + (total_output_tokens / 1e6) * price["output"]
    batch_cost = standard_cost * 0.5

    print(f"\n--- Extrapolated to all 900 videos ({method}) ---")
    print(f"  Estimated total input tokens:  {total_input_tokens:,.0f}")
    print(f"  Estimated total output tokens: {total_output_tokens:,.0f}  ({output_note})")
    print(f"\n  {args.model} standard cost: ${standard_cost:.2f}")
    print(f"  {args.model} batch cost:    ${batch_cost:.2f}")


if __name__ == "__main__":
    main()
