"""
Generate reference summaries with Claude, for scoring the Gemini pipeline's
summaries against (ROUGE/WER need a reference text - Video-MME doesn't ship
one, since it's a QA benchmark, not a summarization benchmark).

Uses a different model family than the one under test (Gemini) on purpose -
see the earlier discussion: comparing Gemini's summary to another Gemini
summary would be self-grading.

Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY=...   # or `ant auth login`

Usage:
    python eval/generate_references.py \
        --subs-dir /path/to/extracted_srts \
        --output eval/results/claude_references.jsonl

Output is JSONL: {"video_id": "...", "reference_summary": "..."} - the exact
shape compute_metrics.py's --references expects.

Resumable, same pattern as generate_summaries.py: --resume skips video_ids
already present in --output, each result is flushed immediately.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from anthropic import Anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent))
from subtitle_utils import parse_subtitle_to_transcript, transcript_to_text

# --- EDIT THIS to match whatever prompt you settle on. Token/cost estimates ---
# --- from check_transcript_tokens.py are only valid if this text matches   ---
# --- what you used there.                                                  ---
REFERENCE_SYSTEM_PROMPT = (
    "You are given the transcript of a video. Write a concise, faithful "
    "summary of what happens in the video, in 3-6 sentences."
)

MAX_TRANSCRIPT_CHARS = 100_000  # generous; Sonnet 5's context window is 1M tokens


def load_done_ids(output_path: Path) -> set:
    if not output_path.exists():
        return set()
    done = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["video_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def generate_reference(client: Anthropic, model: str, transcript_text: str) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        thinking={"type": "disabled"},  # plain summarization, no need to pay for reasoning here
        system=REFERENCE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": transcript_text[:MAX_TRANSCRIPT_CHARS]}],
    )
    return next((b.text for b in response.content if b.type == "text"), "").strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subs-dir", type=Path, default=Path(__file__).resolve().parent / "data" / "subtitles", help="Directory of extracted .srt/.vtt files (default: matches download_chunks.py's output)")
    ap.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results" / "claude_references.jsonl", help="JSONL output path (appended to)")
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--batch-size", type=int, default=25, help="Videos per progress checkpoint")
    ap.add_argument("--sleep-between", type=float, default=0.0, help="Seconds to sleep between calls")
    ap.add_argument("--limit", type=int, default=None, help="Cap the number of videos processed (smoke testing)")
    ap.add_argument("--resume", action="store_true", help="Skip video_ids already present in --output")
    args = ap.parse_args()

    sub_files = sorted(list(args.subs_dir.glob("*.srt")) + list(args.subs_dir.glob("*.vtt")))
    if args.limit:
        sub_files = sub_files[: args.limit]
    if not sub_files:
        sys.exit(f"No .srt/.vtt files found in {args.subs_dir}")

    done_ids = load_done_ids(args.output) if args.resume else set()
    if done_ids:
        print(f"Resuming: {len(done_ids)} videos already in {args.output}, skipping those.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    client = Anthropic()
    todo = [p for p in sub_files if p.stem not in done_ids]
    print(f"Generating references for {len(todo)} videos ({len(sub_files)} total, {len(done_ids)} already done) with {args.model}")

    errors = []
    t_start = time.time()
    with args.output.open("a", encoding="utf-8") as out_f:
        for i, path in enumerate(todo, 1):
            video_id = path.stem
            try:
                transcript = parse_subtitle_to_transcript(path)
                text = transcript_to_text(transcript)
                if not text.strip():
                    raise ValueError("no extractable subtitle text")

                reference = generate_reference(client, args.model, text)
                out_f.write(json.dumps({"video_id": video_id, "reference_summary": reference}) + "\n")
                out_f.flush()

            except Exception as e:
                errors.append({"video_id": video_id, "error": str(e)})
                print(f"  [ERROR] {video_id}: {e}")

            if i % args.batch_size == 0 or i == len(todo):
                elapsed = time.time() - t_start
                print(f"[checkpoint] {i}/{len(todo)} done, {len(errors)} errors, {elapsed:.1f}s elapsed")

            if args.sleep_between:
                time.sleep(args.sleep_between)

    if errors:
        err_path = args.output.with_suffix(".errors.json")
        err_path.write_text(json.dumps(errors, indent=2))
        print(f"\n{len(errors)} videos failed - details in {err_path}")

    print(f"\nDone. References written to {args.output}")


if __name__ == "__main__":
    main()
