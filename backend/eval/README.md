# Summarization pipeline eval

Tests the general-summary quality of the real Gemini pipeline (the
`SemanticChunker -> KnowledgeGraph -> VectorStore -> RetrievalEngine ->
Gemini` path in `pipeline/orchestrator.py`) against a stratified subset of
Video-MME, using WER and ROUGE-1/2/L scored against Claude-generated
reference summaries. Only the general-summary path is tested - transcription
is mocked with the video's existing subtitles, and topic-summary/keyframes
are out of scope.

## What's already done for you

- `chunk_1-7_video_ids.txt` - the 317 video IDs in Video-MME chunks 1-7,
  computed by listing each chunk's remote zip central directory (a few KB of
  HTTP range requests) without downloading any of the ~46GB of actual video
  files. Chunks 1-7 span all 6 domains and all 3 duration buckets in roughly
  the same proportions as the full dataset (checked before committing to
  this split).
- `results/claude_references.jsonl` - reference summaries for whichever of
  those 317 videos have subtitles on YouTube (see "266, not 317" below),
  generated once with Claude Sonnet 5. **You don't need an Anthropic API key
  or to regenerate this** - it's already here and both of you should score
  against the same file.

## Directory layout

```
eval/
  chunk_1-7_video_ids.txt      # manifest: which 317 videos this eval uses
  download_chunks.py           # pulls subtitle.zip + metadata, filters by the manifest
  subtitle_utils.py            # shared SRT/VTT -> transcript-dict parser
  generate_summaries.py        # runs the REAL pipeline + Gemini -> results/gemini_summaries.jsonl
  generate_references.py       # Claude reference summaries (already run - see above)
  compute_metrics.py           # WER/ROUGE + stats, generated vs. reference
  check_transcript_tokens.py   # standalone token/cost estimator (not part of the main flow)
  data/                        # downloaded subtitles + metadata (gitignored, regenerate via download_chunks.py)
  results/                     # jsonl outputs + the final metrics report (tracked)
```

## Two environments, on purpose

`generate_summaries.py` imports the real `pipeline` package (chunking, KG,
retrieval, orchestrator) - it needs the **full backend environment**:

```bash
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

`generate_references.py`, `compute_metrics.py`, and `download_chunks.py`
only need `anthropic` / `rouge-score` / `jiwer` / `huggingface_hub` - a
handful of small packages that conflict with the backend venv's pinned
`anyio` (fastapi wants `<4.0`, `anthropic` pulls in `>=4.10` via its HTTP
client). Use a **separate, lightweight venv** for these three scripts so you
don't destabilize the backend venv:

```bash
cd backend/eval
python3 -m venv venv_eval
venv_eval/bin/pip install rouge-score jiwer scipy huggingface_hub  # + anthropic, only if you ever need to regenerate references
```

## Steps (for you)

1. **Get the data** (lightweight venv):
   ```bash
   venv_eval/bin/python3 download_chunks.py --output-dir data
   ```
   This fetches `subtitle.zip` (7.5MB) + the metadata parquet (405KB) - not
   the video files - and extracts the subtitles matching
   `chunk_1-7_video_ids.txt` into `data/subtitles/`.

   **266, not 317**: of the 317 videos in chunks 1-7, only 266 actually have
   a subtitle file on YouTube (51 don't have captions at all). The script
   reports this count when it runs - it's expected, not a bug. Everything
   downstream (including the already-generated `claude_references.jsonl`)
   is scoped to those 266.

2. **Generate Gemini summaries** (full backend venv, needs `GEMINI_API_KEY`):
   ```bash
   cd backend
   export GEMINI_API_KEY=...   # or set it in backend/.env
   venv/bin/python3 eval/generate_summaries.py --batch-size 25
   ```
   Defaults to `--subs-dir eval/data/subtitles` and
   `--output eval/results/gemini_summaries.jsonl`, so no flags are strictly
   required if you ran step 1 as shown. It's resumable - if it dies partway
   (rate limit, crash), rerun with `--resume` and it'll skip video_ids
   already in the output file.

3. **Score it** (lightweight venv):
   ```bash
   cd backend/eval
   venv_eval/bin/python3 compute_metrics.py
   ```
   Defaults to `--generated results/gemini_summaries.jsonl` and
   `--references results/claude_references.jsonl` - again, no flags needed
   if you used the defaults above. Writes `results/metrics_report.json` and
   prints mean/std/95% CI for WER and ROUGE-1/2/L to the terminal.

   A p-value only gets printed if you pass `--compare-generated <second run>`
   - with one condition there's nothing to test significance against, so by
   default you get descriptive stats + confidence intervals only.

## On "batch size" and the context window

You asked whether the batch size fits Gemini's context window - it isn't
actually a context-window question. `--batch-size` (default 25) only
controls how often progress gets logged and the JSONL gets checkpointed; it
does **not** put multiple videos into one prompt. The real pipeline
(`orchestrator.py`) makes one Gemini call per video, and this eval mirrors
that exactly - no multi-video batching exists to check.

Per-call size: `_summarize_with_gemini` slices the retrieved+deduped chunk
text to 14,000 characters (roughly 3,500-4,000 tokens) plus a short KG
relations block and prompt template - call it ~5,000 tokens total per
request, worst case. The model is `gemini-2.5-flash-lite`, whose context
window is **1,048,576 tokens** (with up to 65,535 output tokens) - so a
single call uses well under 1% of the available context regardless of what
`--batch-size` is set to. There's no tuning needed here; `--batch-size` is
free to set based on how often you want a checkpoint, not based on any size
limit.

## If something doesn't match

- `pip install anthropic` (or rouge-score/jiwer) into the **backend** venv
  breaks `fastapi` (anyio version conflict) - always use `venv_eval` for
  those three scripts, never the backend venv.
- If `generate_summaries.py` fails to import `pipeline.*`, you're missing
  something from `backend/requirements.txt` or the `en_core_web_sm` spaCy
  model - it needs the full backend environment, not `venv_eval`.
- If `compute_metrics.py` says a file isn't found, either step 1 or step 2
  above wasn't run yet, or you're running it from a different working
  directory than expected - the default paths are relative to the `eval/`
  folder itself, not your shell's cwd, so they should work regardless of
  where you invoke the script from.
