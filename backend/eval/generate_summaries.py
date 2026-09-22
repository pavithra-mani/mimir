"""
Batch general-summary generation over a Video-MME subset, using the REAL
production pipeline components (SemanticChunker, KnowledgeGraph, VectorStore,
RetrievalEngine, and PipelineOrchestrator._run_summarization_via_retrieval /
_summarize_with_gemini) — only the transcription (ASR) phase is replaced with
the video's existing subtitle file, and keyframe extraction / topic summary
are skipped since this eval only targets the general summary.

Run this on the machine that will do the actual Gemini calls (e.g. your
college GPU box) with the full backend environment installed:
    pip install -r requirements.txt          # from backend/
    python -m spacy download en_core_web_sm  # if not already present
    export GEMINI_API_KEY=...                # or set it in backend/.env

Usage:
    cd backend
    python eval/generate_summaries.py \
        --subs-dir /path/to/extracted_srts \
        --output eval/results/gemini_summaries.jsonl \
        --batch-size 25

Resumable: re-running with --resume skips video_ids already present in
--output, so a crash or rate-limit partway through doesn't lose prior work.
Each video's result is appended and flushed to the JSONL immediately after
its Gemini call returns — a batch is a pacing/logging checkpoint, not a
single combined request (the real pipeline calls Gemini once per video;
there is no multi-video batching in production to mirror).
"""

import argparse
import asyncio
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.config import config
from pipeline.orchestrator import PipelineOrchestrator
from pipeline.semantic_chunking import SemanticChunker
from pipeline.knowledge_graph import KnowledgeGraph
from pipeline.vector_store import VectorStore
from pipeline.retrieval_engine import RetrievalEngine
from subtitle_utils import parse_subtitle_to_transcript


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


async def process_one(orchestrator: PipelineOrchestrator, video_id: str, transcript: dict, summary_length: str) -> dict:
    orchestrator.knowledge_graph.reset()
    orchestrator.vector_store.reset()

    semantic_chunks = orchestrator.semantic_chunker.chunk_transcript(transcript, video_id)
    orchestrator.knowledge_graph.build_knowledge_graph(semantic_chunks, video_id)
    orchestrator.vector_store.add_chunks(semantic_chunks, compute_embeddings=False)
    orchestrator.retrieval_engine.initialize(
        orchestrator.knowledge_graph, orchestrator.vector_store, semantic_chunks
    )

    result = await orchestrator._run_summarization_via_retrieval(
        orchestrator.retrieval_engine, topic=None, summary_length=summary_length,
    )

    return {
        "video_id": video_id,
        "summary": result.get("summary", ""),
        "key_points": result.get("key_points", []),
        "word_count": result.get("word_count", 0),
        "model": result.get("model", ""),
        "summary_type": result.get("summary_type", ""),
        "num_chunks": len(semantic_chunks),
        "retrieved_chunk_count": result.get("retrieved_chunk_count", 0),
    }


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subs-dir", type=Path, default=Path(__file__).resolve().parent / "data" / "subtitles", help="Directory of extracted .srt/.vtt files (default: matches download_chunks.py's output)")
    ap.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results" / "gemini_summaries.jsonl", help="JSONL output path (appended to)")
    ap.add_argument("--batch-size", type=int, default=25, help="Videos per pacing/logging checkpoint")
    ap.add_argument("--summary-length", default="medium", choices=["short", "medium", "long"])
    ap.add_argument("--sleep-between", type=float, default=0.0, help="Seconds to sleep between Gemini calls")
    ap.add_argument("--limit", type=int, default=None, help="Cap the number of videos processed (smoke testing)")
    ap.add_argument("--resume", action="store_true", help="Skip video_ids already present in --output")
    args = ap.parse_args()

    if not config.gemini_api_key and not __import__("os").getenv("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY not set (backend/.env or environment) - aborting before spending nothing, but also doing nothing.")

    sub_files = sorted(list(args.subs_dir.glob("*.srt")) + list(args.subs_dir.glob("*.vtt")))
    if args.limit:
        sub_files = sub_files[: args.limit]
    if not sub_files:
        sys.exit(f"No .srt/.vtt files found in {args.subs_dir}")

    done_ids = load_done_ids(args.output) if args.resume else set()
    if done_ids:
        print(f"Resuming: {len(done_ids)} videos already in {args.output}, skipping those.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    orchestrator = PipelineOrchestrator()
    orchestrator.semantic_chunker = SemanticChunker()
    orchestrator.knowledge_graph = KnowledgeGraph(spacy_model=config.spacy_model)
    orchestrator.vector_store = VectorStore()
    orchestrator.retrieval_engine = RetrievalEngine(spacy_model=config.spacy_model)

    todo = [p for p in sub_files if p.stem not in done_ids]
    print(f"Processing {len(todo)} videos ({len(sub_files)} total, {len(done_ids)} already done)")

    errors = []
    t_start = time.time()
    with args.output.open("a", encoding="utf-8") as out_f:
        for i, path in enumerate(todo, 1):
            video_id = path.stem
            try:
                transcript = parse_subtitle_to_transcript(path)
                if not transcript["segments"]:
                    raise ValueError("no extractable subtitle text")

                record = await process_one(orchestrator, video_id, transcript, args.summary_length)
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()

            except Exception as e:
                errors.append({"video_id": video_id, "error": str(e), "traceback": traceback.format_exc()})
                print(f"  [ERROR] {video_id}: {e}")

            if i % args.batch_size == 0 or i == len(todo):
                elapsed = time.time() - t_start
                print(f"[batch checkpoint] {i}/{len(todo)} done, {len(errors)} errors, {elapsed:.1f}s elapsed")

            if args.sleep_between:
                await asyncio.sleep(args.sleep_between)

    if errors:
        err_path = args.output.with_suffix(".errors.json")
        err_path.write_text(json.dumps(errors, indent=2))
        print(f"\n{len(errors)} videos failed - details in {err_path}")

    print(f"\nDone. Summaries written to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
