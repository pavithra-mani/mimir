"""
Ablation study for the cross-modal (visual) channel, on Video-MME multiple-choice QA.

Question: does adding vision (CLIP frame retrieval, a visual knowledge graph linked to
spoken entities, on-screen OCR text) improve answers over the text-only pipeline?
Video-MME tags every question with a task_type (e.g. "OCR Problems", "Counting Problem",
"Attribute Perception"), so gains should concentrate on vision-dependent types and be
~zero on speech-answerable ones - that contrast is the evidence.

Conditions (each toggles an independent config flag, see pipeline/config.py):
    text_only         no visual layer at all (the baseline = existing pipeline)
    visual_retrieval  + CLIP text->frame channel in hybrid retrieval
    visual_graph      + visual/cross-modal nodes & edges in the knowledge graph
    ocr_context       + on-screen text of retrieved chunks' keyframes given to the answerer
    full              all of the above

Pipeline per video (expensive parts run ONCE and are shared by all conditions):
    keyframes (cached) -> chunks -> keyframe<->chunk link -> CLIP+OCR index
    then per condition: rebuild KG/vector store, retrieve per question, answer with Gemini.

Only the final answering step uses the Gemini API (1 call per question x condition);
CLIP and OCR run locally. Budget calls with --limit-videos / --task-types / --conditions.
Resumable: finished (video, question, condition) triples are skipped on --resume.

Needs the VIDEO FILES (not just subtitles):
    python eval/download_chunks.py --output-dir eval/data --with-videos

Usage:
    cd backend
    python eval/ablation_visual.py --videos-dir eval/data/videos --limit-videos 20 \
        --task-types "OCR Problems,Counting Problem,Attribute Perception,Information Synopsis" \
        --conditions text_only,full --resume
    python eval/analyze_ablation.py
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from pipeline.config import config
from pipeline.orchestrator import PipelineOrchestrator
from pipeline.semantic_chunking import SemanticChunker
from pipeline.knowledge_graph import KnowledgeGraph
from pipeline.vector_store import VectorStore
from pipeline.retrieval_engine import RetrievalEngine
from pipeline.visual_index import VisualIndex
from pipeline.keyframe_extraction import KeyframeExtractor
from subtitle_utils import parse_subtitle_to_transcript

HERE = Path(__file__).resolve().parent
VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".avi")

CONDITIONS = {
    #                    index  graph  retrieval ocr_ctx
    "text_only":        (False, False, False, False),
    "visual_retrieval": (True,  False, True,  False),
    "visual_graph":     (True,  True,  False, False),
    "ocr_context":      (True,  False, False, True),
    "full":             (True,  True,  True,  True),
}


def set_condition(name: str) -> None:
    idx, graph, retr, ocr = CONDITIONS[name]
    config.enable_visual_index = idx
    config.enable_visual_graph = graph
    config.enable_visual_retrieval = retr
    config.enable_ocr_in_prompt = ocr


def fmt_ts(sec: float) -> str:
    m, s = divmod(int(sec or 0), 60)
    return f"{m:02d}:{s:02d}"


def build_context(retrieved: list, keyframes: list, use_ocr: bool) -> str:
    """Answerer context: retrieved transcript chunks (time-ordered), plus - only when
    ocr_context is on - the on-screen text / visible entities of keyframes that belong
    to those chunks. Same signals the summariser's visual anchors carry."""
    chunks = sorted(retrieved, key=lambda c: c.get("time_start", 0))
    parts = [f"[{fmt_ts(c['time_start'])}-{fmt_ts(c['time_end'])}] {c['text']}" for c in chunks]
    if use_ocr:
        ids = {c["chunk_id"] for c in chunks}
        for kf in keyframes:
            if kf.get("chunk_id") in ids and (kf.get("ocr_text") or kf.get("grounded_entities")):
                line = f"[frame @ {fmt_ts(kf.get('timestamp', 0))}]"
                if kf.get("ocr_lines"):
                    line += f' on-screen text: "{VisualIndex.fit_lines(kf["ocr_lines"], 300)}"'
                if kf.get("grounded_entities"):
                    line += " visible: " + ", ".join(g["entity"] for g in kf["grounded_entities"])
                parts.append(line)
    return "\n".join(parts)


async def ask_gemini(model, question: str, options: list, context: str, retries: int = 4) -> str:
    prompt = (
        "Answer the multiple-choice question about a video using ONLY the context below "
        "(transcript excerpts, and frame text where given). If the context is insufficient, "
        "pick the most likely option anyway.\n\n"
        f"CONTEXT:\n{context[:12000]}\n\nQUESTION: {question}\n" + "\n".join(options) +
        "\n\nRespond with a single letter (A, B, C or D) and nothing else."
    )
    for attempt in range(retries):
        try:
            resp = await asyncio.to_thread(
                model.generate_content, prompt, generation_config={"temperature": 0.0}
            )
            m = re.search(r"\b([ABCD])\b", resp.text.strip().upper())
            return m.group(1) if m else ""
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "quota" in msg or "rate" in msg:
                await asyncio.sleep(10 * (attempt + 1))
                continue
            print(f"  gemini error: {e}")
            return ""
    return ""


def load_done(path: Path) -> set:
    done = set()
    if path.exists():
        for line in path.open(encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["videoID"], r["question_id"], r["condition"]))
            except Exception:
                pass
    return done


def find_video(videos_dir: Path, video_id: str):
    for ext in VIDEO_EXTS:
        p = videos_dir / f"{video_id}{ext}"
        if p.exists():
            return p
    return None


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos-dir", type=Path, default=HERE / "data" / "videos")
    ap.add_argument("--subs-dir", type=Path, default=HERE / "data" / "subtitles")
    ap.add_argument("--metadata", type=Path, default=HERE / "data" / "metadata.parquet")
    ap.add_argument("--output", type=Path, default=HERE / "results" / "ablation_visual.jsonl")
    ap.add_argument("--conditions", default=",".join(CONDITIONS), help=f"comma list of {list(CONDITIONS)}")
    ap.add_argument("--task-types", default="", help="comma list of Video-MME task_type to keep (default all)")
    ap.add_argument("--limit-videos", type=int, default=None)
    ap.add_argument("--questions-per-video", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=8, help="chunks retrieved per question")
    ap.add_argument("--include-options", action="store_true", help="append answer options to the retrieval query")
    ap.add_argument("--sleep-between", type=float, default=0.0, help="seconds between Gemini calls (free-tier pacing)")
    ap.add_argument("--dry-run", action="store_true", help="build indexes + retrieval, skip Gemini; reports index stats only")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conds:
        if c not in CONDITIONS:
            sys.exit(f"unknown condition {c}")
    if "text_only" not in conds:
        print("WARNING: no text_only baseline in --conditions; analysis needs one for deltas.")

    model = None
    if not args.dry_run:
        key = config.gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            sys.exit("GEMINI_API_KEY not set (backend/.env or env). Use --dry-run to test without it.")
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-2.5-flash-lite")

    df = pd.read_parquet(args.metadata)
    if args.task_types:
        keep = {t.strip() for t in args.task_types.split(",")}
        df = df[df["task_type"].isin(keep)]

    # videos that have both a subtitle file and a video file
    avail = []
    for vid, g in df.groupby("videoID", sort=True):
        sub = next((args.subs_dir / f"{vid}{e}" for e in (".srt", ".vtt") if (args.subs_dir / f"{vid}{e}").exists()), None)
        vp = find_video(args.videos_dir, vid)
        if sub and vp:
            avail.append((vid, sub, vp, g.head(args.questions_per_video)))
    if args.limit_videos:
        avail = avail[: args.limit_videos]
    if not avail:
        sys.exit(f"No videos with both subtitles and video file found (videos-dir={args.videos_dir}).")
    print(f"{len(avail)} videos x {len(conds)} conditions")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stats_path = args.output.with_name(args.output.stem + "_index_stats.jsonl")
    done = load_done(args.output) if args.resume else set()

    o = PipelineOrchestrator()
    o.semantic_chunker = SemanticChunker()
    o.knowledge_graph = KnowledgeGraph(spacy_model=config.spacy_model)
    o.vector_store = VectorStore()
    o.retrieval_engine = RetrievalEngine(spacy_model=config.spacy_model)
    o.visual_index = VisualIndex(config.visual_model)
    extractor = KeyframeExtractor()

    cache_root = args.output.parent / "ablation_cache"
    out_f = args.output.open("a", encoding="utf-8")
    stats_f = stats_path.open("a", encoding="utf-8")

    for n, (vid, sub, vp, qs) in enumerate(avail, 1):
        todo = [(c, q) for c in conds for q in qs.itertuples() if (vid, q.question_id, c) not in done]
        if not todo:
            continue
        t0 = time.time()
        print(f"[{n}/{len(avail)}] {vid}")
        try:
            transcript = parse_subtitle_to_transcript(sub)
            if not transcript["segments"]:
                raise ValueError("empty subtitles")

            # keyframes (cached; the 1fps seek pass is the slow part on long videos)
            cdir = cache_root / vid
            cdir.mkdir(parents=True, exist_ok=True)
            kf_cache = cdir / "keyframes.json"
            if kf_cache.exists():
                kfs = json.loads(kf_cache.read_text())
            else:
                kfs = extractor.extract_keyframes(str(vp))["keyframes"]
                kf_cache.write_text(json.dumps(kfs))

            chunks = o.semantic_chunker.chunk_transcript(transcript, vid)
            o._link_keyframes_to_chunks(kfs, chunks)

            # CLIP + OCR once per video, shared by every condition
            o.visual_index.reset()
            config.enable_visual_index = True
            await asyncio.to_thread(o.visual_index.build, kfs, True)
            base_kfs = [dict(k) for k in kfs]   # snapshot (with ocr_text, no graph fields)

            for cond in conds:
                cond_todo = [q for q in qs.itertuples() if (vid, q.question_id, cond) not in done]
                if not cond_todo:
                    continue
                set_condition(cond)
                kf_c = [dict(k) for k in base_kfs]
                # Re-point the shared index rows (embeddings stay put) at THIS condition's
                # keyframe dicts, so graph/grounding writes never leak across conditions.
                by_id = {k["keyframe_id"]: k for k in kf_c}
                o.visual_index.frames = [by_id[f["keyframe_id"]] for f in o.visual_index.frames]

                o.knowledge_graph.reset(); o.vector_store.reset()
                o.knowledge_graph.build_knowledge_graph(chunks, vid)
                o.vector_store.add_chunks(chunks, compute_embeddings=False)
                vstats, vidx = await o._build_visual_layer(
                    kf_c, chunks, str(cdir / "idx"), vid, reuse_built=True
                )
                o.retrieval_engine.initialize(o.knowledge_graph, o.vector_store, chunks, visual_index=vidx)
                stats_f.write(json.dumps({
                    "videoID": vid, "condition": cond, "visual": vstats,
                    "kg_nodes": o.knowledge_graph.graph.number_of_nodes(),
                    "kg_edges": o.knowledge_graph.graph.number_of_edges(),
                    "n_chunks": len(chunks), "n_keyframes": len(kf_c),
                }) + "\n"); stats_f.flush()

                for q in cond_todo:
                    query = q.question + (" " + " ".join(q.options) if args.include_options else "")
                    retrieved = await o.retrieval_engine.retrieve(query, top_k_final=args.top_k, topic_mode=True)
                    ctx = build_context(retrieved, kf_c, config.enable_ocr_in_prompt)
                    pred = ""
                    if model is not None:
                        pred = await ask_gemini(model, q.question, list(q.options), ctx)
                        if args.sleep_between:
                            await asyncio.sleep(args.sleep_between)
                    rec = {
                        "videoID": vid, "question_id": q.question_id, "task_type": q.task_type,
                        "duration": q.duration, "condition": cond, "predicted": pred, "answer": q.answer,
                        "correct": bool(pred) and pred == q.answer,
                        "n_retrieved": len(retrieved),
                        "n_visual_hits": sum(1 for c in retrieved if c.get("visual_score", 0) > 0),
                        "ctx_chars": len(ctx),
                    }
                    out_f.write(json.dumps(rec) + "\n"); out_f.flush()
            print(f"   done in {time.time() - t0:.0f}s")
        except Exception as e:
            print(f"   FAILED {vid}: {e}")
            traceback.print_exc()

    out_f.close(); stats_f.close()
    print(f"Wrote {args.output}. Next: python eval/analyze_ablation.py")


if __name__ == "__main__":
    asyncio.run(main())
