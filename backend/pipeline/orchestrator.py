"""Pipeline Orchestrator - Coordinates all pipeline phases

Architecture (source of truth: -latest notebook):
  Phase 1 : Transcription (WhisperX) + Keyframe extraction (k-means++ clustering)
  Phase 2 : Dual-channel indexing — Semantic chunks -> Knowledge Graph + FAISS Vector Store
  Phase 3 : Hybrid retrieval (KG + Vector) -> Gemini summarization (retrieved chunks ONLY)
  Phase 4 : Subtitle generation
  Phase 5 : Citation assembly

Two and only two summary outputs:
  results['summary']       — general summary (all main topics)
  results['topic_summary'] — topic-focused summary (only when topic is provided)
Both go through the same hybrid retrieval pipeline before reaching Gemini.
"""

import os
import re
import math
import uuid
import time
import logging
from typing import Optional, Dict, Any, Callable, List
from datetime import datetime
import asyncio

from pipeline.config import config
from pipeline.transcription import TranscriptionEngine, TranscriptionError
from pipeline.keyframe_extraction import KeyframeExtractor, KeyframeExtractionError
from pipeline.summarization import TextSummarizer, SummarizationError
from pipeline.subtitle_generation import SubtitleGenerator, SubtitleError
from pipeline.semantic_chunking import SemanticChunker
from pipeline.knowledge_graph import KnowledgeGraph
from pipeline.vector_store import VectorStore
from pipeline.retrieval_engine import RetrievalEngine
from pipeline.visual_index import VisualIndex

logger = logging.getLogger(__name__)

# Minimum absolute cosine similarity (vector channel) for a topic to be considered
# present in the video. Below this, topic summaries return a standard "no relevant
# content" message instead of forcing Gemini to summarise unrelated chunks.
# "Balanced" default; tunable. NOTE: only vector_score is an absolute signal —
# graph_score is normalised per-batch in retrieval, so it cannot gate relevance.
RELEVANCE_MIN_COSINE = 0.25
# A topic that is only SHOWN (never spoken) has no text cosine; accept it if a keyframe
# matches via CLIP text->image similarity. CLIP cosines run lower than text-text ones and
# are uncalibrated; 0.27 is a conservative starting point, tune on held-out videos.
RELEVANCE_MIN_VISUAL = 0.27


# Spoken-language disfluencies, stripped before text goes to the LLM or the UI.
_FILLER_RE = re.compile(
    r"\b(you know|right|okay|basically|actually|kind of|sort of|I mean)\b[,.]?\s*",
    re.IGNORECASE,
)
_REPEAT_RE = re.compile(r"\b(\w+)(\s+\1\b)+", re.IGNORECASE)


def _clean_speech(text: str) -> str:
    """Remove fillers and stutters ("the the") and normalise whitespace."""
    text = _FILLER_RE.sub("", text)
    text = _REPEAT_RE.sub(r"\1", text)
    return " ".join(text.split())


class PipelineOrchestrator:
    """Orchestrates the complete video processing pipeline."""

    LENGTH_PROFILES = {
        "short":  {"paragraphs": "1-2 paragraphs", "kp_count": 3, "kp_label": "3 key points"},
        "medium": {"paragraphs": "3-5 paragraphs", "kp_count": 5, "kp_label": "5 key points"},
        "long":   {"paragraphs": "6-8 paragraphs", "kp_count": 8, "kp_label": "8 key points"},
    }

    def __init__(self):
        self.device = config.device
        self.transcription_engine: Optional[TranscriptionEngine] = None
        self.keyframe_extractor: Optional[KeyframeExtractor] = None
        self.summarizer: Optional[TextSummarizer] = None
        self.subtitle_generator: Optional[SubtitleGenerator] = None
        self.semantic_chunker: Optional[SemanticChunker] = None
        self.knowledge_graph: Optional[KnowledgeGraph] = None
        self.vector_store: Optional[VectorStore] = None
        self.retrieval_engine: Optional[RetrievalEngine] = None
        self.visual_index: Optional[VisualIndex] = None

    def initialize_components(self):
        logger.info("Initializing pipeline components...")

        if config.enable_transcription:
            try:
                self.transcription_engine = TranscriptionEngine(self.device)
            except Exception as e:
                logger.warning(f"TranscriptionEngine init failed: {e}")

        try:
            self.keyframe_extractor = KeyframeExtractor()
        except Exception as e:
            logger.warning(f"KeyframeExtractor init failed: {e}")

        if config.enable_summarization:
            try:
                self.summarizer = TextSummarizer()
            except Exception as e:
                logger.warning(f"TextSummarizer init failed: {e}")

        if config.enable_subtitles:
            try:
                self.subtitle_generator = SubtitleGenerator()
            except Exception as e:
                logger.warning(f"SubtitleGenerator init failed: {e}")

        # RAG components always initialised (they are the core pipeline)
        try:
            self.semantic_chunker = SemanticChunker()
            self.knowledge_graph = KnowledgeGraph(spacy_model=config.spacy_model)
            self.vector_store = VectorStore()
            self.retrieval_engine = RetrievalEngine(spacy_model=config.spacy_model)
            self.visual_index = VisualIndex(config.visual_model)   # model loads lazily on first build
        except Exception as e:
            logger.warning(f"RAG components init failed: {e}")

        logger.info("Pipeline components initialised.")

    # ------------------------------------------------------------------ #
    # Main entry point                                                     #
    # ------------------------------------------------------------------ #

    async def process_video(
        self,
        video_path: str,
        options: Dict[str, Any],
        progress_callback: Optional[Callable[[str, int], None]] = None,
    ) -> Dict[str, Any]:
        try:
            if progress_callback:
                progress_callback("Initializing pipeline...", 5)

            if not any([
                self.transcription_engine, self.keyframe_extractor,
                self.semantic_chunker, self.retrieval_engine,
            ]):
                self.initialize_components()

            results: Dict[str, Any] = {}
            pipeline_logs: Dict[str, Any] = {}

            # ── Phase 1: Transcription + Keyframe Extraction (parallel) ── #
            t0 = time.time()

            if progress_callback:
                progress_callback("Transcribing + extracting keyframes (parallel)...", 10)

            run_transcription = (
                options.get("transcript", True) and self.transcription_engine is not None
            )
            run_keyframes = self.keyframe_extractor is not None

            if run_transcription and run_keyframes:
                # Both available — run concurrently (they use different resources)
                transcript, keyframe_data = await asyncio.gather(
                    self._run_transcription(video_path, progress_callback),
                    self._run_keyframe_extraction(video_path, progress_callback, transcript=None),
                )
            elif run_transcription:
                transcript = await self._run_transcription(video_path, progress_callback)
                keyframe_data = {}
            elif run_keyframes:
                transcript = None
                keyframe_data = await self._run_keyframe_extraction(
                    video_path, progress_callback, transcript=None
                )
            else:
                transcript = None
                keyframe_data = {}

            results["transcript"] = transcript
            results["keyframes"] = keyframe_data if keyframe_data else None

            phase1_time = time.time() - t0
            pipeline_logs["phase1"] = {
                "transcript_segments": len(transcript["segments"]) if transcript else 0,
                "total_frames_at_1fps": keyframe_data.get("total_frames_at_1fps", 0),
                "cluster_k": keyframe_data.get("cluster_k", 0),
                "scene_count": keyframe_data.get("scene_count", 0),
                "keyframes_selected": keyframe_data.get("extracted_count", 0),
                "frame_scores_sample": keyframe_data.get("frame_scores_sample", []),
                "scene_segments": keyframe_data.get("scene_segments", []),
                "timing_sec": round(phase1_time, 2),
            }

            # ── Phase 2: Dual-channel indexing ───────────────────────── #
            if self.semantic_chunker and self.knowledge_graph and self.vector_store and self.retrieval_engine:
                if progress_callback:
                    progress_callback("Building semantic chunks...", 40)
                t1 = time.time()

                # Reset per-task state so uploads don't bleed into each other
                self.vector_store.reset()
                self.knowledge_graph.reset()

                video_id = self._derive_video_id(video_path, options)
                index_dir = os.path.join(config.upload_dir, video_id, "index")
                chunks_dir = os.path.join(config.upload_dir, video_id, "chunks")

                semantic_chunks = self.semantic_chunker.chunk_transcript(
                    transcript or {}, video_id
                )
                try:
                    self.semantic_chunker.save_chunks(semantic_chunks, chunks_dir, video_id)
                except Exception as e:
                    logger.warning(f"save_chunks failed (non-fatal): {e}")

                # Link each keyframe to the semantic chunk that spans its timestamp, so
                # keyframe descriptions + summaries can reference the transcript portion.
                anchor_candidates = self._link_keyframes_to_chunks(
                    keyframe_data.get("keyframes", []) if keyframe_data else [],
                    semantic_chunks,
                    transcript,
                )

                if progress_callback:
                    progress_callback("Building knowledge graph (spaCy NER + SVO)...", 52)
                kg = self.knowledge_graph.build_knowledge_graph(semantic_chunks, video_id)
                try:
                    self.knowledge_graph.save_graph(index_dir, video_id)
                except Exception as e:
                    logger.warning(f"save_graph failed (non-fatal): {e}")

                if progress_callback:
                    progress_callback("Building FAISS vector store...", 63)
                # Use pre-computed embeddings from SemanticChunker — avoids a second
                # SentenceTransformer pass and eliminates the most common [Errno 22] path.
                self.vector_store.add_chunks(semantic_chunks, compute_embeddings=False)
                try:
                    self.vector_store.save(index_dir, video_id)
                except Exception as e:
                    logger.warning(f"vector_store.save failed (non-fatal): {e}")

                # Visual channel (local CLIP + OCR): embed keyframes, add the visual
                # modality to the KG, link on-screen/visible entities to spoken ones.
                kf_list = keyframe_data.get("keyframes", []) if keyframe_data else []
                visual_stats, visual_for_retrieval = await self._build_visual_layer(
                    kf_list, semantic_chunks, index_dir, video_id, progress_callback
                )

                # Short keyframe titles via ONE batched LLM call, in the background; awaited
                # after summarisation so it never blocks the main path.
                label_task = (
                    asyncio.create_task(self._label_keyframes(kf_list)) if kf_list else None
                )

                self.retrieval_engine.initialize(
                    self.knowledge_graph, self.vector_store, semantic_chunks,
                    visual_index=visual_for_retrieval,
                )

                # KG sample edges for visualisation
                kg_sample_edges: List[Dict] = []
                if self.knowledge_graph.graph:
                    sorted_edges = sorted(
                        self.knowledge_graph.graph.edges(data=True),
                        key=lambda e: e[2].get("weight", 1),
                        reverse=True,
                    )[:40]
                    for src, dst, data in sorted_edges:
                        kg_sample_edges.append(
                            {
                                "source": str(src),
                                "target": str(dst),
                                "relation": str(
                                    data.get("predicates", [""])[0]
                                    if data.get("predicates")
                                    else ""
                                ),
                                "weight": data.get("weight", 1),
                            }
                        )

                kg_nodes = kg.number_of_nodes()
                kg_edges = kg.number_of_edges()
                vs_size = self.vector_store.index.ntotal if self.vector_store.index else 0

                phase2_time = time.time() - t1
                pipeline_logs["phase2"] = {
                    "total_chunks": len(semantic_chunks),
                    "kg_nodes": kg_nodes,
                    "kg_edges": kg_edges,
                    "kg_sample_edges": kg_sample_edges,
                    "vector_index_size": vs_size,
                    "chunk_sizes": [c["token_count"] for c in semantic_chunks],
                    "visual": visual_stats,
                    "timing_sec": round(phase2_time, 2),
                }

                results["rag"] = {
                    "video_id": video_id,
                    "num_chunks": len(semantic_chunks),
                    "kg_nodes": kg_nodes,
                    "kg_edges": kg_edges,
                    "vector_index_size": vs_size,
                    "retrieval_ready": True,
                    "visual": visual_stats,
                }

                # ── Phase 3: Hybrid retrieval + Summarization ─────────── #
                if progress_callback:
                    progress_callback("Hybrid retrieval -> General summary...", 70)
                t2 = time.time()

                summary_length = options.get("summary_length", "medium")
                general_summary = await self._run_summarization_via_retrieval(
                    self.retrieval_engine,
                    topic=None,
                    summary_length=summary_length,
                    progress_callback=progress_callback,
                    anchor_candidates=anchor_candidates,
                )
                results["summary"] = general_summary

                topic = options.get("topic") or None
                if topic:
                    if progress_callback:
                        progress_callback(f"Hybrid retrieval -> Topic summary: {topic[:30]}...", 80)
                    topic_summary = await self._run_summarization_via_retrieval(
                        self.retrieval_engine,
                        topic=topic,
                        summary_length=summary_length,
                        progress_callback=progress_callback,
                        anchor_candidates=anchor_candidates,
                    )
                    results["topic_summary"] = topic_summary
                else:
                    results["topic_summary"] = None

                phase3_time = time.time() - t2
                # Dual-channel coverage stats per channel
                gen_scores = general_summary.get("retrieval_scores", [])
                gen_graph_hits = sum(1 for s in gen_scores if s.get("graph_score", 0) > 0)
                gen_vector_hits = sum(1 for s in gen_scores if s.get("vector_score", 0) > 0)
                gen_both_hits = sum(
                    1 for s in gen_scores
                    if s.get("graph_score", 0) > 0 and s.get("vector_score", 0) > 0
                )
                topic_scores = (
                    results["topic_summary"].get("retrieval_scores", [])
                    if results["topic_summary"] else []
                )
                topic_graph_hits = sum(1 for s in topic_scores if s.get("graph_score", 0) > 0)
                topic_vector_hits = sum(1 for s in topic_scores if s.get("vector_score", 0) > 0)

                pipeline_logs["phase3"] = {
                    "general_query": "summarize main topics, key insights, and important content",
                    "general_retrieved_count": general_summary.get("retrieved_chunk_count", 0),
                    "general_graph_hits": gen_graph_hits,
                    "general_vector_hits": gen_vector_hits,
                    "general_both_hits": gen_both_hits,
                    "general_retrieval_scores": gen_scores,
                    "general_retrieved_chunks": general_summary.get("retrieved_chunks", []),
                    "topic": topic,
                    "topic_retrieved_count": (
                        results["topic_summary"].get("retrieved_chunk_count", 0)
                        if results["topic_summary"] else 0
                    ),
                    "topic_graph_hits": topic_graph_hits,
                    "topic_vector_hits": topic_vector_hits,
                    "general_top_k": general_summary.get("retrieval_top_k"),
                    "topic_top_k": (
                        results["topic_summary"].get("retrieval_top_k")
                        if results["topic_summary"] else None
                    ),
                    "keyframe_refs": {
                        "total_keyframes": len(kf_list),
                        "hosting_chunks": len(anchor_candidates),
                        "general": self._ref_log(general_summary),
                        "topic": self._ref_log(results["topic_summary"]),
                    },
                    "timing_sec": round(phase3_time, 2),
                }

                if label_task is not None:
                    try:
                        pipeline_logs["phase3"]["keyframe_labels"] = await label_task
                    except Exception as e:
                        logger.warning(f"keyframe label task failed (non-fatal): {e}")

                # Citations from general retrieval
                results["citations"] = self._build_citations(
                    general_summary.get("retrieval_scores", []),
                    keyframe_data,
                )

            else:
                results["rag"] = None
                results["summary"] = {"summary": "RAG components unavailable.", "key_points": []}
                results["topic_summary"] = None
                results["citations"] = []
                pipeline_logs["phase2"] = {"error": "RAG components not initialised"}
                pipeline_logs["phase3"] = {"error": "RAG components not initialised"}

            # ── Phase 4: Subtitle generation ─────────────────────────── #
            if options.get("subtitles", True) and self.subtitle_generator:
                if progress_callback:
                    progress_callback("Generating subtitles...", 90)
                subtitle_result = await self._run_subtitle_generation(
                    results["transcript"],
                    options.get("download_format", "srt"),
                    progress_callback,
                )
                results["subtitles"] = subtitle_result
            else:
                results["subtitles"] = None

            results["pipeline_logs"] = pipeline_logs

            if progress_callback:
                progress_callback("Pipeline complete!", 100)

            return results

        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            raise PipelineError(f"Pipeline processing failed: {e}")

    # ------------------------------------------------------------------ #
    # Phase runners                                                        #
    # ------------------------------------------------------------------ #

    async def _run_transcription(
        self,
        video_path: str,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        # transcribe_video is sync (CPU/GPU bound). Run in a worker thread so it doesn't
        # block the event loop AND can run concurrently with keyframe extraction.
        try:
            return await asyncio.to_thread(
                self.transcription_engine.transcribe_video,
                video_path,
                progress_callback,
            )
        except TranscriptionError as e:
            logger.error(f"Transcription failed: {e}")
            raise

    async def _run_keyframe_extraction(
        self,
        video_path: str,
        progress_callback: Optional[Callable] = None,
        transcript: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            return await asyncio.to_thread(
                self.keyframe_extractor.extract_keyframes,
                video_path,
                progress_callback,
                transcript,
            )
        except KeyframeExtractionError as e:
            logger.error(f"Keyframe extraction failed: {e}")
            return {}

    async def _run_subtitle_generation(
        self,
        transcript: Optional[Dict[str, Any]],
        format_type: str,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        try:
            gen = self.subtitle_generator.generate_subtitles(
                transcript, format_type, progress_callback
            )
            # Support both sync and async subtitle generators
            if asyncio.iscoroutine(gen):
                return await gen
            return gen
        except SubtitleError as e:
            logger.error(f"Subtitle generation failed: {e}")
            raise

    async def _build_visual_layer(
        self,
        kf_list: List[Dict[str, Any]],
        semantic_chunks: List[Dict[str, Any]],
        index_dir: str,
        video_id: str,
        progress_callback: Optional[Callable] = None,
        reuse_built: bool = False,
    ):
        """Build the cross-modal layer. Returns (visual_stats, visual_index_or_None).

        Each stage is gated by its own config flag (enable_visual_index / _graph /
        _retrieval) so the ablation harness can switch components independently.
        Failures degrade to the text-only pipeline. reuse_built=True skips re-embedding
        when the index for these keyframes is already built (ablation reuse).
        """
        visual_stats: Dict[str, Any] = {"enabled": False}
        if not (config.enable_visual_index and self.visual_index and kf_list):
            return visual_stats, None
        tv = time.time()
        try:
            if not (reuse_built and self.visual_index.ready):
                if progress_callback:
                    progress_callback("Building visual index (CLIP + OCR)...", 66)
                self.visual_index.reset()
                await asyncio.to_thread(self.visual_index.build, kf_list, True)
                if self.visual_index.ready:
                    try:
                        self.visual_index.save(index_dir, video_id)
                    except Exception as e:
                        logger.warning(f"visual_index.save failed (non-fatal): {e}")
            if not self.visual_index.ready:
                return visual_stats, None
            for kf in kf_list:
                kf["ocr_display"] = self._ocr_display(kf)
            visual_stats = {
                "enabled": True,
                "frames_indexed": len(self.visual_index.frames),
                "frames_with_ocr": sum(1 for k in kf_list if k.get("ocr_text")),
            }
            if config.enable_visual_graph:
                visual_stats["graph"] = await asyncio.to_thread(
                    self.knowledge_graph.add_visual_entities,
                    kf_list, semantic_chunks, self.visual_index,
                    config.visual_ground_min_sim, config.visual_ground_top_k,
                )
                try:
                    self.knowledge_graph.save_graph(index_dir, video_id)
                except Exception as e:
                    logger.warning(f"save_graph (visual) failed (non-fatal): {e}")
            visual_stats["timing_sec"] = round(time.time() - tv, 2)
            return visual_stats, (self.visual_index if config.enable_visual_retrieval else None)
        except Exception as e:
            logger.warning(f"Visual indexing failed (non-fatal, text-only fallback): {e}")
            return {"enabled": False, "error": str(e)}, None

    # ------------------------------------------------------------------ #
    # Core: retrieval-based summarisation                                  #
    # ------------------------------------------------------------------ #

    async def _run_summarization_via_retrieval(
        self,
        retrieval_engine: RetrievalEngine,
        topic: Optional[str] = None,
        summary_length: str = "medium",
        progress_callback: Optional[Callable] = None,
        anchor_candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Summarise using ONLY hybrid-retrieved chunks — never the full transcript.

        anchor_candidates (optional): keyframe-hosting chunks from
        _link_keyframes_to_chunks. When given, up to MAX_VISUAL_ANCHORS of them are
        added to the context and the LLM is asked to reference them as "Keyframe N".
        Omitting it (e.g. the eval harness) leaves behaviour unchanged.
        """
        query = (
            topic
            if topic
            else "summarize the main topics, key insights, and important content"
        )

        is_topic_query = bool(topic)
        top_k = self._retrieval_budget(len(retrieval_engine.semantic_chunks), is_topic_query)
        logger.info(
            f"Retrieval budget: top_k={top_k} for {len(retrieval_engine.semantic_chunks)} chunks "
            f"({'topic' if is_topic_query else 'general'})"
        )
        try:
            retrieved_chunks = await retrieval_engine.retrieve(
                query,
                top_k_final=top_k,
                topic_mode=is_topic_query,
            )
        except Exception as e:
            logger.warning(f"Retrieval failed ({e}); using empty context")
            retrieved_chunks = []

        # Relevance gate (topic queries only): if the topic is essentially absent from
        # the video, return a standard message instead of a forced/hallucinated summary.
        # General summaries are always about the video itself and are never gated.
        if is_topic_query:
            best_vec = max(
                (c.get("vector_score", 0.0) for c in retrieved_chunks), default=0.0
            )
            best_visual = max(
                (c.get("visual_score", 0.0) for c in retrieved_chunks), default=0.0
            )
            if not retrieved_chunks or (
                best_vec < RELEVANCE_MIN_COSINE and best_visual < RELEVANCE_MIN_VISUAL
            ):
                logger.info(
                    f"Topic '{topic}' gated as off-topic "
                    f"(best cosine={best_vec:.3f} < {RELEVANCE_MIN_COSINE}); "
                    f"returning standard no-content message"
                )
                return {
                    "summary": "There is no matching or relevant content to summarize.",
                    "key_points": [],
                    "summary_type": "no_content",
                    "model": "relevance_gate",
                    "summary_length": summary_length,
                    "word_count": 0,
                    "retrieved_chunk_count": len(retrieved_chunks),
                    "retrieval_scores": [],
                    "retrieved_chunks": [],
                    "topic": topic,
                }

        # Sort by timestamp for coherent context
        sorted_chunks = sorted(
            retrieved_chunks, key=lambda c: c.get("time_start", 0)
        )

        # Visual anchors: keyframe-hosting chunks. General summary -> most informative
        # ones; topic summary -> only those the topic retrieval already surfaced, so
        # keyframe references stay on-topic. Anchor chunks missing from retrieval are
        # added to the LLM context only (not to retrieval_scores/diagnostics).
        anchors = self._select_visual_anchors(
            anchor_candidates or [],
            {c.get("chunk_id") for c in sorted_chunks} if is_topic_query else None,
        )
        retrieved_ids = {c.get("chunk_id") for c in sorted_chunks}
        extra_chunks = [
            a["chunk"] for a in anchors if a["chunk"].get("chunk_id") not in retrieved_ids
        ]
        context_chunks = sorted(
            sorted_chunks + extra_chunks, key=lambda c: c.get("time_start", 0)
        )

        # Strip fillers before sending to Gemini, then fit the context to a budget that
        # scales with k. If over budget, drop the LOWEST-scoring chunks (never the tail of
        # the video, which a plain character slice would silently cut) and protect the
        # visual-anchor chunks.
        max_chars = min(
            self.CONTEXT_CHARS_MAX,
            max(self.CONTEXT_CHARS_MIN, top_k * self.CHARS_PER_CHUNK_BUDGET),
        )
        anchor_ids = {a["chunk"].get("chunk_id") for a in anchors}
        cleaned = {
            id(c): _FILLER_RE.sub("", c["text"]).strip() for c in context_chunks
        }
        total_chars = sum(len(t) for t in cleaned.values())
        dropped = 0
        if total_chars > max_chars:
            droppable = sorted(
                (c for c in context_chunks if c.get("chunk_id") not in anchor_ids),
                key=lambda c: c.get("final_score", 0.0),
            )
            drop_ids = set()
            for c in droppable:
                if total_chars <= max_chars:
                    break
                total_chars -= len(cleaned[id(c)])
                drop_ids.add(id(c))
            context_chunks = [c for c in context_chunks if id(c) not in drop_ids]
            dropped = len(drop_ids)
            logger.info(f"Context over budget ({max_chars} chars): dropped {dropped} lowest-scoring chunks")
        context_text = " ".join(cleaned[id(c)] for c in context_chunks)

        # Surface actual KG relations behind these same chunks — gives Gemini
        # structured facts that are provably grounded in the retrieved text
        # (never relations from chunks outside the current context).
        relations_text = ""
        if self.knowledge_graph:
            chunk_ids = {c.get("chunk_id", "") for c in context_chunks if c.get("chunk_id")}
            relations = self.knowledge_graph.get_relations_for_chunks(chunk_ids)
            if relations:
                relations_text = "\n".join(
                    f"- {r['subject']} {r['predicate']} {r['object']}" for r in relations
                )

        # Dual-channel diagnostics — confirm BOTH KG and vector contributed signal.
        n_graph = sum(1 for c in sorted_chunks if c.get("graph_score", 0) > 0)
        n_vector = sum(1 for c in sorted_chunks if c.get("vector_score", 0) > 0)
        n_both = sum(
            1 for c in sorted_chunks
            if c.get("graph_score", 0) > 0 and c.get("vector_score", 0) > 0
        )
        logger.info(
            f"Sending {len(sorted_chunks)} retrieved chunks "
            f"({len(context_text.split())} words) to Gemini for "
            f"{'topic: ' + topic if topic else 'general'} summary | "
            f"dual-channel coverage: graph={n_graph}, vector={n_vector}, "
            f"both={n_both} of {len(sorted_chunks)}"
        )
        if len(sorted_chunks) > 0 and n_graph == 0:
            logger.warning(
                "Knowledge graph contributed ZERO chunks to the LLM context — "
                "dual-channel constraint degraded to vector-only."
            )

        try:
            result = await self._summarize_with_gemini(
                context_text, topic, summary_length, relations_text, anchors, max_chars
            )
        except Exception as e:
            logger.warning(f"Gemini failed ({e}); extractive fallback")
            fallback_max = {"short": 3, "medium": 5, "long": 9}.get(summary_length, 5)
            fallback_text = self._generate_extractive_summary(
                context_text, max_sentences=fallback_max
            )
            result = {
                "summary": fallback_text,
                "key_points": [],
                "summary_type": "extractive_fallback",
                "model": "extractive",
                "word_count": len(fallback_text.split()),
                "summary_length": summary_length,
            }

        cited, invalid = self._keyframe_ref_stats(result, {a["keyframe_number"] for a in anchors})
        result["visual_anchors"] = [
            {
                "keyframe_number": a["keyframe_number"],
                "timestamp": a["timestamp"],
                "chunk_id": a["chunk"].get("chunk_id", ""),
                "time_start": a["chunk"].get("time_start", 0),
                "time_end": a["chunk"].get("time_end", 0),
                "has_ocr": bool(a["keyframe"].get("ocr_text")),
                "grounded_entities": [g["entity"] for g in a["keyframe"].get("grounded_entities", [])],
                "cited": a["keyframe_number"] in cited,
            }
            for a in anchors
        ]
        result["keyframe_refs"] = cited
        result["keyframe_refs_invalid"] = invalid
        result["context_chunks_dropped"] = dropped
        result["retrieval_top_k"] = top_k
        if anchors:
            logger.info(
                f"Keyframe anchors offered={[a['keyframe_number'] for a in anchors]} "
                f"cited={cited} invalid={invalid} "
                f"({'topic' if is_topic_query else 'general'} summary)"
            )
        else:
            logger.info(
                f"No keyframe anchors fed to the {'topic' if is_topic_query else 'general'} summary "
                f"(candidates={len(anchor_candidates or [])})"
            )
        result["retrieved_chunk_count"] = len(sorted_chunks)
        result["retrieval_scores"] = [
            {
                "chunk_id": c.get("chunk_id", ""),
                "score": round(c.get("final_score", 0), 4),
                "vector_score": round(c.get("vector_score", 0), 4),
                "graph_score": round(c.get("graph_score", 0), 4),
                "time_start": c.get("time_start", 0),
                "time_end": c.get("time_end", 0),
            }
            for c in sorted_chunks
        ]
        result["retrieved_chunks"] = [
            {
                "chunk_id": c.get("chunk_id", ""),
                "text": c.get("text", "")[:250],
                "time_start": c.get("time_start", 0),
                "time_end": c.get("time_end", 0),
                "score": round(c.get("final_score", 0), 4),
            }
            for c in sorted_chunks
        ]
        if topic:
            result["topic"] = topic

        return result

    async def _summarize_with_gemini(
        self,
        transcript_text: str,
        topic: Optional[str] = None,
        summary_length: str = "medium",
        relations_text: str = "",
        anchors: Optional[List[Dict[str, Any]]] = None,
        max_chars: int = 120000,
    ) -> Dict[str, Any]:
        import google.generativeai as genai

        api_key = config.gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise Exception("GEMINI_API_KEY not configured")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash-lite")

        profile = self.LENGTH_PROFILES.get(summary_length, self.LENGTH_PROFILES["medium"])
        if topic:
            topic_hint = (
                f'IMPORTANT: Summarize ONLY the content related to: "{topic}". '
                f'Do NOT write a general summary. Focus exclusively on what the video says about "{topic}". '
                f"Ignore content that is not directly about this topic."
            )
        else:
            topic_hint = "Summarize the full content broadly, covering all main themes."
        text_slice = transcript_text[:max_chars]   # safety only; budgeting happens upstream
        bullets_template = "\n".join(
            f"- <key point {i+1}>" for i in range(profile["kp_count"])
        )

        relations_block = (
            f"\n\nEXTRACTED RELATIONS (from the video, for cross-reference only — "
            f"use these to keep entity/relationship details consistent, but do not "
            f"state anything as fact unless it also appears in the transcript excerpts above):\n"
            f"{relations_text}"
            if relations_text
            else ""
        )

        anchors_block = ""
        anchors_rule = ""
        cite_summary = ""
        cite_point = ""
        if anchors:
            lines = []
            for a in anchors:
                c = a["chunk"]
                snippet = " ".join(c.get("text", "").split())[:400]
                line = (
                    f'- Keyframe {a["keyframe_number"]} '
                    f'[{self._fmt_ts(c.get("time_start", 0))}-{self._fmt_ts(c.get("time_end", 0))}]: "{snippet}"'
                )
                if config.enable_ocr_in_prompt:
                    kf = a.get("keyframe", {})
                    if kf.get("ocr_lines"):
                        line += f'\n    on-screen text: "{VisualIndex.fit_lines(kf["ocr_lines"], 300)}"'
                    seen = [g["entity"] for g in kf.get("grounded_entities", [])]
                    if seen:
                        line += f'\n    visible in frame: {", ".join(seen)}'
                lines.append(line)
            anchors_block = (
                "\n\nVISUAL ANCHORS (video keyframes, each paired with the transcript "
                "portion spoken at that point, plus any text read from the frame itself):\n" + "\n".join(lines)
            )
            valid = ", ".join(str(a["keyframe_number"]) for a in anchors)
            anchors_rule = (
                "\nWhere a point in your summary is covered by one of the VISUAL ANCHORS, "
                "state the topic and how it is done/explained, then add a reference in exactly "
                'this form: "(see Keyframe N)" for visual context. Only reference keyframe '
                f"numbers from this list: {valid}. Never invent keyframe numbers; if no anchor "
                "fits a point, give no reference. Aim to cite several different keyframes."
            )
            ex = anchors[0]["keyframe_number"]
            cite_summary = (
                f' Where a sentence is covered by a VISUAL ANCHOR, end it with "(see Keyframe N)", '
                f'for example "...explained step by step (see Keyframe {ex})."'
            )
            cite_point = " Add (see Keyframe N) to a point when an anchor covers it."

        prompt = f"""You are summarizing a video transcript. Write clearly and concisely.

{topic_hint}{anchors_rule}

Only use information explicitly present in the transcript excerpts (and, if given, the extracted relations and the on-screen text / visible items of the visual anchors) below. Do not add facts, names, or claims that are not present there.

TRANSCRIPT EXCERPTS:
{text_slice}{relations_block}{anchors_block}

Respond in exactly this format (no extra text before or after):

SUMMARY:
<{profile["paragraphs"]} summary of the video content. Write in third person. Be specific about what is taught or discussed.{cite_summary}>

KEY POINTS ({profile["kp_label"]}):{cite_point}
{bullets_template}"""

        response = await asyncio.to_thread(model.generate_content, prompt)
        raw = response.text.strip()

        summary = ""
        key_points: List[str] = []

        if "SUMMARY:" in raw and "KEY POINTS" in raw:
            parts = raw.split("KEY POINTS", 1)
            summary = parts[0].replace("SUMMARY:", "").strip()
            bullets_block = parts[1].split("\n", 1)[1] if "\n" in parts[1] else parts[1]
            bullets = bullets_block.strip().split("\n")
            key_points = [
                b.lstrip("- •*").strip()
                for b in bullets
                if b.strip() and b.strip() not in ("-", "•", "*")
            ]
        else:
            summary = raw

        return {
            "summary": summary,
            "key_points": key_points,
            "summary_type": "gemini",
            "model": "gemini-2.5-flash-lite",
            "summary_length": summary_length,
            "word_count": len(summary.split()),
        }

    # ------------------------------------------------------------------ #
    # Keyframe <-> chunk linking                                           #
    # ------------------------------------------------------------------ #

    MAX_VISUAL_ANCHORS = 6
    CHARS_PER_CHUNK_BUDGET = 2500      # LLM context allowance per retrieved chunk
    CONTEXT_CHARS_MIN = 24000
    CONTEXT_CHARS_MAX = 120000         # safety ceiling (flash-lite handles far more)
    KEYFRAME_EXCERPT_WINDOW = 6.0      # +/- seconds of speech shown beside a keyframe
    KEYFRAME_EXCERPT_CHARS = 220

    @staticmethod
    def _retrieval_budget(n_chunks: int, is_topic: bool) -> int:
        """How many chunks to retrieve: a floor for short videos, growing with video
        length (chunk count), capped so the prompt stays bounded.
        general: clamp(ceil(0.15*n), 15, 60)   topic: clamp(ceil(0.08*n), 8, 30)"""
        if not config.adaptive_retrieval_budget:
            return 15 if is_topic else 30
        frac, lo, hi = (0.08, 8, 30) if is_topic else (0.15, 15, 60)
        return max(lo, min(hi, math.ceil(frac * max(n_chunks, 0))))

    @staticmethod
    def _fmt_ts(sec: float) -> str:
        m, s = divmod(int(sec or 0), 60)
        return f"{m:02d}:{s:02d}"

    def _link_keyframes_to_chunks(
        self,
        keyframes: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]],
        transcript: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Annotate each keyframe (in place) with the chunk spanning its timestamp.

        Adds display_index (1-based, by timestamp), chunk_id, time_start/time_end,
        chunk_text_snippet and a transcript-aware description. Returns one anchor
        candidate per hosting chunk ({keyframe_number, timestamp, chunk}).
        Additive only; no-op without chunks.
        """
        if not keyframes:
            return []
        ordered = sorted(keyframes, key=lambda k: k.get("timestamp", 0))
        for i, kf in enumerate(ordered, start=1):
            kf["display_index"] = i
        if not chunks:
            return []

        candidates: Dict[str, Dict[str, Any]] = {}
        for kf in ordered:
            ts = kf.get("timestamp", 0)

            def dist(c: Dict[str, Any]) -> float:
                if c["time_start"] <= ts <= c["time_end"]:
                    return 0.0
                return min(abs(ts - c["time_start"]), abs(ts - c["time_end"]))

            chunk = min(chunks, key=dist)
            snippet = self._keyframe_excerpt(ts, chunk, transcript)
            rng = f'{self._fmt_ts(chunk["time_start"])}–{self._fmt_ts(chunk["time_end"])}'
            kf["chunk_id"] = chunk.get("chunk_id", "")
            kf["time_start"] = chunk["time_start"]
            kf["time_end"] = chunk["time_end"]
            kf["chunk_text_snippet"] = snippet
            kf["description"] = f"Visual context for {rng}: {snippet}" if snippet else (
                f"Visual context for {rng}"
            )
            # all keyframes of a chunk are kept; the representative one is chosen at
            # selection time (after OCR is known)
            candidates.setdefault(kf["chunk_id"], {"chunk": chunk, "keyframes": []})["keyframes"].append(kf)
        return list(candidates.values())

    def _keyframe_excerpt(
        self, ts: float, chunk: Dict[str, Any], transcript: Optional[Dict[str, Any]]
    ) -> str:
        """Clean, readable text for what is being SAID around a keyframe.

        Uses the transcript segments within +/- KEYFRAME_EXCERPT_WINDOW seconds of the
        keyframe (not the start of the whole chunk), strips fillers/stutters, and trims
        to whole sentences. Falls back to the cleaned start of the chunk when no
        timed transcript is available.
        """
        raw = ""
        segs = (transcript or {}).get("segments") or []
        if segs:
            w = self.KEYFRAME_EXCERPT_WINDOW
            near = [
                sg.get("text", "") for sg in segs
                if sg.get("end", 0) >= ts - w and sg.get("start", 0) <= ts + w
            ]
            raw = " ".join(near)
        if not raw.strip():
            raw = chunk.get("text", "")
        text = _clean_speech(raw)
        if not text:
            return ""
        sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]
        # drop a leading fragment (segment boundaries cut mid-sentence) when others exist
        if len(sentences) > 1 and sentences[0][0].islower():
            sentences = sentences[1:]
        out = ""
        for sent in sentences:
            if out and len(out) + len(sent) + 1 > self.KEYFRAME_EXCERPT_CHARS:
                break
            out = f"{out} {sent}".strip()
        truncated = len(out) < len(" ".join(sentences))
        if len(out) > self.KEYFRAME_EXCERPT_CHARS:
            out = out[: self.KEYFRAME_EXCERPT_CHARS].rsplit(" ", 1)[0]
            truncated = True
        out = out[0].upper() + out[1:]
        return out + ("…" if truncated and not out.endswith(("…", "...")) else "")

    def _ocr_display(self, kf: Dict[str, Any], max_lines: int = 3, max_chars: int = 150) -> str:
        """A short, readable 'what is on screen' for the UI: the few OCR lines most related
        to what is being said at that moment (word overlap with the spoken excerpt), padded
        by line length, shown in reading order. The full cleaned text stays in ocr_text."""
        lines = kf.get("ocr_lines") or []
        if not lines:
            return ""
        spoken = {
            w for w in re.findall(r"[a-z][a-z0-9_-]{2,}", (kf.get("chunk_text_snippet") or "").lower())
        }
        def score(line: str) -> float:
            words = set(re.findall(r"[a-z][a-z0-9_-]{2,}", line.lower()))
            return 3.0 * len(words & spoken) + min(len(line), 60) / 60.0
        top = sorted(range(len(lines)), key=lambda i: score(lines[i]), reverse=True)[:max_lines]
        picked = [lines[i] for i in sorted(top)]
        return VisualIndex.fit_lines(picked, max_chars, sep=" · ")

    async def _label_keyframes(self, keyframes: List[Dict[str, Any]]) -> int:
        """ONE batched Gemini call giving each keyframe a short title (<=8 words), from
        its cleaned excerpt + OCR. Best-effort: any failure leaves labels unset and the UI
        falls back to the excerpt. Runs as a background task so it adds no pipeline wait."""
        if not config.enable_keyframe_labels:
            return 0
        items = [k for k in keyframes if k.get("chunk_text_snippet") or k.get("ocr_text")]
        api_key = config.gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not items or not api_key:
            return 0
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash-lite")
            lines = []
            for k in items:
                line = f'{k["display_index"]}. said: "{k.get("chunk_text_snippet", "")}"'
                if k.get("ocr_display"):
                    line += f' | on screen: "{k["ocr_display"]}"'
                lines.append(line)
            prompt = (
                "Give each numbered video moment a short descriptive title (max 8 words, no "
                "quotes, no trailing period) naming what is being discussed or shown. Use only "
                "the information given. Reply with one line per item in the form `N: title`, "
                "nothing else.\n\n" + "\n".join(lines)
            )
            resp = await asyncio.wait_for(asyncio.to_thread(model.generate_content, prompt), timeout=60)
            by_num = {k["display_index"]: k for k in items}
            n = 0
            for line in resp.text.splitlines():
                m = re.match(r"^\s*(\d+)\s*[:.\-]\s*(.+?)\s*$", line)
                if m and int(m.group(1)) in by_num:
                    by_num[int(m.group(1))]["label"] = m.group(2).strip().strip('"*')[:80]
                    n += 1
            logger.info(f"Keyframe labels: {n}/{len(items)} generated")
            return n
        except Exception as e:
            logger.warning(f"Keyframe labelling failed (non-fatal): {e}")
            return 0

    def _ref_log(self, summary: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Compact keyframe-reference diagnostics for the pipeline-logs page."""
        if not summary:
            return None
        offered = summary.get("visual_anchors", [])
        cited = summary.get("keyframe_refs", [])
        return {
            "offered": offered,
            "cited": cited,
            "invalid": summary.get("keyframe_refs_invalid", []),
            "offered_count": len(offered),
            "cited_count": len(cited),
            "cited_ratio": round(len(cited) / len(offered), 3) if offered else 0.0,
            "retrieval_top_k": summary.get("retrieval_top_k"),
            "context_chunks_dropped": summary.get("context_chunks_dropped", 0),
        }

    def _select_visual_anchors(
        self,
        candidates: List[Dict[str, Any]],
        allowed_chunk_ids: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """Pick up to MAX_VISUAL_ANCHORS keyframe-hosting chunks, most informative first,
        optionally restricted to chunk ids (topic mode); returned in time order."""
        pool = [
            a for a in candidates
            if allowed_chunk_ids is None or a["chunk"].get("chunk_id") in allowed_chunk_ids
        ]
        pool.sort(key=lambda a: a["chunk"].get("informativeness", 0.0), reverse=True)
        anchors = []
        for a in pool[: self.MAX_VISUAL_ANCHORS]:
            # representative keyframe: the one with the most on-screen text, else earliest
            kf = max(a["keyframes"], key=lambda k: (len(k.get("ocr_text", "")), -k.get("timestamp", 0)))
            anchors.append({
                "chunk": a["chunk"],
                "keyframe": kf,
                "keyframe_number": kf["display_index"],
                "timestamp": kf.get("timestamp", 0),
            })
        return sorted(anchors, key=lambda a: a["keyframe_number"])

    @staticmethod
    def _keyframe_ref_stats(result: Dict[str, Any], valid: set):
        """(cited, invalid): keyframe numbers the LLM referenced that were offered /
        that were NOT offered (hallucinated or from the wrong summary)."""
        text = result.get("summary", "") + " " + " ".join(result.get("key_points", []))
        found = {int(n) for n in re.findall(r"Keyframes?\s+(\d+)", text)}
        return sorted(found & valid), sorted(found - valid)

    # ------------------------------------------------------------------ #
    # Citation assembly                                                    #
    # ------------------------------------------------------------------ #

    def _build_citations(
        self,
        retrieval_scores: List[Dict[str, Any]],
        keyframe_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        keyframes = keyframe_data.get("keyframes", []) if keyframe_data else []
        citations: List[Dict[str, Any]] = []

        for item in retrieval_scores[:15]:
            ts = item.get("time_start", 0)
            m, s = divmod(int(ts), 60)
            timestamp_label = f"{m:02d}:{s:02d}"

            kf_timestamp = None
            kf_path = None
            best_dist = float("inf")
            for kf in keyframes:
                dist = abs(kf.get("timestamp", 0) - ts)
                if dist < best_dist and dist <= 2.0:
                    best_dist = dist
                    kf_timestamp = kf.get("timestamp")
                    kf_path = kf.get("frame_path")

            citations.append(
                {
                    "chunk_id": item.get("chunk_id", ""),
                    "timestamp_label": timestamp_label,
                    "time_start_sec": ts,
                    "time_end_sec": item.get("time_end", 0),
                    "relevance_score": item.get("score", 0),
                    "keyframe_timestamp": kf_timestamp,
                    "keyframe_path": kf_path,
                }
            )

        return citations

    # ------------------------------------------------------------------ #
    # Utility helpers                                                      #
    # ------------------------------------------------------------------ #

    def _derive_video_id(self, video_path: str, options: Dict[str, Any]) -> str:
        if options.get("video_id"):
            return options["video_id"]
        parts = video_path.replace("\\", "/").split("/")
        return parts[-2] if len(parts) > 1 else f"video_{uuid.uuid4().hex[:8]}"

    def _generate_extractive_summary(self, text: str, max_sentences: int = 5) -> str:
        try:
            sentences = re.split(r"[.!?]+", text)
            sentences = [s.strip() for s in sentences if s.strip()]

            if len(sentences) <= max_sentences:
                return ". ".join(sentences)

            stop_words = {
                "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
                "with", "by", "from", "is", "are", "was", "were", "be", "have",
                "has", "had", "do", "does", "did", "will", "would", "could",
                "should", "may", "might", "must", "can", "this", "that", "these",
                "those", "i", "you", "he", "she", "it", "we", "they", "a", "an",
                "so", "then", "there", "here", "now", "just", "only", "very",
                "really", "also", "too", "even", "well",
            }

            word_freq: Dict[str, int] = {}
            for sent in sentences:
                for word in sent.lower().split():
                    w = word.strip(".,!?;:\"()[]{}").strip()
                    if w and w not in stop_words and len(w) > 2:
                        word_freq[w] = word_freq.get(w, 0) + 1

            scored = []
            for i, sent in enumerate(sentences):
                words = [
                    w.strip(".,!?;:\"()[]{}").lower()
                    for w in sent.split()
                ]
                score = sum(word_freq.get(w, 0) for w in words if w not in stop_words)
                if i < len(sentences) * 0.2:
                    score *= 1.4
                elif i > len(sentences) * 0.8:
                    score *= 1.2
                scored.append((sent, score, i))

            scored.sort(key=lambda x: x[1], reverse=True)
            top = sorted(scored[:max_sentences], key=lambda x: x[2])
            return ". ".join(s[0] for s in top)

        except Exception:
            sentences = re.split(r"[.!?]+", text)
            return ". ".join(s.strip() for s in sentences if s.strip())[:max_sentences]

    def _generate_fallback_summary(self, chunks: List[Dict[str, Any]]) -> str:
        important = []
        for chunk in chunks[:10]:
            text = chunk.get("text", "").strip()
            if len(text) > 20 and not text.startswith(("And", "So", "But")):
                important.append(text)
        if not important:
            important = [c.get("text", "") for c in chunks[:5]]
        summary = " ".join(important[:3])
        return (summary[:300] + "...") if len(summary) > 300 else summary.strip()

    def get_pipeline_status(self) -> Dict[str, Any]:
        return {
            "device": str(self.device),
            "components": {
                "transcription": self.transcription_engine is not None,
                "keyframe_extraction": self.keyframe_extractor is not None,
                "summarization": self.summarizer is not None,
                "subtitle_generation": self.subtitle_generator is not None,
                "rag": self.retrieval_engine is not None,
            },
            "configuration": {
                "use_gpu": config.use_gpu,
                "whisper_model": config.whisper_model,
                "enabled_features": {
                    "transcription": config.enable_transcription,
                    "summarization": config.enable_summarization,
                    "subtitles": config.enable_subtitles,
                    "keyframes": config.enable_keyframes,
                    "rag": getattr(config, "enable_rag", True),
                },
            },
        }


class PipelineError(Exception):
    pass
