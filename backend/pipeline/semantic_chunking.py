"""Semantic Chunking - Greedy Sequential Cosine-Drop Algorithm

Follows the notebook (-latest) Phase 2.1-2.2 exactly:
- Sentence segmentation with word-level timestamp preservation
- SentenceTransformer embeddings (all-MiniLM-L6-v2)
- Greedy loop: start new chunk when cosine similarity drops below SIM_THRESHOLD
  OR token count exceeds MAX_CHUNK_TOKENS
- Stores prev/next chunk pointers for temporal context expansion in retrieval
"""

import logging
import json
import os
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

SIM_THRESHOLD = 0.55      # cosine drop below this → start new chunk
MAX_CHUNK_TOKENS = 350    # hard token ceiling per chunk
MIN_CHUNK_TOKENS = 60     # don't finalize a chunk smaller than this — produces topical chunks


class SemanticChunker:
    """Greedy cosine-drop semantic chunker (VideoRAG notebook implementation)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", **kwargs):
        # Accept legacy params but ignore them
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None

    def initialize(self):
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
        self.model = SentenceTransformer(self.model_name, device=device)
        logger.info(f"Loaded SentenceTransformer: {self.model_name} on {device}")

    def chunk_transcript(
        self,
        transcript: Dict[str, Any],
        video_id: str,
        progress_callback: Optional[callable] = None,
    ) -> List[Dict[str, Any]]:
        if not self.model:
            self.initialize()

        if progress_callback:
            progress_callback("Segmenting transcript into sentences...", 10)

        sentences = self._extract_sentences(transcript)
        if not sentences:
            logger.warning("No sentences extracted from transcript")
            return []

        if progress_callback:
            progress_callback("Computing sentence embeddings...", 30)

        texts = [s["text"] for s in sentences]
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=64,
            show_progress_bar=False,
        )
        # L2 normalise for cosine via dot product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        if progress_callback:
            progress_callback("Greedy cosine-drop chunking...", 60)

        chunks = self._greedy_chunk(sentences, embeddings, video_id)

        # Wire up prev/next pointers
        for i, chunk in enumerate(chunks):
            chunk["prev_chunk_id"] = chunks[i - 1]["chunk_id"] if i > 0 else None
            chunk["next_chunk_id"] = (
                chunks[i + 1]["chunk_id"] if i < len(chunks) - 1 else None
            )

        if progress_callback:
            progress_callback(f"Created {len(chunks)} semantic chunks", 100)

        return chunks

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _extract_sentences(
        self, transcript: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build timestamped sentence list from transcript (word or segment level)."""
        sentences: List[Dict[str, Any]] = []

        # Collect flat word list (word-level timestamps are ideal)
        words: List[Dict] = []
        if "words" in transcript:
            words = transcript["words"]
        elif "segments" in transcript:
            for seg in transcript["segments"]:
                words.extend(seg.get("words", []))

        if words:
            buf: List[Dict] = []
            for w in words:
                buf.append(w)
                text = w.get("word", "").strip()
                if text and text[-1] in ".?!":
                    sent_text = " ".join(
                        cw.get("word", "").strip() for cw in buf
                    ).strip()
                    if sent_text:
                        sentences.append(
                            {
                                "text": sent_text,
                                "time_start": buf[0].get("start", 0),
                                "time_end": buf[-1].get("end", 0),
                            }
                        )
                    buf = []
            # Flush remainder
            if buf:
                sent_text = " ".join(
                    cw.get("word", "").strip() for cw in buf
                ).strip()
                if sent_text:
                    sentences.append(
                        {
                            "text": sent_text,
                            "time_start": buf[0].get("start", 0),
                            "time_end": buf[-1].get("end", 0),
                        }
                    )
        elif "segments" in transcript:
            # Use each Whisper segment as an atomic unit — segments are natural speech
            # pauses and already sentence-like; sub-sentence splitting produces embeddings
            # that are too short and semantically unstable.
            for seg in transcript["segments"]:
                seg_text = seg.get("text", "").strip()
                if not seg_text:
                    continue
                sentences.append(
                    {
                        "text": seg_text,
                        "time_start": seg.get("start", 0),
                        "time_end": seg.get("end", 0),
                    }
                )

        return sentences

    def _greedy_chunk(
        self,
        sentences: List[Dict[str, Any]],
        embeddings: np.ndarray,
        video_id: str,
    ) -> List[Dict[str, Any]]:
        if not sentences:
            return []

        chunks: List[Dict[str, Any]] = []
        chunk_idx = 0

        buf_sentences = [sentences[0]]
        buf_emb = embeddings[0].copy()
        buf_tokens = len(sentences[0]["text"].split())

        for i in range(1, len(sentences)):
            sent = sentences[i]
            emb = embeddings[i]
            tokens = len(sent["text"].split())

            # Cosine similarity between current buffer centroid and next sentence
            sim = float(np.dot(buf_emb, emb))

            if (sim < SIM_THRESHOLD or buf_tokens + tokens > MAX_CHUNK_TOKENS) and buf_tokens >= MIN_CHUNK_TOKENS:
                chunks.append(
                    self._make_chunk(buf_sentences, chunk_idx, video_id, buf_emb)
                )
                chunk_idx += 1
                buf_sentences = [sent]
                buf_emb = emb.copy()
                buf_tokens = tokens
            else:
                buf_sentences.append(sent)
                # Running mean of L2-normalised embeddings (approximate centroid)
                n = len(buf_sentences)
                buf_emb = ((n - 1) * buf_emb + emb) / n
                norm = np.linalg.norm(buf_emb)
                if norm > 0:
                    buf_emb /= norm
                buf_tokens += tokens

        if buf_sentences:
            chunks.append(
                self._make_chunk(buf_sentences, chunk_idx, video_id, buf_emb)
            )

        return chunks

    @staticmethod
    def _make_chunk(
        sentences: List[Dict[str, Any]],
        idx: int,
        video_id: str,
        embedding: np.ndarray,
    ) -> Dict[str, Any]:
        text = " ".join(s["text"] for s in sentences)
        return {
            "chunk_id": f"{video_id}_chunk_{idx:04d}",
            "chunk_index": idx,
            "video_id": video_id,
            "text": text,
            "time_start": sentences[0]["time_start"],
            "time_end": sentences[-1]["time_end"],
            "token_count": len(text.split()),
            "sentence_count": len(sentences),
            "embedding": embedding.tolist(),
            "prev_chunk_id": None,   # filled in after all chunks are built
            "next_chunk_id": None,
        }

    def save_chunks(
        self,
        chunks: List[Dict[str, Any]],
        output_dir: str,
        video_id: str,
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        chunks_path = os.path.join(output_dir, f"{video_id}_chunks.json")
        readable = [
            {k: v for k, v in chunk.items() if k != "embedding"}
            for chunk in chunks
        ]
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(readable, f, indent=2)
        logger.info(f"Saved {len(chunks)} chunks to {chunks_path}")
        return chunks_path
