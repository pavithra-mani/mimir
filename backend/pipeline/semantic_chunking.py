"""Semantic Chunking - Topic-Clustered Greedy Cosine-Drop Algorithm

- Sentence segmentation with word-level timestamp preservation
- SentenceTransformer embeddings (all-MiniLM-L6-v2)
- TF-IDF informativeness score per sentence (content-word density)
- Global K-Means clustering over sentence embeddings (k chosen via silhouette
  score), temporally smoothed, to find real topic-level segments spanning the
  whole transcript — not just adjacent-sentence similarity
- Greedy loop (unchanged): within each topic segment, start a new chunk when
  cosine similarity drops below SIM_THRESHOLD OR token count exceeds
  MAX_CHUNK_TOKENS
- Stores prev/next chunk pointers for temporal context expansion in retrieval
"""

import logging
import json
import os
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.signal import medfilt

logger = logging.getLogger(__name__)

SIM_THRESHOLD = 0.55      # cosine drop below this → start new chunk
MAX_CHUNK_TOKENS = 350    # hard token ceiling per chunk
MIN_CHUNK_TOKENS = 60     # don't finalize a chunk smaller than this — produces topical chunks

# Topic clustering — below this many sentences there isn't enough signal to
# cluster meaningfully; the whole transcript is treated as one topic segment
# (identical to the old flat-greedy behaviour).
MIN_SENTENCES_FOR_CLUSTERING = 12
MIN_SENTENCES_PER_CLUSTER = 15    # bounds how many clusters we'll even try
MAX_CLUSTERS = 10
MEDFILT_KERNEL = 5                # same smoothing window used for keyframe cluster labels


class SemanticChunker:
    """Topic-clustered, informativeness-aware semantic chunker."""

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
            progress_callback("Scoring sentence informativeness...", 45)

        informativeness = self._compute_informativeness([s["text"] for s in sentences])
        for sent, score in zip(sentences, informativeness):
            sent["informativeness"] = score

        if progress_callback:
            progress_callback("Clustering sentences into topic segments...", 55)

        labels = self._cluster_sentences(embeddings)
        runs = self._runs_from_labels(labels)
        runs = self._merge_small_runs(runs, sentences)
        logger.info(
            f"Topic clustering: {len(sentences)} sentences -> {len(runs)} topic segment(s)"
        )

        if progress_callback:
            progress_callback("Greedy cosine-drop chunking within topic segments...", 60)

        chunks: List[Dict[str, Any]] = []
        chunk_idx = 0
        for run in runs:
            seg_sentences = sentences[run["start"]:run["end"]]
            seg_embeddings = embeddings[run["start"]:run["end"]]
            seg_chunks = self._greedy_chunk(
                seg_sentences, seg_embeddings, video_id, start_idx=chunk_idx
            )
            # _greedy_chunk always flushes its final buffer regardless of size — fine
            # when it runs once over the whole transcript (only the very last chunk
            # could ever be undersized), but now it runs once per topic segment, so
            # an undersized tail could appear at every segment boundary. Fold it back
            # into the previous chunk in the same segment when that happens.
            while len(seg_chunks) > 1 and seg_chunks[-1]["token_count"] < MIN_CHUNK_TOKENS:
                last = seg_chunks.pop()
                seg_chunks[-1] = self._merge_two_chunks(seg_chunks[-1], last)
            for c in seg_chunks:
                c["cluster_id"] = run["cluster_id"]
            chunks.extend(seg_chunks)
            chunk_idx += len(seg_chunks)

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

    def _compute_informativeness(self, texts: List[str]) -> List[float]:
        """Per-sentence informativeness: mean TF-IDF weight of its content words,
        min-max normalized across the transcript. Low score = generic/filler
        ("okay so yeah"), high score = specific/content-dense. Exposed as chunk
        metadata and used to decide which neighbor absorbs an undersized topic
        segment (see _merge_small_runs) — content-rich segments anchor merges,
        filler-heavy ones get folded in rather than the reverse.
        """
        if len(texts) < 2:
            return [1.0] * len(texts)
        try:
            vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
            matrix = vectorizer.fit_transform(texts)
        except ValueError:
            return [1.0] * len(texts)

        scores = []
        for i in range(matrix.shape[0]):
            row = matrix.getrow(i).data
            scores.append(float(row.mean()) if len(row) else 0.0)

        lo, hi = min(scores), max(scores)
        if hi - lo < 1e-9:
            return [1.0] * len(texts)
        return [(s - lo) / (hi - lo) for s in scores]

    def _cluster_sentences(self, embeddings: np.ndarray) -> np.ndarray:
        """Global K-Means over sentence embeddings (k chosen via silhouette score),
        temporally smoothed. Finds real topic segments spanning the whole transcript
        instead of only reacting to adjacent-sentence similarity drops.
        """
        n = embeddings.shape[0]
        if n < MIN_SENTENCES_FOR_CLUSTERING:
            return np.zeros(n, dtype=int)

        max_k = min(MAX_CLUSTERS, n // MIN_SENTENCES_PER_CLUSTER)
        if max_k < 2:
            return np.zeros(n, dtype=int)

        best_k, best_labels, best_score = 1, np.zeros(n, dtype=int), -1.0
        for k in range(2, max_k + 1):
            try:
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = km.fit_predict(embeddings)
                score = silhouette_score(embeddings, labels)
            except Exception as e:
                logger.warning(f"KMeans k={k} failed: {e}")
                continue
            if score > best_score:
                best_k, best_labels, best_score = k, labels, score

        if best_k < 2:
            return np.zeros(n, dtype=int)

        # Temporal smoothing — same fix already used for keyframe cluster-label
        # thrashing (keyframe_extraction.py): raw per-sentence cluster ids flicker
        # between adjacent topics; a median filter removes single-sentence noise
        # without erasing genuine topic shifts.
        kernel = min(MEDFILT_KERNEL, n if n % 2 == 1 else n - 1)
        if kernel >= 3:
            smoothed = medfilt(best_labels.astype(float), kernel_size=kernel).astype(int)
        else:
            smoothed = best_labels

        logger.info(f"Sentence clustering: k={best_k}, silhouette={best_score:.3f}")
        return smoothed

    @staticmethod
    def _runs_from_labels(labels: np.ndarray) -> List[Dict[str, Any]]:
        """Collapse a (smoothed) per-sentence label sequence into contiguous runs."""
        runs = []
        start = 0
        n = len(labels)
        for i in range(1, n + 1):
            if i == n or labels[i] != labels[start]:
                runs.append({"start": start, "end": i, "cluster_id": int(labels[start])})
                start = i
        return runs

    @staticmethod
    def _merge_small_runs(
        runs: List[Dict[str, Any]], sentences: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Fold topic segments too small to form a real chunk into a neighbor.

        Picks whichever neighbor has HIGHER mean informativeness — the merged
        segment anchors on the more content-rich context rather than diluting it.
        """
        runs = list(runs)

        def seg_tokens(run: Dict[str, Any]) -> int:
            return sum(len(sentences[j]["text"].split()) for j in range(run["start"], run["end"]))

        def seg_informativeness(run: Dict[str, Any]) -> float:
            vals = [sentences[j].get("informativeness", 0.0) for j in range(run["start"], run["end"])]
            return sum(vals) / len(vals) if vals else 0.0

        changed = True
        while changed and len(runs) > 1:
            changed = False
            for i, run in enumerate(runs):
                if seg_tokens(run) >= MIN_CHUNK_TOKENS:
                    continue

                prev_i = i - 1 if i > 0 else None
                next_i = i + 1 if i < len(runs) - 1 else None
                if prev_i is not None and next_i is not None:
                    target_i = prev_i if seg_informativeness(runs[prev_i]) >= seg_informativeness(runs[next_i]) else next_i
                else:
                    target_i = prev_i if prev_i is not None else next_i

                target = runs[target_i]
                merged = {
                    "start": min(run["start"], target["start"]),
                    "end": max(run["end"], target["end"]),
                    "cluster_id": target["cluster_id"],
                }
                keep_idx = min(i, target_i)
                drop_idx = max(i, target_i)
                runs = runs[:keep_idx] + [merged] + runs[keep_idx + 1:drop_idx] + runs[drop_idx + 1:]
                changed = True
                break

        return runs

    def _greedy_chunk(
        self,
        sentences: List[Dict[str, Any]],
        embeddings: np.ndarray,
        video_id: str,
        start_idx: int = 0,
    ) -> List[Dict[str, Any]]:
        if not sentences:
            return []

        chunks: List[Dict[str, Any]] = []
        chunk_idx = start_idx

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
        info_scores = [s.get("informativeness", 0.0) for s in sentences]
        return {
            "chunk_id": f"{video_id}_chunk_{idx:04d}",
            "chunk_index": idx,
            "video_id": video_id,
            "text": text,
            "time_start": sentences[0]["time_start"],
            "time_end": sentences[-1]["time_end"],
            "token_count": len(text.split()),
            "sentence_count": len(sentences),
            "informativeness": round(sum(info_scores) / len(info_scores), 4) if info_scores else 0.0,
            "embedding": embedding.tolist(),
            "prev_chunk_id": None,   # filled in after all chunks are built
            "next_chunk_id": None,
        }

    @staticmethod
    def _merge_two_chunks(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        """Merge chunk b into chunk a (a precedes b in time). Keeps a's chunk_id."""
        n_a, n_b = a["sentence_count"], b["sentence_count"]
        total = n_a + n_b
        emb_a = np.array(a["embedding"])
        emb_b = np.array(b["embedding"])
        merged_emb = (n_a * emb_a + n_b * emb_b) / total
        norm = np.linalg.norm(merged_emb)
        if norm > 0:
            merged_emb /= norm

        text = a["text"] + " " + b["text"]
        return {
            **a,
            "text": text,
            "time_end": b["time_end"],
            "token_count": len(text.split()),
            "sentence_count": total,
            "informativeness": round(
                (n_a * a["informativeness"] + n_b * b["informativeness"]) / total, 4
            ),
            "embedding": merged_emb.tolist(),
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
