"""Visual channel: CLIP embeddings + OCR over selected keyframes.

Everything runs locally (sentence-transformers CLIP, RapidOCR/ONNX) — no API quota.

Per keyframe it produces
  - a normalised CLIP image embedding (kept in a FAISS inner-product index; never
    serialised into task results),
  - OCR text (on-screen words: slides, code, captions, chart labels),
  - "grounded" spoken entities: which entities the speaker mentions around that
    timestamp are actually visible in the frame, scored by CLIP text<->image cosine.

CLIP puts text and images in one space, so a text query can be matched directly
against frames (cross-modal retrieval), and spoken entities can be matched against
the frame they co-occur with (cross-modal grounding).
"""

import logging
import os
import re
import json
import unicodedata
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss
    _FAISS = True
except ImportError:  # pragma: no cover
    _FAISS = False

try:
    from rapidocr_onnxruntime import RapidOCR
    _OCR = True
except ImportError:
    _OCR = False
    logger.warning("rapidocr-onnxruntime not installed; OCR disabled (pip install rapidocr-onnxruntime)")


class VisualIndex:
    """CLIP + OCR index over a video's keyframes."""

    OCR_MAX_CHARS = 600

    def __init__(self, model_name: str = "clip-ViT-B-32"):
        self.model_name = model_name
        self.model = None
        self._ocr = None
        self.index = None
        self.frames: List[Dict[str, Any]] = []   # parallel to index rows
        self.embeddings: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #

    def _ensure_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
            self.model = SentenceTransformer(self.model_name, device=device)
            logger.info(f"Loaded visual model {self.model_name} on {device}")

    def reset(self):
        self.index = None
        self.frames = []
        self.embeddings = None

    @property
    def ready(self) -> bool:
        return self.index is not None and len(self.frames) > 0

    # ------------------------------------------------------------------ #
    # Indexing                                                             #
    # ------------------------------------------------------------------ #

    def build(self, keyframes: List[Dict[str, Any]], run_ocr: bool = True) -> None:
        """Embed + OCR every keyframe (in place: adds ocr_text to each keyframe dict)."""
        from PIL import Image

        self.reset()
        usable = [kf for kf in keyframes if kf.get("frame_path") and os.path.isfile(kf["frame_path"])]
        if not usable:
            return
        self._ensure_model()

        images = []
        for kf in usable:
            with Image.open(kf["frame_path"]) as im:
                images.append(im.convert("RGB").copy())
        emb = self.model.encode(
            images, batch_size=16, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        ).astype(np.float32)

        if run_ocr and _OCR and self._ocr is None:
            self._ocr = RapidOCR()

        # OCR every frame, then filter: per-box quality, status-bar bands, and
        # boilerplate that repeats across frames (UI chrome, watermarks, clocks).
        raw_lines = []
        for kf, img in zip(usable, images):
            raw_lines.append(
                self._ocr_lines(kf["frame_path"], img.height) if (run_ocr and self._ocr) else []
            )
        kept = self._drop_boilerplate(raw_lines)
        for kf, lines in zip(usable, kept):
            kf["ocr_lines"] = lines
            kf["ocr_text"] = self.fit_lines(lines, self.OCR_MAX_CHARS, sep=" . ")

        self.frames = usable
        self.embeddings = emb
        if _FAISS:
            self.index = faiss.IndexFlatIP(emb.shape[1])
            self.index.add(emb)
        else:  # numpy fallback marker
            self.index = "numpy"

    # OCR filtering knobs
    OCR_MIN_CONF = 0.65          # per-box confidence floor
    OCR_MIN_ALPHA = 3            # a box needs >= this many letters
    OCR_MIN_ALPHA_RATIO = 0.5    # letters / characters; drops symbol/number soup
    OCR_EDGE_BAND = 0.06         # ignore boxes in the top/bottom 6% (status bars, taskbars)
    OCR_BOILERPLATE_FRAC = 0.4   # a line on >= 40% of frames is UI chrome, not content
    OCR_BOILERPLATE_MIN_FRAMES = 6

    @staticmethod
    def _norm_line(text: str) -> str:
        """Key for 'same line across frames': lowercase, digits/punctuation removed, so a
        clock or temperature that changes between frames still matches itself."""
        return " ".join(re.sub(r"[\d\W_]+", " ", text.lower()).split())

    def _ocr_lines(self, path: str, height: int) -> List[str]:
        """OCR one frame -> cleaned lines in reading order."""
        try:
            result, _ = self._ocr(path)
        except Exception as e:
            logger.warning(f"OCR failed for {path}: {e}")
            return []
        lines: List[str] = []
        for box, txt, conf in result or []:
            if float(conf) < self.OCR_MIN_CONF:
                continue
            y_mid = (min(p[1] for p in box) + max(p[1] for p in box)) / 2 / max(height, 1)
            if y_mid < self.OCR_EDGE_BAND or y_mid > 1 - self.OCR_EDGE_BAND:
                continue
            # NFKC folds full-width punctuation (e.g. "（+1 0.00)") to plain ASCII
            t = unicodedata.normalize("NFKC", txt).strip()
            # split run-together CamelCase that OCR emits when spacing is tight
            t = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", t)
            letters = sum(ch.isalpha() for ch in t)
            if letters < self.OCR_MIN_ALPHA or letters / max(len(t), 1) < self.OCR_MIN_ALPHA_RATIO:
                continue
            lines.append(" ".join(t.split()))
        return lines

    def _drop_boilerplate(self, per_frame: List[List[str]]) -> List[List[str]]:
        """Remove lines that recur across many frames (status bar, 'Rec', watermark)."""
        n = len(per_frame)
        if n < self.OCR_BOILERPLATE_MIN_FRAMES:
            return per_frame
        df: Dict[str, int] = {}
        for lines in per_frame:
            for key in {self._norm_line(l) for l in lines}:
                if key:
                    df[key] = df.get(key, 0) + 1
        limit = max(3, self.OCR_BOILERPLATE_FRAC * n)
        return [
            [l for l in lines if df.get(self._norm_line(l), 0) < limit and self._norm_line(l)]
            for lines in per_frame
        ]

    @staticmethod
    def fit_lines(lines: List[str], max_chars: int, sep: str = " | ") -> str:
        """Join whole lines up to max_chars. Never cuts a word or a line in half."""
        out, used = [], 0
        for l in lines:
            add = len(l) + (len(sep) if out else 0)
            if used + add > max_chars:
                if not out:   # a single over-long line: cut at a word boundary
                    return l[:max_chars].rsplit(" ", 1)[0].rstrip(" ,.;:") + "…"
                break
            out.append(l)
            used += add
        return sep.join(out)

    # ------------------------------------------------------------------ #
    # Search / grounding                                                   #
    # ------------------------------------------------------------------ #

    def _encode_text(self, texts: List[str]) -> np.ndarray:
        self._ensure_model()
        return self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Text -> frame retrieval. Returns [{keyframe, score}] by CLIP cosine."""
        if not self.ready or not query.strip():
            return []
        q = self._encode_text([query])
        k = min(k, len(self.frames))
        if _FAISS and self.index != "numpy":
            scores, idx = self.index.search(q, k)
            pairs = list(zip(idx[0].tolist(), scores[0].tolist()))
        else:
            sims = (self.embeddings @ q[0])
            order = np.argsort(-sims)[:k]
            pairs = [(int(i), float(sims[i])) for i in order]
        return [{"keyframe": self.frames[i], "score": float(s)} for i, s in pairs if i >= 0]

    def ground_entities(
        self, keyframe: Dict[str, Any], entities: List[str], min_sim: float, top_k: int
    ) -> List[Dict[str, Any]]:
        """Which of these spoken entities are visible in this frame? (CLIP cosine)

        Scores are compared WITHIN the candidate set: an entity is kept only if it is
        in the top_k and above min_sim. CLIP cosines are low in absolute terms
        (~0.15-0.35) and uncalibrated, so min_sim is a tunable, not a probability.
        """
        if not self.ready or not entities:
            return []
        try:
            row = self.frames.index(keyframe)
        except ValueError:
            return []
        phrases = [f"a photo of {e}" for e in entities]
        txt = self._encode_text(phrases)
        sims = txt @ self.embeddings[row]
        order = np.argsort(-sims)[:top_k]
        return [
            {"entity": entities[i], "similarity": round(float(sims[i]), 4)}
            for i in order if float(sims[i]) >= min_sim
        ]

    # ------------------------------------------------------------------ #

    def save(self, out_dir: str, video_id: str) -> None:
        """Persist embeddings + OCR (diagnostic/ablation cache)."""
        if not self.ready:
            return
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, f"{video_id}_visual_embeddings.npy"), self.embeddings)
        meta = [
            {"keyframe_id": f.get("keyframe_id"), "timestamp": f.get("timestamp"),
             "ocr_text": f.get("ocr_text", "")}
            for f in self.frames
        ]
        with open(os.path.join(out_dir, f"{video_id}_visual_meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
