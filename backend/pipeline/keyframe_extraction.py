"""Phase 1: Keyframe Extraction with Adaptive k-means++ Clustering (VideoRAG-KAIST)

Performance notes:
- Seek directly to each 1-second mark (no sequential full-fps decode)
- Score computed inline on a 320x180 medium frame; full frame discarded immediately
- Only selected keyframes are written to disk (not all 1fps frames)

Selection notes:
- Candidates are first picked per scene (contiguous run of one visual cluster).
- A cluster that recurs across multiple scenes (e.g. the video keeps cutting back to
  the same shot) is capped globally across ALL its occurrences, not per occurrence —
  otherwise its keyframe budget scales with how many times it recurs, not with how
  much unique visual information it contains.
- If candidates still exceed the final budget, the cut is diversity-aware (MMR over
  the clustering feature vectors) instead of chronological truncation, so a long
  video doesn't just lose its later sections.
"""

import os
import math
import time
import logging
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from scipy.signal import medfilt

logger = logging.getLogger(__name__)

try:
    import imagehash
    from PIL import Image as PILImage
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False
    logger.warning("imagehash not installed; perceptual dedup disabled. Run: pip install imagehash")


class KeyframeExtractor:
    """Adaptive clustering-based keyframe extractor (VideoRAG architecture)."""

    FRAME_FPS = 1
    CLUSTERS_PER_MIN = 4.0
    MIN_CLUSTERS = 8
    MAX_CLUSTERS = 120
    W_SHARPNESS = 0.4
    W_CONTRAST  = 0.3
    W_EDGE      = 0.3
    MIN_SCORE_THRESHOLD  = 0.15
    MIN_FRAMES_PER_SCENE = 1
    MAX_FRAMES_PER_SCENE = 5
    SEC_PER_EXTRA_FRAME  = 15
    HASH_THRESHOLD       = 8
    MAX_FRAMES_PER_CLUSTER = 4   # global cap: a recurring visual setup (e.g. "host talking")
                                 # keeps re-triggering as a new scene, but it's the same
                                 # cluster_id every time — cap its TOTAL budget across the
                                 # whole video, not per occurrence, so it can't crowd out
                                 # clusters that only ever appear once.

    def __init__(self, **kwargs):
        pass

    def extract_keyframes(
        self,
        video_path: str,
        progress_callback: Optional[callable] = None,
        transcript: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        t0 = time.time()
        try:
            if progress_callback:
                progress_callback("Extracting frames at 1FPS...", 30)

            frames_data, duration = self._sample_frames(video_path)

            if not frames_data:
                raise KeyframeExtractionError("No frames extracted from video")

            if progress_callback:
                progress_callback("Clustering frames (k-means++)...", 50)

            cluster_labels, k = self._cluster_frames(frames_data, duration)

            if progress_callback:
                progress_callback("Segmenting visual scenes...", 62)

            scene_segments = self._build_scene_segments(frames_data, cluster_labels, transcript)

            if progress_callback:
                progress_callback("Selecting and saving keyframes...", 72)

            keyframes_dir = os.path.join(os.path.dirname(video_path), "frames")
            os.makedirs(keyframes_dir, exist_ok=True)
            keyframes = self._select_and_write_keyframes(
                video_path, frames_data, scene_segments, keyframes_dir
            )

            if progress_callback:
                progress_callback("Keyframe extraction complete!", 82)

            sample_size = min(200, len(frames_data))
            frame_scores_sample = [
                {
                    "timestamp": round(frames_data[i]["timestamp"], 2),
                    "score":     round(frames_data[i]["score"], 4),
                    "cluster":   int(cluster_labels[i]),
                }
                for i in range(sample_size)
            ]

            return {
                "keyframes":            keyframes,
                "total_frames":         len(frames_data),
                "total_frames_at_1fps": len(frames_data),
                "duration":             round(duration, 2),
                "fps":                  self.FRAME_FPS,
                "extracted_count":      len(keyframes),
                "cluster_k":            int(k),
                "scene_count":          len(scene_segments),
                "scene_segments":       scene_segments,
                "frame_scores_sample":  frame_scores_sample,
                "timing_sec":           round(time.time() - t0, 2),
            }

        except Exception as e:
            logger.error(f"Keyframe extraction failed: {e}")
            raise KeyframeExtractionError(f"Failed to extract keyframes: {e}")

    # ------------------------------------------------------------------ #
    # Step 1: sample 1 frame per second, score inline, keep only small    #
    # ------------------------------------------------------------------ #

    def _sample_frames(
        self, video_path: str
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Seek directly to each 1-second mark; compute score inline; never keep full frame."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise KeyframeExtractionError("Cannot open video file")

        native_fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration    = total_frames / native_fps if native_fps > 0 else 0.0

        frames_data: List[Dict[str, Any]] = []
        n_seconds = max(1, int(math.ceil(duration)))

        for sec in range(n_seconds):
            target_frame = int(round(sec * native_fps))
            if target_frame >= total_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if not ret:
                continue

            score = self._score_frame(frame)

            small = cv2.resize(frame, (64, 64))
            small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            # full frame is discarded here — never stored in the list

            frames_data.append({
                "timestamp": float(sec),
                "frame_idx": target_frame,
                "small_rgb": small_rgb,   # 64×64 — used for clustering
                "score":     score,       # raw (unnormalized) — normalized below
            })

        cap.release()

        # Min-max normalize scores across all frames
        if frames_data:
            raw_scores = [fd["score"] for fd in frames_data]
            mn, mx = min(raw_scores), max(raw_scores)
            rng = mx - mn if mx != mn else 1.0
            for fd in frames_data:
                fd["score"] = (fd["score"] - mn) / rng

        return frames_data, duration

    def _score_frame(self, frame: np.ndarray) -> float:
        """Sharpness + contrast + edge density on a 320×180 downsample."""
        med = cv2.resize(frame, (320, 180))
        gray = cv2.cvtColor(med, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast  = float(gray.std())
        edges     = cv2.Canny(gray, 100, 200)
        edge_den  = float(np.count_nonzero(edges)) / edges.size
        return self.W_SHARPNESS * sharpness + self.W_CONTRAST * contrast + self.W_EDGE * edge_den

    # ------------------------------------------------------------------ #
    # Step 2: k-means++ clustering on small_rgb features                  #
    # ------------------------------------------------------------------ #

    def _cluster_frames(
        self, frames_data: List[Dict[str, Any]], duration: float
    ) -> Tuple[np.ndarray, int]:
        n = len(frames_data)
        k = int(math.ceil((duration / 60.0) * self.CLUSTERS_PER_MIN))
        k = max(self.MIN_CLUSTERS, min(self.MAX_CLUSTERS, k, n))

        features = np.array(
            [fd["small_rgb"].flatten().astype(np.float32) for fd in frames_data]
        )
        features = normalize(features)

        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=3, random_state=42)
        labels = kmeans.fit_predict(features)
        # Smooth label sequence to eliminate single-frame cluster flickers
        smooth_labels = medfilt(labels.astype(float), kernel_size=5).astype(int)
        return smooth_labels, k

    # ------------------------------------------------------------------ #
    # Step 3: scene segmentation from contiguous cluster-label runs       #
    # ------------------------------------------------------------------ #

    def _build_scene_segments(
        self,
        frames_data: List[Dict[str, Any]],
        labels: np.ndarray,
        transcript: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not frames_data:
            return []

        words: List[Dict] = []
        if transcript:
            if "words" in transcript:
                words = transcript["words"]
            elif "segments" in transcript:
                for seg in transcript["segments"]:
                    words.extend(seg.get("words", []))

        scenes: List[Dict[str, Any]] = []
        scene_id = 0
        current_label = int(labels[0])
        scene_start_idx = 0

        for i in range(1, len(frames_data) + 1):
            is_last = i == len(frames_data)
            label_changed = is_last or (int(labels[i]) != current_label)

            if label_changed:
                start_sec = frames_data[scene_start_idx]["timestamp"]
                end_sec   = frames_data[i - 1]["timestamp"]
                frame_indices = list(range(scene_start_idx, i))
                scene_words = [
                    w.get("word", "")
                    for w in words
                    if start_sec <= w.get("start", 0) <= end_sec
                ]
                scenes.append({
                    "scene_id":        scene_id,
                    "cluster_id":      current_label,
                    "start_sec":       round(start_sec, 2),
                    "end_sec":         round(end_sec, 2),
                    "duration_sec":    round(end_sec - start_sec, 2),
                    "frame_indices":   frame_indices,
                    "frame_count":     len(frame_indices),
                    "transcript_words": scene_words[:50],
                })
                scene_id += 1
                if not is_last:
                    current_label  = int(labels[i])
                    scene_start_idx = i

        return scenes

    # ------------------------------------------------------------------ #
    # Step 4: select keyframes per scene, write only those to disk        #
    # ------------------------------------------------------------------ #

    def _select_and_write_keyframes(
        self,
        video_path: str,
        frames_data: List[Dict[str, Any]],
        scene_segments: List[Dict[str, Any]],
        keyframes_dir: str,
    ) -> List[Dict[str, Any]]:
        """Pick best frames per scene, then re-seek video to write only those."""
        selected: List[Dict[str, Any]] = []  # {scene, frame_idx, timestamp, score, scene_id}

        for scene in scene_segments:
            indices   = scene["frame_indices"]
            scene_dur = scene["duration_sec"]

            n_kf = max(
                self.MIN_FRAMES_PER_SCENE,
                min(self.MAX_FRAMES_PER_SCENE, int(scene_dur / self.SEC_PER_EXTRA_FRAME)),
            )

            ranked = sorted(indices, key=lambda i: frames_data[i]["score"], reverse=True)
            candidates = [i for i in ranked if frames_data[i]["score"] >= self.MIN_SCORE_THRESHOLD][:n_kf]

            for idx in candidates:
                selected.append({
                    "list_idx":  idx,
                    "frame_idx": frames_data[idx]["frame_idx"],
                    "timestamp": frames_data[idx]["timestamp"],
                    "score":     frames_data[idx]["score"],
                    "scene_id":  scene["scene_id"],
                    "cluster_id": scene["cluster_id"],
                    "small_rgb": frames_data[idx]["small_rgb"],
                })

        # Global per-cluster cap: a scene that's the SAME cluster_id as an earlier scene
        # (e.g. the video keeps cutting back to the same shot) only gets to keep its top
        # MAX_FRAMES_PER_CLUSTER frames across ALL its occurrences combined, not per
        # occurrence. Non-recurring clusters are unaffected (a single scene's own count is
        # already <= MAX_FRAMES_PER_SCENE from the loop above).
        by_cluster: Dict[int, List[Dict[str, Any]]] = {}
        for sel in selected:
            by_cluster.setdefault(sel["cluster_id"], []).append(sel)
        deduped: List[Dict[str, Any]] = []
        for group in by_cluster.values():
            group.sort(key=lambda s: s["score"], reverse=True)
            deduped.extend(group[: self.MAX_FRAMES_PER_CLUSTER])

        # Final budget: bounded, not purely duration-linear — a longer video isn't
        # necessarily more visually varied, just longer, so scaling the cap with duration
        # alone re-introduces the redundancy the per-cluster cap above just removed.
        if frames_data:
            duration_min = (frames_data[-1]["timestamp"] + 1) / 60.0
            max_kf = min(60, max(10, int(duration_min * 2)))
        else:
            max_kf = 10

        if len(deduped) > max_kf:
            # Diversity-aware final cut (MMR) instead of chronological truncation — same
            # principle used for retrieval chunk selection: trade off quality against
            # similarity to what's already been picked, reusing the small_rgb feature
            # vectors already computed for clustering (no extra cost). Chronological
            # truncation would arbitrarily favor whatever comes first in scene order and
            # could drop an entire later section of the video.
            selected = self._mmr_select_keyframes(deduped, max_kf)
        else:
            selected = deduped

        selected.sort(key=lambda s: s["timestamp"])

        # Re-open video once, seek to each selected frame, write JPEG
        cap = cv2.VideoCapture(video_path)
        keyframes: List[Dict[str, Any]] = []
        seen_hashes: List = []
        kf_id = 0

        for sel in selected:
            cap.set(cv2.CAP_PROP_POS_FRAMES, sel["frame_idx"])
            ret, full_frame = cap.read()
            if not ret:
                continue

            if IMAGEHASH_AVAILABLE:
                pil_img = PILImage.fromarray(cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB))
                h = imagehash.phash(pil_img)
                if any(h - ph < self.HASH_THRESHOLD for ph in seen_hashes):
                    continue
                seen_hashes.append(h)

            ts = sel["timestamp"]
            kf_filename = f"keyframe_sc{sel['scene_id']:03d}_kf{kf_id:03d}_t{ts:.3f}.jpg"
            kf_path = os.path.join(keyframes_dir, kf_filename)
            cv2.imwrite(kf_path, full_frame)

            keyframes.append({
                "keyframe_id":          f"kf_{kf_id:04d}",
                "scene_id":             sel["scene_id"],
                "cluster_id":           sel["cluster_id"],
                "timestamp":            round(ts, 3),
                "informativeness_score": round(sel["score"], 4),
                "frame_path":           kf_path,
                "description":          f"Scene {sel['scene_id']} keyframe at {ts:.1f}s",
            })
            kf_id += 1

        cap.release()
        return keyframes

    @staticmethod
    def _mmr_select_keyframes(
        candidates: List[Dict[str, Any]], k: int, lambda_param: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Maximal Marginal Relevance over keyframe candidates, trading off quality score
        against visual similarity to already-selected frames (cosine over the small_rgb
        feature vectors already computed for clustering — no extra cost).
        """
        if len(candidates) <= k:
            return list(candidates)

        scores = [c["score"] for c in candidates]
        lo, hi = min(scores), max(scores)
        span = (hi - lo) or 1.0
        relevance = [(s - lo) / span for s in scores]

        feats = [
            normalize(c["small_rgb"].flatten().astype(np.float32).reshape(1, -1))[0]
            for c in candidates
        ]

        remaining = list(range(len(candidates)))
        first = max(remaining, key=lambda i: relevance[i])
        chosen = [first]
        remaining.remove(first)

        while remaining and len(chosen) < k:
            best_i, best_mmr = remaining[0], -float("inf")
            for i in remaining:
                max_sim = max(float(np.dot(feats[i], feats[j])) for j in chosen)
                mmr = lambda_param * relevance[i] - (1 - lambda_param) * max_sim
                if mmr > best_mmr:
                    best_mmr, best_i = mmr, i
            chosen.append(best_i)
            remaining.remove(best_i)

        return [candidates[i] for i in chosen]


class KeyframeExtractionError(Exception):
    pass
