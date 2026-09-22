"""SRT/VTT -> transcript-dict parsing, shared by generate_summaries.py and
generate_references.py. No dependency on the pipeline package on purpose -
generate_references.py only needs `anthropic`, not the full backend ML stack.
"""

import re
from pathlib import Path

TIMING_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[.,](\d{3})"
)
TAG_RE = re.compile(r"<[^>]+>")


def _to_seconds(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_subtitle_to_transcript(path: Path) -> dict:
    """SRT/VTT -> the transcript dict shape SemanticChunker.chunk_transcript expects:
    {"segments": [{"start": float, "end": float, "text": str}, ...], "language": "en"}
    """
    raw = path.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", raw.strip())

    segments = []
    for block in blocks:
        lines = [l for l in block.split("\n") if l.strip()]
        if not lines:
            continue
        timing_idx = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if timing_idx is None:
            continue
        m = TIMING_RE.search(lines[timing_idx])
        if not m:
            continue
        g = m.groups()
        start, end = _to_seconds(*g[0:4]), _to_seconds(*g[4:8])
        text = " ".join(TAG_RE.sub("", l).strip() for l in lines[timing_idx + 1:]).strip()
        if text:
            segments.append({"start": start, "end": end, "text": text})

    # Dedupe consecutive identical cues (common in rolling auto-captions)
    deduped = []
    for seg in segments:
        if deduped and deduped[-1]["text"] == seg["text"]:
            deduped[-1]["end"] = seg["end"]
        else:
            deduped.append(seg)

    return {"segments": deduped, "language": "en"}


def transcript_to_text(transcript: dict) -> str:
    return " ".join(seg["text"] for seg in transcript["segments"])
