# YouTube Link Ingestion — Feasibility & Plan (SHELVED for later)

> **Status: idea parked, not implemented.** Captured 2026-05-25. Locked decision: **720p quality cap**.
> One decision still open (see "Open decision"). No code written.

## Goal
Add a second way to get a video into Mimir: paste a **YouTube URL** → app fetches and shows the
video's metadata (title, thumbnail, channel, duration) → **on explicit user confirmation**, it
downloads with **yt-dlp** and runs the video through the existing pipeline.
**Hard requirement: do not change the current processing architecture** — this is purely an
additional ingestion path.

## Why it's highly feasible (verified against the code)
- **Pipeline is format-agnostic.** Audio extraction (`backend/pipeline/transcription.py:_extract_audio`,
  ffmpeg → `pcm_s16le -ac 1 -ar 16000`) and keyframes (`backend/pipeline/keyframe_extraction.py`,
  OpenCV `cv2.VideoCapture`) accept anything ffmpeg/OpenCV can decode. Upload validation is only a
  MIME + size check. A downloaded mp4 is indistinguishable from an uploaded one.
- **Ingestion flow is reusable.** The whole backend pipeline is reached via
  `task_manager.create_task(task_id, video_path, options)` +
  `background_tasks.add_task(process_video_task, …)` (`backend/api/video.py:43-112`;
  `process_video_task` at `backend/api/video.py:285-316`). A YouTube task just needs to land a file in
  `uploads/{task_id}/` and call those same two functions.
- **Status polling, results enrichment, the inline player, and the subtitled-video download endpoint
  all work unchanged** — they key off `task_id` and `/uploads/{task_id}/…`.
- yt-dlp is the **only new dependency**; ffmpeg (needed to merge best video+audio) is already installed.

## Approach (new front door, zero pipeline change)

### Dependency
- Add **`yt-dlp`** to `backend/requirements.txt` and install into `backend/venv`. It shells out to the
  existing ffmpeg for merging. Keep it reasonably updated (stale yt-dlp breaks on YouTube changes).

### Backend — new `backend/api/youtube.py` router, mounted in `backend/main.py` (one `include_router` line)
No edits to the orchestrator, task manager, or pipeline.
1. `POST /api/v1/youtube/metadata` `{ "url" }` → `yt_dlp.YoutubeDL({"quiet":True,"skip_download":True})
   .extract_info(url, download=False)`. Reject live streams. Return
   `{ title, channel, duration, duration_string, thumbnail, webpage_url, id, filesize_approx }`.
   Fast, no download — powers the preview card.
2. `POST /api/v1/youtube/process` `{ "url", "options" }` → make `task_id` + `uploads/{task_id}/`,
   `task_manager.create_task(task_id, "", options)` (immediately pollable), return `{ task_id }`,
   `background_tasks.add_task(download_and_process_youtube_task, task_id, url, options)`.
3. `download_and_process_youtube_task(task_id, url, options)` (sync, threadpool — like `process_video_task`):
   - status → "Downloading from YouTube…" (progress ~2).
   - yt-dlp **720p cap**: `format="bestvideo[height<=720]+bestaudio/best[height<=720]/best"`,
     `merge_output_format="mp4"`, `outtmpl="uploads/{task_id}/source.%(ext)s"`
     (fixed `source.*` name sidesteps title sanitisation; the subtitled-download endpoint globs by
     extension, not name).
   - On yt-dlp error (private/age-restricted/region-locked/removed) → status "failed" + message, return.
   - Then run the **identical body** as `process_video_task` against the downloaded path (same
     orchestrator call + progress callback). Simplest: refactor the post-download part of
     `process_video_task` into a shared helper, or just call `process_video_task(task_id, path, options)`
     after download. Pipeline call is unchanged either way.

### Frontend — `frontend/src/services/api.js`, `frontend/src/pages/VideoUpload.jsx`
- `api.js`: add `fetchYouTubeMetadata(url)` + `startYouTubeProcess(url, options)` (same fetch pattern as
  `uploadVideo`/`getStatus`). Reuse `pollStatus` as-is.
- `VideoUpload.jsx`:
  - Source toggle at top of the upload card: **"Upload file" | "YouTube link"** (default file → current
    behaviour preserved).
  - YouTube mode: URL input + "Fetch" → `fetchYouTubeMetadata` → **preview card** (thumbnail, title,
    channel, duration) + **"Confirm & Process"**. Natural placement: where the file-pill sits today
    (`VideoUpload.jsx:619-628`), before the options grid.
  - Reuse the **existing options/topic/length/format controls verbatim** — identical processing after
    download.
  - "Confirm & Process" maps `selectedOptions` to the same `options` object from `handleProcess`
    (`VideoUpload.jsx:470-529`), calls `startYouTubeProcess`, then `pollStatus`. Results UI, inline
    player, and subtitled-video download all work unchanged (`video_url` points to the downloaded file).
  - New state: `sourceMode`('file'|'youtube'), `youtubeUrl`, `youtubeMeta`, `isFetchingMeta`; clear them
    in `resetUpload` (`VideoUpload.jsx:551-555`).

## Open decision (deferred)
**Over-limit (size/length) handling.** Recommended default: show duration + estimated size in the
preview card; if over caps, warn but allow proceed; and raise `config.max_file_size_mb` (currently 500)
for the YouTube path so capped-720p long videos aren't rejected. Alternatives: hard-block, or no guard.

## Pre-existing constraints (NOT introduced by this feature)
- `task_manager.max_tasks = 10` (`backend/models/task_manager.py:89`) caps all ingestion, not just YouTube.
- CPU-only Whisper makes very long videos slow regardless of source.
- Legal/ToS of downloading YouTube content is the user's call (local/personal use).

## Files that would change
- `backend/requirements.txt` — add `yt-dlp`.
- `backend/api/youtube.py` — **new** (2 endpoints + 1 background wrapper).
- `backend/main.py` — one `app.include_router(youtube_router)` line.
- `frontend/src/services/api.js` — 2 new methods.
- `frontend/src/pages/VideoUpload.jsx` — source toggle + URL input + preview/confirm card + state.
- (Maybe) small helper extraction from `process_video_task` in `backend/api/video.py` — no pipeline behaviour change.

## Verification (when built)
1. `/youtube/metadata` public URL → fast metadata, no download; live URL → rejected; bad URL → 400.
2. `/youtube/process` → `task_id`; `/status/{task_id}` shows "Downloading from YouTube…" then normal
   pipeline phases to `completed`.
3. `uploads/{task_id}/source.mp4` exists and is ≤720p (`ffprobe`).
4. UI: paste → preview → Confirm → live progress → results; inline player plays; existing
   "download subtitled video" button works on the YouTube-sourced video.
5. Regression: a normal file upload still works unchanged.
6. Teardown: downloads live under `uploads/` and yt-dlp in `backend/venv` — both already wiped by
   `MIMIR_TEARDOWN.sh`.
