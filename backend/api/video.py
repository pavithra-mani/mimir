"""Video processing API routes"""

import os
import uuid
import asyncio
import logging
import subprocess
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
from pipeline.orchestrator import PipelineError
from pipeline.config import config
from app_state import orchestrator as pipeline_orchestrator, task_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["video"])

# Pydantic models
class ProcessingOptions(BaseModel):
    transcript: bool = True
    summary: bool = True
    subtitles: bool = True
    keyframes: bool = True   # always on — keyframes are Phase 1 of the pipeline
    rag: bool = True         # always on — RAG is the core indexing pipeline
    topic_based: bool = False
    topic: Optional[str] = None
    download_format: str = "srt"        # 'srt' | 'vtt' (user choice for download)
    summary_length: str = "medium"      # 'short' | 'medium' | 'long'
    use_gpu: bool = True
    video_id: Optional[str] = None

    class Config:
        extra = "allow"  # Allow extra fields

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        if 'topic' not in data:
            data['topic'] = None
        return data

@router.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    options: str = Form(None)
):
    """Upload video and start processing"""
    
    # Validate file
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Check file size
    file_size = 0
    content = await file.read()
    file_size = len(content)
    await file.seek(0)  # Reset file pointer
    
    max_size_mb = config.max_file_size_mb
    if file_size > max_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=400, 
            detail=f"File size exceeds {max_size_mb}MB limit"
        )
    
    # Parse options
    try:
        if options:
            import json
            options_dict = json.loads(options)
            processing_options = ProcessingOptions(**options_dict)
        else:
            processing_options = ProcessingOptions()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid options: {str(e)}")
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Create upload directory
    upload_dir = os.path.join(config.upload_dir, task_id)
    os.makedirs(upload_dir, exist_ok=True)
    
    # Save video file
    video_path = os.path.join(upload_dir, file.filename)
    with open(video_path, "wb") as buffer:
        buffer.write(content)
    
    # Initialize task
    task_manager.create_task(task_id, video_path, processing_options.model_dump())
    
    # Start background processing
    try:
        background_tasks.add_task(
            process_video_task, 
            task_id, 
            video_path, 
            processing_options
        )
        logger.info(f"Background task started for task {task_id}")
    except Exception as e:
        logger.error(f"Failed to start background task: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start processing: {str(e)}")
    
    return {
        "task_id": task_id,
        "message": "Video uploaded successfully",
        "filename": file.filename,
        "file_size": file_size
    }

@router.get("/status/{task_id}")
async def get_status(task_id: str):
    """Get processing status"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task.to_dict()

@router.get("/results/{task_id}")
async def get_results(task_id: str):
    """Get processing results"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task.status != "completed":
        raise HTTPException(status_code=400, detail="Processing not completed")
    
    return task.results

@router.delete("/task/{task_id}")
async def delete_task(task_id: str):
    """Delete task and cleanup files"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Cleanup files
    task_manager.cleanup_task(task_id)
    
    return {"message": "Task deleted successfully"}

@router.get("/tasks")
async def list_tasks():
    """List all tasks"""
    tasks = task_manager.get_all_tasks()
    return {"tasks": [task.to_dict() for task in tasks]}

@router.get("/pipeline-logs/{task_id}")
async def get_pipeline_logs(task_id: str):
    """Return per-stage pipeline logs for the diagnostics page"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    results = task.results or {}
    logs = results.get("pipeline_logs", {})
    return {"task_id": task_id, "logs": logs, "status": task.status}

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")


@router.get("/download/subtitled-video/{task_id}")
async def download_subtitled_video(task_id: str):
    """Return the source video with subtitles embedded, generated on demand.

    Soft-muxes the already-generated ``subtitles.vtt`` into the source video as a
    selectable ``mov_text`` track using ``ffmpeg -c copy`` (no re-encode), then caches
    the result. This is lazy and request-triggered — it does NOT touch the processing
    pipeline.
    """
    task_dir = os.path.join(config.upload_dir, task_id)
    if not os.path.isdir(task_dir):
        raise HTTPException(status_code=404, detail="Task not found")

    vtt_path = os.path.join(task_dir, "subtitles.vtt")
    if not os.path.isfile(vtt_path):
        raise HTTPException(status_code=404, detail="Subtitles not generated for this video")

    # Locate the source video, skipping the cached subtitled output and sidecars.
    source_video = None
    for name in sorted(os.listdir(task_dir)):
        lower = name.lower()
        if lower.endswith("_subtitled.mp4"):
            continue
        if lower.endswith(VIDEO_EXTS):
            source_video = os.path.join(task_dir, name)
            break
    if not source_video:
        raise HTTPException(status_code=404, detail="Source video not found")

    stem = os.path.splitext(os.path.basename(source_video))[0]
    out_path = os.path.join(task_dir, f"{stem}_subtitled.mp4")
    download_name = f"{stem}_subtitled.mp4"

    # Serve the cached file if it is newer than both inputs.
    if os.path.isfile(out_path):
        out_mtime = os.path.getmtime(out_path)
        if out_mtime >= os.path.getmtime(source_video) and out_mtime >= os.path.getmtime(vtt_path):
            return FileResponse(out_path, media_type="video/mp4", filename=download_name)

    # Soft-mux subtitles as a selectable mov_text track — fast, no re-encode.
    # NOTE: mov_text targets MP4-family containers; exotic inputs may need a different -c:s.
    cmd = [
        "ffmpeg", "-y", "-i", source_video, "-i", vtt_path,
        "-map", "0", "-map", "1", "-c", "copy", "-c:s", "mov_text",
        "-metadata:s:s:0", "language=eng", out_path,
    ]
    try:
        proc = await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True, timeout=600
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Subtitle muxing timed out")

    if proc.returncode != 0:
        if os.path.isfile(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
        stderr_tail = (proc.stderr or "")[-800:]
        logger.error(f"ffmpeg mux failed for {task_id}: {stderr_tail}")
        raise HTTPException(status_code=500, detail=f"Failed to mux subtitles: {stderr_tail}")

    return FileResponse(out_path, media_type="video/mp4", filename=download_name)


def _to_url_path(path: str) -> str:
    """Normalise a backend file path to a forward-slash URL path under /uploads."""
    p = path.replace("\\", "/")
    # Strip any leading "./"
    if p.startswith("./"):
        p = p[2:]
    # Ensure it starts with /uploads/
    if not p.startswith("/"):
        p = "/" + p
    return p


def _enrich_results_with_urls(results: Dict[str, Any], task_id: str, video_path: str, loop: asyncio.AbstractEventLoop) -> None:
    """Add browser-loadable URLs to the results dict for the frontend."""
    filename = os.path.basename(video_path)
    upload_dir = os.path.dirname(video_path)

    # Original video for the inline player
    results["video_url"] = f"/uploads/{task_id}/{filename}"
    results["task_id"] = task_id
    results["filename"] = filename

    # Always emit a sibling .vtt next to the video so the <track> tag can load it,
    # regardless of the user's chosen download format.
    if results.get("subtitles") and isinstance(results["subtitles"], dict):
        try:
            from pipeline.subtitle_generation import SubtitleGenerator
            transcript = results.get("transcript")
            if transcript and isinstance(transcript, dict) and transcript.get("segments"):
                vtt_payload = loop.run_until_complete(
                    SubtitleGenerator().generate_subtitles(transcript, "vtt")
                )
                vtt_path = os.path.join(upload_dir, "subtitles.vtt")
                with open(vtt_path, "w", encoding="utf-8") as f:
                    f.write(vtt_payload.get("subtitles", ""))
                results["subtitles_vtt_url"] = f"/uploads/{task_id}/subtitles.vtt"
        except Exception as e:
            logger.warning(f"Failed to write subtitles.vtt for task {task_id}: {e}")

    # Keyframe image URLs — keyframe_extraction stores frame_path as
    # uploads/{video_stem}/frames/keyframe_{ts}.jpg (relative, OS-native sep).
    kf_block = results.get("keyframes")
    if isinstance(kf_block, dict) and isinstance(kf_block.get("keyframes"), list):
        for kf in kf_block["keyframes"]:
            fp = kf.get("frame_path")
            if not fp:
                continue
            url_path = fp.replace("\\", "/")
            if not url_path.startswith("/"):
                url_path = "/" + url_path
            kf["frame_url"] = url_path


def process_video_task(task_id: str, video_path: str, options: ProcessingOptions):
    """Background task for video processing - runs in thread pool so event loop stays free for polls"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        task_manager.update_status(task_id, "processing", "Initializing pipeline...", 5)

        results = loop.run_until_complete(
            pipeline_orchestrator.process_video(
                video_path=video_path,
                options=options.model_dump(),
                progress_callback=lambda step, progress: task_manager.update_status(
                    task_id, "processing", step, progress
                )
            )
        )

        # Enrich with URLs the frontend can load directly via /uploads static mount.
        try:
            _enrich_results_with_urls(results, task_id, video_path, loop)
        except Exception as e:
            logger.warning(f"URL enrichment failed for task {task_id}: {e}")

        task_manager.update_status(task_id, "completed", "Processing completed!", 100)
        task_manager.set_results(task_id, results)

    except Exception as e:
        logger.error(f"Processing failed for task {task_id}: {str(e)}")
        task_manager.update_status(task_id, "failed", f"Processing failed: {str(e)}", 0)
    finally:
        loop.close()
