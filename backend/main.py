from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import logging
from contextlib import asynccontextmanager
from datetime import datetime
import torch

# Import components
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.config import config
from app_state import orchestrator as pipeline_orchestrator, task_manager
from api.video import router as video_router
from api.auth import router as auth_router
from api.rag import router as rag_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Mimir Video Processing API...")
    logger.info(f"Using device: {config.device}")
    logger.info(f"GPU enabled: {config.use_gpu}")
    
    # Initialize pipeline components
    try:
        pipeline_orchestrator.initialize_components()
        logger.info("Pipeline components initialized successfully")
    except Exception as e:
        logger.warning(f"Some pipeline components failed to initialize: {str(e)}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Mimir Video Processing API...")
    # Cleanup tasks
    task_manager.cleanup_completed_tasks(max_age_hours=0)

# Create FastAPI app
app = FastAPI(
    title="Mimir Video Processing API",
    version="1.0.0",
    description="AI-powered video processing pipeline with transcription, summarization, and subtitle generation",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded files (videos, keyframes, subtitles) for the frontend
os.makedirs(config.upload_dir, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=config.upload_dir), name="uploads")

# Include API routes
app.include_router(auth_router)
app.include_router(video_router)
app.include_router(rag_router)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Mimir Video Processing API",
        "version": "1.0.0",
        "description": "AI-powered video processing pipeline",
        "endpoints": {
            "health": "/health",
            "pipeline_status": "/pipeline/status",
            "api_docs": "/docs",
            "auth": {
                "register": "/api/v1/auth/register",
                "login": "/api/v1/auth/login",
                "me": "/api/v1/auth/me"
            },
            "video_processing": "/api/v1/upload"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/stats")
async def get_pipeline_stats():
    """Get pipeline statistics"""
    return pipeline_orchestrator.get_pipeline_status()

@app.get("/pipeline/status")
async def pipeline_status():
    """Get pipeline component status"""
    return pipeline_orchestrator.get_pipeline_status()

@app.get("/statistics")
async def get_statistics():
    """Get system statistics"""
    return {
        "task_statistics": task_manager.get_statistics(),
        "pipeline_status": pipeline_orchestrator.get_pipeline_status(),
        "system_info": {
            "device": str(config.device),
            "gpu_enabled": config.use_gpu,
            "supported_formats": ["mp4", "avi", "mov", "mkv"],
            "max_file_size_mb": config.max_file_size_mb
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.api_host,
        port=config.api_port,
        reload=True,
        # Without this, --reload watches the whole backend/ tree by default —
        # including uploads/ and mimir.db, which the pipeline itself writes to
        # continuously while processing a video (audio extraction, keyframes,
        # chunks, FAISS index, KG pickle, every progress update). Each of those
        # writes looks like a code change, so the reloader restarts the server
        # mid-request and silently kills whatever task was in flight.
        reload_excludes=["uploads/*", "mimir.db", "*.db", "__pycache__/*"],
    )
