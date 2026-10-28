"""Configuration management for the video processing pipeline"""

from pydantic_settings import BaseSettings
from typing import Optional
import torch
import os

class PipelineConfig(BaseSettings):
    """Pipeline configuration settings"""
    
    # GPU Configuration
    use_gpu: bool = True
    cuda_visible_devices: str = "0"
    
    # Model Configuration
    whisper_model: str = "tiny"
    transformers_cache_dir: str = "models"
    spacy_model: str = "en_core_web_sm"
    
    # Processing Configuration
    max_file_size_mb: int = 500
    chunk_size_seconds: int = 30
    max_workers: int = 4
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:5173,http://localhost:5174,http://localhost:3000"
    
    # Storage Configuration
    upload_dir: str = "uploads"
    temp_dir: str = "temp"
    redis_url: str = "redis://redis:6379"
    
    # AI API Keys
    gemini_api_key: str = ""

    # Feature Flags
    enable_transcription: bool = True
    enable_summarization: bool = True
    enable_subtitles: bool = True
    enable_keyframes: bool = True
    enable_topic_analysis: bool = True
    enable_rag: bool = True

    # Cross-modal (visual) channel. Each flag is independent so the ablation
    # harness can switch components on/off; all use LOCAL models (no API quota).
    adaptive_retrieval_budget: bool = True  # scale retrieved-chunk count with video length (off => fixed 30 general / 15 topic)
    enable_keyframe_labels: bool = True     # one batched Gemini call per video for short keyframe titles

    enable_visual_index: bool = True        # CLIP embeddings + OCR over keyframes
    enable_visual_graph: bool = True        # visual/cross-modal nodes+edges in the KG
    enable_visual_retrieval: bool = True    # CLIP text->frame channel in hybrid retrieval
    enable_ocr_in_prompt: bool = True       # on-screen text + grounded entities in the LLM prompt
    visual_model: str = "clip-ViT-B-32"     # sentence-transformers CLIP (text+image, 512-d)
    visual_ground_min_sim: float = 0.20     # min CLIP cosine to link a frame to a spoken entity
    visual_ground_top_k: int = 3            # max grounded spoken entities per frame
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

    def setup_device(self) -> torch.device:
        """Setup and return the appropriate device"""
        if self.use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = torch.device("cpu")
            print("GPU not available or disabled, using CPU")
        return device

    @property
    def device(self) -> torch.device:
        """Get the configured device"""
        return self.setup_device()

# Global configuration instance
config = PipelineConfig()
