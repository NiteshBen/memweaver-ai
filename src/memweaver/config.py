"""Configuration and hyperparameter settings for MemWeaver-AI."""

from __future__ import annotations

import os
from typing import Optional
from pydantic import BaseModel, Field, model_validator


class MemoryConfig(BaseModel):
    """Runtime configuration for memory retrieval, decay, and persistence."""
    
    # Hybrid Retrieval Weights (Must sum to ~1.0)
    weight_semantic: float = Field(default=0.50, ge=0.0, le=1.0)
    weight_recency: float = Field(default=0.30, ge=0.0, le=1.0)
    weight_importance: float = Field(default=0.20, ge=0.0, le=1.0)

    # Cognitive Decay Settings
    decay_half_life_seconds: float = Field(default=86400.0, gt=0.0)  # 24 hours baseline
    decay_power: float = Field(default=1.0, gt=0.0)
    access_boost_factor: float = Field(default=0.15, ge=0.0)

    # Capacity Limits
    working_memory_max_items: int = Field(default=10, ge=1)
    working_memory_max_tokens: int = Field(default=2048, ge=1)

    # Semantic Consolidation & Belief Revision
    enable_belief_revision: bool = True
    contradiction_similarity_threshold: float = Field(default=0.72, ge=0.0, le=1.0)
    consolidation_batch_size: int = Field(default=20, ge=1)

    # Storage & Persistence
    storage_path: Optional[str] = None  # None = In-Memory, or path to .db SQLite
    
    # Embedding Configuration
    embedding_dimension: int = Field(default=128, ge=16, le=1024)
    offline_mode: bool = True
    gemini_api_key: Optional[str] = None

    @model_validator(mode="after")
    def validate_weights(self) -> MemoryConfig:
        total = self.weight_semantic + self.weight_recency + self.weight_importance
        if abs(total - 1.0) > 1e-4:
            # Normalize automatically
            self.weight_semantic = self.weight_semantic / total
            self.weight_recency = self.weight_recency / total
            self.weight_importance = self.weight_importance / total
        
        # Pull environment key if not explicitly set
        if not self.gemini_api_key:
            self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        return self
