"""Data models and schema definitions for MemWeaver-AI."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MemoryTier(str, Enum):
    """Hierarchical tiers of agent memory."""
    WORKING = "working"      # Bounded scratchpad / immediate conversational context
    EPISODIC = "episodic"    # Temporal sequence of interactions, observations, and events
    SEMANTIC = "semantic"    # Distilled long-term atomic facts, user profiles, and beliefs


class MemoryType(str, Enum):
    """Categorical types of memory items."""
    OBSERVATION = "observation"
    FACT = "fact"
    PREFERENCE = "preference"
    GOAL = "goal"
    REFLECTION = "reflection"
    DIALOGUE = "dialogue"


class MemoryStatus(str, Enum):
    """Lifecycle status of a memory item."""
    ACTIVE = "active"
    SUPERSEDED = "superseded"  # Invalidated by newer contradictory knowledge
    ARCHIVED = "archived"


class MemoryItem(BaseModel):
    """Core memory item representation across all tiers."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    tier: MemoryTier = MemoryTier.EPISODIC
    memory_type: MemoryType = MemoryType.OBSERVATION
    status: MemoryStatus = MemoryStatus.ACTIVE
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: float = Field(default_factory=time.time)
    last_accessed_at: float = Field(default_factory=time.time)
    access_count: int = 0
    vector: Optional[List[float]] = None
    session_id: str = "default"
    agent_id: str = "agent_default"
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    superseded_by: Optional[str] = None
    supersedes_id: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    def touch(self) -> None:
        """Updates access metadata when retrieved."""
        self.last_accessed_at = time.time()
        self.access_count += 1

    def mark_superseded(self, newer_id: str) -> None:
        """Marks this memory as superseded by a newer contradictory item."""
        self.status = MemoryStatus.SUPERSEDED
        self.superseded_by = newer_id

    @property
    def token_estimate(self) -> int:
        """Approximates token count based on whitespace and punctuation."""
        return max(1, len(self.content) // 4)


class RetrievalResult(BaseModel):
    """Ranked memory item with transparent scoring breakdown."""
    item: MemoryItem
    score: float
    semantic_similarity: float
    recency_score: float
    importance_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.item.id,
            "content": self.item.content,
            "tier": self.item.tier.value,
            "type": self.item.memory_type.value,
            "score": round(self.score, 4),
            "semantic_similarity": round(self.semantic_similarity, 4),
            "recency_score": round(self.recency_score, 4),
            "importance_score": round(self.importance_score, 4),
            "status": self.item.status.value,
        }


class ContradictionResult(BaseModel):
    """Result of contradiction detection between old and new knowledge."""
    has_conflict: bool
    conflicting_item_id: Optional[str] = None
    reason: Optional[str] = None
    confidence: float = 0.0


class ConsolidationReport(BaseModel):
    """Summary of episodic-to-semantic consolidation cycle."""
    episodic_items_analyzed: int = 0
    semantic_facts_generated: int = 0
    contradictions_resolved: int = 0
    compression_ratio: float = 0.0
    duration_ms: float = 0.0
    new_semantic_ids: List[str] = Field(default_factory=list)


class MemoryStats(BaseModel):
    """Aggregate telemetry and operational health stats."""
    total_memories: int = 0
    working_count: int = 0
    episodic_count: int = 0
    semantic_count: int = 0
    superseded_count: int = 0
    retrieval_count: int = 0
    average_retrieval_latency_ms: float = 0.0
    consolidation_cycles: int = 0
