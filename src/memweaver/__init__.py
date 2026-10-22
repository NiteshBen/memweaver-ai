"""MemWeaver-AI: Production Hierarchical Long-Term Memory & State Engine for AI Agents."""

__version__ = "0.1.0"
__author__ = "Nitesh Beniwal"

from memweaver.config import MemoryConfig
from memweaver.conflict_resolver import ConflictResolver
from memweaver.consolidation import ConsolidationEngine
from memweaver.dashboard import DashboardReporter
from memweaver.decay import CognitiveDecayEngine
from memweaver.embeddings import EmbeddingEngine
from memweaver.engine import MemoryEngine
from memweaver.episodic_memory import EpisodicMemory
from memweaver.models import (
    ConsolidationReport,
    ContradictionResult,
    MemoryItem,
    MemoryStats,
    MemoryStatus,
    MemoryTier,
    MemoryType,
    RetrievalResult,
)
from memweaver.semantic_memory import SemanticMemory
from memweaver.storage import BaseStorage, InMemoryStorage, SQLiteStorage
from memweaver.working_memory import WorkingMemory

__all__ = [
    "MemoryEngine",
    "MemoryConfig",
    "MemoryItem",
    "MemoryTier",
    "MemoryType",
    "MemoryStatus",
    "RetrievalResult",
    "ConsolidationReport",
    "ContradictionResult",
    "MemoryStats",
    "WorkingMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "EmbeddingEngine",
    "CognitiveDecayEngine",
    "ConflictResolver",
    "ConsolidationEngine",
    "DashboardReporter",
    "BaseStorage",
    "InMemoryStorage",
    "SQLiteStorage",
]
