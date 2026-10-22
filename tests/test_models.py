"""Tests for data models and schema definitions."""

import time
import pytest
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


def test_memory_item_initialization_defaults():
    item = MemoryItem(content="User prefers Python for data science")
    assert item.content == "User prefers Python for data science"
    assert item.tier == MemoryTier.EPISODIC
    assert item.memory_type == MemoryType.OBSERVATION
    assert item.status == MemoryStatus.ACTIVE
    assert item.importance == 0.5
    assert item.access_count == 0
    assert item.superseded_by is None
    assert item.token_estimate >= 5


def test_memory_item_touch():
    item = MemoryItem(content="Observation test")
    t0 = item.last_accessed_at
    time.sleep(0.01)
    item.touch()
    assert item.access_count == 1
    assert item.last_accessed_at > t0


def test_memory_item_mark_superseded():
    item = MemoryItem(content="Old config setting")
    item.mark_superseded("new-uuid-1234")
    assert item.status == MemoryStatus.SUPERSEDED
    assert item.superseded_by == "new-uuid-1234"


def test_retrieval_result_serialization():
    item = MemoryItem(content="Test fact", tier=MemoryTier.SEMANTIC, memory_type=MemoryType.FACT)
    res = RetrievalResult(
        item=item,
        score=0.91234,
        semantic_similarity=0.85432,
        recency_score=0.98765,
        importance_score=0.8,
    )
    d = res.to_dict()
    assert d["content"] == "Test fact"
    assert d["tier"] == "semantic"
    assert d["score"] == 0.9123
    assert d["semantic_similarity"] == 0.8543
    assert d["status"] == "active"


def test_consolidation_report_defaults():
    rep = ConsolidationReport()
    assert rep.episodic_items_analyzed == 0
    assert rep.semantic_facts_generated == 0
    assert rep.contradictions_resolved == 0
    assert rep.compression_ratio == 0.0


def test_contradiction_result():
    res = ContradictionResult(
        has_conflict=True,
        conflicting_item_id="item-abc",
        reason="Antonym conflict detected",
        confidence=0.89,
    )
    assert res.has_conflict is True
    assert res.conflicting_item_id == "item-abc"
