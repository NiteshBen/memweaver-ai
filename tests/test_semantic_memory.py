"""Tests for Tier 3: Semantic Memory."""

import pytest
from memweaver.config import MemoryConfig
from memweaver.embeddings import EmbeddingEngine
from memweaver.models import MemoryStatus, MemoryTier, MemoryType
from memweaver.semantic_memory import SemanticMemory


@pytest.fixture
def semantic_setup():
    config = MemoryConfig()
    embedder = EmbeddingEngine(dimension=64)
    sem = SemanticMemory(config, embedder)
    return sem


def test_semantic_store_fact(semantic_setup):
    sem = semantic_setup
    fact = sem.store_fact(
        content="PostgreSQL database port is 5432",
        memory_type=MemoryType.FACT,
        importance=0.9,
        tags=["database", "network"],
    )
    assert fact.tier == MemoryTier.SEMANTIC
    assert fact.memory_type == MemoryType.FACT
    assert fact.importance == 0.9
    assert fact.vector is not None
    assert sem.count() == 1


def test_semantic_search_active_vs_superseded(semantic_setup):
    sem = semantic_setup
    f1 = sem.store_fact("User prefers dark mode UI theme")
    f2 = sem.store_fact("Application deployed on port 8080")

    # Invalidate f1
    f1.status = MemoryStatus.SUPERSEDED

    # Search active only
    active_res = sem.search("UI theme dark mode", include_superseded=False)
    assert len(active_res) == 0

    # Search including superseded
    all_res = sem.search("UI theme dark mode", include_superseded=True)
    assert len(all_res) >= 1
    assert all_res[0].item.id == f1.id
    assert all_res[0].item.status == MemoryStatus.SUPERSEDED


def test_semantic_get_by_id(semantic_setup):
    sem = semantic_setup
    fact = sem.store_fact("Target audience is AI researchers")
    found = sem.get_by_id(fact.id)
    assert found is not None
    assert found.content == "Target audience is AI researchers"
    assert sem.get_by_id("non-existent-id") is None
