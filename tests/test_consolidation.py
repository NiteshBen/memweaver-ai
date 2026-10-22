"""Tests for Autonomous Memory Consolidation."""

import pytest
from memweaver.config import MemoryConfig
from memweaver.conflict_resolver import ConflictResolver
from memweaver.consolidation import ConsolidationEngine
from memweaver.embeddings import EmbeddingEngine
from memweaver.models import MemoryItem, MemoryTier, MemoryType
from memweaver.semantic_memory import SemanticMemory


@pytest.fixture
def consolidation_setup():
    config = MemoryConfig()
    embedder = EmbeddingEngine(dimension=64)
    semantic = SemanticMemory(config, embedder)
    resolver = ConflictResolver(config, embedder)
    engine = ConsolidationEngine(config, semantic, resolver)
    return engine, semantic


def test_consolidation_empty_items(consolidation_setup):
    engine, semantic = consolidation_setup
    report = engine.consolidate([])
    assert report.episodic_items_analyzed == 0
    assert report.semantic_facts_generated == 0
    assert report.compression_ratio == 1.0


def test_consolidation_extracts_preferences_and_goals(consolidation_setup):
    engine, semantic = consolidation_setup

    items = [
        MemoryItem(content="In this conversation, user prefers pytest over unittest for testing", tier=MemoryTier.EPISODIC),
        MemoryItem(content="The user goal is to build a production portfolio project in Python", tier=MemoryTier.EPISODIC),
        MemoryItem(content="Random casual chitchat hello there how are you", tier=MemoryTier.EPISODIC, importance=0.2),
    ]

    report = engine.consolidate(items)
    assert report.episodic_items_analyzed == 3
    assert report.semantic_facts_generated >= 2
    assert semantic.count() >= 2
    # Check that episodic items were marked as consolidated
    assert items[0].metadata.get("consolidated") is True
    assert items[1].metadata.get("consolidated") is True


def test_consolidation_resolves_contradictions_during_cycle(consolidation_setup):
    engine, semantic = consolidation_setup

    # Pre-populate existing semantic fact
    semantic.store_fact("User preference/fact: user prefers dark theme", memory_type=MemoryType.PREFERENCE)

    # Consolidate an opposing episodic memory
    new_episodic = [
        MemoryItem(content="User prefers light theme for all dashboards", tier=MemoryTier.EPISODIC)
    ]
    report = engine.consolidate(new_episodic)
    assert report.contradictions_resolved >= 1
