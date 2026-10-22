"""Tests for Truth Maintenance & Conflict Resolution."""

import pytest
from memweaver.config import MemoryConfig
from memweaver.conflict_resolver import ConflictResolver
from memweaver.embeddings import EmbeddingEngine
from memweaver.models import MemoryItem, MemoryStatus, MemoryTier, MemoryType


@pytest.fixture
def conflict_setup():
    config = MemoryConfig(enable_belief_revision=True, contradiction_similarity_threshold=0.65)
    embedder = EmbeddingEngine(dimension=64)
    resolver = ConflictResolver(config, embedder)
    return resolver, embedder


def test_conflict_resolution_antonym_state_transition(conflict_setup):
    resolver, embedder = conflict_setup

    old_item = MemoryItem(
        content="User preferred dark theme for interface",
        tier=MemoryTier.SEMANTIC,
        memory_type=MemoryType.PREFERENCE,
        status=MemoryStatus.ACTIVE,
    )
    old_item.vector = embedder.embed(old_item.content)

    new_item = MemoryItem(
        content="User switched to light theme for interface",
        tier=MemoryTier.SEMANTIC,
        memory_type=MemoryType.PREFERENCE,
    )
    new_item.vector = embedder.embed(new_item.content)

    result = resolver.detect_and_resolve(new_item, [old_item])
    assert result.has_conflict is True
    assert result.conflicting_item_id == old_item.id
    # Old item should be marked SUPERSEDED
    assert old_item.status == MemoryStatus.SUPERSEDED
    assert old_item.superseded_by == new_item.id
    assert new_item.supersedes_id == old_item.id


def test_conflict_resolution_explicit_revision_phrase(conflict_setup):
    resolver, embedder = conflict_setup

    old_item = MemoryItem(
        content="User works in Chicago office location",
        tier=MemoryTier.SEMANTIC,
        status=MemoryStatus.ACTIVE,
    )
    old_item.vector = embedder.embed(old_item.content)

    new_item = MemoryItem(
        content="User moved to New York office location",
        tier=MemoryTier.SEMANTIC,
    )
    new_item.vector = embedder.embed(new_item.content)

    result = resolver.detect_and_resolve(new_item, [old_item])
    assert result.has_conflict is True
    assert old_item.status == MemoryStatus.SUPERSEDED


def test_conflict_no_conflict_on_different_topics(conflict_setup):
    resolver, embedder = conflict_setup

    item1 = MemoryItem(
        content="User prefers Python for machine learning workflows",
        tier=MemoryTier.SEMANTIC,
        status=MemoryStatus.ACTIVE,
    )
    item1.vector = embedder.embed(item1.content)

    item2 = MemoryItem(
        content="Server memory is 64GB DDR5 ECC RAM",
        tier=MemoryTier.SEMANTIC,
    )
    item2.vector = embedder.embed(item2.content)

    result = resolver.detect_and_resolve(item2, [item1])
    assert result.has_conflict is False
    assert item1.status == MemoryStatus.ACTIVE
