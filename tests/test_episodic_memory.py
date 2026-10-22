"""Tests for Tier 2: Episodic Memory."""

import pytest
from memweaver.config import MemoryConfig
from memweaver.decay import CognitiveDecayEngine
from memweaver.embeddings import EmbeddingEngine
from memweaver.episodic_memory import EpisodicMemory
from memweaver.models import MemoryItem, MemoryStatus, MemoryTier


@pytest.fixture
def episodic_setup():
    config = MemoryConfig()
    embedder = EmbeddingEngine(dimension=64)
    decay = CognitiveDecayEngine(config)
    ep = EpisodicMemory(config, embedder, decay)
    return ep, config


def test_episodic_record_and_embedding(episodic_setup):
    ep, config = episodic_setup
    item = MemoryItem(content="User executed SQL SELECT query on database")
    recorded = ep.record(item)

    assert recorded.tier == MemoryTier.EPISODIC
    assert recorded.vector is not None
    assert len(recorded.vector) == 64
    assert ep.count() == 1


def test_episodic_search_ranking(episodic_setup):
    ep, config = episodic_setup
    ep.record(MemoryItem(content="Python pandas dataframe groupby aggregation", importance=0.8))
    ep.record(MemoryItem(content="C++ pointer arithmetic and memory leaks", importance=0.4))
    ep.record(MemoryItem(content="Machine learning scikit-learn model evaluation", importance=0.7))

    results = ep.search("Python pandas dataframe analysis", k=2)
    assert len(results) == 2
    # The pandas memory should rank first
    assert "pandas" in results[0].item.content
    assert results[0].score > results[1].score
    # Access metadata should update
    assert results[0].item.access_count == 1


def test_episodic_unconsolidated_filtering(episodic_setup):
    ep, config = episodic_setup
    item1 = ep.record(MemoryItem(content="Interaction turn 1"))
    item2 = ep.record(MemoryItem(content="Interaction turn 2"))

    unconsolidated = ep.get_unconsolidated()
    assert len(unconsolidated) == 2

    # Mark one as consolidated
    item1.metadata["consolidated"] = True
    unconsolidated_after = ep.get_unconsolidated()
    assert len(unconsolidated_after) == 1
    assert unconsolidated_after[0].id == item2.id
