"""Integration tests for the primary MemoryEngine orchestration facade."""

import pytest
from memweaver.config import MemoryConfig
from memweaver.engine import MemoryEngine
from memweaver.models import MemoryStatus, MemoryTier, MemoryType


def test_engine_remember_and_recall_hybrid():
    engine = MemoryEngine()

    engine.remember(
        content="User frequently analyzes CSV files with Pandas in Python",
        tier=MemoryTier.EPISODIC,
        memory_type=MemoryType.OBSERVATION,
        importance=0.8,
    )
    engine.remember(
        content="System memory limit is set to 8GB",
        tier=MemoryTier.SEMANTIC,
        memory_type=MemoryType.FACT,
        importance=0.9,
    )

    recalled = engine.recall("Pandas dataframe in Python", k=2)
    assert len(recalled) >= 1
    assert "Pandas" in recalled[0].item.content
    assert recalled[0].score > 0.1


def test_engine_prompt_context_assembly():
    engine = MemoryEngine()
    engine.remember("User wants to build high quality portfolio", tier=MemoryTier.WORKING, memory_type=MemoryType.GOAL)
    engine.remember("PostgreSQL connection string host: localhost", tier=MemoryTier.SEMANTIC, memory_type=MemoryType.FACT)

    prompt = engine.get_context_prompt(query="PostgreSQL database", k=1)
    assert "Current Working Memory Context:" in prompt
    assert "User wants to build high quality portfolio" in prompt
    assert "Relevant Long-Term Memories" in prompt
    assert "PostgreSQL connection string" in prompt


def test_engine_consolidation_integration():
    engine = MemoryEngine()
    engine.remember("User prefers Python over Java for backend services", tier=MemoryTier.EPISODIC)
    engine.remember("User goal is to deploy Docker microservices", tier=MemoryTier.EPISODIC)

    report = engine.consolidate()
    assert report.episodic_items_analyzed == 2
    assert report.semantic_facts_generated >= 1

    stats = engine.get_stats()
    assert stats.consolidation_cycles == 1
    assert stats.semantic_count >= 1


def test_engine_remember_call_decorator():
    engine = MemoryEngine()

    @engine.remember_call(memory_type=MemoryType.OBSERVATION, importance=0.7)
    def fetch_weather(city: str) -> str:
        return f"Weather in {city} is Sunny, 24C"

    res = fetch_weather("Jaipur")
    assert "Sunny" in res

    # Check that episodic memory was automatically recorded
    episodic_items = engine.episodic.get_all()
    assert len(episodic_items) == 1
    assert "fetch_weather" in episodic_items[0].content
    assert "Jaipur" in episodic_items[0].content


def test_engine_decorator_records_exception():
    engine = MemoryEngine()

    @engine.remember_call()
    def failing_tool():
        raise ValueError("Simulated network timeout")

    with pytest.raises(ValueError):
        failing_tool()

    episodic_items = engine.episodic.get_all()
    assert len(episodic_items) == 1
    assert "failing_tool" in episodic_items[0].content
    assert "Simulated network timeout" in episodic_items[0].content
