"""Tests for Tier 1: Working Memory."""

import pytest
from memweaver.config import MemoryConfig
from memweaver.models import MemoryItem, MemoryTier, MemoryType
from memweaver.working_memory import WorkingMemory


def test_working_memory_add_and_count():
    config = MemoryConfig(working_memory_max_items=5)
    wm = WorkingMemory(config)

    item1 = MemoryItem(content="First thought", importance=0.5)
    wm.add(item1)
    assert wm.count() == 1
    assert wm.get_all()[0].tier == MemoryTier.WORKING


def test_working_memory_capacity_eviction_by_count():
    # Max 3 items
    config = MemoryConfig(working_memory_max_items=3)
    wm = WorkingMemory(config)

    wm.add(MemoryItem(content="Item 1 (low)", importance=0.1))
    wm.add(MemoryItem(content="Item 2 (high)", importance=0.9))
    wm.add(MemoryItem(content="Item 3 (med)", importance=0.5))
    assert wm.count() == 3

    # Add 4th item -> should evict lowest importance (Item 1)
    wm.add(MemoryItem(content="Item 4 (new)", importance=0.8))
    assert wm.count() == 3
    contents = [i.content for i in wm.get_all()]
    assert "Item 1 (low)" not in contents
    assert "Item 2 (high)" in contents
    assert "Item 4 (new)" in contents


def test_working_memory_token_capacity_eviction():
    config = MemoryConfig(working_memory_max_items=10, working_memory_max_tokens=20)
    wm = WorkingMemory(config)

    # 1 token estimate is len // 4; 40 characters is ~10 tokens
    wm.add(MemoryItem(content="A" * 40))
    wm.add(MemoryItem(content="B" * 40))
    # Adding third pushes total tokens beyond 20
    wm.add(MemoryItem(content="C" * 40))
    assert wm.total_tokens() <= config.working_memory_max_tokens or wm.count() == 1


def test_working_memory_format_for_prompt():
    config = MemoryConfig()
    wm = WorkingMemory(config)
    wm.add(MemoryItem(content="User wants data in CSV", memory_type=MemoryType.GOAL))
    wm.add(MemoryItem(content="Loaded dataset containing 10,000 rows", memory_type=MemoryType.OBSERVATION))

    prompt = wm.format_for_prompt()
    assert "Current Working Memory Context:" in prompt
    assert "User wants data in CSV" in prompt
    assert "Loaded dataset containing 10,000 rows" in prompt


def test_working_memory_clear():
    config = MemoryConfig()
    wm = WorkingMemory(config)
    wm.add(MemoryItem(content="Scratch note"))
    assert wm.count() == 1
    cleared = wm.clear()
    assert cleared == 1
    assert wm.count() == 0
