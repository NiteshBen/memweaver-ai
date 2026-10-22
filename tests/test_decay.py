"""Tests for Ebbinghaus cognitive decay engine."""

import time
import pytest
from memweaver.config import MemoryConfig
from memweaver.decay import CognitiveDecayEngine
from memweaver.models import MemoryItem


def test_recency_score_fresh_memory():
    config = MemoryConfig(decay_half_life_seconds=3600.0)
    engine = CognitiveDecayEngine(config)
    item = MemoryItem(content="Fresh observation", importance=0.5)

    r = engine.calculate_recency_score(item, current_time=item.last_accessed_at)
    assert 0.99 <= r <= 1.0


def test_recency_score_at_half_life():
    half_life = 3600.0
    config = MemoryConfig(decay_half_life_seconds=half_life, access_boost_factor=0.0)
    engine = CognitiveDecayEngine(config)
    item = MemoryItem(content="Test half life", importance=0.0)  # zero importance baseline

    # Advance time by exactly one half life
    r = engine.calculate_recency_score(item, current_time=item.last_accessed_at + half_life)
    # Retention should be close to 0.5 (exp(-ln(2)) = 0.5)
    assert 0.45 <= r <= 0.55


def test_importance_and_access_boost_slows_decay():
    config = MemoryConfig(decay_half_life_seconds=1000.0)
    engine = CognitiveDecayEngine(config)

    item_low = MemoryItem(content="Low priority", importance=0.1)
    item_high = MemoryItem(content="High priority", importance=0.9)
    item_high.access_count = 5  # Frequently recalled

    elapsed = 2000.0
    sim_time = item_low.last_accessed_at + elapsed

    r_low = engine.calculate_recency_score(item_low, current_time=sim_time)
    r_high = engine.calculate_recency_score(item_high, current_time=sim_time)

    assert r_high > r_low


def test_hybrid_score_calculation():
    config = MemoryConfig(weight_semantic=0.5, weight_recency=0.3, weight_importance=0.2)
    engine = CognitiveDecayEngine(config)

    score = engine.calculate_hybrid_score(semantic_sim=1.0, recency=1.0, importance=1.0)
    assert 0.98 <= score <= 1.0

    score_low = engine.calculate_hybrid_score(semantic_sim=0.0, recency=0.2, importance=0.1)
    assert score_low < score


def test_decay_curve_generation():
    config = MemoryConfig()
    engine = CognitiveDecayEngine(config)
    item = MemoryItem(content="Curve test", importance=0.7)

    curve = engine.generate_decay_curve(item, horizon_hours=48.0, steps=10)
    assert len(curve) == 10
    # First point at 0h should have retention ~1.0
    assert curve[0][0] == 0.0
    assert curve[0][1] > 0.95
    # Retention should decrease monotonically
    for i in range(1, len(curve)):
        assert curve[i][1] <= curve[i - 1][1]
