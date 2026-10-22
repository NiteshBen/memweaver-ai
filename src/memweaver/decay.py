"""Cognitive retention and Ebbinghaus forgetting curve decay calculator."""

from __future__ import annotations

import math
import time
from typing import List, Tuple
from memweaver.config import MemoryConfig
from memweaver.models import MemoryItem


class CognitiveDecayEngine:
    """Calculates memory retention, temporal decay, and access frequency boost.
    
    Grounds agent memory in cognitive science principles:
    - Ebbinghaus exponential forgetting curve.
    - Memory strength/stability increases with importance and repeated recall.
    - Fresh memories or frequently recalled memories resist decay.
    """

    def __init__(self, config: MemoryConfig):
        self.config = config

    def calculate_recency_score(self, item: MemoryItem, current_time: float | None = None) -> float:
        """Computes exponential retention score R in [0.0, 1.0]."""
        now = current_time if current_time is not None else time.time()
        elapsed_seconds = max(0.0, now - item.last_accessed_at)

        # Baseline stability (S0) calibrated to target half-life
        s0 = self.config.decay_half_life_seconds / math.log(2.0)

        # Modulate stability based on intrinsic importance and access frequency
        # Repeated accesses strengthen consolidation (spaced repetition effect)
        importance_boost = 1.0 + (item.importance * 1.5)
        access_boost = 1.0 + (item.access_count * self.config.access_boost_factor)
        stability = s0 * importance_boost * access_boost

        decay_arg = (elapsed_seconds / stability) ** self.config.decay_power
        retention = math.exp(-decay_arg)
        return max(0.0, min(1.0, retention))

    def calculate_hybrid_score(
        self,
        semantic_sim: float,
        recency: float,
        importance: float,
    ) -> float:
        """Combines semantic similarity, temporal recency, and priority."""
        # Normalize semantic similarity from [-1, 1] to [0, 1]
        norm_sem = max(0.0, (semantic_sim + 1.0) / 2.0)
        
        score = (
            self.config.weight_semantic * norm_sem
            + self.config.weight_recency * recency
            + self.config.weight_importance * importance
        )
        return max(0.0, min(1.0, score))

    def generate_decay_curve(
        self,
        item: MemoryItem,
        horizon_hours: float = 72.0,
        steps: int = 25,
    ) -> List[Tuple[float, float]]:
        """Simulates retention curve for UI visualizer and graphs."""
        points: List[Tuple[float, float]] = []
        base_time = item.last_accessed_at
        step_seconds = (horizon_hours * 3600.0) / max(1, steps - 1)

        for i in range(steps):
            sim_time = base_time + (i * step_seconds)
            sim_hours = (i * step_seconds) / 3600.0
            r = self.calculate_recency_score(item, current_time=sim_time)
            points.append((round(sim_hours, 1), round(r, 4)))
        return points
