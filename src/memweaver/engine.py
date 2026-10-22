"""Primary MemoryEngine orchestration facade for MemWeaver-AI."""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Dict, List, Optional
from memweaver.config import MemoryConfig
from memweaver.conflict_resolver import ConflictResolver
from memweaver.consolidation import ConsolidationEngine
from memweaver.decay import CognitiveDecayEngine
from memweaver.embeddings import EmbeddingEngine
from memweaver.episodic_memory import EpisodicMemory
from memweaver.models import (
    ConsolidationReport,
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


class MemoryEngine:
    """Production cognitive long-term memory engine for AI agents.
    
    Seamlessly orchestrates three tiers of memory:
    1. Working Memory: Immediate attention scratchpad.
    2. Episodic Memory: Autobiographical interaction logs with Ebbinghaus decay.
    3. Semantic Memory: Consolidated factual knowledge with conflict resolution.
    """

    def __init__(self, config: Optional[MemoryConfig] = None, storage: Optional[BaseStorage] = None):
        self.config = config or MemoryConfig()
        self.embedder = EmbeddingEngine(
            dimension=self.config.embedding_dimension,
            offline_mode=self.config.offline_mode,
            api_key=self.config.gemini_api_key,
        )
        self.decay = CognitiveDecayEngine(self.config)
        self.conflict_resolver = ConflictResolver(self.config, self.embedder)

        # Storage backend
        if storage is not None:
            self.storage = storage
        elif self.config.storage_path:
            self.storage = SQLiteStorage(self.config.storage_path)
        else:
            self.storage = InMemoryStorage()

        # Memory tiers
        self.working = WorkingMemory(self.config)
        self.episodic = EpisodicMemory(self.config, self.embedder, self.decay)
        self.semantic = SemanticMemory(self.config, self.embedder)
        self.consolidation = ConsolidationEngine(self.config, self.semantic, self.conflict_resolver)

        # Telemetry metrics
        self._retrieval_latencies: List[float] = []
        self._consolidation_count: int = 0

        # Load existing items from storage if any
        self._hydrate_from_storage()

    def _hydrate_from_storage(self) -> None:
        """Hydrates in-memory tiers from persistent storage backend."""
        stored = self.storage.get_all()
        for itm in stored:
            if itm.tier == MemoryTier.WORKING:
                self.working.add(itm)
            elif itm.tier == MemoryTier.EPISODIC:
                self.episodic._items.append(itm)
            elif itm.tier == MemoryTier.SEMANTIC:
                self.semantic._items.append(itm)

    def remember(
        self,
        content: str,
        tier: MemoryTier = MemoryTier.EPISODIC,
        memory_type: MemoryType = MemoryType.OBSERVATION,
        importance: float = 0.5,
        session_id: str = "default",
        agent_id: str = "agent_default",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryItem:
        """Stores a new memory into the specified cognitive tier with vector embedding."""
        item = MemoryItem(
            content=content,
            tier=tier,
            memory_type=memory_type,
            importance=importance,
            session_id=session_id,
            agent_id=agent_id,
            tags=tags or [],
            metadata=metadata or {},
            vector=self.embedder.embed(content),
        )

        # If semantic or high importance, perform belief revision against active semantic items
        if tier == MemoryTier.SEMANTIC or memory_type in (MemoryType.PREFERENCE, MemoryType.FACT):
            self.conflict_resolver.detect_and_resolve(item, self.semantic.get_all())

        if tier == MemoryTier.WORKING:
            stored_item = self.working.add(item)
        elif tier == MemoryTier.EPISODIC:
            stored_item = self.episodic.record(item)
        elif tier == MemoryTier.SEMANTIC:
            self.semantic._items.append(item)
            stored_item = item
        else:
            stored_item = self.episodic.record(item)

        self.storage.save(stored_item)
        return stored_item

    def recall(
        self,
        query: str,
        k: int = 5,
        tier: Optional[MemoryTier] = None,
        session_id: Optional[str] = None,
        min_score: float = 0.05,
    ) -> List[RetrievalResult]:
        """Hybrid multi-tier memory search ranking by semantic similarity, recency, and priority."""
        t0 = time.time()
        candidates: List[RetrievalResult] = []

        if tier is None or tier == MemoryTier.EPISODIC:
            candidates.extend(self.episodic.search(query, k=k * 2, session_id=session_id, min_score=min_score))

        if tier is None or tier == MemoryTier.SEMANTIC:
            candidates.extend(self.semantic.search(query, k=k * 2, min_similarity=min_score))

        # Rank all candidates by composite score
        candidates.sort(key=lambda r: r.score, reverse=True)
        top_k = candidates[:k]

        # Record operational latency
        latency_ms = (time.time() - t0) * 1000.0
        self._retrieval_latencies.append(latency_ms)

        # Update storage access stats
        for r in top_k:
            self.storage.update(r.item)

        return top_k

    def consolidate(self, batch_size: Optional[int] = None) -> ConsolidationReport:
        """Executes cognitive consolidation: abstracts episodic interactions into semantic facts."""
        limit = batch_size or self.config.consolidation_batch_size
        unconsolidated = self.episodic.get_unconsolidated(limit=limit)
        report = self.consolidation.consolidate(unconsolidated)

        # Persist newly created semantic facts and update episodic states
        for ep in unconsolidated:
            self.storage.update(ep)
        for sem_id in report.new_semantic_ids:
            sem_item = self.semantic.get_by_id(sem_id)
            if sem_item:
                self.storage.save(sem_item)

        self._consolidation_count += 1
        return report

    def get_context_prompt(self, query: Optional[str] = None, k: int = 4) -> str:
        """Builds an injection-ready prompt block containing working context + relevant long-term memories."""
        sections = []

        # 1. Working memory block
        working_block = self.working.format_for_prompt()
        if working_block:
            sections.append(working_block)

        # 2. Recalled long-term memories if query is provided
        if query:
            recalled = self.recall(query, k=k)
            if recalled:
                lines = ["### Relevant Long-Term Memories (Episodic & Semantic):"]
                for idx, r in enumerate(recalled, 1):
                    lines.append(
                        f"[{idx}] ({r.item.tier.value.upper()} | {r.item.memory_type.value}): "
                        f"{r.item.content} (Score: {r.score:.2f})"
                    )
                sections.append("\n".join(lines))

        return "\n\n".join(sections)

    def get_stats(self) -> MemoryStats:
        """Returns aggregate operational metrics and telemetry."""
        all_items = self.storage.get_all() if self.config.storage_path else (
            self.working.get_all() + self.episodic.get_all() + self.semantic.get_all()
        )
        avg_lat = (
            sum(self._retrieval_latencies) / len(self._retrieval_latencies)
            if self._retrieval_latencies
            else 0.0
        )
        superseded = sum(1 for itm in all_items if itm.status == MemoryStatus.SUPERSEDED)

        return MemoryStats(
            total_memories=len(all_items),
            working_count=self.working.count(),
            episodic_count=self.episodic.count(),
            semantic_count=self.semantic.count(active_only=True),
            superseded_count=superseded,
            retrieval_count=len(self._retrieval_latencies),
            average_retrieval_latency_ms=round(avg_lat, 3),
            consolidation_cycles=self._consolidation_count,
        )

    def remember_call(
        self,
        memory_type: MemoryType = MemoryType.OBSERVATION,
        importance: float = 0.5,
    ) -> Callable:
        """Decorator to automatically record function inputs, outputs, and errors as episodic memories."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                fn_name = func.__name__
                arg_summary = f"args={args[:3]}, kwargs={list(kwargs.keys())}"
                try:
                    result = func(*args, **kwargs)
                    content = f"Tool execution '{fn_name}' completed. Inputs: {arg_summary}. Result: {str(result)[:200]}"
                    self.remember(
                        content=content,
                        tier=MemoryTier.EPISODIC,
                        memory_type=memory_type,
                        importance=importance,
                        tags=["tool_call", fn_name],
                    )
                    return result
                except Exception as exc:
                    content = f"Tool execution '{fn_name}' failed with error: {str(exc)}"
                    self.remember(
                        content=content,
                        tier=MemoryTier.EPISODIC,
                        memory_type=MemoryType.OBSERVATION,
                        importance=0.9,
                        tags=["tool_failure", fn_name],
                    )
                    raise exc
            return wrapper
        return decorator
