"""Tier 3: Semantic Memory (Consolidated facts, knowledge graph triples & beliefs)."""

from __future__ import annotations

from typing import List, Optional
from memweaver.config import MemoryConfig
from memweaver.embeddings import EmbeddingEngine
from memweaver.models import MemoryItem, MemoryStatus, MemoryTier, MemoryType, RetrievalResult


class SemanticMemory:
    """Manages long-term crystallized knowledge, facts, and persistent preferences.
    
    Semantic memories have high stability and represent distilled truths about the
    user, system, or world. They are continuously updated and audited for contradictions.
    """

    def __init__(self, config: MemoryConfig, embedder: EmbeddingEngine):
        self.config = config
        self.embedder = embedder
        self._items: List[MemoryItem] = []

    def store_fact(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.FACT,
        importance: float = 0.8,
        tags: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
        confidence: float = 1.0,
    ) -> MemoryItem:
        """Stores a verified semantic fact with embedding."""
        item = MemoryItem(
            content=content,
            tier=MemoryTier.SEMANTIC,
            memory_type=memory_type,
            importance=importance,
            tags=tags or [],
            metadata=metadata or {},
            confidence=confidence,
            vector=self.embedder.embed(content),
        )
        self._items.append(item)
        return item

    def search(
        self,
        query: str,
        k: int = 5,
        include_superseded: bool = False,
        min_similarity: float = 0.20,
    ) -> List[RetrievalResult]:
        """Searches long-term semantic knowledge based on dense semantic similarity."""
        query_vec = self.embedder.embed(query)
        results: List[RetrievalResult] = []

        for itm in self._items:
            if not include_superseded and itm.status != MemoryStatus.ACTIVE:
                continue

            sim = self.embedder.cosine_similarity(query_vec, itm.vector or [])
            if sim >= min_similarity:
                # In semantic memory, score is primarily semantic similarity modulated by importance
                score = (0.75 * max(0.0, (sim + 1.0) / 2.0)) + (0.25 * itm.importance)
                results.append(
                    RetrievalResult(
                        item=itm,
                        score=score,
                        semantic_similarity=sim,
                        recency_score=1.0,  # Semantic facts don't decay aggressively
                        importance_score=itm.importance,
                    )
                )

        results.sort(key=lambda r: r.score, reverse=True)
        top = results[:k]
        for r in top:
            r.item.touch()
        return top

    def get_by_id(self, item_id: str) -> Optional[MemoryItem]:
        for itm in self._items:
            if itm.id == item_id:
                return itm
        return None

    def get_active(self) -> List[MemoryItem]:
        return [itm for itm in self._items if itm.status == MemoryStatus.ACTIVE]

    def count(self, active_only: bool = True) -> int:
        if active_only:
            return sum(1 for itm in self._items if itm.status == MemoryStatus.ACTIVE)
        return len(self._items)

    def get_all(self) -> List[MemoryItem]:
        return list(self._items)
