"""Tier 2: Episodic Memory (Temporal interaction log & autobiographical history)."""

from __future__ import annotations

from typing import List, Optional
from memweaver.config import MemoryConfig
from memweaver.decay import CognitiveDecayEngine
from memweaver.embeddings import EmbeddingEngine
from memweaver.models import MemoryItem, MemoryStatus, MemoryTier, RetrievalResult


class EpisodicMemory:
    """Manages sequential autobiographical interactions, tool runs, and dialogues.
    
    Items are organized chronologically, decayed via the Ebbinghaus forgetting curve,
    and queried through hybrid semantic + recency ranking.
    """

    def __init__(self, config: MemoryConfig, embedder: EmbeddingEngine, decay: CognitiveDecayEngine):
        self.config = config
        self.embedder = embedder
        self.decay = decay
        self._items: List[MemoryItem] = []

    def record(self, item: MemoryItem) -> MemoryItem:
        """Stores an episodic event, calculating vector embeddings if missing."""
        item.tier = MemoryTier.EPISODIC
        if item.vector is None:
            item.vector = self.embedder.embed(item.content)
        self._items.append(item)
        return item

    def search(
        self,
        query: str,
        k: int = 5,
        session_id: Optional[str] = None,
        min_score: float = 0.05,
    ) -> List[RetrievalResult]:
        """Hybrid retrieval across episodic memories using semantic + recency + importance."""
        query_vec = self.embedder.embed(query)
        results: List[RetrievalResult] = []

        for itm in self._items:
            if itm.status != MemoryStatus.ACTIVE:
                continue
            if session_id and itm.session_id != session_id:
                continue

            sim = self.embedder.cosine_similarity(query_vec, itm.vector or [])
            recency = self.decay.calculate_recency_score(itm)
            score = self.decay.calculate_hybrid_score(sim, recency, itm.importance)

            if score >= min_score:
                results.append(
                    RetrievalResult(
                        item=itm,
                        score=score,
                        semantic_similarity=sim,
                        recency_score=recency,
                        importance_score=itm.importance,
                    )
                )

        results.sort(key=lambda r: r.score, reverse=True)
        top = results[:k]
        for r in top:
            r.item.touch()
        return top

    def get_unconsolidated(self, limit: int = 50) -> List[MemoryItem]:
        """Returns active episodic memories that have not yet been consolidated into semantic facts."""
        unconsolidated = [
            itm for itm in self._items
            if itm.status == MemoryStatus.ACTIVE and not itm.metadata.get("consolidated", False)
        ]
        return unconsolidated[:limit]

    def count(self) -> int:
        return len(self._items)

    def get_all(self) -> List[MemoryItem]:
        return list(self._items)
