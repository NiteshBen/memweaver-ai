"""Tier 1: Working Memory (Immediate scratchpad and attention buffer)."""

from __future__ import annotations

from typing import List, Optional
from memweaver.config import MemoryConfig
from memweaver.models import MemoryItem, MemoryTier


class WorkingMemory:
    """Manages short-term working memory with token and item capacity budgeting.
    
    Serves as the high-speed active scratchpad for an ongoing agent reasoning loop.
    Enforces capacity limits through priority-aware FIFO eviction.
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        self._items: List[MemoryItem] = []

    def add(self, item: MemoryItem) -> MemoryItem:
        """Adds an item into working memory, evicting if capacity is exceeded."""
        item.tier = MemoryTier.WORKING
        self._items.append(item)
        self._enforce_capacity()
        return item

    def get_all(self) -> List[MemoryItem]:
        """Returns all active items in working memory in insertion order."""
        return list(self._items)

    def count(self) -> int:
        return len(self._items)

    def total_tokens(self) -> int:
        return sum(item.token_estimate for item in self._items)

    def clear(self) -> int:
        """Clears working memory and returns count of cleared items."""
        cnt = len(self._items)
        self._items.clear()
        return cnt

    def format_for_prompt(self, prefix: str = "### Current Working Memory Context:\n") -> str:
        """Formats active working items as a system prompt block."""
        if not self._items:
            return ""
        lines = [prefix]
        for idx, itm in enumerate(self._items, 1):
            lines.append(f"[{idx}] ({itm.memory_type.value.upper()}): {itm.content}")
        return "\n".join(lines)

    def _enforce_capacity(self) -> None:
        """Evicts lower priority/oldest items when item or token limits are breached."""
        # Evict by count
        while len(self._items) > self.config.working_memory_max_items:
            # Find item with lowest importance (FIFO tie-breaker)
            lowest_idx = 0
            lowest_imp = self._items[0].importance
            for i in range(1, len(self._items)):
                if self._items[i].importance < lowest_imp:
                    lowest_imp = self._items[i].importance
                    lowest_idx = i
            self._items.pop(lowest_idx)

        # Evict by token budget
        while self.total_tokens() > self.config.working_memory_max_tokens and len(self._items) > 1:
            self._items.pop(0)
