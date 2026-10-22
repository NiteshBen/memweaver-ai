"""Autonomous Memory Consolidation Engine (Episodic-to-Semantic Synthesis)."""

from __future__ import annotations

import re
import time
from typing import List, Optional
from memweaver.config import MemoryConfig
from memweaver.conflict_resolver import ConflictResolver
from memweaver.models import ConsolidationReport, MemoryItem, MemoryTier, MemoryType
from memweaver.semantic_memory import SemanticMemory


class ConsolidationEngine:
    """Consolidates granular episodic memories into crystallized semantic knowledge.
    
    Operates like cognitive slow-wave sleep: replaying episodic events, clustering
    common themes, abstracting user habits and preferences, and reconciling conflicts.
    """

    def __init__(
        self,
        config: MemoryConfig,
        semantic_memory: SemanticMemory,
        conflict_resolver: ConflictResolver,
    ):
        self.config = config
        self.semantic = semantic_memory
        self.conflict_resolver = conflict_resolver

    def consolidate(self, episodic_items: List[MemoryItem]) -> ConsolidationReport:
        """Runs a consolidation pass over unconsolidated episodic items."""
        start_time = time.time()
        if not episodic_items:
            return ConsolidationReport(
                episodic_items_analyzed=0,
                semantic_facts_generated=0,
                contradictions_resolved=0,
                compression_ratio=1.0,
                duration_ms=0.0,
            )

        extracted_facts: List[str] = []
        new_semantic_ids: List[str] = []
        contradictions_resolved = 0

        # Pattern extractors for offline deterministic synthesis
        patterns = [
            (r"(?:i prefer|user prefers|likes)\s+([^.]+)", MemoryType.PREFERENCE, 0.85),
            (r"(?:i want to|goal is to|target is to)\s+([^.]+)", MemoryType.GOAL, 0.90),
            (r"(?:is located in|lives in|works at)\s+([^.]+)", MemoryType.FACT, 0.80),
            (r"(?:always use|never use|default to)\s+([^.]+)", MemoryType.PREFERENCE, 0.85),
            (r"(?:the answer is|result was)\s+([^.]+)", MemoryType.FACT, 0.70),
        ]

        for ep in episodic_items:
            text = ep.content
            matched = False

            for regex, mtype, imp in patterns:
                m = re.search(regex, text, re.IGNORECASE)
                if m:
                    fact_str = f"User preference/fact: {m.group(0).strip()}"
                    extracted_facts.append(fact_str)
                    
                    new_item = MemoryItem(
                        content=fact_str,
                        tier=MemoryTier.SEMANTIC,
                        memory_type=mtype,
                        importance=imp,
                        tags=["consolidated", mtype.value],
                        session_id=ep.session_id,
                        agent_id=ep.agent_id,
                    )
                    
                    # Audit for contradictions with existing semantic items
                    c_res = self.conflict_resolver.detect_and_resolve(
                        new_item, self.semantic.get_all()
                    )
                    if c_res.has_conflict:
                        contradictions_resolved += 1

                    stored = self.semantic.store_fact(
                        content=new_item.content,
                        memory_type=new_item.memory_type,
                        importance=new_item.importance,
                        tags=new_item.tags,
                    )
                    new_semantic_ids.append(stored.id)
                    matched = True
                    break

            # If no pattern matched, but item is marked high-importance, summarize directly
            if not matched and ep.importance >= 0.75:
                fact_str = f"Consolidated observation: {ep.content}"
                new_item = MemoryItem(
                    content=fact_str,
                    tier=MemoryTier.SEMANTIC,
                    memory_type=MemoryType.FACT,
                    importance=ep.importance,
                    tags=["consolidated", "high_importance"],
                )
                c_res = self.conflict_resolver.detect_and_resolve(
                    new_item, self.semantic.get_all()
                )
                if c_res.has_conflict:
                    contradictions_resolved += 1
                
                stored = self.semantic.store_fact(
                    content=new_item.content,
                    memory_type=new_item.memory_type,
                    importance=new_item.importance,
                    tags=new_item.tags,
                )
                new_semantic_ids.append(stored.id)

            # Mark episodic item as consolidated
            ep.metadata["consolidated"] = True

        duration_ms = (time.time() - start_time) * 1000.0
        n_analyzed = len(episodic_items)
        n_generated = len(new_semantic_ids)
        compression = round(n_analyzed / max(1, n_generated), 2) if n_generated > 0 else 1.0

        return ConsolidationReport(
            episodic_items_analyzed=n_analyzed,
            semantic_facts_generated=n_generated,
            contradictions_resolved=contradictions_resolved,
            compression_ratio=compression,
            duration_ms=round(duration_ms, 2),
            new_semantic_ids=new_semantic_ids,
        )
