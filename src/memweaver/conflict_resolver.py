"""Belief Revision and Truth Maintenance System for resolving contradictory memories."""

from __future__ import annotations

import re
from typing import List, Optional, Set, Tuple
from memweaver.config import MemoryConfig
from memweaver.embeddings import EmbeddingEngine
from memweaver.models import ContradictionResult, MemoryItem, MemoryStatus


# Antonym pairs and state transitions commonly indicating fact revision
MUTUAL_EXCLUSIONS = [
    ("dark", "light"),
    ("enable", "disable"),
    ("enabled", "disabled"),
    ("true", "false"),
    ("yes", "no"),
    ("remote", "onsite"),
    ("python", "rust"),
    ("mac", "windows"),
    ("linux", "windows"),
    ("morning", "evening"),
    ("active", "inactive"),
    ("high", "low"),
]

STOP_WORDS: Set[str] = {
    "a", "an", "the", "in", "on", "at", "for", "to", "is", "was", "are", "were", "of", "and", "or", "it", "with"
}


class ConflictResolver:
    """Detects when incoming knowledge invalidates prior beliefs and updates state.
    
    Prevents hallucination and cognitive dissonance in AI agents by superseding
    outdated memories while preserving the historical audit trail.
    """

    def __init__(self, config: MemoryConfig, embedder: EmbeddingEngine):
        self.config = config
        self.embedder = embedder

    def detect_and_resolve(
        self,
        new_item: MemoryItem,
        existing_items: List[MemoryItem],
    ) -> ContradictionResult:
        """Audits incoming memory against active memories and supersedes conflicting items."""
        if not self.config.enable_belief_revision:
            return ContradictionResult(has_conflict=False)

        new_vec = new_item.vector or self.embedder.embed(new_item.content)
        new_text_lower = new_item.content.lower()

        for old_item in existing_items:
            if old_item.status != MemoryStatus.ACTIVE:
                continue
            if old_item.id == new_item.id:
                continue

            old_vec = old_item.vector or self.embedder.embed(old_item.content)
            similarity = self.embedder.cosine_similarity(new_vec, old_vec)
            old_text_lower = old_item.content.lower()

            has_opposition, reason = self._check_opposing_values(old_text_lower, new_text_lower)

            # Conflict condition:
            # 1. Antonym/explicit revision pattern present AND sufficient subject overlap (similarity >= 0.35)
            # 2. Or high semantic similarity indicating topic collision (similarity >= threshold)
            is_conflict = False
            if has_opposition and similarity >= 0.35:
                is_conflict = True
            elif similarity >= self.config.contradiction_similarity_threshold and (has_opposition or similarity >= 0.88):
                is_conflict = True

            if is_conflict:
                old_item.mark_superseded(new_item.id)
                new_item.supersedes_id = old_item.id

                return ContradictionResult(
                    has_conflict=True,
                    conflicting_item_id=old_item.id,
                    reason=reason or f"Semantic belief update (cosine similarity: {similarity:.2f})",
                    confidence=similarity,
                )

        return ContradictionResult(has_conflict=False)

    def _check_opposing_values(self, old_text: str, new_text: str) -> Tuple[bool, Optional[str]]:
        """Scans for conflicting antonyms or updated state attributes."""
        old_words = set(re.findall(r"\b\w+\b", old_text))
        new_words = set(re.findall(r"\b\w+\b", new_text))

        for word_a, word_b in MUTUAL_EXCLUSIONS:
            if (word_a in old_words and word_b in new_words) or (word_b in old_words and word_a in new_words):
                return True, f"Conflicting state transition detected: '{word_a}' vs '{word_b}'"

        # Check for phrase updates like "moved to", "switched to", "now prefers", "changed to"
        revision_triggers = ["now prefers", "moved to", "changed to", "switched to", "updated to", "instead of"]
        for trigger in revision_triggers:
            if trigger in new_text:
                # Ensure they share non-stopword tokens (e.g. subject / entity)
                shared_content_words = (old_words & new_words) - STOP_WORDS
                if len(shared_content_words) >= 2:
                    return True, f"Explicit revision phrase '{trigger}' identified with shared entity context"

        return False, None
