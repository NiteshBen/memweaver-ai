"""Tests for In-Memory and SQLite WAL Persistence."""

import os
import tempfile
import pytest
from memweaver.models import MemoryItem, MemoryStatus, MemoryTier, MemoryType
from memweaver.storage import InMemoryStorage, SQLiteStorage


def test_in_memory_storage_crud():
    storage = InMemoryStorage()
    item = MemoryItem(content="In-memory test fact", tier=MemoryTier.SEMANTIC)

    # Save
    storage.save(item)
    assert storage.get(item.id) is not None

    # Get all by tier
    assert len(storage.get_all(MemoryTier.SEMANTIC)) == 1
    assert len(storage.get_all(MemoryTier.WORKING)) == 0

    # Update
    item.importance = 0.95
    storage.update(item)
    assert storage.get(item.id).importance == 0.95

    # Delete
    assert storage.delete(item.id) is True
    assert storage.get(item.id) is None


def test_sqlite_storage_wal_persistence():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_memweaver.db")
        storage = SQLiteStorage(db_path)

        item = MemoryItem(
            content="Persistent SQLite item",
            tier=MemoryTier.EPISODIC,
            memory_type=MemoryType.OBSERVATION,
            importance=0.85,
            vector=[0.1, 0.2, 0.3],
            tags=["sqlite", "test"],
            metadata={"source": "pytest"},
        )

        storage.save(item)

        # Reopen new connection from disk
        storage2 = SQLiteStorage(db_path)
        retrieved = storage2.get(item.id)

        assert retrieved is not None
        assert retrieved.content == "Persistent SQLite item"
        assert retrieved.tier == MemoryTier.EPISODIC
        assert retrieved.importance == 0.85
        assert retrieved.vector == [0.1, 0.2, 0.3]
        assert retrieved.tags == ["sqlite", "test"]
        assert retrieved.metadata["source"] == "pytest"

        # Clear
        cleared_count = storage2.clear()
        assert cleared_count == 1
        assert len(storage2.get_all()) == 0
