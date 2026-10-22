"""Persistence backends for MemWeaver-AI (In-Memory & SQLite WAL)."""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional
from memweaver.models import MemoryItem, MemoryStatus, MemoryTier, MemoryType


class BaseStorage(ABC):
    """Abstract interface for memory persistence."""

    @abstractmethod
    def save(self, item: MemoryItem) -> None:
        pass

    @abstractmethod
    def get(self, item_id: str) -> Optional[MemoryItem]:
        pass

    @abstractmethod
    def get_all(self, tier: Optional[MemoryTier] = None) -> List[MemoryItem]:
        pass

    @abstractmethod
    def update(self, item: MemoryItem) -> None:
        pass

    @abstractmethod
    def delete(self, item_id: str) -> bool:
        pass

    @abstractmethod
    def clear(self) -> int:
        pass

    def close(self) -> None:
        """Closes any open resources or database handles."""
        pass


class InMemoryStorage(BaseStorage):
    """Thread-safe in-memory memory repository."""

    def __init__(self):
        self._lock = threading.RLock()
        self._items: Dict[str, MemoryItem] = {}

    def save(self, item: MemoryItem) -> None:
        with self._lock:
            self._items[item.id] = item

    def get(self, item_id: str) -> Optional[MemoryItem]:
        with self._lock:
            return self._items.get(item_id)

    def get_all(self, tier: Optional[MemoryTier] = None) -> List[MemoryItem]:
        with self._lock:
            if tier is None:
                return list(self._items.values())
            return [itm for itm in self._items.values() if itm.tier == tier]

    def update(self, item: MemoryItem) -> None:
        with self._lock:
            self._items[item.id] = item

    def delete(self, item_id: str) -> bool:
        with self._lock:
            if item_id in self._items:
                del self._items[item_id]
                return True
            return False

    def clear(self) -> int:
        with self._lock:
            cnt = len(self._items)
            self._items.clear()
            return cnt


class SQLiteStorage(BaseStorage):
    """Production file-backed SQLite database with Write-Ahead Logging (WAL)."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._init_db()

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Provides a managed SQLite connection that is explicitly closed."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        tier TEXT NOT NULL,
                        memory_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        importance REAL NOT NULL,
                        created_at REAL NOT NULL,
                        last_accessed_at REAL NOT NULL,
                        access_count INTEGER NOT NULL,
                        vector TEXT,
                        session_id TEXT NOT NULL,
                        agent_id TEXT NOT NULL,
                        tags TEXT,
                        metadata TEXT,
                        superseded_by TEXT,
                        supersedes_id TEXT,
                        confidence REAL NOT NULL
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_tier ON memories(tier);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON memories(status);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON memories(session_id);")
                conn.commit()

    def _row_to_item(self, row: sqlite3.Row) -> MemoryItem:
        vector = json.loads(row["vector"]) if row["vector"] else None
        tags = json.loads(row["tags"]) if row["tags"] else []
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}

        return MemoryItem(
            id=row["id"],
            content=row["content"],
            tier=MemoryTier(row["tier"]),
            memory_type=MemoryType(row["memory_type"]),
            status=MemoryStatus(row["status"]),
            importance=row["importance"],
            created_at=row["created_at"],
            last_accessed_at=row["last_accessed_at"],
            access_count=row["access_count"],
            vector=vector,
            session_id=row["session_id"],
            agent_id=row["agent_id"],
            tags=tags,
            metadata=metadata,
            superseded_by=row["superseded_by"],
            supersedes_id=row["supersedes_id"],
            confidence=row["confidence"],
        )

    def save(self, item: MemoryItem) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO memories (
                        id, content, tier, memory_type, status, importance,
                        created_at, last_accessed_at, access_count, vector,
                        session_id, agent_id, tags, metadata, superseded_by,
                        supersedes_id, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    item.id,
                    item.content,
                    item.tier.value,
                    item.memory_type.value,
                    item.status.value,
                    item.importance,
                    item.created_at,
                    item.last_accessed_at,
                    item.access_count,
                    json.dumps(item.vector) if item.vector else None,
                    item.session_id,
                    item.agent_id,
                    json.dumps(item.tags),
                    json.dumps(item.metadata),
                    item.superseded_by,
                    item.supersedes_id,
                    item.confidence,
                ))
                conn.commit()

    def get(self, item_id: str) -> Optional[MemoryItem]:
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute("SELECT * FROM memories WHERE id = ?", (item_id,))
                row = cursor.fetchone()
                if row:
                    return self._row_to_item(row)
                return None

    def get_all(self, tier: Optional[MemoryTier] = None) -> List[MemoryItem]:
        with self._lock:
            with self._connect() as conn:
                if tier is None:
                    cursor = conn.execute("SELECT * FROM memories ORDER BY created_at ASC")
                else:
                    cursor = conn.execute("SELECT * FROM memories WHERE tier = ? ORDER BY created_at ASC", (tier.value,))
                return [self._row_to_item(r) for r in cursor.fetchall()]

    def update(self, item: MemoryItem) -> None:
        self.save(item)

    def delete(self, item_id: str) -> bool:
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute("DELETE FROM memories WHERE id = ?", (item_id,))
                conn.commit()
                return cursor.rowcount > 0

    def clear(self) -> int:
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute("DELETE FROM memories")
                conn.commit()
                return cursor.rowcount
