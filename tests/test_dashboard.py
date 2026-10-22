"""Tests for Executive Dashboard generation."""

import os
import tempfile
import pytest
from memweaver.dashboard import DashboardReporter
from memweaver.engine import MemoryEngine
from memweaver.models import MemoryTier, MemoryType


def test_dashboard_generation():
    engine = MemoryEngine()
    engine.remember("Working prompt memory", tier=MemoryTier.WORKING)
    engine.remember("User prefers dark theme", tier=MemoryTier.EPISODIC)
    engine.remember("User switched to light theme", tier=MemoryTier.SEMANTIC, memory_type=MemoryType.PREFERENCE)
    engine.recall("theme preference", k=2)

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_html = os.path.join(tmp_dir, "test_dashboard.html")
        path = DashboardReporter.generate_html(engine, output_path=out_html)

        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "<!DOCTYPE html>" in content
        assert "MemWeaver-AI" in content
        assert "Ebbinghaus Cognitive Decay Curves" in content
        assert "<svg" in content
        assert "Nitesh Beniwal" in content
