"""Tests for MemWeaver-AI Click CLI."""

import os
import tempfile
import pytest
from click.testing import CliRunner
from memweaver.cli import main


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "memweaver" in result.output


def test_cli_remember_and_recall():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        db_path = os.path.join(tmp_dir, "cli_test.db")
        runner = CliRunner()

        # Remember
        res_rem = runner.invoke(main, [
            "--db", db_path,
            "remember",
            "--content", "User prefers Python 3.12 for microservices",
            "--tier", "semantic",
            "--type", "preference",
            "--importance", "0.85"
        ])
        assert res_rem.exit_code == 0
        assert "Stored #" in res_rem.output

        # Recall
        res_rec = runner.invoke(main, [
            "--db", db_path,
            "recall",
            "--query", "Python microservices",
            "--k", "2"
        ])
        assert res_rec.exit_code == 0
        assert "Recalled" in res_rec.output
        assert "Python" in res_rec.output


def test_cli_consolidate_and_stats():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        db_path = os.path.join(tmp_dir, "cli_test.db")
        runner = CliRunner()

        # Store episodic items
        runner.invoke(main, [
            "--db", db_path,
            "remember",
            "--content", "In this run, user prefers FastAPI over Flask",
            "--tier", "episodic"
        ])

        # Consolidate
        res_cons = runner.invoke(main, ["--db", db_path, "consolidate"])
        assert res_cons.exit_code == 0
        assert "Consolidation Complete" in res_cons.output

        # Stats
        res_stats = runner.invoke(main, ["--db", db_path, "stats"])
        assert res_stats.exit_code == 0
        assert "MemWeaver-AI Operational Stats" in res_stats.output


def test_cli_dashboard_and_benchmark():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        db_path = os.path.join(tmp_dir, "cli_test.db")
        html_path = os.path.join(tmp_dir, "report.html")
        runner = CliRunner()

        res_dash = runner.invoke(main, ["--db", db_path, "dashboard", "--output", html_path])
        assert res_dash.exit_code == 0
        assert os.path.exists(html_path)

        res_bench = runner.invoke(main, ["--db", db_path, "benchmark", "--n-queries", "20"])
        assert res_bench.exit_code == 0
        assert "Benchmark Results:" in res_bench.output
        assert "Throughput:" in res_bench.output
