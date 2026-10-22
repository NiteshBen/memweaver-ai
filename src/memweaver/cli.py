"""Command Line Interface for MemWeaver-AI."""

from __future__ import annotations

import sys
import time
import click
from memweaver.config import MemoryConfig
from memweaver.dashboard import DashboardReporter
from memweaver.engine import MemoryEngine
from memweaver.models import MemoryTier, MemoryType


@click.group()
@click.version_option("0.1.0", prog_name="memweaver")
@click.option("--db", default=None, help="Path to SQLite persistence database (default: in-memory)")
@click.pass_context
def main(ctx, db):
    """MemWeaver-AI: Production Hierarchical Long-Term Memory & State Engine for AI Agents."""
    ctx.ensure_object(dict)
    config = MemoryConfig(storage_path=db)
    ctx.obj["engine"] = MemoryEngine(config=config)


@main.command()
@click.option("--content", "-c", required=True, help="Memory text to store")
@click.option("--tier", "-t", type=click.Choice(["working", "episodic", "semantic"]), default="episodic")
@click.option("--type", "mtype", type=click.Choice(["observation", "fact", "preference", "goal", "reflection", "dialogue"]), default="observation")
@click.option("--importance", "-i", type=float, default=0.7, help="Importance score [0.0 - 1.0]")
@click.option("--session", "-s", default="default", help="Session ID")
@click.pass_context
def remember(ctx, content, tier, mtype, importance, session):
    """Stores a new memory item across working, episodic, or semantic tiers."""
    engine: MemoryEngine = ctx.obj["engine"]
    item = engine.remember(
        content=content,
        tier=MemoryTier(tier),
        memory_type=MemoryType(mtype),
        importance=importance,
        session_id=session,
    )
    click.echo(f"Stored #{item.id[:8]} [{item.tier.value.upper()} | {item.memory_type.value}] (importance: {item.importance:.2f})")
    click.echo(f"  Content: {item.content}")


@main.command()
@click.option("--query", "-q", required=True, help="Semantic search query")
@click.option("--k", default=5, help="Number of items to retrieve")
@click.pass_context
def recall(ctx, query, k):
    """Performs hybrid memory retrieval (semantic similarity + recency + importance)."""
    engine: MemoryEngine = ctx.obj["engine"]
    t0 = time.time()
    results = engine.recall(query=query, k=k)
    duration_ms = (time.time() - t0) * 1000.0

    click.echo(f"\nRecalled {len(results)} memories for: '{query}' ({duration_ms:.2f} ms)\n" + "-" * 60)
    for idx, r in enumerate(results, 1):
        click.echo(f"[{idx}] Score: {r.score:.3f} | Sim: {r.semantic_similarity:.3f} | Recency: {r.recency_score:.3f}")
        click.echo(f"    Tier: {r.item.tier.value.upper()} | Type: {r.item.memory_type.value} | ID: #{r.item.id[:8]}")
        click.echo(f"    Content: {r.item.content}\n")


@main.command()
@click.pass_context
def consolidate(ctx):
    """Executes cognitive consolidation from episodic interactions into semantic facts."""
    engine: MemoryEngine = ctx.obj["engine"]
    report = engine.consolidate()
    click.echo(f"Consolidation Complete in {report.duration_ms:.2f} ms:")
    click.echo(f"  • Episodic Items Analyzed: {report.episodic_items_analyzed}")
    click.echo(f"  • Semantic Facts Created:  {report.semantic_facts_generated}")
    click.echo(f"  • Contradictions Resolved: {report.contradictions_resolved}")
    click.echo(f"  • Knowledge Compression:   {report.compression_ratio:.1f}x")


@main.command()
@click.pass_context
def stats(ctx):
    """Displays telemetry metrics, memory distribution, and average retrieval latency."""
    engine: MemoryEngine = ctx.obj["engine"]
    s = engine.get_stats()
    click.echo("\nMemWeaver-AI Operational Stats:")
    click.echo("-" * 40)
    click.echo(f"Total Memories:          {s.total_memories}")
    click.echo(f"  • Working Memory:      {s.working_count}")
    click.echo(f"  • Episodic Memory:     {s.episodic_count}")
    click.echo(f"  • Semantic Facts:      {s.semantic_count}")
    click.echo(f"  • Superseded (Outdated): {s.superseded_count}")
    click.echo(f"Total Retrievals:        {s.retrieval_count}")
    click.echo(f"Avg Retrieval Latency:   {s.average_retrieval_latency_ms:.3f} ms")
    click.echo(f"Consolidation Cycles:    {s.consolidation_cycles}\n")


@main.command()
@click.option("--output", "-o", default="memweaver_dashboard.html", help="Output path for HTML report")
@click.pass_context
def dashboard(ctx, output):
    """Generates a standalone, dark-mode visual executive dashboard."""
    engine: MemoryEngine = ctx.obj["engine"]
    path = DashboardReporter.generate_html(engine, output_path=output)
    click.echo(f"Dashboard generated at: {path}")


@main.command()
@click.option("--n-queries", "-n", default=300, help="Number of benchmark queries to run")
@click.pass_context
def benchmark(ctx, n_queries):
    """Profiles retrieval latency and throughput."""
    engine: MemoryEngine = ctx.obj["engine"]
    click.echo(f"Populating test dataset and running {n_queries} retrieval queries...")

    # Seed memories
    for i in range(50):
        engine.remember(f"Agent observation event #{i}: User performed data transformation on column {i}", importance=0.6)
    for i in range(20):
        engine.remember(f"Permanent fact #{i}: System parameter timeout is set to {i * 10} seconds", tier=MemoryTier.SEMANTIC, importance=0.9)

    t0 = time.time()
    for i in range(n_queries):
        engine.recall(f"timeout parameter {i % 10}", k=3)
    elapsed = time.time() - t0
    qps = n_queries / elapsed
    avg_latency = (elapsed / n_queries) * 1000.0

    click.echo("\nBenchmark Results:")
    click.echo("-" * 40)
    click.echo(f"Throughput:       {qps:.1f} Queries / Sec")
    click.echo(f"Average Latency:  {avg_latency:.3f} ms")
    click.echo(f"Total Time:       {elapsed:.3f} s")


if __name__ == "__main__":
    main()
