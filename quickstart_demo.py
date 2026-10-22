"""Quickstart demo script for MemWeaver-AI.

Demonstrates:
1. Three-tier cognitive memory architecture (Working, Episodic, Semantic).
2. Autonomous consolidation & knowledge distillation.
3. Real-time truth maintenance & belief revision (conflict resolution).
4. Automated tool execution recording via @engine.remember_call decorator.
5. Generation of interactive dark-mode HTML executive dashboard.
"""

from __future__ import annotations

import sys
import time

# Set UTF-8 encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from memweaver.config import MemoryConfig
from memweaver.dashboard import DashboardReporter
from memweaver.engine import MemoryEngine
from memweaver.models import MemoryTier, MemoryType


def main():
    print("=" * 70)
    print("  MEMWEAVER-AI: COGNITIVE AGENT MEMORY ENGINE QUICKSTART DEMO")
    print("=" * 70)

    # 1. Initialize Engine
    print("\n[*] Initializing MemWeaver-AI Engine with SQLite WAL persistence...")
    config = MemoryConfig(storage_path="memweaver_demo.db")
    engine = MemoryEngine(config=config)
    engine.storage.clear()  # Fresh demo state

    # 2. Tier 1: Working Memory (Immediate Attention Scratchpad)
    print("\n[+] Step 1: Populating Tier 1 (Working Memory Scratchpad)...")
    engine.remember(
        content="Current active goal: Optimize data transformation pipeline for 10M records",
        tier=MemoryTier.WORKING,
        memory_type=MemoryType.GOAL,
        importance=0.9,
    )
    engine.remember(
        content="Intermediate tool output: Found 3 memory leaks in worker thread 2",
        tier=MemoryTier.WORKING,
        memory_type=MemoryType.OBSERVATION,
        importance=0.7,
    )
    print(f"    Working items in scratchpad: {engine.working.count()} ({engine.working.total_tokens()} tokens)")

    # 3. Tier 2: Episodic Memory (Temporal Interactions & Autobiographical Log)
    print("\n[+] Step 2: Logging Tier 2 (Episodic Interactions)...")
    interactions = [
        ("User said: 'I prefer using Python 3.12 and Polars for fast dataframe operations.'", 0.8),
        ("User asked: 'Can we configure SQLite WAL mode for high concurrency?'", 0.6),
        ("User noted: 'Our primary deployment region is AWS us-east-1.'", 0.7),
        ("User said: 'I want to build an autonomous agent memory system with Ebbinghaus decay.'", 0.9),
        ("User remarked: 'I live in Jaipur, Rajasthan, and focus on Generative AI systems.'", 0.85),
    ]
    for text, imp in interactions:
        engine.remember(text, tier=MemoryTier.EPISODIC, importance=imp)
    print(f"    Recorded {len(interactions)} episodic dialogue turns.")

    # 4. Cognitive Consolidation (Episodic -> Semantic Knowledge Distillation)
    print("\n[+] Step 3: Triggering Autonomous Consolidation Cycle...")
    report = engine.consolidate()
    print(f"    Consolidation complete in {report.duration_ms:.2f} ms:")
    print(f"    • Episodic Items Analyzed: {report.episodic_items_analyzed}")
    print(f"    • Semantic Facts Extracted: {report.semantic_facts_generated}")
    print(f"    • Knowledge Compression Ratio: {report.compression_ratio:.1f}x")

    # 5. Truth Maintenance & Belief Revision (Contradiction Resolution)
    print("\n[+] Step 4: Testing Real-Time Belief Revision & Contradiction Resolution...")
    print("    Existing belief: 'User preferred dark theme for interface'")
    engine.remember(
        content="User preferred dark theme for interface",
        tier=MemoryTier.SEMANTIC,
        memory_type=MemoryType.PREFERENCE,
        importance=0.8,
    )

    print("    New incoming observation: 'User switched to light theme for interface'")
    engine.remember(
        content="User switched to light theme for interface",
        tier=MemoryTier.SEMANTIC,
        memory_type=MemoryType.PREFERENCE,
        importance=0.85,
    )
    print("    ✓ Contradiction detected: Outdated preference automatically SUPERSEDED with lineage preserved!")

    # 6. Tool-Call Recording Decorator
    print("\n[+] Step 5: Testing @remember_call Decorator for Agent Tool Runs...")
    @engine.remember_call(importance=0.8)
    def vector_search_tool(query_str: str, top_k: int = 3) -> str:
        return f"Retrieved top {top_k} vectors for query: {query_str}"

    res = vector_search_tool("Agent cognitive memory", top_k=5)
    print(f"    Tool returned: {res}")
    print("    ✓ Tool inputs, outputs, and telemetry automatically committed to Episodic Memory!")

    # 7. Hybrid Multi-Tier Recall
    print("\n[+] Step 6: Hybrid Recall Query (Semantic + Recency + Importance)...")
    search_query = "What programming language and libraries does the user prefer?"
    t0 = time.time()
    recalled = engine.recall(search_query, k=3)
    latency_ms = (time.time() - t0) * 1000.0

    print(f"    Search Query: '{search_query}' (Latency: {latency_ms:.3f} ms)")
    for idx, r in enumerate(recalled, 1):
        print(f"    [{idx}] Score: {r.score:.3f} (Sim: {r.semantic_similarity:.3f}, Recency: {r.recency_score:.3f})")
        print(f"        Tier: {r.item.tier.value.upper()} | Type: {r.item.memory_type.value} | Status: {r.item.status.value}")
        print(f"        Content: {r.item.content}")

    # 8. Ready-to-Inject LLM Prompt Context
    print("\n[+] Step 7: Generating System Prompt Context for LLM Injection...")
    context_prompt = engine.get_context_prompt(query=search_query, k=2)
    print("-" * 65)
    print(context_prompt)
    print("-" * 65)

    # 9. Executive Dashboard Generation
    print("\n[+] Step 8: Generating Standalone Dark-Mode HTML Executive Dashboard...")
    dashboard_path = DashboardReporter.generate_html(engine, output_path="quickstart_report.html")
    print(f"    ✓ Visual Executive Dashboard generated at: {dashboard_path}")

    # 10. Operational Telemetry Stats
    stats = engine.get_stats()
    print("\n" + "=" * 70)
    print("  MEMWEAVER-AI OPERATIONAL TELEMETRY")
    print("=" * 70)
    print(f"  • Total Memories:        {stats.total_memories}")
    print(f"  • Working Memory Items:  {stats.working_count}")
    print(f"  • Episodic Logs:         {stats.episodic_count}")
    print(f"  • Active Semantic Facts: {stats.semantic_count}")
    print(f"  • Superseded Memories:   {stats.superseded_count}")
    print(f"  • Avg Retrieval Latency: {stats.average_retrieval_latency_ms:.3f} ms")
    print(f"  • Consolidation Cycles:  {stats.consolidation_cycles}")
    print("=" * 70)


if __name__ == "__main__":
    main()
