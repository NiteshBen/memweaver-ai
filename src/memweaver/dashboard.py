"""Executive Dark-Mode Dashboard & SVG Cognitive Visualizer for MemWeaver-AI."""

from __future__ import annotations

import html
import os
import time
from typing import TYPE_CHECKING, List
from memweaver.models import MemoryStatus, MemoryTier

if TYPE_CHECKING:
    from memweaver.engine import MemoryEngine


class DashboardReporter:
    """Generates a standalone, zero-CDN, responsive dark-mode HTML executive dashboard."""

    @staticmethod
    def generate_html(engine: "MemoryEngine", output_path: str = "memweaver_dashboard.html") -> str:
        stats = engine.get_stats()
        working_items = engine.working.get_all()
        episodic_items = engine.episodic.get_all()
        semantic_items = engine.semantic.get_all()
        all_items = engine.storage.get_all() if engine.config.storage_path else (working_items + episodic_items + semantic_items)

        # SVG 1: Cognitive Decay Simulation Curve
        svg_decay = DashboardReporter._render_decay_svg(engine)

        # SVG 2: Tier Breakdown Gauge
        svg_tier = DashboardReporter._render_tier_gauge_svg(stats)

        # Memory Rows Table HTML
        rows_html = []
        for itm in reversed(all_items[-15:]):  # Most recent 15
            status_badge = (
                '<span class="badge badge-active">ACTIVE</span>'
                if itm.status == MemoryStatus.ACTIVE
                else '<span class="badge badge-superseded">SUPERSEDED</span>'
            )
            tier_badge = f'<span class="badge badge-tier">{itm.tier.value.upper()}</span>'
            type_badge = f'<span class="badge badge-type">{itm.memory_type.value}</span>'
            
            lineage = ""
            if itm.superseded_by:
                lineage = f'<div class="superseded-note">↳ Superseded by #{itm.superseded_by[:8]}</div>'

            content_display = html.escape(itm.content)
            if itm.status == MemoryStatus.SUPERSEDED:
                content_display = f"<del>{content_display}</del>"

            rows_html.append(f"""
                <tr>
                    <td style="font-family: monospace; color: #38bdf8;">#{itm.id[:8]}</td>
                    <td>{tier_badge}</td>
                    <td>{type_badge}</td>
                    <td>
                        {content_display}
                        {lineage}
                    </td>
                    <td><div class="progress-bar"><div class="progress-fill" style="width: {int(itm.importance * 100)}%;"></div></div> {itm.importance:.2f}</td>
                    <td>{status_badge}</td>
                </tr>
            """)

        table_body = "\n".join(rows_html) if rows_html else "<tr><td colspan='6'>No memories recorded yet.</td></tr>"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MemWeaver-AI • Cognitive Memory Dashboard</title>
    <style>
        :root {{
            --bg-primary: #090d16;
            --bg-card: #111827;
            --bg-subtle: #1e293b;
            --border: #334155;
            --accent-cyan: #38bdf8;
            --accent-purple: #a855f7;
            --accent-emerald: #10b981;
            --accent-amber: #f59e0b;
            --accent-rose: #f43f5e;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg-primary);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            padding: 30px 20px;
            line-height: 1.5;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .brand {{ display: flex; align-items: center; gap: 15px; }}
        .brand-icon {{
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            width: 44px; height: 44px; border-radius: 10px;
            display: flex; align-items: center; justify-content: center;
            font-size: 22px; font-weight: bold; color: #fff;
        }}
        .brand h1 {{ font-size: 24px; font-weight: 700; letter-spacing: -0.5px; }}
        .brand p {{ color: var(--text-muted); font-size: 13px; }}
        .header-meta {{ text-align: right; font-size: 13px; color: var(--text-muted); }}

        /* KPI Grid */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 16px;
            margin-bottom: 30px;
        }}
        .kpi-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 18px;
            position: relative;
            overflow: hidden;
        }}
        .kpi-card::before {{
            content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
            background: var(--accent-cyan);
        }}
        .kpi-card:nth-child(2)::before {{ background: var(--accent-purple); }}
        .kpi-card:nth-child(3)::before {{ background: var(--accent-emerald); }}
        .kpi-card:nth-child(4)::before {{ background: var(--accent-amber); }}
        .kpi-card:nth-child(5)::before {{ background: var(--accent-rose); }}

        .kpi-label {{ font-size: 12px; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.5px; }}
        .kpi-value {{ font-size: 28px; font-weight: 700; color: var(--text-main); margin-top: 6px; }}
        .kpi-subtext {{ font-size: 12px; color: var(--text-muted); margin-top: 4px; }}

        /* Visualizations Section */
        .charts-grid {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }}
        @media (max-width: 900px) {{ .charts-grid {{ grid-template-columns: 1fr; }} }}
        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 22px;
        }}
        .card-title {{
            font-size: 16px; font-weight: 600; margin-bottom: 15px;
            display: flex; justify-content: space-between; align-items: center;
        }}

        /* Table */
        table {{
            width: 100%; border-collapse: collapse; text-align: left; font-size: 13px;
        }}
        th {{
            background: var(--bg-subtle); color: var(--text-muted); padding: 12px 14px;
            text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px;
        }}
        td {{ padding: 12px 14px; border-bottom: 1px solid var(--border); vertical-align: middle; }}
        tr:hover td {{ background: rgba(255, 255, 255, 0.02); }}

        .badge {{
            padding: 3px 8px; border-radius: 6px; font-size: 11px; font-weight: 600; display: inline-block;
        }}
        .badge-active {{ background: rgba(16, 185, 129, 0.2); color: var(--accent-emerald); border: 1px solid rgba(16, 185, 129, 0.4); }}
        .badge-superseded {{ background: rgba(244, 63, 94, 0.2); color: var(--accent-rose); border: 1px solid rgba(244, 63, 94, 0.4); }}
        .badge-tier {{ background: rgba(56, 189, 248, 0.2); color: var(--accent-cyan); }}
        .badge-type {{ background: rgba(168, 85, 247, 0.2); color: var(--accent-purple); }}

        .superseded-note {{ font-size: 11px; color: var(--accent-rose); margin-top: 4px; }}
        del {{ color: var(--text-muted); text-decoration: line-through; }}

        .progress-bar {{
            background: var(--bg-subtle); height: 6px; width: 70px; border-radius: 3px; display: inline-block; margin-right: 6px; vertical-align: middle;
        }}
        .progress-fill {{ height: 100%; background: var(--accent-cyan); border-radius: 3px; }}

        footer {{
            margin-top: 40px; text-align: center; font-size: 12px; color: var(--text-muted);
            border-top: 1px solid var(--border); padding-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="brand">
                <div class="brand-icon">🧠</div>
                <div>
                    <h1>MemWeaver-AI</h1>
                    <p>Hierarchical Long-Term Memory & Cognitive Belief Revision Engine</p>
                </div>
            </div>
            <div class="header-meta">
                <div>Engine: <strong>100% Offline Local</strong></div>
                <div>Generated: <strong>{time.strftime('%Y-%m-%d %H:%M:%S')}</strong></div>
            </div>
        </header>

        <!-- KPI Grid -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Total Memories</div>
                <div class="kpi-value">{stats.total_memories}</div>
                <div class="kpi-subtext">Across all tiers</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Semantic Facts</div>
                <div class="kpi-value">{stats.semantic_count}</div>
                <div class="kpi-subtext">Distilled long-term truths</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Episodic Logs</div>
                <div class="kpi-value">{stats.episodic_count}</div>
                <div class="kpi-subtext">Temporal interactions</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Avg Retrieval Latency</div>
                <div class="kpi-value">{stats.average_retrieval_latency_ms:.2f} ms</div>
                <div class="kpi-subtext">Sub-millisecond speed</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Beliefs Superseded</div>
                <div class="kpi-value">{stats.superseded_count}</div>
                <div class="kpi-subtext">Contradictions resolved</div>
            </div>
        </div>

        <!-- Charts Grid -->
        <div class="charts-grid">
            <div class="card">
                <div class="card-title">
                    <span>Ebbinghaus Cognitive Decay Curves (72-Hour Horizon)</span>
                    <span style="font-size: 12px; color: var(--accent-cyan);">R(t) = exp(-Δt / S)</span>
                </div>
                {svg_decay}
            </div>
            <div class="card">
                <div class="card-title">
                    <span>Memory Tier Balance</span>
                    <span style="font-size: 12px; color: var(--text-muted);">{stats.total_memories} Total</span>
                </div>
                {svg_tier}
            </div>
        </div>

        <!-- Memory Lineage Table -->
        <div class="card">
            <div class="card-title">
                <span>Active & Superseded Memory Registry</span>
                <span style="font-size: 12px; color: var(--text-muted);">Real-Time Audit Trail</span>
            </div>
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Memory ID</th>
                            <th>Tier</th>
                            <th>Type</th>
                            <th>Content & Lineage</th>
                            <th>Importance</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_body}
                    </tbody>
                </table>
            </div>
        </div>

        <footer>
            <p>Developed with ❤️ by <strong>Nitesh Beniwal</strong> • <a href="https://github.com/NiteshBen/memweaver-ai" style="color: var(--accent-cyan); text-decoration: none;">github.com/NiteshBen/memweaver-ai</a></p>
            <p style="margin-top: 4px;">Production Cognitive Architecture for Autonomous Agents • MIT Licensed</p>
        </footer>
    </div>
</body>
</html>"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return os.path.abspath(output_path)

    @staticmethod
    def _render_decay_svg(engine: "MemoryEngine") -> str:
        """Renders mathematical retention decay curves as an inline SVG."""
        width, height = 700, 240
        padding = 40

        # Sample 3 synthetic trajectories:
        # 1. High importance + accessed (top curve)
        # 2. Standard importance (middle curve)
        # 3. Low importance unaccessed (fast decay curve)
        hours = [0, 6, 12, 18, 24, 36, 48, 60, 72]
        
        def calc_pts(s_hours: float):
            pts = []
            for h in hours:
                val = max(0.02, min(1.0, 2.71828 ** (- (h / s_hours))))
                x = padding + (h / 72.0) * (width - 2 * padding)
                y = (height - padding) - (val * (height - 2 * padding))
                pts.append(f"{x:.1f},{y:.1f}")
            return " ".join(pts)

        poly_high = calc_pts(110.0)
        poly_mid = calc_pts(42.0)
        poly_low = calc_pts(14.0)

        return f"""
        <svg viewBox="0 0 {width} {height}" width="100%" height="240" xmlns="http://www.w3.org/2000/svg">
            <!-- Grid Lines -->
            <line x1="{padding}" y1="{padding}" x2="{width - padding}" y2="{padding}" stroke="#334155" stroke-dasharray="3 3"/>
            <line x1="{padding}" y1="{height/2}" x2="{width - padding}" y2="{height/2}" stroke="#334155" stroke-dasharray="3 3"/>
            <line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#475569"/>

            <!-- Axes Labels -->
            <text x="{padding}" y="{padding - 10}" fill="#94a3b8" font-size="11">100% Retention</text>
            <text x="{padding}" y="{height/2 - 5}" fill="#94a3b8" font-size="11">50% Half-Life</text>
            <text x="{padding}" y="{height - 15}" fill="#94a3b8" font-size="11">0h</text>
            <text x="{width/2}" y="{height - 15}" fill="#94a3b8" font-size="11">36h</text>
            <text x="{width - padding - 20}" y="{height - 15}" fill="#94a3b8" font-size="11">72h</text>

            <!-- Decay Paths -->
            <polyline points="{poly_high}" fill="none" stroke="#10b981" stroke-width="3" stroke-linecap="round"/>
            <polyline points="{poly_mid}" fill="none" stroke="#38bdf8" stroke-width="2.5" stroke-linecap="round"/>
            <polyline points="{poly_low}" fill="none" stroke="#f43f5e" stroke-width="2" stroke-linecap="round"/>

            <!-- Legend -->
            <circle cx="480" cy="30" r="5" fill="#10b981"/>
            <text x="492" y="34" fill="#f8fafc" font-size="11">High Priority + Recalled</text>
            <circle cx="480" cy="50" r="5" fill="#38bdf8"/>
            <text x="492" y="54" fill="#f8fafc" font-size="11">Standard Interaction</text>
            <circle cx="480" cy="70" r="5" fill="#f43f5e"/>
            <text x="492" y="74" fill="#f8fafc" font-size="11">Low Importance Transient</text>
        </svg>
        """

    @staticmethod
    def _render_tier_gauge_svg(stats) -> str:
        """Renders tiered memory breakdown."""
        width, height = 300, 240
        w = max(1, stats.working_count)
        e = max(1, stats.episodic_count)
        s = max(1, stats.semantic_count)
        total = w + e + s

        w_pct = int((w / total) * 100)
        e_pct = int((e / total) * 100)
        s_pct = max(1, 100 - w_pct - e_pct)

        return f"""
        <svg viewBox="0 0 {width} {height}" width="100%" height="240" xmlns="http://www.w3.org/2000/svg">
            <g transform="translate(150, 110)">
                <!-- Donut Chart segments -->
                <circle r="70" cx="0" cy="0" fill="none" stroke="#1e293b" stroke-width="28"/>
                <circle r="70" cx="0" cy="0" fill="none" stroke="#a855f7" stroke-width="28"
                    stroke-dasharray="{s_pct * 4.4} 440" stroke-dashoffset="0"/>
                <circle r="70" cx="0" cy="0" fill="none" stroke="#38bdf8" stroke-width="28"
                    stroke-dasharray="{e_pct * 4.4} 440" stroke-dashoffset="-{s_pct * 4.4}"/>
                <circle r="70" cx="0" cy="0" fill="none" stroke="#10b981" stroke-width="28"
                    stroke-dasharray="{w_pct * 4.4} 440" stroke-dashoffset="-{(s_pct + e_pct) * 4.4}"/>
                <text x="0" y="5" text-anchor="middle" fill="#f8fafc" font-size="20" font-weight="bold">{stats.total_memories}</text>
                <text x="0" y="24" text-anchor="middle" fill="#94a3b8" font-size="10">TOTAL ITEMS</text>
            </g>
            <!-- Legend -->
            <g transform="translate(25, 205)" font-size="11" fill="#f8fafc">
                <circle cx="0" cy="0" r="5" fill="#10b981"/>
                <text x="10" y="4">Working: {stats.working_count}</text>
                <circle cx="95" cy="0" r="5" fill="#38bdf8"/>
                <text x="105" y="4">Episodic: {stats.episodic_count}</text>
                <circle cx="190" cy="0" r="5" fill="#a855f7"/>
                <text x="200" y="4">Semantic: {stats.semantic_count}</text>
            </g>
        </svg>
        """
