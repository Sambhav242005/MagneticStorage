"""
Performance tracker for NeuroSavant.

Tracks query latency, ingestion throughput, memory usage, and embedding time.
All terminal output is ASCII-only so it renders cleanly on Windows consoles.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class PerformanceMetrics:
    """Store performance metrics with rolling averages."""

    query_times: deque = field(default_factory=lambda: deque(maxlen=100))
    ingest_times: deque = field(default_factory=lambda: deque(maxlen=100))
    embed_times: deque = field(default_factory=lambda: deque(maxlen=100))
    batch_sizes: deque = field(default_factory=lambda: deque(maxlen=100))

    total_queries: int = 0
    total_ingests: int = 0
    total_cells: int = 0

    tool_usage: Dict[str, int] = field(default_factory=dict)
    tool_errors: int = 0
    total_tool_calls: int = 0


class PerformanceTracker:
    """Tracks performance metrics for NeuroSavant operations."""

    def __init__(self):
        self.metrics = PerformanceMetrics()
        self._current_timer: Optional[float] = None

    def start_timer(self) -> float:
        """Start a timing operation."""
        return time.perf_counter()

    def record_query(self, start_time: float):
        """Record a query operation time."""
        elapsed = (time.perf_counter() - start_time) * 1000
        self.metrics.query_times.append(elapsed)
        self.metrics.total_queries += 1
        return elapsed

    def record_ingest(self, start_time: float, count: int = 1):
        """Record an ingest operation time."""
        elapsed = (time.perf_counter() - start_time) * 1000
        self.metrics.ingest_times.append(elapsed)
        self.metrics.batch_sizes.append(count)
        self.metrics.total_ingests += count
        return elapsed

    def record_embed(self, start_time: float):
        """Record an embedding operation time."""
        elapsed = (time.perf_counter() - start_time) * 1000
        self.metrics.embed_times.append(elapsed)
        return elapsed

    def record_tool_usage(self, tool_name: str, success: bool = True):
        """Record a tool execution."""
        self.metrics.total_tool_calls += 1
        self.metrics.tool_usage[tool_name] = self.metrics.tool_usage.get(tool_name, 0) + 1
        if not success:
            self.metrics.tool_errors += 1

    def get_stats(self) -> Dict:
        """Get current statistics."""

        def avg(values):
            return sum(values) / len(values) if values else 0

        def percentile(values, p):
            if not values:
                return 0
            sorted_vals = sorted(values)
            idx = int(len(sorted_vals) * p / 100)
            return sorted_vals[min(idx, len(sorted_vals) - 1)]

        ingest_seconds = sum(self.metrics.ingest_times) / 1000 if self.metrics.ingest_times else 0

        return {
            "query": {
                "avg_ms": avg(self.metrics.query_times),
                "p95_ms": percentile(self.metrics.query_times, 95),
                "total": self.metrics.total_queries,
            },
            "ingest": {
                "avg_ms": avg(self.metrics.ingest_times),
                "total": self.metrics.total_ingests,
                "throughput": (self.metrics.total_ingests / ingest_seconds) if ingest_seconds else 0,
            },
            "embed": {
                "avg_ms": avg(self.metrics.embed_times),
                "p95_ms": percentile(self.metrics.embed_times, 95),
            },
            "tools": {
                "total": self.metrics.total_tool_calls,
                "errors": self.metrics.tool_errors,
                "usage": self.metrics.tool_usage,
            },
        }

    def display_stats(self) -> str:
        """Generate formatted performance stats display."""
        stats = self.get_stats()

        output = [
            "",
            "Performance Metrics",
            "-------------------",
            "Queries",
            f"  Total:        {stats['query']['total']}",
            f"  Avg Latency:  {stats['query']['avg_ms']:.2f} ms",
            f"  P95 Latency:  {stats['query']['p95_ms']:.2f} ms",
            "",
            "Ingestion",
            f"  Total Cells:  {stats['ingest']['total']}",
            f"  Avg Time:     {stats['ingest']['avg_ms']:.2f} ms",
            f"  Throughput:   {stats['ingest']['throughput']:.1f} cells/sec",
            "",
            "Embedding",
            f"  Avg Time:     {stats['embed']['avg_ms']:.2f} ms",
            f"  P95 Time:     {stats['embed']['p95_ms']:.2f} ms",
            "",
            "Agentic Tools",
            f"  Total Calls:  {stats['tools']['total']}",
            f"  Errors:       {stats['tools']['errors']}",
        ]

        if stats["tools"]["usage"]:
            output.append("  Usage Breakdown:")
            for name, count in sorted(stats["tools"]["usage"].items()):
                output.append(f"    - {name}: {count}")

        return "\n".join(output)


class VisualDisplay:
    """ASCII-only terminal display helpers."""

    @staticmethod
    def bar_chart(data: Dict[str, int], max_width: int = 40, title: str = "Chart") -> str:
        """Generate an ASCII horizontal bar chart."""
        if not data:
            return "  (No data)"

        output = ["", title, "-" * len(title)]
        max_val = max(data.values()) if data else 1
        max_label = max(len(str(label)) for label in data) if data else 10

        for label, value in data.items():
            bar_width = int((value / max_val) * max_width) if max_val > 0 else 0
            bar = "#" * bar_width
            output.append(f"{label:<{max_label}} | {bar:<{max_width}} | {value:>6}")

        return "\n".join(output)

    @staticmethod
    def memory_stats_visual(cells: int, groups: int, entities: int) -> str:
        """Display memory statistics with a visual representation."""
        display_total = cells + groups + entities
        denominator_total = display_total or 1

        data = {
            "Cells": cells,
            "Groups": groups,
            "Entity Index": entities,
        }
        max_val = max(data.values()) if any(data.values()) else 1
        bar_width = 35

        output = ["", "Memory Visualization", "--------------------"]
        for label, value in data.items():
            width = int((value / max_val) * bar_width) if max_val > 0 else 0
            bar = "#" * width + "-" * (bar_width - width)
            pct = (value / denominator_total) * 100 if display_total > 0 else 0
            output.append(f"  {label:<12} [{bar}] {value:>6} ({pct:>5.1f}%)")

        output.append(f"  Total Items: {display_total}")
        return "\n".join(output)

    @staticmethod
    def progress_bar(current: int, total: int, width: int = 40, prefix: str = "") -> str:
        """Generate a progress bar string."""
        pct = 100 if total == 0 else (current / total) * 100
        filled = int(width * current // total) if total > 0 else width
        bar = "#" * filled + "-" * (width - filled)
        return f"\r{prefix} [{bar}] {current}/{total} ({pct:.1f}%)"

    @staticmethod
    def group_distribution(groups: Dict[str, int], top_n: int = 10) -> str:
        """Show group size distribution."""
        if not groups:
            return "  (No groups)"

        sorted_groups = sorted(groups.items(), key=lambda x: x[1], reverse=True)[:top_n]
        output = ["", "Group Size Distribution", "-----------------------"]

        if not sorted_groups:
            output.append("  (No groups found)")
        else:
            max_count = sorted_groups[0][1] if sorted_groups else 1
            bar_width = 30
            for index, (gid, count) in enumerate(sorted_groups, start=1):
                short_id = gid[:15] + "..." if len(gid) > 15 else gid
                width = int((count / max_count) * bar_width) if max_count > 0 else 0
                bar = "#" * width
                output.append(f"  {index:>2}. {short_id:<18} | {bar:<{bar_width}} | {count:>4}")

        return "\n".join(output)
