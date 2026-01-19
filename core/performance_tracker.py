"""
Performance Tracker for NeuroSavant

Tracks and displays performance metrics:
- Query latency
- Ingestion throughput  
- Memory usage
- Embedding time

Provides terminal visualization with ASCII charts.
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque


@dataclass
class PerformanceMetrics:
    """Store performance metrics with rolling averages"""
    query_times: deque = field(default_factory=lambda: deque(maxlen=100))
    ingest_times: deque = field(default_factory=lambda: deque(maxlen=100))
    embed_times: deque = field(default_factory=lambda: deque(maxlen=100))
    batch_sizes: deque = field(default_factory=lambda: deque(maxlen=100))
    
    total_queries: int = 0
    total_ingests: int = 0
    total_cells: int = 0
    
    # Tool metrics
    tool_usage: Dict[str, int] = field(default_factory=dict)
    tool_errors: int = 0
    total_tool_calls: int = 0


class PerformanceTracker:
    """
    Tracks performance metrics for NeuroSavant operations.
    Provides timing decorators and visualization.
    """
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self._current_timer: Optional[float] = None
    
    def start_timer(self) -> float:
        """Start a timing operation"""
        return time.perf_counter()
    
    def record_query(self, start_time: float):
        """Record a query operation time"""
        elapsed = (time.perf_counter() - start_time) * 1000  # ms
        self.metrics.query_times.append(elapsed)
        self.metrics.total_queries += 1
        return elapsed
    
    def record_ingest(self, start_time: float, count: int = 1):
        """Record an ingest operation time"""
        elapsed = (time.perf_counter() - start_time) * 1000  # ms
        self.metrics.ingest_times.append(elapsed)
        self.metrics.batch_sizes.append(count)
        self.metrics.total_ingests += count
        return elapsed
    
    def record_embed(self, start_time: float):
        """Record an embedding operation time"""
        elapsed = (time.perf_counter() - start_time) * 1000  # ms
        self.metrics.embed_times.append(elapsed)
        return elapsed
        
    def record_tool_usage(self, tool_name: str, success: bool = True):
        """Record a tool execution"""
        self.metrics.total_tool_calls += 1
        self.metrics.tool_usage[tool_name] = self.metrics.tool_usage.get(tool_name, 0) + 1
        if not success:
            self.metrics.tool_errors += 1
    
    def get_stats(self) -> Dict:
        """Get current statistics"""
        def avg(dq):
            return sum(dq) / len(dq) if dq else 0
        
        def percentile(dq, p):
            if not dq:
                return 0
            sorted_vals = sorted(dq)
            idx = int(len(sorted_vals) * p / 100)
            return sorted_vals[min(idx, len(sorted_vals) - 1)]
        
        return {
            'query': {
                'avg_ms': avg(self.metrics.query_times),
                'p95_ms': percentile(self.metrics.query_times, 95),
                'total': self.metrics.total_queries
            },
            'ingest': {
                'avg_ms': avg(self.metrics.ingest_times),
                'total': self.metrics.total_ingests,
                'throughput': (self.metrics.total_ingests / 
                              (sum(self.metrics.ingest_times) / 1000)) 
                              if self.metrics.ingest_times else 0
            },
            'embed': {
                'avg_ms': avg(self.metrics.embed_times),
                'p95_ms': percentile(self.metrics.embed_times, 95)
            },
            'tools': {
                'total': self.metrics.total_tool_calls,
                'errors': self.metrics.tool_errors,
                'usage': self.metrics.tool_usage
            }
        }
    
    def display_stats(self) -> str:
        """Generate formatted performance stats display"""
        stats = self.get_stats()
        
        output = []
        output.append("")
        output.append("╔══════════════════════════════════════════════════════════════╗")
        output.append("║                    PERFORMANCE METRICS                       ║")
        output.append("╠══════════════════════════════════════════════════════════════╣")
        output.append(f"║  QUERIES                                                     ║")
        output.append(f"║    Total:        {stats['query']['total']:>8}                              ║")
        output.append(f"║    Avg Latency:  {stats['query']['avg_ms']:>8.2f} ms                          ║")
        output.append(f"║    P95 Latency:  {stats['query']['p95_ms']:>8.2f} ms                          ║")
        output.append("║                                                              ║")
        output.append(f"║  INGESTION                                                   ║")
        output.append(f"║    Total Cells:  {stats['ingest']['total']:>8}                              ║")
        output.append(f"║    Avg Time:     {stats['ingest']['avg_ms']:>8.2f} ms                          ║")
        output.append(f"║    Throughput:   {stats['ingest']['throughput']:>8.1f} cells/sec                   ║")
        output.append("║                                                              ║")
        output.append(f"║  EMBEDDING                                                   ║")
        output.append(f"║    Avg Time:     {stats['embed']['avg_ms']:>8.2f} ms                          ║")
        output.append(f"║    P95 Time:     {stats['embed']['p95_ms']:>8.2f} ms                          ║")
        output.append("║                                                              ║")
        output.append(f"║  AGENTIC TOOLS                                               ║")
        output.append(f"║    Total Calls:  {stats['tools']['total']:>8}                              ║")
        output.append(f"║    Errors:       {stats['tools']['errors']:>8}                              ║")
        
        if stats['tools']['usage']:
            output.append("║    Usage Breakdown:                                          ║")
            for name, count in stats['tools']['usage'].items():
                output.append(f"║      - {name:<20}: {count:>4}                          ║")
                
        output.append("╚══════════════════════════════════════════════════════════════╝")
        
        return "\n".join(output)


class VisualDisplay:
    """
    Terminal-based visual display for NeuroSavant data.
    Uses ASCII/Unicode for charts and graphs.
    """
    
    BAR_CHARS = " ▏▎▍▌▋▊▉█"
    
    @staticmethod
    def bar_chart(data: Dict[str, int], max_width: int = 40, title: str = "Chart") -> str:
        """Generate ASCII horizontal bar chart"""
        if not data:
            return "  (No data)"
        
        output = []
        output.append(f"\n┌─ {title} ─" + "─" * (max_width - len(title) - 4) + "┐")
        
        max_val = max(data.values()) if data.values() else 1
        max_label = max(len(str(k)) for k in data.keys()) if data else 10
        
        for label, value in data.items():
            # Calculate bar width
            bar_width = int((value / max_val) * max_width) if max_val > 0 else 0
            bar = "█" * bar_width
            
            # Format line
            line = f"│ {label:<{max_label}} │{bar:<{max_width}}│ {value:>6}"
            output.append(line)
        
        output.append("└" + "─" * (max_label + max_width + 12) + "┘")
        return "\n".join(output)
    
    @staticmethod
    def memory_stats_visual(cells: int, groups: int, entities: int) -> str:
        """Display memory statistics with visual representation"""
        total = cells + groups + entities
        if total == 0:
            total = 1  # Avoid division by zero
        
        output = []
        output.append("")
        output.append("╔══════════════════════════════════════════════════════════════╗")
        output.append("║                     MEMORY VISUALIZATION                     ║")
        output.append("╠══════════════════════════════════════════════════════════════╣")
        
        # Bar chart data
        data = {
            "Cells": cells,
            "Groups": groups,
            "Entity Index": entities
        }
        
        max_val = max(data.values()) if any(data.values()) else 1
        bar_width = 35
        
        for label, value in data.items():
            width = int((value / max_val) * bar_width) if max_val > 0 else 0
            bar = "█" * width + "░" * (bar_width - width)
            pct = (value / total) * 100 if total > 0 else 0
            output.append(f"║  {label:<12} │{bar}│ {value:>6} ({pct:>5.1f}%)  ║")
        
        output.append("║                                                              ║")
        output.append(f"║  Total Items: {total:<46} ║")
        output.append("╚══════════════════════════════════════════════════════════════╝")
        
        return "\n".join(output)
    
    @staticmethod
    def progress_bar(current: int, total: int, width: int = 40, prefix: str = "") -> str:
        """Generate a progress bar string"""
        if total == 0:
            pct = 100
        else:
            pct = (current / total) * 100
        
        filled = int(width * current // total) if total > 0 else width
        bar = "█" * filled + "░" * (width - filled)
        
        return f"\r{prefix} │{bar}│ {current}/{total} ({pct:.1f}%)"
    
    @staticmethod
    def group_distribution(groups: Dict[str, int], top_n: int = 10) -> str:
        """Show group size distribution"""
        if not groups:
            return "  (No groups)"
        
        # Sort by count descending and take top N
        sorted_groups = sorted(groups.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        output = []
        output.append("")
        output.append("╔══════════════════════════════════════════════════════════════╗")
        output.append("║                 GROUP SIZE DISTRIBUTION                      ║")
        output.append("╠══════════════════════════════════════════════════════════════╣")
        
        if not sorted_groups:
            output.append("║  (No groups found)                                           ║")
        else:
            max_count = sorted_groups[0][1] if sorted_groups else 1
            bar_width = 30
            
            for i, (gid, count) in enumerate(sorted_groups):
                short_id = gid[:15] + "..." if len(gid) > 15 else gid
                width = int((count / max_count) * bar_width) if max_count > 0 else 0
                bar = "▓" * width
                output.append(f"║  {i+1:>2}. {short_id:<18} │{bar:<{bar_width}}│ {count:>4}  ║")
        
        output.append("╚══════════════════════════════════════════════════════════════╝")
        
        return "\n".join(output)
