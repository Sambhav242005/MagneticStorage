#!/usr/bin/env python3
"""
NeuroSavant 2.0 Benchmark
=========================

Compares:
1. Baseline RAG (Flat Vector)
2. NeuroSavant 2.0 (Hybrid Graph-Vector)
3. GraphRAG (Simulated)

Target:
- Exact Recall: >80%
- Multi-hop: >40%
"""

import os
import sys
import time
import json
import random
import hashlib
import re
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict

import chromadb
from chromadb.config import Settings

# Import NeuroSavant V2
from neuro_savant_v2 import NeuroSavantV2

# Reuse data structures from benchmark_full.py
from benchmark_full import (
    Needle, QueryResult, SystemResult, 
    BaselineRAG, GraphRAG, DataGenerator, BenchmarkRunner
)

# =============================================================================
# ADAPTER FOR NEURO SAVANT V2
# =============================================================================

class NeuroSavantV2Adapter:
    """Adapts NeuroSavantV2 to the benchmark interface."""
    
    def __init__(self):
        self.ns = NeuroSavantV2()
        self.ns.memory.clear()
        self.num_collections = 2
        self.num_embeddings = 0
        
    def ingest(self, text: str) -> float:
        stats = self.ns.ingest(text)
        self.num_embeddings = stats['chunks'] + stats['entities']
        return stats['time']
    
    def query(self, question: str) -> str:
        return self.ns.query(question)


# =============================================================================
# V2 BENCHMARK RUNNER
# =============================================================================

class V2BenchmarkRunner(BenchmarkRunner):
    
    def run(self, target_tokens: int = 250_000) -> None:
        print(f"\n{'='*70}")
        print(f"🚀 NEURO SAVANT 2.0 BENCHMARK")
        print(f"   Target: {target_tokens:,} tokens")
        print(f"{'='*70}")
        
        gen = DataGenerator()
        text, needles = gen.generate(target_tokens)
        print(f"\n🔧 Generated {len(text):,} bytes, {len(needles)} needles")
        
        results = []
        
        # 1. Baseline RAG
        results.append(self._benchmark_system(BaselineRAG(), "Baseline RAG", text, needles))
        
        # 2. NeuroSavant 2.0
        results.append(self._benchmark_system(NeuroSavantV2Adapter(), "NeuroSavant 2.0", text, needles))
        
        # 3. GraphRAG
        results.append(self._benchmark_system(GraphRAG(), "GraphRAG (Simulated)", text, needles))
        
        self._generate_report(results, target_tokens)

    def _generate_report(self, results: List[SystemResult], tokens: int) -> None:
        b, ns2, g = results[0], results[1], results[2]
        
        report = f"""# NeuroSavant 2.0 Benchmark Report
Generated: {datetime.now().isoformat()}
Data Scale: {tokens:,} tokens

---

## Executive Summary

| Metric | Baseline | NeuroSavant 2.0 | GraphRAG | Winner |
|--------|----------|-----------------|----------|--------|
| **Exact Recall** | {b.exact_accuracy*100:.0f}% | {ns2.exact_accuracy*100:.0f}% | {g.exact_accuracy*100:.0f}% | {max([(b.exact_accuracy, 'Baseline'), (ns2.exact_accuracy, 'NS 2.0'), (g.exact_accuracy, 'Graph')], key=lambda x: x[0])[1]} |
| **Multi-hop** | {b.multihop_accuracy*100:.0f}% | {ns2.multihop_accuracy*100:.0f}% | {g.multihop_accuracy*100:.0f}% | {max([(b.multihop_accuracy, 'Baseline'), (ns2.multihop_accuracy, 'NS 2.0'), (g.multihop_accuracy, 'Graph')], key=lambda x: x[0])[1]} |
| **Overall** | **{b.overall_accuracy*100:.0f}%** | **{ns2.overall_accuracy*100:.0f}%** | **{g.overall_accuracy*100:.0f}%** | **{max([(b.overall_accuracy, 'Baseline'), (ns2.overall_accuracy, 'NS 2.0'), (g.overall_accuracy, 'Graph')], key=lambda x: x[0])[1]}** |
| Cold Latency | {b.cold_latency_avg_ms:.0f}ms | {ns2.cold_latency_avg_ms:.0f}ms | {g.cold_latency_avg_ms:.0f}ms | {min([(b.cold_latency_avg_ms, 'Baseline'), (ns2.cold_latency_avg_ms, 'NS 2.0'), (g.cold_latency_avg_ms, 'Graph')], key=lambda x: x[0])[1]} |

---

## Success Criteria Validation

| Goal | Target | NeuroSavant 2.0 | Status |
|------|--------|-----------------|--------|
| **Exact Recall** | >80% | {ns2.exact_accuracy*100:.0f}% | {'✅ PASS' if ns2.exact_accuracy >= 0.8 else '❌ FAIL'} |
| **Multi-hop** | >40% | {ns2.multihop_accuracy*100:.0f}% | {'✅ PASS' if ns2.multihop_accuracy >= 0.4 else '❌ FAIL'} |
| **Latency** | <500ms | {ns2.cold_latency_avg_ms:.0f}ms | {'✅ PASS' if ns2.cold_latency_avg_ms <= 500 else '⚠️ WARN'} |

---

## Analysis

**NeuroSavant 2.0 Architecture:**
- **Hybrid Memory**: Combines Raw Chunks (Vector) + Entities (Graph).
- **Unified Query**: Runs both searches in parallel.

**Why it works:**
- **Exact Recall**: The vector search path ensures no data loss (unlike v1's summaries).
- **Multi-hop**: The graph path finds relationships (e.g. "Reyes" -> "Titan").

---

## Raw Data

SHA256: `{hashlib.sha256(str(results).encode()).hexdigest()}`
"""
        
        # Save
        with open(os.path.join(self.output_dir, "benchmark_v2_proof.md"), 'w') as f:
            f.write(report)
        
        print(f"\n{'='*70}")
        print("📊 BENCHMARK COMPLETE")
        print(f"{'='*70}")
        print(report)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="NeuroSavant 2.0 Benchmark")
    parser.add_argument("--scale", choices=["100k", "250k", "500k", "1m"], default="250k")
    parser.add_argument("--output", default="reports")
    args = parser.parse_args()
    
    scale_map = {"100k": 100_000, "250k": 250_000, "500k": 500_000, "1m": 1_000_000}
    V2BenchmarkRunner(output_dir=args.output).run(target_tokens=scale_map[args.scale])


if __name__ == "__main__":
    main()
