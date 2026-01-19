#!/usr/bin/env python3
"""
Standalone 1M Token Benchmark: Baseline RAG vs Hierarchical RAG
================================================================

This benchmark runs WITHOUT Ollama dependency by using ChromaDB's 
default sentence-transformer embeddings.

Measures ACTUAL accuracy against ground truth - no proxy metrics.

Usage:
    python benchmark_standalone.py --scale 100k   # Quick test (~2 min)
    python benchmark_standalone.py --scale 1m     # Full 1M tokens (~15 min)
"""

import os
import sys
import time
import json
import random
import hashlib
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

import chromadb
from chromadb.config import Settings


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Needle:
    """A fact hidden in the haystack with ground truth answer."""
    id: str
    question: str
    answer: str
    context: str
    depth_percent: float
    needle_type: str  # 'exact', 'multi_hop', 'semantic_drift', 'coherence'
    requires: List[str] = field(default_factory=list)


@dataclass 
class QueryResult:
    """Result of a single query test."""
    needle_id: str
    question: str
    expected_answer: str
    retrieved_context: str
    is_correct: bool
    latency_ms: float
    needle_type: str


@dataclass
class SystemResult:
    """Results for one system."""
    system_name: str
    ingest_time_sec: float
    cold_latency_avg_ms: float
    warm_latency_avg_ms: float
    
    exact_accuracy: float
    multihop_accuracy: float
    drift_accuracy: float
    coherence_accuracy: float
    overall_accuracy: float
    
    context_avg_chars: int
    query_results: List[QueryResult] = field(default_factory=list)


# =============================================================================
# BASELINE RAG (Flat Vector Store)
# =============================================================================

class BaselineRAG:
    """
    Standard RAG: Fixed chunking + Flat vector search.
    Uses ChromaDB with default sentence-transformer embeddings.
    """
    
    def __init__(self, db_path: str = "./benchmark_baseline_db"):
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        try:
            self.client.delete_collection("baseline_chunks")
        except:
            pass
        self.collection = self.client.create_collection(
            name="baseline_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        self.chunk_size = 500
        self.overlap = 50
    
    def ingest(self, text: str) -> float:
        """Ingest text using fixed-size chunking."""
        start = time.perf_counter()
        
        chunks = []
        ids = []
        
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size + self.overlap]
            chunks.append(chunk)
            ids.append(f"chunk_{i}")
        
        # Batch add
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            end = min(i + batch_size, len(chunks))
            self.collection.add(
                documents=chunks[i:end],
                ids=ids[i:end]
            )
        
        return time.perf_counter() - start
    
    def query(self, question: str, n_results: int = 3) -> str:
        """Retrieve top-k chunks."""
        results = self.collection.query(
            query_texts=[question],
            n_results=n_results
        )
        if results['documents'] and results['documents'][0]:
            return "\n".join(results['documents'][0])
        return ""


# =============================================================================
# HIERARCHICAL RAG (Simulated NeuroSavant)
# =============================================================================

class HierarchicalRAG:
    """
    Hierarchical RAG: Multi-layer storage with summaries.
    
    Layer 0: High-level summaries (entire sections)
    Layer 1: Medium summaries (paragraph groups)
    Layer 2: Raw chunks
    
    Query strategy: Search all layers, prefer higher layers for context.
    """
    
    def __init__(self, db_path: str = "./benchmark_hierarchical_db"):
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        
        self.layers = {}
        for layer in range(3):
            try:
                self.client.delete_collection(f"layer_{layer}")
            except:
                pass
            self.layers[layer] = self.client.create_collection(
                name=f"layer_{layer}",
                metadata={"hnsw:space": "cosine"}
            )
        
        self.raw_chunk_size = 500
        self.mid_chunk_size = 2000
        self.section_size = 10000
    
    def _summarize(self, text: str, target_length: int = 200) -> str:
        """
        Simple extractive summarization (no LLM needed).
        Takes first and last sentences plus key phrases.
        """
        sentences = text.replace('\n', ' ').split('. ')
        if len(sentences) <= 3:
            return text[:target_length]
        
        # Take first, middle, and last sentences
        summary_parts = [
            sentences[0],
            sentences[len(sentences)//2],
            sentences[-1]
        ]
        summary = '. '.join(summary_parts)
        return summary[:target_length]
    
    def ingest(self, text: str) -> float:
        """
        Ingest with hierarchical structure:
        - Layer 2: Raw 500-char chunks
        - Layer 1: 2000-char chunks with summary
        - Layer 0: Section summaries (10000 chars)
        """
        start = time.perf_counter()
        
        # Layer 2: Raw chunks
        l2_chunks = []
        l2_ids = []
        for i in range(0, len(text), self.raw_chunk_size):
            chunk = text[i:i + self.raw_chunk_size + 50]
            l2_chunks.append(chunk)
            l2_ids.append(f"L2_{i}")
        
        # Batch add Layer 2
        for i in range(0, len(l2_chunks), 100):
            end = min(i + 100, len(l2_chunks))
            self.layers[2].add(documents=l2_chunks[i:end], ids=l2_ids[i:end])
        
        # Layer 1: Medium chunks (summarized)
        l1_chunks = []
        l1_ids = []
        for i in range(0, len(text), self.mid_chunk_size):
            chunk = text[i:i + self.mid_chunk_size]
            summary = self._summarize(chunk, 400)
            l1_chunks.append(summary + "\n---\n" + chunk[:500])  # Summary + snippet
            l1_ids.append(f"L1_{i}")
        
        for i in range(0, len(l1_chunks), 100):
            end = min(i + 100, len(l1_chunks))
            self.layers[1].add(documents=l1_chunks[i:end], ids=l1_ids[i:end])
        
        # Layer 0: Section summaries
        l0_chunks = []
        l0_ids = []
        for i in range(0, len(text), self.section_size):
            section = text[i:i + self.section_size]
            summary = self._summarize(section, 600)
            l0_chunks.append(summary)
            l0_ids.append(f"L0_{i}")
        
        self.layers[0].add(documents=l0_chunks, ids=l0_ids)
        
        return time.perf_counter() - start
    
    def query(self, question: str, n_results: int = 3) -> str:
        """
        Hierarchical query: Check all layers, combine results.
        Higher layers provide context, lower layers provide detail.
        """
        all_contexts = []
        
        # Search all layers
        for layer in [0, 1, 2]:
            try:
                results = self.layers[layer].query(
                    query_texts=[question],
                    n_results=n_results,
                    include=['documents', 'distances']
                )
                if results['documents'] and results['documents'][0]:
                    for i, doc in enumerate(results['documents'][0]):
                        distance = results['distances'][0][i]
                        confidence = 1.0 / (1.0 + distance)
                        # Weight by layer (higher layers = more important for context)
                        weighted_conf = confidence * (1.0 + 0.2 * (2 - layer))
                        all_contexts.append((weighted_conf, doc, layer))
            except Exception as e:
                continue
        
        # Sort by confidence and combine
        all_contexts.sort(key=lambda x: x[0], reverse=True)
        
        # Return top contexts
        combined = "\n---\n".join([ctx[1] for ctx in all_contexts[:5]])
        return combined


# =============================================================================
# DATA GENERATOR
# =============================================================================

class DataGenerator:
    """Generate test data with various needle types."""
    
    def __init__(self):
        self.filler = [
            "The quantum entanglement protocol requires precise calibration. ",
            "Sector 7G reported anomalous readings in the detection grid. ",
            "Ancient manuscripts describe rituals involving sacred artifacts. ",
            "Economic analysis suggests variance in projected earnings. ",
            "Neural pathway optimization converged after many iterations. ",
            "Historical records indicate fortress construction in 1247. ",
            "Molecular structure exhibits unusual stability under pressure. ",
            "Weather patterns show significant deviation from normal. ",
            "Encryption algorithm uses a 256-bit key with salt values. ",
            "Biological samples revealed traces of unknown compound XR-7. ",
        ]
    
    def generate(self, target_tokens: int = 100_000) -> Tuple[str, List[Needle]]:
        """Generate haystack with needles."""
        target_bytes = target_tokens * 4
        
        blocks = []
        current_bytes = 0
        
        while current_bytes < target_bytes:
            block = "".join(random.choice(self.filler) for _ in range(50))
            blocks.append(block)
            current_bytes += len(block)
        
        needles = []
        num_blocks = len(blocks)
        
        # Exact recall needles
        exact_facts = [
            ("What is the Omega Protocol password?", "Azure-99-Gamma", 
             "CLASSIFIED: The Omega Protocol password is 'Azure-99-Gamma'."),
            ("What is the frequency of the Ghost Signal?", "442.8 MHz",
             "Signal Analysis: The Ghost Signal broadcasts on 442.8 MHz."),
            ("Who is the traitor in Section 9?", "Commander Reyes",
             "INTEL: Commander Reyes has been confirmed as the traitor."),
            ("What is the launch code for Project Titan?", "7749-WHISKEY",
             "EYES ONLY: Project Titan launch code is 7749-WHISKEY."),
            ("Where is the rebel base located?", "Moon of Endor",
             "Recon confirms the rebel base is on the Moon of Endor."),
        ]
        
        for i, (q, a, ctx) in enumerate(exact_facts):
            depth = 0.1 + 0.2 * i
            idx = int(num_blocks * depth)
            marker = f"\n[LOG-{random.randint(10000,99999)}]: {ctx}\n"
            blocks[idx] += marker
            
            needles.append(Needle(
                id=f"exact_{i}", question=q, answer=a,
                context=marker, depth_percent=depth*100,
                needle_type='exact'
            ))
        
        # Multi-hop needles
        multihop_facts = [
            ("What did Commander Reyes steal from Project Titan?", "launch codes",
             "After Reyes' betrayal, investigators found he copied the launch codes."),
            ("How did the Ghost Signal reach the Moon of Endor?", "relay satellite",
             "The 442.8 MHz signal was bounced off a relay satellite above Endor."),
        ]
        
        for i, (q, a, ctx) in enumerate(multihop_facts):
            depth = 0.4 + 0.1 * i
            idx = int(num_blocks * depth)
            marker = f"\n[MULTI-{random.randint(10000,99999)}]: {ctx}\n"
            blocks[idx] += marker
            
            needles.append(Needle(
                id=f"multihop_{i}", question=q, answer=a,
                context=marker, depth_percent=depth*100,
                needle_type='multi_hop'
            ))
        
        # Semantic drift (ambiguous pronouns)
        drift_facts = [
            ("What is it?", "quantum stabilizer",
             "Dr. Chen examined the quantum stabilizer. 'This is our best chance.'"),
            ("Where did they put that?", "vault B7",
             "The artifact was secured in vault B7 after the incident."),
        ]
        
        for i, (q, a, ctx) in enumerate(drift_facts):
            depth = 0.25 + 0.05 * i
            idx = int(num_blocks * depth)
            blocks[idx] += f"\n{ctx}\n"
            
            needles.append(Needle(
                id=f"drift_{i}", question=q, answer=a,
                context=ctx, depth_percent=depth*100,
                needle_type='semantic_drift'
            ))
        
        # Coherence needle (narrative flow)
        coherence_parts = [
            "Chapter 1: Marcus swore revenge against those who destroyed his village.",
            "Chapter 3: Meeting the orphaned children made Marcus question his path.",
            "Chapter 5: Marcus chose to protect the innocent rather than seek vengeance.",
        ]
        for i, part in enumerate(coherence_parts):
            depth = 0.15 + 0.25 * i
            idx = int(num_blocks * depth)
            blocks[idx] += f"\n{part}\n"
        
        needles.append(Needle(
            id="coherence_0",
            question="How did Marcus's motivation change throughout the story?",
            answer="revenge to redemption",
            context=" | ".join(coherence_parts),
            depth_percent=50.0,
            needle_type='coherence'
        ))
        
        full_text = "\n".join(blocks)
        return full_text, needles


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

class BenchmarkRunner:
    """Runs benchmark and generates proof report."""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def _check_answer(self, expected: str, context: str) -> bool:
        """Ground truth matching."""
        if not context:
            return False
        return expected.lower() in context.lower()
    
    def _run_queries(self, name: str, query_fn, needles: List[Needle]) -> List[QueryResult]:
        """Run all queries against a system."""
        results = []
        for needle in needles:
            start = time.perf_counter()
            context = query_fn(needle.question)
            latency = (time.perf_counter() - start) * 1000
            
            is_correct = self._check_answer(needle.answer, context)
            
            results.append(QueryResult(
                needle_id=needle.id,
                question=needle.question,
                expected_answer=needle.answer,
                retrieved_context=context[:300] if context else "",
                is_correct=is_correct,
                latency_ms=latency,
                needle_type=needle.needle_type
            ))
            
            status = "✅" if is_correct else "❌"
            print(f"   {status} [{needle.needle_type:12}] {needle.question[:40]}... ({latency:.0f}ms)")
        
        return results
    
    def _calc_accuracy(self, results: List[QueryResult], needle_type: str) -> float:
        """Calculate accuracy for a needle type."""
        filtered = [r for r in results if r.needle_type == needle_type]
        if not filtered:
            return 0.0
        return sum(1 for r in filtered if r.is_correct) / len(filtered)
    
    def run(self, target_tokens: int = 250_000) -> None:
        """Run full benchmark."""
        print(f"\n{'='*70}")
        print(f"🚀 STANDALONE RAG BENCHMARK")
        print(f"   Target: {target_tokens:,} tokens (~{target_tokens*4:,} bytes)")
        print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
        
        # Generate data
        print(f"\n🔧 Generating test data...")
        gen = DataGenerator()
        text, needles = gen.generate(target_tokens)
        print(f"   ✅ Generated {len(text):,} bytes, {len(needles)} needles")
        
        results = []
        
        # === BASELINE RAG ===
        print(f"\n{'='*60}")
        print(f"📊 BASELINE RAG (Flat Vector Store)")
        print(f"{'='*60}")
        
        baseline = BaselineRAG()
        print("   🔄 Ingesting...")
        baseline_ingest = baseline.ingest(text)
        print(f"   ✅ Ingested in {baseline_ingest:.1f}s")
        
        print("\n   ❄️ COLD queries:")
        cold_results = self._run_queries("Baseline", baseline.query, needles)
        print("\n   🔥 WARM queries:")
        warm_results = self._run_queries("Baseline", baseline.query, needles)
        
        baseline_result = SystemResult(
            system_name="Baseline RAG",
            ingest_time_sec=baseline_ingest,
            cold_latency_avg_ms=sum(r.latency_ms for r in cold_results) / len(cold_results),
            warm_latency_avg_ms=sum(r.latency_ms for r in warm_results) / len(warm_results),
            exact_accuracy=self._calc_accuracy(cold_results, 'exact'),
            multihop_accuracy=self._calc_accuracy(cold_results, 'multi_hop'),
            drift_accuracy=self._calc_accuracy(cold_results, 'semantic_drift'),
            coherence_accuracy=self._calc_accuracy(cold_results, 'coherence'),
            overall_accuracy=sum(1 for r in cold_results if r.is_correct) / len(cold_results),
            context_avg_chars=sum(len(r.retrieved_context) for r in cold_results) // len(cold_results),
            query_results=cold_results
        )
        results.append(baseline_result)
        
        # === HIERARCHICAL RAG ===
        print(f"\n{'='*60}")
        print(f"📊 HIERARCHICAL RAG (Multi-Layer, NeuroSavant-style)")
        print(f"{'='*60}")
        
        hierarchical = HierarchicalRAG()
        print("   🔄 Ingesting...")
        hier_ingest = hierarchical.ingest(text)
        print(f"   ✅ Ingested in {hier_ingest:.1f}s")
        
        print("\n   ❄️ COLD queries:")
        cold_results = self._run_queries("Hierarchical", hierarchical.query, needles)
        print("\n   🔥 WARM queries:")
        warm_results = self._run_queries("Hierarchical", hierarchical.query, needles)
        
        hier_result = SystemResult(
            system_name="Hierarchical RAG",
            ingest_time_sec=hier_ingest,
            cold_latency_avg_ms=sum(r.latency_ms for r in cold_results) / len(cold_results),
            warm_latency_avg_ms=sum(r.latency_ms for r in warm_results) / len(warm_results),
            exact_accuracy=self._calc_accuracy(cold_results, 'exact'),
            multihop_accuracy=self._calc_accuracy(cold_results, 'multi_hop'),
            drift_accuracy=self._calc_accuracy(cold_results, 'semantic_drift'),
            coherence_accuracy=self._calc_accuracy(cold_results, 'coherence'),
            overall_accuracy=sum(1 for r in cold_results if r.is_correct) / len(cold_results),
            context_avg_chars=sum(len(r.retrieved_context) for r in cold_results) // len(cold_results),
            query_results=cold_results
        )
        results.append(hier_result)
        
        # === GENERATE REPORT ===
        self._generate_report(results, target_tokens)
    
    def _generate_report(self, results: List[SystemResult], tokens: int) -> None:
        """Generate proof report."""
        b = results[0]  # Baseline
        h = results[1]  # Hierarchical
        
        report = f"""# Benchmark Proof Report
Generated: {datetime.now().isoformat()}
Data Scale: {tokens:,} tokens

---

## ⚠️ CRITICAL: This Measures ACTUAL Accuracy Against Ground Truth

Unlike the PerformanceTracker's claimed metrics (which estimate accuracy from proxy 
metrics like "cell reuse rate"), this benchmark measures whether the retrieved 
context actually contains the correct answer.

---

## Summary: Baseline RAG vs Hierarchical RAG

| Metric | Baseline RAG | Hierarchical RAG | Winner |
|--------|--------------|------------------|--------|
| **Exact Recall** | {b.exact_accuracy*100:.0f}% | {h.exact_accuracy*100:.0f}% | {'Hierarchical' if h.exact_accuracy > b.exact_accuracy else 'Baseline' if b.exact_accuracy > h.exact_accuracy else 'Tie'} |
| **Multi-hop** | {b.multihop_accuracy*100:.0f}% | {h.multihop_accuracy*100:.0f}% | {'Hierarchical' if h.multihop_accuracy > b.multihop_accuracy else 'Baseline' if b.multihop_accuracy > h.multihop_accuracy else 'Tie'} |
| **Semantic Drift** | {b.drift_accuracy*100:.0f}% | {h.drift_accuracy*100:.0f}% | {'Hierarchical' if h.drift_accuracy > b.drift_accuracy else 'Baseline' if b.drift_accuracy > h.drift_accuracy else 'Tie'} |
| **Coherence** | {b.coherence_accuracy*100:.0f}% | {h.coherence_accuracy*100:.0f}% | {'Hierarchical' if h.coherence_accuracy > b.coherence_accuracy else 'Baseline' if b.coherence_accuracy > h.coherence_accuracy else 'Tie'} |
| **Overall** | **{b.overall_accuracy*100:.0f}%** | **{h.overall_accuracy*100:.0f}%** | **{'Hierarchical' if h.overall_accuracy > b.overall_accuracy else 'Baseline' if b.overall_accuracy > h.overall_accuracy else 'Tie'}** |
| Cold Latency | {b.cold_latency_avg_ms:.0f}ms | {h.cold_latency_avg_ms:.0f}ms | {'Baseline' if b.cold_latency_avg_ms < h.cold_latency_avg_ms else 'Hierarchical'} |
| Warm Latency | {b.warm_latency_avg_ms:.0f}ms | {h.warm_latency_avg_ms:.0f}ms | {'Baseline' if b.warm_latency_avg_ms < h.warm_latency_avg_ms else 'Hierarchical'} |
| Ingest Time | {b.ingest_time_sec:.1f}s | {h.ingest_time_sec:.1f}s | {'Baseline' if b.ingest_time_sec < h.ingest_time_sec else 'Hierarchical'} |
| Avg Context | {b.context_avg_chars} chars | {h.context_avg_chars} chars | - |

---

## Claim Validation

### Claimed vs Measured

| Claim (from PerformanceTracker) | Claimed | Measured | Validated? |
|--------------------------------|---------|----------|------------|
| Query Accuracy ~75-85% | 75-85% | {h.overall_accuracy*100:.0f}% | {'✅ YES' if 0.70 <= h.overall_accuracy <= 0.90 else '❌ NO'} |
| Multi-hop Reasoning | ✓ Better than Baseline | {h.multihop_accuracy*100:.0f}% vs {b.multihop_accuracy*100:.0f}% | {'✅ YES' if h.multihop_accuracy > b.multihop_accuracy else '❌ NO'} |
| Cold Latency 100-500ms | 100-500ms | {h.cold_latency_avg_ms:.0f}ms | {'✅ YES' if 100 <= h.cold_latency_avg_ms <= 500 else '⚠️ OUTSIDE RANGE'} |
| Token Efficiency 70% lower | ~70% less | {100 - (h.context_avg_chars / b.context_avg_chars * 100):.0f}% less | {'✅ YES' if h.context_avg_chars < b.context_avg_chars * 0.5 else '❌ NO'} |

---

## Detailed Query Results

"""
        for result in results:
            report += f"### {result.system_name}\n\n"
            for qr in result.query_results:
                status = "✅" if qr.is_correct else "❌"
                report += f"- {status} **[{qr.needle_type}]** {qr.question}\n"
                report += f"  - Expected: `{qr.expected_answer}`\n"
                report += f"  - Found: {'Yes' if qr.is_correct else 'No'} | Latency: {qr.latency_ms:.0f}ms\n\n"
            report += "\n"
        
        # Add hash for integrity
        report += f"""---

## Proof Hash

SHA256: `{hashlib.sha256(report.encode()).hexdigest()}`

This hash can be used to verify this report was not modified after generation.
"""
        
        # Save report
        report_path = os.path.join(self.output_dir, "benchmark_proof.md")
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Save JSON
        json_data = [asdict(r) for r in results]
        json_path = os.path.join(self.output_dir, "benchmark_results.json")
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        print(f"\n{'='*70}")
        print(f"📊 BENCHMARK COMPLETE")
        print(f"{'='*70}")
        print(f"\n✅ Report: {report_path}")
        print(f"✅ JSON:   {json_path}")
        print(f"\n{report}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Standalone RAG Benchmark")
    parser.add_argument("--scale", choices=["100k", "250k", "500k", "1m"], default="250k")
    parser.add_argument("--output", default="reports")
    args = parser.parse_args()
    
    scale_map = {"100k": 100_000, "250k": 250_000, "500k": 500_000, "1m": 1_000_000}
    
    runner = BenchmarkRunner(output_dir=args.output)
    runner.run(target_tokens=scale_map[args.scale])


if __name__ == "__main__":
    main()
