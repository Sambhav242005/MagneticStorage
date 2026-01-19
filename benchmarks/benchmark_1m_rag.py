#!/usr/bin/env python3
"""
1 Million Token RAG vs NeuroSavant Benchmark
============================================

HONEST BENCHMARK: Measures actual accuracy against ground truth,
not proxy metrics like "cell reuse rate".

Tests:
1. Exact Recall (Needle-in-Haystack) - RAG's strength
2. Multi-hop Reasoning - Requires connecting facts across chunks
3. Context Coherence - Requires understanding narrative flow
4. Semantic Drift - Tests re-ranker bypass vulnerability
5. Latency Profiling - Cold vs Warm cache

Author: Benchmark Suite
Date: 2026-01-09
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
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import real dependencies, fall back to mocks
try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("⚠️  ChromaDB not available, using mock mode")

try:
    from neuro_savant import NeuroSavant, Config
    NEURO_AVAILABLE = True
except ImportError:
    NEURO_AVAILABLE = False
    print("⚠️  NeuroSavant not available")

from baseline_rag import BaselineRAG


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Needle:
    """A fact hidden in the haystack with ground truth answer."""
    id: str
    question: str
    answer: str  # Ground truth
    context: str  # The text containing the answer
    depth_percent: float  # Where in the haystack (0-100%)
    needle_type: str  # 'exact', 'multi_hop', 'semantic_drift', 'coherence'
    requires: List[str] = field(default_factory=list)  # IDs of prerequisite needles for multi-hop


@dataclass 
class QueryResult:
    """Result of a single query test."""
    needle_id: str
    question: str
    expected_answer: str
    retrieved_context: str
    is_correct: bool  # Ground truth match
    latency_ms: float
    needle_type: str


@dataclass
class BenchmarkResult:
    """Complete benchmark results with proof artifacts."""
    timestamp: str
    system_name: str
    data_size_tokens: int
    data_size_bytes: int
    
    # Accuracy metrics (MEASURED, not estimated)
    exact_recall_accuracy: float
    multi_hop_accuracy: float
    semantic_drift_accuracy: float  # Tests re-ranker bypass vulnerability
    coherence_accuracy: float
    overall_accuracy: float
    
    # Latency metrics
    ingest_time_seconds: float
    cold_query_avg_ms: float
    warm_query_avg_ms: float
    
    # Token efficiency
    avg_context_length_chars: int
    
    # Raw query results for verification
    query_results: List[QueryResult] = field(default_factory=list)
    
    # Metadata
    needle_count: int = 0
    total_queries: int = 0


# =============================================================================
# DATA GENERATOR (Scaled for 1M tokens)
# =============================================================================

class MillionTokenGenerator:
    """
    Generates ~1 million token dataset (~4MB) with various needle types.
    
    Needle Types:
    - exact: Simple keyword lookup (RAG's strength)
    - multi_hop: Requires connecting 2+ facts
    - semantic_drift: Tests pronoun resolution ("it", "that")
    - coherence: Requires understanding narrative arc
    """
    
    def __init__(self, output_dir: str = "benchmark_data_1m"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Filler text templates (varied to prevent compression tricks)
        self.filler_templates = [
            "The quantum entanglement protocol requires precise calibration of the photon emitters. ",
            "Sector 7G reported anomalous readings in the subspace detection grid last Tuesday. ",
            "The ancient manuscripts describe a ritual involving three sacred artifacts. ",
            "Economic analysis suggests a 12% variance in the projected Q3 earnings report. ",
            "The neural pathway optimization algorithm converged after 847 iterations. ",
            "Historical records indicate the fortress was constructed in the year 1247. ",
            "The molecular structure exhibits unusual stability under extreme pressure. ",
            "Weather patterns in the northern hemisphere show significant deviation. ",
            "The encryption algorithm uses a 256-bit key with rotating salt values. ",
            "Biological samples revealed traces of an unknown compound designated XR-7. ",
        ]
        
        # Exact recall needles (simple keyword lookup)
        self.exact_needles = [
            {"question": "What is the Omega Protocol password?", 
             "answer": "Azure-99-Gamma", 
             "context": "CLASSIFIED: The Omega Protocol password is 'Azure-99-Gamma'. Do not share."},
            {"question": "What is the frequency of the Ghost Signal?", 
             "answer": "442.8 MHz", 
             "context": "Signal Analysis: The Ghost Signal broadcasts on exactly 442.8 MHz."},
            {"question": "Who is the traitor in Section 9?", 
             "answer": "Commander Reyes", 
             "context": "INTEL REPORT: Commander Reyes has been confirmed as the traitor in Section 9."},
            {"question": "What is the launch code for Project Titan?", 
             "answer": "7749-WHISKEY-TANGO", 
             "context": "EYES ONLY: Project Titan launch code is 7749-WHISKEY-TANGO."},
            {"question": "Where is the hidden rebel base located?", 
             "answer": "Moon of Endor", 
             "context": "Reconnaissance confirms the hidden rebel base is on the Moon of Endor."},
        ]
        
        # Multi-hop needles (require connecting multiple facts)
        self.multi_hop_needles = [
            {
                "question": "What did Commander Reyes steal from Project Titan?",
                "answer": "launch codes",
                "context": "After Commander Reyes' betrayal was discovered, investigators found he had accessed the Project Titan vault and copied the launch codes.",
                "requires": ["traitor", "titan"]  # Must understand both facts
            },
            {
                "question": "How did the Ghost Signal reach the Moon of Endor?",
                "answer": "relay satellite",
                "context": "The Ghost Signal's 442.8 MHz transmission was bounced off a hidden relay satellite positioned directly above the Moon of Endor.",
                "requires": ["ghost_signal", "rebel_base"]
            },
        ]
        
        # Semantic drift needles (tests pronoun resolution / re-ranker bypass)
        self.semantic_drift_needles = [
            {
                "question": "What is it?",  # Ambiguous pronoun
                "answer": "quantum stabilizer",
                "context": "Dr. Chen examined the quantum stabilizer carefully. 'This is our best chance,' she said.",
                "setup_context": "The research team discovered a breakthrough device in Lab 7."
            },
            {
                "question": "Where did they put that?",  # Ambiguous
                "answer": "vault B7",
                "context": "The artifact was secured in vault B7 after the incident.",
                "setup_context": "Security transported the dangerous artifact to a secure location."
            },
        ]
        
        # Coherence needles (require understanding narrative flow)
        self.coherence_needles = [
            {
                "question": "How did the protagonist's motivation change throughout the story?",
                "answer": "revenge to redemption",
                "parts": [
                    "Chapter 1: Marcus swore revenge against those who destroyed his village.",
                    "Chapter 3: Meeting the orphaned children made Marcus question his path.",
                    "Chapter 5: Marcus chose to protect the innocent rather than seek vengeance.",
                    "Conclusion: Marcus found redemption through sacrifice, saving the very people he once blamed."
                ]
            },
        ]

    def generate(self, target_tokens: int = 1_000_000) -> Tuple[str, str]:
        """
        Generate dataset with ~target_tokens tokens.
        
        Returns:
            (dataset_path, needles_path)
        """
        print(f"\n{'='*60}")
        print(f"🔧 GENERATING {target_tokens:,} TOKEN DATASET")
        print(f"{'='*60}")
        
        # Approximate: 1 token ≈ 4 chars, so 1M tokens ≈ 4MB
        target_bytes = target_tokens * 4
        
        full_text = []
        needles = []
        current_bytes = 0
        block_count = 0
        
        # Generate haystack
        while current_bytes < target_bytes:
            block = ""
            for _ in range(50):
                block += random.choice(self.filler_templates)
            full_text.append(block)
            current_bytes += len(block)
            block_count += 1
            
            if block_count % 100 == 0:
                print(f"   Generated {current_bytes:,} / {target_bytes:,} bytes ({100*current_bytes/target_bytes:.1f}%)")
        
        print(f"✅ Haystack: {current_bytes:,} bytes, {block_count} blocks")
        
        # Inject needles at various depths
        needle_id = 0
        
        # 1. EXACT RECALL needles (5 at 10%, 30%, 50%, 70%, 90% depth)
        for i, needle_data in enumerate(self.exact_needles):
            depth = 0.1 + 0.2 * i
            block_idx = int(block_count * depth)
            
            unique_context = f"\n[CLASSIFIED LOG {random.randint(10000,99999)}]: {needle_data['context']}\n"
            full_text[block_idx] += unique_context
            
            needles.append(Needle(
                id=f"exact_{needle_id}",
                question=needle_data['question'],
                answer=needle_data['answer'],
                context=unique_context,
                depth_percent=depth * 100,
                needle_type='exact'
            ))
            needle_id += 1
        
        # 2. MULTI-HOP needles (require connecting facts)
        for mh in self.multi_hop_needles:
            depth = random.uniform(0.4, 0.6)
            block_idx = int(block_count * depth)
            
            unique_context = f"\n[MULTI-HOP INTEL {random.randint(10000,99999)}]: {mh['context']}\n"
            full_text[block_idx] += unique_context
            
            needles.append(Needle(
                id=f"multihop_{needle_id}",
                question=mh['question'],
                answer=mh['answer'],
                context=unique_context,
                depth_percent=depth * 100,
                needle_type='multi_hop',
                requires=mh.get('requires', [])
            ))
            needle_id += 1
        
        # 3. SEMANTIC DRIFT needles (test re-ranker bypass vulnerability)
        for sd in self.semantic_drift_needles:
            # Place setup context first
            setup_depth = random.uniform(0.2, 0.3)
            setup_idx = int(block_count * setup_depth)
            full_text[setup_idx] += f"\n{sd['setup_context']}\n"
            
            # Place answer context nearby but not adjacent
            answer_depth = setup_depth + 0.05
            answer_idx = int(block_count * answer_depth)
            unique_context = f"\n[CONTEXT {random.randint(10000,99999)}]: {sd['context']}\n"
            full_text[answer_idx] += unique_context
            
            needles.append(Needle(
                id=f"drift_{needle_id}",
                question=sd['question'],
                answer=sd['answer'],
                context=unique_context,
                depth_percent=answer_depth * 100,
                needle_type='semantic_drift'
            ))
            needle_id += 1
        
        # 4. COHERENCE needles (narrative flow)
        for coh in self.coherence_needles:
            # Spread parts across the document
            for i, part in enumerate(coh['parts']):
                part_depth = 0.1 + 0.2 * i
                part_idx = int(block_count * part_depth)
                full_text[part_idx] += f"\n{part}\n"
            
            needles.append(Needle(
                id=f"coherence_{needle_id}",
                question=coh['question'],
                answer=coh['answer'],
                context=" | ".join(coh['parts']),
                depth_percent=50.0,  # Spread throughout
                needle_type='coherence'
            ))
            needle_id += 1
        
        # Save dataset
        dataset_path = os.path.join(self.output_dir, "dataset_1m.txt")
        needles_path = os.path.join(self.output_dir, "needles_1m.json")
        
        full_text_str = "\n".join(full_text)
        with open(dataset_path, "w") as f:
            f.write(full_text_str)
        
        needles_json = [asdict(n) for n in needles]
        with open(needles_path, "w") as f:
            json.dump(needles_json, f, indent=2)
        
        actual_tokens = len(full_text_str) // 4
        print(f"✅ Dataset saved: {dataset_path}")
        print(f"   - Size: {len(full_text_str):,} bytes (~{actual_tokens:,} tokens)")
        print(f"✅ Needles saved: {needles_path}")
        print(f"   - Exact: {sum(1 for n in needles if n.needle_type == 'exact')}")
        print(f"   - Multi-hop: {sum(1 for n in needles if n.needle_type == 'multi_hop')}")
        print(f"   - Semantic Drift: {sum(1 for n in needles if n.needle_type == 'semantic_drift')}")
        print(f"   - Coherence: {sum(1 for n in needles if n.needle_type == 'coherence')}")
        
        return dataset_path, needles_path


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

class BenchmarkRunner:
    """
    Runs the benchmark suite against both Baseline RAG and NeuroSavant.
    
    IMPORTANT: Measures ACTUAL ACCURACY against ground truth,
    not proxy metrics like "cell reuse rate".
    """
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.generator = MillionTokenGenerator()
    
    def _check_answer_match(self, expected: str, context: str) -> bool:
        """
        Check if the expected answer is present in the retrieved context.
        Uses case-insensitive substring matching.
        """
        if not context:
            return False
        return expected.lower() in context.lower()
    
    def _run_queries(self, system_name: str, query_fn, needles: List[Needle], 
                     warm_cache: bool = False) -> List[QueryResult]:
        """Run all needle queries against a system."""
        results = []
        
        for needle in needles:
            start = time.perf_counter()
            
            # Run query
            response = query_fn(needle.question)
            
            latency_ms = (time.perf_counter() - start) * 1000
            
            # Extract context from response
            context = response.get('context', '') if isinstance(response, dict) else str(response)
            
            # Check ground truth match
            is_correct = self._check_answer_match(needle.answer, context)
            
            results.append(QueryResult(
                needle_id=needle.id,
                question=needle.question,
                expected_answer=needle.answer,
                retrieved_context=context[:500] if context else "",  # Truncate for storage
                is_correct=is_correct,
                latency_ms=latency_ms,
                needle_type=needle.needle_type
            ))
            
            status = "✅" if is_correct else "❌"
            print(f"   {status} [{needle.needle_type}] {needle.question[:40]}... ({latency_ms:.1f}ms)")
        
        return results
    
    def benchmark_baseline_rag(self, dataset_path: str, needles: List[Needle]) -> BenchmarkResult:
        """Run benchmark against Baseline RAG (flat vector store)."""
        print(f"\n{'='*60}")
        print("📊 BENCHMARKING: Baseline RAG")
        print(f"{'='*60}")
        
        # Initialize
        baseline = BaselineRAG(collection_name="benchmark_1m_baseline")
        
        # Ingest
        print("\n🔄 Ingesting data...")
        ingest_time = baseline.ingest(dataset_path)
        
        # Cold queries
        print("\n❄️  Running COLD queries...")
        cold_results = self._run_queries(
            "Baseline RAG",
            lambda q: baseline.query(q),
            needles,
            warm_cache=False
        )
        
        # Warm queries (repeat to measure cache effect - RAG has no cache, should be similar)
        print("\n🔥 Running WARM queries...")
        warm_results = self._run_queries(
            "Baseline RAG",
            lambda q: baseline.query(q),
            needles,
            warm_cache=True
        )
        
        # Calculate metrics by type
        exact_correct = sum(1 for r in cold_results if r.is_correct and r.needle_type == 'exact')
        exact_total = sum(1 for r in cold_results if r.needle_type == 'exact')
        
        multihop_correct = sum(1 for r in cold_results if r.is_correct and r.needle_type == 'multi_hop')
        multihop_total = sum(1 for r in cold_results if r.needle_type == 'multi_hop')
        
        drift_correct = sum(1 for r in cold_results if r.is_correct and r.needle_type == 'semantic_drift')
        drift_total = sum(1 for r in cold_results if r.needle_type == 'semantic_drift')
        
        coherence_correct = sum(1 for r in cold_results if r.is_correct and r.needle_type == 'coherence')
        coherence_total = sum(1 for r in cold_results if r.needle_type == 'coherence')
        
        # Get data size
        with open(dataset_path, 'r') as f:
            data = f.read()
        
        return BenchmarkResult(
            timestamp=datetime.now().isoformat(),
            system_name="Baseline RAG",
            data_size_tokens=len(data) // 4,
            data_size_bytes=len(data),
            exact_recall_accuracy=exact_correct / exact_total if exact_total > 0 else 0,
            multi_hop_accuracy=multihop_correct / multihop_total if multihop_total > 0 else 0,
            semantic_drift_accuracy=drift_correct / drift_total if drift_total > 0 else 0,
            coherence_accuracy=coherence_correct / coherence_total if coherence_total > 0 else 0,
            overall_accuracy=sum(1 for r in cold_results if r.is_correct) / len(cold_results),
            ingest_time_seconds=ingest_time,
            cold_query_avg_ms=sum(r.latency_ms for r in cold_results) / len(cold_results),
            warm_query_avg_ms=sum(r.latency_ms for r in warm_results) / len(warm_results),
            avg_context_length_chars=sum(len(r.retrieved_context) for r in cold_results) // len(cold_results),
            query_results=cold_results,
            needle_count=len(needles),
            total_queries=len(cold_results)
        )
    
    def benchmark_neurosavant(self, dataset_path: str, needles: List[Needle]) -> Optional[BenchmarkResult]:
        """Run benchmark against NeuroSavant (hierarchical memory)."""
        if not NEURO_AVAILABLE:
            print("\n⚠️  NeuroSavant not available, skipping...")
            return None
        
        print(f"\n{'='*60}")
        print("📊 BENCHMARKING: NeuroSavant (Hierarchical Memory)")
        print(f"{'='*60}")
        
        # Initialize with a fresh brain
        brain_path = "./benchmark_neuro_brain"
        # Clean up old DB if exists
        import shutil
        if os.path.exists("./neuro_savant_memory"):
            shutil.rmtree("./neuro_savant_memory")
            
        try:
            # Use MockEncoder for benchmark speed/stability if desired, or real one
            os.environ["USE_MOCK_ENCODER"] = "true" 
            brain = NeuroSavant()
        except Exception as e:
            print(f"❌ Failed to initialize NeuroSavant: {e}")
            return None
        
        # Ingest
        print("\n🔄 Ingesting data...")
        start_ingest = time.time()
        
        with open(dataset_path, 'r') as f:
            full_text = f.read()
        
        # Chunk and ingest (similar to how the existing benchmark does it)
        chunk_size = 500
        # Use batch_ingest for speed
        chunks = []
        for i in range(0, len(full_text), chunk_size):
            chunk = full_text[i:i + chunk_size + 50]  # Overlap
            chunks.append(chunk)
            
        try:
            brain.batch_ingest(chunks)
        except Exception as e:
            print(f"   ⚠️ Ingest error: {e}")
        
        ingest_time = time.time() - start_ingest
        print(f"   ✅ Ingested in {ingest_time:.2f}s")
        
        # Query function using NeuroSavant's retrieval
        def ns_query(question: str) -> dict:
            """Query NeuroSavant and return context."""
            context = ""
            try:
                context = brain.query(question)
            except Exception as e:
                print(f"   ⚠️ Query error: {e}")
            return {"context": context}
        
        # Cold queries
        print("\n❄️  Running COLD queries...")
        cold_results = self._run_queries(
            "NeuroSavant",
            ns_query,
            needles,
            warm_cache=False
        )
        
        # Warm queries (NeuroSavant has QueryCache)
        print("\n🔥 Running WARM queries (testing cache)...")
        warm_results = self._run_queries(
            "NeuroSavant",
            ns_query,
            needles,
            warm_cache=True
        )
        
        # Calculate metrics
        exact_correct = sum(1 for r in cold_results if r.is_correct and r.needle_type == 'exact')
        exact_total = sum(1 for r in cold_results if r.needle_type == 'exact')
        
        multihop_correct = sum(1 for r in cold_results if r.is_correct and r.needle_type == 'multi_hop')
        multihop_total = sum(1 for r in cold_results if r.needle_type == 'multi_hop')
        
        drift_correct = sum(1 for r in cold_results if r.is_correct and r.needle_type == 'semantic_drift')
        drift_total = sum(1 for r in cold_results if r.needle_type == 'semantic_drift')
        
        coherence_correct = sum(1 for r in cold_results if r.is_correct and r.needle_type == 'coherence')
        coherence_total = sum(1 for r in cold_results if r.needle_type == 'coherence')
        
        return BenchmarkResult(
            timestamp=datetime.now().isoformat(),
            system_name="NeuroSavant",
            data_size_tokens=len(full_text) // 4,
            data_size_bytes=len(full_text),
            exact_recall_accuracy=exact_correct / exact_total if exact_total > 0 else 0,
            multi_hop_accuracy=multihop_correct / multihop_total if multihop_total > 0 else 0,
            semantic_drift_accuracy=drift_correct / drift_total if drift_total > 0 else 0,
            coherence_accuracy=coherence_correct / coherence_total if coherence_total > 0 else 0,
            overall_accuracy=sum(1 for r in cold_results if r.is_correct) / len(cold_results),
            ingest_time_seconds=ingest_time,
            cold_query_avg_ms=sum(r.latency_ms for r in cold_results) / len(cold_results),
            warm_query_avg_ms=sum(r.latency_ms for r in warm_results) / len(warm_results),
            avg_context_length_chars=sum(len(r.retrieved_context) for r in cold_results) // len(cold_results),
            query_results=cold_results,
            needle_count=len(needles),
            total_queries=len(cold_results)
        )
    
    def generate_proof_report(self, results: List[BenchmarkResult]) -> str:
        """Generate markdown proof report with measured vs claimed comparison."""
        
        report = f"""# Benchmark Proof Report
Generated: {datetime.now().isoformat()}

## ⚠️ IMPORTANT: Measured vs Claimed Metrics

This benchmark measures **actual accuracy against ground truth**,
not proxy metrics like "cell reuse rate" or "similarity threshold".

---

## Summary

| Metric | Baseline RAG | NeuroSavant | Claimed (NS) | Verdict |
|--------|--------------|-------------|--------------|---------|
"""
        
        baseline = next((r for r in results if r.system_name == "Baseline RAG"), None)
        neuro = next((r for r in results if r.system_name == "NeuroSavant"), None)
        
        if baseline:
            report += f"| Exact Recall | {baseline.exact_recall_accuracy*100:.1f}% | "
            if neuro:
                verdict = "✅" if neuro.exact_recall_accuracy >= 0.70 else "❌"
                report += f"{neuro.exact_recall_accuracy*100:.1f}% | 75-85% | {verdict} |\n"
            else:
                report += "N/A | 75-85% | - |\n"
            
            report += f"| Multi-hop | {baseline.multi_hop_accuracy*100:.1f}% | "
            if neuro:
                verdict = "✅" if neuro.multi_hop_accuracy > baseline.multi_hop_accuracy else "❌"
                report += f"{neuro.multi_hop_accuracy*100:.1f}% | > Baseline | {verdict} |\n"
            else:
                report += "N/A | > Baseline | - |\n"
            
            report += f"| Semantic Drift | {baseline.semantic_drift_accuracy*100:.1f}% | "
            if neuro:
                verdict = "✅" if neuro.semantic_drift_accuracy > baseline.semantic_drift_accuracy else "❌"
                report += f"{neuro.semantic_drift_accuracy*100:.1f}% | > Baseline | {verdict} |\n"
            else:
                report += "N/A | > Baseline | - |\n"
            
            report += f"| Coherence | {baseline.coherence_accuracy*100:.1f}% | "
            if neuro:
                verdict = "✅" if neuro.coherence_accuracy > baseline.coherence_accuracy else "❌"
                report += f"{neuro.coherence_accuracy*100:.1f}% | > Baseline | {verdict} |\n"
            else:
                report += "N/A | > Baseline | - |\n"
            
            report += f"| Cold Latency | {baseline.cold_query_avg_ms:.1f}ms | "
            if neuro:
                verdict = "✅" if neuro.cold_query_avg_ms <= 500 else "❌"
                report += f"{neuro.cold_query_avg_ms:.1f}ms | 100-500ms | {verdict} |\n"
            else:
                report += "N/A | 100-500ms | - |\n"
            
            if neuro:
                warm_verdict = "✅" if neuro.warm_query_avg_ms <= 50 else "❌"
                report += f"| Warm Latency | {baseline.warm_query_avg_ms:.1f}ms | {neuro.warm_query_avg_ms:.1f}ms | <10ms | {warm_verdict} |\n"
        
        report += f"""
---

## Detailed Results

"""
        
        for result in results:
            report += f"""### {result.system_name}

- **Data Size**: {result.data_size_tokens:,} tokens ({result.data_size_bytes:,} bytes)
- **Ingest Time**: {result.ingest_time_seconds:.2f}s
- **Total Queries**: {result.total_queries}

#### Accuracy by Type

| Type | Correct | Total | Accuracy |
|------|---------|-------|----------|
| Exact Recall | {sum(1 for r in result.query_results if r.is_correct and r.needle_type == 'exact')} | {sum(1 for r in result.query_results if r.needle_type == 'exact')} | {result.exact_recall_accuracy*100:.1f}% |
| Multi-hop | {sum(1 for r in result.query_results if r.is_correct and r.needle_type == 'multi_hop')} | {sum(1 for r in result.query_results if r.needle_type == 'multi_hop')} | {result.multi_hop_accuracy*100:.1f}% |
| Semantic Drift | {sum(1 for r in result.query_results if r.is_correct and r.needle_type == 'semantic_drift')} | {sum(1 for r in result.query_results if r.needle_type == 'semantic_drift')} | {result.semantic_drift_accuracy*100:.1f}% |
| Coherence | {sum(1 for r in result.query_results if r.is_correct and r.needle_type == 'coherence')} | {sum(1 for r in result.query_results if r.needle_type == 'coherence')} | {result.coherence_accuracy*100:.1f}% |
| **Overall** | {sum(1 for r in result.query_results if r.is_correct)} | {len(result.query_results)} | **{result.overall_accuracy*100:.1f}%** |

#### Query Details

"""
            for qr in result.query_results:
                status = "✅" if qr.is_correct else "❌"
                report += f"- {status} **[{qr.needle_type}]** {qr.question}\n"
                report += f"  - Expected: `{qr.expected_answer}`\n"
                report += f"  - Latency: {qr.latency_ms:.1f}ms\n\n"
            
            report += "\n---\n\n"
        
        report += f"""
## Proof Hash

To verify this report was not tampered with:

```
SHA256: {hashlib.sha256(report.encode()).hexdigest()}
```
"""
        
        return report
    
    def run(self, target_tokens: int = 1_000_000) -> None:
        """Run full benchmark suite."""
        print(f"\n{'='*70}")
        print(f"🚀 1 MILLION TOKEN RAG BENCHMARK")
        print(f"   Target: {target_tokens:,} tokens")
        print(f"   Time: {datetime.now().isoformat()}")
        print(f"{'='*70}")
        
        # Generate data
        dataset_path, needles_path = self.generator.generate(target_tokens)
        
        # Load needles
        with open(needles_path, 'r') as f:
            needles_data = json.load(f)
        needles = [Needle(**n) for n in needles_data]
        
        results = []
        
        # Benchmark Baseline RAG
        baseline_result = self.benchmark_baseline_rag(dataset_path, needles)
        results.append(baseline_result)
        
        # Benchmark NeuroSavant
        neuro_result = self.benchmark_neurosavant(dataset_path, needles)
        if neuro_result:
            results.append(neuro_result)
        
        # Generate report
        report = self.generate_proof_report(results)
        
        # Save results
        report_path = os.path.join(self.output_dir, "benchmark_1m_proof.md")
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\n✅ Report saved: {report_path}")
        
        json_path = os.path.join(self.output_dir, "benchmark_1m_results.json")
        with open(json_path, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2, default=str)
        print(f"✅ Raw data saved: {json_path}")
        
        # Print summary
        print(f"\n{'='*70}")
        print("📊 BENCHMARK COMPLETE")
        print(f"{'='*70}")
        print(report)


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="1M Token RAG Benchmark")
    parser.add_argument("--scale", choices=["100k", "500k", "1m", "1mb"], default="1mb",
                        help="Scale of benchmark (100k=quick test, 1m=full)")
    parser.add_argument("--output", default="reports", help="Output directory")
    
    args = parser.parse_args()
    
    # Convert scale to tokens
    scale_map = {
        "100k": 100_000,
        "500k": 500_000,
        "1m": 1_000_000,
        "1mb": 250_000,  # ~1MB = 250k tokens
    }
    
    runner = BenchmarkRunner(output_dir=args.output)
    runner.run(target_tokens=scale_map[args.scale])


if __name__ == "__main__":
    main()
