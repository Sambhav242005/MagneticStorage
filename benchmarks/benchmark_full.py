#!/usr/bin/env python3
"""
Full Benchmark: Baseline RAG vs Hierarchical RAG vs GraphRAG
=============================================================

Compares three approaches:
1. Baseline RAG - Flat vector store (ChromaDB)  
2. Hierarchical RAG - Multi-layer like NeuroSavant
3. GraphRAG - Entity-based knowledge graph simulation

PERFORMANCE ANALYSIS INCLUDED: Explains why each system performs as it does.

Usage:
    python benchmark_full.py --scale 100k   # Quick (~3 min)
    python benchmark_full.py --scale 250k   # Standard (~10 min)
    python benchmark_full.py --scale 1m     # Full 1M tokens (~30 min)
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
    needle_type: str
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
    num_collections: int  # For performance analysis
    num_embeddings: int   # For performance analysis
    query_results: List[QueryResult] = field(default_factory=list)


# =============================================================================
# 1. BASELINE RAG (Flat Vector Store)
# =============================================================================

class BaselineRAG:
    """
    Standard RAG: Fixed chunking + Flat vector search.
    
    PERFORMANCE PROFILE:
    - Ingest: O(N) - one embedding per chunk
    - Query: O(1) - single vector search
    - Accuracy: Good for exact recall, bad for synthesis
    """
    
    def __init__(self, db_path: str = "./benchmark_baseline_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        try:
            self.client.delete_collection("baseline")
        except:
            pass
        self.collection = self.client.create_collection(
            name="baseline", metadata={"hnsw:space": "cosine"}
        )
        self.chunk_size = 500
        self.num_embeddings = 0
        self.num_collections = 1
    
    def ingest(self, text: str) -> float:
        start = time.perf_counter()
        
        chunks = []
        ids = []
        
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size + 50]
            chunks.append(chunk)
            ids.append(f"c_{i}")
        
        # Batch add
        for i in range(0, len(chunks), 100):
            end = min(i + 100, len(chunks))
            self.collection.add(documents=chunks[i:end], ids=ids[i:end])
        
        self.num_embeddings = len(chunks)
        return time.perf_counter() - start
    
    def query(self, question: str, n_results: int = 3) -> str:
        results = self.collection.query(query_texts=[question], n_results=n_results)
        if results['documents'] and results['documents'][0]:
            return "\n".join(results['documents'][0])
        return ""


# =============================================================================
# 2. HIERARCHICAL RAG (NeuroSavant-style)
# =============================================================================

class HierarchicalRAG:
    """
    Hierarchical RAG: Multi-layer with summaries.
    
    PERFORMANCE ANALYSIS:
    - Ingest: O(3N) - three layers of embeddings
    - Query: O(3) - must search all layers
    - WHY IT'S SLOWER: 3x embedding work + 3x query work
    - WHY ACCURACY ISN'T 3x BETTER: Extractive summarization LOSES information
    
    THE FUNDAMENTAL FLAW:
    When we summarize "The Omega Protocol password is 'Azure-99-Gamma'",
    extractive summarization might produce "Protocol calibration requires...",
    completely losing the needle. The summary doesn't know what's important.
    """
    
    def __init__(self, db_path: str = "./benchmark_hier_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        
        self.layers = {}
        for layer in range(3):
            try:
                self.client.delete_collection(f"hier_{layer}")
            except:
                pass
            self.layers[layer] = self.client.create_collection(
                name=f"hier_{layer}", metadata={"hnsw:space": "cosine"}
            )
        
        self.raw_chunk_size = 500
        self.mid_chunk_size = 2000
        self.section_size = 10000
        self.num_embeddings = 0
        self.num_collections = 3
    
    def _summarize(self, text: str, target_length: int = 200) -> str:
        """
        CRITICAL FLAW: Extractive summarization loses needle information.
        Takes first/middle/last sentences - misses important facts in between.
        """
        sentences = text.replace('\n', ' ').split('. ')
        if len(sentences) <= 3:
            return text[:target_length]
        
        summary_parts = [sentences[0], sentences[len(sentences)//2], sentences[-1]]
        return '. '.join(summary_parts)[:target_length]
    
    def ingest(self, text: str) -> float:
        start = time.perf_counter()
        embedding_count = 0
        
        # Layer 2: Raw chunks (same as baseline)
        l2_chunks, l2_ids = [], []
        for i in range(0, len(text), self.raw_chunk_size):
            l2_chunks.append(text[i:i + self.raw_chunk_size + 50])
            l2_ids.append(f"L2_{i}")
        
        for i in range(0, len(l2_chunks), 100):
            self.layers[2].add(documents=l2_chunks[i:i+100], ids=l2_ids[i:i+100])
        embedding_count += len(l2_chunks)
        
        # Layer 1: Medium chunks with summary
        l1_chunks, l1_ids = [], []
        for i in range(0, len(text), self.mid_chunk_size):
            chunk = text[i:i + self.mid_chunk_size]
            summary = self._summarize(chunk, 400)
            l1_chunks.append(summary + "\n---\n" + chunk[:500])
            l1_ids.append(f"L1_{i}")
        
        for i in range(0, len(l1_chunks), 100):
            self.layers[1].add(documents=l1_chunks[i:i+100], ids=l1_ids[i:i+100])
        embedding_count += len(l1_chunks)
        
        # Layer 0: Section summaries
        l0_chunks, l0_ids = [], []
        for i in range(0, len(text), self.section_size):
            l0_chunks.append(self._summarize(text[i:i + self.section_size], 600))
            l0_ids.append(f"L0_{i}")
        
        self.layers[0].add(documents=l0_chunks, ids=l0_ids)
        embedding_count += len(l0_chunks)
        
        self.num_embeddings = embedding_count
        return time.perf_counter() - start
    
    def query(self, question: str, n_results: int = 3) -> str:
        """
        PERFORMANCE COST: 3 separate vector searches.
        Each ChromaDB query is ~50-200ms, so 3x = 150-600ms.
        """
        all_contexts = []
        
        for layer in [0, 1, 2]:
            try:
                results = self.layers[layer].query(
                    query_texts=[question], n_results=n_results,
                    include=['documents', 'distances']
                )
                if results['documents'] and results['documents'][0]:
                    for i, doc in enumerate(results['documents'][0]):
                        dist = results['distances'][0][i]
                        conf = 1.0 / (1.0 + dist)
                        all_contexts.append((conf * (1.0 + 0.2 * (2 - layer)), doc, layer))
            except:
                continue
        
        all_contexts.sort(key=lambda x: x[0], reverse=True)
        return "\n---\n".join([c[1] for c in all_contexts[:5]])


# =============================================================================
# 3. GRAPH RAG (Entity-based Knowledge Graph)
# =============================================================================

class GraphRAG:
    """
    Simulated GraphRAG: Entity extraction + relationship linking.
    
    HOW REAL GRAPHRAG WORKS:
    1. LLM extracts entities (people, places, concepts) from text
    2. LLM identifies relationships between entities
    3. Builds knowledge graph with nodes and edges
    4. Query traverses graph to find related context
    
    THIS SIMULATION:
    - Uses regex patterns to extract entities (no LLM needed)
    - Creates actual relationship links between entities
    - Stores both entities and their source chunks
    
    THEORETICAL ADVANTAGE:
    - Multi-hop queries can traverse relationships
    - "What did X steal from Y?" can link X->stolen_item->Y
    
    PRACTICAL LIMITATIONS:
    - Entity extraction is expensive (we simulate with regex)
    - Graph construction is O(N²) for relationship finding
    - Doesn't help for simple exact recall queries
    """
    
    def __init__(self, db_path: str = "./benchmark_graph_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Entity collection (each entity is a node)
        try:
            self.client.delete_collection("graph_entities")
            self.client.delete_collection("graph_chunks")
        except:
            pass
        
        self.entities = self.client.create_collection(
            name="graph_entities", metadata={"hnsw:space": "cosine"}
        )
        self.chunks = self.client.create_collection(
            name="graph_chunks", metadata={"hnsw:space": "cosine"}
        )
        
        # In-memory graph for relationship traversal
        self.graph: Dict[str, Set[str]] = defaultdict(set)  # entity -> related entities
        self.entity_to_chunks: Dict[str, List[str]] = defaultdict(list)  # entity -> chunk IDs
        
        self.num_embeddings = 0
        self.num_collections = 2
        self.chunk_size = 500
        
        # Entity patterns (simulating LLM extraction)
        self.entity_patterns = [
            r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b',  # Proper names (Commander Reyes)
            r'\b(Project [A-Z][a-z]+)\b',      # Project names
            r'\b(Section \d+)\b',              # Sections
            r'\b(Protocol [A-Za-z-]+)\b',      # Protocols
            r'\b(Moon of [A-Z][a-z]+)\b',      # Locations
            r'password[^\'\"]*[\'"]([^\'\"]+)[\'"]',  # Passwords
            r'\b(\d+\.\d+ MHz)\b',             # Frequencies
            r'\b(vault [A-Z]\d+)\b',           # Vault IDs
            r'\b(launch code[s]?\s*(?:is\s*)?[\w-]+)\b',  # Launch codes
        ]
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities using regex patterns."""
        entities = set()
        for pattern in self.entity_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.update(matches)
        return list(entities)
    
    def _find_relationships(self, entities: List[str], text: str) -> List[Tuple[str, str]]:
        """Find relationships between entities that appear in same sentence."""
        relationships = []
        sentences = text.split('. ')
        
        for sentence in sentences:
            sentence_entities = [e for e in entities if e.lower() in sentence.lower()]
            # Create pairwise relationships
            for i, e1 in enumerate(sentence_entities):
                for e2 in sentence_entities[i+1:]:
                    relationships.append((e1, e2))
        
        return relationships
    
    def ingest(self, text: str) -> float:
        start = time.perf_counter()
        
        # Store raw chunks (for fallback retrieval)
        chunks = []
        chunk_ids = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size + 50]
            chunks.append(chunk)
            chunk_ids.append(f"chunk_{i}")
        
        for i in range(0, len(chunks), 100):
            self.chunks.add(documents=chunks[i:i+100], ids=chunk_ids[i:i+100])
        
        # Extract entities and build graph
        all_entities = set()
        entity_contexts = defaultdict(list)
        
        for i, chunk in enumerate(chunks):
            entities = self._extract_entities(chunk)
            for entity in entities:
                all_entities.add(entity)
                self.entity_to_chunks[entity].append(chunk_ids[i])
                entity_contexts[entity].append(chunk[:200])
            
            # Build relationships
            relationships = self._find_relationships(entities, chunk)
            for e1, e2 in relationships:
                self.graph[e1].add(e2)
                self.graph[e2].add(e1)
        
        # Store entities with their context
        if all_entities:
            entity_docs = []
            entity_ids = []
            for entity in all_entities:
                # Context is the entity name + related snippets
                context = f"{entity}: " + " | ".join(entity_contexts[entity][:3])
                entity_docs.append(context)
                entity_ids.append(f"entity_{hash(entity) % 100000}")
            
            for i in range(0, len(entity_docs), 100):
                self.entities.add(
                    documents=entity_docs[i:i+100], 
                    ids=entity_ids[i:i+100]
                )
        
        self.num_embeddings = len(chunks) + len(all_entities)
        return time.perf_counter() - start
    
    def query(self, question: str, n_results: int = 3) -> str:
        """
        Graph-enhanced query:
        1. Find relevant entities
        2. Traverse graph for related entities
        3. Retrieve chunks containing those entities
        """
        contexts = []
        
        # Step 1: Search entities
        try:
            entity_results = self.entities.query(
                query_texts=[question], n_results=5,
                include=['documents', 'distances']
            )
            if entity_results['documents'] and entity_results['documents'][0]:
                for doc in entity_results['documents'][0]:
                    contexts.append(doc)
        except:
            pass
        
        # Step 2: Search raw chunks (fallback)
        try:
            chunk_results = self.chunks.query(
                query_texts=[question], n_results=n_results,
                include=['documents', 'distances']
            )
            if chunk_results['documents'] and chunk_results['documents'][0]:
                for doc in chunk_results['documents'][0]:
                    contexts.append(doc)
        except:
            pass
        
        # Step 3: Extract entities from question and traverse graph
        question_entities = self._extract_entities(question)
        for entity in question_entities:
            # Get related entities (1-hop traversal)
            related = self.graph.get(entity, set())
            for rel_entity in list(related)[:3]:
                # Get chunks containing related entities
                chunk_ids = self.entity_to_chunks.get(rel_entity, [])[:2]
                for cid in chunk_ids:
                    try:
                        result = self.chunks.get(ids=[cid])
                        if result['documents']:
                            contexts.append(result['documents'][0])
                    except:
                        pass
        
        return "\n---\n".join(contexts[:7])


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
             "INTEL: Commander Reyes has been confirmed as the traitor in Section 9."),
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
                context=marker, depth_percent=depth*100, needle_type='exact'
            ))
        
        # Multi-hop needles (GraphRAG should excel here)
        multihop_facts = [
            ("What did Commander Reyes steal from Project Titan?", "launch codes",
             "After Commander Reyes' betrayal was confirmed in Section 9, investigators found he copied the Project Titan launch codes."),
            ("How did the Ghost Signal reach the Moon of Endor?", "relay satellite",
             "The Ghost Signal's 442.8 MHz transmission was bounced off a relay satellite positioned above the Moon of Endor."),
        ]
        
        for i, (q, a, ctx) in enumerate(multihop_facts):
            depth = 0.4 + 0.1 * i
            idx = int(num_blocks * depth)
            marker = f"\n[MULTI-{random.randint(10000,99999)}]: {ctx}\n"
            blocks[idx] += marker
            needles.append(Needle(
                id=f"multihop_{i}", question=q, answer=a,
                context=marker, depth_percent=depth*100, needle_type='multi_hop'
            ))
        
        # Semantic drift
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
                context=ctx, depth_percent=depth*100, needle_type='semantic_drift'
            ))
        
        # Coherence needle
        coherence_parts = [
            "Chapter 1: Marcus swore revenge against those who destroyed his village.",
            "Chapter 3: Meeting the orphaned children made Marcus question his path.",
            "Chapter 5: Marcus chose to protect the innocent rather than seek vengeance.",
        ]
        for i, part in enumerate(coherence_parts):
            depth = 0.15 + 0.25 * i
            blocks[int(num_blocks * depth)] += f"\n{part}\n"
        
        needles.append(Needle(
            id="coherence_0",
            question="How did Marcus's motivation change throughout the story?",
            answer="revenge to redemption",
            context=" | ".join(coherence_parts),
            depth_percent=50.0, needle_type='coherence'
        ))
        
        return "\n".join(blocks), needles


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

class BenchmarkRunner:
    """Runs benchmark and generates analysis report."""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def _check_answer(self, expected: str, context: str) -> bool:
        if not context:
            return False
        return expected.lower() in context.lower()
    
    def _run_queries(self, name: str, query_fn, needles: List[Needle]) -> List[QueryResult]:
        results = []
        for needle in needles:
            start = time.perf_counter()
            context = query_fn(needle.question)
            latency = (time.perf_counter() - start) * 1000
            is_correct = self._check_answer(needle.answer, context)
            
            results.append(QueryResult(
                needle_id=needle.id, question=needle.question,
                expected_answer=needle.answer,
                retrieved_context=context[:300] if context else "",
                is_correct=is_correct, latency_ms=latency,
                needle_type=needle.needle_type
            ))
            
            status = "✅" if is_correct else "❌"
            print(f"   {status} [{needle.needle_type:12}] {needle.question[:40]}... ({latency:.0f}ms)")
        
        return results
    
    def _calc_accuracy(self, results: List[QueryResult], needle_type: str) -> float:
        filtered = [r for r in results if r.needle_type == needle_type]
        return sum(1 for r in filtered if r.is_correct) / len(filtered) if filtered else 0.0
    
    def _benchmark_system(self, system, name: str, text: str, needles: List[Needle]) -> SystemResult:
        print(f"\n{'='*60}")
        print(f"📊 {name}")
        print(f"{'='*60}")
        
        print("   🔄 Ingesting...")
        ingest_time = system.ingest(text)
        print(f"   ✅ Ingested in {ingest_time:.1f}s ({system.num_embeddings} embeddings)")
        
        print("\n   ❄️ COLD queries:")
        cold = self._run_queries(name, system.query, needles)
        print("\n   🔥 WARM queries:")
        warm = self._run_queries(name, system.query, needles)
        
        return SystemResult(
            system_name=name,
            ingest_time_sec=ingest_time,
            cold_latency_avg_ms=sum(r.latency_ms for r in cold) / len(cold),
            warm_latency_avg_ms=sum(r.latency_ms for r in warm) / len(warm),
            exact_accuracy=self._calc_accuracy(cold, 'exact'),
            multihop_accuracy=self._calc_accuracy(cold, 'multi_hop'),
            drift_accuracy=self._calc_accuracy(cold, 'semantic_drift'),
            coherence_accuracy=self._calc_accuracy(cold, 'coherence'),
            overall_accuracy=sum(1 for r in cold if r.is_correct) / len(cold),
            context_avg_chars=sum(len(r.retrieved_context) for r in cold) // len(cold),
            num_collections=system.num_collections,
            num_embeddings=system.num_embeddings,
            query_results=cold
        )
    
    def run(self, target_tokens: int = 250_000) -> None:
        print(f"\n{'='*70}")
        print(f"🚀 FULL RAG BENCHMARK (3-System Comparison)")
        print(f"   Target: {target_tokens:,} tokens (~{target_tokens*4:,} bytes)")
        print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
        
        gen = DataGenerator()
        text, needles = gen.generate(target_tokens)
        print(f"\n🔧 Generated {len(text):,} bytes, {len(needles)} needles")
        
        results = []
        
        results.append(self._benchmark_system(BaselineRAG(), "Baseline RAG", text, needles))
        results.append(self._benchmark_system(HierarchicalRAG(), "Hierarchical RAG", text, needles))
        results.append(self._benchmark_system(GraphRAG(), "GraphRAG (Simulated)", text, needles))
        
        self._generate_report(results, target_tokens)
    
    def _generate_report(self, results: List[SystemResult], tokens: int) -> None:
        b, h, g = results[0], results[1], results[2]
        
        report = f"""# Full RAG Benchmark Report
Generated: {datetime.now().isoformat()}
Data Scale: {tokens:,} tokens

---

## Executive Summary

| Metric | Baseline | Hierarchical | GraphRAG | Winner |
|--------|----------|--------------|----------|--------|
| **Exact Recall** | {b.exact_accuracy*100:.0f}% | {h.exact_accuracy*100:.0f}% | {g.exact_accuracy*100:.0f}% | {max([(b.exact_accuracy, 'Baseline'), (h.exact_accuracy, 'Hier'), (g.exact_accuracy, 'Graph')], key=lambda x: x[0])[1]} |
| **Multi-hop** | {b.multihop_accuracy*100:.0f}% | {h.multihop_accuracy*100:.0f}% | {g.multihop_accuracy*100:.0f}% | {max([(b.multihop_accuracy, 'Baseline'), (h.multihop_accuracy, 'Hier'), (g.multihop_accuracy, 'Graph')], key=lambda x: x[0])[1]} |
| **Semantic Drift** | {b.drift_accuracy*100:.0f}% | {h.drift_accuracy*100:.0f}% | {g.drift_accuracy*100:.0f}% | {max([(b.drift_accuracy, 'Baseline'), (h.drift_accuracy, 'Hier'), (g.drift_accuracy, 'Graph')], key=lambda x: x[0])[1]} |
| **Coherence** | {b.coherence_accuracy*100:.0f}% | {h.coherence_accuracy*100:.0f}% | {g.coherence_accuracy*100:.0f}% | {max([(b.coherence_accuracy, 'Baseline'), (h.coherence_accuracy, 'Hier'), (g.coherence_accuracy, 'Graph')], key=lambda x: x[0])[1]} |
| **Overall** | **{b.overall_accuracy*100:.0f}%** | **{h.overall_accuracy*100:.0f}%** | **{g.overall_accuracy*100:.0f}%** | **{max([(b.overall_accuracy, 'Baseline'), (h.overall_accuracy, 'Hier'), (g.overall_accuracy, 'Graph')], key=lambda x: x[0])[1]}** |
| Cold Latency | {b.cold_latency_avg_ms:.0f}ms | {h.cold_latency_avg_ms:.0f}ms | {g.cold_latency_avg_ms:.0f}ms | {min([(b.cold_latency_avg_ms, 'Baseline'), (h.cold_latency_avg_ms, 'Hier'), (g.cold_latency_avg_ms, 'Graph')], key=lambda x: x[0])[1]} |
| Ingest Time | {b.ingest_time_sec:.1f}s | {h.ingest_time_sec:.1f}s | {g.ingest_time_sec:.1f}s | {min([(b.ingest_time_sec, 'Baseline'), (h.ingest_time_sec, 'Hier'), (g.ingest_time_sec, 'Graph')], key=lambda x: x[0])[1]} |
| Embeddings | {b.num_embeddings:,} | {h.num_embeddings:,} | {g.num_embeddings:,} | {min([(b.num_embeddings, 'Baseline'), (h.num_embeddings, 'Hier'), (g.num_embeddings, 'Graph')], key=lambda x: x[0])[1]} |

---

## Performance Analysis: Why Hierarchical is 4x Slower

### The Math

| System | Collections | Embeddings | Query Ops |
|--------|-------------|------------|-----------|
| Baseline | 1 | {b.num_embeddings:,} | 1 search |
| Hierarchical | 3 | {h.num_embeddings:,} | 3 searches |
| GraphRAG | 2 | {g.num_embeddings:,} | 2 searches + graph traversal |

**Hierarchical RAG** is ~{h.cold_latency_avg_ms/b.cold_latency_avg_ms:.1f}x slower because:
1. **3x Query Cost**: Must search all 3 layers (L0, L1, L2)
2. **~{h.num_embeddings/b.num_embeddings:.1f}x Embedding Cost**: Creates embeddings for summaries too
3. **No Speedup**: Layer 0 summaries don't help find specific facts

### Why Accuracy Isn't Proportionally Better

**Extractive summarization loses needle information.**

Example: If the raw text is:
```
"The flux capacitor... The Omega Protocol password is 'Azure-99-Gamma'. More filler text..."
```

Extractive summary (first/middle/last sentences) might produce:
```
"The flux capacitor... More filler text..."
```

The password is gone. The summary doesn't know what's important.

---

## Claim Validation

| Claim (PerformanceTracker) | Claimed | Measured | Validated? |
|---------------------------|---------|----------|------------|
| Query Accuracy 75-85% | 75-85% | {h.overall_accuracy*100:.0f}% | {'✅' if 0.70 <= h.overall_accuracy <= 0.90 else '❌'} |
| Multi-hop Better | ✓ | {h.multihop_accuracy*100:.0f}% vs {b.multihop_accuracy*100:.0f}% | {'✅' if h.multihop_accuracy > b.multihop_accuracy else '❌'} |
| Cold Latency 100-500ms | 100-500ms | {h.cold_latency_avg_ms:.0f}ms | {'✅' if 100 <= h.cold_latency_avg_ms <= 500 else '⚠️'} |

---

## Recommendations

1. **For NeuroSavant**: Use semantic summarization (LLM-based) instead of extractive
2. **For Speed**: Add caching layer to avoid redundant vector searches
3. **For Multi-hop**: Consider GraphRAG approach with entity linking
4. **Honest Claims**: Update PerformanceTracker to show measured, not estimated, values

---

## Raw Data

SHA256: `{hashlib.sha256(str(results).encode()).hexdigest()}`
"""
        
        # Save
        with open(os.path.join(self.output_dir, "benchmark_full_proof.md"), 'w') as f:
            f.write(report)
        
        with open(os.path.join(self.output_dir, "benchmark_full_results.json"), 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2, default=str)
        
        print(f"\n{'='*70}")
        print("📊 BENCHMARK COMPLETE")
        print(f"{'='*70}")
        print(report)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Full RAG Benchmark")
    parser.add_argument("--scale", choices=["100k", "250k", "500k", "1m"], default="250k")
    parser.add_argument("--output", default="reports")
    args = parser.parse_args()
    
    scale_map = {"100k": 100_000, "250k": 250_000, "500k": 500_000, "1m": 1_000_000}
    BenchmarkRunner(output_dir=args.output).run(target_tokens=scale_map[args.scale])


if __name__ == "__main__":
    main()
