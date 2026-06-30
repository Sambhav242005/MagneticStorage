---
contentKind: case-study
title: "MagneticStorage — Biologically-Inspired Agentic Memory"
slug: "magnetic-storage"
summary: "A local, biologically-inspired agentic memory system that organizes information into hierarchical concept groups and cells, going beyond flat RAG with three-stage hybrid search, entity routing, and LLM-driven consolidation."
status: published
order: 1
featured: true
updatedAt: 2026-06-30
tags:
  - Agentic Memory
  - ChromaDB
  - Vector Search
  - Ollama
  - Python
---

## Problem

Flat retrieval-augmented generation (RAG) systems treat all information as independent chunks in a single index. This loses the conceptual context that groups related facts together — a retrieved snippet about "mitochondria" is returned without connecting it to the broader "cellular biology" concept it belongs to. For long-running agents, this erodes coherence and weakens recall for multi-turn tasks like story generation, research synthesis, or personal memory.

Existing vector databases support filtering and metadata, but no common agent framework provides *online hierarchical clustering* — the ability to incrementally build and maintain a two-level concept taxonomy as data arrives, then search through the concept layer before retrieving details.

## Approach

I built MagneticStorage as a local-first agentic memory engine in Python. It stores information in two layers — **Groups** (conceptual centroids) and **Cells** (individual text/vector units nested under groups) — and searches through the concept layer before selecting detailed memories, mirroring how biological memory organizes information.

The system runs entirely locally via Ollama for both LLM inference and embeddings, with ChromaDB as the persistent vector substrate. It exposes a CLI with agentic function calling so the LLM can autonomously search and store memories during a session.

## Technical Decisions

**Hierarchical memory structure.** A flat index lets you find similar chunks, but it cannot tell you which conceptual bucket a chunk belongs to. By layering Groups (centroid vectors) over Cells (detail vectors), queries first identify relevant concepts, then retrieve details only from those concept groups. This preserves the topic context that flat search drops.

**Online clustering at ingest.** Each new cell is compared against all group centroids via cosine similarity. If the closest centroid is within a configurable threshold, the cell joins that group; otherwise, a new group is created. Batch operations use vectorized NumPy for performance. This means the hierarchy builds itself incrementally — no separate clustering pass required.

**Three-stage hybrid search.** Queries run through three concurrent paths: (1) HNSW centroid search to pick top-k groups, (2) parallel per-group HNSW queries on N/G-sized subsets (keeping latency stable as N grows), and (3) a dedicated entity index that boosts scores for groups containing named entities from the query. If all three paths return insufficient results, a full flat fallback guarantees recall.

**Entity routing at ingest and query.** Entities (filenames, paths, capitalized phrases, emails, URLs) are extracted via regex at ingest time and indexed in a dedicated ChromaDB collection. At query time, entities in the query text identify relevant groups directly — ensuring recall for specific named concepts even when semantic similarity is low.

**Dual persistence strategy.** A master cells collection (`ns_cells`) persists across restarts as the source of truth. Per-group collections (`ns_grp_{id}`) are ephemeral — created each session and dropped on restart — working around a ChromaDB bug where dynamically created collections do not persist their HNSW indices reliably.

**Sleep mode consolidation.** A background consolidation pass merges groups whose centroids exceed a cosine similarity threshold (default 0.85). The merge uses greedy vectorized comparison (O(G²) on centroids, avoiding O(N²) on raw cells). After merging, an LLM agent analyzes the combined cell contents for factual contradictions, adding an agentic reasoning layer to what would normally be a purely numerical operation.

**Modular tool framework.** Extensions live in `tools/` as `BaseTool` subclasses with stable `name` and `command` attributes. Current tools include agent behavior/persona management, interactive story generation with world consistency tracking, infinite continuous generation, and GitHub repository ingestion.

**Ollama-first, with fallbacks.** The system prefers Ollama embeddings (`nomic-embed-text`, 768-dim) for quality, falls back to `sentence-transformers` (`all-MiniLM-L6-v2`) when Ollama is unavailable, and provides a `MockEncoder` for offline testing.

## Result

MagneticStorage demonstrates that biologically-inspired hierarchical memory is practical with local tooling. Key outcomes:

- **Stable query latency (~1.7ms)** regardless of total cell count, because per-group queries operate on N/G-sized subsets rather than the full index. Flat RAG latency grows linearly with N in comparison.
- **100% recall** in needle-in-haystack tests (5 specific needles found in top-3 results from a 200-doc haystack), driven by the entity-routing safety net.
- **Online clustering** that builds and maintains a concept hierarchy incrementally, without offline batch processing.
- **11 integration and unit tests** covering clustering, consolidation, search correctness, infinite mode, and story consistency.
- **Extensive benchmarks** confirming the cellular architecture's latency advantage at 25K–100K cell scales.

The project adds an agent-memory signal to the portfolio — retrieval architecture, local persistence, consolidation behavior, and tool-oriented AI system design — and is available as an open-source prototype for anyone building long-running local agents.
