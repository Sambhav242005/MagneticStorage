---
contentKind: article
slug: hierarchical-memory-chromadb
title: Hierarchical Memory Retrieval with ChromaDB
type: technical-note
status: published
date: 2026-06-25
summary: How to structure persistent agent memory using concept groups and detailed cell retrieval in ChromaDB collections.
tags:
  - ChromaDB
  - Vector Search
  - Agentic Memory
---

Flat vector search (standard RAG) is not enough for persistent agent memory. It retrieves independent chunks of text without preserving the overarching concept, leading to lost context.

To solve this, I designed a hierarchical cellular memory system using ChromaDB, modeling relationship contexts that flat indices drop.

## The Cellular Approach

Instead of a single flat index, the memory architecture splits data into two distinct layers:
1. **Layer 0: Groups (Concepts)** - High-level semantic centroids representing conceptual topics.
2. **Layer 1: Cells (Details)** - Precise text and vector segments associated with a parent Group.

## Centroid-driven Query Processing

When a user or agent queries the database, the search proceeds in three distinct stages:

* **Centroid Search**: Querying the group centroids collection (`ns_cells`) to identify the top-k relevant Concepts.
* **Parallel Per-Group Queries**: Concurrently querying the HNSW index of the selected Groups via Python's `ThreadPoolExecutor` (reducing retrieval time to `t_{N/G}` instead of `k * t_{N/G}`).
* **Entity Routing**: Running a parallel query on a dedicated entity index. If named entities match a Group, that Group's score is boosted, guaranteeing recall for specific concepts even when semantic similarity is low.

## Workarounds for ChromaDB Collections

During development, dynamically creating and deleting dozens of collections led to descriptor locks. To workaround this:
- Stale dynamically created group collections (`ns_grp_{id}`) are cached in-memory and re-created on session restarts.
- The master cells collection (`ns_cells`) acts as the single source of truth, persisting coordinates and raw content across runs.

## Sleep Mode & Agentic Consolidation

To keep the concept hierarchy compact, a background thread runs a consolidation pass:
* **Vector Merge**: A greedy merge runs on centroids with cosine similarity > 0.85, grouping related nodes.
* **Agentic Analysis**: An LLM agent (`_run_conflict_analysis_agent`) inspects the merged cells to resolve factual contradictions and merge semantic groups.
