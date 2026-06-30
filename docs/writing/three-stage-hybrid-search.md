---
contentKind: article
slug: "three-stage-hybrid-search"
title: "Three-Stage Hybrid Search with Entity Routing"
type: technical-note
status: published
date: 2026-06-28
summary: "How to combine centroid search, parallel per-group HNSW queries, and entity routing into a recall-safe hybrid search pipeline."
tags:
  - Vector Search
  - ChromaDB
  - Retrieval
  - Architecture
---

Standard vector search queries a single HNSW index and returns the nearest neighbours. For agent memory that needs both semantic breadth and named-entity precision, a single pass is not enough.

## The Three Paths

MagneticStorage runs three concurrent search paths on every query:

### 1. Centroid search

Group centroids live in a dedicated ChromaDB collection (`ns_groups`). An HNSW query returns the top-k groups whose centroid is nearest to the query vector. This identifies which conceptual buckets are relevant before looking at any individual cell.

### 2. Parallel per-group HNSW queries

Each group has its own collection (`ns_grp_{id}`) containing only that group's cell vectors. Once the centroid search picks the top-k groups, Python's `ThreadPoolExecutor` queries all of them in parallel. Since each collection has roughly N/G vectors, the wall-clock time stays close to `t_{N/G}` — the time to search a single small index — rather than `k * t_{N/G}`.

Benchmarks confirm this scales: at 100K cells spread across 200 groups (~500 cells/group), query latency holds at ~1.7ms, while flat RAG grows linearly.

### 3. Entity routing

A third parallel query hits a dedicated entity index (`ns_entity_index`). At ingest time, entities (capitalised phrases, filenames, file paths, URLs, emails) are extracted via regex and stored as separate documents tied to their parent group IDs. At query time, entities present in the query text are matched against this index. Groups containing matching entities receive a score boost of 1.0, guaranteeing they appear in the final result set even if semantic similarity is low.

This is the safety net that catches queries like "find the notes about the AsyncProcessor refactor" — the entity "AsyncProcessor" routes directly to the right group regardless of embedding quality.

### Fallback: flat full scan

If the three parallel paths collectively return fewer results than `cell_top_k`, a full flat query against the master `ns_cells` collection fires as a safety net. This ensures the system never silently returns fewer results than requested.

## Why Three?

Each path covers a weakness of the others:

| Path | Strength | Weakness |
|---|---|---|
| Centroid search | Fast concept identification | Misses cells whose group centroid is off-centre |
| Per-group HNSW | Precise detail retrieval within the right groups | Requires the right groups to be selected first |
| Entity routing | Guaranteed recall for named concepts | Only works when the query contains extractable entities |

Running them concurrently and fusing the results gives both the breadth of semantic search and the precision of entity lookup, with a flat fallback as insurance.

## Implementation Notes

- The entity extractor uses `re.findall` with patterns for filenames (with extensions), capitalised phrases (2+ words), URLs, emails, and file paths.
- Entity documents in ChromaDB store the group ID in metadata, so entity matches can be mapped back to groups for score boosting.
- Per-group collections are rebuilt on each session start (see the dual persistence article for why), but the entity index is persistent.
