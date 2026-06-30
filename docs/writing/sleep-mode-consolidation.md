---
contentKind: article
slug: "sleep-mode-agentic-consolidation"
title: "Sleep Mode: Agentic Memory Consolidation"
type: technical-note
status: published
date: 2026-06-26
summary: "Designing a consolidation pass that merges similar concept groups using both vector similarity and LLM-based contradiction analysis."
tags:
  - Agentic Memory
  - Consolidation
  - LLM
  - ChromaDB
---

Memory consolidation in biological systems happens during sleep — the brain replays, strengthens, and merges related patterns from the day. MagneticStorage's "Sleep Mode" is a software analogue: a consolidation pass that merges overlapping concept groups and resolves factual inconsistencies.

## The Merge Problem

As data streams in via online clustering, related groups can drift close to each other. Two groups about "Python async patterns" and "Python asyncio internals" might have centroids with cosine similarity above 0.85 — effectively the same concept but stored as separate groups. Over time, this fragments the hierarchy and increases the number of centroid lookups per query.

## Greedy Vectorized Merge

Sleep Mode iterates over all group centroids and computes pairwise cosine similarities. Any pair exceeding `merge_threshold` (default 0.85) is merged:

1. **Compute the merged centroid** as the mean of all cell vectors in both groups.
2. **Reassign all cells** from the secondary group to the primary group.
3. **Update the in-memory registry** and the persistent `ns_cells` collection with the new group ID.
4. **Drop the stale per-group collection** — it will be rebuilt on the next session start.

The comparison is O(G²) on centroids, not O(N²) on raw cells, which keeps it tractable for thousands of groups.

## Agentic Contradiction Analysis

After the vector merge, an LLM agent (`_run_conflict_analysis_agent`) reviews the concatenated cell contents of the newly merged group for factual contradictions. For example, if one cell states "the API key is sent as a header" and another says "the API key is sent as a query parameter", the agent flags the conflict.

This is implemented as a structured prompt that lists all cell texts in the merged group and asks the LLM to identify any contradictory statements. The results are surfaced to the user rather than auto-resolved, since automated fact resolution risks data loss.

## Orchestration

Sleep Mode runs on demand via the `/sleep` CLI command. A full pass:

1. Loads all group centroids from ChromaDB
2. Runs the greedy merge loop until no pairs exceed the threshold
3. Updates ChromaDB collections and the in-memory cell registry
4. Runs the conflict analysis agent on each newly merged group
5. Reports merges performed and any contradictions found

The CLI output shows which groups were merged and whether any conflicts were detected, giving the user visibility into the consolidation process.

## Why Not Just Cosine Similarity?

Pure vector consolidation captures semantic overlap but cannot detect factual contradictions — two cells can be semantically similar (both about API authentication) yet factually contradictory (different protocols). The agentic step adds a reasoning layer that vector math alone cannot provide. This hybrid approach — numerical merge followed by LLM verification — is the pattern I would extend for more sophisticated memory maintenance.
