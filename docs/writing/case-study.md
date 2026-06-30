---
contentKind: case-study
---

## Problem

Flat retrieval systems can return isolated chunks without preserving the larger concept they belong to, which weakens long-running agent memory and story consistency.

## Approach

I built MagneticStorage as a local agentic memory prototype that stores high-level groups and lower-level cells, then retrieves through the concept layer before selecting detailed memories.

## Technical Decisions

- Hierarchical memory structure with groups for concepts and cells for detailed text/vector units.
- ChromaDB acts as the persistent vector substrate for group centroids and metadata-filtered retrieval.
- Sleep mode consolidates related groups through merge and optimization passes.
- Tool commands support memory inspection, visualization, persona behavior, story generation, GitHub ingestion, and performance checks.
- Ollama models provide local LLM and embedding workflows for the prototype.

## Result

The project adds an agent-memory signal to the portfolio: retrieval architecture, local persistence, consolidation behavior, and tool-oriented AI system design.
