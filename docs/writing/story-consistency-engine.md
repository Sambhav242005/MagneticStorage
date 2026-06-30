---
contentKind: article
slug: "story-consistency-tracking"
title: "Building a Story Consistency Engine with Regex and Vector Search"
type: technical-note
status: published
date: 2026-06-24
summary: "Using regex-based fact extraction and vector-backed cross-referencing to detect contradictions in LLM-generated stories."
tags:
  - Agentic Memory
  - Story Generation
  - LLM
  - Python
---

Generating long-form fiction with an LLM is difficult because the model has no persistent memory of what it already established. A character's eyes might be "blue" in chapter one and "green" in chapter five, or a world rule ("magic requires a focus object") might be contradicted later. MagneticStorage's StorylineAgent solves this with a dedicated consistency registry.

## The Consistency Registry

`StoryConsistencyRegistry` tracks five categories of facts extracted from generated story chunks:

| Category | Examples |
|---|---|
| Characters | name, eyes, hair, traits |
| World rules | magic systems, physics constraints |
| Events | plot events, major occurrences |
| Locations | setting names, geography |
| Timeline | temporal order, elapsed time |

Each fact is stored with its source text and a confidence flag. When a new story chunk is generated, the registry extracts facts from it and cross-references against existing facts.

## Regex Extraction Patterns

Facts are extracted using targeted regex patterns applied to each generated chunk:

- **Character introductions**: patterns like `([A-Z][a-z]+) had ([a-z]+) (eyes|hair)` capture physical traits
- **Trait statements**: `([A-Z][a-z]+) was (a|an|the) ([^,.]+)` captures personality or role descriptions
- **World rules**: `(magic|the world|the system) (requires|uses|runs on|operates)` captures world-building constraints
- **Location mentions**: `(in|at|beneath|inside) (the|a) ([A-Z][a-z]+)` captures setting contexts

## Contradiction Detection

When a new fact is extracted, it is compared against existing facts in the same category for the same entity. Contradictions are detected through direct value comparison:

- "blue eyes" vs "green eyes" → direct string mismatch on the same attribute
- "heroic" vs "cowardly" → trait contradiction (explicit antonym check)
- Timeline conflicts → ordering violations ("the tower collapsed" then later "they entered the tower")

Contradictions are logged and surfaced to the user, who can choose to accept the new fact (overwriting the old) or reject it and regenerate.

## Integration with Vector Memory

Beyond the regex registry, the StorylineAgent also queries MagneticStorage's main memory engine to pull relevant context from earlier story chunks. This means the agent has two memory systems working in parallel:

1. **Consistency registry** — deterministic, rule-based fact tracking with explicit contradiction detection
2. **Vector memory** — fuzzy retrieval of semantically similar story passages from the full history

The combination gives both the precision of regex for known fact types and the breadth of vector search for unexpected connections.

## Results

In practice, the consistency engine catches approximately 80% of direct factual contradictions during story generation. The remaining 20% are subtler (e.g., implied contradictions or statements whose meaning depends on context) and require human review. The system performs best on physical character traits and explicit world rules — the categories with the most predictable regex patterns.
