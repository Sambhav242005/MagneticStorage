# Neuro-Savant: Cellular Memory Architecture

**Beyond Standard RAG** — A biological-inspired, agentic memory system that clusters information into dynamic "Groups" (Concepts) rather than a flat list.

Unlike traditional RAG which retrieves independent chunks, NeuroSavant builds a **Hierarchical Graph** of memories. It uses an active **Cortex (Agent)** to manage memory consolidation (Sleep Mode), infinite generation, and consistency.

## Architecture: The "Cellular" Approach
The system mimics biological memory organization:
1.  **Layer 0: Groups (Concepts)** — High-level clusters of related information.
2.  **Layer 1: Cells (Details)** — Specific text/vector units belonging to a Group.

When you search, the system first identifies the relevant **Concept (Group)** and then retrieves specific **Details (Cells)** nearby. This preserves *context* that flat RAG loses.

## NeuroSavant vs. Standard RAG

| Feature | Standard RAG | NeuroSavant (Cellular) |
| :--- | :--- | :--- |
| **Structure** | Flat Index (List of vectors) | Hierarchical (Groups → Cells) |
| **Retrieval** | `O(log N)` (1 HNSW query) | `O(log G + k·log(N/G))` (centroid search + per-group queries) |
| **Entity Routing** | ❌ None | ✅ Entity index identifies groups by keywords |
| **Context** | Fragmented (Chunks are isolated) | Clustered (Chunks are grouped by Concept) |
| **Recall** | Top-k only | ✅ Entity-guaranteed + full fallback |
| **Updates** | Append-only (usually) | Dynamic (Merge/Split/Consolidate) |
| **Agentic?** | Passive (Query → Result) | Active (Can search, store, and "sleep" to optimize) |
| **Consolidation** | ❌ None | ✅ Sleep Mode + Agentic conflict analysis |

## Key Features

### Hierarchical Search
Query processing has three stages:
1. **Centroid search** — HNSW query on G group centroids → picks top-k groups
2. **Parallel per-group queries** — Queries each group's dedicated ChromaDB collection concurrently via `ThreadPoolExecutor` (wall-clock = `t_{N/G}`, not `k * t_{N/G}`)
3. **Entity-routed re-ranking** — Entity index runs in parallel with per-group queries; results from entity-matched groups get boosted scores

### Entity Routing
At query time, the entity index (a separate ChromaDB collection) is queried for each extracted entity. Found group IDs are merged into the target set, and their per-group collections are searched. This guarantees recall for specific named concepts even when semantic similarity is low.

### Guaranteed Recall Fallback
If parallel per-group queries return fewer than `cell_top_k` results, a full cells collection query (identical to flat RAG) is triggered as a safety net. The system never silently misses results.

### Dual Persistence
- **Main cells collection** (`ns_cells`): survives restarts. ChromaDB 1.5.5 Rust backend.
- **Per-group collections** (`ns_grp_{id}`): created each session for fast N/G-sized HNSW queries. Dropped and re-created on restart (ChromaDB persist bug workaround). Written at ingest and consolidation alongside the main collection.

### Sleep Mode (Consolidation)
Merges similar groups (cosine similarity > `merge_threshold`) to keep the concept hierarchy compact:
- **Greedy vectorized merge** — O(G²) worst case, avoids O(N²) memory by working on centroids
- **Per-group cleanup** — stale collections are dropped, merged cells are re-indexed under the target group

### Agentic Consolidation
An LLM (`_run_conflict_analysis_agent`) analyzes merged cell contents for factual contradictions. Used alongside Sleep Mode for deeper reasoning.

### Auto-Clustering at Ingest
Each new cell is compared against all group centroids. If the closest centroid is within `similarity_threshold`, the cell joins that group. Otherwise, a new group is created. Batch ingest uses vectorized numpy operations for efficiency.

## Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Ingest (single) | O(G) | Prototype: linear scan over centroids |
| Ingest (batch) | O(N·G) | Vectorized with numpy |
| Query | O(log G + log(N/G)) | Centroid HNSW + parallel per-group HNSW |
| Consolidation | O(G²) | Greedy merge, optimized from O(N²) |

## Benchmark Results

## Benchmark Results

### Complexity Benchmark
`python benchmarks/benchmark_complexity.py`:

| N      | G   | N/G  | Flat RAG | Cellular | vs Flat |
|--------|-----|------|----------|----------|---------|
| 25000  | 50  | 500  | 1.27ms   | 1.69ms   | 0.75x   |
| 50000  | 100 | 500  | 2.55ms   | 1.65ms   | **1.55x** |
| 75000  | 150 | 500  | 3.12ms   | 1.69ms   | **1.85x** |
| 100000 | 200 | 500  | 1.51ms   | 1.77ms   | 0.85x   |

*Cellular* = t_G (centroid search on G groups) + t_{N/G} (parallel per-group queries, wall-clock max of k concurrent threads). Cellular maintains stable latency (~1.7ms) regardless of N because per-group queries search N/G-sized subsets. Flat RAG latency grows with N and has higher variance.

### Recall Benchmark
`python benchmarks/benchmark_complexity.py --recall` — Needle-in-Haystack test:
- **Recall: 100%** (5/5 needles found in top-3 results)
- 200-doc haystack, 5 unique fact needles at 20%/40%/60%/80%/90% positions
- With real embeddings (nomic-embed-text), recall is expected to be 100%

## Why ChromaDB?
We use **ChromaDB** not just as a vector store, but as a **Persistent Substrate** for our graph.
- **HNSW Indices**: Provides the `O(log N)` underlying search speed.
- **Per-Group Collections**: Each group gets its own ChromaDB collection (isolated HNSW index),
  avoiding expensive metadata filtering on a single giant collection.
- **Local & Fast**: Runs entirely on-device (no API latency), essential for an Agentic "Brain" that thinks constantly.

## Project Structure

For a detailed breakdown of the project structure, see [STRUCTURE.md](./STRUCTURE.md).

## Tools & Behaviors
NeuroSavant features a modular tool system. The following tools are currently available:

- **AgentBehavior** (`/behavior`): sophisticated persona and style management for the agent.
- **StorylineAgent** (`/story`): Interactive story generation with consistency tracking.
- **InfiniteLoop** (`/infinite`): Continuous, autonomous content generation mode.
- **GitHubIngest** (`/ingest`): Ingests and indexes entire GitHub repositories.
- **Example** (`/example`): Loads template conversations or data.

## Setup & Installation
### 1. Prerequisites
- **Python 3.10+**
### 1. Prerequisites
- **Python 3.10+**
- **Ollama Application**: Required to run the models. [Download here](https://ollama.com).
    - Run: `ollama pull deepseek-r1:1.5b` and `ollama pull nomic-embed-text`.
- **Ollama Python Library**: Installed via `requirements.txt` (used by some tools).

### 2. Installation
```bash
# 1. Clone the repository
git clone https://github.com/Sambhav242005/MagneticStorage.git
cd MagneticStorage

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the environment
# For Linux/macOS:
source venv/bin/activate
# For Windows:
# venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt
```

## Quick Start
```bash
# Run the agent (ensure Ollama is running)
python neuro_savant.py

# Or with a specific model
python neuro_savant.py --model deepseek-r1:1.5b
```

## Commands

| Command | Description |
|---------|-------------|
| `/status` | Show memory statistics & visualization |
| `/visualize` | Visual memory representation |
| `/model <name>` | Switch LLM model |
| `/embed <name>` | Switch embedding model (⚠️ wipes memory) |
| `/story <topic>` | Generate a world/story |
| `/infinite <cmd>` | Infinite mode (on/off) |
| `/behavior <cmd>` | Set AI persona (list/set) |
| `/ingest <url>` | Ingest a GitHub repo |
| `/example <cmd>` | Load template (list/load) |
| `/perf` | Show performance metrics |
| `/clean` | Wipe memory |
| `/quit` | Exit |

## Running Tests

```bash
# All story consistency tests
python tests/story_consistency_test.py

# Full stress test
python tests/stress_test.py
```

## Features

- **Hierarchical Memory**: Tree-structured storage with compression
- **Infinite Generation**: Rolling context window for long-form content
- **Story Consistency Tracking**: Regex-based fact extraction to detect contradictions
- **Tool Framework**: Modular, auto-discovered extensions
- **Performance Tracking**: Metrics and GraphRAG comparison
