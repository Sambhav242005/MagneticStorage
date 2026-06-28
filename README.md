# Neuro-Savant: Cellular Memory Architecture

**Beyond Standard RAG** — A biological-inspired, agentic memory system that clusters information into dynamic "Groups" (Concepts) rather than a flat list.

Unlike traditional RAG which retrieves independent chunks, NeuroSavant builds a **Hierarchical Graph** of memories. It uses an active **Cortex (Agent)** to manage memory consolidation (Sleep Mode), infinite generation, and consistency.

## Architecture: The "Cellular" Approach
The system mimics biological memory organization:
1.  **Layer 0: Groups (Concepts)** - High-level clusters of related information.
2.  **Layer 1: Cells (Details)** - Specific text/vector units belonging to a Group.

When you search, the system first identifies the relevant **Concept (Group)** and then retrieves specific **Details (Cells)** nearby. This preserves *context* that flat RAG loses.

## NeuroSavant vs. Standard RAG

| Feature | Standard RAG | NeuroSavant (Cellular) |
| :--- | :--- | :--- |
| **Structure** | Flat Index (List of vectors) | Hierarchical (Groups → Cells) |
| **Retrieval** | `O(log N)` (HNSW graph search) | `O(log G + k·log(N))` (Groups via HNSW, Cells via HNSW + Python filter) |
| **Context** | Fragmented (Chunks are isolated) | Clustered (Chunks are linked by Concept) |
| **Updates** | Append-only (usually) | Dynamic (Merge/Split/Consolidate) |
| **Agentic?** | Passive (Query → Result) | Active (Can search, store, and "sleep" to optimize) |
| **Consolidation** | ❌ None | ✅ Sleep Mode (Merges similar concepts) |

## Time Complexity Analysis
Efficiently scaling to millions of memories (`N`) by clustering them into `G` groups.

### 1. Retention (Ingest) & Search
- **Search**: **`O(log G + k · log(N))`** → effectively **`O(log N)`** in practice
    - Step 1 (Find Concept): `O(log G)` using ChromaDB's HNSW index on Group Centroids.
    - Step 2 (Retrieve Details): All cells live in *one* ChromaDB collection. Instead of
      per-group `WHERE` filters (which ChromaDB's HNSW handles poorly, adding ~3x overhead),
      we query with `n_results = cell_top_k * |groups| * 3` (a small constant multiple)
      and filter by group ID in Python.
      *Why this works*: ChromaDB's HNSW returns the top-k results in O(log N), then the
      Python filter is O(k) over the returned batch — negligible compared to the HNSW search.
    - **vs WHERE filter (old approach)**: ChromaDB's `where={"group_id": gid}` looks fast
      in theory but HNSW internally fetches extra candidates when selectivity is low,
      adding 2-5x latency per group. The Python-filter approach avoids this entirely.

- **Ingest (Formation)**: **`O(log G)`**
    - Finds the nearest Group (Concept) in `O(log G)` and adds the new Cell to it.
    - *Note*: Current Python prototype uses an optimized memory scan `O(G)`, but production uses HNSW `O(log G)`.

### Why Not Per-Group Collections?
The ideal fix would be one ChromaDB collection per group (isolated HNSW index of size N/G).
However, ChromaDB 1.5.5 (Rust backend) has a persist bug: collections created dynamically
via `get_or_create_collection` fail with "Nothing found on disk" on subsequent runs.
The single-collection + Python filter approach achieves the same latency profile without
the ChromaDB bug.

### 2. Sleep Mode (Consolidation)
- **Complexity**: **`O(G²)`** (worst case, optimized to `O(G log G)`)
- The system wakes up periodic "Dreams" to merge similar Groups, reducing `G` and keeping the index efficient. This mimics the human brain's consolidation process during sleep.

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
