# Neuro-Savant

**AI with Hierarchical Memory** - A conversational AI system with tree-structured persistent memory, enabling long-term context retention and intelligent story generation.

## Project Structure

```
MagneticStorage/
├── neuro_savant.py          # Main application (NeuroSavant agent + memory grid)
├── mock_setup.py            # Mock utilities for testing
│
├── tools/                   # Modular tool extensions
│   ├── __init__.py          # Tool framework (BaseTool, auto-discovery)
│   ├── agent_behavior.py    # Persona/style modifier (/behavior)
│   ├── example.py           # Template loader (/example)
│   ├── github_ingest.py     # GitHub repo ingestion (/ingest)
│   ├── infinite.py          # Infinite generation mode (/infinite)
│   ├── story_registry.py    # Story consistency tracking (regex-based)
│   └── storyline_agent.py   # Agentic story generator (/story)
│
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── story_consistency_test.py  # Story consistency stress tests
│   ├── stress_test.py       # Full workflow stress test
│   ├── test_adaptive.py     # Adaptive memory tests
│   └── test_clustering.py   # Clustering tests
│
├── reports/                 # Documentation & analysis
│   ├── architecture_comparison.md
│   ├── benchmark_debugging_journey.md
│   └── project_critique.md
│
├── baseline_rag.py          # Baseline RAG comparison
├── benchmark_data_gen.py    # Benchmark data generator
└── benchmark_recall.py      # Recall benchmark
```

## Quick Start

```bash
# Activate environment
source venv/bin/activate

# Run the agent
python neuro_savant.py

# Or with a specific model
python neuro_savant.py --model qwen2.5:3b
```

## Commands

| Command | Description |
|---------|-------------|
| `/status` | Show memory statistics |
| `/tree` | Visualize memory tree |
| `/story <topic>` | Generate a world/story |
| `/infinite on` | Enable infinite generation mode |
| `/behavior set <persona>` | Set agent persona |
| `/ingest <github-url>` | Ingest a GitHub repo |
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
