# MagneticStorage Project Structure

## Root Files
- `neuro_savant.py` - Main application (agentic vector memory system)
- `README.md` - Project documentation
- `requirements.txt` - Python dependencies
- `.neuro_savant_config` - Configuration file

## Directories

### `/core/`
Core modules and utilities
- `performance_tracker.py` - Performance metrics and visualization
- `chat_agentic.py` - Agentic chat logic and tool orchestration

### `/tools/`
Agentic tools for NeuroSavant
- `agent_behavior.py` - AI persona management
- `example.py` - Template/example system
- `infinite.py` - Continuous generation
- `github_ingest.py` - Repository ingestion
- `storyline_agent.py` - Story generation
- `story_registry.py` - Story consistency tracking
- `__init__.py` - Tool loading framework

### `/benchmarks/`
Benchmark scripts and comparisons
- `baseline_rag.py` - Standard RAG baseline
- `benchmark_1m_rag.py` - 1M scale RAG benchmark
- `benchmark_data_gen.py` - Data generation for benchmarks
- `benchmark_full.py` - comprehensive system benchmark
- `benchmark_recall.py` - Recall accuracy benchmark
- `benchmark_standalone.py` - Standalone benchmark runner

### `/scripts/`
Utility and test scripts
- `debug_v2.py` - Debugging utilities
- `mock_setup.py` - Mock testing setup
- `query_1m_cellular.py` - Large-scale query testing
- `verify_1m_cellular.py` - Cellular memory verification
- `verify_sleep_mode.py` - Sleep mode verification

### `/data/`
Data files and logs
- `dream_log.jsonl` - Dream/consolidation logs
- `memory_tree.txt` - Memory hierarchy visualization
- `benchmark_data_1m/` - Large-scale test datasets

### `/tests/`
Unit and integration tests
- `stress_test.py`
- `story_consistency_test.py`

### `/reports/`
Benchmark and analysis reports

### `/old_architecture/`
Legacy implementation for reference

## Generated Directories
- `neuro_savant_memory/` - ChromaDB vector database (created at runtime)
- `__pycache__/` - Python bytecode cache
