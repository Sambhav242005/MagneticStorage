# Neuro-Savant

**AI with Hierarchical Memory** - A conversational AI system with tree-structured persistent memory, enabling long-term context retention and intelligent story generation.

## Project Structure

For a detailed breakdown of the project structure, see [STRUCTURE.md](./STRUCTURE.md).

## Tools & Behaviors
NeuroSavant features a modular tool system. The following tools are currently available:

- **AgentBehavior** (`/behavior`): sophisticated persona and style management for the agent.
- **StorylineAgent** (`/story`): Interactive story generation with consistency tracking.
- **InfiniteLoop** (`/infinite`): Continuous, autonomous content generation mode.
- **GitHubIngest** (`/ingest`): Ingests and indexes entire GitHub repositories.
- **Example** (`/example`): Loads template conversations or data.

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
| `/example list/load` | Manage templates |
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
