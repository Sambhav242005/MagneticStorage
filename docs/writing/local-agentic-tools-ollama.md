---
contentKind: article
slug: "local-agentic-tools-with-ollama"
title: "Local Agentic Tools with Ollama Function Calling"
type: technical-note
status: published
date: 2026-06-22
summary: "How to wire local LLM function calling into a modular tool framework, and what I learned building six tools for NeuroSavant."
tags:
  - Ollama
  - Agentic
  - LLM
  - Tool Design
---

Ollama supports tool/function calling for models that expose a tool-calling API. NeuroSavant wraps this into a framework where tools are auto-discovered, registered with the LLM, and invoked through a clean base class.

## BaseTool Design

Every tool lives in `tools/` and subclasses `BaseTool`:

```python
class BaseTool(ABC):
    name: str       # unique identifier
    command: str    # CLI command (e.g., "/story")
    description: str  # LLM-facing description for function selection

    @abstractmethod
    def run(self, **kwargs) -> str: ...
```

The loader (`tools/__init__.py`) scans the `tools` package for subclasses and populates a registry. The LLM receives the tool names and descriptions as available functions, and the agent loop routes calls back to the matching `run` method.

## The Tool Suite

Six tools are currently registered:

| Tool | Name | Purpose |
|---|---|---|
| AgentBehavior | behavior | Manage persona/style (storyteller, critic, coder, custom) |
| StorylineAgent | story | Interactive story generation with world configuration |
| InfiniteLoop | infinite | Continuous autonomous generation with rolling context |
| GitHubIngest | ingest | Clone and index GitHub repos into memory |
| StoryRegistry | registry | Track and validate story consistency |
| Example | example | Load generation templates (hero journey, noir, technical) |

## Function Calling Loop

The agentic chat loop works as follows:

1. The user sends a message (or the LLM decides to continue)
2. The system prompt includes tool descriptions serialised into Ollama's tool format
3. The LLM responds either with text or a function call request
4. If a function call is requested, NeuroSavant runs the matching tool and appends the result to the message history
5. The LLM receives the tool output and continues

This loop lets the LLM autonomously decide when to search memory, add new information, or invoke specialised tools like story generation — without the user explicitly switching modes.

## Key Lessons

**Tool descriptions matter more than tool implementations.** The LLM selects tools based on their description strings. Spending time on clear, specific descriptions (e.g., "Generate an interactive story with world configuration, physics rules, and character tracking" instead of "Generate a story") dramatically improves correct tool selection.

**Keep tool outputs short.** Ollama context windows fill quickly when tool outputs are verbose. Each tool's `run` method should return a concise summary, not raw data. For story generation, the tool returns a plot summary and character list, not the full generated text.

**Handle tool failures gracefully.** Tools can fail (network errors, missing dependencies, invalid user input). The agent loop catches exceptions and returns an error message to the LLM, which can then decide whether to retry, ask the user for clarification, or proceed without the tool output. This prevents a single tool failure from crashing the entire session.

## Limitations

Ollama's tool-calling API is less mature than OpenAI's. Tool call parsing can produce malformed JSON, especially with smaller models (sub-7B parameters). I found that using the `deepseek-r1:1.5b` model for agentic tool selection required careful prompt engineering and occasional retry logic. Larger models like `llama3.2:3b` perform noticeably better at correct tool selection and output format adherence.
