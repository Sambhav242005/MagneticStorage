# Repository Guidelines

## Project Structure & Module Organization
`neuro_savant.py` is the main entrypoint and contains the Chroma-backed memory engine, CLI, and runtime wiring. `core/` holds shared runtime helpers such as agentic chat and performance tracking. `tools/` contains optional extensions; new tools should live here and subclass `tools.BaseTool` with a stable `name` and `command`. `tests/` contains the `unittest` suite. `benchmarks/` and `scripts/` are for performance runs and local verification helpers. Treat `old_architecture/` as reference-only, and avoid editing `benchmark_data_1m/` unless you are explicitly updating datasets.

## Build, Test, and Development Commands
- `pip install -r requirements.txt`: install runtime dependencies.
- `python neuro_savant.py`: start the local CLI agent.
- `python -m unittest discover -s tests -p "*.py" -v`: run the full test suite.
- `python -m py_compile neuro_savant.py tools\\__init__.py tools\\agent_behavior.py tools\\example.py tools\\github_ingest.py tools\\infinite.py tools\\story_registry.py tools\\storyline_agent.py`: quick syntax smoke check for core modules.
- `python scripts\\verify_sleep_mode.py` or `python scripts\\verify_1m_cellular.py`: targeted verification scripts for memory behavior.

## Coding Style & Naming Conventions
Use 4-space indentation and keep Python style close to PEP 8. Prefer `snake_case` for functions, variables, and module names; use `CamelCase` for classes and dataclasses. Keep comments short and only where behavior is not obvious. New tool modules should expose one clear class and avoid side effects at import time. No formatter or linter is checked in, so keep changes consistent with nearby code and avoid broad reformatting.

## Testing Guidelines
Write tests with `unittest` under `tests/` and name files `test_*.py` or `*_test.py`. Favor deterministic tests with mocks over live Ollama or network calls. If you touch story generation, tool wiring, or memory clustering, add or update a focused regression test. Before opening a PR, run full discovery and the relevant script-level verification for the area you changed.

## Commit & Pull Request Guidelines
Current history uses short, informal subjects such as `update md files` and `adding project`. Keep commit messages brief, imperative, and scoped, for example `fix batch ingest dedupe` or `update story workflow tests`. PRs should include a short summary, the reason for the change, test commands run, and any behavior changes to CLI commands or memory layout. Include sample output only when it clarifies a user-visible change.

## Security & Configuration Tips
The project expects local services such as Ollama and writes state under the configured Chroma database path. Do not commit local virtualenvs, generated memory stores, or secrets. Keep external URLs and repo ingestion inputs explicit and validated.
