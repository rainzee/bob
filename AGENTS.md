# Repository Guidelines

## Project Structure & Module Organization

Core code lives under `src/`, and there is no `builtin/` layer: the library ships no tools, no provider and no skills.

- `src/bub/framework.py`: the composition root: paths, resource slots and `running()`.
- `src/bub/hooks.py`: the interception contracts, the `Hooks` value, and its per-slot execution semantics.
- `src/bub/agent.py`: the turn loop over hooks, tools and the tape.
- `src/bub/model_runner.py`: the `ChatClient`/`ChatRequest` boundary and the parsing of streamed chunks.
- `src/bub/tape.py` / `src/bub/store.py` / `src/bub/context.py`: append-only records, the `TapeStore` protocol with its in-memory and JSONL implementations, and the entry-to-message replay.
- `src/bub/tools.py`: the tool framework and executor.
- `src/bub/{streaming,errors,sidecars,turn,utils}.py`: small kernel vocabulary owned by each concern.

Everything is instance-scoped: no environment variables, no process-wide registries, no implicit user or working directories, no configuration file. The framework takes `workspace` and `home`; the agent takes its model, its `ChatClient`, its tools and its hooks as plain keyword arguments. Settings only exist as the host's own dict, and the provider is the host's own SDK.

Because the library emits no telemetry and configures no logging, the only observability surface is the `Hooks` slots and the tape itself.

Tests live in `tests/`.

## Build, Test, and Development Commands

- `uv sync`: sync the Python dependencies.
- `uv run ruff check .`: run Ruff directly when you only want lint feedback.
- `uv run ruff format .`: format the tree.
- `uv run mypy src`: run mypy directly against `src/`.
- `uv run pytest -q`: run the main test suite without doctests.
- `uv run python -m pytest --doctest-modules`: run pytest with doctests enabled.
- `uv lock --locked` then Ruff, formatting, and typing: validate the lockfile and the tree.

## Coding Style & Naming Conventions

- Python 3.13+, 4-space indentation, and type hints for new or modified logic. Use the 3.13 standard library and `typing`; there are no compatibility shims or version guards in this tree.
- Use `snake_case` for modules/functions/variables, `PascalCase` for classes, and `UPPER_CASE` for constants.
- Keep functions focused and composable; avoid hidden side effects.
- Format and lint with Ruff. Keep line length within 120 unless an existing file clearly follows a different local convention.

## Testing Guidelines
- Framework: `pytest`.
- Name test files `tests/test_<feature>.py`.
- Prefer behavior-oriented test names such as `test_gateway_uses_enabled_channels_only`.
- Cover hook precedence, turn lifecycle, and tape persistence when changing runtime behavior.
- Update or add tests in the same change when behavior moves.

## Commit & Pull Request Guidelines

- Follow the Conventional Commit style used in history, for example `feat:`, `fix:`, `docs:`, `chore:`.
- Keep commits focused; avoid mixing unrelated refactors with behavior changes.
- For PRs, include:
  - what changed and why
  - impacted modules or commands
  - verification performed (`ruff`, `mypy`, `pytest`, docs build if relevant)
  - docs updates when CLI behavior, commands, or architecture changed

## Security & Configuration Tips

- Never commit credentials. Construct your own client with them and pass it to `Agent`.
- Bub reads no environment variables and holds no credentials; a provider SDK you bring may still read its own (for example `OPENROUTER_API_KEY`).
