# Repository Guidelines

## Project Structure & Module Organization

Core code lives under `src/`:

- `src/bub/framework.py`: turn orchestration, hook loading, and resource lifecycle.
- `src/bub/hooks/`: hook specifications, execution helpers, and interception contracts.
- `src/bub/configure.py`: `Settings`, the `@config()` section registry, and the per-instance `Config`.
- `src/bub/{streaming,errors}.py`: small kernel vocabulary owned by each concern.
- `src/bub/builtin/`: builtin runtime, settings, tools, tape services, and provider adapters.
- `src/bub/builtin/hook_impl.py`: `BuiltinImpl` (turn pipeline) and `BatteryImpl` (opt-in tape store, spill, shells).
- `src/bub/skills.py` / `src/bub/tools.py`: skill discovery over explicit roots, and the tool framework.

Everything is instance-scoped: no environment variables, no process-wide registries, no implicit user or working directories. The framework takes `workspace`, `home`, and an optional `Config`; the agent takes its tools and skill roots.

Tests live in `tests/`.

## Build, Test, and Development Commands

- `make install`: sync the Python dependencies.
- `uv run ruff check .`: run Ruff directly when you only want lint feedback.
- `uv run ruff format .`: format the tree.
- `uv run mypy src`: run mypy directly against `src/`.
- `uv run pytest -q`: run the main test suite without doctests.
- `make test`: run pytest with doctests enabled.
- `make check`: lock validation, Ruff, formatting, and typing.

## Coding Style & Naming Conventions

- Python 3.12+, 4-space indentation, and type hints for new or modified logic.
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

- Never commit credentials. Pass them to `Config` from your own secret store or an untracked file.
- Bub reads no environment variables; provider SDKs may still read their own (for example `OPENROUTER_API_KEY`) when a key is not supplied through configuration.
