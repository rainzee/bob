# Repository Guidelines

## Project Structure & Module Organization

Core code lives under `src/`:

- `src/bub/framework.py`: turn orchestration, hook loading, and resource lifecycle.
- `src/bub/hooks/`: hook specifications, execution helpers, and interception contracts.
- `src/bub/message.py`: the concrete inbound envelope (`Message`) and media items.
- `src/bub/{envelope,turn,streaming,errors,model_selection}.py`: small kernel vocabulary owned by each concern.
- `src/bub/builtin/`: builtin runtime, settings, tools, tape services, and provider adapters.
- `src/bub/skills.py` / `src/bub/tools.py`: skill discovery and tool registry.

Tests live in `tests/`.

## Build, Test, and Development Commands

- `uv sync`: install or update the Python environment.
- `make install`: full local development bootstrap; sync Python deps and install `prek` hooks.
- `uv run ruff check .`: run Ruff directly when you only want lint feedback.
- `uv run mypy src`: run mypy directly against `src/`.
- `uv run pytest -q`: run the main test suite without doctests.
- `make test`: run pytest with doctests enabled.
- `make check`: lock validation, `prek`, and typing.

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

- Use `.env` for local secrets; never commit credentials.
- Bub runtime settings are driven by `BUB_*` variables such as `BUB_MODEL`, `BUB_API_KEY`, and `BUB_API_BASE`.
- Provider-specific keys such as `OPENROUTER_API_KEY` may still be consumed by downstream SDKs.
