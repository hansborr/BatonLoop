# Repository Guidelines

## Project Structure & Module Organization

`batonloop/` contains the Python package. The CLI entry point is `batonloop/cli.py`, the loop engine lives in `batonloop/runner.py`, config parsing is in `batonloop/config.py`, resume summary logic is in `batonloop/handoff.py`, and provider adapters live under `batonloop/providers/`. Add new providers by extending `batonloop/providers/base.py` and registering them in the CLI. `tests/` mirrors the package by behavior area, and `example_logs/` holds sample iteration artifacts used for reference.

## Build, Test, and Development Commands

- `python3 -m batonloop --help`: run the CLI from the working tree.
- `python3 -m batonloop --provider codex -f ./PROMPT.md --check "pytest -q"`: run a local loop with a post-iteration test check.
- `python3 -m batonloop handoff-summary ./batonloop-logs/iteration-000001.json`: inspect a saved iteration summary.
- `pip install -e .`: install the package and expose the `batonloop` console script.
- `pytest -q`: run the full test suite.

The project uses setuptools via `pyproject.toml` and requires Python 3.11 or newer.

## Coding Style & Naming Conventions

Use standard Python formatting with 4-space indentation, typed function signatures where practical, and `from __future__ import annotations` in source and test files. Keep modules focused on one responsibility and prefer dataclasses or small helper functions over broad dictionaries of untyped state. Name test files `test_<feature>.py`, provider modules after their CLI provider name, and provider classes as `<Name>Provider`.

## Testing Guidelines

Tests use `unittest` assertions and pytest discovery. Add or update tests in `tests/` for every behavior change, especially provider command construction, error classification, config parsing, and loop stop conditions. Keep tests deterministic by using temporary directories and mocked subprocess behavior instead of invoking real provider CLIs.

## Commit & Pull Request Guidelines

Recent commits use concise imperative subjects such as `Add GitHub Copilot provider support` and `Fix handoff summary extraction and CLI help`. Follow that style: start with a verb, describe the user-visible change, and keep the subject focused. Pull requests should include a short problem statement, a summary of changes, test results such as `pytest -q`, and any CLI examples or log snippets needed to review provider behavior.

## Security & Configuration Tips

Do not commit local `batonloop-providers.toml` files containing private binary paths, credentials, or machine-specific settings. Treat provider logs as potentially sensitive because prompts and model output are stored verbatim. Prefer profile `args` for provider-specific flags instead of adding new core config options unless the setting applies across providers.
