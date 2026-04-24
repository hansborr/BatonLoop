# Ralph Loop (Python)

This is a Python rewrite of the `ralph.sh` loop runner. The first version keeps the original Claude-focused behavior, but the code is structured around provider adapters so Codex and other agents can be added without turning the main loop into a pile of shell conditionals.

## Current status

- Typed Python CLI with `argparse` and `dataclasses`
- Core loop engine for prompt cycling, retries, limits, and log retention
- Graceful shutdown with first `Ctrl+C` waiting for the current run and second `Ctrl+C` force-terminating it
- Claude provider adapter with command building, error classification, and cost extraction
- Codex provider adapter with `codex exec` support and provider-specific validation
- Small stdlib test suite

## Run it

```bash
python3 -m ralph --help
python3 -m ralph --provider claude -f ./PROMPT.md
python3 -m ralph --provider codex -f ./PROMPT.md
```

You can also install the local package and use the console script:

```bash
pip install -e .
ralph --provider claude -f ./PROMPT.md
```

## Notes

- `codex` currently supports `stream-json` only; `--no-stream` is rejected for that provider.
- `codex` does not expose explicit cost data in the local JSON stream today, so cost tracking remains `0` unless the CLI starts emitting cost fields.
- The core loop no longer depends on `jq`, `bc`, or `setsid`.
- Adding another provider should mostly be a matter of implementing another adapter in `ralph/providers/`.
