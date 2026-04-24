# Ralph Loop (Python)

This is a Python rewrite of the `ralph.sh` loop runner. The first version keeps the original Claude-focused behavior, but the code is structured around provider adapters so Codex and other agents can be added without turning the main loop into a pile of shell conditionals.

## Current status

- Typed Python CLI with `argparse` and `dataclasses`
- Core loop engine for prompt cycling, retries, limits, and log retention
- Graceful shutdown with first `Ctrl+C` waiting for the current run and second `Ctrl+C` force-terminating it
- Claude provider adapter with command building, error classification, and cost extraction
- Small stdlib test suite

## Run it

```bash
python3 -m ralph --help
python3 -m ralph --provider claude -f ./PROMPT.md
```

You can also install the local package and use the console script:

```bash
pip install -e .
ralph --provider claude -f ./PROMPT.md
```

## Notes

- Only the `claude` provider is implemented so far.
- The core loop no longer depends on `jq`, `bc`, or `setsid`.
- Adding Codex should mostly be a matter of implementing another provider in `ralph/providers/`.

