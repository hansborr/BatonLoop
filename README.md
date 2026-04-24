# BatonLoop (Python)

This is a Python rewrite of the old `ralph.sh` loop runner. The first version keeps the original Claude-focused behavior, but the code is structured around provider adapters so Codex and other agents can be added without turning the main loop into a pile of shell conditionals.

## Current status

- Typed Python CLI with `argparse` and `dataclasses`
- Core loop engine for prompt cycling, retries, limits, and log retention
- Iteration watchdog plus post-iteration checks and explicit stop conditions
- Graceful shutdown with first `Ctrl+C` waiting for the current run and second `Ctrl+C` force-terminating it
- Claude provider adapter with command building, error classification, and cost extraction
- Codex provider adapter with `codex exec` support and provider-specific validation
- Automatic in-process failover across multiple providers on eligible failures such as rate limits
- Provider-specific settings loaded from `batonloop-providers.toml`
- Resume handoff support so a new run can pick up from a prior iteration log, including cross-provider handoff
- Resume handoff summaries extracted from prior Claude/Codex logs so the next provider gets a compact state snapshot instead of raw log noise
- Per-iteration metadata artifacts for future resume/failover runs
- Small stdlib test suite

## Run it

```bash
python3 -m batonloop --help
python3 -m batonloop --provider claude -f ./PROMPT.md
python3 -m batonloop --provider codex -f ./PROMPT.md
python3 -m batonloop --provider claude --provider codex -f ./PROMPT.md
python3 -m batonloop --provider codex -f ./PROMPT.md --iteration-timeout 20 --check "pytest -q"
python3 -m batonloop -f ./PROMPT.md --stop-on-regex "DONE" --stop-when-file ./DONE.flag
python3 -m batonloop --provider codex -f ./PROMPT.md --resume-from ./batonloop-logs --resume-note "Claude hit a usage limit; continue from the in-progress work."
```

You can also install the local package and use the console script:

```bash
pip install -e .
batonloop --provider claude -f ./PROMPT.md
```

## Provider Config

Create `./batonloop-providers.toml` to define provider-specific settings once:

```toml
[providers.claude]
model = "opus-4.7"
safe = true

[providers.codex]
model = "gpt-5.4"
bare = true
```

BatonLoop automatically loads that file when it exists. It also falls back to `./ralph-providers.toml` for compatibility. You can point at another file with `--provider-config`.

When you run:

```bash
python3 -m batonloop --provider claude --provider codex -f ./PROMPT.md
```

BatonLoop starts with Claude using the Claude profile and, if it hits an auto-failover condition such as a rate limit, switches to Codex in the same process using the Codex profile.

## Notes

- `codex` currently supports `stream-json` only; `--no-stream` is rejected for that provider.
- `codex` does not expose explicit cost data in the local JSON stream today, so cost tracking remains `0` unless the CLI starts emitting cost fields.
- Repeat `--provider` to define failover order. BatonLoop keeps using the current provider until it hits an eligible failover condition or you stop the loop.
- `--check` commands are run with the current shell and stop the loop when all configured checks pass.
- `--stop-on-clean-git` ignores the configured log directory so BatonLoop's own log files do not keep the repo dirty.
- `--resume-from` accepts either an `iteration-*.json` log or a BatonLoop log directory. BatonLoop resolves that to a prior iteration, writes per-iteration `.meta.json` artifacts, extracts a compact summary from the prior log when possible, and appends a generated handoff block to each resumed prompt.
- Auto-failover reuses the same handoff mechanism internally: the failed iteration becomes the resume source for the next provider in the configured order.
- The core loop no longer depends on `jq`, `bc`, or `setsid`.
- Adding another provider should mostly be a matter of implementing another adapter in `batonloop/providers/`.
