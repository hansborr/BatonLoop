# Interactive tmux provider mode research and implementation plan

Research date: 2026-06-06

Local versions observed in this workspace:

- `codex-cli 0.137.0`
- `Claude Code 2.1.167`
- `tmux 3.3a`

## Goal

BatonLoop currently runs provider CLIs as one process per iteration:

- Claude: `claude -p --output-format ...`
- Codex: `codex exec --json ...`

The requested mode is different: keep each provider in an interactive terminal session, drive it through `tmux send-keys`, and run `/clear` between BatonLoop prompts so each iteration starts a fresh provider conversation without paying process startup cost or losing the interactive CLI behavior.

## Confirmed provider behavior

### Codex

The Codex CLI has two relevant surfaces:

- `codex exec` is the documented non-interactive automation surface. It can emit JSONL events and read prompts from stdin.
- `codex` without a subcommand launches the interactive TUI. It accepts global flags such as `--model`, `--cd`, `--sandbox`, `--ask-for-approval`, `--dangerously-bypass-approvals-and-sandbox`, and `--no-alt-screen`.

Sources:

- https://developers.openai.com/codex/cli/reference#codex-interactive
- https://developers.openai.com/codex/noninteractive

Codex documents `/clear` in the interactive CLI. It clears the terminal, resets the visible transcript, and starts a fresh chat in the same CLI session. The docs explicitly distinguish this from `Ctrl+L`, which only clears the terminal view and keeps the chat. Codex also disables `/clear` while a task is in progress, so BatonLoop must only send it after the prior turn is complete.

Source:

- https://developers.openai.com/codex/cli/slash-commands#clear-the-terminal-and-start-a-new-chat-with-clear

Important Codex caveats for this repo:

- Current BatonLoop Codex command construction uses `codex exec --json`, so the runner receives structured per-turn events and a process exit code. Interactive mode will not provide either of those directly.
- Local `codex --help` shows `--no-alt-screen`, which is useful for tmux capture and scrollback preservation.
- Local `codex --help` does not expose `--ignore-user-config` or `--ignore-rules` for the interactive command. Those are available to `codex exec`, so BatonLoop's existing `bare=true` behavior cannot be mapped exactly in interactive Codex mode.
- Local `codex exec --help` still supports structured JSONL output, so keeping non-interactive mode as the default remains important.

### Claude Code

Claude Code starts an interactive session by default. `-p`/`--print` is its non-interactive mode, and its JSON and stream-json output options apply to print mode.

Sources:

- https://code.claude.com/docs/en/cli-usage

Claude Code documents `/clear [name]` as starting a new conversation with empty context while leaving the previous conversation available in `/resume`. `/reset` and `/new` are aliases.

Source:

- https://code.claude.com/docs/en/commands

Interactive mode supports slash commands at the start of a message, `Ctrl+C` interruption, multiline paste, shell mode, and command history. The docs note that input history resets when `/clear` starts a new session.

Source:

- https://code.claude.com/docs/en/interactive-mode

Claude Code has an opt-in fullscreen renderer that uses the alternate screen buffer. That is usually bad for automation that wants terminal scrollback or simple tmux captures. The docs say the classic renderer can be forced with `CLAUDE_CODE_DISABLE_ALTERNATE_SCREEN=1`, which is the safer default for BatonLoop's tmux mode.

Source:

- https://code.claude.com/docs/en/fullscreen

Important Claude caveats for this repo:

- `--max-turns` is documented and shown locally as print-mode-only. BatonLoop should reject `max_turns` for Claude interactive mode.
- `--fallback-model`, `--output-format`, `--input-format`, `--max-budget-usd`, `--no-session-persistence`, and related structured automation flags are print-mode-only and should not be passed to interactive mode.
- `--bare` is available for interactive mode.
- Permission handling remains an operator choice. `safe=false` can keep using `--dangerously-skip-permissions`. For `safe=true`, BatonLoop should not silently choose a mode that can block forever on permission dialogs without documenting it.

## tmux primitives

tmux supports the control operations BatonLoop needs:

- `new-session -d -P -F '#{pane_id}'` can start a detached session and return a stable pane id.
- `send-keys -t <pane> ... Enter` can send key presses to a pane.
- `send-keys -l` disables key-name lookup and sends literal UTF-8 text.
- `capture-pane -p -S - -E -` can capture pane history and visible content.
- `pipe-pane -O` can stream output from the pane to a file.
- `load-buffer` plus `paste-buffer -p` can paste prompt file contents using bracketed paste when the application requested it.

Sources:

- https://github.com/tmux/tmux/wiki/Advanced-Use
- https://man.openbsd.org/tmux

For BatonLoop prompts, prefer tmux buffers over raw `send-keys -l`:

```bash
tmux load-buffer -b batonloop-prompt /path/to/iteration.prompt.txt
tmux paste-buffer -p -d -b batonloop-prompt -t %pane_id
tmux send-keys -t %pane_id Enter
```

This avoids argv length limits, avoids shell quoting issues, and lets interactive CLIs treat multiline text as a paste into the composer instead of a sequence of submitted lines.

## Impact on BatonLoop's current architecture

The main structural issue is that the runner assumes each iteration has a subprocess that exits:

- `Provider.build_command(...)` returns a non-interactive command.
- `_run_iteration(...)` opens the prompt file as stdin and waits for process exit.
- `_run_subprocess(...)` writes stdout to one per-iteration log.
- Success is `exit_code == 0`.
- Failure classification reads the completed log.
- Cost extraction reads structured provider output.
- Resume summaries expect Claude/Codex JSONL-style logs, with a small fallback for interruption text.

An interactive provider session has different mechanics:

- The provider process remains alive across iterations.
- The prompt is typed or pasted into a TUI.
- There is no natural per-iteration process exit code.
- The TUI output is not provider JSONL.
- `/clear` must happen after the prior turn is complete and before the next prompt.
- If BatonLoop times out or receives a force stop, it must interrupt or kill a tmux pane/session, not a child process spawned for the current iteration.

This is feasible, but it should be implemented as a distinct execution mode rather than by contorting `build_command`.

## Recommended architecture

Add an explicit provider execution mode:

```toml
[run]
provider_mode = "exec" # default, current behavior

[providers.claude]
mode = "tmux"

[providers.codex]
mode = "tmux"
```

CLI spelling can be:

```bash
batonloop --provider claude --provider-mode tmux -f ./PROMPT.md
```

The default must remain `exec` to preserve current JSONL logging, cost extraction, and CI-friendly behavior.

### Provider contract changes

Keep current `build_command(...)` for non-interactive execution. Add provider-owned interactive launch behavior:

```python
class Provider(Protocol):
    def build_command(...): ...
    def build_interactive_command(...): ...
    def interactive_environment(...): ...
    def validate_interactive_config(...): ...
```

Provider-specific mappings:

- Claude interactive command starts with `claude`, not `claude -p`.
- Codex interactive command starts with `codex`, not `codex exec`.
- Both should set the working directory through provider flags where available (`codex -C`, tmux `-c` for both, and Claude's cwd through tmux).
- Codex should add `--no-alt-screen`.
- Claude should set `CLAUDE_CODE_DISABLE_ALTERNATE_SCREEN=1` and likely `CLAUDE_CODE_ENABLE_PROMPT_SUGGESTION=false`.
- Unsafe modes can map to the existing bypass flags.
- Unsupported profile options should be rejected in tmux mode instead of silently ignored.

### Tmux session manager

Add a small tmux driver module, for example `batonloop/tmux_session.py`.

Responsibilities:

- Check `tmux` exists when any provider uses tmux mode.
- Create a private tmux server socket name such as `batonloop-<pid>-<runid>`.
- Lazily start one detached session per provider slot.
- Record stable pane ids.
- Log attach commands, for example:

```bash
tmux -L batonloop-12345 attach -t batonloop-codex-1
```

- Start a continuous raw pane log with `pipe-pane -O`.
- Paste each prompt through `load-buffer` and `paste-buffer -p`.
- Send `/clear` between prompts.
- Poll for turn completion, timeouts, provider death, and known failure text.
- On normal BatonLoop shutdown, kill tmux sessions by default.
- Add an opt-in `keep_tmux_sessions = true` or `--keep-tmux-sessions` for debugging.

Use stable pane ids and private socket names. Do not rely on tmux's current active pane, because BatonLoop may run inside another tmux session or the user may attach while the loop is running.

### Turn completion detection

This is the riskiest part.

Recommended MVP: wrap each prompt with a unique BatonLoop completion marker instruction and wait for that marker in captured pane output:

```text
=== BATONLOOP CONTROL ===
When you are completely finished with this turn, end your final assistant message
with this line exactly:
BATONLOOP_TURN_COMPLETE <uuid>
Do not emit that line until all intended edits and verification for this turn are done.
=== END BATONLOOP CONTROL ===
```

The tmux driver should:

1. Record the raw pane log byte offset before the prompt.
2. Paste and submit the wrapped prompt.
3. Poll the new raw log chunk after that offset.
4. Strip ANSI/control sequences for matching.
5. Return success when the marker appears.
6. Return timeout if `iteration_timeout` expires.
7. Return failure if the provider process exits, the pane disappears, or obvious auth/rate-limit/overload text appears and the pane has gone idle.

Why a marker is preferable to UI parsing:

- Provider TUI wording changes over time.
- Prompt boxes, spinners, and footers are not stable APIs.
- tmux captures may include redraws, carriage returns, or alternate screen behavior.
- Codex and Claude expose structured completion only in non-interactive modes.

The marker is not perfect. The model can omit it or mention it early. BatonLoop can reduce false positives by using a per-turn UUID and requiring it to appear after the prompt submission offset. Timeouts remain the fallback.

### Log files

Interactive logs should not pretend to be JSON.

Recommended filename behavior:

- Keep existing `iteration-000001.json` for current exec mode.
- Use `iteration-000001.log` for tmux mode.
- Update resume discovery to accept both suffixes.
- Keep metadata names based on the stem, for example `iteration-000001.meta.json`.
- Add `provider_mode` and `log_format` fields to metadata.

The per-iteration interactive log should be a sanitized text slice from the continuous pane log. Also keep provider-session raw logs for debugging:

```text
batonloop-logs/
  iteration-000001.log
  iteration-000001.meta.json
  iteration-000001.prompt.txt
  tmux-claude.raw.log
  tmux-codex.raw.log
```

The handoff extractor needs a new text-log path:

- Remove ANSI/control sequences.
- Ignore BatonLoop control-marker lines.
- Extract the last assistant-looking response blocks when possible.
- Preserve current interruption detection.
- Fall back to the last N non-empty lines if no structured messages are found.

Stop regexes can continue to run against the sanitized per-iteration log.

### Runner integration

Do not replace `_run_subprocess`. Add a sibling path:

```python
def _run_iteration(...):
    if execution.mode is ProviderMode.TMUX:
        return _run_tmux_iteration(...)
    return _run_subprocess_iteration(...)
```

`_run_tmux_iteration` should still return `CommandResult` so the existing outcome logic remains mostly intact:

- `exit_code=0` when the completion marker is observed.
- `timed_out=True` on turn timeout.
- `exit_code=1` for provider/pane failure or classified textual failures.

Provider `classify_failure(...)` can still read the log because both Claude and Codex adapters already have text-pattern fallback logic. Add patterns if interactive logs expose different wording.

Cost extraction in tmux mode should return `0` at first. Interactive TUI cost/usage output is not a stable structured stream. Cost tracking can be revisited later with provider-specific `/usage` or `/status` scraping, but that should not block the first implementation.

### Signal handling

`StopController` currently tracks a `subprocess.Popen`. tmux mode needs a generic terminable handle:

```python
class Terminable(Protocol):
    def terminate(self) -> None: ...
    def force_terminate(self) -> None: ...
```

For subprocess mode, this wraps the existing process-tree termination.

For tmux mode:

- First stop request: keep current BatonLoop behavior and wait for the current turn to finish.
- Second stop request: send an interrupt key (`Esc` or `C-c`) and then kill the tmux session if it does not stop promptly.
- Timeout: send interrupt, capture final log slice, then kill or restart the session depending on configuration.

This must be tested because otherwise BatonLoop could leave long-running interactive agents behind.

## Implementation plan

### MVP acceptance criteria

The first implementation should be considered complete when:

- Existing non-interactive provider execution remains the default and keeps the current `.json` log behavior.
- `--provider-mode tmux` and provider TOML `mode = "tmux"` can launch supported providers in private tmux sessions.
- BatonLoop can run at least two iterations through the same interactive session and send `/clear` between prompts.
- Prompt submission uses tmux buffers, not argv-heavy `send-keys -l` prompt text.
- A per-turn completion marker produces `CommandResult(exit_code=0)`, while timeout, pane death, and provider startup failure produce classified failure results.
- Tmux mode writes `iteration-*.log`, per-iteration metadata, prompt artifacts, and raw session logs.
- Resume discovery, handoff summaries, stop regexes, and failover can consume interactive text logs.
- Unsupported provider/profile options in tmux mode fail with clear validation errors.
- The automated test suite covers the tmux driver with a fake interactive CLI and passes with `pytest -q`.

### Phase 1: configuration and provider command support

1. Add `ProviderMode` enum with `EXEC` and `TMUX`.
2. Add `mode` to `ProviderProfile` and `ProviderExecution`.
3. Add `--provider-mode {exec,tmux}` and TOML parsing aliases.
4. Allow per-provider `[providers.<name>].mode`.
5. Keep default mode as `exec`.
6. Add provider tests for Claude and Codex interactive command construction.
7. Add validation tests for unsupported tmux-mode options:
   - Claude `max_turns`
   - Claude print-only output flags if present through core options
   - Codex `bare=true` unless a supported mapping is found

### Phase 2: tmux driver

1. Add `TmuxSessionManager`.
2. Implement session creation with private socket name and stable pane id capture.
3. Implement raw `pipe-pane` logging.
4. Implement prompt paste through `load-buffer` and `paste-buffer -p`.
5. Implement `/clear` submission and basic readiness wait.
6. Implement per-turn completion marker generation and polling.
7. Implement timeout and provider-death handling.
8. Add a fake interactive CLI fixture for tests so unit/integration tests do not invoke real Claude or Codex.

### Phase 3: runner integration

1. Route tmux-mode iterations through `TmuxSessionManager`.
2. Return existing `CommandResult`.
3. Write `iteration-*.log` for tmux mode.
4. Update metadata to include provider mode, tmux session name, pane id, raw pane log path, and attach command.
5. Update `_latest_existing_iteration_number`, log rotation, and resume resolution to understand both `.json` and `.log`.
6. Update `StopController` to support tmux termination.
7. Ensure provider failover and alternate strategy can reuse or lazily start provider sessions.

### Phase 4: text-log handoff and live output

1. Add ANSI/control sequence stripping.
2. Teach `extract_handoff_details` to summarize interactive text logs.
3. Ignore BatonLoop control-marker lines in summaries.
4. Keep interruption detection working for plain text.
5. Add basic live output for tmux mode by tailing sanitized log slices, or explicitly disable filtered live output in tmux mode for the first release.

### Phase 5: documentation and opt-in integration tests

1. Update `README.md` with tmux mode examples.
2. Document attach and cleanup behavior.
3. Document unsupported options and cost-tracking limitations.
4. Add tests:
   - Config parse and CLI override tests.
   - Provider command construction tests.
   - Tmux driver tests using a fake interactive program.
   - Runner tests for two iterations proving `/clear` is sent between prompts.
   - Timeout test proving tmux session cleanup or restart behavior.
   - Resume/failover test from an interactive text log.
5. Run `pytest -q`.

## First-pass out of scope

The first implementation should explicitly avoid:

- Structured cost or usage extraction from interactive TUI output. Return zero cost in tmux mode until there is a stable provider-specific source.
- UI-state parsing for turn completion. Use the per-turn marker and timeout path instead of parsing spinners, prompt boxes, or footer text.
- Full live-output parity with JSONL provider streams. Add a basic sanitized tail if straightforward, otherwise document tmux live output as limited for the first release.
- Adding tmux support for providers beyond Claude and Codex unless their interactive behavior is separately researched and validated.

## Default implementation decisions

- Kill tmux sessions by default at the end of a run. Preserve them only when `--keep-tmux-sessions` or `keep_tmux_sessions = true` is set.
- During failover cooldown, keep a failed provider session only if it is still healthy. Clear it before the next prompt after cooldown.
- Restart the interactive session after timeout or pane death. Reuse it after rate limit, overload, or normal completion when the pane remains healthy.
- Support prompt cycling exactly as exec mode does. `/clear` isolates the provider conversation; BatonLoop's resume handoff still handles failed iterations.
- Use `.log` for interactive iteration logs immediately. Writing terminal text to `.json` would make future tooling harder.

## Feasibility assessment

This is feasible as an opt-in mode. The main risk is reliable turn-completion detection because interactive CLIs are optimized for humans, not machine protocols. A per-turn completion marker plus timeout handling is the most practical MVP. It keeps BatonLoop's current loop, stop, retry, and failover logic mostly intact while making tmux the transport layer for providers that support interactive sessions.

The implementation should not remove or weaken the existing non-interactive path. `codex exec --json` and `claude -p --output-format stream-json` remain more reliable for structured automation, cost extraction, and machine-readable logs. The tmux mode should be presented as a provider-experience option for users who specifically want the interactive CLIs and `/clear` semantics.
