from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import (
    OutputFormat,
    ProviderMode,
    ProviderStrategy,
    build_config,
    parse_decimal_at_least_one,
    parse_fraction,
    parse_non_negative_decimal,
    parse_non_negative_int,
    parse_positive_int,
    resolve_path,
)
from .handoff import prompt_artifact_path_for, resolve_resume_context
from .providers import ClaudeProvider, CopilotProvider, CodexProvider
from .runner import run_loop

PROVIDERS = {
    "claude": ClaudeProvider(),
    "copilot": CopilotProvider(),
    "codex": CodexProvider(),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="batonloop",
        description="Run AI coding agents in a loop.",
    )

    parser.add_argument(
        "--config",
        help=(
            "TOML file with run settings. Defaults to ./batonloop.toml when that file exists."
        ),
    )
    parser.add_argument(
        "--provider",
        action="append",
        dest="provider_names",
        choices=sorted(PROVIDERS),
        help="Provider adapter to use. Repeat to define automatic failover order.",
    )
    parser.add_argument(
        "--provider-config",
        help=(
            "TOML file with per-provider settings. Defaults to ./batonloop-providers.toml "
            "when that file exists."
        ),
    )
    parser.add_argument(
        "--provider-binary",
        help="Default executable name or path for providers without a profile-specific binary.",
    )
    parser.add_argument(
        "-f",
        "--prompt-file",
        action="append",
        dest="prompt_specs",
        help="Prompt file, optionally with :N repeat count. Can be specified multiple times.",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        dest="max_iterations",
        type=parse_non_negative_int,
        default=None,
        help="Max provider-run attempts, including failed iterations. Use 0 for unlimited.",
    )
    parser.add_argument(
        "-c",
        "--max-cost",
        type=parse_non_negative_decimal,
        default=None,
        help="Max cumulative cost in USD. Use 0 for unlimited.",
    )
    parser.add_argument(
        "-d",
        "--duration",
        dest="max_duration_hours",
        type=parse_non_negative_decimal,
        default=None,
        help="Max duration in hours. Use 0 for unlimited.",
    )
    parser.add_argument(
        "-p",
        "--pause",
        dest="pause_seconds",
        type=parse_non_negative_int,
        default=None,
        help="Pause between iterations in seconds.",
    )
    parser.add_argument(
        "--iteration-timeout",
        dest="iteration_timeout_minutes",
        type=parse_non_negative_decimal,
        default=None,
        help="Timeout in minutes for each provider run and post-iteration check. Use 0 for unlimited.",
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Default model name for providers without a profile-specific model.",
    )
    parser.add_argument(
        "-w",
        "--wait-on-limit",
        dest="wait_on_limit_mins",
        type=parse_non_negative_int,
        default=None,
        help="Minutes to wait after a detected rate limit.",
    )
    parser.add_argument(
        "--retry-backoff-base",
        dest="retry_backoff_base_seconds",
        type=parse_non_negative_int,
        default=None,
        help=(
            "Initial retry backoff in seconds for nonfatal provider failures. "
            "Use 0 to disable exponential retry backoff."
        ),
    )
    parser.add_argument(
        "--retry-backoff-multiplier",
        type=parse_decimal_at_least_one,
        default=None,
        help="Multiplier applied to retry backoff after each consecutive nonfatal error.",
    )
    parser.add_argument(
        "--retry-backoff-max",
        dest="retry_backoff_max_seconds",
        type=parse_non_negative_int,
        default=None,
        help="Maximum retry backoff in seconds. Use 0 for no cap.",
    )
    parser.add_argument(
        "--retry-jitter",
        dest="retry_jitter_fraction",
        type=parse_fraction,
        default=None,
        help="Random retry jitter as a fraction from 0 to 1, for example 0.2 for +/-20%%.",
    )
    parser.add_argument(
        "--provider-cooldown",
        dest="provider_cooldown_seconds",
        type=parse_non_negative_int,
        default=None,
        help=(
            "Seconds to keep a provider out of failover rotation after a failover-eligible "
            "failure. Use 0 to disable provider cooldowns."
        ),
    )
    parser.add_argument(
        "--provider-strategy",
        choices=[strategy.value for strategy in ProviderStrategy],
        default=None,
        help=(
            "Provider selection strategy: failover keeps using the current provider until "
            "an eligible failure; alternate advances after each successful iteration."
        ),
    )
    parser.add_argument(
        "--provider-mode",
        choices=[mode.value for mode in ProviderMode],
        default=None,
        help="Provider execution mode. Defaults to exec; tmux keeps an interactive provider session alive.",
    )
    parser.add_argument(
        "-e",
        "--max-errors",
        dest="max_consecutive_errors",
        type=parse_positive_int,
        default=None,
        help="Max consecutive errors before stopping.",
    )
    parser.add_argument(
        "--max-turns",
        type=parse_positive_int,
        help="Default max agentic turns for providers without a profile-specific setting.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for logs.",
    )
    parser.add_argument(
        "--log-retain",
        type=parse_non_negative_int,
        default=None,
        help="Keep only the last N iteration logs. Use 0 to keep all.",
    )
    parser.add_argument(
        "--check",
        action="append",
        dest="check_commands",
        help="Run a shell command after each successful iteration. Stop when all configured checks pass.",
    )
    parser.add_argument(
        "--stop-on-regex",
        action="append",
        dest="stop_on_regexes",
        help="Stop when the given regular expression matches the current iteration log.",
    )
    parser.add_argument(
        "--stop-on-clean-git",
        action="store_true",
        default=None,
        help="Stop when the current Git worktree is clean after a successful iteration.",
    )
    parser.add_argument(
        "--stop-when-file",
        action="append",
        dest="stop_when_files",
        help="Stop when the given file exists after a successful iteration. Can be specified multiple times.",
    )
    parser.add_argument(
        "--output-format",
        choices=[format_.value for format_ in OutputFormat],
        default=None,
        help="Provider output format.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        default=None,
        help="Shortcut for --output-format json.",
    )
    parser.add_argument(
        "--live-output",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show filtered live provider progress in the console and BatonLoop log.",
    )
    parser.add_argument(
        "--bare",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Default bare/minimal mode for providers without a profile-specific setting.",
    )
    parser.add_argument(
        "--safe",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Default non-bypass/sandboxed mode for providers without a profile-specific setting.",
    )
    parser.add_argument(
        "--resume",
        dest="resume_latest",
        action="store_true",
        default=None,
        help=(
            "Resume from the latest iteration in the configured log directory. "
            "This appends generated handoff context to the next prompt."
        ),
    )
    parser.add_argument(
        "--resume-from",
        help=(
            "Previous iteration log or BatonLoop log directory to resume from. "
            "BatonLoop appends a generated handoff block to each prompt so a new provider "
            "can pick up interrupted work."
        ),
    )
    parser.add_argument(
        "--resume-note",
        help="Additional operator note to include in the generated resume handoff block.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Show config and exit.",
    )
    parser.add_argument(
        "--keep-tmux-sessions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep private tmux sessions after BatonLoop exits for debugging.",
    )
    subparsers = parser.add_subparsers(dest="command")
    handoff_parser = subparsers.add_parser(
        "handoff-summary",
        help="Print the extracted resume handoff summary for an iteration log.",
        description="Print the extracted resume handoff summary for an iteration log.",
    )
    _add_handoff_summary_arguments(handoff_parser)
    inspect_parser = subparsers.add_parser(
        "inspect-handoff",
        help="Inspect whether iterations received generated handoff prompts.",
        description="Inspect whether iterations received generated handoff prompts.",
    )
    _add_inspect_handoff_arguments(inspect_parser)
    return parser


def _add_handoff_summary_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "resume_source",
        help=(
            "Iteration log, iteration artifact, or BatonLoop log directory to summarize."
        ),
    )


def _add_inspect_handoff_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("log_dir", help="BatonLoop log directory to inspect.")
    parser.add_argument(
        "--iterations",
        nargs="+",
        type=parse_positive_int,
        help="Iteration numbers to inspect. Defaults to all iterations with metadata.",
    )
    parser.add_argument(
        "--first",
        type=parse_positive_int,
        default=3,
        help="Number of initial progress messages and tasks to show per iteration.",
    )


def main(argv: Sequence[str] | None = None) -> int:
    argv = tuple(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "handoff-summary":
        try:
            resume_context = resolve_resume_context(
                resolve_path(Path(args.resume_source).expanduser(), Path.cwd())
            )
        except (FileNotFoundError, ValueError) as exc:
            parser.exit(status=1, message=f"ERROR: {exc}\n")

        print(resume_context.previous_handoff_summary or "<no summary>")
        return 0

    if args.command == "inspect-handoff":
        try:
            return _inspect_handoff_command(args)
        except (FileNotFoundError, ValueError) as exc:
            parser.exit(status=1, message=f"ERROR: {exc}\n")

    try:
        config = build_config(args)
        return run_loop(config, PROVIDERS)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(status=1, message=f"ERROR: {exc}\n")


def _inspect_handoff_command(args: argparse.Namespace) -> int:
    log_dir = resolve_path(Path(args.log_dir).expanduser(), Path.cwd())
    if not log_dir.is_dir():
        raise FileNotFoundError(f"Handoff log directory not found: {log_dir}")

    iterations = (
        tuple(args.iterations)
        if args.iterations
        else _discover_metadata_iterations(log_dir)
    )
    if not iterations:
        print(f"No iteration metadata found in {log_dir}")
        return 0

    for index, iteration_number in enumerate(iterations):
        if index:
            print()
        _print_iteration_inspection(
            log_dir=log_dir,
            iteration_number=iteration_number,
            first_count=args.first,
        )

    return 0


def _discover_metadata_iterations(log_dir: Path) -> tuple[int, ...]:
    iteration_numbers: list[int] = []
    for path in log_dir.glob("iteration-*.meta.json"):
        try:
            iteration_numbers.append(
                int(path.name.removeprefix("iteration-").removesuffix(".meta.json"))
            )
        except ValueError:
            continue
    return tuple(sorted(iteration_numbers))


def _print_iteration_inspection(
    *,
    log_dir: Path,
    iteration_number: int,
    first_count: int,
) -> None:
    metadata_path = log_dir / f"iteration-{iteration_number:06d}.meta.json"
    previous_metadata_path = log_dir / f"iteration-{iteration_number - 1:06d}.meta.json"
    metadata = _load_json_object(metadata_path)
    previous_metadata = _load_json_object(previous_metadata_path)

    print(f"Iteration {iteration_number:06d}")
    if metadata is None:
        print("Metadata: missing")
        return

    log_path = _resolve_iteration_log_path(
        log_dir=log_dir,
        iteration_number=iteration_number,
        metadata=metadata,
    )

    if previous_metadata is not None:
        print("Previous iteration summary:")
        print(_indent(previous_metadata.get("handoff_summary") or "<none>"))

    prompt_artifact_path = prompt_artifact_path_for(log_path)
    input_prompt_path = _string_value(metadata.get("input_prompt_path"))
    base_prompt_path = _string_value(metadata.get("base_prompt_path"))
    generated_prompt = (
        input_prompt_path == str(prompt_artifact_path) and prompt_artifact_path.is_file()
    )
    print(f"Generated prompt artifact: {'yes' if generated_prompt else 'no'}")
    print(f"Input prompt: {input_prompt_path or '<missing>'}")
    resume_source_log = _string_value(metadata.get("resume_source_log_path")) or "none"
    print(f"Resume source log: {resume_source_log}")
    print(
        "Resume source metadata: "
        f"{_string_value(metadata.get('resume_source_metadata_path')) or 'none'}"
    )

    if (
        previous_metadata is not None
        and previous_metadata.get("success") is False
        and input_prompt_path == base_prompt_path
    ):
        print(
            "WARNING: previous iteration failed but this iteration used "
            "the base prompt directly."
        )

    progress_messages, tasks = _extract_iteration_start(log_path, first_count)
    print(f"First {first_count} progress message(s):")
    print(_indent("\n".join(progress_messages) if progress_messages else "<none>"))
    print(f"First {first_count} task/tool call(s):")
    print(_indent("\n".join(tasks) if tasks else "<none>"))


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _resolve_iteration_log_path(
    *,
    log_dir: Path,
    iteration_number: int,
    metadata: dict[str, Any],
) -> Path:
    metadata_log_path = _string_value(metadata.get("log_path"))
    if metadata_log_path:
        path = Path(metadata_log_path)
        if path.is_file():
            return path

    for suffix in (".json", ".log"):
        path = log_dir / f"iteration-{iteration_number:06d}{suffix}"
        if path.is_file():
            return path

    return log_dir / f"iteration-{iteration_number:06d}.json"


def _extract_iteration_start(
    log_path: Path,
    limit: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    progress_messages: list[str] = []
    tasks: list[str] = []

    try:
        with log_path.open(encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                if len(progress_messages) >= limit and len(tasks) >= limit:
                    break
                try:
                    payload = json.loads(raw_line)
                except json.JSONDecodeError:
                    if len(progress_messages) < limit:
                        text = _clean_interactive_cli_line(raw_line)
                        if text:
                            progress_messages.append(text)
                    continue
                if isinstance(payload, dict):
                    if len(progress_messages) < limit:
                        progress_messages.extend(_progress_messages_from_payload(payload))
                    if len(tasks) < limit:
                        tasks.extend(_tasks_from_payload(payload))
    except OSError:
        return (), ()

    return tuple(progress_messages[:limit]), tuple(tasks[:limit])


def _progress_messages_from_payload(payload: dict[str, Any]) -> list[str]:
    payload_type = payload.get("type")
    if payload_type == "item.completed":
        item = payload.get("item")
        if isinstance(item, dict) and item.get("type") == "agent_message":
            text = _clean_cli_text(item.get("text"))
            return [text] if text else []

    if payload_type != "assistant":
        return []

    message = payload.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return []
    return _content_texts(message.get("content"))


def _tasks_from_payload(payload: dict[str, Any]) -> list[str]:
    if payload.get("type") == "system" and payload.get("subtype") == "task_started":
        description = _clean_cli_text(payload.get("description"))
        prompt = _clean_cli_text(payload.get("prompt"))
        return [_join_task_parts(description, prompt)] if description or prompt else []

    item = payload.get("item")
    if isinstance(item, dict) and item.get("type") == "collab_tool_call":
        tool = _clean_cli_text(item.get("tool"))
        prompt = _clean_cli_text(item.get("prompt"))
        return [_join_task_parts(tool, prompt)] if tool or prompt else []

    return []


def _content_texts(content: object) -> list[str]:
    if isinstance(content, str):
        text = _clean_cli_text(content)
        return [text] if text else []
    if not isinstance(content, list):
        return []
    texts: list[str] = []
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "text":
            continue
        text = _clean_cli_text(item.get("text"))
        if text:
            texts.append(text)
    return texts


def _join_task_parts(description: str | None, prompt: str | None) -> str:
    if description and prompt:
        return f"{description}: {prompt}"
    return description or prompt or ""


def _clean_cli_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    return text or None


def _clean_interactive_cli_line(value: str) -> str | None:
    text = _clean_cli_text(value)
    if not text:
        return None
    lowered = text.lower()
    if text.startswith("BATONLOOP_TURN_COMPLETE ") or "batonloop control" in lowered:
        return None
    if lowered.startswith(
        (
            "when you are completely finished with this turn",
            "marker prefix:",
            "marker id:",
            "do not emit that line until",
        )
    ):
        return None
    return text


def _string_value(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _indent(text: object) -> str:
    return "\n".join(f"  {line}" for line in str(text).splitlines())
