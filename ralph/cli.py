from __future__ import annotations

import argparse
from collections.abc import Sequence

from .config import (
    OutputFormat,
    build_config,
    parse_non_negative_decimal,
    parse_non_negative_int,
    parse_positive_int,
)
from .providers import ClaudeProvider, CodexProvider
from .runner import run_loop

PROVIDERS = {
    "claude": ClaudeProvider(),
    "codex": CodexProvider(),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ralph",
        description="Run AI coding agents in a loop.",
    )

    parser.add_argument(
        "--provider",
        choices=sorted(PROVIDERS),
        default="claude",
        help="Provider adapter to use.",
    )
    parser.add_argument(
        "--provider-binary",
        help="Override the executable name or path for the selected provider.",
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
        default=0,
        help="Max iterations. Use 0 for unlimited.",
    )
    parser.add_argument(
        "-c",
        "--max-cost",
        type=parse_non_negative_decimal,
        default=parse_non_negative_decimal("0"),
        help="Max cumulative cost in USD. Use 0 for unlimited.",
    )
    parser.add_argument(
        "-d",
        "--duration",
        dest="max_duration_hours",
        type=parse_non_negative_decimal,
        default=parse_non_negative_decimal("0"),
        help="Max duration in hours. Use 0 for unlimited.",
    )
    parser.add_argument(
        "-p",
        "--pause",
        dest="pause_seconds",
        type=parse_non_negative_int,
        default=5,
        help="Pause between iterations in seconds.",
    )
    parser.add_argument(
        "--iteration-timeout",
        dest="iteration_timeout_minutes",
        type=parse_non_negative_decimal,
        default=parse_non_negative_decimal("0"),
        help="Timeout in minutes for each provider run and post-iteration check. Use 0 for unlimited.",
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Model name to pass to the provider.",
    )
    parser.add_argument(
        "-w",
        "--wait-on-limit",
        dest="wait_on_limit_mins",
        type=parse_non_negative_int,
        default=30,
        help="Minutes to wait after a detected rate limit.",
    )
    parser.add_argument(
        "-e",
        "--max-errors",
        dest="max_consecutive_errors",
        type=parse_positive_int,
        default=5,
        help="Max consecutive errors before stopping.",
    )
    parser.add_argument(
        "--max-turns",
        type=parse_positive_int,
        help="Max agentic turns per iteration.",
    )
    parser.add_argument(
        "--log-dir",
        default="./ralph-logs",
        help="Directory for logs.",
    )
    parser.add_argument(
        "--log-retain",
        type=parse_non_negative_int,
        default=0,
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
        default=OutputFormat.STREAM_JSON.value,
        help="Provider output format.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Shortcut for --output-format json.",
    )
    parser.add_argument(
        "--bare",
        action="store_true",
        help="Use the provider's bare/minimal mode when supported.",
    )
    parser.add_argument(
        "--safe",
        action="store_true",
        help="Use the provider's non-bypass/sandboxed mode when supported.",
    )
    parser.add_argument(
        "--resume-from",
        help=(
            "Previous iteration log or Ralph log directory to resume from. "
            "Ralph appends a generated handoff block to each prompt so a new provider "
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
        help="Show config and exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = build_config(args)
        provider = PROVIDERS[config.provider_name]
        return run_loop(config, provider)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(status=1, message=f"ERROR: {exc}\n")
