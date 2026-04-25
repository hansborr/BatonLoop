from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .config import (
    OutputFormat,
    build_config,
    parse_decimal_at_least_one,
    parse_fraction,
    parse_non_negative_decimal,
    parse_non_negative_int,
    parse_positive_int,
    resolve_path,
)
from .handoff import resolve_resume_context
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
    subparsers = parser.add_subparsers(dest="command")
    handoff_parser = subparsers.add_parser(
        "handoff-summary",
        help="Print the extracted resume handoff summary for an iteration log.",
        description="Print the extracted resume handoff summary for an iteration log.",
    )
    _add_handoff_summary_arguments(handoff_parser)
    return parser


def _add_handoff_summary_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "resume_source",
        help=(
            "Iteration log, iteration artifact, or BatonLoop log directory to summarize."
        ),
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

    try:
        config = build_config(args)
        return run_loop(config, PROVIDERS)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(status=1, message=f"ERROR: {exc}\n")
