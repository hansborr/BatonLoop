from __future__ import annotations

import json
import unittest
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from batonloop.config import (
    OutputFormat,
    PromptSpec,
    ProviderProfile,
    RunnerConfig,
    resolve_provider_execution,
)
from batonloop.providers import ClaudeProvider, FailureKind


class ClaudeProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = ClaudeProvider()

    def test_build_command_matches_claude_flags(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config = _make_config(Path(tmp_dir))
            execution = resolve_provider_execution(config, "claude")
            command = self.provider.build_command(config, execution)

        self.assertEqual(
            command,
            [
                "claude",
                "-p",
                "--output-format",
                "stream-json",
                "--verbose",
                "--dangerously-skip-permissions",
                "--model",
                "sonnet",
                "--max-turns",
                "7",
            ],
        )

    def test_build_command_appends_extra_provider_args(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config = _make_config(
                Path(tmp_dir),
                extra_args=("--custom-flag", "custom-value"),
            )
            execution = resolve_provider_execution(config, "claude")
            command = self.provider.build_command(config, execution)

        self.assertEqual(
            command,
            [
                "claude",
                "-p",
                "--output-format",
                "stream-json",
                "--verbose",
                "--dangerously-skip-permissions",
                "--model",
                "sonnet",
                "--max-turns",
                "7",
                "--custom-flag",
                "custom-value",
            ],
        )

    def test_extract_cost_from_stream_json_log(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "iteration-000001.json"
            log_path.write_text(
                "\n".join(
                    [
                        json.dumps({"type": "message", "content": "hello"}),
                        json.dumps({"type": "result", "total_cost_usd": 1.25}),
                    ]
                ),
                encoding="utf-8",
            )

            cost = self.provider.extract_cost(log_path, OutputFormat.STREAM_JSON)

        self.assertEqual(cost, Decimal("1.25"))

    def test_classify_rate_limit_error(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            log_path = temp_root / "iteration-000001.json"
            log_path.write_text("hit rate_limit while calling API", encoding="utf-8")
            config = _make_config(temp_root)
            decision = self.provider.classify_failure(
                exit_code=1,
                log_path=log_path,
                config=config,
                execution=resolve_provider_execution(config, "claude"),
            )

        self.assertFalse(decision.fatal)
        self.assertEqual(decision.kind, FailureKind.RATE_LIMIT)
        self.assertEqual(decision.wait_seconds, 1800)
        self.assertTrue(decision.reset_error_count)
        self.assertTrue(decision.skip_pause)
        self.assertTrue(decision.should_failover)

    def test_classify_rate_limit_from_structured_stream_events(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            log_path = temp_root / "iteration-000001.json"
            log_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "type": "rate_limit_event",
                                "rate_limit_info": {
                                    "status": "rejected",
                                    "rateLimitType": "five_hour",
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "assistant",
                                "error": "rate_limit",
                            }
                        ),
                        json.dumps(
                            {
                                "type": "result",
                                "is_error": True,
                                "api_error_status": 429,
                                "result": "You've hit your limit. Try again later.",
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            config = _make_config(temp_root)
            decision = self.provider.classify_failure(
                exit_code=1,
                log_path=log_path,
                config=config,
                execution=resolve_provider_execution(config, "claude"),
            )

        self.assertEqual(decision.kind, FailureKind.RATE_LIMIT)
        self.assertEqual(decision.wait_seconds, 1800)
        self.assertTrue(decision.reset_error_count)
        self.assertTrue(decision.skip_pause)
        self.assertTrue(decision.should_failover)

    def test_classify_overload_from_structured_result(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            log_path = temp_root / "iteration-000001.json"
            log_path.write_text(
                json.dumps(
                    {
                        "type": "result",
                        "is_error": True,
                        "api_error_status": 529,
                        "result": "Temporarily unavailable.",
                    }
                ),
                encoding="utf-8",
            )
            config = _make_config(temp_root)
            decision = self.provider.classify_failure(
                exit_code=1,
                log_path=log_path,
                config=config,
                execution=resolve_provider_execution(config, "claude"),
            )

        self.assertEqual(decision.kind, FailureKind.OVERLOADED)
        self.assertEqual(decision.wait_seconds, 120)
        self.assertTrue(decision.skip_pause)

    def test_classify_rate_limit_from_json_output(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            log_path = temp_root / "iteration-000001.json"
            log_path.write_text(
                json.dumps(
                    {
                        "is_error": True,
                        "api_error_status": 429,
                        "result": "You've hit your limit. Try again later.",
                    }
                ),
                encoding="utf-8",
            )
            config = _make_config(temp_root, output_format=OutputFormat.JSON)
            decision = self.provider.classify_failure(
                exit_code=1,
                log_path=log_path,
                config=config,
                execution=resolve_provider_execution(config, "claude"),
            )

        self.assertEqual(decision.kind, FailureKind.RATE_LIMIT)
        self.assertEqual(decision.wait_seconds, 1800)
        self.assertTrue(decision.should_failover)


def _make_config(
    temp_root: Path,
    *,
    output_format: OutputFormat = OutputFormat.STREAM_JSON,
    extra_args: tuple[str, ...] = (),
) -> RunnerConfig:
    prompt_path = temp_root / "PROMPT.md"
    prompt_path.write_text("prompt", encoding="utf-8")
    return RunnerConfig(
        working_dir=temp_root,
        provider_names=("claude",),
        provider_profiles={
            "claude": ProviderProfile(
                model="sonnet",
                max_turns=7,
                extra_args=extra_args,
            )
        },
        provider_config_path=None,
        default_provider_profile=ProviderProfile(),
        prompt_specs=(PromptSpec(path=prompt_path, repeat=1),),
        prompt_sequence=(prompt_path,),
        max_iterations=0,
        max_cost=Decimal("0"),
        max_duration_hours=Decimal("0"),
        iteration_timeout_minutes=Decimal("0"),
        pause_seconds=5,
        wait_on_limit_mins=30,
        max_consecutive_errors=5,
        log_dir=temp_root / "logs",
        log_retain=0,
        check_commands=(),
        stop_on_regexes=(),
        stop_on_clean_git=False,
        stop_when_files=(),
        output_format=output_format,
        live_output=True,
        resume_from=None,
        resume_note=None,
        dry_run=False,
    )


if __name__ == "__main__":
    unittest.main()
