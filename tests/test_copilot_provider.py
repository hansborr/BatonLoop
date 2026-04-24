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
from batonloop.providers import CopilotProvider, FailureKind


class CopilotProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = CopilotProvider()

    def test_build_command_uses_autopilot_and_full_permissions_by_default(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config = _make_config(Path(tmp_dir))
            execution = resolve_provider_execution(config, "copilot")
            command = self.provider.build_command(config, execution)

        self.assertEqual(
            command,
            [
                "copilot",
                "--output-format",
                "json",
                "--autopilot",
                "--no-ask-user",
                "--allow-all",
                "--model",
                "gpt-5.2",
                "--max-autopilot-continues",
                "7",
            ],
        )

    def test_build_command_uses_safe_and_bare_flags_when_requested(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config = _make_config(
                Path(tmp_dir),
                safe_mode=True,
                use_bare=True,
                model=None,
                max_turns=None,
            )
            execution = resolve_provider_execution(config, "copilot")
            command = self.provider.build_command(config, execution)

        self.assertEqual(
            command,
            [
                "copilot",
                "--output-format",
                "json",
                "--autopilot",
                "--no-ask-user",
                "--allow-all-tools",
                "--no-custom-instructions",
                "--disable-builtin-mcps",
            ],
        )

    def test_build_command_appends_extra_provider_args(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config = _make_config(
                Path(tmp_dir),
                extra_args=("--effort", "high"),
            )
            execution = resolve_provider_execution(config, "copilot")
            command = self.provider.build_command(config, execution)

        self.assertEqual(
            command,
            [
                "copilot",
                "--output-format",
                "json",
                "--autopilot",
                "--no-ask-user",
                "--allow-all",
                "--model",
                "gpt-5.2",
                "--max-autopilot-continues",
                "7",
                "--effort",
                "high",
            ],
        )

    def test_validate_config_rejects_non_stream_json_mode(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config = _make_config(Path(tmp_dir), output_format=OutputFormat.JSON)

            with self.assertRaisesRegex(ValueError, "stream-json"):
                self.provider.validate_config(
                    config,
                    resolve_provider_execution(config, "copilot"),
                )

    def test_extract_cost_reads_nested_usage_cost_when_present(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "iteration-000001.json"
            log_path.write_text(
                "\n".join(
                    [
                        json.dumps({"type": "turn.started"}),
                        json.dumps({"type": "turn.completed", "usage": {"total_cost_usd": "0.42"}}),
                    ]
                ),
                encoding="utf-8",
            )

            cost = self.provider.extract_cost(log_path, OutputFormat.STREAM_JSON)

        self.assertEqual(cost, Decimal("0.42"))

    def test_classify_auth_failure_from_plain_text_output(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            log_path = temp_root / "iteration-000001.json"
            log_path.write_text(
                "\n".join(
                    [
                        "Error: No authentication information found.",
                        "",
                        "Copilot can be authenticated with GitHub using an OAuth Token or a Fine-Grained Personal Access Token.",
                    ]
                ),
                encoding="utf-8",
            )
            config = _make_config(temp_root)

            decision = self.provider.classify_failure(
                exit_code=1,
                log_path=log_path,
                config=config,
                execution=resolve_provider_execution(config, "copilot"),
            )

        self.assertTrue(decision.fatal)
        self.assertEqual(decision.kind, FailureKind.AUTH)
        self.assertTrue(decision.should_failover)

    def test_classify_rate_limit_from_structured_output(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            log_path = temp_root / "iteration-000001.json"
            log_path.write_text(
                json.dumps(
                    {
                        "type": "error",
                        "message": "Too Many Requests: Sorry, you've exhausted this model's rate limit.",
                    }
                ),
                encoding="utf-8",
            )
            config = _make_config(temp_root)

            decision = self.provider.classify_failure(
                exit_code=1,
                log_path=log_path,
                config=config,
                execution=resolve_provider_execution(config, "copilot"),
            )

        self.assertEqual(decision.kind, FailureKind.RATE_LIMIT)
        self.assertEqual(decision.wait_seconds, 1800)
        self.assertTrue(decision.reset_error_count)
        self.assertTrue(decision.skip_pause)
        self.assertTrue(decision.should_failover)

    def test_classify_invalid_request_from_exit_code_2(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            log_path = temp_root / "iteration-000001.json"
            log_path.write_text("unknown option '--bogus-flag'", encoding="utf-8")
            config = _make_config(temp_root)

            decision = self.provider.classify_failure(
                exit_code=2,
                log_path=log_path,
                config=config,
                execution=resolve_provider_execution(config, "copilot"),
            )

        self.assertTrue(decision.fatal)
        self.assertEqual(decision.kind, FailureKind.INVALID_REQUEST)
        self.assertTrue(decision.should_failover)


def _make_config(
    temp_root: Path,
    *,
    safe_mode: bool = False,
    use_bare: bool = False,
    model: str | None = "gpt-5.2",
    output_format: OutputFormat = OutputFormat.STREAM_JSON,
    max_turns: int | None = 7,
    extra_args: tuple[str, ...] = (),
) -> RunnerConfig:
    prompt_path = temp_root / "PROMPT.md"
    prompt_path.write_text("prompt", encoding="utf-8")
    return RunnerConfig(
        working_dir=temp_root,
        provider_names=("copilot",),
        provider_profiles={
            "copilot": ProviderProfile(
                model=model,
                max_turns=max_turns,
                use_bare=use_bare,
                safe_mode=safe_mode,
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
