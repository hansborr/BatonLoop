from __future__ import annotations

import json
import unittest
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from ralph.config import OutputFormat, PromptSpec, RunnerConfig
from ralph.providers import CodexProvider


class CodexProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = CodexProvider()

    def test_build_command_uses_bypass_mode_by_default(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config = _make_config(Path(tmp_dir))
            command = self.provider.build_command(config)

        self.assertEqual(
            command,
            [
                "codex",
                "exec",
                "--json",
                "--skip-git-repo-check",
                "-C",
                str(config.working_dir),
                "--dangerously-bypass-approvals-and-sandbox",
                "-m",
                "gpt-5",
            ],
        )

    def test_build_command_uses_safe_and_bare_flags_when_requested(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config = _make_config(
                Path(tmp_dir),
                safe_mode=True,
                use_bare=True,
                model=None,
            )
            command = self.provider.build_command(config)

        self.assertEqual(
            command,
            [
                "codex",
                "exec",
                "--json",
                "--skip-git-repo-check",
                "-C",
                str(config.working_dir),
                "--full-auto",
                "--ignore-user-config",
                "--ignore-rules",
            ],
        )

    def test_validate_config_rejects_unsupported_options(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            json_config = _make_config(Path(tmp_dir), output_format=OutputFormat.JSON)
            with self.assertRaisesRegex(ValueError, "stream-json"):
                self.provider.validate_config(json_config)

            turn_limited_config = _make_config(Path(tmp_dir), max_turns=7)
            with self.assertRaisesRegex(ValueError, "max-turns"):
                self.provider.validate_config(turn_limited_config)

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

    def test_classify_invalid_request_as_fatal(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            log_path = temp_root / "iteration-000001.json"
            log_path.write_text(
                '{"type":"error","message":"{\\"type\\":\\"error\\",\\"status\\":400,\\"error\\":{\\"type\\":\\"invalid_request_error\\"}}"}',
                encoding="utf-8",
            )

            decision = self.provider.classify_failure(
                exit_code=1,
                log_path=log_path,
                config=_make_config(temp_root),
            )

        self.assertTrue(decision.fatal)


def _make_config(
    temp_root: Path,
    *,
    safe_mode: bool = False,
    use_bare: bool = False,
    model: str | None = "gpt-5",
    output_format: OutputFormat = OutputFormat.STREAM_JSON,
    max_turns: int | None = None,
) -> RunnerConfig:
    prompt_path = temp_root / "PROMPT.md"
    prompt_path.write_text("prompt", encoding="utf-8")
    return RunnerConfig(
        working_dir=temp_root,
        provider_name="codex",
        provider_binary=None,
        prompt_specs=(PromptSpec(path=prompt_path, repeat=1),),
        prompt_sequence=(prompt_path,),
        max_iterations=0,
        max_cost=Decimal("0"),
        max_duration_hours=Decimal("0"),
        iteration_timeout_minutes=Decimal("0"),
        pause_seconds=5,
        model=model,
        wait_on_limit_mins=30,
        max_consecutive_errors=5,
        max_turns=max_turns,
        log_dir=temp_root / "logs",
        log_retain=0,
        check_commands=(),
        stop_on_regexes=(),
        stop_on_clean_git=False,
        stop_when_files=(),
        output_format=output_format,
        use_bare=use_bare,
        safe_mode=safe_mode,
        dry_run=False,
    )



if __name__ == "__main__":
    unittest.main()
