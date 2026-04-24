from __future__ import annotations

import json
import unittest
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from ralph.config import OutputFormat, PromptSpec, RunnerConfig
from ralph.providers import ClaudeProvider


class ClaudeProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = ClaudeProvider()

    def test_build_command_matches_claude_flags(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config = _make_config(Path(tmp_dir))
            command = self.provider.build_command(config)

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
            decision = self.provider.classify_failure(
                exit_code=1,
                log_path=log_path,
                config=_make_config(temp_root),
            )

        self.assertFalse(decision.fatal)
        self.assertEqual(decision.wait_seconds, 1800)
        self.assertTrue(decision.reset_error_count)
        self.assertTrue(decision.skip_pause)


def _make_config(temp_root: Path) -> RunnerConfig:
    prompt_path = temp_root / "PROMPT.md"
    prompt_path.write_text("prompt", encoding="utf-8")
    return RunnerConfig(
        working_dir=temp_root,
        provider_name="claude",
        provider_binary=None,
        prompt_specs=(PromptSpec(path=prompt_path, repeat=1),),
        prompt_sequence=(prompt_path,),
        max_iterations=0,
        max_cost=Decimal("0"),
        max_duration_hours=Decimal("0"),
        iteration_timeout_minutes=Decimal("0"),
        pause_seconds=5,
        model="sonnet",
        wait_on_limit_mins=30,
        max_consecutive_errors=5,
        max_turns=7,
        log_dir=temp_root / "logs",
        log_retain=0,
        check_commands=(),
        stop_on_regexes=(),
        stop_on_clean_git=False,
        stop_when_files=(),
        output_format=OutputFormat.STREAM_JSON,
        use_bare=False,
        safe_mode=False,
        resume_from=None,
        resume_note=None,
        dry_run=False,
    )


if __name__ == "__main__":
    unittest.main()
