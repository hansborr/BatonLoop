from __future__ import annotations

import unittest
from argparse import Namespace
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from ralph.config import OutputFormat, build_config, parse_prompt_spec, resolve_provider_execution


class PromptSpecTests(unittest.TestCase):
    def test_parse_repeat_suffix(self) -> None:
        prompt_spec = parse_prompt_spec("REVIEW.md:3")
        self.assertEqual(prompt_spec.path, Path("REVIEW.md"))
        self.assertEqual(prompt_spec.repeat, 3)

    def test_colon_without_numeric_suffix_is_not_a_repeat(self) -> None:
        prompt_spec = parse_prompt_spec("notes:v2.md")
        self.assertEqual(prompt_spec.path, Path("notes:v2.md"))
        self.assertEqual(prompt_spec.repeat, 1)

    def test_build_config_expands_prompt_sequence(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            prompt_path = temp_root / "PROMPT.md"
            review_path = temp_root / "REVIEW.md"
            provider_config_path = temp_root / "ralph-providers.toml"
            prompt_path.write_text("develop", encoding="utf-8")
            review_path.write_text("review", encoding="utf-8")
            provider_config_path.write_text(
                "\n".join(
                    [
                        "[providers.claude]",
                        'model = "opus"',
                        "",
                        "[providers.codex]",
                        'model = "gpt-5.4"',
                        "safe = true",
                    ]
                ),
                encoding="utf-8",
            )

            config = build_config(
                Namespace(
                    provider_names=["claude", "codex"],
                    provider_config=str(provider_config_path),
                    provider_binary=None,
                    prompt_specs=[f"{prompt_path}:2", str(review_path)],
                    max_iterations=0,
                    max_cost=Decimal("0"),
                    max_duration_hours=Decimal("0"),
                    iteration_timeout_minutes=Decimal("0"),
                    pause_seconds=5,
                    model=None,
                    wait_on_limit_mins=30,
                    max_consecutive_errors=5,
                    max_turns=None,
                    log_dir=str(temp_root / "logs"),
                    log_retain=0,
                    check_commands=["pytest -q"],
                    stop_on_regexes=["DONE"],
                    stop_on_clean_git=False,
                    stop_when_files=[str(temp_root / "done.flag")],
                    output_format=OutputFormat.STREAM_JSON.value,
                    no_stream=False,
                    bare=None,
                    safe=None,
                    resume_from=str(temp_root / "old-logs"),
                    resume_note="resume after usage limit",
                    dry_run=False,
                )
            )

            self.assertEqual(
                config.prompt_sequence,
                (prompt_path, prompt_path, review_path),
            )
            self.assertEqual(config.output_format, OutputFormat.STREAM_JSON)
            self.assertEqual(config.iteration_timeout_minutes, Decimal("0"))
            self.assertEqual(config.check_commands, ("pytest -q",))
            self.assertEqual(config.stop_on_regexes, ("DONE",))
            self.assertEqual(config.stop_when_files, (temp_root / "done.flag",))
            self.assertEqual(config.provider_names, ("claude", "codex"))
            self.assertEqual(config.provider_config_path, provider_config_path)
            self.assertEqual(config.resume_from, temp_root / "old-logs")
            self.assertEqual(config.resume_note, "resume after usage limit")
            self.assertEqual(resolve_provider_execution(config, "claude").model, "opus")
            self.assertEqual(resolve_provider_execution(config, "codex").model, "gpt-5.4")
            self.assertTrue(resolve_provider_execution(config, "codex").safe_mode)


if __name__ == "__main__":
    unittest.main()
