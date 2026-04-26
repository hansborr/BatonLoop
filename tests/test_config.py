from __future__ import annotations

import unittest
from argparse import Namespace
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from batonloop.config import OutputFormat, build_config, parse_prompt_spec, resolve_provider_execution


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
            provider_config_path = temp_root / "batonloop-providers.toml"
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
                        'args = ["--profile", "baton", "--sandbox", "workspace-write"]',
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
                    retry_backoff_base_seconds=10,
                    retry_backoff_multiplier=Decimal("3"),
                    retry_backoff_max_seconds=120,
                    retry_jitter_fraction=Decimal("0.25"),
                    provider_cooldown_seconds=300,
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
                    live_output=False,
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
            self.assertFalse(config.live_output)
            self.assertEqual(config.iteration_timeout_minutes, Decimal("0"))
            self.assertEqual(config.check_commands, ("pytest -q",))
            self.assertEqual(config.stop_on_regexes, ("DONE",))
            self.assertEqual(config.stop_when_files, (temp_root / "done.flag",))
            self.assertEqual(config.retry_backoff_base_seconds, 10)
            self.assertEqual(config.retry_backoff_multiplier, Decimal("3"))
            self.assertEqual(config.retry_backoff_max_seconds, 120)
            self.assertEqual(config.retry_jitter_fraction, Decimal("0.25"))
            self.assertEqual(config.provider_cooldown_seconds, 300)
            self.assertEqual(config.provider_names, ("claude", "codex"))
            self.assertEqual(config.provider_config_path, provider_config_path)
            self.assertEqual(config.resume_from, temp_root / "old-logs")
            self.assertEqual(config.resume_note, "resume after usage limit")
            self.assertEqual(resolve_provider_execution(config, "claude").model, "opus")
            self.assertEqual(resolve_provider_execution(config, "codex").model, "gpt-5.4")
            self.assertTrue(resolve_provider_execution(config, "codex").safe_mode)
            self.assertEqual(
                resolve_provider_execution(config, "codex").extra_args,
                ("--profile", "baton", "--sandbox", "workspace-write"),
            )

    def test_resume_flag_uses_configured_log_dir(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            prompt_path = temp_root / "PROMPT.md"
            prompt_path.write_text("develop", encoding="utf-8")

            config = build_config(
                _make_args(
                    prompt_specs=[str(prompt_path)],
                    log_dir=str(temp_root / "logs"),
                    resume_latest=True,
                )
            )

            self.assertEqual(config.log_dir, temp_root / "logs")
            self.assertEqual(config.resume_from, temp_root / "logs")

    def test_resume_flag_rejects_explicit_resume_from(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            prompt_path = temp_root / "PROMPT.md"
            prompt_path.write_text("develop", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "--resume cannot be combined"):
                build_config(
                    _make_args(
                        prompt_specs=[str(prompt_path)],
                        log_dir=str(temp_root / "logs"),
                        resume_latest=True,
                        resume_from=str(temp_root / "old-logs"),
                    )
                )

    def test_build_config_loads_run_config_toml(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            prompt_path = temp_root / "PROMPT.md"
            prompt_path.write_text("develop", encoding="utf-8")
            run_config_path = temp_root / "batonloop.toml"
            run_config_path.write_text(
                "\n".join(
                    [
                        "[run]",
                        'providers = ["claude", "codex", "copilot"]',
                        'prompt_files = ["./PROMPT.md"]',
                        "iterations = 20",
                        "iteration_timeout = 30",
                        "pause = 10",
                        "max_errors = 4",
                        "wait_on_limit = 30",
                        "retry_backoff_base = 30",
                        'retry_backoff_multiplier = "2"',
                        "retry_backoff_max = 600",
                        'retry_jitter = "0.2"',
                        "provider_cooldown = 1800",
                        'checks = ["pytest -q"]',
                        "safe = true",
                        "",
                        "[providers.codex]",
                        'model = "gpt-5.4"',
                        'args = ["--profile", "baton"]',
                    ]
                ),
                encoding="utf-8",
            )

            with patch("pathlib.Path.cwd", return_value=temp_root):
                config = build_config(_make_args())

            self.assertEqual(config.run_config_path, run_config_path)
            self.assertEqual(config.provider_names, ("claude", "codex", "copilot"))
            self.assertEqual(config.prompt_sequence, (prompt_path,))
            self.assertEqual(config.max_iterations, 20)
            self.assertEqual(config.iteration_timeout_minutes, Decimal("30"))
            self.assertEqual(config.pause_seconds, 10)
            self.assertEqual(config.max_consecutive_errors, 4)
            self.assertEqual(config.retry_backoff_base_seconds, 30)
            self.assertEqual(config.retry_backoff_multiplier, Decimal("2"))
            self.assertEqual(config.retry_backoff_max_seconds, 600)
            self.assertEqual(config.retry_jitter_fraction, Decimal("0.2"))
            self.assertEqual(config.provider_cooldown_seconds, 1800)
            self.assertEqual(config.check_commands, ("pytest -q",))
            self.assertTrue(resolve_provider_execution(config, "claude").safe_mode)
            self.assertEqual(resolve_provider_execution(config, "codex").model, "gpt-5.4")
            self.assertEqual(resolve_provider_execution(config, "codex").extra_args, ("--profile", "baton"))

    def test_cli_values_override_run_config_toml(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            prompt_path = temp_root / "PROMPT.md"
            prompt_path.write_text("develop", encoding="utf-8")
            (temp_root / "batonloop.toml").write_text(
                "\n".join(
                    [
                        "[run]",
                        'providers = ["claude"]',
                        "iterations = 20",
                        "safe = true",
                        'checks = ["pytest -q"]',
                    ]
                ),
                encoding="utf-8",
            )

            with patch("pathlib.Path.cwd", return_value=temp_root):
                config = build_config(
                    _make_args(
                        provider_names=["codex"],
                        max_iterations=3,
                        safe=False,
                        check_commands=["python -m unittest"],
                    )
                )

            self.assertEqual(config.provider_names, ("codex",))
            self.assertEqual(config.max_iterations, 3)
            self.assertFalse(resolve_provider_execution(config, "codex").safe_mode)
            self.assertEqual(config.check_commands, ("python -m unittest",))

    def test_build_config_rejects_non_string_provider_args(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            prompt_path = temp_root / "PROMPT.md"
            provider_config_path = temp_root / "batonloop-providers.toml"
            prompt_path.write_text("develop", encoding="utf-8")
            provider_config_path.write_text(
                "\n".join(
                    [
                        "[providers.codex]",
                        'args = ["--profile", 7]',
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, r"\.args must be an array of strings"):
                build_config(
                    Namespace(
                        provider_names=["codex"],
                        provider_config=str(provider_config_path),
                        provider_binary=None,
                        prompt_specs=[str(prompt_path)],
                        max_iterations=0,
                        max_cost=Decimal("0"),
                        max_duration_hours=Decimal("0"),
                        iteration_timeout_minutes=Decimal("0"),
                        pause_seconds=5,
                        model=None,
                        wait_on_limit_mins=30,
                        retry_backoff_base_seconds=0,
                        retry_backoff_multiplier=Decimal("2"),
                        retry_backoff_max_seconds=0,
                        retry_jitter_fraction=Decimal("0"),
                        provider_cooldown_seconds=0,
                        max_consecutive_errors=5,
                        max_turns=None,
                        log_dir=str(temp_root / "logs"),
                        log_retain=0,
                        check_commands=[],
                        stop_on_regexes=[],
                        stop_on_clean_git=False,
                        stop_when_files=[],
                        output_format=OutputFormat.STREAM_JSON.value,
                        no_stream=False,
                        live_output=False,
                        bare=None,
                        safe=None,
                        resume_from=None,
                        resume_note=None,
                        dry_run=False,
                    )
                )


def _make_args(**overrides: object) -> Namespace:
    values = {
        "config": None,
        "provider_names": None,
        "provider_config": None,
        "provider_binary": None,
        "prompt_specs": None,
        "max_iterations": None,
        "max_cost": None,
        "max_duration_hours": None,
        "iteration_timeout_minutes": None,
        "pause_seconds": None,
        "model": None,
        "wait_on_limit_mins": None,
        "retry_backoff_base_seconds": None,
        "retry_backoff_multiplier": None,
        "retry_backoff_max_seconds": None,
        "retry_jitter_fraction": None,
        "provider_cooldown_seconds": None,
        "max_consecutive_errors": None,
        "max_turns": None,
        "log_dir": None,
        "log_retain": None,
        "check_commands": None,
        "stop_on_regexes": None,
        "stop_on_clean_git": None,
        "stop_when_files": None,
        "output_format": None,
        "no_stream": None,
        "live_output": None,
        "bare": None,
        "safe": None,
        "resume_latest": None,
        "resume_from": None,
        "resume_note": None,
        "dry_run": None,
    }
    values.update(overrides)
    return Namespace(**values)


if __name__ == "__main__":
    unittest.main()
