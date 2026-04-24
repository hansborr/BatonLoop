from __future__ import annotations

import json
import shlex
import subprocess
import sys
import unittest
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from ralph.config import OutputFormat, PromptSpec, RunnerConfig
from ralph.handoff import metadata_path_for, prompt_artifact_path_for
from ralph.providers.base import FailureDecision
from ralph.runner import run_loop


class RunnerTests(unittest.TestCase):
    def test_iteration_timeout_stops_hung_process(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            config = _make_config(
                temp_root,
                iteration_timeout_minutes=Decimal("0.001"),
                max_consecutive_errors=1,
            )
            provider = FakeProvider(
                [sys.executable, "-c", "import time; time.sleep(1)"],
            )

            exit_code = run_loop(config, provider)

            self.assertEqual(exit_code, 1)
            log_text = (config.log_dir / "ralph.log").read_text(encoding="utf-8")
            self.assertIn("timed out", log_text.lower())

    def test_check_commands_stop_after_all_pass(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            config = _make_config(
                temp_root,
                check_commands=(f"{shlex.quote(sys.executable)} -c 'import sys; sys.exit(0)'",),
            )
            provider = FakeProvider(
                [sys.executable, "-c", "print('work complete')"],
            )

            exit_code = run_loop(config, provider)

            self.assertEqual(exit_code, 0)
            self.assertTrue((config.log_dir / "iteration-000001-check-01.log").is_file())
            log_text = (config.log_dir / "ralph.log").read_text(encoding="utf-8")
            self.assertIn("All post-iteration checks passed", log_text)

    def test_stop_on_regex_matches_iteration_log(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            config = _make_config(temp_root, stop_on_regexes=("DONE",))
            provider = FakeProvider([sys.executable, "-c", "print('DONE')"])

            exit_code = run_loop(config, provider)

            self.assertEqual(exit_code, 0)
            log_text = (config.log_dir / "ralph.log").read_text(encoding="utf-8")
            self.assertIn("Stop regex matched iteration output", log_text)

    def test_stop_when_file_detects_marker(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            marker_path = temp_root / "DONE.flag"
            config = _make_config(temp_root, stop_when_files=(marker_path,))
            provider = FakeProvider(
                [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; Path('DONE.flag').touch(); print('flag written')",
                ],
            )

            exit_code = run_loop(config, provider)

            self.assertEqual(exit_code, 0)
            self.assertTrue(marker_path.exists())
            log_text = (config.log_dir / "ralph.log").read_text(encoding="utf-8")
            self.assertIn("Stop file detected", log_text)

    def test_stop_on_clean_git_ignores_log_directory(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            subprocess.run(["git", "init", "-q"], cwd=temp_root, check=True)
            prompt_path = temp_root / "PROMPT.md"
            prompt_path.write_text("prompt", encoding="utf-8")
            subprocess.run(["git", "add", "PROMPT.md"], cwd=temp_root, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Test User",
                    "-c",
                    "user.email=test@example.com",
                    "commit",
                    "-qm",
                    "init",
                ],
                cwd=temp_root,
                check=True,
            )

            config = _make_config(temp_root, stop_on_clean_git=True)
            provider = FakeProvider([sys.executable, "-c", "print('no repo changes')"])

            exit_code = run_loop(config, provider)

            self.assertEqual(exit_code, 0)
            log_text = (config.log_dir / "ralph.log").read_text(encoding="utf-8")
            self.assertIn("Git worktree is clean", log_text)

    def test_resume_from_directory_uses_latest_iteration_log(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            previous_logs = temp_root / "previous-logs"
            previous_logs.mkdir()
            older_log = previous_logs / "iteration-000003.json"
            latest_log = previous_logs / "iteration-000014.json"
            older_log.write_text("older", encoding="utf-8")
            latest_log.write_text("latest", encoding="utf-8")
            metadata_path_for(latest_log).write_text(
                json.dumps(
                    {
                        "provider_name": "claude",
                        "base_prompt_path": str(temp_root / "OLD_PROMPT.md"),
                        "exit_code": 1,
                        "timed_out": False,
                        "failure_message": "RATE LIMITED detected in output.",
                    }
                ),
                encoding="utf-8",
            )

            config = _make_config(
                temp_root,
                max_iterations=1,
                resume_from=previous_logs,
                resume_note="Switch providers and continue from the partial work.",
            )
            provider = FakeProvider([sys.executable, "-c", "import sys; print(sys.stdin.read())"])

            exit_code = run_loop(config, provider)

            self.assertEqual(exit_code, 0)
            current_log = config.log_dir / "iteration-000001.json"
            prompt_artifact = prompt_artifact_path_for(current_log)
            log_text = current_log.read_text(encoding="utf-8")
            self.assertTrue(prompt_artifact.is_file())
            self.assertIn("=== RALPH RESUME CONTEXT ===", log_text)
            self.assertIn(f"Previous raw log: {latest_log}", log_text)
            self.assertIn("Previous provider: claude", log_text)
            self.assertIn("Previous exit code: 1", log_text)
            self.assertIn(
                "Operator note: Switch providers and continue from the partial work.",
                log_text,
            )

    def test_failure_iteration_writes_metadata_for_future_resume(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            config = _make_config(temp_root, max_consecutive_errors=1, max_iterations=1)
            provider = FakeProvider(
                [sys.executable, "-c", "import sys; print('boom'); sys.exit(1)"],
            )

            exit_code = run_loop(config, provider)

            self.assertEqual(exit_code, 1)
            metadata = json.loads(
                metadata_path_for(config.log_dir / "iteration-000001.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metadata["provider_name"], "fake")
            self.assertEqual(metadata["exit_code"], 1)
            self.assertFalse(metadata["success"])
            self.assertEqual(metadata["failure_message"], "process failed with exit code 1")
            self.assertEqual(metadata["resume_source_log_path"], None)
            self.assertIn("git_status", metadata)


class FakeProvider:
    name = "fake"

    def __init__(self, command: list[str]) -> None:
        self._command = command

    def executable_name(self, config: RunnerConfig) -> str:
        del config
        return self._command[0]

    def validate_config(self, config: RunnerConfig) -> None:
        del config

    def build_command(self, config: RunnerConfig) -> list[str]:
        del config
        return self._command

    def extract_cost(self, log_path: Path, output_format: OutputFormat) -> Decimal:
        del log_path, output_format
        return Decimal("0")

    def classify_failure(
        self,
        exit_code: int,
        log_path: Path,
        config: RunnerConfig,
    ) -> FailureDecision:
        del log_path, config
        return FailureDecision(message=f"process failed with exit code {exit_code}")


def _make_config(
    temp_root: Path,
    *,
    max_iterations: int = 0,
    iteration_timeout_minutes: Decimal = Decimal("0"),
    check_commands: tuple[str, ...] = (),
    stop_on_regexes: tuple[str, ...] = (),
    stop_on_clean_git: bool = False,
    stop_when_files: tuple[Path, ...] = (),
    max_consecutive_errors: int = 5,
    resume_from: Path | None = None,
    resume_note: str | None = None,
) -> RunnerConfig:
    prompt_path = temp_root / "PROMPT.md"
    prompt_path.write_text("prompt", encoding="utf-8")
    return RunnerConfig(
        working_dir=temp_root,
        provider_name="fake",
        provider_binary=None,
        prompt_specs=(PromptSpec(path=prompt_path, repeat=1),),
        prompt_sequence=(prompt_path,),
        max_iterations=max_iterations,
        max_cost=Decimal("0"),
        max_duration_hours=Decimal("0"),
        iteration_timeout_minutes=iteration_timeout_minutes,
        pause_seconds=0,
        model=None,
        wait_on_limit_mins=30,
        max_consecutive_errors=max_consecutive_errors,
        max_turns=None,
        log_dir=temp_root / "ralph-logs",
        log_retain=0,
        check_commands=check_commands,
        stop_on_regexes=stop_on_regexes,
        stop_on_clean_git=stop_on_clean_git,
        stop_when_files=stop_when_files,
        output_format=OutputFormat.STREAM_JSON,
        use_bare=False,
        safe_mode=False,
        resume_from=resume_from,
        resume_note=resume_note,
        dry_run=False,
    )


if __name__ == "__main__":
    unittest.main()
