from __future__ import annotations

import json
import shlex
import subprocess
import sys
import unittest
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from batonloop.config import (
    OutputFormat,
    PromptSpec,
    ProviderExecution,
    ProviderProfile,
    RunnerConfig,
)
from batonloop.handoff import extract_handoff_summary, metadata_path_for, prompt_artifact_path_for
from batonloop.providers.base import FailureDecision, FailureKind
from batonloop.runner import run_loop


class RunnerTests(unittest.TestCase):
    def test_iteration_timeout_stops_hung_process(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            config = _make_config(
                temp_root,
                iteration_timeout_minutes=Decimal("0.001"),
                max_consecutive_errors=1,
            )
            provider = StaticCommandProvider(
                [sys.executable, "-c", "import time; time.sleep(1)"],
            )

            exit_code = run_loop(config, {"fake": provider})

            self.assertEqual(exit_code, 1)
            log_text = (config.log_dir / "batonloop.log").read_text(encoding="utf-8")
            self.assertIn("timed out", log_text.lower())

    def test_check_commands_stop_after_all_pass(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            config = _make_config(
                temp_root,
                check_commands=(f"{shlex.quote(sys.executable)} -c 'import sys; sys.exit(0)'",),
            )
            provider = StaticCommandProvider(
                [sys.executable, "-c", "print('work complete')"],
            )

            exit_code = run_loop(config, {"fake": provider})

            self.assertEqual(exit_code, 0)
            self.assertTrue((config.log_dir / "iteration-000001-check-01.log").is_file())
            log_text = (config.log_dir / "batonloop.log").read_text(encoding="utf-8")
            self.assertIn("All post-iteration checks passed", log_text)

    def test_stop_on_regex_matches_iteration_log(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            config = _make_config(temp_root, stop_on_regexes=("DONE",))
            provider = StaticCommandProvider([sys.executable, "-c", "print('DONE')"])

            exit_code = run_loop(config, {"fake": provider})

            self.assertEqual(exit_code, 0)
            log_text = (config.log_dir / "batonloop.log").read_text(encoding="utf-8")
            self.assertIn("Stop regex matched iteration output", log_text)

    def test_stop_when_file_detects_marker(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            marker_path = temp_root / "DONE.flag"
            config = _make_config(temp_root, stop_when_files=(marker_path,))
            provider = StaticCommandProvider(
                [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; Path('DONE.flag').touch(); print('flag written')",
                ],
            )

            exit_code = run_loop(config, {"fake": provider})

            self.assertEqual(exit_code, 0)
            self.assertTrue(marker_path.exists())
            log_text = (config.log_dir / "batonloop.log").read_text(encoding="utf-8")
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
            provider = StaticCommandProvider([sys.executable, "-c", "print('no repo changes')"])

            exit_code = run_loop(config, {"fake": provider})

            self.assertEqual(exit_code, 0)
            log_text = (config.log_dir / "batonloop.log").read_text(encoding="utf-8")
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
                        "handoff_summary": (
                            "Previous iteration summary:\n"
                            "- Goal: Resume the interrupted task.\n"
                            "- Interruption: RATE LIMITED."
                        ),
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
            provider = StaticCommandProvider([sys.executable, "-c", "import sys; print(sys.stdin.read())"])

            exit_code = run_loop(config, {"fake": provider})

            self.assertEqual(exit_code, 0)
            current_log = config.log_dir / "iteration-000001.json"
            prompt_artifact = prompt_artifact_path_for(current_log)
            log_text = current_log.read_text(encoding="utf-8")
            self.assertTrue(prompt_artifact.is_file())
            self.assertIn("=== BATONLOOP RESUME CONTEXT ===", log_text)
            self.assertIn(f"Previous raw log: {latest_log}", log_text)
            self.assertIn("Previous provider: claude", log_text)
            self.assertIn("Previous exit code: 1", log_text)
            self.assertIn("Previous iteration summary:", log_text)
            self.assertIn("Goal: Resume the interrupted task.", log_text)
            self.assertIn(
                "Operator note: Switch providers and continue from the partial work.",
                log_text,
            )

    def test_extract_handoff_summary_from_claude_log(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "iteration-000001.json"
            log_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "type": "assistant",
                                "message": {
                                    "role": "assistant",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": (
                                                "Now I have the picture. Phase 4.3 is the next "
                                                "recommended task - wire weapon `Atk` / `Dmg` via "
                                                "the `target-pick` tool to `encounterCombat.attemptAttack`."
                                            ),
                                        }
                                    ],
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "assistant",
                                "message": {
                                    "role": "assistant",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": "Lint, typecheck, and test all pass. Now update the roadmap and agent notes.",
                                        }
                                    ],
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "system",
                                "subtype": "task_started",
                                "description": "Critical review of Phase 4.3a",
                                "prompt": (
                                    "You are an independent code reviewer. Give a critical review "
                                    "of commit `5fed429` on the current branch. Focus areas: "
                                    "correctness of the new hook and drawer wiring."
                                ),
                            }
                        ),
                        json.dumps(
                            {
                                "type": "user",
                                "message": {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": "You've hit your limit - resets 7am.",
                                        }
                                    ],
                                },
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            summary = extract_handoff_summary(log_path, provider_hint="claude")

            assert summary is not None
            self.assertIn("Previous iteration summary:", summary)
            self.assertIn("Goal: Now I have the picture. Phase 4.3 is the next recommended task", summary)
            self.assertIn("Progress checkpoint: Lint, typecheck, and test all pass.", summary)
            self.assertIn("In-flight task: Critical review of Phase 4.3a:", summary)
            self.assertIn("Interruption: You've hit your limit - resets 7am.", summary)
            self.assertLessEqual(len(summary.split()), 120)

    def test_extract_handoff_summary_from_codex_log(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "iteration-000002.json"
            log_path.write_text(
                "\n".join(
                    [
                        "Reading prompt from stdin...",
                        json.dumps(
                            {
                                "type": "item.completed",
                                "item": {
                                    "type": "agent_message",
                                    "text": (
                                        "I have enough context to work the actual 5.0 slice. "
                                        "The plan is: add `areaOfEffect` to the shared spell "
                                        "contract and Prisma model, seed it from a curated "
                                        "override map, add coverage, then update notes."
                                    ),
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "item.completed",
                                "item": {
                                    "type": "agent_message",
                                    "text": (
                                        "Lint and typecheck are clean. I am making one attempt "
                                        "at the repo's normal test path as the last gate."
                                    ),
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "item.updated",
                                "item": {
                                    "type": "todo_list",
                                    "items": [
                                        {"text": "Implement schema and seed changes", "completed": True},
                                        {"text": "Add targeted tests", "completed": True},
                                        {
                                            "text": "Run verification, update notes, review, and commit",
                                            "completed": False,
                                        },
                                    ],
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "item.started",
                                "item": {
                                    "type": "collab_tool_call",
                                    "tool": "spawn_agent",
                                    "prompt": (
                                        "Review the current uncommitted diff critically. Focus on "
                                        "bugs, behavioral regressions, schema mismatches, and "
                                        "whether the updated roadmap notes match what landed."
                                    ),
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "turn.failed",
                                "error": {
                                    "message": "You've hit your usage limit. Try again later."
                                },
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            summary = extract_handoff_summary(log_path, provider_hint="codex")

            assert summary is not None
            self.assertIn("Previous iteration summary:", summary)
            self.assertIn("Goal: I have enough context to work the actual 5.0 slice.", summary)
            self.assertIn("Progress checkpoint: Lint and typecheck are clean.", summary)
            self.assertIn(
                "Checklist: 2/3 complete; remaining: Run verification, update notes, review, and commit",
                summary,
            )
            self.assertIn("In-flight task: spawn_agent: Review the current uncommitted diff critically.", summary)
            self.assertIn("Interruption: You've hit your usage limit. Try again later.", summary)
            self.assertLessEqual(len(summary.split()), 140)

    def test_failure_iteration_writes_metadata_for_future_resume(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            config = _make_config(temp_root, max_consecutive_errors=1, max_iterations=1)
            provider = StaticCommandProvider(
                [sys.executable, "-c", "import sys; print('boom'); sys.exit(1)"],
            )

            exit_code = run_loop(config, {"fake": provider})

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
            self.assertEqual(metadata["failover_target_provider"], None)
            self.assertEqual(metadata["handoff_summary"], None)
            self.assertIn("git_status", metadata)

    def test_rate_limit_failure_automatically_fails_over_to_next_provider(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            config = _make_config(
                temp_root,
                max_iterations=1,
                provider_names=("limited", "backup"),
                provider_profiles={"backup": ProviderProfile(model="gpt-5.4")},
            )
            providers = {
                "limited": RateLimitedProvider(),
                "backup": ResumeEchoProvider(),
            }

            exit_code = run_loop(config, providers)

            self.assertEqual(exit_code, 0)
            first_iteration_metadata = json.loads(
                metadata_path_for(config.log_dir / "iteration-000001.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(first_iteration_metadata["provider_name"], "limited")
            self.assertEqual(first_iteration_metadata["failover_target_provider"], "backup")

            second_iteration_log = config.log_dir / "iteration-000002.json"
            second_prompt_artifact = prompt_artifact_path_for(second_iteration_log)
            second_log_text = second_iteration_log.read_text(encoding="utf-8")
            self.assertTrue(second_prompt_artifact.is_file())
            self.assertIn("MODEL=gpt-5.4", second_log_text)
            self.assertIn("=== BATONLOOP RESUME CONTEXT ===", second_log_text)
            self.assertIn(
                f"Previous raw log: {config.log_dir / 'iteration-000001.json'}",
                second_log_text,
            )
            self.assertIn("Current provider: backup", second_log_text)
            self.assertIn("Previous provider: limited", second_log_text)
            self.assertIn("Previous iteration summary:", second_log_text)


class StaticCommandProvider:
    name = "fake"

    def __init__(
        self,
        command: list[str],
        *,
        failure_decision: FailureDecision | None = None,
    ) -> None:
        self._command = command
        self._failure_decision = failure_decision

    def executable_name(self, execution: ProviderExecution) -> str:
        del execution
        return self._command[0]

    def validate_config(self, config: RunnerConfig, execution: ProviderExecution) -> None:
        del config, execution

    def build_command(self, config: RunnerConfig, execution: ProviderExecution) -> list[str]:
        del config, execution
        return self._command

    def extract_cost(self, log_path: Path, output_format: OutputFormat) -> Decimal:
        del log_path, output_format
        return Decimal("0")

    def classify_failure(
        self,
        exit_code: int,
        log_path: Path,
        config: RunnerConfig,
        execution: ProviderExecution,
    ) -> FailureDecision:
        del log_path, config, execution
        if self._failure_decision is not None:
            return self._failure_decision
        return FailureDecision(message=f"process failed with exit code {exit_code}")


class RateLimitedProvider:
    name = "limited"

    def executable_name(self, execution: ProviderExecution) -> str:
        del execution
        return sys.executable

    def validate_config(self, config: RunnerConfig, execution: ProviderExecution) -> None:
        del config, execution

    def build_command(self, config: RunnerConfig, execution: ProviderExecution) -> list[str]:
        del config, execution
        return [sys.executable, "-c", "import sys; print('rate limited'); sys.exit(1)"]

    def extract_cost(self, log_path: Path, output_format: OutputFormat) -> Decimal:
        del log_path, output_format
        return Decimal("0")

    def classify_failure(
        self,
        exit_code: int,
        log_path: Path,
        config: RunnerConfig,
        execution: ProviderExecution,
    ) -> FailureDecision:
        del exit_code, log_path, config, execution
        return FailureDecision(
            message="RATE LIMITED detected in output. Waiting 30 minutes before retrying...",
            kind=FailureKind.RATE_LIMIT,
            wait_seconds=1800,
            reset_error_count=True,
            skip_pause=True,
            should_failover=True,
        )


class ResumeEchoProvider:
    name = "backup"

    def executable_name(self, execution: ProviderExecution) -> str:
        del execution
        return sys.executable

    def validate_config(self, config: RunnerConfig, execution: ProviderExecution) -> None:
        del config, execution

    def build_command(self, config: RunnerConfig, execution: ProviderExecution) -> list[str]:
        del config
        return [
            sys.executable,
            "-c",
            "import sys; print('MODEL=' + sys.argv[1]); print(sys.stdin.read())",
            execution.model or "none",
        ]

    def extract_cost(self, log_path: Path, output_format: OutputFormat) -> Decimal:
        del log_path, output_format
        return Decimal("0")

    def classify_failure(
        self,
        exit_code: int,
        log_path: Path,
        config: RunnerConfig,
        execution: ProviderExecution,
    ) -> FailureDecision:
        del exit_code, log_path, config, execution
        return FailureDecision(message="unexpected failure")


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
    provider_names: tuple[str, ...] = ("fake",),
    provider_profiles: dict[str, ProviderProfile] | None = None,
) -> RunnerConfig:
    prompt_path = temp_root / "PROMPT.md"
    prompt_path.write_text("prompt", encoding="utf-8")
    return RunnerConfig(
        working_dir=temp_root,
        provider_names=provider_names,
        provider_profiles=provider_profiles or {},
        provider_config_path=None,
        default_provider_profile=ProviderProfile(),
        prompt_specs=(PromptSpec(path=prompt_path, repeat=1),),
        prompt_sequence=(prompt_path,),
        max_iterations=max_iterations,
        max_cost=Decimal("0"),
        max_duration_hours=Decimal("0"),
        iteration_timeout_minutes=iteration_timeout_minutes,
        pause_seconds=0,
        wait_on_limit_mins=30,
        max_consecutive_errors=max_consecutive_errors,
        log_dir=temp_root / "batonloop-logs",
        log_retain=0,
        check_commands=check_commands,
        stop_on_regexes=stop_on_regexes,
        stop_on_clean_git=stop_on_clean_git,
        stop_when_files=stop_when_files,
        output_format=OutputFormat.STREAM_JSON,
        resume_from=resume_from,
        resume_note=resume_note,
        dry_run=False,
    )


if __name__ == "__main__":
    unittest.main()
