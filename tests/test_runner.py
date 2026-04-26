from __future__ import annotations

import json
import logging
import shlex
import subprocess
import sys
import unittest
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from batonloop.config import (
    OutputFormat,
    PromptSpec,
    ProviderExecution,
    ProviderProfile,
    ProviderStrategy,
    RunnerConfig,
)
from batonloop.handoff import (
    build_resume_prompt,
    extract_handoff_details,
    extract_handoff_summary,
    metadata_path_for,
    prompt_artifact_path_for,
    resolve_resume_context,
)
from batonloop.providers.base import FailureDecision, FailureKind
from batonloop.runner import (
    CommandResult,
    IterationExecution,
    ProviderSlot,
    RunState,
    StopController,
    _handle_iteration_outcome,
    run_loop,
)


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

    def test_live_output_logs_filtered_provider_messages_by_default(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            config = _make_config(temp_root, max_iterations=1)
            provider = StaticCommandProvider(
                [
                    sys.executable,
                    "-c",
                    (
                        "import json; "
                        "print('Reading prompt from stdin...'); "
                        "print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':'Planning the refactor.'}})); "
                        "print(json.dumps({'type':'item.updated','item':{'type':'todo_list','items':[{'text':'Implement live output','completed':True},{'text':'Run tests','completed':False}]}}))"
                    ),
                ],
            )

            exit_code = run_loop(config, {"fake": provider})

            self.assertEqual(exit_code, 0)
            log_text = (config.log_dir / "batonloop.log").read_text(encoding="utf-8")
            self.assertIn("[fake] Planning the refactor.", log_text)
            self.assertIn(
                "[fake] Checklist: 1/2 complete; remaining: Run tests",
                log_text,
            )
            self.assertNotIn("Reading prompt from stdin...", log_text)

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
                        "handoff_extractor_version": 2,
                        "base_prompt_path": str(temp_root / "OLD_PROMPT.md"),
                        "exit_code": 1,
                        "timed_out": False,
                        "failure_message": "RATE LIMITED detected in output.",
                        "handoff_summary": (
                            "Previous iteration summary:\n"
                            "- State: Resume the interrupted task.\n"
                            "- Interruption: RATE LIMITED."
                        ),
                        "retry_recommended_next_step": "Resume the interrupted task.",
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
            self.assertIn("State: Resume the interrupted task.", log_text)
            self.assertIn("Recommended resume point: Resume the interrupted task.", log_text)
            self.assertIn(
                "Operator note: Switch providers and continue from the partial work.",
                log_text,
            )

    def test_resume_from_own_log_dir_continues_iteration_numbering(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            log_dir = temp_root / "batonloop-logs"
            log_dir.mkdir()
            previous_log = log_dir / "iteration-000003.json"
            previous_log.write_text(
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "agent_message",
                            "text": "Continue from the cancelled edit.",
                        },
                    }
                ),
                encoding="utf-8",
            )
            config = _make_config(temp_root, max_iterations=1, resume_from=log_dir)
            provider = StaticCommandProvider(
                [sys.executable, "-c", "import sys; print(sys.stdin.read())"]
            )

            exit_code = run_loop(config, {"fake": provider})

            self.assertEqual(exit_code, 0)
            self.assertTrue(previous_log.is_file())
            self.assertFalse((log_dir / "iteration-000001.json").exists())
            current_log = log_dir / "iteration-000004.json"
            self.assertTrue(current_log.is_file())
            log_text = current_log.read_text(encoding="utf-8")
            self.assertIn(f"Previous raw log: {previous_log}", log_text)
            self.assertIn("State: Continue from the cancelled edit.", log_text)

    def test_resume_note_is_only_added_to_explicit_resume_attempt(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            previous_logs = temp_root / "previous-logs"
            previous_logs.mkdir()
            previous_log = previous_logs / "iteration-000001.json"
            previous_log.write_text(
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "agent_message",
                            "text": "Continue the cancelled provider run.",
                        },
                    }
                ),
                encoding="utf-8",
            )
            config = _make_config(
                temp_root,
                max_iterations=2,
                max_consecutive_errors=5,
                resume_from=previous_logs,
                resume_note="Previous iteration was force-cancelled.",
            )
            provider = SequencedCommandProvider(
                (
                    [
                        sys.executable,
                        "-c",
                        (
                            "import json, sys; "
                            "print(json.dumps({'type':'item.completed','item':"
                            "{'type':'agent_message','text':sys.stdin.read()}})); "
                            "sys.exit(1)"
                        ),
                    ],
                    [sys.executable, "-c", "import sys; print(sys.stdin.read())"],
                )
            )

            exit_code = run_loop(config, {"fake": provider})

            self.assertEqual(exit_code, 0)
            first_prompt = prompt_artifact_path_for(config.log_dir / "iteration-000001.json")
            second_prompt = prompt_artifact_path_for(config.log_dir / "iteration-000002.json")
            self.assertIn(
                "Operator note: Previous iteration was force-cancelled.",
                first_prompt.read_text(encoding="utf-8"),
            )
            second_prompt_text = second_prompt.read_text(encoding="utf-8")
            self.assertNotIn(
                "Operator note: Previous iteration was force-cancelled.",
                second_prompt_text,
            )
            self.assertNotIn("Previous iteration was force-cancelled.", second_prompt_text)
            self.assertIn(
                f"Previous raw log: {config.log_dir / 'iteration-000001.json'}",
                second_prompt_text,
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
            self.assertIn("State: Now I have the picture. Phase 4.3 is the next recommended task", summary)
            self.assertIn("Progress checkpoint: Lint, typecheck, and test all pass.", summary)
            self.assertIn("In-flight task: Critical review of Phase 4.3a:", summary)
            self.assertIn("Interruption: You've hit your limit - resets 7am.", summary)
            self.assertLessEqual(len(summary.split()), 120)

    def test_extract_handoff_summary_from_successful_claude_result_log(self) -> None:
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
                                            "text": "Phase 5.0 is the next recommended task.",
                                        }
                                    ],
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "result",
                                "subtype": "success",
                                "is_error": False,
                                "result": "Completed successfully",
                                "total_cost_usd": 0.42,
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            summary = extract_handoff_summary(log_path, provider_hint="claude")

            assert summary is not None
            self.assertIn("State: Phase 5.0 is the next recommended task.", summary)
            self.assertNotIn("Completed successfully", summary)
            self.assertNotIn("Interruption:", summary)

    def test_extract_handoff_summary_from_claude_json_log(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "iteration-000001.json"
            log_path.write_text(
                json.dumps(
                    {
                        "type": "result",
                        "subtype": "success",
                        "is_error": False,
                        "result": (
                            "Phase 5.0 is the next recommended task. "
                            "Lint and tests pass."
                        ),
                        "total_cost_usd": 0.03,
                        "session_id": "abc123",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            summary = extract_handoff_summary(log_path, provider_hint="claude")

            assert summary is not None
            self.assertIn("Previous iteration summary:", summary)
            self.assertIn("State: Phase 5.0 is the next recommended task.", summary)
            self.assertNotIn("Interruption:", summary)

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
            self.assertIn("State: I have enough context to work the actual 5.0 slice.", summary)
            self.assertIn("Progress checkpoint: Lint and typecheck are clean.", summary)
            self.assertIn(
                "Checklist: 2/3 complete; remaining: Run verification, update notes, review, and commit",
                summary,
            )
            self.assertIn("In-flight task: spawn_agent: Review the current uncommitted diff critically.", summary)
            self.assertIn("Interruption: You've hit your usage limit. Try again later.", summary)
            self.assertLessEqual(len(summary.split()), 140)

    def test_extract_handoff_summary_streams_jsonl_logs(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "iteration-000003.json"
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
                                            "text": "Phase 5.1 is the next recommended task.",
                                        }
                                    ],
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "turn.failed",
                                "error": {"message": "You've hit your usage limit."},
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            with patch("pathlib.Path.read_text", side_effect=AssertionError("unexpected read_text")):
                summary = extract_handoff_summary(log_path, provider_hint="claude")

            assert summary is not None
            self.assertIn("State: Phase 5.1 is the next recommended task.", summary)
            self.assertIn("Interruption: You've hit your usage limit.", summary)

    def test_extract_handoff_summary_prefers_late_actionable_recovery_notes(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "iteration-000004.json"
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
                                            "text": "Now I have enough context. Let me start implementing.",
                                        }
                                    ],
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "system",
                                "subtype": "task_started",
                                "description": "Critical review of Phase 5.4",
                                "prompt": "Review the uncommitted diff and flag correctness blockers.",
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
                                            "text": (
                                                "Reviewer flagged two blockers: guard "
                                                "`useCastPlacement.dispatch` while `map.get` is "
                                                "unresolved and clear template/target-pick mode "
                                                "when the strip unwinds."
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
                                            "text": "Now I'll apply the two fixes.",
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
            self.assertIn("State: Reviewer flagged two blockers:", summary)
            self.assertIn("useCastPlacement.dispatch", summary)
            self.assertIn("Last activity: Now I'll apply the two fixes.", summary)
            self.assertNotIn("State: Now I have enough context.", summary)

    def test_extract_handoff_summary_prefers_shipped_state_over_filler_progress(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "iteration-000002.json"
            log_path.write_text(
                "\n".join(
                    [
                        _agent_message("Now I have enough context. Let me update the store."),
                        _agent_message(
                            "Phase 5.2 (cast-rail slot picker interactivity) is shipped "
                            "on `vtt` as commits `5c050b6` + `d6ff0c7`."
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            summary = extract_handoff_summary(log_path, provider_hint="codex")

            assert summary is not None
            self.assertIn("State: Phase 5.2 (cast-rail slot picker interactivity) is shipped", summary)
            self.assertNotIn("State: Now I have enough context.", summary)

    def test_extract_handoff_summary_prefers_committed_state_over_exploration(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "iteration-000003.json"
            log_path.write_text(
                "\n".join(
                    [
                        _agent_message("Now let me check the current roadmap notes."),
                        _agent_message(
                            "Phase 5.3 is shipped and committed. Both commits are on `vtt`, "
                            "working tree is clean, and the roadmap docs point the next agent "
                            "at Phase 5.4."
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            summary = extract_handoff_summary(log_path, provider_hint="codex")

            assert summary is not None
            self.assertIn("State: Phase 5.3 is shipped and committed.", summary)
            self.assertNotIn("State: Now let me check", summary)

    def test_extract_handoff_summary_does_not_treat_commit_intent_as_terminal(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "iteration-000004.json"
            log_path.write_text(
                "\n".join(
                    [
                        _agent_message("These changes must be committed before continuing."),
                        _agent_message("Next task is update the regression tests."),
                    ]
                ),
                encoding="utf-8",
            )

            summary = extract_handoff_summary(log_path, provider_hint="codex")

            assert summary is not None
            self.assertIn("State: Next task is update the regression tests.", summary)
            self.assertNotIn("State: These changes must be committed", summary)

    def test_extract_handoff_summary_selects_commit_failed_recovery_note(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "iteration-000005.json"
            log_path.write_text(
                "\n".join(
                    [
                        _agent_message("Now update phase-5-spell-cast.md to mark 5.5 as shipped:"),
                        _agent_message("All 181 files / 2931 tests pass. Now let me commit the work."),
                        _agent_message("The commit failed. Let me check the output."),
                    ]
                ),
                encoding="utf-8",
            )

            details = extract_handoff_details(log_path, provider_hint="codex")

            assert details.summary is not None
            self.assertIn("State: The commit failed. Let me check the output.", details.summary)
            self.assertEqual(
                details.retry_recommended_next_step,
                "The commit failed. Let me check the output.",
            )

    def test_extract_handoff_summary_does_not_demote_apply_fix_recovery_note(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "iteration-000006.json"
            log_path.write_text(
                "\n".join(
                    [
                        _agent_message("The work is blocked on a small parser bug."),
                        _agent_message("Now I'll apply the fix."),
                    ]
                ),
                encoding="utf-8",
            )

            summary = extract_handoff_summary(log_path, provider_hint="codex")

            assert summary is not None
            self.assertIn("State: Now I'll apply the fix.", summary)

    def test_extract_handoff_summary_avoids_complete_picture_for_rate_limit(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "iteration-000006.json"
            log_path.write_text(
                "\n".join(
                    [
                        _agent_message("Now I have a complete picture. Let me start the implementation."),
                        _agent_message(
                            "The lint error is in pre-existing staged work from Phase 5.4. "
                            "Let me first commit my Phase 6.1 changes separately."
                        ),
                        json.dumps(
                            {
                                "type": "turn.failed",
                                "error": {"message": "You've hit your usage limit."},
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            summary = extract_handoff_summary(log_path, provider_hint="codex")

            assert summary is not None
            self.assertIn("State: The lint error is in pre-existing staged work", summary)
            self.assertNotIn("State: Now I have a complete picture.", summary)
            self.assertIn("Interruption: You've hit your usage limit.", summary)

    def test_extract_handoff_summary_keeps_generic_chatter_as_last_resort(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "iteration-000007.json"
            log_path.write_text(
                _agent_message("Now I have enough context. Let me check the next file."),
                encoding="utf-8",
            )

            summary = extract_handoff_summary(log_path, provider_hint="codex")

            assert summary is not None
            self.assertIn("Previous iteration summary:", summary)
            self.assertIn("State: Now I have enough context. Let me check the next file.", summary)

    def test_resolve_resume_context_reextracts_old_metadata_summary(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            log_path = temp_root / "iteration-000001.json"
            log_path.write_text(
                _agent_message("Phase 8.1 is shipped and committed. Working tree is clean."),
                encoding="utf-8",
            )
            metadata_path_for(log_path).write_text(
                json.dumps(
                    {
                        "provider_name": "codex",
                        "handoff_summary": (
                            "Previous iteration summary:\n"
                            "- Goal: Now I have enough context. Let me update the store."
                        ),
                        "retry_recommended_next_step": "Now I have enough context.",
                    }
                ),
                encoding="utf-8",
            )

            context = resolve_resume_context(log_path)

            assert context.previous_handoff_summary is not None
            self.assertIn("State: Phase 8.1 is shipped and committed.", context.previous_handoff_summary)
            self.assertNotIn("Now I have enough context", context.previous_handoff_summary)
            self.assertEqual(
                context.previous_retry_recommended_next_step,
                "Phase 8.1 is shipped and committed. Working tree is clean.",
            )
            metadata = json.loads(metadata_path_for(log_path).read_text(encoding="utf-8"))
            self.assertEqual(metadata["handoff_extractor_version"], 2)
            self.assertEqual(
                metadata["retry_recommended_next_step"],
                "Phase 8.1 is shipped and committed. Working tree is clean.",
            )
            self.assertIn("State: Phase 8.1 is shipped and committed.", metadata["handoff_summary"])

    def test_resolve_resume_context_reuses_current_version_metadata(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            log_path = temp_root / "iteration-000001.json"
            log_path.write_text(
                _agent_message("Phase 8.1 is shipped and committed. Working tree is clean."),
                encoding="utf-8",
            )
            metadata_path_for(log_path).write_text(
                json.dumps(
                    {
                        "provider_name": "codex",
                        "handoff_extractor_version": 2,
                        "handoff_summary": "Previous iteration summary:\n- State: Cached summary.",
                        "retry_recommended_next_step": "Cached next step.",
                    }
                ),
                encoding="utf-8",
            )

            with patch("batonloop.handoff.extract_handoff_details", side_effect=AssertionError):
                context = resolve_resume_context(log_path)

            self.assertEqual(
                context.previous_handoff_summary,
                "Previous iteration summary:\n- State: Cached summary.",
            )
            self.assertEqual(context.previous_retry_recommended_next_step, "Cached next step.")

    def test_build_resume_prompt_includes_recommended_resume_point(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            base_prompt_path = temp_root / "PROMPT.md"
            base_prompt_path.write_text("base prompt", encoding="utf-8")
            log_path = temp_root / "iteration-000001.json"
            log_path.write_text(
                _agent_message("All tests pass. Now commit the work."),
                encoding="utf-8",
            )
            metadata_path_for(log_path).write_text(
                json.dumps(
                    {
                        "handoff_extractor_version": 2,
                        "handoff_summary": (
                            "Previous iteration summary:\n"
                            "- State: All tests pass. Now commit the work."
                        ),
                        "retry_recommended_next_step": "All tests pass. Now commit the work.",
                    }
                ),
                encoding="utf-8",
            )
            context = resolve_resume_context(log_path)

            prompt = build_resume_prompt(
                base_prompt_path=base_prompt_path,
                current_provider_name="codex",
                working_dir=temp_root,
                log_dir=temp_root / "logs",
                resume_context=context,
                resume_note=None,
            )

            self.assertIn(
                "Recommended resume point: All tests pass. Now commit the work.",
                prompt,
            )

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
            self.assertEqual(metadata["handoff_extractor_version"], 2)
            self.assertEqual(metadata["exit_code"], 1)
            self.assertFalse(metadata["success"])
            self.assertEqual(metadata["failure_message"], "process failed with exit code 1")
            self.assertEqual(metadata["resume_source_log_path"], None)
            self.assertEqual(metadata["failover_target_provider"], None)
            self.assertEqual(metadata["handoff_summary"], None)
            self.assertEqual(metadata["last_progress_messages"], [])
            self.assertEqual(metadata["last_tasks"], [])
            self.assertEqual(metadata["last_interruption"], None)
            self.assertEqual(metadata["retry_recommended_next_step"], None)
            self.assertIn("git_status", metadata)

    def test_timeout_retry_injects_resume_context_into_next_attempt(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            config = _make_config(
                temp_root,
                max_iterations=2,
                iteration_timeout_minutes=Decimal("0.001"),
                max_consecutive_errors=5,
            )
            provider = SequencedCommandProvider(
                (
                    [sys.executable, "-c", "import time; time.sleep(1)"],
                    [sys.executable, "-c", "import sys; print(sys.stdin.read())"],
                )
            )

            exit_code = run_loop(config, {"fake": provider})

            self.assertEqual(exit_code, 0)
            second_iteration_log = config.log_dir / "iteration-000002.json"
            second_prompt_artifact = prompt_artifact_path_for(second_iteration_log)
            second_metadata = json.loads(
                metadata_path_for(second_iteration_log).read_text(encoding="utf-8")
            )
            self.assertTrue(second_prompt_artifact.is_file())
            self.assertEqual(
                second_metadata["resume_source_log_path"],
                str(config.log_dir / "iteration-000001.json"),
            )
            self.assertEqual(
                second_metadata["resume_source_metadata_path"],
                str(config.log_dir / "iteration-000001.meta.json"),
            )
            self.assertEqual(second_metadata["input_prompt_path"], str(second_prompt_artifact))
            prompt_text = second_prompt_artifact.read_text(encoding="utf-8")
            self.assertIn("=== BATONLOOP RESUME CONTEXT ===", prompt_text)
            self.assertIn(
                f"Previous raw log: {config.log_dir / 'iteration-000001.json'}",
                prompt_text,
            )
            self.assertIn("Previous timed out: True", prompt_text)
            self.assertIn("Previous failure summary: Iteration 1 timed out", prompt_text)

    def test_retryable_failure_injects_resume_context_without_advancing_prompt(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            config = _make_config(
                temp_root,
                max_iterations=2,
                max_consecutive_errors=5,
                prompt_texts=("first prompt", "second prompt"),
            )
            provider = SequencedCommandProvider(
                (
                    [
                        sys.executable,
                        "-c",
                        (
                            "import json, sys; "
                            "print(json.dumps({'type':'item.completed','item':"
                            "{'type':'agent_message','text':'All tests pass. Now commit the work.'}})); "
                            "sys.exit(1)"
                        ),
                    ],
                    [sys.executable, "-c", "import sys; print(sys.stdin.read())"],
                )
            )

            exit_code = run_loop(config, {"fake": provider})

            self.assertEqual(exit_code, 0)
            second_iteration_log = config.log_dir / "iteration-000002.json"
            second_prompt_artifact = prompt_artifact_path_for(second_iteration_log)
            self.assertTrue(second_prompt_artifact.is_file())
            prompt_text = second_prompt_artifact.read_text(encoding="utf-8")
            self.assertIn("first prompt", prompt_text)
            self.assertNotIn("second prompt", prompt_text)
            self.assertIn("=== BATONLOOP RESUME CONTEXT ===", prompt_text)
            self.assertIn(
                f"Previous raw log: {config.log_dir / 'iteration-000001.json'}",
                prompt_text,
            )
            self.assertIn("Previous exit code: 1", prompt_text)
            self.assertIn("State: All tests pass. Now commit the work.", prompt_text)
            self.assertIn(
                "Recommended resume point: All tests pass. Now commit the work.",
                prompt_text,
            )

            second_metadata = json.loads(
                metadata_path_for(second_iteration_log).read_text(encoding="utf-8")
            )
            self.assertEqual(second_metadata["base_prompt_path"], str(config.prompt_sequence[0]))
            self.assertEqual(second_metadata["input_prompt_path"], str(second_prompt_artifact))
            self.assertEqual(
                second_metadata["resume_source_log_path"],
                str(config.log_dir / "iteration-000001.json"),
            )

    def test_failed_iterations_count_toward_iteration_limit(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            config = _make_config(temp_root, max_iterations=3, max_consecutive_errors=5)
            provider = StaticCommandProvider(
                [sys.executable, "-c", "import sys; print('boom'); sys.exit(1)"],
            )

            exit_code = run_loop(config, {"fake": provider})

            self.assertEqual(exit_code, 1)
            self.assertTrue((config.log_dir / "iteration-000001.json").is_file())
            self.assertTrue((config.log_dir / "iteration-000002.json").is_file())
            self.assertTrue((config.log_dir / "iteration-000003.json").is_file())
            self.assertFalse((config.log_dir / "iteration-000004.json").exists())

            log_text = (config.log_dir / "batonloop.log").read_text(encoding="utf-8")
            self.assertIn("STOP: Iteration limit reached (3)", log_text)
            self.assertIn("Attempts:    3", log_text)
            self.assertIn("Completed:   0", log_text)
            self.assertNotIn("FATAL: 5 consecutive errors. Stopping.", log_text)

    def test_rate_limit_failure_automatically_fails_over_to_next_provider(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            config = _make_config(
                temp_root,
                max_iterations=2,
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

    def test_alternate_provider_strategy_rotates_after_success(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            config = _make_config(
                temp_root,
                max_iterations=3,
                provider_names=("one", "two"),
                provider_strategy=ProviderStrategy.ALTERNATE,
            )
            providers = {
                "one": StaticCommandProvider([sys.executable, "-c", "print('one')"]),
                "two": StaticCommandProvider([sys.executable, "-c", "print('two')"]),
            }

            exit_code = run_loop(config, providers)

            self.assertEqual(exit_code, 0)
            provider_names = [
                json.loads(
                    metadata_path_for(
                        config.log_dir / f"iteration-00000{index}.json"
                    ).read_text(encoding="utf-8")
                )["provider_name"]
                for index in range(1, 4)
            ]
            self.assertEqual(provider_names, ["one", "two", "one"])

    def test_failover_respects_iteration_limit(self) -> None:
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

            self.assertEqual(exit_code, 1)
            self.assertTrue((config.log_dir / "iteration-000001.json").is_file())
            self.assertFalse((config.log_dir / "iteration-000002.json").exists())

            log_text = (config.log_dir / "batonloop.log").read_text(encoding="utf-8")
            self.assertNotIn("AUTO FAILOVER: Switching provider from limited to backup.", log_text)
            self.assertIn("Iteration limit reached before another retry.", log_text)
            self.assertIn("STOP: Iteration limit reached (1)", log_text)
            self.assertIn("Attempts:    1", log_text)
            self.assertIn("Completed:   0", log_text)

    def test_handle_iteration_outcome_returns_failover_target(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            config = _make_config(
                temp_root,
                max_iterations=2,
                provider_names=("limited", "backup"),
                provider_profiles={"backup": ProviderProfile(model="gpt-5.4")},
            )
            config.log_dir.mkdir()
            iteration_log = config.log_dir / "iteration-000001.json"
            iteration_log.write_text("rate limited", encoding="utf-8")
            prompt_path = config.prompt_sequence[0]

            logger = logging.getLogger("batonloop.test.outcome")
            logger.handlers.clear()
            logger.addHandler(logging.NullHandler())
            controller = StopController(logger)
            state = RunState(start_time=0.0, attempted_iterations=1)

            limited_slot = ProviderSlot(
                provider=RateLimitedProvider(),
                execution=ProviderExecution(
                    name="limited",
                    binary=sys.executable,
                    model=None,
                    max_turns=None,
                    use_bare=False,
                    safe_mode=False,
                ),
            )
            backup_slot = ProviderSlot(
                provider=ResumeEchoProvider(),
                execution=ProviderExecution(
                    name="backup",
                    binary=sys.executable,
                    model="gpt-5.4",
                    max_turns=None,
                    use_bare=False,
                    safe_mode=False,
                ),
            )
            iteration = IterationExecution(
                number=1,
                slot=limited_slot,
                prompt_path=prompt_path,
                prompt_input_path=prompt_path,
                log_path=iteration_log,
                result=CommandResult(exit_code=1),
            )

            outcome = _handle_iteration_outcome(
                logger=logger,
                config=config,
                state=state,
                controller=controller,
                iteration=iteration,
                provider_slots=(limited_slot, backup_slot),
                current_provider_index=0,
            )

            self.assertEqual(outcome.exit_code, 1)
            self.assertFalse(outcome.success)
            self.assertEqual(
                outcome.failure_message,
                "RATE LIMITED detected in output. Waiting 30 minutes before retrying...",
            )
            self.assertEqual(outcome.failover_target_provider, "backup")
            self.assertEqual(outcome.next_provider_index, 1)
            self.assertFalse(outcome.should_break)
            self.assertEqual(state.consecutive_errors, 0)

    def test_retry_backoff_applies_to_nonfatal_failures(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            config = _make_config(
                temp_root,
                max_iterations=3,
                retry_backoff_base_seconds=2,
                retry_backoff_multiplier=Decimal("3"),
                retry_backoff_max_seconds=5,
            )
            config.log_dir.mkdir()
            iteration_log = config.log_dir / "iteration-000002.json"
            iteration_log.write_text("boom", encoding="utf-8")
            prompt_path = config.prompt_sequence[0]

            logger = logging.getLogger("batonloop.test.retry-backoff")
            logger.handlers.clear()
            logger.addHandler(logging.NullHandler())
            controller = StopController(logger)
            state = RunState(
                start_time=0.0,
                attempted_iterations=2,
                consecutive_errors=1,
            )
            slot = ProviderSlot(
                provider=StaticCommandProvider(
                    [sys.executable, "-c", "import sys; sys.exit(1)"],
                    failure_decision=FailureDecision(message="temporary failure"),
                ),
                execution=ProviderExecution(
                    name="fake",
                    binary=sys.executable,
                    model=None,
                    max_turns=None,
                    use_bare=False,
                    safe_mode=False,
                ),
            )
            iteration = IterationExecution(
                number=2,
                slot=slot,
                prompt_path=prompt_path,
                prompt_input_path=prompt_path,
                log_path=iteration_log,
                result=CommandResult(exit_code=1),
            )

            outcome = _handle_iteration_outcome(
                logger=logger,
                config=config,
                state=state,
                controller=controller,
                iteration=iteration,
                provider_slots=(slot,),
                current_provider_index=0,
            )

            self.assertEqual(outcome.wait_seconds, 5)
            self.assertTrue(outcome.skip_pause_after_wait)
            self.assertFalse(outcome.should_break)

    def test_provider_cooldown_waits_for_next_available_failover_target(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            config = _make_config(
                temp_root,
                max_iterations=3,
                provider_names=("limited", "backup"),
                provider_cooldown_seconds=60,
            )
            config.log_dir.mkdir()
            iteration_log = config.log_dir / "iteration-000001.json"
            iteration_log.write_text("rate limited", encoding="utf-8")
            prompt_path = config.prompt_sequence[0]

            logger = logging.getLogger("batonloop.test.provider-cooldown")
            logger.handlers.clear()
            logger.addHandler(logging.NullHandler())
            controller = StopController(logger)
            state = RunState(
                start_time=0.0,
                attempted_iterations=1,
                provider_cooldowns={"limited": 150.0},
            )
            limited_slot = ProviderSlot(
                provider=RateLimitedProvider(),
                execution=ProviderExecution(
                    name="limited",
                    binary=sys.executable,
                    model=None,
                    max_turns=None,
                    use_bare=False,
                    safe_mode=False,
                ),
            )
            backup_slot = ProviderSlot(
                provider=RateLimitedProvider(),
                execution=ProviderExecution(
                    name="backup",
                    binary=sys.executable,
                    model=None,
                    max_turns=None,
                    use_bare=False,
                    safe_mode=False,
                ),
            )
            iteration = IterationExecution(
                number=1,
                slot=backup_slot,
                prompt_path=prompt_path,
                prompt_input_path=prompt_path,
                log_path=iteration_log,
                result=CommandResult(exit_code=1),
            )

            with patch("batonloop.runner.time.monotonic", return_value=100.0):
                outcome = _handle_iteration_outcome(
                    logger=logger,
                    config=config,
                    state=state,
                    controller=controller,
                    iteration=iteration,
                    provider_slots=(limited_slot, backup_slot),
                    current_provider_index=1,
                )

            self.assertEqual(outcome.failover_target_provider, "limited")
            self.assertEqual(outcome.next_provider_index, 0)
            self.assertEqual(outcome.wait_seconds, 50)
            self.assertTrue(outcome.skip_pause_after_wait)
            self.assertEqual(state.provider_cooldowns["backup"], 160.0)


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


class SequencedCommandProvider(StaticCommandProvider):
    def __init__(
        self,
        commands: tuple[list[str], ...],
        *,
        failure_decision: FailureDecision | None = None,
    ) -> None:
        if not commands:
            raise ValueError("commands may not be empty")
        super().__init__(commands[0], failure_decision=failure_decision)
        self._commands = commands
        self._next_command_index = 0

    def build_command(self, config: RunnerConfig, execution: ProviderExecution) -> list[str]:
        del config, execution
        index = min(self._next_command_index, len(self._commands) - 1)
        self._next_command_index += 1
        return self._commands[index]


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


def _agent_message(text: str) -> str:
    return json.dumps(
        {
            "type": "item.completed",
            "item": {
                "type": "agent_message",
                "text": text,
            },
        }
    )


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
    live_output: bool = True,
    retry_backoff_base_seconds: int = 0,
    retry_backoff_multiplier: Decimal = Decimal("2"),
    retry_backoff_max_seconds: int = 0,
    retry_jitter_fraction: Decimal = Decimal("0"),
    provider_cooldown_seconds: int = 0,
    provider_strategy: ProviderStrategy = ProviderStrategy.FAILOVER,
    prompt_texts: tuple[str, ...] = ("prompt",),
) -> RunnerConfig:
    prompt_paths: list[Path] = []
    for index, prompt_text in enumerate(prompt_texts):
        prompt_path = temp_root / ("PROMPT.md" if index == 0 else f"PROMPT-{index + 1}.md")
        prompt_path.write_text(prompt_text, encoding="utf-8")
        prompt_paths.append(prompt_path)
    return RunnerConfig(
        working_dir=temp_root,
        run_config_path=None,
        provider_names=provider_names,
        provider_profiles=provider_profiles or {},
        provider_config_path=None,
        default_provider_profile=ProviderProfile(),
        prompt_specs=tuple(PromptSpec(path=prompt_path, repeat=1) for prompt_path in prompt_paths),
        prompt_sequence=tuple(prompt_paths),
        max_iterations=max_iterations,
        max_cost=Decimal("0"),
        max_duration_hours=Decimal("0"),
        iteration_timeout_minutes=iteration_timeout_minutes,
        pause_seconds=0,
        wait_on_limit_mins=30,
        retry_backoff_base_seconds=retry_backoff_base_seconds,
        retry_backoff_multiplier=retry_backoff_multiplier,
        retry_backoff_max_seconds=retry_backoff_max_seconds,
        retry_jitter_fraction=retry_jitter_fraction,
        provider_cooldown_seconds=provider_cooldown_seconds,
        provider_strategy=provider_strategy,
        max_consecutive_errors=max_consecutive_errors,
        log_dir=temp_root / "batonloop-logs",
        log_retain=0,
        check_commands=check_commands,
        stop_on_regexes=stop_on_regexes,
        stop_on_clean_git=stop_on_clean_git,
        stop_when_files=stop_when_files,
        output_format=OutputFormat.STREAM_JSON,
        live_output=live_output,
        resume_from=resume_from,
        resume_note=resume_note,
        dry_run=False,
    )


if __name__ == "__main__":
    unittest.main()
