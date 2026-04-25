from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from batonloop.cli import main


class CliTests(unittest.TestCase):
    def test_handoff_summary_command_prints_extracted_summary(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            log_path = temp_root / "iteration-000001.json"
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
                                            "text": "Phase 4.3 is the next recommended task.",
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

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["handoff-summary", str(log_path)])

            self.assertEqual(exit_code, 0)
            output = stdout.getvalue()
            self.assertIn("Previous iteration summary:", output)
            self.assertIn("State: Phase 4.3 is the next recommended task.", output)
            self.assertIn("Interruption: You've hit your usage limit.", output)

    def test_handoff_summary_command_accepts_log_directory(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            log_dir = temp_root / "logs"
            log_dir.mkdir()
            (log_dir / "iteration-000001.json").write_text("older", encoding="utf-8")
            (log_dir / "iteration-000002.json").write_text(
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "Current hot task is Phase 5.0."}
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["handoff-summary", str(log_dir)])

            self.assertEqual(exit_code, 0)
            self.assertIn("Phase 5.0", stdout.getvalue())

    def test_main_help_lists_handoff_summary_command(self) -> None:
        stdout = io.StringIO()
        with self.assertRaises(SystemExit) as exc:
            with redirect_stdout(stdout):
                main(["--help"])

        self.assertEqual(exc.exception.code, 0)
        self.assertIn("handoff-summary", stdout.getvalue())
        self.assertIn("inspect-handoff", stdout.getvalue())

    def test_inspect_handoff_warns_when_retry_used_base_prompt(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            log_dir = temp_root / "logs"
            log_dir.mkdir()
            prompt_path = temp_root / "PROMPT.md"
            prompt_path.write_text("prompt", encoding="utf-8")
            (log_dir / "iteration-000001.json").write_text("boom", encoding="utf-8")
            (log_dir / "iteration-000001.meta.json").write_text(
                json.dumps(
                    {
                        "success": False,
                        "handoff_summary": "Previous iteration summary:\n- State: Fix retry.",
                    }
                ),
                encoding="utf-8",
            )
            (log_dir / "iteration-000002.json").write_text(
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "agent_message",
                            "text": "Starting from the base prompt.",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (log_dir / "iteration-000002.meta.json").write_text(
                json.dumps(
                    {
                        "success": True,
                        "base_prompt_path": str(prompt_path),
                        "input_prompt_path": str(prompt_path),
                        "resume_source_log_path": None,
                        "resume_source_metadata_path": None,
                    }
                ),
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    ["inspect-handoff", str(log_dir), "--iterations", "2", "--first", "1"]
                )

            self.assertEqual(exit_code, 0)
            output = stdout.getvalue()
            self.assertIn("Iteration 000002", output)
            self.assertIn("State: Fix retry.", output)
            self.assertIn("Generated prompt artifact: no", output)
            self.assertIn("WARNING: previous iteration failed", output)
            self.assertIn("Starting from the base prompt.", output)


if __name__ == "__main__":
    unittest.main()
