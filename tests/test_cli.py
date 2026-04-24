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
            self.assertIn("Goal: Phase 4.3 is the next recommended task.", output)
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


if __name__ == "__main__":
    unittest.main()
