from __future__ import annotations

import io
import json
import logging
import unittest

from batonloop.live_output import LiveOutputConsumer


class LiveOutputConsumerTests(unittest.TestCase):
    def test_emits_agent_messages_and_ignores_known_noise(self) -> None:
        consumer, stream = _make_consumer("codex")

        consumer.consume_line("Reading prompt from stdin...\n")
        consumer.consume_line(
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "type": "agent_message",
                        "text": "I have enough context to implement the retry fix.",
                    },
                }
            )
        )
        consumer.consume_line(
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "type": "command_execution",
                        "command": "pytest -q",
                        "aggregated_output": "noise",
                    },
                }
            )
        )

        output = stream.getvalue()
        self.assertIn("[codex] I have enough context to implement the retry fix.", output)
        self.assertNotIn("Reading prompt from stdin...", output)
        self.assertNotIn("command_execution", output)

    def test_emits_checklist_and_task_summaries_without_duplicates(self) -> None:
        consumer, stream = _make_consumer("codex")
        todo_payload = json.dumps(
            {
                "type": "item.updated",
                "item": {
                    "type": "todo_list",
                    "items": [
                        {"text": "Implement filtered live output", "completed": True},
                        {"text": "Run tests", "completed": False},
                    ],
                },
            }
        )

        consumer.consume_line(todo_payload)
        consumer.consume_line(todo_payload)
        consumer.consume_line(
            json.dumps(
                {
                    "type": "item.started",
                    "item": {
                        "type": "collab_tool_call",
                        "tool": "spawn_agent",
                        "prompt": (
                            "Review the current diff critically. Focus on output noise, "
                            "filter correctness, and behavior regressions."
                        ),
                    },
                }
            )
        )

        output = stream.getvalue()
        self.assertEqual(output.count("Checklist:"), 1)
        self.assertIn(
            "[codex] Checklist: 1/2 complete; remaining: Run tests",
            output,
        )
        self.assertIn(
            "[codex] Task: spawn_agent: Review the current diff critically. Focus on output noise, filter correctness, and behavior regressions.",
            output,
        )

    def test_emits_duplicate_interruptions_once(self) -> None:
        consumer, stream = _make_consumer("codex")
        interruption = "You've hit your usage limit. Try again later."

        consumer.consume_line(json.dumps({"type": "error", "message": interruption}))
        consumer.consume_line(
            json.dumps({"type": "turn.failed", "error": {"message": interruption}})
        )

        output = stream.getvalue()
        self.assertEqual(output.count("Interruption:"), 1)
        self.assertIn(f"[codex] Interruption: {interruption}", output)


def _make_consumer(provider_name: str) -> tuple[LiveOutputConsumer, io.StringIO]:
    stream = io.StringIO()
    logger = logging.getLogger(f"test-live-output-{provider_name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)
    return LiveOutputConsumer(logger, provider_name), stream


if __name__ == "__main__":
    unittest.main()
