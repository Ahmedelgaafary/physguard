"""Run: python examples/demo.py  (no API key needed; uses a scripted fake LLM)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physguard import guarded_answer  # noqa: E402

INPUTS = {"F": "1 kN", "L": "2 m", "E": "200 GPa", "I": "8e-6 m**4"}


class FakeLLM:
    """First reply has a unit-prefix mistake; second reply fixes it after feedback."""

    def __init__(self):
        self.replies = [
            json.dumps({"answer": "1.667 m", "law": "cantilever_tip_deflection", "inputs": INPUTS}),
            json.dumps({"answer": "1.667 mm", "law": "cantilever_tip_deflection", "inputs": INPUTS}),
        ]

    def complete(self, prompt: str) -> str:
        return self.replies.pop(0)


res = guarded_answer("Tip deflection of a 2 m steel cantilever, 1 kN end load?", FakeLLM())
print("accepted:", res.accepted, "| attempts:", res.attempts)
print("answer:", res.report.answer_value, "| reference:", res.report.reference_value)
