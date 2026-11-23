"""Optional LLM integration: ask for a structured claim, verify it, retry with feedback.

No vendor SDK is imported. Pass any object with `complete(prompt: str) -> str`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from .guard import Claim, Guard, Report
from .laws import LAWS


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


def build_prompt(question: str, feedback: str | None = None) -> str:
    laws = "\n".join(f"- {n}({', '.join(spec.params)}): {spec.doc}" for n, spec in LAWS.items())
    prompt = (
        "Answer the engineering question. Reply with JSON only, with keys:\n"
        '  "answer": value with unit (string), e.g. "12.4 mm"\n'
        '  "law": one of the law names below, or null\n'
        '  "inputs": object mapping each law parameter to a value with unit\n'
        f"Available laws:\n{laws}\n\nQuestion: {question}\n"
    )
    if feedback:
        prompt += f"\nYour previous answer failed verification:\n{feedback}\nFix it and reply with JSON only.\n"
    return prompt


def parse_claim(text: str) -> Claim:
    """Extract the first JSON object from a model reply (tolerates code fences)."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("no JSON object found in model reply")
    return Claim(**json.loads(match.group(0)))


@dataclass
class GuardedResult:
    claim: Claim | None
    report: Report | None
    attempts: int
    accepted: bool


def guarded_answer(
    question: str, client: LLMClient, guard: Guard | None = None, max_attempts: int = 3
) -> GuardedResult:
    """Ask, verify, and feed failures back to the model until it passes or attempts run out."""
    guard = guard or Guard()
    feedback: str | None = None
    claim: Claim | None = None
    report: Report | None = None
    for attempt in range(1, max_attempts + 1):
        raw = client.complete(build_prompt(question, feedback))
        try:
            claim = parse_claim(raw)
        except Exception as exc:
            feedback = f"- reply was not valid JSON for the schema: {exc}"
            continue
        report = guard.verify(claim)
        if report.ok:
            return GuardedResult(claim, report, attempt, True)
        feedback = report.feedback()
    return GuardedResult(claim, report, max_attempts, False)
