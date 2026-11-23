"""Core verification: compare a model's numeric answer with a reference law."""
from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel, Field

from .bounds import Bound, universal_violations
from .laws import LAWS
from .units import UnitError, has_dimension, parse_quantity


class Verdict(str, Enum):
    PASS = "pass"
    WARN = "warn"  # nothing wrong found, but could not verify against a law
    FAIL = "fail"


class Claim(BaseModel):
    """A structured numeric answer produced by an LLM."""

    answer: str = Field(description="Value with unit, e.g. '12.4 mm'")
    law: str | None = Field(default=None, description="Name of a registered law")
    inputs: dict[str, str] = Field(default_factory=dict, description="Law inputs with units")
    expected_dimension: str | None = Field(
        default=None, description="Unit or dimension the answer must have"
    )


class Check(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class Report(BaseModel):
    verdict: Verdict
    checks: list[Check]
    answer_value: str | None = None
    reference_value: str | None = None
    relative_error: float | None = None

    @property
    def ok(self) -> bool:
        return self.verdict != Verdict.FAIL

    def feedback(self) -> str:
        """Short text to send back to the LLM for a repair attempt."""
        lines = [f"- {c.name}: {c.detail}" for c in self.checks if not c.passed]
        if self.reference_value:
            lines.append(f"- reference value from the physical law: {self.reference_value}")
        return "\n".join(lines)


def _prefix_hint(ratio: float) -> str | None:
    """Detect answers off by (almost exactly) a power of ten (typical unit-prefix mistake)."""
    if ratio <= 0 or not math.isfinite(ratio):
        return None
    exp = math.log10(ratio)
    k = round(exp)
    if k != 0 and abs(exp - k) < 0.005:
        return f"answer is off by a factor of 10^{k}; likely a unit-prefix error"
    return None


class Guard:
    """Verify claims. `rtol` is the allowed relative error against the reference."""

    def __init__(self, rtol: float = 0.01, bounds: Bound | None = None):
        self.rtol = rtol
        self.bounds = bounds

    def verify(self, claim: Claim) -> Report:
        checks: list[Check] = []

        try:
            answer = parse_quantity(claim.answer)
        except UnitError as exc:
            return Report(verdict=Verdict.FAIL, checks=[Check(name="parse", passed=False, detail=str(exc))])
        checks.append(Check(name="parse", passed=True))

        for problem in universal_violations(answer):
            checks.append(Check(name="universal_limits", passed=False, detail=problem))

        if self.bounds is not None:
            try:
                inside = self.bounds.contains(answer)
                rng = f"[{self.bounds.min}, {self.bounds.max}] {self.bounds.unit}"
                detail = "" if inside else f"outside allowed range {rng}"
            except Exception as exc:
                inside, detail = False, f"bound check failed: {exc}"
            checks.append(Check(name="bounds", passed=inside, detail=detail))

        if claim.expected_dimension:
            try:
                ok = has_dimension(answer, claim.expected_dimension)
                got, want = answer.dimensionality, claim.expected_dimension
                detail = "" if ok else f"answer has dimension {got}, expected {want}"
            except UnitError as exc:
                ok, detail = False, str(exc)
            checks.append(Check(name="expected_dimension", passed=ok, detail=detail))

        reference = None
        rel_err = None
        if claim.law is None:
            verdict_if_clean = Verdict.WARN
        elif claim.law not in LAWS:
            checks.append(Check(name="law", passed=False, detail=f"unknown law {claim.law!r}"))
            verdict_if_clean = Verdict.WARN
        else:
            verdict_if_clean = Verdict.PASS
            spec = LAWS[claim.law]
            missing = [p for p in spec.params if p not in claim.inputs]
            if missing:
                checks.append(Check(name="inputs", passed=False, detail=f"missing inputs: {missing}"))
            else:
                try:
                    args = {p: parse_quantity(claim.inputs[p]) for p in spec.params}
                    reference = spec.func(**args)
                except Exception as exc:
                    checks.append(Check(name="reference", passed=False, detail=f"cannot compute reference: {exc}"))
                else:
                    if answer.dimensionality != reference.dimensionality:
                        checks.append(Check(
                            name="dimension",
                            passed=False,
                            detail=(
                                f"answer has dimension {answer.dimensionality}, "
                                f"law gives {reference.dimensionality}"
                            ),
                        ))
                    else:
                        ref_val = reference.magnitude
                        ans_val = answer.to(reference.units).magnitude
                        if ref_val == 0:
                            rel_err = abs(ans_val)
                        else:
                            rel_err = abs(ans_val - ref_val) / abs(ref_val)
                        ok = rel_err <= self.rtol
                        detail = "" if ok else f"relative error {rel_err:.3g} exceeds tolerance {self.rtol}"
                        if not ok and ref_val != 0:
                            hint = _prefix_hint(ans_val / ref_val)
                            if hint:
                                detail += f"; {hint}"
                        checks.append(Check(name="magnitude", passed=ok, detail=detail))

        failed = any(not c.passed for c in checks)
        verdict = Verdict.FAIL if failed else verdict_if_clean
        return Report(
            verdict=verdict,
            checks=checks,
            answer_value=f"{answer:.6g~P}",
            reference_value=f"{reference:.6g~P}" if reference is not None else None,
            relative_error=rel_err,
        )
