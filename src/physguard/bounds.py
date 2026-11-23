"""Sanity bounds: universal physical limits and user-defined ranges."""
from __future__ import annotations

from dataclasses import dataclass

import pint

from .units import Q_, has_dimension, parse_quantity

C_LIGHT = Q_(299_792_458, "m/s")


@dataclass(frozen=True)
class Bound:
    """Inclusive range. `unit` fixes the dimension the bound applies to."""

    min: float | None = None
    max: float | None = None
    unit: str = "dimensionless"

    def contains(self, q: pint.Quantity) -> bool:
        value = q.to(self.unit).magnitude
        if self.min is not None and value < self.min:
            return False
        return not (self.max is not None and value > self.max)


def universal_violations(q: pint.Quantity) -> list[str]:
    """Hard physical impossibilities, independent of the problem."""
    problems: list[str] = []
    if has_dimension(q, "[velocity]") and abs(q.to("m/s")) > C_LIGHT:
        problems.append("speed exceeds the speed of light")
    if has_dimension(q, "[temperature]"):
        try:
            if q.to("kelvin").magnitude < 0:
                problems.append("temperature below absolute zero")
        except Exception:  # offset-unit edge cases: do not guess
            pass
    if has_dimension(q, "[mass]") and q.to("kg").magnitude < 0:
        problems.append("negative mass")
    return problems


def check_bound(answer: str | pint.Quantity, bound: Bound) -> bool:
    q = parse_quantity(answer) if isinstance(answer, str) else answer
    return bound.contains(q)
