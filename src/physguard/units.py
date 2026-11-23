"""Unit parsing and dimensional checks (built on pint)."""
from __future__ import annotations

import re

import pint

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity
ureg.define("pcm = 1e-5 = pcm")  # per cent mille, common in reactor physics


class UnitError(ValueError):
    """Raised when a quantity cannot be parsed or has the wrong dimension."""


_NUMBER_THEN_UNIT = re.compile(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(\S.*)$")


def parse_quantity(text: str) -> pint.Quantity:
    """Parse strings such as '5 kN', '9.81 m/s**2' or '25 degC' into a pint Quantity."""
    if not isinstance(text, str) or not text.strip():
        raise UnitError("empty quantity")
    s = text.strip()
    try:
        q = ureg.Quantity(s)
    except Exception as first:  # pint raises several error types
        # pint < 0.26 cannot parse offset units from text ('25 degC' reads as 25 * degC and is rejected as
        # ambiguous), so fall back to building the quantity from its number and its unit.
        m = _NUMBER_THEN_UNIT.match(s)
        if not m:
            raise UnitError(f"cannot parse quantity {text!r}: {first}") from first
        try:
            q = ureg.Quantity(float(m.group(1)), m.group(2))
        except Exception as exc:
            raise UnitError(f"cannot parse quantity {text!r}: {exc}") from exc
    if not isinstance(q, ureg.Quantity):  # bare number
        q = ureg.Quantity(q, "dimensionless")
    return q


def dimension_of(spec: str) -> pint.util.UnitsContainer:
    """Dimensionality of a unit ('meter') or a dimension name ('[length]')."""
    try:
        return ureg.get_dimensionality(spec)
    except Exception as exc:
        raise UnitError(f"unknown unit or dimension {spec!r}: {exc}") from exc


def has_dimension(q: pint.Quantity, spec: str) -> bool:
    return q.dimensionality == dimension_of(spec)
