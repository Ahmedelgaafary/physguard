"""Reference implementations of analytical physics/engineering laws.

Each law takes pint Quantities and returns a pint Quantity, so dimensional errors in
the *inputs* are caught by pint itself. Add a law by decorating a function with @law.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import pint

from .units import Q_

G = Q_(9.80665, "m/s**2")
R_GAS = Q_(8.314462618, "J/(mol*K)")


@dataclass(frozen=True)
class Law:
    name: str
    func: Callable[..., pint.Quantity]
    doc: str
    params: tuple[str, ...]


LAWS: dict[str, Law] = {}


def law(name: str, params: tuple[str, ...]):
    def deco(fn: Callable[..., pint.Quantity]):
        LAWS[name] = Law(name, fn, (fn.__doc__ or "").strip(), params)
        return fn

    return deco


@law("cantilever_tip_deflection", ("F", "L", "E", "I"))
def cantilever_tip_deflection(F, L, E, I):  # noqa: E741
    """Tip deflection of a cantilever with end load: F L^3 / (3 E I)."""
    return (F * L**3 / (3 * E * I)).to("m")


@law("simply_supported_midspan_deflection", ("F", "L", "E", "I"))
def simply_supported_midspan_deflection(F, L, E, I):  # noqa: E741
    """Mid-span deflection of a simply supported beam, central load: F L^3 / (48 E I)."""
    return (F * L**3 / (48 * E * I)).to("m")


@law("ohms_law_current", ("V", "R"))
def ohms_law_current(V, R):
    """Current through a resistor: I = V / R."""
    return (V / R).to("A")


@law("kinetic_energy", ("m", "v"))
def kinetic_energy(m, v):
    """Kinetic energy: 0.5 m v^2."""
    return (0.5 * m * v**2).to("J")


@law("projectile_range", ("v", "angle_deg"))
def projectile_range(v, angle_deg):
    """Range on level ground without drag: v^2 sin(2 theta) / g. angle_deg is a plain number."""
    theta = float(angle_deg.to("dimensionless").magnitude)
    return (v**2 * math.sin(2 * math.radians(theta)) / G).to("m")


@law("ideal_gas_pressure", ("n", "T", "V"))
def ideal_gas_pressure(n, T, V):
    """Pressure of an ideal gas: n R T / V (T converted to kelvin)."""
    return (n * R_GAS * T.to("kelvin") / V).to("Pa")


@law("hydrostatic_pressure", ("rho", "h"))
def hydrostatic_pressure(rho, h):
    """Gauge pressure at depth h: rho g h."""
    return (rho * G * h).to("Pa")
