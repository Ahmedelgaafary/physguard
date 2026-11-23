"""Law packs: nuclear, thermal, mechanical, electrical and civil engineering.

Importing this module registers the laws in `physguard.laws.LAWS`.
Inputs may use any compatible unit; results are returned in SI-style units.
Temperature DIFFERENCES must be given in kelvin (e.g. '20 K'), not degC.
"""
from __future__ import annotations

import math

from .laws import law

# ----------------------------------------------------------------------------- nuclear


@law("decay_constant", ("half_life",))
def decay_constant(half_life):
    """Radioactive decay constant: lambda = ln(2) / T_half."""
    return (math.log(2) / half_life).to("1/s")


@law("activity_after_time", ("A0", "half_life", "t"))
def activity_after_time(A0, half_life, t):
    """Activity after time t: A0 * 2^(-t / T_half)."""
    n = (t / half_life).to("dimensionless").magnitude
    return (A0 * 0.5**n).to("Bq")


@law("gamma_attenuation", ("I0", "mu", "x"))
def gamma_attenuation(I0, mu, x):
    """Narrow-beam attenuation through a shield: I = I0 exp(-mu x). Result keeps the units of I0."""
    return I0 * math.exp(-(mu * x).to("dimensionless").magnitude)


@law("dose_rate_inverse_square", ("D1", "r1", "r2"))
def dose_rate_inverse_square(D1, r1, r2):
    """Point-source dose rate at distance r2 given rate D1 at r1: D1 (r1/r2)^2 (no shielding/scatter)."""
    return D1 * ((r1 / r2).to("dimensionless").magnitude) ** 2


@law("k_infinity_four_factor", ("eta", "f", "p", "eps"))
def k_infinity_four_factor(eta, f, p, eps):
    """Infinite-medium multiplication factor: k_inf = eta * f * p * epsilon."""
    return (eta * f * p * eps).to("dimensionless")


@law("reactivity_from_keff", ("k_eff",))
def reactivity_from_keff(k_eff):
    """Reactivity: rho = (k_eff - 1) / k_eff (dimensionless; 1 pcm = 1e-5)."""
    return ((k_eff - 1) / k_eff).to("dimensionless")


@law("reactor_power_exponential", ("P0", "T", "t"))
def reactor_power_exponential(P0, T, t):
    """Power during a steady exponential transient with period T: P = P0 exp(t / T)."""
    return (P0 * math.exp((t / T).to("dimensionless").magnitude)).to("W")


@law("decay_heat_way_wigner", ("P0", "t", "T_op"))
def decay_heat_way_wigner(P0, t, T_op):
    """Decay heat after shutdown (Way-Wigner approximation, roughly 10 s < t < 100 days):
    P/P0 = 0.0622 (t^-0.2 - (t + T_op)^-0.2), t = time since shutdown, T_op = operating time."""
    ts = t.to("s").magnitude
    to = T_op.to("s").magnitude
    return (P0 * 0.0622 * (ts**-0.2 - (ts + to) ** -0.2)).to("W")


@law("fuel_centerline_temperature", ("Ts", "q_vol", "R", "k_th"))
def fuel_centerline_temperature(Ts, q_vol, R, k_th):
    """Centerline temperature of a cylindrical fuel pellet with uniform heat generation and
    constant conductivity: T0 = Ts + q''' R^2 / (4 k)."""
    return (Ts.to("kelvin") + (q_vol * R**2 / (4 * k_th)).to("kelvin")).to("kelvin")


# ----------------------------------------------------------------------------- thermal


@law("conduction_plane_wall", ("k_th", "A", "dT", "L"))
def conduction_plane_wall(k_th, A, dT, L):
    """Steady 1D conduction through a plane wall: Q = k A dT / L (dT in kelvin)."""
    return (k_th * A * dT / L).to("W")


# ----------------------------------------------------------------------------- mechanical


@law("darcy_weisbach_pressure_drop", ("f", "L", "D", "rho", "v"))
def darcy_weisbach_pressure_drop(f, L, D, rho, v):
    """Pipe pressure drop: dp = f (L/D) rho v^2 / 2."""
    return (f * (L / D) * rho * v**2 / 2).to("Pa")


@law("thin_wall_hoop_stress", ("P", "r", "t"))
def thin_wall_hoop_stress(P, r, t):
    """Hoop stress in a thin-walled cylinder: sigma = P r / t (valid for r/t > ~10)."""
    return (P * r / t).to("Pa")


# ----------------------------------------------------------------------------- electrical


@law("three_phase_power", ("V_ll", "I", "pf"))
def three_phase_power(V_ll, I, pf):  # noqa: E741
    """Real power of a balanced three-phase load: P = sqrt(3) V_LL I pf."""
    return (math.sqrt(3) * V_ll * I * pf).to("W")


# ----------------------------------------------------------------------------- civil


@law("euler_buckling_load", ("E", "I", "L", "K"))
def euler_buckling_load(E, I, L, K):  # noqa: E741
    """Euler critical column load: P = pi^2 E I / (K L)^2 (K = effective length factor)."""
    return (math.pi**2 * E * I / (K * L) ** 2).to("N")
