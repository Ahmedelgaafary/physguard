import math

import pytest

import physguard  # noqa: F401  (registers extra laws)
from physguard import Claim, Guard, Verdict
from physguard.laws import LAWS
from physguard.units import parse_quantity as P


def run(name, **inputs):
    return LAWS[name].func(**{k: P(v) for k, v in inputs.items()})


def test_decay_constant_cs137():
    lam = run("decay_constant", half_life="30.17 year")
    assert lam.to("1/s").magnitude == pytest.approx(math.log(2) / (30.17 * 365.25 * 86400), rel=1e-9)


def test_activity_three_half_lives():
    assert run("activity_after_time", A0="100 Bq", half_life="10 s", t="30 s").magnitude == pytest.approx(12.5)


def test_activity_accepts_curies():
    a = run("activity_after_time", A0="1 Ci", half_life="1 hour", t="1 hour")
    assert a.to("Ci").magnitude == pytest.approx(0.5)


def test_gamma_attenuation_one_mean_free_path():
    out = run("gamma_attenuation", I0="1000 mSv/h", mu="0.5 1/cm", x="2 cm")
    assert out.magnitude == pytest.approx(1000 / math.e)


def test_inverse_square():
    assert run("dose_rate_inverse_square", D1="100 mSv/h", r1="1 m", r2="2 m").magnitude == pytest.approx(25)


def test_four_factor():
    assert run("k_infinity_four_factor", eta="1.65", f="0.71", p="0.87", eps="1.02").magnitude == pytest.approx(
        1.03958, rel=1e-4
    )


def test_reactivity_and_pcm():
    rho = run("reactivity_from_keff", k_eff="1.01")
    assert rho.magnitude == pytest.approx(0.01 / 1.01)
    assert rho.to("pcm").magnitude == pytest.approx(990.099, rel=1e-4)


def test_reactor_period():
    assert run("reactor_power_exponential", P0="100 MW", T="20 s", t="10 s").to("MW").magnitude == pytest.approx(
        100 * math.exp(0.5)
    )


def test_way_wigner_one_hour_after_one_year():
    t, to = 3600.0, 365.25 * 86400
    expected = 3000e6 * 0.0622 * (t**-0.2 - (t + to) ** -0.2)
    out = run("decay_heat_way_wigner", P0="3000 MW", t="1 hour", T_op="1 year")
    assert out.to("W").magnitude == pytest.approx(expected, rel=1e-9)
    assert 0.01 < out.to("MW").magnitude / 3000 < 0.03  # roughly 1-3 % of full power


def test_fuel_centerline():
    out = run("fuel_centerline_temperature", Ts="600 K", q_vol="3e8 W/m**3", R="4.7 mm", k_th="3 W/m/K")
    assert out.to("K").magnitude == pytest.approx(600 + 3e8 * 4.7e-3**2 / 12, rel=1e-9)


def test_conduction_wall():
    assert run("conduction_plane_wall", k_th="0.8 W/m/K", A="10 m**2", dT="20 K", L="0.2 m").to("W").magnitude == (
        pytest.approx(800)
    )


def test_darcy():
    assert run("darcy_weisbach_pressure_drop", f="0.02", L="100 m", D="0.5 m", rho="1000 kg/m**3", v="3 m/s").to(
        "Pa"
    ).magnitude == pytest.approx(18000)


def test_hoop_stress_rpv():
    assert run("thin_wall_hoop_stress", P="15.5 MPa", r="2 m", t="0.2 m").to("MPa").magnitude == pytest.approx(155)


def test_three_phase():
    assert run("three_phase_power", V_ll="400 V", I="100 A", pf="0.9").to("W").magnitude == pytest.approx(
        math.sqrt(3) * 400 * 100 * 0.9
    )


def test_euler_buckling():
    out = run("euler_buckling_load", E="200 GPa", I="8e-6 m**4", L="3 m", K="1")
    assert out.to("N").magnitude == pytest.approx(math.pi**2 * 200e9 * 8e-6 / 9, rel=1e-9)


def test_guard_catches_wrong_decay_heat_prefix():
    claim = Claim(
        answer="40 kW", law="decay_heat_way_wigner", inputs={"P0": "3000 MW", "t": "1 hour", "T_op": "1 year"}
    )
    r = Guard().verify(claim)
    assert r.verdict == Verdict.FAIL
