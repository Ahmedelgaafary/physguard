import pint
import pytest

from physguard.laws import LAWS
from physguard.units import parse_quantity as P


def test_cantilever_known_value():
    # F=1 kN, L=2 m, E=200 GPa, I=8e-6 m^4 -> 8000/(3*200e9*8e-6)=1.6667e-3 m
    d = LAWS["cantilever_tip_deflection"].func(P("1 kN"), P("2 m"), P("200 GPa"), P("8e-6 m**4"))
    assert d.to("m").magnitude == pytest.approx(1.6667e-3, rel=1e-3)


def test_ohms_law():
    assert LAWS["ohms_law_current"].func(P("12 V"), P("4 ohm")).to("A").magnitude == pytest.approx(3)


def test_projectile_45_degrees():
    r = LAWS["projectile_range"].func(P("20 m/s"), P("45"))
    assert r.to("m").magnitude == pytest.approx(400 / 9.80665, rel=1e-6)


def test_ideal_gas_accepts_celsius():
    p = LAWS["ideal_gas_pressure"].func(P("1 mol"), P("0 degC"), P("22.414 L"))
    assert p.to("kPa").magnitude == pytest.approx(101.325, rel=1e-3)


def test_wrong_input_dimension_is_caught_by_pint():
    with pytest.raises(pint.DimensionalityError):
        LAWS["ohms_law_current"].func(P("12 kg"), P("4 ohm"))
