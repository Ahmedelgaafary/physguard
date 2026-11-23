import pytest

from physguard.units import UnitError, has_dimension, parse_quantity


def test_parse_with_prefix():
    assert parse_quantity("5 kN").to("N").magnitude == pytest.approx(5000)


def test_parse_rejects_garbage():
    with pytest.raises(UnitError):
        parse_quantity("five bananas")


def test_dimension_check():
    q = parse_quantity("3 m/s")
    assert has_dimension(q, "[velocity]")
    assert has_dimension(q, "km/h")
    assert not has_dimension(q, "[mass]")


def test_offset_units_parse_on_every_supported_pint_version():
    # regression: pint < 0.26 rejects '25 degC' when given as text; found by running CI on Python 3.10 and 3.11
    assert parse_quantity("25 degC").to("kelvin").magnitude == pytest.approx(298.15)
    assert parse_quantity("-40 degC").to("kelvin").magnitude == pytest.approx(233.15)
    assert parse_quantity("0 degC").to("degF").magnitude == pytest.approx(32.0)
    assert parse_quantity("1.5e2 kPa").to("Pa").magnitude == pytest.approx(150000.0)  # normal units are unaffected
    with pytest.raises(UnitError):
        parse_quantity("warm degC")


def test_below_absolute_zero_in_celsius_is_caught():
    from physguard import Claim, Guard, Verdict

    assert Guard().verify(Claim(answer="-300 degC")).verdict == Verdict.FAIL
