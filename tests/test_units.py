"""The unit registry: prefixed symbols as before, and compound units built from
them, compatible whenever they measure the same quantity."""

from __future__ import annotations

import math

import pytest

from sigflow import units


@pytest.mark.parametrize(
    "unit, dimension, scale",
    [
        ("mV", "voltage", 1e-3),
        ("kHz", "frequency", 1e3),
        ("us", "time", 1e-6),
        ("deg", "angle", math.pi / 180),
        ("count", "dimensionless", 1.0),
        ("", "dimensionless", 1.0),
        ("volts", "voltage", 1.0),
        ("dBm", "level_dBm", 1.0),
        ("kg", "mass", 1.0),
        ("min", "time", 60.0),
        ("mrad", "angle", 1e-3),
        ("mdeg", "angle", 1e-3 * math.pi / 180),
        ("urad", "angle", 1e-6),
    ],
)
def test_simple_units_read_as_before(unit, dimension, scale):
    d, s = units.parse(unit)
    assert d == dimension and s == pytest.approx(scale)


@pytest.mark.parametrize(
    "unit, dimension, scale",
    [
        ("W*s", "energy", 1.0),
        ("N*m", "energy", 1.0),
        ("N m", "energy", 1.0),
        ("kg*m^2/s^2", "energy", 1.0),
        ("V*A", "power", 1.0),
        ("1/s", "frequency", 1.0),
        ("m/s", "velocity", 1.0),
        ("km/h", "velocity", 1000 / 3600),
        ("m/s^2", "acceleration", 1.0),
        ("m/s**2", "acceleration", 1.0),
        ("m·s^-2", "acceleration", 1.0),
        ("mA*h", "charge", 3.6),
        ("(m/s)*s", "length", 1.0),
        ("V*s", "m^2*kg/(A*s^2)", 1.0),
        ("V/Hz^(1/2)", "m^2*kg/(A*s^(5/2))", 1.0),
        ("deg/s", "rad/s", math.pi / 180),
        ("mrad/s", "rad/s", 1e-3),
    ],
)
def test_compound_units_measure_what_they_are_made_of(unit, dimension, scale):
    d, s = units.parse(unit)
    assert d == dimension and s == pytest.approx(scale)


def test_units_of_one_quantity_are_compatible_however_written():
    assert units.compatible("J", "W*s") and units.compatible("J", "N*m")
    assert units.compatible("Hz", "1/s")
    assert units.factor("W*s", "kJ") == pytest.approx(1e-3)
    assert units.factor("km/h", "m/s") == pytest.approx(1 / 3.6)
    assert units.factor("mrad", "rad") == pytest.approx(1e-3)
    assert units.factor("mdeg", "deg") == pytest.approx(1e-3)
    assert not units.compatible("V", "s")
    assert not units.compatible("rad", "1")  # angles stay their own quantity
    assert not units.compatible("b", "B")


def test_levels_and_unknown_symbols_do_not_combine():
    assert units.parse("dB*s")[0] == "unknown:dB*s"
    # a symbol no registry has, pint included
    assert units.parse("zorkmid")[0].startswith("unknown:")
    assert not units.compatible("zorkmid", "zorkmid/1")
    with pytest.raises(units.UnitError):
        units.factor("zorkmid", "m")


def test_pint_units_combine_with_the_built_in_ones():
    pytest.importorskip("pint")
    assert units.parse("inch/s") == (units.parse("m/s")[0], pytest.approx(0.0254))
    assert units.factor("inch/s", "m/s") == pytest.approx(0.0254)
    assert units.factor("mile/h", "km/h") == pytest.approx(1.609344)
    assert units.compatible("inch*V", "m*V")
    assert units.factor("rpm", "rad/s") == pytest.approx(2 * math.pi / 60)
    assert units.parse("furlong")[0] == "length"


@pytest.mark.parametrize(
    "unit, simple",
    [
        ("W*s", "J"),
        ("A*s", "C"),
        ("V*A", "W"),
        ("J/s", "W"),
        ("(m/s)*s", "m"),
        ("(m/s^2)*s", "m/s"),
        ("(rad/s)*s", "rad"),
        ("V*s", "V*s"),
        ("mV*s", "mV*s"),
        ("zorkmid*s", "zorkmid*s"),  # not readable: unchanged
    ],
)
def test_simplify_tidies_without_changing_the_value(unit, simple):
    assert units.simplify(unit) == simple
    if not simple.startswith("zorkmid"):
        assert units.factor(unit, simple) == pytest.approx(1.0)


def test_products_and_quotients_compose():
    assert units.product("V", "A") == "V*A"
    assert units.product("m/s", "s") == "(m/s)*s"
    assert units.quotient("m", "s") == "m/s"
    assert units.simplify(units.quotient(units.product("W", "s"), "s")) == "W"
