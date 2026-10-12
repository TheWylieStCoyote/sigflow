"""Units for signal types: a small SI registry, compatibility and conversion.

    Sig[float, "N", None, "mV"]           # a type with units
    units.compatible("mV", "V")           # True (same dimension, different scale)
    units.factor("mV", "V")               # 0.001
    dsp.to_units("V")                     # a block that converts (and retypes) a stream

Unification treats *different but convertible* units as a mismatch with a hint
(insert ``to_units``), and *incompatible* units (V vs Hz) as a plain mismatch.
Logarithmic units (dB, dBm, dBV, dBFS) are their own dimensions. If ``pint`` is
installed it is used for parsing anything this registry does not know.
"""

from __future__ import annotations

from typing import Any

__all__ = ["parse", "compatible", "factor", "convert", "describe", "UnitError"]


class UnitError(ValueError):
    pass


# base symbol -> dimension; scale is applied on top via SI prefixes
_BASE: dict[str, str] = {
    "V": "voltage",
    "A": "current",
    "W": "power",
    "Hz": "frequency",
    "s": "time",
    "m": "length",
    "rad": "angle",
    "deg": "angle",
    "1": "dimensionless",
    "": "dimensionless",
    "count": "dimensionless",
    "dB": "level",
    "dBm": "level_dBm",
    "dBV": "level_dBV",
    "dBFS": "level_dBFS",
    "dBc": "level_dBc",
    "K": "temperature",
    "Pa": "pressure",
    "g": "mass",
    "b": "bit",
    "B": "byte",
    "sym": "symbol",
}
_PREFIX: dict[str, float] = {
    "T": 1e12,
    "G": 1e9,
    "M": 1e6,
    "k": 1e3,
    "": 1.0,
    "m": 1e-3,
    "u": 1e-6,
    "µ": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
}
_ALIAS = {
    "volt": "V",
    "volts": "V",
    "hertz": "Hz",
    "sec": "s",
    "seconds": "s",
    "degrees": "deg",
    "radians": "rad",
    "samples": "1",
    "sample": "1",
}
_SPECIAL_SCALE = {"deg": 3.141592653589793 / 180.0}  # deg expressed in rad


def parse(unit: str | None) -> tuple[str, float]:
    """``(dimension, scale-to-base)`` for a unit string, e.g. ``"mV" -> ("voltage", 1e-3)``."""
    if unit is None:
        return "unspecified", 1.0
    u = unit.strip()
    u = _ALIAS.get(u.lower(), u)
    if u in _BASE:
        return _BASE[u], _SPECIAL_SCALE.get(u, 1.0)
    for p, scale in _PREFIX.items():
        if (
            p
            and u.startswith(p)
            and u[len(p) :] in _BASE
            and not u[len(p) :].startswith("dB")
        ):
            base = u[len(p) :]
            return _BASE[base], scale * _SPECIAL_SCALE.get(base, 1.0)
    # exponent / compound forms via pint when available
    try:
        import pint

        q = pint.UnitRegistry()(u)
        return str(q.dimensionality), float(q.to_base_units().magnitude)
    except Exception:
        return f"unknown:{u}", 1.0


def compatible(a: str | None, b: str | None) -> bool:
    """Same dimension (possibly different scale)."""
    if a is None or b is None:
        return True
    return parse(a)[0] == parse(b)[0]


def factor(src: str, dst: str) -> float:
    """Multiply a value in ``src`` by this to get ``dst``."""
    (da, sa), (db, sb) = parse(src), parse(dst)
    if da != db:
        raise UnitError(f"cannot convert {src!r} ({da}) to {dst!r} ({db})")
    return sa / sb


def convert(value: Any, src: str, dst: str) -> Any:
    return value * factor(src, dst)


def describe(unit: str | None) -> str:
    if unit is None:
        return "no units"
    d, s = parse(unit)
    return f"{unit} ({d}, x{s:g})"
