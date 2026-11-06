"""Units for signal types: a small SI registry, compatibility and conversion.

    Sig[float, N, None, Literal["mV"]]    # a type with units
    units.compatible("mV", "V")           # True (same dimension, different scale)
    units.factor("mV", "V")               # 0.001
    units.factor("W*s", "kJ")             # 0.001: compound units work too
    dsp.to_units("V")                     # a block that converts (and retypes) a stream

A unit is a symbol with an optional SI prefix (``mV``, ``kHz``, ``us``), or a
product, quotient and powers of them: ``V*s``, ``m/s^2``, ``kg*m^2/s^2``,
``W/Hz``, ``V/Hz^0.5``; ``*``, ``·`` and ``.`` multiply, ``/`` divides (left to
right: ``a/b*c`` is ``(a/b)*c``), ``^`` or ``**`` raise to a number, and
parentheses group. Each unit stands for powers of mass, length, time, current
and temperature (with angle, bits, bytes and symbols as quantities of their
own), so units of the same quantity are compatible however they are written:
``J``, ``W*s`` and ``N*m`` are all energy, ``Hz`` and ``1/s`` are both
frequency.

Unification treats *different but convertible* units as a mismatch with a hint
(insert ``to_units``), and *incompatible* units (V vs Hz) as a plain mismatch.
Logarithmic units (dB, dBm, dBV, dBFS, dBc) are their own dimensions and do not
combine with others. If ``pint`` is installed it reads the symbols this
registry does not know (``inch``, ``mile``, ``rpm``), in the same terms, so
they combine and convert with the built-in ones: ``units.factor("inch/s",
"m/s")`` is 0.0254.

:func:`product`, :func:`quotient` and :func:`simplify` build the units of a
computed quantity: the integral of a signal in ``W`` over time is in
``simplify(product("W", "s"))``, which is ``"J"``.
"""

from __future__ import annotations

import functools
import math
import re
from fractions import Fraction
from typing import Any

__all__ = [
    "UnitError",
    "compatible",
    "convert",
    "describe",
    "factor",
    "parse",
    "product",
    "quotient",
    "simplify",
]


class UnitError(ValueError):
    pass


# The quantities every unit is a product of powers of.
_M, _L, _T, _I, _K = "mass", "length", "time", "current", "temperature"
Dim = tuple[tuple[str, Fraction], ...]  # sorted (quantity, power), no zero powers


def _dim(**powers: int) -> Dim:
    return tuple(sorted((k, Fraction(v)) for k, v in powers.items() if v))


# symbol -> (dimension, scale to the SI unit of that dimension)
_UNITS: dict[str, tuple[Dim, float]] = {
    "s": (_dim(time=1), 1.0),
    "min": (_dim(time=1), 60.0),
    "h": (_dim(time=1), 3600.0),
    "m": (_dim(length=1), 1.0),
    "g": (_dim(mass=1), 1e-3),
    "A": (_dim(current=1), 1.0),
    "K": (_dim(temperature=1), 1.0),
    "Hz": (_dim(time=-1), 1.0),
    "N": (_dim(mass=1, length=1, time=-2), 1.0),
    "Pa": (_dim(mass=1, length=-1, time=-2), 1.0),
    "J": (_dim(mass=1, length=2, time=-2), 1.0),
    "W": (_dim(mass=1, length=2, time=-3), 1.0),
    "C": (_dim(current=1, time=1), 1.0),
    "V": (_dim(mass=1, length=2, time=-3, current=-1), 1.0),
    "ohm": (_dim(mass=1, length=2, time=-3, current=-2), 1.0),
    "Ω": (_dim(mass=1, length=2, time=-3, current=-2), 1.0),
    "rad": (_dim(angle=1), 1.0),
    "deg": (_dim(angle=1), math.pi / 180.0),
    "1": ((), 1.0),
    "": ((), 1.0),
    "count": ((), 1.0),
    "b": (_dim(bit=1), 1.0),
    "B": (_dim(byte=1), 1.0),
    "sym": (_dim(symbol=1), 1.0),
}
# Logarithmic levels: each its own dimension, never combined with other units.
_LEVELS: dict[str, str] = {
    "dB": "level",
    "dBm": "level_dBm",
    "dBV": "level_dBV",
    "dBFS": "level_dBFS",
    "dBc": "level_dBc",
}
_PREFIX: dict[str, float] = {
    "T": 1e12,
    "G": 1e9,
    "M": 1e6,
    "k": 1e3,
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
_NOT_PREFIXED = {"min", "h", "1", "", "count"}  # "mh" is no unit; "mrad" is

# names for the dimensions people talk about; the rest are spelled out
_NAMES: dict[Dim, str] = {
    (): "dimensionless",
    _dim(time=1): "time",
    _dim(time=-1): "frequency",
    _dim(length=1): "length",
    _dim(length=1, time=-1): "velocity",
    _dim(length=1, time=-2): "acceleration",
    _dim(mass=1): "mass",
    _dim(current=1): "current",
    _dim(temperature=1): "temperature",
    _dim(mass=1, length=1, time=-2): "force",
    _dim(mass=1, length=-1, time=-2): "pressure",
    _dim(mass=1, length=2, time=-2): "energy",
    _dim(mass=1, length=2, time=-3): "power",
    _dim(current=1, time=1): "charge",
    _dim(mass=1, length=2, time=-3, current=-1): "voltage",
    _dim(mass=1, length=2, time=-3, current=-2): "resistance",
    _dim(angle=1): "angle",
    _dim(bit=1): "bit",
    _dim(byte=1): "byte",
    _dim(symbol=1): "symbol",
}
# derived units a compound simplifies to when it is exactly one of them
_NAMED_RESULTS = ("J", "W", "C", "N")


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


class _Unit:
    """A parsed unit: the symbols it is written with and their powers, its
    dimension and its scale to SI."""

    def __init__(
        self, symbols: dict[str, Fraction], dim: dict[str, Fraction], scale: float
    ) -> None:
        self.symbols, self.dim, self.scale = symbols, dim, scale

    @classmethod
    def one(cls) -> _Unit:
        return cls({}, {}, 1.0)

    def times(self, other: _Unit, sign: int = 1) -> _Unit:
        symbols, dim = dict(self.symbols), dict(self.dim)
        for k, p in other.symbols.items():
            symbols[k] = symbols.get(k, Fraction(0)) + sign * p
        for k, p in other.dim.items():
            dim[k] = dim.get(k, Fraction(0)) + sign * p
        return _Unit(symbols, dim, self.scale * other.scale**sign)

    def power(self, p: Fraction) -> _Unit:
        return _Unit(
            {k: v * p for k, v in self.symbols.items()},
            {k: v * p for k, v in self.dim.items()},
            self.scale ** float(p),
        )

    @property
    def dimension(self) -> Dim:
        return tuple(sorted((k, v) for k, v in self.dim.items() if v))


class _Unknown(Exception):
    """A symbol this registry does not know, or a malformed unit."""


_TOKEN = re.compile(
    r"\s*(?:(?P<num>[-+]?\d+(?:\.\d+)?)|(?P<op>\*\*|[*·./^()])|(?P<sym>[A-Za-zµΩ]+))"
)


def _tokens(text: str) -> list[tuple[str, str]]:
    out, i = [], 0
    text = text.strip()
    while i < len(text):
        m = _TOKEN.match(text, i)
        if not m or m.end() == i:
            raise _Unknown(text)
        kind = m.lastgroup or ""
        out.append((kind, m.group(kind)))
        i = m.end()
    return out


def _symbol(sym: str) -> _Unit:
    sym = _ALIAS.get(sym.lower(), sym) if sym.lower() in _ALIAS else sym
    if sym in _LEVELS:
        raise _Unknown(f"{sym} is a level: it does not combine with other units")
    if sym in _UNITS:
        dim, scale = _UNITS[sym]
        return _Unit({sym: Fraction(1)}, dict(dim), scale)
    for p, s in _PREFIX.items():
        base = sym[len(p) :]
        if sym.startswith(p) and base in _UNITS and base not in _NOT_PREFIXED:
            dim, scale = _UNITS[base]
            return _Unit({sym: Fraction(1)}, dict(dim), s * scale)
    found = _from_pint(sym)
    if found is None:
        raise _Unknown(sym)
    return found


@functools.cache
def _pint_registry() -> Any:
    try:
        import pint  # type: ignore[import-not-found]
    except ImportError:
        return None
    return pint.UnitRegistry()


# pint's base quantities, in this registry's names
_FROM_PINT = {
    "[mass]": _M,
    "[length]": _L,
    "[time]": _T,
    "[current]": _I,
    "[temperature]": _K,
}


def _from_pint(text: str) -> _Unit | None:
    """A unit ``pint`` knows and this registry does not (``inch``, ``mile``,
    ``rpm``), in this registry's terms, so that it combines and compares with
    the built-in units: ``inch/s`` is a velocity like ``m/s``. None without
    pint, or if pint does not know it either."""
    reg = _pint_registry()
    if reg is None:
        return None
    try:
        q = reg(text)
        base = q.to_base_units()
        scale = float(base.magnitude)
    except Exception:
        return None
    dim: dict[str, Fraction] = {}
    for name, p in dict(q.dimensionality).items():
        key = _FROM_PINT.get(str(name), str(name).strip("[]"))  # substance...
        dim[key] = Fraction(p).limit_denominator(1000)
    angle = dict(base.units._units).get("radian")  # pint counts angles as 1
    if angle:
        dim["angle"] = Fraction(angle).limit_denominator(1000)
    return _Unit({text: Fraction(1)}, dim, scale)


class _Parser:
    def __init__(self, text: str) -> None:
        self.toks = _tokens(text)
        self.i = 0

    def peek(self) -> tuple[str, str] | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self) -> tuple[str, str]:
        tok = self.peek()
        if tok is None:
            raise _Unknown("unexpected end")
        self.i += 1
        return tok

    def parse(self) -> _Unit:
        u = self.expr()
        if self.peek() is not None:
            raise _Unknown(f"unexpected {self.peek()}")
        return u

    def expr(self) -> _Unit:
        u = self.term()
        while (tok := self.peek()) is not None and tok != ("op", ")"):
            if tok in (("op", "*"), ("op", "·"), ("op", ".")):
                self.take()
                u = u.times(self.term())
            elif tok == ("op", "/"):
                self.take()
                u = u.times(self.term(), -1)
            elif tok[0] == "sym" or tok == ("op", "("):  # "N m": side by side
                u = u.times(self.term())
            else:
                raise _Unknown(f"unexpected {tok[1]!r}")
        return u

    def term(self) -> _Unit:
        u = self.atom()
        if self.peek() in (("op", "^"), ("op", "**")):
            self.take()
            u = u.power(self.exponent())
        return u

    def exponent(self) -> Fraction:
        kind, text = self.take()
        if kind == "num":
            return Fraction(text)
        if (kind, text) == ("op", "("):  # ^(1/2)
            num = Fraction(self.take()[1])
            if self.peek() == ("op", "/"):
                self.take()
                num /= Fraction(self.take()[1])
            if self.take() != ("op", ")"):
                raise _Unknown("unclosed exponent")
            return num
        raise _Unknown(f"bad exponent {text!r}")

    def atom(self) -> _Unit:
        kind, text = self.take()
        if kind == "sym":
            return _symbol(text)
        if kind == "num" and Fraction(text) == 1:  # "1/s"
            return _Unit.one()
        if (kind, text) == ("op", "("):
            u = self.expr()
            if self.take() != ("op", ")"):
                raise _Unknown("unclosed parenthesis")
            return u
        raise _Unknown(f"unexpected {text!r}")


def _parse(unit: str) -> _Unit:
    u = unit.strip()
    if u.lower() in _ALIAS:
        u = _ALIAS[u.lower()]
    return _Parser(u).parse()


# how a dimension without a name is written: in SI base units
_SI = {
    "mass": "kg",
    "length": "m",
    "time": "s",
    "current": "A",
    "temperature": "K",
    "angle": "rad",
    "bit": "b",
    "byte": "B",
    "symbol": "sym",
}


def _name(dim: Dim) -> str:
    """``"voltage"`` for a dimension with a name, else its SI base units:
    ``"kg*m^2/(A*s^2)"``."""
    if dim in _NAMES:
        return _NAMES[dim]
    return _spell({_SI[k]: p for k, p in dim})


def _spell(powers: dict[str, Fraction]) -> str:
    """``a*b^2/(c*d)`` from symbol powers."""
    up = [k if p == 1 else f"{k}^{_power(p)}" for k, p in powers.items() if p > 0]
    down = [k if p == -1 else f"{k}^{_power(-p)}" for k, p in powers.items() if p < 0]
    text = "*".join(up) or "1"
    if down:
        text += "/" + (down[0] if len(down) == 1 else "(" + "*".join(down) + ")")
    return text


def _power(p: Fraction) -> str:
    return str(p.numerator) if p.denominator == 1 else f"({p})"


def parse(unit: str | None) -> tuple[str, float]:
    """``(dimension, scale-to-SI)`` for a unit string: ``"mV" -> ("voltage",
    1e-3)``, ``"W*s" -> ("energy", 1.0)``, ``"km/h" -> ("velocity", 0.2777...)``.
    A unit this registry cannot read gives ``("unknown:<unit>", 1.0)``."""
    if unit is None:
        return "unspecified", 1.0
    u = unit.strip()
    if u in _LEVELS:
        return _LEVELS[u], 1.0
    if not u:
        return "dimensionless", 1.0
    parsed: _Unit | None
    try:
        parsed = _parse(u)
    except (_Unknown, ZeroDivisionError, ValueError, OverflowError):
        parsed = _from_pint(u)  # spellings only pint reads, if it is installed
    if parsed is None:
        return f"unknown:{u}", 1.0
    return _name(parsed.dimension), parsed.scale


def compatible(a: str | None, b: str | None) -> bool:
    """Same dimension (possibly different scale)."""
    if a is None or b is None:
        return True
    da, db = parse(a)[0], parse(b)[0]
    return da == db and not da.startswith("unknown:")


def factor(src: str, dst: str) -> float:
    """Multiply a value in ``src`` by this to get ``dst``."""
    (da, sa), (db, sb) = parse(src), parse(dst)
    if da != db or da.startswith("unknown:"):
        raise UnitError(f"cannot convert {src!r} ({da}) to {dst!r} ({db})")
    return sa / sb


def convert(value: Any, src: str, dst: str) -> Any:
    return value * factor(src, dst)


def describe(unit: str | None) -> str:
    if unit is None:
        return "no units"
    d, s = parse(unit)
    return f"{unit} ({d}, x{s:g})"


# --------------------------------------------------------------------------- #
# Building units
# --------------------------------------------------------------------------- #


def _wrap(unit: str) -> str:
    return f"({unit})" if re.search(r"[*·./^ ]", unit.strip()) else unit.strip()


def product(a: str, b: str) -> str:
    """The units of a product: ``product("V", "A")`` is ``"V*A"``. Pass the
    result to :func:`simplify` for a tidier spelling."""
    return f"{_wrap(a)}*{_wrap(b)}"


def quotient(a: str, b: str) -> str:
    """The units of a quotient: ``quotient("m", "s")`` is ``"m/s"``."""
    return f"{_wrap(a)}/{_wrap(b)}"


def simplify(unit: str) -> str:
    """A tidier spelling of the same unit: symbols that appear more than once are
    combined (``"(m/s)*s"`` is ``"m"``, ``"(m/s^2)*s"`` is ``"m/s"``), and a
    compound that is exactly a joule, watt, coulomb or newton is written as one
    (``"W*s"`` is ``"J"``, ``"A*s"`` is ``"C"``). The value is unchanged: the
    result converts to ``unit`` with a factor of 1. A unit this registry cannot
    read is returned as it is."""
    try:
        parsed = _parse(unit)
    except (_Unknown, ZeroDivisionError, ValueError, OverflowError):
        return unit
    symbols = {k: p for k, p in parsed.symbols.items() if p}
    compound = len(symbols) != 1 or next(iter(symbols.values()), 1) != 1
    if compound:
        for name in _NAMED_RESULTS:
            dim, scale = _UNITS[name]
            if dim == parsed.dimension and math.isclose(scale, parsed.scale):
                return name
    return _spell(symbols)
