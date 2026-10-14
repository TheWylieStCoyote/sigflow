"""Stream-rate contracts: how many frames a block emits per frame it consumes.

    @block(ratio="1:1")     # one out per in (what map blocks and taps are by definition)
    @block(ratio="N:1")     # fewer out than in: a decimator, a windowing block, a filter
    @block(ratio="1:N")     # more out than in: an interpolator, a splitter
    @block(ratio=(4, 1))    # exactly one out per four in
    @block(ratio="any")     # no promise (the default for stream blocks)
    @block(ratio=lambda n, n_in=None, **_: Ratio.of(n_in, n) if n_in else Ratio.ANY)

A callable receives the block's configuration plus ``n_in``, the leading dimension of
the actual input type when it is concrete (so a re-framer can state its ratio once it
knows the input frame length). Ratios compose along a pipeline (they multiply) and are
compared across the branches of a fan-out or zip: two branches with *different exact*
ratios cannot be paired, and the graph refuses to build instead of silently stopping at
the shortest branch at run time.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

__all__ = ["Ratio", "RatioMismatch", "as_ratio"]


class RatioMismatch(TypeError):
    """Branches of a fan-out / zip produce frames at incompatible rates."""


class Ratio:
    """Outputs per input frame: an exact fraction, or a bound (``N:1`` fewer-or-equal,
    ``1:N`` more-or-equal), or ``any``."""

    __slots__ = ("value", "kind")

    def __init__(self, value: Fraction | None = None, kind: str = "exact") -> None:
        if value is not None:
            kind = "exact"
        elif kind not in ("down", "up", "any"):
            raise ValueError(f"bad ratio kind {kind!r}")
        self.value = value
        self.kind = kind

    @classmethod
    def of(cls, n_in: int | float | Fraction, n_out: int | float | Fraction) -> "Ratio":
        """``n_out`` frames for every ``n_in`` frames (``Ratio.of(4, 1)`` is ``"4:1"``)."""
        return cls(Fraction(n_out) / Fraction(n_in))

    # -- algebra ------------------------------------------------------------ #
    def __mul__(self, other: "Ratio") -> "Ratio":
        a, b = self, other
        if a.kind == "any" or b.kind == "any":
            return Ratio.ANY
        if a.kind == "exact" and b.kind == "exact":
            return Ratio(a.value * b.value)  # type: ignore[operator]
        if a.kind == "exact":
            a, b = b, a  # a is now the bound, b exact
        if b.kind == "exact":
            v = b.value
            assert v is not None
            if v == 1:
                return a
            if (a.kind == "down" and v < 1) or (a.kind == "up" and v > 1):
                return a
            return Ratio.ANY
        return a if a.kind == b.kind else Ratio.ANY

    def compatible(self, other: "Ratio") -> bool:
        """Could two branches with these ratios emit the same number of frames?"""
        a, b = self, other
        if a.kind == "any" or b.kind == "any":
            return True
        if a.kind == "exact" and b.kind == "exact":
            return a.value == b.value
        if a.kind != "exact":
            a, b = b, a
        if b.kind == "exact":
            return a.value == b.value
        v = a.value
        assert v is not None
        return v <= 1 if b.kind == "down" else v >= 1

    def meet(self, other: "Ratio") -> "Ratio":
        """The more specific of two compatible ratios."""
        if self.kind == "exact":
            return self
        if other.kind == "exact":
            return other
        return self if self.kind != "any" else other

    # -- display ------------------------------------------------------------ #
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Ratio)
            and self.kind == other.kind
            and self.value == other.value
        )

    def __hash__(self) -> int:
        return hash((self.kind, self.value))

    def __bool__(self) -> bool:
        return self.kind != "any"

    def __str__(self) -> str:
        if self.kind == "exact":
            v = self.value
            assert v is not None
            return f"{v.denominator}:{v.numerator}"
        return {"down": "N:1", "up": "1:N", "any": "any"}[self.kind]

    def __repr__(self) -> str:
        return f"Ratio({self})"

    def to_json(self) -> str:
        return str(self)

    ONE: "Ratio"
    ANY: "Ratio"
    DOWN: "Ratio"
    UP: "Ratio"


Ratio.ONE = Ratio(Fraction(1))
Ratio.ANY = Ratio(kind="any")
Ratio.DOWN = Ratio(kind="down")
Ratio.UP = Ratio(kind="up")


def as_ratio(spec: Any) -> Ratio:
    """``"4:1"``, ``"N:1"``, ``"1:N"``, ``"any"``, ``(4, 1)``, an int / Fraction (outputs per
    input), or a :class:`Ratio`."""
    if isinstance(spec, Ratio):
        return spec
    if spec is None:
        return Ratio.ANY
    if isinstance(spec, str):
        s = spec.strip().replace(" ", "")
        if s in ("any", "*"):
            return Ratio.ANY
        if s.upper() == "N:1":
            return Ratio.DOWN
        if s.upper() == "1:N":
            return Ratio.UP
        if ":" in s:
            i, o = s.split(":", 1)
            return Ratio.of(int(i), int(o))
        return Ratio(Fraction(s))
    if isinstance(spec, tuple) and len(spec) == 2:
        return Ratio.of(spec[0], spec[1])
    if isinstance(spec, (int, Fraction, float)):
        return Ratio(Fraction(spec))
    raise TypeError(f"cannot read a stream ratio from {spec!r}")


def leading_dim(t: Any) -> int | None:
    """The concrete leading dimension of a signal type's shape, or None."""
    shape = getattr(t, "shape", None)
    if isinstance(shape, tuple) and shape and isinstance(shape[0], int):
        return shape[0]
    return None
