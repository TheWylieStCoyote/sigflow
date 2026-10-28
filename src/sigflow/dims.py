"""Dimension symbols for :class:`~sigflow.types.Sig` annotations.

``Sig[float, N]`` and ``Sig[float, "N"]`` build the same contract: at runtime each
symbol *is* its one-letter string, so unification, ``describe()`` and scoping work
exactly as for the string spelling. The difference is what a static checker sees:
a string inside a type expression is a forward reference (``"N" is not defined``),
while ``N`` imported from here is a class, so block signatures type-check as
written. ``T``-style dtype variables are spelled the same way: ``Sig[T, N]``.

The whole uppercase alphabet is provided; ``from sigflow import N, M`` or
``from sigflow.dims import N, M``. If a letter is already a name in your module
(``N = 1024``), use the qualified form ``dims.N`` in the annotation instead.
"""

from typing import TYPE_CHECKING

__all__ = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

if TYPE_CHECKING:
    # one declaration per line: a tuple-unpacking assignment in the runtime branch
    # is not exported as a symbol by pyright
    class A: ...

    class B: ...

    class C: ...

    class D: ...

    class E: ...

    class F: ...

    class G: ...

    class H: ...

    class I: ...

    class J: ...

    class K: ...

    class L: ...

    class M: ...

    class N: ...

    class O: ...

    class P: ...

    class Q: ...

    class R: ...

    class S: ...

    class T: ...

    class U: ...

    class V: ...

    class W: ...

    class X: ...

    class Y: ...

    class Z: ...
else:
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"
    H = "H"
    I = "I"
    J = "J"
    K = "K"
    L = "L"
    M = "M"
    N = "N"
    O = "O"
    P = "P"
    Q = "Q"
    R = "R"
    S = "S"
    T = "T"
    U = "U"
    V = "V"
    W = "W"
    X = "X"
    Y = "Y"
    Z = "Z"
