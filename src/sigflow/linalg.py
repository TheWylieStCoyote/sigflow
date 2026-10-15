"""Linear-algebra primitives as shape-checked blocks.

Symbolic dimensions make the contracts explicit and checkable at graph
construction::

    matmul : (M x K, K x N) -> M x N
    solve  : (N x N, N)     -> N

so ``(a & b) >> matmul`` fails to build if the inner dimensions cannot agree.
"""

from __future__ import annotations
# pyright: reportUndefinedVariable=false, reportInvalidTypeForm=false, reportReturnType=false, reportGeneralTypeIssues=false

from typing import Any

import numpy as np

from .block import block
from .types import Sig

__all__ = [
    "matmul",
    "matvec",
    "dot",
    "outer",
    "transpose",
    "inv",
    "solve",
    "lstsq",
    "eig",
    "eigh",
    "svd",
    "norm",
    "normalize",
    "project",
    "covariance",
    "gram",
    "hermitian",
    "Vector",
    "Matrix",
    "Square",
]

Vector = Sig["T", ("N",)]
Matrix = Sig["T", ("M", "N")]
Square = Sig["T", ("N", "N")]


@block
def matmul(a: Sig["T", ("M", "K")], b: Sig["T", ("K", "N")]) -> Sig["T", ("M", "N")]:
    return a @ b


@block
def matvec(a: Sig["T", ("M", "N")], x: Sig["T", ("N",)]) -> Sig["T", ("M",)]:
    return a @ x


@block
def dot(a: Sig["T", ("N",)], b: Sig["T", ("N",)]) -> Sig["T", ()]:
    return np.vdot(a, b)


@block
def outer(a: Sig["T", ("M",)], b: Sig["T", ("N",)]) -> Sig["T", ("M", "N")]:
    return np.outer(a, b)


@block
def transpose(a: Sig["T", ("M", "N")]) -> Sig["T", ("N", "M")]:
    return a.T


@block
def hermitian(a: Sig["T", ("M", "N")]) -> Sig["T", ("N", "M")]:
    return a.conj().T


@block
def inv(a: Sig["T", ("N", "N")]) -> Sig["T", ("N", "N")]:
    return np.linalg.inv(a)


@block
def solve(a: Sig["T", ("N", "N")], b: Sig["T", ("N",)]) -> Sig["T", ("N",)]:
    return np.linalg.solve(a, b)


@block
def lstsq(a: Sig["T", ("M", "N")], b: Sig["T", ("M",)]) -> Sig["T", ("N",)]:
    return np.linalg.lstsq(a, b, rcond=None)[0]


@block
def eig(a: Sig["T", ("N", "N")]) -> (Sig[complex, ("N",)], Sig[complex, ("N", "N")]):
    w, v = np.linalg.eig(a)
    return w.astype(complex), v.astype(complex)


@block
def eigh(a: Sig["T", ("N", "N")]) -> (Sig[float, ("N",)], Sig["T", ("N", "N")]):
    w, v = np.linalg.eigh(a)
    return w, v


@block
def svd(
    a: Sig["T", ("M", "N")], full: bool = False
) -> (Sig["T", ("M", "K")], Sig[float, ("K",)], Sig["T", ("K", "N")]):
    return np.linalg.svd(a, full_matrices=full)


@block
def norm(x: Sig["T", None], ord: Any = None) -> Sig[float, ()]:
    return float(np.linalg.norm(x, ord=ord))


@block
def normalize(x: Sig["T", ("N",)]) -> Sig["T", ("N",)]:
    n = np.linalg.norm(x)
    return x / n if n else x


@block
def project(x: Sig["T", ("N",)], basis: Sig["T", ("N", "K")]) -> Sig["T", ("K",)]:
    """Coefficients of ``x`` in an (orthonormal) ``basis``: basis^H x."""
    return basis.conj().T @ x


@block
def covariance(x: Sig["T", ("M", "N")]) -> Sig["T", ("N", "N")]:
    """Sample covariance of M observations of an N-vector (channels on the last axis):
    R[i, j] = E[x_i conj(x_j)], the same convention as ``np.cov``."""
    xc = x - x.mean(axis=0, keepdims=True)
    return xc.T @ xc.conj() / max(x.shape[0] - 1, 1)


@block
def gram(a: Sig["T", ("M", "N")]) -> Sig["T", ("N", "N")]:
    return a.conj().T @ a
