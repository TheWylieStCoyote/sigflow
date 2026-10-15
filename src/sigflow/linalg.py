"""Linear-algebra primitives as shape-checked blocks.

Symbolic dimensions make the contracts explicit and checkable at graph
construction::

    matmul : (M x K, K x N) -> M x N
    solve  : (N x N, N)     -> N

so ``(a & b) >> matmul`` fails to build if the inner dimensions cannot agree.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from .block import block
from .types import Sig
from .dims import K, M, N, T

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

Vector = Sig[T, tuple[N]]
Matrix = Sig[T, tuple[M, N]]
Square = Sig[T, tuple[N, N]]


@block
def matmul(a: Sig[T, tuple[M, K]], b: Sig[T, tuple[K, N]]) -> Sig[T, tuple[M, N]]:
    return a @ b


@block
def matvec(a: Sig[T, tuple[M, N]], x: Sig[T, tuple[N]]) -> Sig[T, tuple[M]]:
    return a @ x


@block
def dot(a: Sig[T, tuple[N]], b: Sig[T, tuple[N]]) -> Sig[T, tuple[()]]:
    return np.vdot(a, b)


@block
def outer(a: Sig[T, tuple[M]], b: Sig[T, tuple[N]]) -> Sig[T, tuple[M, N]]:
    return np.outer(a, b)


@block
def transpose(a: Sig[T, tuple[M, N]]) -> Sig[T, tuple[N, M]]:
    return a.T


@block
def hermitian(a: Sig[T, tuple[M, N]]) -> Sig[T, tuple[N, M]]:
    return a.conj().T


@block
def inv(a: Sig[T, tuple[N, N]]) -> Sig[T, tuple[N, N]]:
    return np.linalg.inv(a)


@block
def solve(a: Sig[T, tuple[N, N]], b: Sig[T, tuple[N]]) -> Sig[T, tuple[N]]:
    return np.linalg.solve(a, b)


@block
def lstsq(a: Sig[T, tuple[M, N]], b: Sig[T, tuple[M]]) -> Sig[T, tuple[N]]:
    return np.linalg.lstsq(a, b, rcond=None)[0]


@block
def eig(
    a: Sig[T, tuple[N, N]],
) -> tuple[Sig[complex, tuple[N]], Sig[complex, tuple[N, N]]]:
    w, v = np.linalg.eig(a)
    return w.astype(complex), v.astype(complex)


@block
def eigh(a: Sig[T, tuple[N, N]]) -> tuple[Sig[float, tuple[N]], Sig[T, tuple[N, N]]]:
    w, v = np.linalg.eigh(a)
    return w, v


@block
def svd(
    a: Sig[T, tuple[M, N]], full: bool = False
) -> tuple[Sig[T, tuple[M, K]], Sig[float, tuple[K]], Sig[T, tuple[K, N]]]:
    return np.linalg.svd(a, full_matrices=full)


@block
def norm(x: Sig[T, None], ord: Any = None) -> Sig[float, tuple[()]]:
    return float(np.linalg.norm(x, ord=ord))


@block
def normalize(x: Sig[T, tuple[N]]) -> Sig[T, tuple[N]]:
    n = np.linalg.norm(x)
    return x / n if n else x


@block
def project(x: Sig[T, tuple[N]], basis: Sig[T, tuple[N, K]]) -> Sig[T, tuple[K]]:
    """Coefficients of ``x`` in an (orthonormal) ``basis``: basis^H x."""
    return basis.conj().T @ x


@block
def covariance(x: Sig[T, tuple[M, N]]) -> Sig[T, tuple[N, N]]:
    """Sample covariance of M observations of an N-vector (channels on the last axis):
    R[i, j] = E[x_i conj(x_j)], the same convention as ``np.cov``."""
    xc = x - x.mean(axis=0, keepdims=True)
    return xc.T @ xc.conj() / max(x.shape[0] - 1, 1)


@block
def gram(a: Sig[T, tuple[M, N]]) -> Sig[T, tuple[N, N]]:
    return a.conj().T @ a
