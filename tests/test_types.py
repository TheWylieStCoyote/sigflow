# mypy: disable-error-code="valid-type, misc, syntax, return-value, union-attr, index"
from typing import Literal
import numpy as np
import pytest

from sigflow import Any, Env, Sig, TupleType, TypeMismatch, unify
from sigflow.dims import M, N, T
from sigflow.types import DVar, scoped


def test_sig_construction_and_display():
    t = Sig[np.complex64, N]
    assert t.dtype == np.dtype("complex64")
    assert t.shape == ("N",)
    assert str(t) == "Sig[complex64, (N,)]"
    assert (
        str(Sig[float, (M, N), 48000, Literal["V"]])
        == "Sig[float64, (M, N), 48000Hz, 'V']"
    )
    assert str(Sig[Any]) == "Sig[Any]"
    assert Sig[dict].dtype is dict
    assert isinstance(Sig[T, N].dtype, DVar)


def test_unify_concrete_and_symbolic():
    env = unify(Sig[float, (1024,)], Sig[float, N])
    assert env.bindings["N"] == 1024
    env = unify(Sig[float, N], Sig[float, M], env)
    # N and M are now the same symbol
    assert env.value("M") == 1024


def test_unify_conflicts():
    with pytest.raises(TypeMismatch):
        unify(Sig[float, (10,)], Sig[float, (20,)])
    with pytest.raises(TypeMismatch):
        unify(Sig[float, N], Sig[float, (M, N)])
    with pytest.raises(TypeMismatch):
        unify(Sig[complex, N], Sig[float, N])  # complex cannot feed float
    unify(Sig[float, N], Sig[complex, N])  # float -> complex is a safe cast
    with pytest.raises(TypeMismatch):
        unify(Sig[float, N, 48000], Sig[float, N, 44100])
    with pytest.raises(TypeMismatch):
        unify(Sig[float, N, None, Literal["V"]], Sig[float, N, None, Literal["A"]])


def test_gradual_any_and_python_types():
    unify(Sig[Any], Sig[float, N])
    unify(Sig[float, N], Sig[Any])
    unify(Sig[dict], Sig[dict])
    with pytest.raises(TypeMismatch):
        unify(Sig[list], Sig[dict])


def test_dtype_variables():
    env = unify(Sig[float, N], Sig[T, N])
    assert env.value("T") == np.dtype(float)
    with pytest.raises(TypeMismatch):
        unify(Sig[T, N], Sig[int, N], env)  # T=float64 cannot feed int


def test_tuple_types():
    t = TupleType[Sig[float, N], Sig[float, N]]
    env = unify(TupleType[Sig[float, (4,)], Sig[float, (4,)]], t)
    assert env.bindings["N"] == 4
    with pytest.raises(TypeMismatch):
        unify(TupleType[Sig[float, (4,)], Sig[float, (5,)]], t)
    with pytest.raises(TypeMismatch):
        unify(TupleType[Sig[float, (4,)]], t)


def test_scoping_keeps_symbols_local():
    a = scoped(Sig[float, N], "1")
    b = scoped(Sig[float, N], "2")
    env = Env()
    env.bind(a.shape[0], 10)
    env.bind(b.shape[0], 20)  # different scopes: no conflict
    assert str(a) == "Sig[float64, (N,)]"


def test_runtime_check():
    t = Sig[float, N]
    env = t.check(np.zeros(8))  # type: ignore[attr-defined]
    assert env.bindings["N"] == 8
    with pytest.raises(TypeMismatch):
        Sig[float, (4,)].check(np.zeros(8))  # type: ignore[attr-defined]
    with pytest.raises(TypeMismatch):
        Sig[float, N].check(np.zeros((2, 2)))  # type: ignore[attr-defined]
    with pytest.raises(TypeMismatch):
        Sig[dict].check([1])  # type: ignore[attr-defined]


def test_dimension_symbols_are_the_string_spelling():
    from typing import Literal

    from sigflow import dims
    from sigflow.dims import K

    assert N == "N" and dims.K is K == "K"
    assert Sig[float, N] == Sig[float, "N"]
    assert Sig[T, (M, N)] == Sig["T", ("M", "N")]
    assert Sig[float, N, 48_000, Literal["mV"]] == Sig[float, "N", 48_000, "mV"]  # type: ignore[name-defined]
    assert Sig[float, N, Literal[48_000], Literal["mV"]].rate == 48_000  # type: ignore[attr-defined]
    assert str(Sig[complex, (M, N)]) == "Sig[complex128, (M, N)]"
