from typing import Literal
import numpy as np
import pytest

from sigflow import Any, Env, Sig, TupleType, TypeMismatch, unify
from sigflow import SignalType
from sigflow.dims import M, N, T
from sigflow import SignalType
from sigflow.types import DVar, scoped


def test_sig_construction_and_display():
    t = SignalType[np.complex64, N]
    assert t.dtype == np.dtype("complex64")
    assert t.shape == ("N",)
    assert str(t) == "Sig[complex64, (N,)]"
    assert (
        str(SignalType[float, tuple[M, N], Literal[48000], Literal["V"]])
        == "Sig[float64, (M, N), 48000Hz, 'V']"
    )
    assert str(SignalType[Any]) == "Sig[Any]"
    assert SignalType[dict].dtype is dict
    assert isinstance(SignalType[T, N].dtype, DVar)


def test_unify_concrete_and_symbolic():
    env = unify(SignalType[float, tuple[Literal[1024]]], SignalType[float, N])
    assert env.bindings["N"] == 1024
    env = unify(SignalType[float, N], SignalType[float, M], env)
    # N and M are now the same symbol
    assert env.value("M") == 1024


def test_unify_conflicts():
    with pytest.raises(TypeMismatch):
        unify(
            SignalType[float, tuple[Literal[10]]], SignalType[float, tuple[Literal[20]]]
        )
    with pytest.raises(TypeMismatch):
        unify(SignalType[float, N], SignalType[float, tuple[M, N]])
    with pytest.raises(TypeMismatch):
        unify(SignalType[complex, N], SignalType[float, N])  # complex cannot feed float
    unify(
        SignalType[float, N], SignalType[complex, N]
    )  # float -> complex is a safe cast
    with pytest.raises(TypeMismatch):
        unify(
            SignalType[float, N, Literal[48000]], SignalType[float, N, Literal[44100]]
        )
    with pytest.raises(TypeMismatch):
        unify(
            SignalType[float, N, None, Literal["V"]],
            SignalType[float, N, None, Literal["A"]],
        )


def test_gradual_any_and_python_types():
    unify(SignalType[Any], SignalType[float, N])
    unify(SignalType[float, N], SignalType[Any])
    unify(SignalType[dict], SignalType[dict])
    with pytest.raises(TypeMismatch):
        unify(SignalType[list], SignalType[dict])


def test_dtype_variables():
    env = unify(SignalType[float, N], SignalType[T, N])
    assert env.value("T") == np.dtype(float)
    with pytest.raises(TypeMismatch):
        unify(SignalType[T, N], SignalType[int, N], env)  # T=float64 cannot feed int


def test_tuple_types():
    t = TupleType[SignalType[float, N], SignalType[float, N]]
    env = unify(
        TupleType[
            SignalType[float, tuple[Literal[4]]], SignalType[float, tuple[Literal[4]]]
        ],
        t,
    )
    assert env.bindings["N"] == 4
    with pytest.raises(TypeMismatch):
        unify(
            TupleType[
                SignalType[float, tuple[Literal[4]]],
                SignalType[float, tuple[Literal[5]]],
            ],
            t,
        )
    with pytest.raises(TypeMismatch):
        unify(TupleType[SignalType[float, tuple[Literal[4]]]], t)


def test_scoping_keeps_symbols_local():
    a = scoped(SignalType[float, N], "1")
    b = scoped(SignalType[float, N], "2")
    env = Env()
    env.bind(a.shape[0], 10)
    env.bind(b.shape[0], 20)  # different scopes: no conflict
    assert str(a) == "Sig[float64, (N,)]"


def test_runtime_check():
    t = SignalType(float, (N,))
    env = t.check(np.zeros(8))
    assert env.bindings["N"] == 8
    with pytest.raises(TypeMismatch):
        SignalType(float, (4,)).check(np.zeros(8))
    with pytest.raises(TypeMismatch):
        SignalType(float, (N,)).check(np.zeros((2, 2)))
    with pytest.raises(TypeMismatch):
        SignalType(dict).check([1])


def test_dimension_symbols_are_the_string_spelling():
    from typing import Literal

    from sigflow import dims
    from sigflow.dims import K

    assert N == "N" and dims.K is K == "K"
    assert SignalType[float, N] == SignalType[float, "N"]
    assert SignalType[T, tuple[M, N]] == SignalType["T", tuple["M", "N"]]
    old = SignalType[float, "N", Literal[48_000], "mV"]  # type: ignore[name-defined]
    assert SignalType[float, N, Literal[48_000], Literal["mV"]] == old
    assert SignalType[float, N, Literal[48_000], Literal["mV"]].rate == 48_000
    assert str(SignalType[complex, tuple[M, N]]) == "Sig[complex128, (M, N)]"


def test_frames_always_carry_a_provenance():
    from sigflow import NO_PROV, Frame

    bare = Frame(np.zeros(2))
    assert bare.prov is NO_PROV and not bare.prov and bare.prov.op == ""
    assert bare.trace() == "(no provenance)" and bare.sources() == []
    assert not Frame(np.zeros(2), prov=None).prov  # type: ignore[arg-type]  # old spelling
    assert "x" not in bare.prov and list(bare.lineage()) == []
