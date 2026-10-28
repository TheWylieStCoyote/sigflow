from typing import Literal
import numpy as np
import pytest

from sigflow import Any, Env, Sig, TupleType, TypeMismatch, unify
from sigflow import SignalType
from sigflow.dims import M, N, T
from sigflow import SignalType
from sigflow.types import DVar, narrow, scoped


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


@pytest.mark.parametrize(
    "produced,required,expected",
    [
        ((None,), (4,), (4,)),
        ((N,), (4,), (4,)),
        ((None, 4), (N, N), (4, 4)),
        ((N, N), (4, None), (4, 4)),
        ((None, 3), (4, None), (4, 3)),
    ],
)
def test_narrow_preserves_axis_constraints(produced, required, expected):
    carried = narrow(SignalType(float, produced), SignalType(float, required))
    assert carried.shape == expected
    carried.check(np.zeros(expected))
    wrong = (*expected[:-1], expected[-1] + 1)
    with pytest.raises(TypeMismatch):
        carried.check(np.zeros(wrong))


def test_narrow_retains_unbound_aliases_and_tuple_bindings():
    carried = narrow(SignalType(float, (None, M)), SignalType(float, (N, N)))
    assert carried.shape[0] == carried.shape[1]
    with pytest.raises(TypeMismatch):
        carried.check(np.zeros((2, 3)))

    produced = TupleType((SignalType(Any, (None,)), SignalType(float, (4,))))
    required = TupleType((SignalType(T, (N,)), SignalType(T, (N,))))
    carried = narrow(produced, required)
    assert carried.items == (SignalType(float, (4,)), SignalType(float, (4,)))


def test_gradual_dtype_does_not_hide_later_bindings():
    env = unify(SignalType(Any), SignalType(T))
    assert env.value("T") is None
    unify(SignalType(float), SignalType(T), env)
    assert env.value("T") == np.dtype(float)


def test_narrow_keeps_metadata_and_checks_known_rate_units():
    carried = narrow(
        SignalType(float, (None,), meta=frozenset({"source", "calibrated"})),
        SignalType(float, (4,), 48000, "V", frozenset({"calibrated"})),
    )
    assert carried == SignalType(
        float, (4,), 48000, "V", frozenset({"source", "calibrated"})
    )
    with pytest.raises(TypeMismatch, match="needs tag"):  # a required tag is a tag
        narrow(SignalType(float, (None,)), SignalType(float).requires("calibrated"))
    with pytest.raises(TypeMismatch, match="rate"):
        narrow(SignalType(float, rate=48000), SignalType(float, (4,), 96000))
    with pytest.raises(TypeMismatch, match="units"):
        narrow(SignalType(float, units="V"), SignalType(float, (4,), units="A"))


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
