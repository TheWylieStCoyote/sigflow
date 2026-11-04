"""What Graph.connect checks while a graph is wired (see examples/26_graph_checks.py)."""

from __future__ import annotations

import warnings
from typing import Annotated

import pytest

from sigflow import (
    Graph,
    RatioMismatch,
    Sig,
    SignalType,
    TypeMismatch,
    accepts,
    block,
    dsp,
    from_iterable,
    requires,
    run,
)
from sigflow.dims import M, N
from sigflow.types import TagWarning

needs_256 = block(
    lambda x: x,
    inputs=SignalType(float, (256,)),
    outputs=SignalType(float, (256,)),
    name="needs_256",
)


@block
def combine(x: Sig[float, N], y: Sig[float, M]) -> Sig[float, N]:
    return x


@block
def demod(
    x: Annotated[Sig[float, N], requires("synced", track=int), accepts("snr_db")],
) -> Sig[float, N]:
    return x


def snapshot(g: Graph) -> tuple:
    return (dict(g.members), list(g.edges), g.out_type, g._count)


def test_a_refused_connect_leaves_the_graph_as_it_was():
    g = Graph()
    src = g.add(dsp.sine(n=1024, frames=1), name="src")
    fit = g.add(needs_256)
    before = snapshot(g)
    with pytest.raises(TypeMismatch, match="length 1024 produced, 256 required"):
        g.connect(src, fit)
    assert snapshot(g) == before
    framed = g.add(dsp.chunk(256))
    g.connect(src, framed).connect(framed, fit).output(fit)
    assert len(run(g)) == 4


def test_nodes_added_by_a_refused_connect_are_removed():
    g = Graph()
    src = g.add(dsp.sine(n=1024), name="src")
    before = snapshot(g)
    with pytest.raises(TypeMismatch):
        g.connect(src, needs_256)  # adds needs_256 implicitly, then refuses
    assert snapshot(g) == before


def test_the_same_problem_is_found_in_any_wiring_order():
    forward = Graph()
    src, gain = (
        forward.add(dsp.sine(n=1024), name="src"),
        forward.add(dsp.scale(gain=2.0)),
    )
    forward.connect(src, gain)
    with pytest.raises(TypeMismatch):
        forward.connect(gain, needs_256)
    backward = Graph()
    src, gain = (
        backward.add(dsp.sine(n=1024), name="src"),
        backward.add(dsp.scale(gain=2.0)),
    )
    backward.connect(gain, needs_256)
    with pytest.raises(TypeMismatch, match="'N' is 256 but here needs 1024"):
        backward.connect(src, gain)


def test_units_rates_ports_and_cycles():
    g = Graph()
    adc = g.add(from_iterable([], type=SignalType(float, (N,), 48_000.0, "mV")))
    volts = block(
        lambda x: x,
        inputs=SignalType(float, (N,), None, "V"),
        outputs=SignalType(float, (N,), None, "V"),
        name="volts",
    )
    with pytest.raises(TypeMismatch, match="insert dsp.to_units"):
        g.connect(adc, volts)

    g = Graph()
    src = g.add(dsp.sine(n=1024), name="src")
    framed = g.add(dsp.chunk(256))
    both = g.add(combine)
    g.connect(src, framed).connect(src, both, "x")
    with pytest.raises(RatioMismatch, match="incompatible rates"):
        g.connect(framed, both, "y")
    assert len(g.edges) == 2

    with pytest.raises(KeyError, match="inputs: \\['x', 'y'\\]"):
        g.connect(src, both, "z")
    with pytest.raises(TypeMismatch, match="already connected"):
        g.connect(framed, both, "x")

    g = Graph()
    one, two = (
        g.add(dsp.scale(gain=1.0), name="one"),
        g.add(dsp.scale(gain=2.0), name="two"),
    )
    g.connect(one, two)
    with pytest.raises(TypeMismatch, match="use feedback"):
        g.connect(two, one)
    assert len(g.edges) == 1


def test_tags_are_checked_at_connect():
    def tagged(t: SignalType):
        return from_iterable([], type=t)

    g = Graph()
    with pytest.raises(TypeMismatch, match="needs tag"):
        g.connect(g.add(from_iterable([])), g.add(demod))
    g = Graph()
    wrong = SignalType(float, (N,)).provides("synced", track=str)
    with pytest.raises(TypeMismatch, match="tag 'track' is str, needs int"):
        g.connect(g.add(tagged(wrong)), g.add(demod))


def test_a_connect_warns_once():
    g = Graph()
    right = SignalType(float, (N,)).provides("synced", track=int)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", TagWarning)
        g.connect(g.add(from_iterable([], type=right)), g.add(demod))
    assert [str(w.message) for w in caught] == [
        "list -> demod[0]: accepts tag 'snr_db', which nothing upstream provides or may provide"
    ]


def composite_graph() -> tuple[Graph, object, object]:
    g = Graph()
    cplx = g.add(dsp.sine(n=256, frames=1) >> dsp.to_complex, name="cplx")
    inner = g.add(dsp.scale(gain=1.0) >> needs_256, name="inner")
    return g, cplx, inner


def test_a_pipeline_member_is_checked_by_validate_before_any_frame_moves():
    g, cplx, inner = composite_graph()
    g.connect(cplx, inner).output(inner)  # accepted: only the member's entry is seen
    with pytest.raises(TypeMismatch, match="dtype variable 'T'"):
        g.validate()
    with pytest.raises(TypeMismatch):
        run(g, on_error="raise")


@pytest.mark.xfail(
    strict=True,
    reason="known gap: connect checks a pipeline member's entry only; "
    "the rest is checked by validate() / run()",
)
def test_a_pipeline_member_is_checked_through_at_connect():
    g, cplx, inner = composite_graph()
    with pytest.raises(TypeMismatch, match="complex128 cannot feed float64"):
        g.connect(cplx, inner)
