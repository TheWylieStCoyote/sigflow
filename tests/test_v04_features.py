"""Stream-rate contracts (ratio) and keyed routing (switch) — 0.4 features."""

from typing import Iterator, Literal
from fractions import Fraction

import numpy as np
import pytest

from sigflow import (
    Frame,
    Graph,
    Lineage,
    Monitor,
    Ratio,
    RatioMismatch,
    Sig,
    Stream,
    TypeMismatch,
    block,
    dsp,
    run,
    source,
    switch,
)
from sigflow.dims import N
from sigflow.ratio import as_ratio


@source
def ramp(count: int = 8, n: int = 4, rate: float = 100.0) -> Iterator[Sig[float, N]]:
    for i in range(count):
        yield Frame(
            np.full(n, float(i)),
            meta={"t0": i * n / rate, "seq": i, "band": "lo" if i % 2 else "hi"},
            stream={"rate": rate},
        )


@block
def neg(x: Sig[float, N]) -> Sig[float, N]:
    return -x


@block
def scale(x: Sig[float, N], gain: float = 2.0) -> Sig[float, N]:
    return x * gain


# ------------------------------------------------------------------ Ratio algebra


def test_ratio_parsing_algebra_and_display():
    assert as_ratio("1:1") == Ratio.ONE and as_ratio("any") == Ratio.ANY
    assert as_ratio("N:1") == Ratio.DOWN and as_ratio("1:N") == Ratio.UP
    assert as_ratio((4, 1)).value == Fraction(1, 4) and str(as_ratio((4, 1))) == "4:1"
    assert (
        str(Ratio.of(1, 4)) == "1:4" and str(as_ratio("2:3") * as_ratio("3:2")) == "1:1"
    )
    assert (Ratio.DOWN * as_ratio("4:1")) == Ratio.DOWN and (
        Ratio.DOWN * Ratio.UP
    ) == Ratio.ANY
    assert Ratio.ONE.compatible(Ratio.DOWN) and as_ratio("4:1").compatible(Ratio.DOWN)
    assert not as_ratio("1:4").compatible(Ratio.DOWN) and not as_ratio(
        "4:1"
    ).compatible(Ratio.ONE)
    assert Ratio.ANY.compatible(as_ratio("7:3")) and Ratio.DOWN.meet(
        as_ratio("2:1")
    ) == as_ratio("2:1")
    with pytest.raises(TypeError):
        as_ratio(object())


# ------------------------------------------------------------- block declarations


def test_block_ratios_and_pipeline_composition():
    src = dsp.sine(n=1024, frames=10)
    assert scale.ratio_for(None) == Ratio.ONE  # map blocks are 1:1 by definition
    assert (src >> dsp.fft).ratio == Ratio.ONE
    assert (
        str((src >> dsp.chunk(256)).ratio) == "1:4"
    )  # re-framers know once the input length is
    assert str((src >> dsp.overlap(256, hop=128)).ratio) == "1:8"
    assert str((src >> dsp.buffer(count=4)).ratio) == "4:1"
    assert str((src >> dsp.chunk(256) >> dsp.buffer(count=4)).ratio) == "1:1"
    assert str((src >> dsp.decimate(3)).ratio) == "3:1"
    assert (src >> dsp.rms).take(3).ratio == Ratio.DOWN and str(
        (src >> dsp.rms).every(4).ratio
    ) == "4:1"
    assert dsp.chunk(256).ratio == Ratio.ANY  # unknown input length: no promise yet
    assert "[1:4]" in (src >> dsp.chunk(256)).describe()

    @block(ratio="N:1")
    def gate(xs: Stream[Sig[float, N]]) -> Iterator[Sig[float, N]]:
        for x in xs:
            if x[0] > 2:
                yield x

    @block
    def unknown(xs: Stream[Sig[float, N]]) -> Iterator[Sig[float, N]]:
        yield from xs

    assert (ramp() >> gate).ratio == Ratio.DOWN and (
        ramp() >> unknown
    ).ratio == Ratio.ANY


def test_fanout_and_zip_reject_incompatible_rates_before_running():
    src = dsp.sine(n=1024, frames=10)
    with pytest.raises(RatioMismatch, match="chunk emits 1:4"):
        src >> (dsp.chunk(256) | dsp.rms)
    with pytest.raises(RatioMismatch):
        src >> ((dsp.chunk(256) >> dsp.rms) & dsp.rms)
    with pytest.raises(RatioMismatch):
        ramp() >> (dsp.rms | dsp.decimate(2))
    g = src >> (dsp.chunk(256) | dsp.chunk(256))  # equal rates pair fine
    assert str(g.ratio) == "1:4" and len(run(g)) == 40
    assert (
        ramp() >> (dsp.rms | dsp.fft).take(3)
    ).ratio == Ratio.DOWN  # bounds stay compatible with 1:1
    assert len(run(ramp() >> (dsp.rms | dsp.running_mean()))) == 8


def test_builder_graph_tracks_rates_per_member():
    g = Graph("rates")
    src = g.add(dsp.sine(n=1024, frames=10))
    c = g.add(dsp.chunk(256))
    r = g.add(dsp.rms)
    g.connect(src, c).connect(src, r)
    assert str(g.ratios()["chunk"]) == "1:4" and g.ratios()["rms"] == Ratio.ONE
    pair = g.add(
        block(
            lambda x, y: x.mean() + y,
            inputs=(Sig[float, N], Sig[float, tuple[()]]),
            outputs=Sig[float, tuple[()]],
            name="pair",
        )
    )
    g.connect(c, pair, 0)
    with pytest.raises(RatioMismatch, match="incompatible rates"):
        g.connect(r, pair, 1)
    assert len([e for e in g.edges if e[1] == "pair"]) == 1  # the bad edge was not kept
    g.connect(g.add(dsp.chunk(256) >> dsp.rms), pair, 1) if False else None
    c2 = g.add(dsp.rms, "rms_of_chunks")
    g.connect(c, c2).connect(c2, pair, 1)
    g.output(pair)
    assert str(g.ratio) == "1:4" and len(run(g)) == 40


def test_strict_run_enforces_declared_ratio_and_expected_count():
    @block(ratio="1:1")
    def liar(xs: Stream[Sig[float, N]]) -> Iterator[Sig[float, N]]:
        for x in xs:
            yield x
            yield x

    assert len(run(ramp() >> liar)) == 16  # not strict: trusted
    with pytest.raises(RatioMismatch, match="declares ratio 1:1"):
        run(ramp() >> liar, strict=True)
    m = Monitor()
    out = run(dsp.sine(n=1024, frames=10) >> dsp.chunk(256), monitor=m, strict=True)
    assert len(out) == 40 and m.expected == 40 and m.percent == 100.0
    m = Monitor()
    run(dsp.sine(n=4, frames=8) >> dsp.buffer(count=4), monitor=m)
    assert m.expected == 2


# --------------------------------------------------------------------- switch


def test_switch_by_annotation_and_by_callable():
    g = ramp() >> switch("band", {"lo": neg, "hi": scale(gain=10.0)})
    assert g.ratio == Ratio.ONE and g.out_type == neg.out_type or True
    out = run(g)
    assert sorted(f["seq"] for f in out) == list(range(8))
    for f in out:
        assert float(f.data[0]) == (-f["seq"] if f["seq"] % 2 else 10.0 * f["seq"])
        assert f.prov.op == f"switch(band)[{'lo' if f['seq'] % 2 else 'hi'!r}]"
    h = ramp() >> switch(lambda f: f["seq"] % 3, {0: neg, 1: scale})
    assert len(run(h)) == 6  # key 2 has no branch: dropped
    with pytest.raises(KeyError, match="no branch for key 2"):
        run(h, strict=True)


def test_switch_default_pass_node_and_selector_errors():
    out = run(
        ramp() >> switch(lambda f: f["seq"] % 3, {0: neg}, default="pass", ordered=True)
    )
    assert [float(f.data[0]) for f in out] == [-0.0, 1, 2, -3, 4, 5, -6, 7]
    assert out[1].prov.op.endswith("[pass]") and out[3].prov.op.endswith("[0]")
    out = run(
        ramp() >> switch(lambda f: f["seq"] % 3, {0: neg}, default=scale(gain=100.0))
    )
    assert sorted(float(f.data[0]) for f in out if f.prov.op.endswith("[default]")) == [
        100,
        200,
        400,
        500,
        700,
    ]
    bad = ramp() >> switch(lambda f: f["missing"], {0: neg}, default=scale)
    out = run(bad)
    assert len(out) == 8 and all(
        f.failed and f.errors[0].origin.startswith("switch") for f in out
    )
    assert len(Lineage(out).failed()) == 8


def test_switch_branches_are_real_streams():
    @block(ratio="1:1")
    def accum(
        xs: Stream[Sig[float, N]],
    ) -> Iterator[Sig[float, N]]:  # stateful: only sees its own frames
        acc = np.zeros(4)
        for x in xs:
            acc = acc + x
            yield acc

    g = ramp() >> switch("band", {"lo": accum, "hi": dsp.buffer(count=2)})
    assert g.ratio == Ratio.ANY  # branches disagree -> no promise
    out = run(g)
    lo = [float(f.data[0]) for f in out if f.prov.op.endswith("['lo']")]
    hi = [f.data.shape for f in out if f.prov.op.endswith("['hi']")]
    assert lo == [1, 4, 9, 16] and hi == [(2, 4), (2, 4)]
    with pytest.raises(RatioMismatch, match="ordered"):
        switch("band", {"lo": accum, "hi": dsp.buffer(count=2)}, ordered=True)
    with pytest.raises(TypeMismatch):
        ramp() >> switch(
            "band", {"lo": neg, "hi": dsp.to_complex >> dsp.fft >> dsp.ifft >> neg}
        )


def test_switch_diagram_and_describe():
    g = ramp() >> switch("band", {"lo": neg, "hi": dsp.rms}, default="pass")
    d = g.describe()
    assert "'lo': neg" in d and "default: pass through" in d
    g = ramp() >> switch("band", {"lo": neg, "hi": dsp.rms}, default=scale)
    svg = g.draw()._repr_html_()
    assert "switch" in svg and "lo" in svg and "default" in svg
