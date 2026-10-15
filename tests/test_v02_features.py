"""Units, slicing, time axis, align, Graph builder, plotting, persistence (0.2 features)."""

from typing import Iterator, Literal
import time

import numpy as np
import pytest

from sigflow import (
    Frame,
    Graph,
    Provenance,
    ReplayMismatch,
    Sig,
    Stateful,
    Stream,
    TimeSpan,
    TypeMismatch,
    align,
    block,
    dsp,
    from_iterable,
    linalg,
    plot,
    replay,
    run,
    source,
    units,
)
from sigflow.dims import N, T


@source
def ramp(count: int = 6, n: int = 4, rate: float = 100.0) -> Iterator[Sig[float, N]]:
    for i in range(count):
        yield Frame(
            np.full(n, float(i)),
            meta={"t0": i * n / rate, "seq": i},
            stream={"rate": rate, "units": "mV"},
        )


# ------------------------------------------------------------------ units


def test_units_registry_and_type_contract():
    assert (
        units.parse("mV") == ("voltage", 1e-3)
        and units.compatible("kHz", "Hz")
        and not units.compatible("V", "s")
    )
    assert units.factor("mV", "V") == pytest.approx(1e-3)

    @block
    def volts(x: Sig[float, N, None, Literal["V"]]) -> Sig[float, tuple[()]]:
        return float(x.mean())

    @source
    def mv() -> Iterator[Sig[float, N, None, Literal["mV"]]]:
        yield Frame(np.full(3, 1500.0), stream={"units": "mV"})

    with pytest.raises(TypeMismatch, match="to_units"):
        mv() >> volts
    out = run(mv() >> dsp.to_units("V") >> volts)
    assert out[0].data == pytest.approx(1.5) and out[0]["units"] == "V"


# ---------------------------------------------------------------- slicing


def test_data_slicing_static_shapes_and_samples():
    g = ramp(n=10) >> dsp.detect_clipping(limit=3.0)
    h = g[2:8:2]
    assert h.out_type.shape == (3,)
    f = run(h)[4]
    assert f.data.tolist() == [4.0, 4.0, 4.0] and f.samples["clipped"].shape == (3,)
    assert f["t0"] == pytest.approx(
        4 * 10 / 100 + 2 / 100
    )  # t0 advanced by the slice start
    cube = ramp(n=8) >> dsp.buffer(count=4)
    assert (
        cube[:, 2:5].out_type.shape == (4, 3) and len(cube[1].out_type.shape) == 1
    )  # N stays symbolic here
    with pytest.raises(TypeMismatch):
        cube[7]
    with pytest.raises(TypeError):
        (ramp() >> dsp.rms)[0:2]  # scalar: nothing to slice


def test_stream_slicing():
    s = ramp(count=20) >> dsp.rms
    assert (
        len(run(s.take(5))) == 5
        and len(run(s.skip(15))) == 5
        and len(run(s.every(4))) == 5
    )
    assert (
        len(run(s.until(lambda f: f["seq"] >= 3))) == 3
        and len(run(s.while_(lambda f: f["seq"] < 2))) == 2
    )
    assert run(s.every(3).take(2))[1]["seq"] == 3
    assert s.take(2).out_type == s.out_type  # identity-typed


# --------------------------------------------------------------- time axis


def test_time_axis_propagates_through_reframing():
    out = run(ramp(count=5, n=10, rate=100.0) >> dsp.overlap(n=8, hop=4))
    assert [round(f["t0"], 2) for f in out[:3]] == [0.0, 0.04, 0.08]
    span = out[1].time
    assert (
        isinstance(span, TimeSpan)
        and span.rate == 100.0
        and span.n == 8
        and span.t1 == pytest.approx(0.12)
    )
    part = out[1].between(0.05, 0.09).time
    assert out[1].at(0.05) == 0.0 and part is not None and part.n == 4
    rs = run(ramp(count=2, n=10, rate=100.0) >> dsp.resample(up=1, down=2))[1]
    rt = rs.time
    assert rt is not None and rt.rate == 50.0 and rt.n == 5
    assert rs["t0"] == pytest.approx(0.1)


def test_align_joins_by_time():
    @source
    def fast() -> Iterator[Sig[dict]]:
        for i in range(12):
            yield Frame({"i": i}, meta={"t0": i * 0.1})

    @source
    def slow() -> Iterator[Sig[dict]]:
        for j in range(4):
            yield Frame({"j": j}, meta={"t0": j * 0.3 + 0.01})

    out = run(
        align(fast(), slow(), tolerance=0.05) >> (lambda ab: (ab[0]["i"], ab[1]["j"]))
    )
    assert [f.data for f in out] == [(0, 0), (3, 1), (6, 2), (9, 3)]
    assert all(f["skew"] == pytest.approx(0.01) for f in out)


# --------------------------------------------------------------- builder


def test_graph_builder_diamond_ports_and_cycles():
    eigh = block(
        np.linalg.eigh,
        name="eigh",
        inputs=Sig[T, tuple[N, N]],
        outputs={"vals": Sig[float, N], "vecs": Sig[T, tuple[N, N]]},
    )

    @block
    def count_big(v: Sig[float, N]) -> Sig[int]:
        return int((v > 1).sum())

    @block
    def combine(
        vecs: Sig[T, tuple[N, N]], k: Sig[int], raw: Sig[float, tuple[N, N]]
    ) -> Sig[dict]:
        return {"k": k, "n": vecs.shape[0], "trace": float(np.trace(raw))}

    @source
    def spd() -> Iterator[Sig[float, tuple[Literal[3], Literal[3]]]]:
        yield np.diag([1.0, 2.0, 3.0])

    g = Graph("diamond")
    s = g.add(spd(), "src")
    e = g.add(eigh)
    k = g.add(count_big)
    c = g.add(combine)
    g.connect(s, e).connect(e["vals"], k).connect(e["vecs"], c, "vecs").connect(
        k, c, "k"
    ).connect(s, c, "raw")
    g.output(c)
    out = run(g)
    assert out[0].data == {"k": 2, "n": 3, "trace": 6.0}
    assert "eigh" in out[0].prov and out[0].prov[-1].op == "spd"
    with pytest.raises(TypeMismatch, match="cycle"):
        h = Graph()
        a = h.add(dsp.scale(gain=1))
        b = h.add(dsp.scale(gain=2))
        h.connect(a, b)
        h.connect(b, a)
    with pytest.raises(TypeMismatch):
        h = Graph()
        a = h.add(spd())
        b = h.add(dsp.rms)
        h.connect(a, b)  # matrix into a vector block
    sub = Graph("front")
    i = sub.input(Sig[float, N])
    w = i >> dsp.window
    sub.output(w)
    assert run(ramp(count=2) >> sub)[0].data.shape == (4,)
    assert "<svg" in g.draw().svg and "diamond" in g.report()


# --------------------------------------------------------------- plotting


def test_history_and_multi_plots():
    import matplotlib

    matplotlib.use("Agg")
    h = plot.history(key="seq", title="seq")
    m = plot.multi([plot.line(), h], cols=2)
    run(ramp(count=5) >> m)
    assert len(m.frames) == 5 and m.figure.get_size_inches()[0] == 10.0
    assert len(m.png()) > 1000


# ------------------------------------------------------------ persistence


def test_provenance_json_frame_io_and_replay(tmp_path):
    g = ramp(count=5) >> dsp.detect_clipping(limit=3.0) >> dsp.scale(gain=2)
    f = run(g)[3]
    p = Provenance.from_json(f.prov.to_json())
    assert [n.op for n in p.chain()] == [n.op for n in f.prov.chain()] and p[
        -1
    ].seq == 3
    path = str(tmp_path / "f.npz")
    f.save(path)
    back = Frame.load(path)
    assert (
        np.array_equal(back.data, f.data)
        and back["seq"] == 3
        and "clipped" in back.samples
    )
    assert back.prov["ramp"].params["count"] == 5
    again = replay(g, back)
    assert np.array_equal(again.data, f.data) and again.prov[-1].seq == 3
    with pytest.raises(ReplayMismatch):
        replay(
            ramp(count=9) >> dsp.detect_clipping(limit=3.0) >> dsp.scale(gain=2), back
        )


def test_checkpoint_and_resume(tmp_path):
    class Acc(Stateful):
        def __init__(self) -> None:
            self.total = 0.0

        def __call__(self, x: Sig[float, N]) -> Sig[float, tuple[()]]:
            self.total += float(x.sum())
            return self.total

    full = [f.data for f in run(ramp(count=8) >> block(Acc(), name="acc"))]
    r = (ramp(count=8) >> block(Acc(), name="acc")).start(paused=True)
    r.step(3)
    time.sleep(0.3)
    ck = r.checkpoint(str(tmp_path / "ck.pkl"))
    r.stop()
    assert ck.positions == {"ramp": 3} and ck.states["acc"]["total"] == full[2]
    rest = [
        f.data
        for f in run(
            ramp(count=8) >> block(Acc(), name="acc"), resume=str(tmp_path / "ck.pkl")
        )
    ]
    assert rest == full[3:]
    assert (ramp() >> dsp.running_mean()).start(paused=True).checkpoint().skipped == [
        "running_mean"
    ]


def test_strict_fanout():
    from sigflow import RatioMismatch

    with pytest.raises(RatioMismatch):  # declared rates: caught before running
        ramp(count=6) >> (dsp.rms | dsp.buffer(count=2))

    @block  # no declared ratio: caught at run time
    def halve(xs: Stream[Sig[float, N]]) -> Iterator[Sig[float, N]]:
        for i, x in enumerate(xs):
            if i % 2:
                yield x

    g = ramp(count=6) >> (dsp.rms | halve)
    assert len(run(g)) == 3
    with pytest.raises(TypeMismatch, match="different numbers"):
        run(g, strict=True)
