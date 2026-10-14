"""Unified scheduler, provenance modes, inline stream blocks, parallel(), zmq, Lineage (0.3 features)."""

import asyncio
import pickle
import threading

import numpy as np
import pytest

from sigflow import (
    Frame,
    Lineage,
    Sig,
    Stream,
    TypeMismatch,
    block,
    dsp,
    io,
    parallel,
    run,
    source,
)
from sigflow.builder import Chain, lower
from sigflow.frame import PROVENANCE_MODE


@source
def ramp(count: int = 6, n: int = 4, rate: float = 100.0) -> Sig[float, "N"]:
    for i in range(count):
        yield Frame(
            np.full(n, float(i)),
            meta={"t0": i * n / rate, "seq": i},
            stream={"rate": rate},
        )


@block
def scale(x: Sig[float, "N"], gain: float = 2.0) -> Sig[float, "N"]:
    return x * gain


@block
def boom(x: Sig[float, "N"], at: int = 2) -> Sig[float, "N"]:
    if int(x[0]) == at:
        raise ValueError("boom")
    return x


# ------------------------------------------------------------ unified scheduler


def test_operator_trees_lower_to_graph_with_chains():
    g = lower(ramp() >> scale >> dsp.rms)
    members = (
        list(g.members.values())
        if hasattr(g, "members")
        else list(getattr(g, "_members", {}).values())
    )
    assert any(isinstance(m, Chain) for m in members), (
        "linear runs fuse into a Chain member"
    )
    out = run(ramp() >> scale >> dsp.rms)
    assert [float(f.data) for f in out] == pytest.approx([2.0 * i for i in range(6)])


def test_fanout_and_zip_through_scheduler_keep_provenance():
    g = ramp() >> (scale | dsp.rms)
    out = run(g)
    assert len(out) == 6 and all(len(f.parts) == 2 for f in out)
    assert out[0].prov.op == "fanout"
    ops = {n.op for n in out[3].lineage()}
    assert {"scale", "rms", "ramp"} <= ops
    z = run((ramp() >> scale) & (ramp() >> dsp.rms))
    assert len(z) == 6 and z[2].prov.op == "zip"


def test_failures_and_strict_propagate_through_scheduler():
    out = run(ramp() >> boom >> scale, on_error="mark")
    assert sum(f.failed for f in out) == 1 and out[2].failed and out[3].ok
    with pytest.raises(ValueError, match="boom"):
        run(ramp() >> boom >> scale, on_error="raise")
    short = ramp(count=6).take(3)
    with pytest.raises(Exception):
        run((ramp(count=6) >> scale) & short, strict=True)
    assert len(run((ramp(count=6) >> scale) & short)) == 3


def test_fused_maps_keep_sample_annotations():
    g = ramp(n=8) >> dsp.detect_clipping(limit=3.0) >> scale >> scale
    f = run(g)[5]
    assert (
        "clipped" in f.samples
        and f.samples["clipped"].shape == (8,)
        and f.data[0] == 20.0
    )


# -------------------------------------------------------------- provenance modes


def test_provenance_modes():
    g = ramp() >> scale >> scale >> dsp.rms
    full = run(g)[1]
    assert [n.op for n in full.prov.chain()][:2] == ["rms", "scale"]
    roots = run(g, provenance="roots")[1]
    assert roots.prov.op == "rms" and [p.op for p in roots.prov.parents] == ["ramp"]
    assert roots.sources()[0].seq == 1
    off = run(g, provenance="off")[1]
    assert off.prov.op == "rms" and not off.prov.parents and not off.prov.params
    assert PROVENANCE_MODE.get() == "full"  # reset after the run
    with pytest.raises(ValueError):
        run(g, provenance="some")


# -------------------------------------------------------- stream blocks inline


@block
def pairs(xs: Stream[Sig[float, "N"]]) -> Sig[float, "N"]:
    prev = None
    for x in xs:
        if prev is not None:
            yield prev + x
        prev = x


def test_sync_generator_stream_block_runs_inline_or_threaded():
    main = threading.get_ident()
    seen = []

    @block
    def where(xs: Stream[Sig[float, "N"]]) -> Sig[float, "N"]:
        for x in xs:
            seen.append(threading.get_ident())
            yield x

    out = run(ramp() >> pairs)
    assert [float(f.data[0]) for f in out] == [1, 3, 5, 7, 9]
    run(ramp() >> where)
    try:
        import greenlet  # noqa: F401

        assert set(seen) == {main}  # inline on the event loop
    except ImportError:
        assert set(seen) != {main}
    seen.clear()
    run(
        ramp() >> where.clone(threaded=True)
        if hasattr(where, "clone")
        else ramp() >> where
    )


# --------------------------------------------------------------------- parallel


def test_parallel_thread_and_process_keep_order():
    for backend in ("thread", "process"):
        out = run(
            ramp(count=12) >> parallel(scale(gain=3.0), workers=3, backend=backend)
        )
        assert [float(f.data[0]) for f in out] == [3.0 * i for i in range(12)]
        assert out[4].prov.op == "scale" and out[4].sources()[0].seq == 4


def test_parallel_errors_and_restrictions():
    out = run(ramp() >> parallel(boom, workers=2, backend="thread"), on_error="mark")
    assert out[2].failed and out[3].ok and len(out) == 6
    with pytest.raises(TypeError):
        parallel(pairs)  # stream block
    with pytest.raises(ValueError):
        parallel(scale, backend="gpu")
    assert pickle.loads(pickle.dumps(scale)) is scale  # blocks pickle by import path


# -------------------------------------------------------------------------- zmq


def test_zmq_roundtrip():
    pytest.importorskip("zmq")
    addr = "tcp://127.0.0.1:5599"
    sink = io.zmq_sink(addr, kind="PUSH")
    src = io.zmq(addr, dtype=np.float64, kind="PULL", count=4, timeout=5.0)

    async def main():
        rx = asyncio.create_task(asyncio.to_thread(run, src >> dsp.rms))
        await asyncio.sleep(0.2)
        run(ramp(count=4, n=4) >> sink)
        return await rx

    out = asyncio.run(main())
    sink.close()
    assert [round(float(f.data), 6) for f in out] == [0.0, 1.0, 2.0, 3.0]
    assert out[0]["source"].startswith("zmq:")


# ---------------------------------------------------------------------- lineage


def test_lineage_queries():
    g = ramp(count=8) >> (
        (scale(gain=2.0) >> boom(at=6)) | (scale(gain=5.0) >> dsp.rms)
    )
    out = run(g, on_error="mark")
    L = Lineage(out)
    assert len(L) == 8 and len(L.failed()) == 1 and len(L.ok()) == 7
    assert len(L.through("boom")) == 8 and len(L.through("scale", gain=5.0)) == 8
    assert L.through("scale", gain=7.0) == []
    assert L.descendants_of("ramp", 3)[0].failed
    assert L.sources_of(out[:2]) == [("ramp", 0), ("ramp", 1)]
    (origin, src, seq), n = L.blame()[0]
    assert origin == "boom" and src == "ramp" and seq == 3 and n == 1
    assert {d["gain"] for d in L.configs("scale")} == {2.0, 5.0}
    assert "boom" in L.table() and L.ops()[0] in ("ramp", "fanout")
    a = run(ramp() >> scale(gain=2.0) >> dsp.rms)[0]
    b = run(ramp() >> scale(gain=3.0))[0]
    d = Lineage.diff(a, b)
    assert d["scale.gain"] == (2.0, 3.0) and d["rms"] == ("present", "absent")


# ----------------------------------------------------------------------- typing


def test_sig_dsl_still_builds_signal_types():
    t = Sig[np.complex64, ("N", 2), 1e6, "V"]
    assert t.dtype == np.dtype(np.complex64) and t.rate == 1e6 and t.units == "V"
    assert t.shape[1] == 2
    c = block(
        lambda x: x, inputs=Sig[complex, "N"], outputs=Sig[complex, "N"], name="c"
    )
    ramp() >> c  # float -> complex is a safe cast
    with pytest.raises(TypeMismatch, match="dtype"):
        ramp() >> c >> scale  # complex -> float is not


def test_fused_framed_block_sees_upstream_result():
    from sigflow import Framed

    @block
    def mean_of(f: Framed[Sig[float, "N"]]) -> Sig[float, ()]:
        return float(f.data.mean()) + (0.0 if f.prov.op == "scale" else 1000.0)

    out = run(ramp() >> scale(gain=3.0) >> mean_of)
    assert [float(f.data) for f in out] == [3.0 * i for i in range(6)]
