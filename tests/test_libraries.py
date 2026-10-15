from typing import Iterator, Literal
import os

import numpy as np
import pytest

from sigflow import Sig, TypeMismatch, dsp, linalg, plot, run, source


def test_dsp_chain_types_and_metadata():
    g = (
        dsp.sine(freq=1000, rate=8000, n=256, frames=3)
        >> dsp.window
        >> dsp.to_complex
        >> dsp.fft
        >> dsp.power_db
    )
    out = run(g)
    assert out[0].data.shape == (256,)
    assert out[0].stream["rate"] == 8000 and out[2].meta["seq"] == 2
    assert out[0].prov.op == "power_db"
    peak_bin = int(np.argmax(out[0].data[:128]))
    assert abs(peak_bin * 8000 / 256 - 1000) < 8000 / 256


def test_stream_reframing_blocks():
    out = run(dsp.sine(n=100, frames=10) >> dsp.overlap(n=256, hop=128))
    assert len(out) == 6 and out[0].data.shape == (256,)
    out = run(dsp.sine(n=100, frames=10) >> dsp.buffer(count=5))
    assert out[0].data.shape == (5, 100) and len(out[0].prov.parents) == 5
    out = run(dsp.sine(n=100, frames=10) >> dsp.decimate(every=3))
    assert len(out) == 4


def test_fir_keeps_state_across_frames():
    out = run(dsp.impulse(n=8, frames=2) >> dsp.fir(taps=[0.5, 0.5]))
    assert out[0].data[0] == 0.5 and out[0].data[1] == 0.5 and out[1].data.sum() == 0


def test_collect_sink():
    sink = dsp.collect()
    run(dsp.sine(frames=3, n=16) >> dsp.rms >> sink)
    assert len(sink.data) == 3 and all(abs(v - 1 / np.sqrt(2)) < 0.2 for v in sink.data)


def test_linalg_shape_contracts():
    @source
    def A() -> Iterator[Sig[float, tuple[Literal[3], Literal[4]]]]:
        yield np.arange(12.0).reshape(3, 4)

    @source
    def b() -> Iterator[Sig[float, tuple[Literal[4]]]]:
        yield np.ones(4)

    out = run((A() & b()) >> linalg.matvec)
    assert list(out[0].data) == [6.0, 22.0, 38.0]
    with pytest.raises(TypeMismatch):
        (b() & A()) >> linalg.matvec
    with pytest.raises(TypeMismatch):
        (A() & A()) >> linalg.matmul
    g = ((A() >> linalg.transpose) & A()) >> linalg.matmul >> linalg.eigh
    w, v = run(g)[0].data
    assert w.shape == (4,) and v.shape == (4, 4)
    with pytest.raises(TypeMismatch):
        A() >> linalg.inv  # 3x4 is not square


def test_plot_blocks_record_frames_and_animate(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    p = plot.line(title="t")
    s = plot.spectrum()
    run(dsp.chirp(frames=4, n=128) >> p >> dsp.to_complex >> s)
    assert len(p.frames) == 4 and len(s.frames) == 4
    anim = p.animate(interval=50)
    assert anim is not None
    path = p.save(str(tmp_path / "a.gif"), interval=50)
    assert os.path.getsize(path) > 0
    assert "<img" in p._repr_html_()


def test_graph_and_frame_html():
    g = dsp.sine(frames=1, n=8) >> (dsp.rms | dsp.energy)
    html = g._repr_html_()
    assert "fan-out" in html and "rms" in html
    out = run(g)
    assert "provenance" in out[0]._repr_html_()


def test_draw_and_dot():
    g = (
        (dsp.sine(n=64, frames=2) & dsp.noise(n=64, frames=2))
        >> (lambda ab: ab[0])
        >> (dsp.rms | dsp.energy)
    )
    d = g.draw()
    assert d._repr_svg_().startswith("<svg") and "fan-out" in d.svg and "zip" in d.svg
    assert "digraph" in g.to_dot() and "->" in g.to_dot()


def test_covariance_matches_numpy_convention():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 3)) + 1j * rng.normal(size=(50, 3))
    assert np.allclose(linalg.covariance(x), np.cov(x, rowvar=False))
