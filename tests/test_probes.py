"""Probes: watch one block's output while a graph runs, without editing the graph."""

from __future__ import annotations

import numpy as np
import pytest

from sigflow import Frame, Probe, Sig, block, dsp, plot, run
from sigflow.dims import N
from sigflow.progress import Monitor


@block
def boom(x: Sig[float, N], at: int = 3) -> Sig[float, N]:
    boom.calls += 1  # type: ignore[attr-defined]
    if boom.calls == at:  # type: ignore[attr-defined]
        raise ValueError("kaboom")
    return x


def fresh_boom() -> None:
    boom.calls = 0  # type: ignore[attr-defined]


def test_probe_sees_every_frame_of_a_fused_inner_block():
    g = dsp.sine(n=8, frames=10) >> dsp.scale(gain=2.0) >> dsp.rms
    handle = g.start(paused=True)
    scaled = handle.probe("scale", keep=4)
    assert isinstance(scaled, Probe) and handle.probes == [scaled]
    assert scaled.latest is None and scaled.count == 0
    handle.resume()
    handle.wait()
    assert scaled.count == 10 and len(scaled.history) == 4
    f = scaled.latest
    assert isinstance(f, Frame) and f.prov.op == "scale" and f["seq"] == 9
    source = run(dsp.sine(n=8, frames=10))[-1].data
    np.testing.assert_allclose(f.data, 2.0 * source)  # what the next block receives
    assert repr(scaled).startswith("<Probe 'scale': 10 frames")


def test_probe_by_node_and_by_disambiguated_name():
    g = dsp.sine(n=8, frames=4) >> dsp.scale(gain=2.0) >> dsp.scale(gain=3.0)
    handle = g.start(paused=True)
    first = handle.probe("scale")
    second = handle.probe("scale#2")
    by_node = handle.probe(g.stages[-1])  # exact, whatever name the table shows
    assert by_node.name == "scale" and by_node.node is g.stages[-1]
    handle.resume()
    handle.wait()
    assert first.count == second.count == 4
    np.testing.assert_allclose(second.latest.data, 3.0 * first.latest.data)
    np.testing.assert_allclose(by_node.latest.data, second.latest.data)
    with pytest.raises(KeyError, match="no block called 'fft'"):
        handle.probe("fft")


def test_probe_sees_failed_frames_and_survives_a_bad_callback():
    fresh_boom()
    g = dsp.sine(n=8, frames=5) >> boom >> dsp.rms
    handle = g.start(paused=True)
    seen: list[bool] = []

    def bad(frame: Frame) -> None:
        raise RuntimeError("probe callback broke")

    p = handle.probe("boom").on(lambda f: seen.append(f.failed)).on(bad)
    handle.resume()
    handle.wait()
    assert p.count == 5 and p.failed == 1 and seen == [False, False, True, False, False]
    assert len(p.errors) == 1 and handle.state == "done"  # the run was unaffected


def test_probe_feeds_a_plot_block_and_watches_a_join():
    import matplotlib

    matplotlib.use("Agg")
    g = dsp.sine(n=16, frames=6) >> (dsp.scale(gain=1.0) | dsp.rms)
    handle = g.start(paused=True)
    line = plot.line(title="probed")
    scaled = handle.probe("scale", plot=line)
    joined = handle.probe("fanout")
    handle.resume()
    handle.wait()
    assert len(line.frames) == 6 and scaled.plots == [line]
    assert joined.count == 6 and isinstance(joined.latest.data, tuple)
    assert joined.latest[1].data == pytest.approx(
        float(np.sqrt(np.mean(joined.latest[0].data ** 2)))
    )


def test_probe_wait_and_close():
    g = dsp.sine(n=8, frames=20) >> dsp.scale(gain=1.0)
    handle = g.start(paused=True)
    p = handle.probe("scale")
    handle.set_rate(200)
    handle.resume()
    f = p.wait(3)
    assert f is not None and p.count >= 3
    p.close()
    seen = p.count
    handle.wait()
    assert p.count == seen and handle.monitor.produced == 20  # detached, run finished
    assert p.wait(5, timeout=0.1) is p.latest  # a finished run never blocks


def test_monitor_watch_works_for_a_blocking_run():
    mon = Monitor()
    got: list[Frame] = []
    mon.watch("rms", got.append)  # attached before the node exists
    run(dsp.sine(n=8, frames=3) >> dsp.rms, monitor=mon)
    assert len(got) == 3 and all(f.prov.op == "rms" for f in got)
    mon.unwatch("rms", got.append)
    assert mon.stats("rms").watchers == []


def test_probe_html_without_a_notebook():
    g = dsp.sine(n=8, frames=2) >> dsp.scale(gain=1.0)
    handle = g.start(paused=True)
    p = handle.probe("scale")
    assert "no frames yet" in p._repr_html_()
    handle.resume()
    handle.wait()
    assert "probe scale" in p.html() and "2 frames" in p.html() and "<table" in p.html()
