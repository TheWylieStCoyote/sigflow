"""The live diagram: a run's counts, errors and states laid over the graph picture."""

from __future__ import annotations

import re
import time

import pytest

from sigflow import Sig, block, dsp, run
from sigflow.dims import N
from sigflow.draw import Badge, Diagram, LiveDiagram
from sigflow.progress import Monitor


@block
def boom(x: Sig[float, N], at: int = 5) -> Sig[float, N]:
    boom.calls += 1  # type: ignore[attr-defined]
    if boom.calls == at:  # type: ignore[attr-defined]
        raise ValueError("kaboom")
    return x


def graph():
    boom.calls = 0  # type: ignore[attr-defined]
    return dsp.sine(n=64, frames=12) >> (dsp.scale(gain=2.0) >> boom | dsp.rms)


def pills(svg: str) -> list[str]:
    return re.findall(r"font-size='9' fill='[^']+'>([^<]*)</text>", svg)


def test_still_with_stats_shows_counts_errors_and_state():
    g = graph()
    mon = Monitor()
    run(g, monitor=mon)
    plain, still = g.draw(), g.draw(stats=mon)
    assert "→" not in "".join(pills(plain.svg)) and "produced" not in plain.svg
    labels = pills(still.svg)
    assert "12" in labels  # a source shows what it put out
    assert "12→12" in labels and "✕ 1" in labels  # boom saw every frame, raised once
    assert still.svg.count("stroke='#dc2626'") >= 1  # the failing block is red
    assert still.svg.count("stroke='#16a34a' stroke-width='2.2'") >= 2  # others done
    assert "done · 12 produced" in still.svg and "1 error" in still.svg
    assert "in 12, out 12, errors 1" in still.svg  # the tooltip carries the numbers


def test_every_box_including_joins_finds_its_stats():
    g = graph()
    mon = Monitor()
    run(g, monitor=mon)
    d = Diagram(g, stats=mon)
    counts = d.counts(mon)
    assert len(counts) == len(d.cell.boxes)  # sine, scale, boom, rms and the fan-out
    assert all(b.node is not None for b in d.cell.boxes)


def test_badge_states_follow_the_run():
    counts = {1: Badge(4, 4), 2: Badge(0, 0), 3: Badge(4, 4, errors=1)}
    moved = Diagram.badges(dict(counts), {1: (3, 3), 2: (0, 0), 3: (4, 4)}, "running")
    assert [moved[i].state for i in (1, 2, 3)] == ["active", "idle", "error"]
    paused = Diagram.badges({1: Badge(4, 4)}, {1: (4, 4)}, "paused")
    assert paused[1].state == "paused"
    settled = Diagram.badges({1: Badge(4, 4)}, {1: (4, 4)}, "running")
    assert settled[1].state is None  # ordinary border: nothing moved, nothing wrong
    finished = Diagram.badges({1: Badge(4, 4)}, {1: (2, 2)}, "done")
    assert finished[1].state == "done"  # a finished run is green even if it just moved


def test_live_diagram_tracks_a_run_outside_a_notebook():
    g = graph()
    handle = g.start(paused=True)
    live = handle.diagram()  # no notebook: the object comes back, nothing is displayed
    assert isinstance(live, LiveDiagram) and handle.live_diagram is live
    first = live.svg
    assert "paused · 0 produced" in first and "#d97706' stroke-width='2.2'" in first
    handle.step(3)
    for _ in range(100):  # the three frames drain through the graph
        if handle.monitor.produced >= 3:
            break
        time.sleep(0.02)
    stepped = live.svg
    assert "3 produced" in stepped and "#2563eb' stroke-width='2.2'" in stepped  # moved
    handle.resume()
    handle.wait()
    final = live.svg
    assert "done · 12 produced" in final and "✕ 1" in pills(final)
    assert "#2563eb' stroke-width='2.2'" not in final  # finished: green, not active
    assert "<div" in live._repr_html_() and repr(live).endswith("done>")
    live.close()


def test_panel_row_when_widgets_are_available():
    pytest.importorskip("ipywidgets")
    from sigflow.ui import ControlPanel

    g = graph()
    handle = g.start(paused=True)
    panel = ControlPanel(handle, diagram=True)
    assert panel.live is not None and "<svg" in panel.diagram_html.value
    handle.resume()
    handle.wait()
    panel.render()
    assert "done · 12 produced" in panel.diagram_html.value
