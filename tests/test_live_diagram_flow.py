"""Live diagram, phases 2 and 3: queues and backpressure on edges, frame paths."""

from __future__ import annotations

import asyncio
import re
import time

import numpy as np

from sigflow import Graph, Sig, block, dsp, feedback, run, when
from sigflow.dims import N
from sigflow.progress import Monitor

FULL = "fill='none' stroke='#d97706' stroke-width='2.2'"  # an edge whose queue is full
ON_PATH = "#0f172a"
FADED = "opacity='0.22'"


@block
async def slow(x: Sig[float, N]) -> Sig[float, N]:
    await asyncio.sleep(0.02)  # yields while it works, so queues can back up
    return x


@block
def boom(x: Sig[float, N]) -> Sig[float, N]:
    if float(np.abs(x).max()) > 3.5:
        raise ValueError("too loud")
    return x


def fork():
    return dsp.sine(n=32, frames=60) >> (dsp.scale(gain=2.0) >> slow | dsp.rms)


# ------------------------------------------------------------------ phase 2


def test_the_runtime_registers_its_queues_with_the_monitor():
    mon = Monitor()
    g = fork()
    run(g, monitor=mon, queue_size=4)
    sine, branch, rms = g.stages[0], g.stages[1].branches[0], g.stages[1].branches[1]
    to_slow = mon.queue_between(sine, branch.stages[0])
    to_rms = mon.queue_between(sine, rms)
    assert to_slow is not None and to_slow.capacity == 4
    assert to_slow.peak == 4  # the slow branch held the source back: a full queue
    assert to_rms is not None and to_rms.peak < 4
    # edges inside a fused chain of map blocks have no queue
    assert mon.queue_between(branch.stages[0], branch.stages[1]) is None


def test_live_diagram_shows_backpressure_and_throughput():
    handle = fork().start(queue_size=4)
    live = handle.diagram()
    svg = live.svg  # the first picture gives the rates a baseline
    for _ in range(150):
        time.sleep(0.02)
        svg = live.svg
        if FULL in svg and handle.monitor.produced >= 5:
            break
    handle.stop()
    assert FULL in svg  # the edge into the slow branch is full
    assert re.search(r"· \d+/s", svg)  # throughput on the pills while running
    assert "queue 4 of 4" in svg or "queue 3 of 4" in svg  # the edge's tooltip


def test_still_shows_peak_queue_depth():
    mon = Monitor()
    g = fork()
    run(g, monitor=mon, queue_size=4)
    svg = g.draw(stats=mon).svg
    assert FULL in svg and "peak 4" in svg
    assert not re.search(r"· \d+/s", svg)  # no rates on a finished run


def test_builder_graph_edges_have_queues():
    graph = Graph()
    src = graph.add(dsp.sine(n=16, frames=5))
    scale = graph.add(dsp.scale(gain=2.0))
    energy = graph.add(dsp.rms)
    graph.connect(src, scale).connect(scale, energy).output(energy)
    mon = Monitor()
    run(graph, monitor=mon)
    assert mon.queue_between(graph.members["sine"], graph.members["scale"]) is not None
    assert "queue" in graph.draw(stats=mon).svg


# ------------------------------------------------------------------ phase 3


def router():
    return (
        dsp.sine(n=32, frames=8)
        >> dsp.scale(gain=2.0)
        >> dsp.scale(gain=1.0)
        >> when(
            lambda f: f["seq"] % 2 == 0,
            then=dsp.scale(gain=2.0) >> boom,
            otherwise=dsp.rms,
        )
    )


def test_frame_path_through_a_router_and_duplicate_names():
    g = router()
    out = run(g)
    odd = out[3]
    d = g.draw(frame=odd)
    path = d.path(odd)
    ops = sorted(r.op for r in path.values())
    assert ops == ["rms", "scale", "scale", "sine", "when(<lambda>)[else]"]
    on = {d.cell.boxes[[b.id for b in d.cell.boxes].index(i)].label for i in path}
    assert "boom" not in on and "rms" in on
    svg = d.svg
    assert ON_PATH in svg and FADED in svg  # the path stands out, the rest fades
    assert "#3" in svg  # the frame's sequence number at the blocks it passed


def test_failed_frame_path_marks_where_it_broke():
    g = router()
    even = run(g)[2]
    assert even.failed and even.errors[-1].origin == "boom"
    d = g.draw(frame=even)
    ops = sorted(r.op for r in d.path(even).values())
    assert "boom" in ops and "rms" not in ops
    assert "stroke='#dc2626' stroke-width='2.6'" in d.svg  # boom is red


def test_path_through_a_fan_out_takes_both_branches():
    g = dsp.sine(n=16, frames=2) >> (dsp.scale(gain=2.0) | dsp.rms)
    f = run(g)[1]
    names = sorted(r.op for r in g.draw(frame=f).path(f).values())
    assert names == ["fanout", "rms", "scale", "sine"]


def test_path_looks_through_feedback_loop_records():
    @block
    def step(
        pair: tuple[Sig[float, N], Sig[float, N]],
    ) -> tuple[Sig[float, N], Sig[float, N]]:
        x, state = pair
        return x + state, x

    g = dsp.sine(n=4, frames=3) >> feedback(step, init=np.zeros(4)) >> dsp.rms
    f = run(g)[2]
    ops = {r.op for r in g.draw(frame=f).path(f).values()}
    assert {"sine", "rms"} <= ops  # the loop's pair records have no box of their own


def test_live_highlight_switches_between_counts_and_a_path():
    g = router()
    handle = g.start(paused=True)
    live = handle.diagram()
    handle.resume()
    handle.wait()
    counts = live.svg
    assert ON_PATH not in counts
    live.highlight(handle.frames[3])
    assert ON_PATH in live.svg and FADED in live.svg
    live.highlight(None)
    assert ON_PATH not in live.svg


def test_no_provenance_draws_nothing_special():
    from sigflow import Frame

    g = router()
    assert g.draw(frame=Frame(np.zeros(4))).path(Frame(np.zeros(4))) == {}
