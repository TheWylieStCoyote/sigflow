"""Breakpoints: hold the graph on a frame one block puts out, before the next sees it."""

from __future__ import annotations

import numpy as np
import pytest

from sigflow import Breakpoint, Frame, Framed, Sig, block, dsp
from sigflow.dims import N


@block
def boom(x: Sig[float, N], at: int = 3) -> Sig[float, N]:
    boom.calls += 1  # type: ignore[attr-defined]
    if boom.calls == at:  # type: ignore[attr-defined]
        raise ValueError("kaboom")
    return x


@block
def tagger(f: Framed[Sig[float, N]]) -> Sig[float, N]:
    return f.tag("odd") if f["seq"] % 2 else f


def chain():
    return dsp.sine(n=8, frames=10) >> dsp.scale(gain=2.0) >> dsp.rms


def test_a_held_frame_stops_at_its_block():
    handle = chain().start(paused=True)
    bp = handle.break_when("scale", lambda f: f["seq"] == 3)
    assert isinstance(bp, Breakpoint) and handle.breakpoints == [bp]
    handle.resume()
    held = handle.wait_break(5.0)
    assert held is not None and held["seq"] == 3 and held.prov.op == "scale"
    assert handle.state == "break" and handle.held_at == "scale"
    assert "break at scale #3" in handle.status()
    assert handle.monitor.stats("rms").out == 3  # frame 3 has not reached rms
    assert len(handle.frames) == 3
    handle.resume()
    handle.wait()
    assert handle.state == "done" and len(handle.frames) == 10
    assert bp.count == 1 and bp.hits[0] is held and repr(bp).endswith("1 hits>")


def test_step_holds_the_next_frame_of_the_same_block():
    handle = chain().start(paused=True)
    handle.break_at("scale", seq=2)
    handle.resume()
    assert handle.wait_break(5.0)["seq"] == 2
    handle.step()
    assert handle.wait_break(5.0)["seq"] == 3 and handle.control.held_by == "step"
    handle.step()
    assert handle.wait_break(5.0)["seq"] == 4
    handle.resume()
    handle.wait()
    assert len(handle.frames) == 10 and handle.breakpoints[0].count == 1  # once


def test_replace_and_drop_a_held_frame():
    handle = chain().start(paused=True)
    handle.break_when("scale", lambda f: f["seq"] in (4, 6))
    handle.resume()
    held = handle.wait_break(5.0)
    assert held["seq"] == 4
    handle.replace_held(held.evolve(data=np.zeros_like(held.data)))
    held = handle.wait_break(5.0)
    assert held["seq"] == 6
    handle.drop_held()
    handle.wait()
    outputs = {f["seq"]: float(f.data) for f in handle.frames}
    assert outputs[4] == 0.0 and 6 not in outputs and len(outputs) == 9


def test_break_on_error_tag_every_and_disable():
    boom.calls = 0  # type: ignore[attr-defined]
    handle = (dsp.sine(n=8, frames=8) >> boom >> tagger >> dsp.scale(gain=1.0)).start(
        paused=True
    )
    handle.break_on_error()
    odd = handle.break_when("tagger", tag="odd", every=2)  # every second odd frame
    handle.resume()
    held = handle.wait_break(5.0)  # seq 1 is odd but only the first match; seq 2 fails
    assert (
        held.failed and handle.held_at == "boom" and handle.control.held_by == "error"
    )
    handle.resume()
    held = handle.wait_break(5.0)
    assert held["seq"] == 3 and "odd" in held and handle.held_at == "tagger"
    odd.disable()
    handle.resume()
    handle.wait()
    assert handle.state == "done" and odd.count == 1 and not odd.enabled
    assert handle.frames[2].failed  # the failed frame carried on after resume


def test_by_node_unknown_name_and_removal():
    g = chain()
    handle = g.start(paused=True)
    with pytest.raises(KeyError, match="no block called 'fft'"):
        handle.break_when("fft")
    bp = handle.break_when(g.stages[1], lambda f: True)
    assert bp.node is g.stages[1] and bp.name == "scale"
    bp.remove()
    assert handle.breakpoints == [] and handle.monitor.stats("scale").holds == []
    handle.resume()
    handle.wait()
    assert handle.state == "done" and len(handle.frames) == 10


def test_a_bad_predicate_disables_the_breakpoint_not_the_run():
    handle = chain().start(paused=True)

    def broken(frame: Frame) -> bool:
        raise RuntimeError("no")

    bp = handle.break_when("scale", broken)
    handle.resume()
    handle.wait()
    assert handle.state == "done" and len(bp.errors) == 1 and not bp.enabled


def test_stop_while_held_and_stream_and_join_breaks():
    handle = chain().start(paused=True)
    handle.break_at("scale", seq=1)
    handle.resume()
    assert handle.wait_break(5.0) is not None
    handle.stop()
    assert handle.state == "stopped" and handle.held is None

    forked = (dsp.sine(n=16, frames=4) >> (dsp.scale(gain=1.0) | dsp.rms))[
        0
    ] >> dsp.chunk(2)
    handle = forked.start(paused=True)
    handle.break_at("fanout", seq=1)
    handle.break_when("chunk", lambda f: f.prov.seq == 0)
    kinds = {"chunk": np.ndarray, "fanout": tuple}
    seen = set()
    for _ in kinds:  # the branches race, so either break may come first
        handle.resume()
        held = handle.wait_break(5.0)
        assert held is not None and isinstance(held.data, kinds[handle.held_at])
        seen.add(handle.held_at)
    assert seen == set(kinds)
    handle.resume()
    handle.wait()
    assert handle.state == "done"


def test_diagram_and_panel_show_the_held_frame():
    handle = chain().start(paused=True)
    live = handle.diagram()
    handle.break_at("scale", seq=2)
    handle.resume()
    handle.wait_break(5.0)
    svg = live.svg
    assert "break at scale" in svg and "#7c3aed' stroke-width='2.2'" in svg
    from sigflow.ui import ControlPanel, held_html  # held_html needs no widgets

    assert "break at scale #2" in held_html(handle) and "<table" in held_html(handle)
    pytest.importorskip("ipywidgets")
    panel = ControlPanel(handle)
    panel.render()
    assert "break at scale #2" in panel.break_html.value
    assert panel.b_resume.disabled is False and panel.b_pause.disabled is True
    handle.resume()
    handle.wait()
    panel.render()
    assert panel.break_html.value == ""
