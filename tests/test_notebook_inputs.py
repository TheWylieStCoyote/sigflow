"""Notebook inputs and follow-ups: knob kinds, validation, commit, buttons, the run
registry, the knob timeline, the run report, debug pickers and headless panels."""

from __future__ import annotations

import enum
import json
import threading
import time
from collections.abc import Iterator

import numpy as np
import pytest

from sigflow import (
    Button,
    Frame,
    Knob,
    Sig,
    block,
    commit,
    dsp,
    run,
    runs,
    source,
    stop_all,
)
from sigflow.control import InvalidValue
from sigflow.dims import N

w = pytest.importorskip("ipywidgets")


class Window(enum.Enum):
    HANN = "hann"
    HAMMING = "hamming"


@pytest.fixture(autouse=True)
def _no_runs_left():
    yield
    stop_all()


@source
def ticking(frames: int = 10**6, dt: float = 0.002) -> Iterator[Sig[float, N]]:
    for i in range(frames):
        time.sleep(dt)
        yield Frame(np.full(4, float(i)), meta={"seq": i})


# ------------------------------------------------------------------ knob kinds


@pytest.mark.parametrize(
    "knob,widget_type",
    [
        (Knob(3.0, "gain", min=0.0, max=10.0), w.FloatSlider),
        (Knob(48000.0, "rate"), w.FloatText),  # no bounds: an entry field
        (Knob(512, "nfft"), w.IntText),
        (Knob(4, "taps", min=1, max=64, widget="entry"), w.BoundedIntText),
        (Knob(8, "order", widget="stepper"), w.BoundedIntText),
        (Knob(1e-3, "mu", min=1e-6, max=1.0, widget="log"), w.FloatLogSlider),
        (Knob(True, "notch"), w.ToggleButton),
        (Knob("hann", "window", choices=["hann", "hamming"]), w.Dropdown),
        (Knob(Window.HANN, "window"), w.Dropdown),
        (
            Knob(("a",), "chans", choices=["a", "b", "c"], multiple=True),
            w.SelectMultiple,
        ),
        (Knob((300.0, 3400.0), "band", min=0.0, max=24000.0), w.FloatRangeSlider),
        (Knob((2, 8), "bins", min=0, max=64), w.IntRangeSlider),
        (Knob("tcp://h:1", "addr"), w.Text),
    ],
)
def test_each_knob_kind_gets_its_widget(knob, widget_type):
    assert type(knob.widget()) is widget_type


def test_widget_and_knob_stay_in_step_both_ways():
    band = Knob((300.0, 3400.0), "band", min=0.0, max=24000.0)
    wd = band.widget()
    wd.value = (500.0, 3000.0)
    assert band.value == (500.0, 3000.0)
    band.value = (100.0, 200.0)
    assert wd.value == (100.0, 200.0)
    window = Knob(Window.HANN, "window")
    dd = window.widget()
    dd.value = Window.HAMMING
    assert window.value is Window.HAMMING  # the block gets the enum member


def test_validation_refuses_and_marks_the_widget():
    addr = Knob("tcp://host:5555", "addr", pattern=r"tcp://[\w.-]+:\d+")
    wd = addr.widget()
    wd.value = "not an address"
    assert addr.value == "tcp://host:5555"  # unchanged
    assert (
        addr.error is not None
        and "does not match" in addr.error
        and "2px solid" in wd.layout.border
    )
    wd.value = "tcp://other:6000"
    assert addr.value == "tcp://other:6000" and addr.error is None
    assert wd.layout.border == ""
    with pytest.raises(InvalidValue):
        addr.value = "nope"  # code is checked too


def test_parse_turns_text_into_values():
    def hz(text: str) -> float:
        text = text.strip().lower()
        scale = 1e3 if text.endswith("k") else 1.0
        return float(text.rstrip("k")) * scale

    rate = Knob("48k", "rate", parse=hz)
    assert rate.value == 48000.0
    wd = rate.widget()
    wd.value = "44.1k"
    assert rate.value == 44100.0
    wd.value = "fast"
    assert rate.value == 44100.0 and rate.error is not None and "fast" in rate.error
    with pytest.raises(InvalidValue):
        Knob("a", "x", choices=["b"])


def test_commit_changes_knobs_together_and_checks_first():
    low, high = Knob(300.0, "low"), Knob(3400.0, "high")
    seen: list[tuple[float, float]] = []

    @block
    def band(x: Sig[float, N], low: float = 0.0, high: float = 1.0) -> Sig[float, N]:
        seen.append((low, high))
        return x

    stop = threading.Event()

    def flip() -> None:
        while not stop.is_set():
            commit((low, 1.0), (high, 2.0))
            commit((low, 300.0), (high, 3400.0))

    t = threading.Thread(target=flip)
    t.start()
    run(dsp.sine(n=4, frames=400) >> band(low=low, high=high))
    stop.set()
    t.join()
    assert seen and all(pair in ((1.0, 2.0), (300.0, 3400.0)) for pair in seen)
    choice = Knob("a", "c", choices=["a", "b"])
    with pytest.raises(InvalidValue):
        commit((low, 5.0), (choice, "z"))
    assert low.value in (300.0, 1.0)  # nothing changed by the refused commit


def test_buttons_press_and_report_errors():
    hits: list[int] = []
    ok = Button("reset", lambda: hits.append(1))
    bad = Button("boom", lambda: 1 / 0)
    ok.widget().click()
    ok.press()
    assert hits == [1, 1] and ok.presses == 2
    wd = bad.widget()
    wd.click()
    assert (
        bad.error is not None
        and bad.error.startswith("ZeroDivisionError")
        and wd.button_style == "danger"
    )


# ------------------------------------------------------------------ the registry


def test_registry_replace_and_stop_all():
    g = ticking() >> dsp.scale(gain=1.0)
    first = g.start(name="demo")
    assert first in runs() and first.name == "demo"
    second = g.start(name="demo", replace=True)  # re-running the cell
    assert first.finished and first.state == "stopped"
    assert second in runs() and first not in runs()
    other = g.start()
    assert other.name == g.name
    stopped = stop_all()
    assert set(stopped) == {second, other} and runs() == []
    assert first in runs(all=True)


def test_close_releases_panel_diagram_probes_and_breakpoints():
    gain = Knob(1.0, "gain", min=0.0, max=4.0)
    g = ticking() >> dsp.scale(gain=gain)
    handle = g.start()
    panel = handle.controls(knobs=[gain], diagram=True, display=False)
    probe = handle.probe("scale")
    bp = handle.break_when("scale", lambda f: False)
    live = handle.diagram(display=False)
    handle.close()
    assert handle.finished and panel._closed and live._closed and probe._closed
    assert bp not in handle.breakpoints and not gain._listeners


def test_showing_controls_again_replaces_the_panel():
    handle = (ticking() >> dsp.scale(gain=1.0)).start(paused=True)
    one = handle.controls(display=False)
    two = handle.controls(display=False)
    assert one._closed and not two._closed and handle.panel is two


# ------------------------------------------------------------------ timeline


def test_timeline_records_knob_changes_with_their_frame():
    gain = Knob(1.0, "gain")
    handle = (ticking(dt=0.001) >> dsp.scale(gain=gain)).start()
    assert list(handle.knobs()) == ["gain"]
    for v in (2.0, 3.0):
        while handle.monitor.produced < (10 if v == 2.0 else 30):
            time.sleep(0.005)
        gain.value = v
    handle.stop()
    assert [(c.knob, c.value) for c in handle.timeline] == [
        ("gain", 2.0),
        ("gain", 3.0),
    ]
    assert handle.timeline[0].frame >= 10 and handle.timeline[1].frame >= 30
    records = handle.timeline_records()
    assert json.loads(json.dumps(records))[1]["value"] == 3.0


def test_unnamed_knobs_are_named_after_block_and_parameter():
    handle = (ticking() >> dsp.scale(gain=Knob(1.0))).start(paused=True)
    assert list(handle.knobs()) == ["scale.gain"]


def test_replay_applies_a_recorded_timeline_by_frame():
    gain = Knob(1.0, "gain")
    seen: dict[int, float] = {}

    @block
    def record(x: Sig[float, N], gain: float = 1.0) -> Sig[float, N]:
        seen[int(x[0])] = gain
        return x * gain

    recorded = [{"t": 0.0, "frame": 20, "knob": "gain", "value": 5.0}]
    handle = (ticking(frames=60, dt=0.001) >> record(gain=gain)).start(paused=True)
    worker = handle.replay(recorded)
    handle.resume()
    handle.wait()
    worker.join(1)
    assert seen[5] == 1.0 and seen[59] == 5.0
    assert min(i for i, g in seen.items() if g == 5.0) >= 20


# ------------------------------------------------------------------ report


def test_report_is_static_html_with_everything(tmp_path):
    gain = Knob(1.0, "gain")
    handle = (dsp.sine(n=8, frames=30) >> dsp.scale(gain=gain) >> dsp.rms).start(
        paused=True
    )
    probe = handle.probe("rms")
    handle.break_when("scale", lambda f: False)
    handle.resume()
    gain.value = 2.0
    handle.wait()
    report = handle.report()
    page = report.html
    assert "<svg" in page and "30 produced" in page
    assert "Knobs" in page and "gain" in page and "Knob changes (1)" in page
    assert (
        "Breakpoint at &#x27;scale&#x27;" in page
        and f"probe rms: {probe.count} frames" in page
    )
    assert "Latest frame" in page and "<script" not in page  # nothing live in it
    path = report.save(str(tmp_path / "run.html"))
    with open(path, encoding="utf-8") as fh:
        assert fh.read().startswith("<!doctype html>")


# ------------------------------------------------------------------ panel


def test_headless_panel_with_groups_buttons_and_debug_tools():
    low = Knob(300.0, "low", min=0.0, max=8000.0, group="band")
    high = Knob(3400.0, "high", min=0.0, max=8000.0, group="band")
    reset = Button("reset", lambda: None)

    @block
    def marked(x: Sig[float, N], low: float = 0.0, high: float = 1.0) -> Sig[float, N]:
        return x

    def tagger(f: Frame) -> Frame:
        return f.tag("odd") if f["seq"] % 2 else f

    tag_odd = block(tagger, inputs=Frame, outputs=Sig[float, N], name="tag_odd")
    g = ticking(dt=0.001) >> tag_odd >> marked(low=low, high=high)
    handle = g.start(paused=True)
    panel = handle.controls(knobs=[low, high, reset], debug=True, display=False)
    assert panel._thread is None  # headless: no refresh thread
    sections = [r for r in panel.widget.children if isinstance(r, w.VBox)]
    titles = [
        s.children[0].value for s in sections if isinstance(s.children[0], w.HTML)
    ]
    assert any("band" in t for t in titles)
    tools = panel.debug
    assert "marked" in tools.block.options and "tag_odd" in tools.block.options
    tools.watch("marked")
    bp = tools.add_break("marked", 'f["seq"] >= 5')
    handle.resume()
    held = handle.wait_break(5.0)
    assert held is not None and held["seq"] == 5 and bp in handle.breakpoints
    panel.render()
    assert "odd" in tools.tag.options  # tags seen so far are offered
    assert "<table" in tools.latest.value  # the probed block's latest frame
    tools.clear()
    assert handle.breakpoints == []
    handle.resume()
    assert (
        tools.add_break("marked", "f[") is None and "condition" in tools.message.value
    )


def test_a_change_from_code_notifies_once_even_with_a_widget():
    gain = Knob(1.0, "gain", min=0.0, max=4.0)
    calls: list[float] = []
    gain.on_change(calls.append)
    gain.widget()
    gain.value = 2.0  # updates the widget, whose observer must not report it again
    assert calls == [2.0]
    gain._widget.value = 3.0  # a user dragging the slider
    assert calls == [2.0, 3.0] and gain.value == 3.0
