"""Notebook control panels built on ipywidgets.

    run.controls(plots=[...], knobs=[...])   # full panel for one run
    dashboard(runs)                          # several runs, one panel each, stop-all
    Knob(...).widget()                       # a slider / toggle / dropdown / text box bound to a knob

Requires ``ipywidgets`` in the kernel (``pip install ipywidgets``, then restart the
kernel). Without it, ``run.controls()`` shows a status line and says so.
"""

from __future__ import annotations

import html
import threading
import time
from collections.abc import Callable
from typing import Any

__all__ = [
    "ControlPanel",
    "DebugTools",
    "button_widget",
    "dashboard",
    "have_widgets",
    "knob_widget",
    "require_widgets",
]

_COLOURS = {
    "running": "#16a34a",
    "paused": "#d97706",
    "break": "#7c3aed",
    "stopped": "#6b7280",
    "done": "#2563eb",
    "failed": "#dc2626",
}

_HELP = (
    "Interactive controls need <b>ipywidgets</b> in this kernel: "
    "<code>python -m pip install ipywidgets</code> (JupyterLab 3+/Notebook 7 pick it up "
    "automatically), then <b>restart the kernel</b>. Until then use "
    "<code>run.pause()</code>, <code>run.resume()</code>, <code>run.step()</code>, "
    "<code>run.stop()</code> and set <code>knob.value</code> from a cell."
)


def have_widgets() -> bool:
    """Whether ``ipywidgets`` can be imported (the control panel and knob widgets
    need it)."""
    try:
        import ipywidgets  # noqa: F401

        return True
    except ImportError:
        return False


def require_widgets() -> None:
    """Print a clear message (or nothing) about whether interactive panels will work."""
    try:
        from IPython.display import HTML, display
    except ImportError:
        print("Not running under IPython; interactive panels are unavailable.")
        return
    if have_widgets():
        import ipywidgets

        display(
            HTML(
                f"<span style='color:#16a34a'>ipywidgets {ipywidgets.__version__} found — "
                f"buttons, sliders and live figures will work.</span>"
            )
        )
    else:
        display(HTML(f"<div style='color:#dc2626'>{_HELP}</div>"))


# --------------------------------------------------------------------------- #
# Knob widgets
# --------------------------------------------------------------------------- #


def knob_widget(knob: Any) -> Any:
    """The ipywidgets control for a knob (see :class:`~sigflow.control.Knob` for which
    one): sliders, entry fields, toggles, dropdowns, multi-selects, range sliders or
    text, bound both ways. A value the knob refuses leaves it unchanged and marks the
    widget with a red border and the reason as its tooltip."""
    import enum

    import ipywidgets as w

    v: Any = knob.value  # tested with isinstance below
    raw: Any = knob.value  # the same, un-narrowed, for building the widget
    desc = knob.label
    kind = knob.widget_kind
    show: Any = None  # knob value -> widget value, when they differ
    take: Callable[[Any], Any] | None = None  # widget value -> knob value
    # a knob with a parser takes typed text, whatever the value it holds
    numeric = (
        isinstance(v, (int, float)) and not isinstance(v, bool) and knob.parse is None
    )
    bounded = knob.min is not None and knob.max is not None
    if knob.choices is not None:
        labels = [
            (c.name if isinstance(c, enum.Enum) else str(c), c) for c in knob.choices
        ]
        if knob.multiple:
            wd = w.SelectMultiple(options=labels, value=tuple(raw), description=desc)
        else:
            wd = w.Dropdown(options=labels, value=raw, description=desc)
    elif isinstance(v, bool) and knob.parse is None:
        wd = w.ToggleButton(value=raw, description=desc, icon="check" if raw else "")
    elif (
        isinstance(v, tuple)
        and len(raw) == 2
        and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in raw)
    ):
        lo = knob.min if knob.min is not None else min(raw) - abs(raw[1] - raw[0])
        hi = knob.max if knob.max is not None else max(raw) + abs(raw[1] - raw[0])
        if all(isinstance(x, int) for x in raw):
            wd = w.IntRangeSlider(
                value=raw,
                min=int(lo),
                max=int(hi),
                step=int(knob.step or 1),
                description=desc,
            )
        else:
            wd = w.FloatRangeSlider(
                value=raw,
                min=lo,
                max=hi,
                step=knob.step or (hi - lo) / 200,
                description=desc,
                readout_format=".3g",
            )
        take = tuple
    elif numeric and kind == "log":
        lo = knob.min if knob.min is not None else raw / 100
        hi = knob.max if knob.max is not None else raw * 100
        import math

        wd = w.FloatLogSlider(
            value=raw,
            base=10,
            min=math.log10(lo),
            max=math.log10(hi),
            step=0.01,
            description=desc,
            readout_format=".3g",
        )
    elif numeric and (kind == "slider" or (kind is None and bounded)):
        if isinstance(v, int):
            lo = knob.min if knob.min is not None else (0 if raw >= 0 else 2 * raw)
            hi = knob.max if knob.max is not None else max(2 * raw, lo + 10)
            wd = w.IntSlider(
                value=raw,
                min=lo,
                max=hi,
                step=int(knob.step or 1),
                description=desc,
                continuous_update=True,
            )
        else:
            lo = knob.min if knob.min is not None else (0.0 if raw >= 0 else 2 * raw)
            hi = knob.max if knob.max is not None else (2 * raw if raw > 0 else 1.0)
            wd = w.FloatSlider(
                value=raw,
                min=lo,
                max=hi,
                step=knob.step or (hi - lo) / 200,
                description=desc,
                continuous_update=True,
                readout_format=".3g",
            )
    elif numeric and kind == "stepper" and isinstance(v, int):
        wd = w.BoundedIntText(
            value=raw,
            min=knob.min if knob.min is not None else -(2**31),
            max=knob.max if knob.max is not None else 2**31 - 1,
            step=int(knob.step or 1),
            description=desc,
        )
    elif numeric and kind in (None, "entry", "stepper"):
        cls = w.IntText if isinstance(v, int) else w.FloatText
        if bounded:
            cls = w.BoundedIntText if isinstance(v, int) else w.BoundedFloatText
            wd = cls(value=raw, min=knob.min, max=knob.max, description=desc)
        else:
            wd = cls(value=raw, description=desc)
        if knob.step:
            wd.step = knob.step
    else:
        wd = w.Text(value=str(raw), description=desc, continuous_update=False)
        if not isinstance(v, str):
            kind_of = type(raw)

            def from_text(text: Any) -> Any:
                return text if knob.parse is not None else kind_of(text)

            take = from_text
        show = str

    def mark(ok: bool) -> None:
        wd.layout.border = "" if ok else "2px solid #dc2626"
        wd.tooltip = "" if ok else (knob.error or "refused")

    def changed(change: Any) -> None:
        new = change["new"]
        if take is not None:
            try:
                new = take(new)
            except (TypeError, ValueError) as e:
                knob.error = f"{knob.label}: {e}"
                mark(False)
                return
        mark(knob._set_from_widget(new))
        if isinstance(wd, w.ToggleButton):
            wd.icon = "check" if new else ""

    wd.observe(changed, names="value")
    knob._show = show  # how the widget shows a value set from code
    knob._widget = wd
    return wd


def button_widget(button: Any) -> Any:
    """The ipywidgets button for a :class:`~sigflow.control.Button`; a press that fails
    turns it red with the error as its tooltip."""
    import ipywidgets as w

    wd = w.Button(description=button.label, icon=button.icon)

    def pressed(_: Any) -> None:
        button.press()
        wd.button_style = "danger" if button.error else ""
        wd.tooltip = button.error or ""

    wd.on_click(pressed)
    return wd


# --------------------------------------------------------------------------- #
# Control panel
# --------------------------------------------------------------------------- #


class ControlPanel:
    """Buttons, throttle, progress, knobs, live figures and status for one :class:`Run`."""

    def __init__(
        self,
        run: Any,
        plots: Any = (),
        knobs: Any = (),
        refresh: float = 0.4,
        table: bool = True,
        width: int = 520,
        title: str | None = None,
        diagram: bool = False,
        debug: bool = False,
        auto: bool = True,
    ) -> None:
        import ipywidgets as w

        self.run = run
        self._closed = False
        self.live: Any = None
        if diagram:
            from .draw import LiveDiagram

            self.live = LiveDiagram(run)
            self.diagram_html = w.HTML(self.live.html())
        self.plots = [plots] if hasattr(plots, "draw_fn") else list(plots)
        self.knobs = (
            [knobs]
            if hasattr(knobs, "_set_from_widget") or hasattr(knobs, "press")
            else list(knobs)
        )
        self.refresh = refresh
        self.table = table
        self.snapshots: list[Any] = []

        # -- transport --------------------------------------------------- #
        self.b_pause = w.Button(
            description="Pause", icon="pause", tooltip="hold the sources"
        )
        self.b_resume = w.Button(
            description="Resume", icon="play", button_style="success"
        )
        self.b_step = w.Button(
            description="Step", icon="step-forward", tooltip="release one source frame"
        )
        self.steps = w.BoundedIntText(
            value=1,
            min=1,
            max=100000,
            layout=w.Layout(width="70px"),
            tooltip="frames per Step",
        )
        self.b_stop = w.Button(description="Stop", icon="stop", button_style="danger")
        self.b_snap = w.Button(
            description="Snapshot", icon="camera", tooltip="keep the latest frame"
        )
        self.b_record = w.ToggleButton(
            value=False,
            description="Record",
            icon="circle",
            tooltip="collect frames while on",
        )
        self.b_pause.on_click(lambda _: run.pause())
        self.b_resume.on_click(lambda _: run.resume())
        self.b_step.on_click(lambda _: run.step(self.steps.value))
        self.b_stop.on_click(lambda _: run.stop())
        self.b_snap.on_click(lambda _: self._snapshot())
        self.b_record.observe(lambda c: run.set_recording(c["new"]), names="value")
        transport = w.HBox(
            [
                self.b_pause,
                self.b_resume,
                self.b_step,
                self.steps,
                self.b_stop,
                self.b_snap,
                self.b_record,
            ]
        )

        # -- throttle + progress ----------------------------------------- #
        self.rate = w.FloatLogSlider(
            value=run.rate or 1e4,
            base=10,
            min=-1,
            max=4,
            step=0.05,
            description="max fps",
            readout_format=".3g",
            continuous_update=True,
            layout=w.Layout(width="320px"),
        )
        self.rate_on = w.Checkbox(
            value=run.rate is not None,
            description="throttle",
            indent=False,
            layout=w.Layout(width="90px"),
        )
        self.rate.observe(
            lambda c: run.set_rate(c["new"] if self.rate_on.value else None),
            names="value",
        )
        self.rate_on.observe(
            lambda c: run.set_rate(self.rate.value if c["new"] else None), names="value"
        )
        self.progress = w.FloatProgress(
            value=0, min=0, max=100, layout=w.Layout(width="320px")
        )
        self.progress_label = w.HTML()
        pace = w.HBox([self.rate_on, self.rate, self.progress, self.progress_label])

        # -- knobs, figures, status -------------------------------------- #
        rows: list[Any] = []
        if title:
            rows.append(w.HTML(f"<b style='font-family:monospace'>{title}</b>"))
        rows += [transport, pace]
        if self.knobs:
            rows += _knob_rows(self.knobs)
        self.debug: Any = None
        if debug:
            self.debug = DebugTools(run)
            rows.append(self.debug.widget)
        if self.live is not None:
            rows.append(self.diagram_html)
        self.break_html = w.HTML()  # the held frame, while a breakpoint holds one
        rows.append(self.break_html)
        self.images = [w.Image(format="png", width=width) for _ in self.plots]
        if self.images:
            rows.append(w.HBox(self.images, layout=w.Layout(flex_flow="row wrap")))
        self.status = w.HTML()
        rows.append(self.status)
        self.widget = w.VBox(
            rows,
            layout=w.Layout(border="1px solid #8884", padding="6px"),
        )

        run._ui = self.render
        self._thread: threading.Thread | None = None
        if auto:
            self._thread = threading.Thread(target=self._ticker, daemon=True)
            self._thread.start()
        else:
            self.render()

    def close(self) -> None:
        """Stop refreshing and close the widgets (the run itself goes on)."""
        self._closed = True
        if self.run._ui == self.render:
            self.run._ui = None
        if self.live is not None:
            self.live.close()
        try:
            self.widget.close()
        except Exception:
            pass

    # -- actions ---------------------------------------------------------- #
    def _snapshot(self) -> None:
        f = self.run.latest
        if f is not None:
            self.snapshots.append(f)
            self.run.snapshots.append(f)

    # -- rendering -------------------------------------------------------- #
    def render(self) -> None:
        from .notebook import monitor_html

        run, mon = self.run, self.run.monitor
        st = run.state
        head = (
            f"<span style='font-family:monospace'><b style='color:{_COLOURS[st]}'>{st}</b>"
            f" · {len(run.frames)} frames kept · {mon.produced} produced · {mon.elapsed:.1f}s"
            + (f" · recording {len(run.recording)}" if run.recording_on else "")
            + (f" · {len(run.snapshots)} snapshots" if run.snapshots else "")
            + (
                f" · <span style='color:#dc2626'>{run.exception!r}</span>"
                if run.exception
                else ""
            )
            + "</span>"
        )
        self.status.value = head + (monitor_html(mon) if self.table else "")
        pct = mon.percent
        if pct is not None:
            self.progress.value = pct
            self.progress_label.value = f"{pct:5.1f}%"
        else:
            self.progress.value = (
                100 if run.finished else (mon.elapsed * 25) % 100
            )  # activity sweep
            self.progress.bar_style = "" if not run.finished else "success"
            self.progress_label.value = f"{mon.produced} frames"
        if self.live is not None:
            self.diagram_html.value = self.live.html()
        if self.debug is not None:
            self.debug.render()
        self.break_html.value = held_html(run)
        for img, p in zip(self.images, self.plots):
            if p.frames:
                try:
                    img.value = p.png(dpi=80)
                except Exception:  # figure mid-redraw: skip this refresh
                    pass
        done = run.finished
        for b in (
            self.b_pause,
            self.b_resume,
            self.b_step,
            self.b_stop,
            self.b_snap,
            self.b_record,
        ):
            b.disabled = done
        held = run.held is not None
        self.b_pause.disabled = done or run.control.paused or held
        self.b_resume.disabled = done or not (run.control.paused or held)

    def _ticker(self) -> None:
        while not self.run.finished and not self._closed:
            try:
                self.render()
            except Exception:
                pass
            time.sleep(self.refresh)
        if self._closed:
            return
        try:
            self.render()
        except Exception:
            pass

    def _ipython_display_(self) -> None:
        from IPython.display import display

        display(self.widget)


def _knob_rows(items: list[Any]) -> list[Any]:
    """Knob and button widgets in rows: ungrouped ones first, then one labelled
    section per ``group``."""
    import ipywidgets as w

    sections: dict[str | None, list[Any]] = {}
    for item in items:
        sections.setdefault(getattr(item, "group", None), []).append(item)
    rows = []
    for group, members in sections.items():
        widgets = [
            button_widget(m) if hasattr(m, "press") else knob_widget(m) for m in members
        ]
        box = w.HBox(widgets, layout=w.Layout(flex_flow="row wrap"))
        if group is None:
            rows.append(box)
        else:
            rows.append(
                w.VBox(
                    [
                        w.HTML(
                            f"<b style='font-family:monospace'>{html.escape(group)}</b>"
                        ),
                        box,
                    ],
                    layout=w.Layout(border="1px solid #8882", padding="2px 6px"),
                )
            )
    return rows


class DebugTools:
    """The panel's debugging row (``run.controls(debug=True)``): pick a block and
    probe it (its latest frame shows below), or set a breakpoint on it with a
    condition written in Python over ``f``, the frame (``f["snr"] < 3``), and an
    optional tag; Clear removes the breakpoints set here."""

    def __init__(self, run: Any) -> None:
        import ipywidgets as w

        self.run = run
        names = sorted({n.name for n in run.graph.nodes()})
        self.block = w.Dropdown(options=names, description="block")
        self.condition = w.Text(
            value="", placeholder='f["snr"] < 3', description="break if"
        )
        self.tag = w.Combobox(options=[], placeholder="tag", description="tag")
        self.b_probe = w.Button(description="Probe", icon="eye")
        self.b_break = w.Button(description="Break", icon="hand-paper-o")
        self.b_clear = w.Button(description="Clear", icon="times")
        self.message = w.HTML()
        self.latest = w.HTML()
        self.probe: Any = None
        self.breakpoints: list[Any] = []
        self.b_probe.on_click(lambda _: self.watch(str(self.block.value)))
        self.b_break.on_click(
            lambda _: self.add_break(
                str(self.block.value), self.condition.value, self.tag.value or ""
            )
        )
        self.b_clear.on_click(lambda _: self.clear())
        self.widget = w.VBox(
            [
                w.HBox([self.block, self.b_probe]),
                w.HBox([self.condition, self.tag, self.b_break, self.b_clear]),
                self.message,
                self.latest,
            ],
            layout=w.Layout(border="1px solid #8882", padding="2px 6px"),
        )

    def watch(self, name: str) -> Any:
        if self.probe is not None:
            self.probe.close()
        self.probe = self.run.probe(name)
        self.message.value = (
            f"<span style='font-family:monospace'>probing {html.escape(name)}</span>"
        )
        return self.probe

    def add_break(self, name: str, condition: str = "", tag: str = "") -> Any:
        pred: Callable[[Any], Any] | None = None
        if condition.strip():
            try:
                code = compile(condition, "<break if>", "eval")
            except SyntaxError as e:
                self.message.value = f"<span style='color:#dc2626'>condition: {html.escape(str(e))}</span>"
                return None
            import numpy as np

            def holds(f: Any, code: Any = code) -> Any:
                # the user's own condition, typed in their own kernel
                return eval(code, {"np": np}, {"f": f})

            pred = holds

        bp = self.run.break_when(name, pred, tag=tag.strip() or None)
        self.breakpoints.append(bp)
        self.message.value = (
            f"<span style='font-family:monospace'>{html.escape(repr(bp))}</span>"
        )
        return bp

    def clear(self) -> None:
        for bp in self.breakpoints:
            bp.remove()
        self.breakpoints = []
        self.message.value = (
            "<span style='font-family:monospace'>breakpoints cleared</span>"
        )

    def render(self) -> None:
        from .notebook import frame_html

        seen = set(self.tag.options)
        for f in list(self.run.frames[-20:]):
            seen |= set(f.tags.names)
        if self.probe is not None:
            for f in list(self.probe.history):
                seen |= set(f.tags.names)
            latest = self.probe.latest
            self.latest.value = frame_html(latest) if latest is not None else ""
        self.tag.options = sorted(seen)
        errors = [e for bp in self.breakpoints for e in bp.errors]
        if errors:
            self.message.value = f"<span style='color:#dc2626'>condition failed: {html.escape(str(errors[-1]))}</span>"


def held_html(run: Any) -> str:
    """The frame a run holds at a breakpoint, with where and why, or nothing."""
    f = run.held
    if f is None:
        return ""
    from .notebook import frame_html

    by = run.control.held_by
    why = (
        "stepped"
        if by == "step"
        else "failed"
        if by == "error"
        else html.escape(repr(by))
    )
    return (
        f"<div style='border-left:3px solid {_COLOURS['break']};padding:4px 8px;margin:4px 0'>"
        f"<b style='color:{_COLOURS['break']};font-family:monospace'>break at {html.escape(str(run.held_at))}"
        f" #{f.prov.seq}</b> <span style='color:#6b7280'>· {why} · Resume lets it go, Step holds the next</span>"
        f"{frame_html(f)}</div>"
    )


def dashboard(runs: Any, **kw: Any) -> Any:
    """One panel per run (a dict name -> Run or a list), stacked, with a Stop-all button."""
    import ipywidgets as w
    from IPython.display import display

    items = (
        list(runs.items())
        if isinstance(runs, dict)
        else [(r.graph.name, r) for r in runs]
    )
    panels = [ControlPanel(r, title=name, table=False, **kw) for name, r in items]
    stop_all = w.Button(description="Stop all", icon="stop", button_style="danger")
    pause_all = w.Button(description="Pause all", icon="pause")
    resume_all = w.Button(description="Resume all", icon="play", button_style="success")
    stop_all.on_click(lambda _: [r.stop() for _, r in items])
    pause_all.on_click(lambda _: [r.pause() for _, r in items])
    resume_all.on_click(lambda _: [r.resume() for _, r in items])
    box = w.VBox(
        [w.HBox([pause_all, resume_all, stop_all])] + [p.widget for p in panels]
    )
    display(box)
    return panels
