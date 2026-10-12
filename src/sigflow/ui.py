"""Notebook control panels built on ipywidgets.

    run.controls(plots=[...], knobs=[...])   # full panel for one run
    dashboard(runs)                          # several runs, one panel each, stop-all
    Knob(...).widget()                       # a slider / toggle / dropdown / text box bound to a knob

Requires ``ipywidgets`` in the kernel (``pip install ipywidgets``, then restart the
kernel). Without it, ``run.controls()`` shows a status line and says so.
"""
from __future__ import annotations

import threading
import time
from typing import Any

__all__ = ["have_widgets", "require_widgets", "ControlPanel", "dashboard", "knob_widget"]

_COLOURS = {"running": "#16a34a", "paused": "#d97706", "stopped": "#6b7280",
            "done": "#2563eb", "failed": "#dc2626"}

_HELP = ("Interactive controls need <b>ipywidgets</b> in this kernel: "
         "<code>python -m pip install ipywidgets</code> (JupyterLab 3+/Notebook 7 pick it up "
         "automatically), then <b>restart the kernel</b>. Until then use "
         "<code>run.pause()</code>, <code>run.resume()</code>, <code>run.step()</code>, "
         "<code>run.stop()</code> and set <code>knob.value</code> from a cell.")


def have_widgets() -> bool:
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
        display(HTML(f"<span style='color:#16a34a'>ipywidgets {ipywidgets.__version__} found — "
                     f"buttons, sliders and live figures will work.</span>"))
    else:
        display(HTML(f"<div style='color:#dc2626'>{_HELP}</div>"))


# --------------------------------------------------------------------------- #
# Knob widgets
# --------------------------------------------------------------------------- #

def knob_widget(knob: Any) -> Any:
    """The natural ipywidgets control for a knob: dropdown for choices, toggle for
    bools, int/float sliders for numbers, a text box for strings."""
    import ipywidgets as w
    v = knob.value
    desc = knob.name or "value"
    if knob.choices is not None:
        wd = w.Dropdown(options=list(knob.choices), value=v, description=desc)
    elif isinstance(v, bool):
        wd = w.ToggleButton(value=v, description=desc, icon="check" if v else "")
    elif isinstance(v, int):
        lo = knob.min if knob.min is not None else (0 if v >= 0 else 2 * v)
        hi = knob.max if knob.max is not None else max(2 * v, lo + 10)
        wd = w.IntSlider(value=v, min=lo, max=hi, step=int(knob.step or 1), description=desc,
                         continuous_update=True)
    elif isinstance(v, (float,)):
        lo = knob.min if knob.min is not None else (0.0 if v >= 0 else 2 * v)
        hi = knob.max if knob.max is not None else (2 * v if v > 0 else 1.0)
        wd = w.FloatSlider(value=v, min=lo, max=hi, step=knob.step or (hi - lo) / 200, description=desc,
                           continuous_update=True, readout_format=".3g")
    else:
        wd = w.Text(value=str(v), description=desc, continuous_update=False)

    def changed(change: Any) -> None:
        new = change["new"]
        if isinstance(wd, w.Text) and not isinstance(v, str):
            try:
                new = type(v)(new)
            except (TypeError, ValueError):
                return
        knob._set_from_widget(new)
        if isinstance(wd, w.ToggleButton):
            wd.icon = "check" if new else ""

    wd.observe(changed, names="value")
    knob._widget = wd
    return wd


# --------------------------------------------------------------------------- #
# Control panel
# --------------------------------------------------------------------------- #

class ControlPanel:
    """Buttons, throttle, progress, knobs, live figures and status for one :class:`Run`."""

    def __init__(self, run: Any, plots: Any = (), knobs: Any = (), refresh: float = 0.4,
                 table: bool = True, width: int = 520, title: str | None = None) -> None:
        import ipywidgets as w

        self.run = run
        self.plots = [plots] if hasattr(plots, "draw_fn") else list(plots)
        self.knobs = [knobs] if hasattr(knobs, "_set_from_widget") else list(knobs)
        self.refresh = refresh
        self.table = table
        self.snapshots: list[Any] = []

        # -- transport --------------------------------------------------- #
        self.b_pause = w.Button(description="Pause", icon="pause", tooltip="hold the sources")
        self.b_resume = w.Button(description="Resume", icon="play", button_style="success")
        self.b_step = w.Button(description="Step", icon="step-forward", tooltip="release one source frame")
        self.steps = w.BoundedIntText(value=1, min=1, max=100000, layout=w.Layout(width="70px"),
                                      tooltip="frames per Step")
        self.b_stop = w.Button(description="Stop", icon="stop", button_style="danger")
        self.b_snap = w.Button(description="Snapshot", icon="camera", tooltip="keep the latest frame")
        self.b_record = w.ToggleButton(value=False, description="Record", icon="circle",
                                       tooltip="collect frames while on")
        self.b_pause.on_click(lambda _: run.pause())
        self.b_resume.on_click(lambda _: run.resume())
        self.b_step.on_click(lambda _: run.step(self.steps.value))
        self.b_stop.on_click(lambda _: run.stop())
        self.b_snap.on_click(lambda _: self._snapshot())
        self.b_record.observe(lambda c: run.set_recording(c["new"]), names="value")
        transport = w.HBox([self.b_pause, self.b_resume, self.b_step, self.steps, self.b_stop,
                            self.b_snap, self.b_record])

        # -- throttle + progress ----------------------------------------- #
        self.rate = w.FloatLogSlider(value=run.rate or 1e4, base=10, min=-1, max=4, step=0.05,
                                     description="max fps", readout_format=".3g",
                                     continuous_update=True, layout=w.Layout(width="320px"))
        self.rate_on = w.Checkbox(value=run.rate is not None, description="throttle", indent=False,
                                  layout=w.Layout(width="90px"))
        self.rate.observe(lambda c: run.set_rate(c["new"] if self.rate_on.value else None), names="value")
        self.rate_on.observe(lambda c: run.set_rate(self.rate.value if c["new"] else None), names="value")
        self.progress = w.FloatProgress(value=0, min=0, max=100, layout=w.Layout(width="320px"))
        self.progress_label = w.HTML()
        pace = w.HBox([self.rate_on, self.rate, self.progress, self.progress_label])

        # -- knobs, figures, status -------------------------------------- #
        rows: list[Any] = []
        if title:
            rows.append(w.HTML(f"<b style='font-family:monospace'>{title}</b>"))
        rows += [transport, pace]
        if self.knobs:
            rows.append(w.HBox([knob_widget(k) for k in self.knobs], layout=w.Layout(flex_flow="row wrap")))
        self.images = [w.Image(format="png", width=width) for _ in self.plots]
        if self.images:
            rows.append(w.HBox(self.images, layout=w.Layout(flex_flow="row wrap")))
        self.status = w.HTML()
        rows.append(self.status)
        self.widget = w.VBox(rows, layout=w.Layout(border="1px solid #8884", padding="6px", border_radius="6px"))

        run._ui = self.render
        self._thread = threading.Thread(target=self._ticker, daemon=True)
        self._thread.start()

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
        head = (f"<span style='font-family:monospace'><b style='color:{_COLOURS[st]}'>{st}</b>"
                f" · {len(run.frames)} frames kept · {mon.produced} produced · {mon.elapsed:.1f}s"
                + (f" · recording {len(run.recording)}" if run.recording_on else "")
                + (f" · {len(run.snapshots)} snapshots" if run.snapshots else "")
                + (f" · <span style='color:#dc2626'>{run.exception!r}</span>" if run.exception else "")
                + "</span>")
        self.status.value = head + (monitor_html(mon) if self.table else "")
        pct = mon.percent
        if pct is not None:
            self.progress.value = pct
            self.progress_label.value = f"{pct:5.1f}%"
        else:
            self.progress.value = 100 if run.finished else (mon.elapsed * 25) % 100   # activity sweep
            self.progress.bar_style = "" if not run.finished else "success"
            self.progress_label.value = f"{mon.produced} frames"
        for img, p in zip(self.images, self.plots):
            if p.frames:
                try:
                    img.value = p.png(dpi=80)
                except Exception:       # figure mid-redraw: skip this refresh
                    pass
        done = run.finished
        for b in (self.b_pause, self.b_resume, self.b_step, self.b_stop, self.b_snap, self.b_record):
            b.disabled = done
        self.b_pause.disabled = done or run.control.paused
        self.b_resume.disabled = done or not run.control.paused

    def _ticker(self) -> None:
        while not self.run.finished:
            try:
                self.render()
            except Exception:
                pass
            time.sleep(self.refresh)
        try:
            self.render()
        except Exception:
            pass

    def _ipython_display_(self) -> None:
        from IPython.display import display
        display(self.widget)


def dashboard(runs: Any, **kw: Any) -> Any:
    """One panel per run (a dict name -> Run or a list), stacked, with a Stop-all button."""
    import ipywidgets as w
    from IPython.display import display
    items = list(runs.items()) if isinstance(runs, dict) else [(r.graph.name, r) for r in runs]
    panels = [ControlPanel(r, title=name, table=False, **kw) for name, r in items]
    stop_all = w.Button(description="Stop all", icon="stop", button_style="danger")
    pause_all = w.Button(description="Pause all", icon="pause")
    resume_all = w.Button(description="Resume all", icon="play", button_style="success")
    stop_all.on_click(lambda _: [r.stop() for _, r in items])
    pause_all.on_click(lambda _: [r.pause() for _, r in items])
    resume_all.on_click(lambda _: [r.resume() for _, r in items])
    box = w.VBox([w.HBox([pause_all, resume_all, stop_all])] + [p.widget for p in panels])
    display(box)
    return panels
