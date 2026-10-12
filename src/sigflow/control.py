"""Interactive control of long-running graphs: start / pause / step / stop.

    run = g.start(progress=False)      # returns immediately; the graph runs in the background
    run.controls()                      # buttons + live status in a notebook (ipywidgets if available)
    run.pause(); run.step(); run.resume(); run.stop()
    run.frames                          # outputs so far
    run.wait()                          # block until finished, returns all frames
    run.state                           # "running" | "paused" | "stopped" | "done" | "failed"

Pausing gates the *sources*: in-flight frames drain through the graph and then
everything idles until resumed. ``step()`` lets exactly one source frame
through. ``stop()`` cancels the run; frames produced so far are kept.
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Callable

from .frame import Frame

__all__ = ["Control", "Run", "Knob", "live"]


class Knob:
    """A tunable configuration value: ``scale(gain=Knob(1.0))``. Map blocks read it on
    every call, so changing ``knob.value`` (or dragging ``knob.slider()``) takes effect
    while the graph runs. Stream blocks receive the Knob object itself and should read
    ``knob.value`` inside their loop (or just use ``float(knob)`` / ``knob()``)."""

    def __init__(self, value: Any, name: str | None = None, min: float | None = None,
                 max: float | None = None, step: float | None = None, choices: Any = None) -> None:
        self._value = value
        self.name = name
        self.min, self.max, self.step = min, max, step
        self.choices = list(choices) if choices is not None else None
        self._lock = threading.Lock()
        self._widget: Any = None
        self._listeners: list[Callable[[Any], None]] = []

    @property
    def value(self) -> Any:
        with self._lock:
            return self._value

    @value.setter
    def value(self, v: Any) -> None:
        with self._lock:
            self._value = v
        if self._widget is not None and self._widget.value != v:
            try:
                self._widget.value = v
            except Exception:
                pass
        for fn in self._listeners:
            fn(v)

    def _set_from_widget(self, v: Any) -> None:
        with self._lock:
            self._value = v
        for fn in self._listeners:
            fn(v)

    def on_change(self, fn: Callable[[Any], None]) -> "Knob":
        """Call ``fn(new_value)`` whenever the knob changes (from code or a widget)."""
        self._listeners.append(fn)
        return self

    def widget(self) -> Any:
        """The natural ipywidgets control for this knob (slider, toggle, dropdown or text)."""
        from .ui import knob_widget
        return knob_widget(self)

    def __call__(self) -> Any:
        return self.value

    def __float__(self) -> float:
        return float(self.value)

    def __int__(self) -> int:
        return int(self.value)

    def __repr__(self) -> str:
        return f"Knob({self.value!r}{', ' + self.name if self.name else ''})"

    def slider(self, min: float | None = None, max: float | None = None, step: float | None = None) -> Any:
        """An ipywidgets slider bound to this knob (both directions)."""
        if min is not None:
            self.min = min
        if max is not None:
            self.max = max
        if step is not None:
            self.step = step
        return self.widget()

    toggle = widget


def live(config: Any) -> dict:
    """Resolve Knob values in a configuration mapping (used by the runtime per call)."""
    return {k: (v.value if isinstance(v, Knob) else v) for k, v in config.items()}


class Control:
    """Shared gate between the runtime and the user (thread-safe: used from the UI thread)."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._gate: asyncio.Event | None = None
        self._paused = False
        self._steps = 0
        self.stop_requested = False
        self.rate: float | None = None        # max source frames per second (None = unlimited)
        self._last_release = 0.0
        self._lock = threading.Lock()

    # -- runtime side ------------------------------------------------------ #
    def attach(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._gate = asyncio.Event()
        if not self._paused or self._steps > 0:      # step() may have been called before attach
            self._gate.set()

    async def wait(self) -> None:
        """Called by sources before emitting each item."""
        if self._gate is None:
            return
        if self.stop_requested:
            raise asyncio.CancelledError
        await self._gate.wait()
        if self.stop_requested:
            raise asyncio.CancelledError
        rate = self.rate
        if rate:
            now = time.perf_counter()
            delay = self._last_release + 1.0 / rate - now
            if delay > 0:
                await asyncio.sleep(delay)
            self._last_release = max(now, self._last_release + 1.0 / rate)
        with self._lock:
            if self._steps > 0:
                self._steps -= 1
                if self._steps == 0 and self._paused:
                    self._gate.clear()

    # -- user side --------------------------------------------------------- #
    @property
    def paused(self) -> bool:
        return self._paused

    def _apply(self, fn: Callable[[], None]) -> None:
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(fn)
        else:
            fn()

    def pause(self) -> None:
        with self._lock:
            self._paused = True
            self._steps = 0
        if self._gate is not None:
            self._apply(self._gate.clear)

    def resume(self) -> None:
        with self._lock:
            self._paused = False
            self._steps = 0
        if self._gate is not None:
            self._apply(self._gate.set)

    def step(self, n: int = 1) -> None:
        """Let ``n`` source frames through, then (stay) paused."""
        with self._lock:
            self._paused = True
            self._steps += n
        if self._gate is not None:
            self._apply(self._gate.set)

    def stop(self) -> None:
        self.stop_requested = True
        if self._gate is not None:
            self._apply(self._gate.set)      # wake anything waiting so it can cancel


class Run:
    """Handle on a graph running in a background thread with its own event loop."""

    def __init__(self, graph: Any, paused: bool = False, keep: int | None = None, **options: Any) -> None:
        from .progress import Monitor
        self.graph = graph
        self.keep = keep                     # max frames to retain (None = all)
        self.control = Control()
        self.monitor: Monitor = options.pop("monitor", None) or Monitor()
        self.frames: list[Frame] = []
        self.recording: list[Frame] = []     # frames collected while Record is on
        self.recording_on = False
        self.snapshots: list[Frame] = []     # frames kept with the Snapshot button / .snapshot()
        self.exception: BaseException | None = None
        self._options = options
        self._done = threading.Event()
        self._cancelled = False
        self._ui: Any = None
        if paused:
            self.control.pause()
        self._thread = threading.Thread(target=self._main, name="sigflow-run", daemon=True)
        self._thread.start()

    # -- execution --------------------------------------------------------- #
    def _main(self) -> None:
        from .runtime import arun

        async def body() -> None:
            self.control.attach(asyncio.get_running_loop())
            opts = dict(self._options)
            opts.setdefault("collect", False)
            sink = opts.pop("sink", None)

            def collect(f: Frame) -> None:
                self.frames.append(f)
                if self.keep is not None and len(self.frames) > self.keep:
                    del self.frames[: len(self.frames) - self.keep]
                if self.recording_on:
                    self.recording.append(f)
                if sink is not None:
                    sink(f)

            await arun(self.graph, sink=collect, monitor=self.monitor, control=self.control, **opts)

        try:
            asyncio.run(body())
        except asyncio.CancelledError:
            self._cancelled = True
        except BaseException as e:   # noqa: BLE001 - surfaced via .exception
            self.exception = e
        finally:
            self._done.set()

    # -- control ----------------------------------------------------------- #
    def pause(self) -> "Run":
        self.control.pause()
        self._refresh()
        return self

    def resume(self) -> "Run":
        self.control.resume()
        self._refresh()
        return self

    def step(self, n: int = 1) -> "Run":
        self.control.step(n)
        self._refresh()
        return self

    def stop(self) -> "Run":
        self.control.stop()
        self._done.wait(timeout=5)
        self._refresh()
        return self

    @property
    def rate(self) -> float | None:
        return self.control.rate

    def set_rate(self, frames_per_second: float | None) -> "Run":
        """Throttle the sources to at most this many frames per second (None = unlimited)."""
        self.control.rate = frames_per_second
        return self

    def set_recording(self, on: bool) -> "Run":
        self.recording_on = bool(on)
        return self

    def snapshot(self) -> Frame | None:
        f = self.latest
        if f is not None:
            self.snapshots.append(f)
        return f

    def checkpoint(self, path: str | None = None) -> Any:
        """Capture source positions and Stateful block states (pause first for a clean cut);
        ``g.start(resume=ckpt_or_path)`` / ``run(g, resume=...)`` continues from it."""
        from .persist import checkpoint
        return checkpoint(self.graph, self.monitor, path)

    def wait(self, timeout: float | None = None) -> list[Frame]:
        self._done.wait(timeout)
        if self.exception is not None:
            raise self.exception
        return self.frames

    # -- state ------------------------------------------------------------- #
    @property
    def finished(self) -> bool:
        return self._done.is_set()

    @property
    def state(self) -> str:
        if not self.finished:
            return "paused" if self.control.paused else "running"
        if self.exception is not None:
            return "failed"
        return "stopped" if self._cancelled else "done"

    @property
    def data(self) -> list[Any]:
        return [f.data for f in self.frames]

    def drain(self) -> list[Frame]:
        """Take the frames collected so far and clear the buffer (for endless streams)."""
        out, self.frames[:] = list(self.frames), []
        return out

    @property
    def latest(self) -> Frame | None:
        return self.frames[-1] if self.frames else None

    def status(self) -> str:
        pct = self.monitor.percent
        head = f"{pct:5.1f}%" if pct is not None else f"{len(self.frames)} frames"
        return f"[{self.state}] {head}  {self.monitor.elapsed:.2f}s"

    def __repr__(self) -> str:
        return f"<Run {self.graph.name!r} {self.state}: {len(self.frames)} frames>"

    # -- notebook UI ------------------------------------------------------- #
    def controls(self, plots: Any = (), knobs: Any = (), refresh: float = 0.4, table: bool = True,
                 width: int = 520, title: str | None = None) -> Any:
        """Display an interactive panel for this run (kept as ``run.panel``):

        Pause / Resume / Step (with a frame count) / Stop, Snapshot and Record buttons,
        a throttle slider (max source frames per second), a progress bar, one control per
        Knob in ``knobs`` (slider, toggle, dropdown or text), the figures of the plot blocks
        in ``plots`` refreshed live, and the per-block status table.

        Needs ``ipywidgets`` in the kernel; otherwise a status line and instructions appear."""
        from .ui import ControlPanel, have_widgets
        if not have_widgets():
            return self._html_controls(refresh)
        from IPython.display import display
        self.panel = ControlPanel(self, plots=plots, knobs=knobs, refresh=refresh, table=table,
                                  width=width, title=title)
        self.widget = self.panel.widget
        display(self.widget)
        return None

    def _html_controls(self, refresh: float) -> Any:
        from IPython.display import HTML, display
        from .ui import _HELP
        handle = display(HTML(""), display_id=True)

        def render() -> None:
            from .notebook import monitor_html
            handle.update(HTML(
                f"<div style='font-family:monospace'><b>{self.state}</b> · {len(self.frames)} frames"
                f"<div style='color:#dc2626;margin:4px 0'>{_HELP}</div>{monitor_html(self.monitor)}</div>"))

        def ticker() -> None:
            while not self.finished:
                try:
                    render()
                except Exception:
                    pass
                time.sleep(refresh)
            render()

        threading.Thread(target=ticker, daemon=True).start()
        return None

    def _refresh(self) -> None:
        if self._ui is not None:
            try:
                self._ui()
            except Exception:   # pragma: no cover
                pass
