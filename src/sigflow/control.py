"""Interactive control of long-running graphs: start / pause / step / stop.

    run = g.start(progress=False)      # returns immediately; the graph runs in the background
    run.controls()                      # buttons + live status in a notebook (ipywidgets if available)
    run.pause(); run.step(); run.resume(); run.stop()
    run.frames                          # outputs so far
    run.wait()                          # block until finished, returns all frames
    run.state                           # "running" | "paused" | "stopped" | "done" | "failed"
    run.probe("fft")                    # watch one block's output without editing the graph

Pausing gates the *sources*: in-flight frames drain through the graph and then
everything idles until resumed. ``step()`` lets exactly one source frame
through. ``stop()`` cancels the run; frames produced so far are kept.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from typing import Any

from .frame import Frame

__all__ = ["Control", "Knob", "Probe", "Run", "live"]


class Knob:
    """A tunable configuration value: ``scale(gain=Knob(1.0))``. Map blocks read it on
    every call, so changing ``knob.value`` (or dragging ``knob.slider()``) takes effect
    while the graph runs. Stream blocks receive the Knob object itself and should read
    ``knob.value`` inside their loop (or just use ``float(knob)`` / ``knob()``)."""

    def __init__(
        self,
        value: Any,
        name: str | None = None,
        min: float | None = None,
        max: float | None = None,
        step: float | None = None,
        choices: Any = None,
    ) -> None:
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

    def on_change(self, fn: Callable[[Any], None]) -> Knob:
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

    def slider(
        self,
        min: float | None = None,
        max: float | None = None,
        step: float | None = None,
    ) -> Any:
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
        self.rate: float | None = (
            None  # max source frames per second (None = unlimited)
        )
        self._last_release = 0.0
        self._lock = threading.Lock()

    # -- runtime side ------------------------------------------------------ #
    def attach(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._gate = asyncio.Event()
        if (
            not self._paused or self._steps > 0
        ):  # step() may have been called before attach
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
            self._apply(self._gate.set)  # wake anything waiting so it can cancel


class Probe:
    """A window on one block of a running graph (``run.probe("fft")``): every frame the
    block puts out, as the next block would receive it, without editing the graph.

    ``latest`` is the newest frame, ``history`` the last ``keep``, ``count`` how many
    were seen (``failed`` of them failed). ``on(fn)`` calls ``fn(frame)`` for each,
    ``plot(block)`` feeds a plot block as if it were placed after the probed block,
    ``wait(n)`` blocks until ``n`` more frames arrive, ``show()`` displays the latest
    frame in a notebook and keeps it fresh, ``close()`` detaches. Callbacks run on
    the run's thread and must be quick; one that raises is recorded in ``errors``
    and dropped, never the run.
    """

    def __init__(
        self,
        run: Run,
        name: str,
        keep: int = 16,
        plot: Any = None,
        node: Any = None,
    ) -> None:
        from collections import deque

        self.run = run
        self.name = name
        self.node = node  # when probing by node: exact, whatever name it is shown under
        self.history: deque[Frame] = deque(maxlen=max(1, keep))
        self.count = 0
        self.failed = 0
        self.errors: list[BaseException] = []
        self.plots: list[Any] = []
        self._callbacks: list[Callable[[Frame], Any]] = []
        self._cv = threading.Condition()
        self._handle: Any = None
        self._closed = False
        if plot is not None:
            self.plot(plot)
        if node is not None:
            run.monitor.watch_node(node, self._see)
        else:
            run.monitor.watch(name, self._see)

    # -- what arrives --------------------------------------------------------- #
    def _see(self, frame: Frame) -> None:
        with self._cv:
            self.history.append(frame)
            self.count += 1
            if frame.failed:
                self.failed += 1
            self._cv.notify_all()
        for fn in list(self._callbacks):
            try:
                fn(frame)
            except Exception as e:
                self.errors.append(e)
                self._callbacks.remove(fn)
        for p in self.plots:
            try:
                p._on_frame(frame)
            except Exception as e:
                self.errors.append(e)

    @property
    def latest(self) -> Frame | None:
        return self.history[-1] if self.history else None

    @property
    def data(self) -> Any:
        f = self.latest
        return None if f is None else f.data

    def on(self, fn: Callable[[Frame], Any]) -> Probe:
        self._callbacks.append(fn)
        return self

    def plot(self, block: Any) -> Probe:
        """Feed a plot block (``plot.line()``, ``plot.spectrum()``, ...) with every frame;
        show it with ``run.controls(plots=[block])`` or ``block.figure``."""
        if not hasattr(block, "_on_frame"):
            raise TypeError("probe.plot() takes a plot block (sigflow.plot.*)")
        self.plots.append(block)
        return self

    def wait(self, n: int = 1, timeout: float | None = None) -> Frame | None:
        """Block until ``n`` more frames have arrived (or the run finishes, or
        ``timeout`` seconds pass); returns the latest frame."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cv:
            target = self.count + n
            while self.count < target and not self.run.finished and not self._closed:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    break
                self._cv.wait(0.05 if remaining is None else min(0.05, remaining))
            return self.latest

    def close(self) -> None:
        self._closed = True
        self.run.monitor.unwatch(self.name, self._see)
        with self._cv:
            self._cv.notify_all()

    # -- notebook --------------------------------------------------------------- #
    def html(self) -> str:
        from .notebook import frame_html

        f = self.latest
        head = (
            f"<div style='font-family:monospace'><b>probe {self.name}</b> · "
            f"{self.count} frames"
            + (f" · {self.failed} failed" if self.failed else "")
            + "</div>"
        )
        return head + (frame_html(f) if f is not None else "<i>no frames yet</i>")

    def _repr_html_(self) -> str:
        return self.html()

    def show(self, refresh: float = 0.5) -> None:
        """Display the latest frame and keep it fresh until the run finishes or
        ``close()``."""
        from IPython.display import HTML, display

        self._handle = display(HTML(self.html()), display_id=True)

        def ticker() -> None:
            while not self.run.finished and not self._closed:
                self._update()
                time.sleep(refresh)
            self._update()

        threading.Thread(target=ticker, name="sigflow-probe", daemon=True).start()

    def _update(self) -> None:
        if self._handle is None:
            return
        from IPython.display import HTML

        try:
            self._handle.update(HTML(self.html()))
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"<Probe {self.name!r}: {self.count} frames, latest={self.latest!r}>"


class Run:
    """Handle on a graph running in a background thread with its own event loop."""

    def __init__(
        self, graph: Any, paused: bool = False, keep: int | None = None, **options: Any
    ) -> None:
        from .progress import Monitor

        self.graph = graph
        self.keep = keep  # max frames to retain (None = all)
        self.control = Control()
        self.monitor: Monitor = options.pop("monitor", None) or Monitor()
        self.frames: list[Frame] = []
        self.recording: list[Frame] = []  # frames collected while Record is on
        self.recording_on = False
        self.snapshots: list[
            Frame
        ] = []  # frames kept with the Snapshot button / .snapshot()
        self.exception: BaseException | None = None
        self.probes: list[Probe] = []
        self._options = options
        self._done = threading.Event()
        self._cancelled = False
        self._ui: Any = None
        if paused:
            self.control.pause()
        self._thread = threading.Thread(
            target=self._main, name="sigflow-run", daemon=True
        )
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

            await arun(
                self.graph,
                sink=collect,
                monitor=self.monitor,
                control=self.control,
                **opts,
            )

        try:
            asyncio.run(body())
        except asyncio.CancelledError:
            self._cancelled = True
        except BaseException as e:
            self.exception = e
        finally:
            self._done.set()

    # -- control ----------------------------------------------------------- #
    def pause(self) -> Run:
        self.control.pause()
        self._refresh()
        return self

    def resume(self) -> Run:
        self.control.resume()
        self._refresh()
        return self

    def step(self, n: int = 1) -> Run:
        self.control.step(n)
        self._refresh()
        return self

    def stop(self) -> Run:
        self.control.stop()
        self._done.wait(timeout=5)
        self._refresh()
        return self

    @property
    def rate(self) -> float | None:
        return self.control.rate

    def set_rate(self, frames_per_second: float | None) -> Run:
        """Throttle the sources to at most this many frames per second (None = unlimited)."""
        self.control.rate = frames_per_second
        return self

    def set_recording(self, on: bool) -> Run:
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

    # -- probes -------------------------------------------------------------- #
    def probe(self, target: Any, keep: int = 16, plot: Any = None) -> Probe:
        """Watch one block's output while the graph runs, without editing the graph:
        ``target`` is a block's displayed name (``"fft"``, ``"fft#2"`` for a second
        block of the same name, ``"fanout"`` / ``"zip"`` for a join; the names of
        ``run.monitor.table()``) or a node of ``run.graph``. Returns a :class:`Probe`;
        ``keep`` is how many recent frames it holds, ``plot`` a plot block to feed."""
        if isinstance(target, str):
            name = target
            base = name.split("#", 1)[0]
            known = {n.name for n in self.graph.nodes()} | {"fanout", "zip"}
            if base not in known:
                raise KeyError(
                    f"no block called {base!r} in {self.graph.name!r}; "
                    f"blocks: {sorted(n for n in known if n not in ('fanout', 'zip'))}"
                )
            probe = Probe(self, name, keep=keep, plot=plot)
        else:
            probe = Probe(self, target.name, keep=keep, plot=plot, node=target)
        self.probes.append(probe)
        return probe

    # -- notebook UI ------------------------------------------------------- #
    def diagram(self, refresh: float = 0.5, types: bool = True) -> Any:
        """The graph's diagram, live: frame counts, errors and a state colour on every
        block, refreshed while the run goes (kept as ``run.live_diagram``). In a
        notebook it is displayed and None is returned; elsewhere the
        :class:`~sigflow.draw.LiveDiagram` is returned, its ``svg`` always current."""
        from .draw import LiveDiagram
        from .progress import _in_notebook

        self.live_diagram = LiveDiagram(self, refresh=refresh, types=types)
        if not _in_notebook():
            return self.live_diagram
        self.live_diagram.show()
        return None

    def controls(
        self,
        plots: Any = (),
        knobs: Any = (),
        refresh: float = 0.4,
        table: bool = True,
        width: int = 520,
        title: str | None = None,
        diagram: bool = False,
    ) -> Any:
        """Display an interactive panel for this run (kept as ``run.panel``):

        Pause / Resume / Step (with a frame count) / Stop, Snapshot and Record buttons,
        a throttle slider (max source frames per second), a progress bar, one control per
        Knob in ``knobs`` (slider, toggle, dropdown or text), the figures of the plot blocks
        in ``plots`` refreshed live, with ``diagram=True`` the live diagram of the graph,
        and the per-block status table.

        Needs ``ipywidgets`` in the kernel; otherwise a status line and instructions appear."""
        from .ui import ControlPanel, have_widgets

        if not have_widgets():
            return self._html_controls(refresh)
        from IPython.display import display

        self.panel = ControlPanel(
            self,
            plots=plots,
            knobs=knobs,
            refresh=refresh,
            table=table,
            width=width,
            title=title,
            diagram=diagram,
        )
        self.widget = self.panel.widget
        display(self.widget)
        return None

    def _html_controls(self, refresh: float) -> Any:
        from IPython.display import HTML, display

        from .ui import _HELP

        handle = display(HTML(""), display_id=True)

        def render() -> None:
            from .notebook import monitor_html

            if handle is None:
                return
            handle.update(
                HTML(
                    f"<div style='font-family:monospace'><b>{self.state}</b> · {len(self.frames)} frames"
                    f"<div style='color:#dc2626;margin:4px 0'>{_HELP}</div>{monitor_html(self.monitor)}</div>"
                )
            )

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
            except Exception:  # pragma: no cover
                pass
