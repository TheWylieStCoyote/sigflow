"""Plotting blocks built on matplotlib.

A plot block is a *tap*: it draws every frame it sees and passes the frame on
unchanged, so it can sit anywhere in a graph.  Each one keeps its figure and
the sequence of frames it drew, so after a run you can

* look at the final picture        – ``p.figure``
* animate the whole sequence       – ``p.animate()`` (a ``FuncAnimation``)
* save it                          – ``p.save("out.gif")``
* show it inline in Jupyter        – ``p.html()`` / just evaluate ``p``

Set ``live=True`` for a plot that updates in the notebook while running
(blocking ``run()`` only); for a background ``start()`` put the plot in
``run.controls(plots=[...])``.  Plot options (``ylim``, ``lim``, ``db``,
``rows`` ...) may be :class:`~sigflow.Knob` objects and are read at every draw.

Custom plots: ``plot.custom(draw)`` where ``draw(ax, data, frame)`` draws one frame.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .block import tap
from .frame import Frame
from .graph import Block
from .types import Sig

__all__ = [
    "PlotBlock",
    "line",
    "spectrum",
    "waterfall",
    "constellation",
    "custom",
    "animate",
    "history",
    "multi",
    "Multi",
]


def _matplotlib() -> Any:
    import matplotlib
    import matplotlib.pyplot as plt

    return plt


class PlotBlock(Block):
    """Base class: a tap that draws frames with ``draw(ax, data, frame)``."""

    def __init__(
        self,
        draw: Callable[[Any, Any, Frame], None],
        *,
        name: str = "plot",
        in_type: Any = None,
        title: str | None = None,
        figsize: tuple[float, float] = (7, 3.2),
        keep: int | None = None,
        live: bool = False,
        every: int = 1,
    ) -> None:
        self.draw_fn = draw
        self.title = title
        self.figsize = figsize
        self.keep = keep  # how many frames to remember for animation (None = all)
        self.live = live
        self.every = every
        self.frames: list[Frame] = []
        import threading

        self.lock = (
            threading.Lock()
        )  # matplotlib is not thread-safe: redraw vs. rasterise
        self._fig = None
        self._ax = None
        self._handle = None
        self._count = 0
        super().__init__(
            tap(
                self._on_frame,
                name=name,
                inputs={"f": Frame} if in_type is None else {"f": _framed(in_type)},
            )
        )

    # -- drawing ----------------------------------------------------------- #
    @property
    def figure(self) -> Any:
        if self._fig is None:
            plt = _matplotlib()
            self._fig, self._ax = plt.subplots(figsize=self.figsize)
            if self.title:
                self._ax.set_title(self.title)
            # Keep the figure out of pyplot's auto-display (inline backends show
            # every open figure at the end of a cell); it is displayed explicitly
            # via ``p.figure``, ``p.html()`` or live updates.
            plt.close(self._fig)
        return self._fig

    @property
    def ax(self) -> Any:
        self.figure
        return self._ax

    def _on_frame(self, f: Frame) -> None:
        self._count += 1
        if (self._count - 1) % self.every:
            return
        self.frames.append(f)
        if self.keep is not None and len(self.frames) > self.keep:
            del self.frames[0]
        if f.failed:
            return
        self.redraw(f)
        if self.live:
            self._update_live()

    def redraw(self, f: Frame) -> None:
        with self.lock:
            ax = self.ax
            ax.clear()
            if self.title:
                ax.set_title(self.title)
            self.draw_fn(ax, f.data, f)
            self.figure.canvas.draw_idle()

    def png(self, dpi: int = 100) -> bytes:
        """The current figure as PNG bytes (safe to call from another thread)."""
        import io

        buf = io.BytesIO()
        with self.lock:
            self.figure.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        return buf.getvalue()

    def _update_live(self) -> None:
        import matplotlib

        if (
            "ipympl" in matplotlib.get_backend().lower()
            or "widget" in matplotlib.get_backend().lower()
        ):
            # ipympl: the figure is an interactive widget; it redraws itself (zoom/pan keep working).
            try:
                self.figure.canvas.draw_idle()
                self.figure.canvas.flush_events()
            except Exception:
                pass
            return
        try:
            from IPython.display import display
        except ImportError:  # pragma: no cover
            return
        if self._handle is None:
            self._handle = display(self.figure, display_id=True)
        else:
            self._handle.update(self.figure)

    def show(self) -> Any:
        """Display the figure (in ipympl mode this is the live, interactive canvas)."""
        from IPython.display import display

        return display(self.figure)

    # -- animation --------------------------------------------------------- #
    def animate(self, interval: int = 100, repeat: bool = True) -> Any:
        """A ``FuncAnimation`` replaying every recorded frame."""
        from matplotlib.animation import FuncAnimation

        frames = [f for f in self.frames if not f.failed]
        if not frames:
            raise ValueError("no frames to animate")
        fig = self.figure

        def update(i: int) -> Any:
            self.redraw(frames[i])
            return []

        return FuncAnimation(
            fig,
            update,
            frames=len(frames),
            interval=interval,
            repeat=repeat,
            blit=False,
        )

    def save(self, path: str, interval: int = 100, **kw: Any) -> str:
        """Save the animation (``.gif`` via Pillow, ``.mp4`` via ffmpeg when available)."""
        anim = self.animate(interval=interval)
        writer = kw.pop("writer", "pillow" if path.lower().endswith(".gif") else None)
        anim.save(path, writer=writer, **kw)
        return path

    def html(self, interval: int = 100) -> Any:
        """Animation as HTML (JavaScript player) for Jupyter."""
        from IPython.display import HTML

        return HTML(self.animate(interval=interval).to_jshtml())

    def _repr_html_(self) -> str:
        import base64

        if not self.frames:
            return f"<i>{self.name}: no frames yet</i>"
        data = base64.b64encode(self.png()).decode()
        return f"<img src='data:image/png;base64,{data}' alt='{self.name}'/>"


def _v(x: Any) -> Any:
    """Resolve a Knob (or plain value) at draw time, so plot options can be tuned live."""
    return x.value if hasattr(x, "_set_from_widget") else x


def _framed(t: Any) -> Any:
    from .block import Framed

    return Framed(t)


# --------------------------------------------------------------------------- #
# Ready-made plots
# --------------------------------------------------------------------------- #


def line(
    title: str | None = None,
    xlabel: str = "sample",
    ylabel: str = "",
    ylim: Any = None,
    **kw: Any,
) -> PlotBlock:
    """Plot each 1-D frame as a line (complex frames show real and imaginary parts)."""

    def draw(ax: Any, x: Any, f: Frame) -> None:
        x = np.asarray(x)
        rate = f.stream.get("rate")
        t = np.arange(len(x)) / rate if rate else np.arange(len(x))
        if np.iscomplexobj(x):
            ax.plot(t, x.real, label="I")
            ax.plot(t, x.imag, label="Q")
            ax.legend(loc="upper right")
        else:
            ax.plot(t, x)
        ax.set_xlabel("time [s]" if rate else xlabel)
        ax.set_ylabel(ylabel or f.stream.get("units", ""))
        if _v(ylim) is not None:
            ax.set_ylim(*_v(ylim))
        _annotate(ax, f)

    return PlotBlock(draw, name="line", in_type=Sig["T", "N"], title=title, **kw)


def spectrum(
    title: str | None = None,
    db: bool = True,
    centered: bool = True,
    ylim: Any = None,
    **kw: Any,
) -> PlotBlock:
    """Plot the magnitude spectrum of a complex (or real) frame, with a Hz axis when
    the stream carries ``rate``."""

    def draw(ax: Any, x: Any, f: Frame) -> None:
        x = np.asarray(x)
        n = len(x)
        X = np.fft.fft(x)
        rate = f.stream.get("rate")
        freqs = np.fft.fftfreq(n, d=1.0 / rate if rate else 1.0)
        if _v(centered):
            X, freqs = np.fft.fftshift(X), np.fft.fftshift(freqs)
        mag = np.abs(X) ** 2 / n
        y = 10 * np.log10(mag + 1e-12) if _v(db) else mag
        ax.plot(freqs, y)
        ax.set_xlabel("frequency [Hz]" if rate else "bin")
        ax.set_ylabel("power [dB]" if _v(db) else "power")
        if _v(ylim) is not None:
            ax.set_ylim(*_v(ylim))
        _annotate(ax, f)

    return PlotBlock(draw, name="spectrum", in_type=Sig["T", "N"], title=title, **kw)


def waterfall(
    title: str | None = None, rows: int = 64, db: bool = True, **kw: Any
) -> PlotBlock:
    """Scrolling spectrogram: each incoming frame becomes one row."""
    history: list[np.ndarray] = []

    def draw(ax: Any, x: Any, f: Frame) -> None:
        x = np.asarray(x)
        X = np.fft.fftshift(np.abs(np.fft.fft(x)) ** 2 / len(x))
        history.append(10 * np.log10(X + 1e-12) if _v(db) else X)
        while len(history) > _v(rows):
            del history[0]
        img = np.stack(history)
        rate = f.stream.get("rate")
        extent = (-rate / 2, rate / 2, len(history), 0) if rate else None
        ax.imshow(img, aspect="auto", origin="upper", extent=extent, cmap="viridis")
        ax.set_xlabel("frequency [Hz]" if rate else "bin")
        ax.set_ylabel("frame")
        _annotate(ax, f)

    return PlotBlock(draw, name="waterfall", in_type=Sig["T", "N"], title=title, **kw)


def constellation(
    title: str | None = None, lim: float | None = None, **kw: Any
) -> PlotBlock:
    """Scatter plot of complex samples (I/Q)."""

    def draw(ax: Any, x: Any, f: Frame) -> None:
        x = np.asarray(x)
        ax.scatter(x.real, x.imag, s=4, alpha=0.6)
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.set_aspect("equal")
        if _v(lim):
            ax.set_xlim(-_v(lim), _v(lim))
            ax.set_ylim(-_v(lim), _v(lim))
        _annotate(ax, f)

    return PlotBlock(
        draw,
        name="constellation",
        in_type=Sig[complex, "N"],
        title=title,
        figsize=kw.pop("figsize", (4, 4)),
        **kw,
    )


def custom(
    draw: Callable[[Any, Any, Frame], None],
    name: str = "plot",
    in_type: Any = None,
    **kw: Any,
) -> PlotBlock:
    """Make a plot block from ``draw(ax, data, frame)``."""
    return PlotBlock(draw, name=name, in_type=in_type, **kw)


def animate(
    frames: list[Frame] | list[Any],
    draw: Callable[[Any, Any, Frame], None] | None = None,
    interval: int = 100,
    figsize: tuple[float, float] = (7, 3.2),
) -> Any:
    """Animate any sequence of frames / arrays after the fact (default: line plot)."""
    frames = [f if isinstance(f, Frame) else Frame(f) for f in frames]
    p = line(figsize=figsize) if draw is None else custom(draw, figsize=figsize)
    p.frames = frames
    p.redraw(frames[0])
    return p.animate(interval=interval)


def history(
    key: Any = None,
    title: str | None = None,
    window: int = 200,
    ylabel: str = "",
    ylim: Any = None,
    **kw: Any,
) -> PlotBlock:
    """A scalar over frames: the frame's data (a number), ``key="snr_db"`` an annotation, or
    ``key=callable(frame)``; the last ``window`` values are drawn against time (``t0``) when
    the frames carry it, else against frame index."""
    ts: list[float] = []
    vs: list[float] = []

    def pick(f: Frame) -> float:
        if callable(key):
            return float(key(f))
        if isinstance(key, str):
            return float(f[key])
        return float(np.asarray(f.data).reshape(-1)[0])

    def draw(ax: Any, x: Any, f: Frame) -> None:
        vs.append(pick(f))
        ts.append(f.t0 if f.t0 is not None else len(ts))
        del vs[:-window], ts[:-window]
        ax.plot(ts, vs, lw=1.2)
        ax.plot(ts[-1:], vs[-1:], "o", ms=4)
        ax.set_xlabel("time [s]" if f.t0 is not None else "frame")
        ax.set_ylabel(ylabel or (key if isinstance(key, str) else ""))
        if _v(ylim) is not None:
            ax.set_ylim(*_v(ylim))
        ax.grid(True, alpha=0.3)

    return PlotBlock(
        draw,
        name=f"history({key if isinstance(key, str) else ''})",
        in_type=None,
        title=title,
        **kw,
    )


class Multi(PlotBlock):
    """Several plot blocks tiled in one figure (``plot.multi([...], cols=2)``): one tap that
    draws all panels from the same frame, for `controls(plots=[m])`, `animate()` and `save()`."""

    def __init__(
        self,
        panels: list,
        cols: int = 2,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
        **kw: Any,
    ) -> None:
        self.panels = list(panels)
        self.cols = max(1, min(cols, len(self.panels)))
        self.rows = -(-len(self.panels) // self.cols)
        fs = figsize or (5.0 * self.cols, 2.8 * self.rows)
        super().__init__(
            self._draw_all, name="multi", in_type=None, title=title, figsize=fs, **kw
        )
        self._axes: list = []

    @property
    def figure(self) -> Any:
        if self._fig is None:
            plt = _matplotlib()
            self._fig, axes = plt.subplots(
                self.rows, self.cols, figsize=self.figsize, squeeze=False
            )
            self._axes = [a for row in axes for a in row]
            for a in self._axes[len(self.panels) :]:
                a.set_axis_off()
            self._ax = self._axes[0]
            if self.title:
                self._fig.suptitle(self.title)
            plt.close(self._fig)
        return self._fig

    def redraw(self, f: Frame) -> None:
        with self.lock:
            fig = self.figure
            for ax, p in zip(self._axes, self.panels):
                ax.clear()
                if p.title:
                    ax.set_title(p.title)
                try:
                    p.draw_fn(ax, f.data, f)
                except Exception as e:  # a panel that does not fit this frame
                    ax.text(
                        0.5,
                        0.5,
                        f"{p.name}: {e}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        transform=ax.transAxes,
                    )
            fig.canvas.draw_idle()

    def _draw_all(
        self, ax: Any, x: Any, f: Frame
    ) -> None:  # pragma: no cover - redraw() overrides
        pass


def multi(panels: list, cols: int = 2, **kw: Any) -> Multi:
    """Tile several plot blocks (made with line/spectrum/history/custom) into one figure."""
    return Multi(panels, cols=cols, **kw)


def _annotate(ax: Any, f: Frame) -> None:
    bits = []
    if "seq" in f.meta:
        bits.append(f"#{f.meta['seq']}")
    if "t0" in f.meta:
        bits.append(f"t={f.meta['t0']:.4f}s")
    if f.warnings:
        bits.append(f"⚠ {len(f.warnings)}")
    if bits:
        ax.text(
            0.99,
            0.95,
            "  ".join(bits),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            alpha=0.7,
        )
