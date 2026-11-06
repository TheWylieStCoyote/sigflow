"""Integration across frame boundaries.

A stream arrives in frames, and an integral of it runs across them. Integrating
each frame on its own and adding the totals still loses the interval between
the last sample of one frame and the first of the next: one sample period per
frame, an error that grows for as long as the stream runs. These blocks keep
the last samples of the stream and include that interval, so the result is
what integrating the whole signal at once would give:

- :func:`cumulative` gives the running integral at every sample, with a leak
  and resets to keep drift in check;
- :func:`definite` gives the integral over each frame (or the last few), as
  one value per frame, and those values add up to the integral of the stream.

``method`` is ``"trapezoid"`` (exact against :func:`scipy.integrate.cumulative_trapezoid`
on the whole signal) or ``"simpson"`` (:func:`scipy.integrate.cumulative_simpson`,
more accurate for smooth signals; scipy's value at a sample also uses the
sample after it, which a stream does not have yet at the end of a frame, so the
streamed result differs from the whole-signal one by about the method's own
error).

The input needs a ``rate`` stream annotation. Frame start times (``t0``) are
used when present, so a frame that starts late is a hole in the time axis: by
default the integral bridges it (a straight line from the sample before to the
sample after, with a warning issue), or restarts after it (``gaps="reset"``).

Units: the integral of a quantity is in its units times seconds, and the
output's ``units`` annotation says so, as simply as it can: ``W`` gives ``J``,
``m/s^2`` gives ``m/s``, ``A`` gives ``C``, ``V`` gives ``V*s``.

The blocks keep their state in a generator, so a checkpoint cannot capture it:
they are listed in ``Checkpoint.skipped`` and start from zero on resume.
"""

from __future__ import annotations

import itertools
import math
from collections import deque
from collections.abc import Iterator
from typing import Annotated, Any

import numpy as np
from scipy import integrate as si
from scipy.signal import lfilter

from .block import Framed, block
from .dims import N, T
from .frame import Frame
from .types import Sig, Stream, may_provide

__all__ = ["METHODS", "cumulative", "definite"]

RESET = "integral_reset"  # the tag on a frame where the integral restarts

#: The integration rules, and how many samples each keeps from the previous
#: frame.
METHODS = {"trapezoid": 1, "simpson": 2}


def _units(f: Frame) -> dict[str, Any]:
    """The units of the integral: the input's times seconds, tidied
    (``W`` -> ``J``, ``m/s`` -> ``m``, ``V`` -> ``V*s``)."""
    from .units import product, simplify

    u = f.stream.get("units")
    return {"units": simplify(product(u, "s"))} if u else {}


def _runs(mask: np.ndarray) -> list[tuple[int, int, bool]]:
    """``(start, stop, value)`` of the runs of equal values in a boolean array."""
    edges = np.flatnonzero(np.diff(mask.view(np.int8))) + 1
    bounds = [0, *edges.tolist(), len(mask)]
    return [(a, b, bool(mask[a])) for a, b in itertools.pairwise(bounds)]


class _Integrator:
    """The running integral of a stream, frame by frame."""

    def __init__(
        self,
        method: str,
        leak: float | None,
        reset_on: str | None,
        gaps: str,
        initial: float,
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"method must be one of {', '.join(map(repr, METHODS))}")
        if leak is not None and not leak > 0:
            raise ValueError("leak is a time constant in seconds: give a positive one")
        if gaps not in ("bridge", "reset"):
            raise ValueError("gaps must be 'bridge' or 'reset'")
        self.rule = (
            si.cumulative_trapezoid if method == "trapezoid" else si.cumulative_simpson
        )
        self.keep = METHODS[method]
        self.leak, self.reset_on, self.gaps = leak, reset_on, gaps
        self.initial = initial
        self.ctx_t = np.empty(0)  # the last samples, with their times
        self.ctx_x: np.ndarray | None = None
        self.total: Any = None  # the integral at the last sample
        self.pos = 0  # samples seen, for streams without t0

    def step(self, f: Frame) -> tuple[np.ndarray, list[str], bool]:
        """The integral at each sample of ``f``; warnings; whether it restarted."""
        rate = f.stream.get("rate")
        if not rate:
            raise ValueError("integration needs a 'rate' stream annotation")
        x = np.asarray(f.data)
        if x.ndim == 0:
            raise ValueError("integration needs frames with a time axis")
        if not np.issubdtype(x.dtype, np.inexact):
            x = x.astype(float)
        dx = 1.0 / rate
        t0 = f.t0
        t0 = self.pos * dx if t0 is None else float(t0)
        self.pos += len(x)
        t = t0 + np.arange(len(x)) * dx
        notes: list[str] = []
        restarted = False
        if self.ctx_x is not None:
            width = t[0] - self.ctx_t[-1]
            if width > 1.5 * dx:  # whole samples missing
                if self.gaps == "reset":
                    self._restart()
                    restarted = True
                else:
                    notes.append(f"integral bridged a gap of {width - dx:.6g} s")
            elif width < 0.5 * dx:  # overlapping or repeated times
                notes.append(
                    "frames overlap in time: the integral assumes they follow on"
                )
                t = self.ctx_t[-1] + dx * (1 + np.arange(len(x)))
        rel = self._relative(t, x)  # integral from the previous sample (or 0)
        start = self.total if self.total is not None else self._zero(x, self.initial)
        out = self._accumulate(f, rel, start, dx)
        self.total = out[-1]
        self.ctx_t = np.concatenate([self.ctx_t, t])[-self.keep :]
        self.ctx_x = (
            x[-self.keep :]
            if self.ctx_x is None
            else np.concatenate([self.ctx_x, x])[-self.keep :]
        )
        return out, notes, restarted

    def _restart(self) -> None:
        self.ctx_t, self.ctx_x, self.total = np.empty(0), None, None

    @staticmethod
    def _zero(x: np.ndarray, value: float) -> Any:
        return np.full(x.shape[1:], value, dtype=np.result_type(x, float))

    def _relative(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        """At each sample of the frame, the integral since the sample before the
        frame (0 at the first sample of the stream)."""
        if self.ctx_x is None:
            if len(x) == 1:
                return np.zeros_like(x, dtype=np.result_type(x, float))
            return self.rule(x, x=t, axis=0, initial=0)
        zt = np.concatenate([self.ctx_t, t])
        zx = np.concatenate([self.ctx_x, x])
        c = self.rule(zx, x=zt, axis=0, initial=0)
        b = len(self.ctx_x) - 1  # the last sample before this frame
        return c[b + 1 :] - c[b]

    def _accumulate(
        self, f: Frame, rel: np.ndarray, start: Any, dx: float
    ) -> np.ndarray:
        """Running values from ``start``: plain, leaky, and zero where the
        ``reset_on`` tag applies."""
        inc = np.diff(rel, axis=0, prepend=np.zeros((1, *rel.shape[1:])))
        if self.ctx_x is None:  # no interval before the first sample
            inc[0] = 0
        held = (
            f.mask(self.reset_on)
            if self.reset_on is not None and self.reset_on in f.tags
            else np.zeros(len(rel), dtype=bool)
        )
        out = np.empty_like(rel)
        level = start
        for a, b, zero in _runs(held):
            if zero:
                out[a:b] = 0
                level = self._zero(rel, 0.0)
                continue
            out[a:b] = self._run(level, inc[a:b], dx)
            level = out[b - 1]
        return out

    def _run(self, level: Any, inc: np.ndarray, dx: float) -> np.ndarray:
        if self.leak is None:
            return level + np.cumsum(inc, axis=0)
        a = math.exp(-dx / self.leak)  # decay per sample: y' = x - y / leak
        zi = np.expand_dims(a * np.asarray(level), 0)
        y, _ = lfilter([1.0], [1.0, -a], inc, axis=0, zi=zi)
        return y


@block(ratio="1:1")
def cumulative(
    xs: Stream[Framed[Sig[T, N]]],
    method: str = "trapezoid",
    leak: float | None = None,
    reset_on: str | None = None,
    gaps: str = "bridge",
    initial: float = 0.0,
) -> Iterator[Annotated[Sig[T, N], may_provide(RESET)]]:
    """The running integral of a stream over time, at every sample.

    The output has the input's shape and sample grid: sample ``i`` is the
    integral from the start of the stream (where it is ``initial``) to sample
    ``i``, across frame boundaries. ``method`` is ``"trapezoid"`` or
    ``"simpson"``; the time step comes from the ``rate`` annotation and holes
    from ``t0`` (see :mod:`sigflow.integrate`).

    An integral of a real sensor drifts: noise makes it wander and a constant
    bias makes it grow without bound (a still accelerometer with a 0.001 m/s²
    bias drifts 1.8 m in a minute, integrated twice). Two remedies:

    - ``leak=tau`` makes it a leaky integrator, ``y' = x - y / tau``: it
      forgets what happened more than a few ``tau`` seconds ago, so it follows
      changes faster than ``tau`` and does not drift;
    - ``reset_on="still"`` holds the integral at zero on the samples a tag of
      that name covers (a whole frame, or a span of it), and integrates afresh
      after them: zero-velocity updates, when another block knows the sensor is
      at rest.

    ``gaps="reset"`` restarts the integral after a hole in the time axis, and
    tags that frame ``integral_reset``.
    """
    integ = _Integrator(method, leak, reset_on, gaps, initial)
    for f in xs:
        out, notes, restarted = integ.step(f)
        g = f.evolve(data=out, stream=_units(f))
        for note in notes:
            g = g.with_issue(note, "warning", "cumulative")
        yield g.tag(RESET) if restarted else g


def _scalar_frame(f: Frame, value: Any, notes: list[str], restarted: bool) -> Frame:
    """One value for the frame: its annotations, without those about samples."""
    stream = {k: v for k, v in f.stream.items() if k not in ("rate", "units")}
    stream.update(_units(f))
    out = Frame(value, f.meta, stream, {}, f.issues, f.prov, tags=f.tags.whole())
    for note in notes:
        out = out.with_issue(note, "warning", "definite")
    return out.tag(RESET) if restarted else out


@block(ratio="1:1")
def definite(
    xs: Stream[Framed[Sig[T, N]]],
    method: str = "trapezoid",
    window: int | None = None,
    gaps: str = "bridge",
) -> Iterator[Annotated[Sig[T, tuple[()]], may_provide(RESET)]]:
    """The integral over each frame, as one value per frame.

    A frame's value covers its samples *and* the interval before its first
    sample (from the last sample of the frame before), so the values add up to
    the integral of the whole stream: the charge of a current, the energy of a
    power, the area under a pulse. With ``window=k`` each value is the integral
    over the last ``k`` frames. ``method`` and ``gaps`` are as for
    :func:`cumulative`; a frame where the integral restarts covers its own
    samples only.
    """
    if window is not None and window < 1:
        raise ValueError("window must be at least 1 frame")
    integ = _Integrator(method, None, None, gaps, 0.0)
    recent: deque[Any] = deque()
    for f in xs:
        before = integ.total
        out, notes, restarted = integ.step(f)
        value = out[-1] - (before if before is not None and not restarted else 0)
        if restarted:
            recent.clear()
        recent.append(value)
        if window is not None:
            while len(recent) > window:
                recent.popleft()
        total = sum(recent) if window is not None else value
        if window is None:
            recent.clear()
        yield _scalar_frame(f, total, notes, restarted)
