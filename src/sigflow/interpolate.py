"""Interpolation blocks that work across frame boundaries.

A stream arrives in frames, but interpolation needs samples on both sides of the
point it estimates. Interpolating each frame on its own leaves seams where a
frame ends; these blocks keep the last few samples of the stream instead, so
the output is what interpolating the whole signal would give:

- **local** methods (``"linear"``, ``"pchip"``, ``"akima"``, ``"makima"``)
  depend only on nearby samples, and match the whole-signal result exactly;
- **global** splines (``"cubic"``, ``"spline"``) depend a little on every sample,
  and match it to an accuracy set by ``context``: the error falls about tenfold
  for every two more samples kept (``1e-11`` with the default 16).

Every method takes real or complex samples (complex ones are interpolated as
their real and imaginary parts) and a time axis first, with any channels after
it. The input needs a ``rate`` stream annotation; frame start times (``t0``)
are used when present and counted from the samples otherwise.

Samples these blocks invent where the input had none are marked with an
``interpolated`` tag over their time span.

Each block keeps its history in a generator, so a checkpoint cannot capture it:
it is listed in ``Checkpoint.skipped`` and starts fresh on resume.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from typing import Annotated, Any

import numpy as np
from scipy import interpolate as si

from .block import Framed, block
from .dims import M, N, T
from .frame import Frame, Tag, Tags
from .types import Sig, Stream, may_provide

__all__ = ["KINDS", "fill_gaps", "interpolator", "resample"]

INTERPOLATED = "interpolated"  # the tag on invented samples


def _linear(t: np.ndarray, x: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    return si.make_interp_spline(t, x, k=1, axis=0)


def _cubic(t: np.ndarray, x: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    return si.CubicSpline(t, x, axis=0)


def _spline(t: np.ndarray, x: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    return si.make_interp_spline(t, x, k=3, axis=0)


def _pchip(t: np.ndarray, x: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    return si.PchipInterpolator(t, x, axis=0)


def _akima(t: np.ndarray, x: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    return si.Akima1DInterpolator(t, x, axis=0)


def _makima(t: np.ndarray, x: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    return si.Akima1DInterpolator(t, x, axis=0, method="makima")


_MAKERS: dict[str, Callable[[np.ndarray, np.ndarray], Any]] = {
    "linear": _linear,
    "pchip": _pchip,
    "akima": _akima,
    "makima": _makima,
    "cubic": _cubic,
    "spline": _spline,
}
_REAL_ONLY = {"pchip", "akima", "makima"}  # scipy refuses complex samples

#: The interpolation methods, each with the samples of context it keeps on each
#: side by default: for the local methods, what matching the whole-signal
#: result exactly needs (and the least ``context=`` accepts); for the global
#: splines, enough to make the difference negligible.
KINDS: dict[str, int] = {
    "linear": 1,
    "pchip": 2,
    "akima": 3,
    "makima": 3,
    "cubic": 16,
    "spline": 16,
}


def interpolator(
    t: Any, x: Any, kind: str = "pchip"
) -> Callable[[np.ndarray], np.ndarray]:
    """The function through samples ``x`` at times ``t`` (time on the first axis
    of ``x``), by method ``kind``: call it with the times to estimate. Complex
    samples are interpolated as their real and imaginary parts. This is what the
    blocks use on the samples they keep; use it on a whole array."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(map(repr, KINDS))}")
    t, x = np.asarray(t, dtype=float), np.asarray(x)
    make = _MAKERS[kind]
    if kind in _REAL_ONLY and np.iscomplexobj(x):
        re, im = make(t, np.real(x)), make(t, np.imag(x))
        return lambda q: re(q) + 1j * im(q)
    return make(t, x)


def _method(kind: str, context: int | None) -> tuple[Callable[..., Any], int]:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(map(repr, KINDS))}")
    need = KINDS[kind]
    if context is None:
        context = need
    elif context < need and kind not in ("cubic", "spline"):
        raise ValueError(f"kind={kind!r} needs context >= {need}")
    elif context < 1:
        raise ValueError("context must be at least 1")
    return (lambda t, x: interpolator(t, x, kind)), context


class _History:
    """The samples of a stream not yet consumed, with their times, and its tags as
    time intervals, so an output frame gets those that apply to its window."""

    def __init__(self) -> None:
        self.t = np.empty(0)
        self.x: np.ndarray | None = None
        self.pos = 0  # samples seen, for streams without t0
        self.rate: float | None = None
        # (tag, start, end, whole-frame on its input?)
        self.tags: list[tuple[Tag, float, float, bool]] = []
        self.gaps: list[tuple[float, float]] = []  # where input samples are missing

    def push(self, f: Frame) -> tuple[np.ndarray, np.ndarray]:
        rate = f.stream.get("rate")
        if not rate:
            raise ValueError("interpolation needs a 'rate' stream annotation")
        x = np.asarray(f.data)
        if x.ndim == 0:
            raise ValueError("interpolation needs frames with a time axis")
        t0 = f.meta.get("t0")
        t0 = self.pos / rate if t0 is None else float(t0)
        self.pos += len(x)
        t = t0 + np.arange(len(x)) / rate
        if len(self.t):
            last = self.t[-1]
            if t[0] - last > 1.5 / rate:  # whole samples missing in between
                self.gaps.append((last + 1 / self.rate if self.rate else last, t[0]))
            keep = t > last  # overlapping or repeated times: the first wins
            t, x = t[keep], x[keep]
        self.rate = rate
        for tag in f.tags:
            whole = tag.span is None
            s0, s1 = (t0, t0 + len(np.asarray(f.data)) / rate) if whole else tag.span  # type: ignore[misc]
            self.tags.append((tag, s0, s1, whole))
        if len(t):
            self.t = np.concatenate([self.t, t])
            self.x = x if self.x is None else np.concatenate([self.x, x])
        return t, x

    def forget(self, before: float, context: int) -> None:
        """Drop samples before ``before`` except the ``context`` just before it."""
        cut = max(0, int(np.searchsorted(self.t, before)) - context)
        if cut:
            self.t = self.t[cut:]
            assert self.x is not None
            self.x = self.x[cut:]
        horizon = self.t[0] if len(self.t) else before
        self.tags = [e for e in self.tags if e[2] > horizon]
        self.gaps = [g for g in self.gaps if g[1] > horizon]

    def tags_for(self, w0: float, w1: float, half: float) -> Tags:
        """The tags covering ``[w0, w1)``: a whole-frame tag stays whole when its
        frames (adjacent ones with the same value count together) cover the window,
        and becomes a span over the covered part otherwise."""
        out: list[Tag] = []
        whole: dict[tuple[str, Any], list[tuple[float, float]]] = {}
        for tag, s0, s1, is_whole in self.tags:
            if is_whole:
                whole.setdefault((tag.name, tag.value), []).append((s0, s1))
            elif tag.span is not None:
                if tag.span[1] <= tag.span[0]:  # a point
                    if w0 <= tag.span[0] < w1:
                        out.append(tag)
                elif s1 > w0 and s0 < w1:
                    out.append(tag)
        for (name, value), spans in whole.items():
            for a, b in _union(spans, half):
                lo, hi = max(a, w0), min(b, w1)
                if lo >= hi:
                    continue
                cover = lo <= w0 + half and hi >= w1 - half
                out.append(Tag(name, value, None if cover else (lo, hi)))
        for a, b in self.gaps:
            lo, hi = max(a, w0), min(b, w1)
            if lo < hi:
                out.append(Tag(INTERPOLATED, None, (lo, hi)))
        return Tags(out)


def _union(spans: list[tuple[float, float]], slack: float) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + slack:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _frame(base: Frame, data: np.ndarray, t0: float, rate: float, tags: Tags) -> Frame:
    """An output frame: ``base``'s frame annotations, a new time axis and tags."""
    return Frame(
        data,
        {**base.meta, "t0": t0},
        {**base.stream, "rate": rate},
        {},  # sample-level annotations do not survive a new sample grid
        base.issues,
        base.prov,
        tags=tags,
    )


@block
def resample(
    xs: Stream[Framed[Sig[T, N]]],
    rate: float = 48_000.0,
    kind: str = "pchip",
    context: int | None = None,
) -> Iterator[Annotated[Sig[T, M], may_provide(INTERPOLATED)]]:
    """Resample a stream to ``rate`` Hz by interpolation, across frame boundaries.

    Any ratio of rates works, including irrational ones and a jittery input
    whose frames start at uneven times (their ``t0``). Output samples fall on an
    exact grid, ``first input time + k / rate``, and each output frame starts at
    the first grid point after the previous one.

    ``kind`` is one of :data:`KINDS`. ``context`` is how many input samples
    are kept on each side of the output being computed; each kind has a
    default that makes the result match interpolating the whole signal (a
    global spline accepts less, trading accuracy for latency). Output
    lags the input by ``context`` samples, and the rest is emitted when the
    stream ends.

    Frames are not 1:1 with the input. Follow with ``dsp.chunk(n)`` to get
    frames of a fixed length.

    Where input samples are missing (a frame starting more than one sample
    period after the previous one ends), the gap is interpolated across and
    the samples in it are tagged ``interpolated``. Interpolation does not
    filter: resampling down without a low-pass filter first aliases.
    """
    make, ctx = _method(kind, context)
    if rate <= 0:
        raise ValueError("rate must be positive")
    h = _History()
    origin: float | None = None
    k = 0  # index of the next output sample on the grid origin + k / rate
    last: Frame | None = None

    def emit(upto: float, base: Frame) -> Frame | None:
        nonlocal k
        assert origin is not None and h.x is not None
        n = math.floor((upto - origin) * rate + 1e-9) - k + 1
        if n <= 0:
            return None
        tq = origin + (k + np.arange(n)) / rate
        y = make(h.t, h.x)(tq)
        half = 0.5 / rate
        out = _frame(
            base, y, float(tq[0]), rate, h.tags_for(tq[0], tq[-1] + 1 / rate, half)
        )
        k += n
        h.forget(origin + k / rate, ctx)
        return out

    for f in xs:
        h.push(f)
        last = f
        if origin is None and len(h.t):
            origin = float(h.t[0])
        if len(h.t) <= 2 * ctx:
            continue
        out = emit(float(h.t[-1 - ctx]), f)  # beyond this, later samples still matter
        if out is not None:
            yield out
    if last is not None and origin is not None and len(h.t) >= 2:
        out = emit(float(h.t[-1]), last)
        if out is not None:
            yield out


@block
def fill_gaps(
    xs: Stream[Framed[Sig[T, N]]],
    max_gap: float = math.inf,
    kind: str = "pchip",
    context: int | None = None,
) -> Iterator[Annotated[Sig[T, M], may_provide(INTERPOLATED)]]:
    """Fill holes in a stream's time axis with interpolated samples.

    A hole is where a frame starts later than one sample period after the
    previous frame ends: a dropped packet, or a frame that failed upstream and
    was dropped (``on_error="drop"``). Frames pass through unchanged, and before
    the frame after a hole come frames of interpolated samples, on the input's
    sample grid, in frames as long as the one before the hole (the last one
    shorter). Each is tagged ``interpolated``.

    Holes longer than ``max_gap`` seconds are left alone. ``kind`` and
    ``context`` are as for :func:`resample`; the interpolation uses ``context``
    samples on each side of the hole. Frames without a ``t0`` annotation have no
    time to compare and pass through as they are.

    Failed frames still pass around this block (stream blocks never see them),
    so a hole a failed frame left is filled only if that frame was dropped.
    """
    make, ctx = _method(kind, context)
    before: Frame | None = None  # the frame just before a possible hole
    tail_t = np.empty(0)
    tail_x: np.ndarray | None = None
    for f in xs:
        span = f.time
        if span is None or not span.rate:
            yield f
            continue
        x = np.asarray(f.data)
        rate = span.rate
        if before is not None and tail_x is not None:
            prev = before.time
            assert prev is not None
            missing = round((span.t0 - prev.t1) * rate)
            if missing > 0 and missing / rate <= max_gap:
                head_t = span.stamps()[:ctx]
                fn = make(
                    np.concatenate([tail_t, head_t]),
                    np.concatenate([tail_x, x[:ctx]]),
                )
                gap_t = prev.t1 + np.arange(missing) / rate
                y = fn(gap_t)
                step = len(np.asarray(before.data)) or missing
                # whole-frame tags both sides agree on apply to the hole too
                after = {(t.name, t.value) for t in f.tags.whole()}
                shared = [
                    Tag(t.name, t.value)
                    for t in before.tags.whole()
                    if (t.name, t.value) in after
                ]
                for i in range(0, missing, step):
                    yield _frame(
                        before,
                        y[i : i + step],
                        float(gap_t[i]),
                        rate,
                        Tags([*shared, Tag(INTERPOLATED)]),
                    )
        stamps = span.stamps()
        tail_t = (
            stamps[-ctx:]
            if tail_x is None or len(x) >= ctx
            else np.concatenate([tail_t, stamps])[-ctx:]
        )
        tail_x = (
            x[-ctx:]
            if tail_x is None or len(x) >= ctx
            else np.concatenate([tail_x, x])[-ctx:]
        )
        before = f
        yield f
