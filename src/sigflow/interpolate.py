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

The blocks take one-dimensional frames of real or complex samples (complex
ones are interpolated as their real and imaginary parts); :func:`interpolator`
also takes channels after the time axis. The output is the input computed in
floating point (an integer or float32 input gives float64, complex64 gives
complex128), and its frames vary in length; the output contract says both.

The input needs a ``rate`` stream annotation. Frame start times (``t0``) are
used when present; a frame without one follows on from the frame before it
(the first starts at 0).

Samples these blocks invent where the input had none are marked with an
``interpolated`` tag over their time span. Tags the input carried keep their
origin.

Each block keeps its history in a generator, so a checkpoint cannot capture it:
it is listed in ``Checkpoint.skipped`` and starts fresh on resume.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import scipy
from scipy import interpolate as si

from .block import Framed, block
from .dims import N, T
from .frame import Frame, Tag, Tags
from .types import Sig, SignalType, Stream, promoted

__all__ = ["KINDS", "fill_gaps", "interpolator", "resample"]

INTERPOLATED = "interpolated"  # the tag on invented samples


def _linear(t: np.ndarray, x: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    return si.make_interp_spline(t, x, k=1, axis=0)


def _cubic(t: np.ndarray, x: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    return si.CubicSpline(t, x, axis=0)  # not-a-knot ends, scipy's default


def _spline(t: np.ndarray, x: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    return si.make_interp_spline(t, x, k=3, axis=0)


def _pchip(t: np.ndarray, x: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    return si.PchipInterpolator(t, x, axis=0)


def _akima(t: np.ndarray, x: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    return si.Akima1DInterpolator(t, x, axis=0)


# Akima1DInterpolator(method="makima") is in scipy 1.13 and later
_HAS_MAKIMA = "method" in inspect.signature(si.Akima1DInterpolator).parameters


def _makima(t: np.ndarray, x: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    return si.Akima1DInterpolator(t, x, axis=0, method="makima")  # type: ignore[call-arg]


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
#: splines, enough to make the difference negligible. ``"makima"`` needs
#: scipy 1.13 or later.
KINDS: dict[str, int] = {
    "linear": 1,
    "pchip": 2,
    "akima": 3,
    "makima": 3,
    "cubic": 16,
    "spline": 16,
}


def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(map(repr, KINDS))}")
    if kind == "makima" and not _HAS_MAKIMA:
        raise ValueError(
            f"kind='makima' needs scipy 1.13 or later; this is {scipy.__version__}"
        )


def interpolator(
    t: Any, x: Any, kind: str = "pchip"
) -> Callable[[np.ndarray], np.ndarray]:
    """The function through samples ``x`` at times ``t`` (time on the first axis
    of ``x``), by method ``kind``: call it with the times to estimate. Complex
    samples are interpolated as their real and imaginary parts. This is what the
    blocks use on the samples they keep; use it on a whole array."""
    _check_kind(kind)
    t, x = np.asarray(t, dtype=float), np.asarray(x)
    make = _MAKERS[kind]
    if kind in _REAL_ONLY and np.iscomplexobj(x):
        re, im = make(t, np.real(x)), make(t, np.imag(x))
        return lambda q: re(q) + 1j * im(q)
    return make(t, x)


def _method(kind: str, context: int | None) -> tuple[Callable[..., Any], int]:
    _check_kind(kind)
    need = KINDS[kind]
    if context is None:
        context = need
    elif context < need and kind not in ("cubic", "spline"):
        raise ValueError(f"kind={kind!r} needs context >= {need}")
    elif context < 1:
        raise ValueError("context must be at least 1")
    return (lambda t, x: interpolator(t, x, kind)), context


def _promoted(x: np.ndarray) -> np.ndarray:
    """``x`` in its dtype promoted to floating point, as the contract says."""
    dt = np.result_type(x.dtype, np.float64)
    return x if x.dtype == dt else x.astype(dt)


def _rate(f: Frame) -> float:
    rate = f.stream.get("rate")
    if not rate:
        raise ValueError("interpolation needs a 'rate' stream annotation")
    return float(rate)


class _History:
    """The samples of a stream still needed, with their times, and its tags as
    time intervals, so an output frame gets those that apply to its window."""

    def __init__(self) -> None:
        self.t = np.empty(0)
        self.x: np.ndarray | None = None
        self.next_t: float | None = None  # where a frame without t0 starts
        self.rate: float | None = None
        # (tag, start, end, whole-frame on its input?)
        self.tags: list[tuple[Tag, float, float, bool]] = []
        self.gaps: list[tuple[float, float]] = []  # where input samples are missing

    def push(self, f: Frame) -> None:
        rate = _rate(f)
        x = np.asarray(f.data)
        if x.ndim == 0:
            raise ValueError("interpolation needs frames with a time axis")
        t0 = f.meta.get("t0")
        t0 = (self.next_t or 0.0) if t0 is None else float(t0)
        t = t0 + np.arange(len(x)) / rate
        self.next_t = float(t0 + len(x) / rate)
        for tag in f.tags:
            whole = tag.span is None
            s0, s1 = (t0, self.next_t) if whole else tag.span  # type: ignore[misc]
            self.tags.append((tag, s0, s1, whole))
        if len(self.t):
            last = self.t[-1]
            if t[0] - last > 1.5 / rate:  # whole samples missing in between
                self.gaps.append((last + 1 / (self.rate or rate), t[0]))
            keep = t > last  # overlapping or repeated times: the first wins
            t, x = t[keep], x[keep]
        self.rate = rate
        if len(t):
            self.t = np.concatenate([self.t, t])
            self.x = x if self.x is None else np.concatenate([self.x, x])

    def forget(self, before: float, context: int) -> None:
        """Drop what no output at or after ``before`` needs: samples earlier than
        the ``context`` just before it, and tags and holes that end before it."""
        cut = max(0, int(np.searchsorted(self.t, before)) - context)
        if cut:
            self.t = self.t[cut:]
            assert self.x is not None
            self.x = self.x[cut:]
        self.tags = [e for e in self.tags if e[2] > before]
        self.gaps = [g for g in self.gaps if g[1] > before]

    def tags_for(self, w0: float, w1: float, half: float) -> Tags:
        """The tags covering ``[w0, w1)``: a whole-frame tag stays whole when its
        frames (adjacent ones with the same value count together) cover the window,
        and becomes a span over the covered part otherwise. Inherited tags keep
        their origin; the ``interpolated`` tags are new."""
        out: list[Tag] = []
        whole: dict[tuple[str, Any], list[tuple[float, float, Tag]]] = {}
        for tag, s0, s1, is_whole in self.tags:
            if is_whole:
                whole.setdefault((tag.name, tag.value), []).append((s0, s1, tag))
            elif tag.span is not None:
                if tag.span[1] <= tag.span[0]:  # a point
                    if w0 <= tag.span[0] < w1:
                        out.append(tag)
                elif s1 > w0 and s0 < w1:
                    out.append(tag)
        for spans in whole.values():
            for a, b, tag in _union(spans, half):
                lo, hi = max(a, w0), min(b, w1)
                if lo >= hi:
                    continue
                cover = lo <= w0 + half and hi >= w1 - half
                out.append(dataclasses.replace(tag, span=None if cover else (lo, hi)))
        for a, b in self.gaps:
            lo, hi = max(a, w0), min(b, w1)
            if lo < hi:
                out.append(Tag(INTERPOLATED, None, (lo, hi)))
        return Tags(out)


def _union(
    spans: list[tuple[float, float, Tag]], slack: float
) -> list[tuple[float, float, Tag]]:
    """Adjacent or overlapping intervals merged; each keeps its first tag."""
    out: list[list[Any]] = []
    for a, b, tag in sorted(spans, key=lambda s: (s[0], s[1])):
        if out and a <= out[-1][1] + slack:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b, tag])
    return [(a, b, tag) for a, b, tag in out]


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


# one-dimensional, any length, promoted to floating point
_OUT = SignalType(promoted(T), (None,)).may_provide(INTERPOLATED)


@block(outputs=_OUT)
def resample(
    xs: Stream[Framed[Sig[T, N]]],
    rate: float = 48_000.0,
    kind: str = "pchip",
    context: int | None = None,
) -> Iterator[Frame]:
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
    stream ends. Only the samples the next output needs are kept, so memory
    stays bounded however far apart the output samples are.

    Frames are not 1:1 with the input, and vary in length. Follow with
    ``dsp.chunk(n)`` to get frames of a fixed length.

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
        y = _promoted(np.asarray(make(h.t, h.x)(tq)))
        half = 0.5 / rate
        out = _frame(
            base, y, float(tq[0]), rate, h.tags_for(tq[0], tq[-1] + 1 / rate, half)
        )
        k += n
        return out

    for f in xs:
        h.push(f)
        last = f
        if origin is None and len(h.t):
            origin = float(h.t[0])
        if origin is not None:
            h.forget(origin + k / rate, ctx)  # bounded, even with nothing to emit
        if len(h.t) <= 2 * ctx:
            continue
        out = emit(float(h.t[-1 - ctx]), f)  # beyond this, later samples still matter
        if out is not None:
            yield out
    if last is not None and origin is not None and len(h.t) >= 2:
        out = emit(float(h.t[-1]), last)
        if out is not None:
            yield out


@dataclasses.dataclass
class _Hole:
    """A hole waiting for enough samples after it to be filled."""

    start: float  # time of the first missing sample
    missing: int
    rate: float
    before: Frame  # the frame before it: annotations and frame length
    left_t: np.ndarray
    left_x: np.ndarray
    held: list[tuple[Frame, np.ndarray, np.ndarray]]  # frames after it, held back

    def right(self, ctx: int) -> tuple[np.ndarray, np.ndarray]:
        t = np.concatenate([s for _, s, _ in self.held])[:ctx]
        x = np.concatenate([x for _, _, x in self.held])[:ctx]
        return t, x


@block(outputs=_OUT)
def fill_gaps(
    xs: Stream[Framed[Sig[T, N]]],
    max_gap: float = math.inf,
    kind: str = "pchip",
    context: int | None = None,
) -> Iterator[Frame]:
    """Fill holes in a stream's time axis with interpolated samples.

    A hole is where a frame starts later than one sample period after the
    previous frame ends: a dropped packet, or a frame that failed upstream and
    was dropped (``on_error="drop"``). Frames pass through unchanged (but for
    the dtype, promoted to floating point), and before the frame after a hole
    come frames of interpolated samples, on the input's sample grid, in frames
    as long as the one before the hole (the last one shorter). Each is tagged
    ``interpolated``, and carries the whole-frame tags the frames on both sides
    share.

    Holes longer than ``max_gap`` seconds are left alone. ``kind`` and
    ``context`` are as for :func:`resample`; the interpolation uses ``context``
    samples on each side of the hole, so after a hole the frames that follow
    are held back until there are that many (or another hole comes, or the
    stream ends). If there are still too few for ``kind``, the hole is left as
    it is and the frame after it gets a warning issue.

    A frame without a ``t0`` annotation follows on from the frame before it. A
    stream that never had times has no holes to find, and passes through.

    Failed frames still pass around this block (stream blocks never see them),
    so a hole a failed frame left is filled only if that frame was dropped.
    """
    make, ctx = _method(kind, context)
    prev_end: float | None = None  # time just after the last sample seen
    before: Frame | None = None  # the frame before a possible hole
    tail_t = np.empty(0)  # the last ``ctx`` samples, with their times
    tail_x: np.ndarray | None = None
    hole: _Hole | None = None

    def fill(h: _Hole) -> Iterator[Frame]:
        right_t, right_x = h.right(ctx)
        gap_t = h.start + np.arange(h.missing) / h.rate
        first = h.held[0][0]
        try:
            fn = make(
                np.concatenate([h.left_t, right_t]), np.concatenate([h.left_x, right_x])
            )
            y = _promoted(np.asarray(fn(gap_t)))
        except (ValueError, np.linalg.LinAlgError) as e:  # too few samples for kind
            note = f"hole of {h.missing} samples left: {kind} could not fill it ({e})"
            first = first.with_issue(note, "warning", "fill_gaps")
            h.held[0] = (first, h.held[0][1], h.held[0][2])
        else:
            step = len(np.asarray(h.before.data)) or h.missing
            # whole-frame tags both sides agree on apply to the hole too
            after = {(t.name, t.value) for t in first.tags.whole()}
            shared = [t for t in h.before.tags.whole() if (t.name, t.value) in after]
            for i in range(0, h.missing, step):
                yield _frame(
                    h.before,
                    y[i : i + step],
                    float(gap_t[i]),
                    h.rate,
                    Tags([*shared, Tag(INTERPOLATED)]),
                )
        for f, _, _ in h.held:
            yield f

    for f in xs:
        x = _promoted(np.asarray(f.data))
        if x is not f.data:
            f = f.evolve(data=x)
        rate = f.stream.get("rate")
        t0 = f.meta.get("t0")
        if not rate or (t0 is None and prev_end is None):  # no time to compare
            if hole is not None:
                yield from fill(hole)
                hole = None
            prev_end, before, tail_t, tail_x = None, None, np.empty(0), None
            yield f
            continue
        t0 = prev_end if t0 is None else float(t0)
        assert t0 is not None
        stamps = t0 + np.arange(len(x)) / rate
        missing = round((t0 - prev_end) * rate) if prev_end is not None else 0
        if hole is not None and missing > 0:  # another hole first: fill with what came
            yield from fill(hole)
            hole = None
        if hole is not None:
            hole.held.append((f, stamps, x))
            if sum(len(s) for _, s, _ in hole.held) >= ctx:
                yield from fill(hole)
                hole = None
        elif (
            missing > 0
            and missing / rate <= max_gap
            and tail_x is not None
            and before is not None
            and prev_end is not None
        ):
            hole = _Hole(
                prev_end, missing, rate, before, tail_t, tail_x, [(f, stamps, x)]
            )
            if len(x) >= ctx:
                yield from fill(hole)
                hole = None
        else:
            yield f
        tail_t = np.concatenate([tail_t, stamps])[-ctx:]
        tail_x = x[-ctx:] if tail_x is None else np.concatenate([tail_x, x])[-ctx:]
        prev_end = float(t0 + len(x) / rate)
        before = f
    if hole is not None:
        yield from fill(hole)
