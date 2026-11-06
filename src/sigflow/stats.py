"""Statistics across frames: running moments, outliers and change points.

A frame is a small sample. These blocks keep what they need of the stream's
history, so a statistic covers every sample seen (or the last few frames) and
a test has enough data to tell a change from noise.

- :func:`moments` gives count, mean, variance, skewness and kurtosis over the
  whole stream or a window of frames, matching :func:`scipy.stats.describe` on
  the same samples;
- :func:`outliers` tags the samples far from the typical value, measured
  robustly (median and median absolute deviation);
- :func:`change_point` tags the frame where a stream's mean or spread changes,
  with a two-sided CUSUM test.

The stateful blocks keep their state in an object of their own for each place
they are used in a graph, so two graphs, or one graph using a block twice, do
not share it. Every run starts afresh, unless it resumes from a checkpoint,
which captures their state. Name a second use of the same block
(``stats.moments.node().named("moments_b")``) so a checkpoint keeps both.
"""

from __future__ import annotations

import copy
import math
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any

import numpy as np
from scipy import stats as _scipy_stats

from .block import BlockDef, Framed
from .dims import N
from .frame import Tag
from .graph import Block
from .persist import Stateful
from .types import Sig, may_provide

__all__ = ["Moments", "change_point", "moments", "outliers"]


# --------------------------------------------------------------------------- #
# A mergeable summary
# --------------------------------------------------------------------------- #


@dataclass
class Moments:
    """Count, mean and the sums of the 2nd to 4th powers of deviations from it.

    Each frame's summary comes from :func:`scipy.stats.moment`. Two summaries
    combine exactly with :meth:`merge` (Chan's and Pébay's formulas), which
    scipy does not provide, so a summary of a long stream is built frame by
    frame, or from the summaries of its parts, without keeping the samples and
    without the cancellation the textbook sum-of-squares formula suffers when
    the mean is large."""

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    m3: float = 0.0
    m4: float = 0.0

    @classmethod
    def of(cls, x: Any) -> Moments:
        """The summary of the samples in ``x`` (flattened; NaNs are skipped): the
        central moments come from :func:`scipy.stats.moment`."""
        x = np.asarray(x, dtype=float).ravel()
        x = x[~np.isnan(x)]
        n = len(x)
        if not n:
            return cls()
        if x.min() == x.max():  # constant (silence, DC): scipy warns needlessly
            return cls(n, float(x[0]))
        mu = float(np.mean(x))
        c2, c3, c4 = _scipy_stats.moment(  # an order list is accepted at run time
            x,
            order=[2, 3, 4],  # pyright: ignore[reportArgumentType]
            center=mu,
        )
        return cls(n, mu, float(c2) * n, float(c3) * n, float(c4) * n)

    def merge(self, other: Moments) -> Moments:
        """The summary of both sets of samples together."""
        a, b = self, other
        if not a.n:
            return b
        if not b.n:
            return a
        n = a.n + b.n
        d = b.mean - a.mean
        dn = d / n
        ab = a.n * b.n
        m2 = a.m2 + b.m2 + d * dn * ab
        m3 = (
            a.m3
            + b.m3
            + d * dn * dn * ab * (a.n - b.n)
            + 3 * dn * (a.n * b.m2 - b.n * a.m2)
        )
        m4 = (
            a.m4
            + b.m4
            + d * dn**3 * ab * (a.n * a.n - ab + b.n * b.n)
            + 6 * dn * dn * (a.n * a.n * b.m2 + b.n * b.n * a.m2)
            + 4 * dn * (a.n * b.m3 - b.n * a.m3)
        )
        return Moments(n, a.mean + dn * b.n, m2, m3, m4)

    def variance(self, ddof: int = 1) -> float:
        """Variance; ``ddof=1`` (the default) is the unbiased estimate, as
        :func:`scipy.stats.describe` reports."""
        return self.m2 / (self.n - ddof) if self.n > ddof else math.nan

    def skewness(self) -> float:
        """Sample skewness (biased, as ``scipy.stats.skew`` by default)."""
        if self.n < 2 or self.m2 == 0:
            return math.nan
        return math.sqrt(self.n) * self.m3 / self.m2**1.5

    def kurtosis(self) -> float:
        """Excess kurtosis, Fisher's definition (0 for a normal distribution)."""
        if self.n < 2 or self.m2 == 0:
            return math.nan
        return self.n * self.m4 / (self.m2 * self.m2) - 3.0


# --------------------------------------------------------------------------- #
# Blocks with state of their own per node
# --------------------------------------------------------------------------- #


class _OwnState(BlockDef):
    """A block definition whose function is a stateful object: each node made
    from it gets a copy of that object, so uses of the block do not share state."""

    def node(self, **config: Any) -> Block:
        from .xarray import adapt

        own = object.__new__(type(self))
        own.__dict__.update(self.__dict__)
        own.fn = copy.deepcopy(self.fn)
        own.adapted = adapt(own)
        return Block(own, config)

    def __reduce__(self) -> Any:
        raise TypeError(
            f"{self.name} keeps state across frames, which worker processes would "
            "split between them: run it in the graph itself, not in parallel()"
        )


def _stateful(obj: Callable[..., Any], **options: Any) -> BlockDef:
    return _OwnState(obj, **options)


# --------------------------------------------------------------------------- #
# moments
# --------------------------------------------------------------------------- #

_SCALAR = Sig[float, tuple[()]]
_COUNT = Sig[int, tuple[()]]


class _Moments(Stateful):
    """Count, mean, variance, skewness and excess kurtosis of the samples seen.

    Over every sample of the stream so far, or with ``window=k`` over the
    samples of the last ``k`` frames. Each frame's samples count individually
    (its array is flattened, NaNs skipped); a stream of one value per frame, such
    as ``dsp.rms``, therefore gives statistics of those values. The results are
    those of :func:`scipy.stats.describe` on the same samples: unbiased
    variance, biased skewness, Fisher's kurtosis. Ports: ``count``, ``mean``,
    ``var``, ``skew``, ``kurt``; select one with ``moments["var"]``. Values that
    need more samples than there are (a variance of one sample) are NaN.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total = Moments()
        self.recent: deque[Moments] = deque()

    def __call__(
        self, x: Sig[float, None], window: int | None = None
    ) -> tuple[int, float, float, float, float]:
        part = Moments.of(x)
        if window is None:
            self.total = self.total.merge(part)
            m = self.total
        else:
            if window < 1:
                raise ValueError("window must be at least 1 frame")
            self.recent.append(part)
            while len(self.recent) > window:
                self.recent.popleft()
            m = Moments()
            for p in self.recent:
                m = m.merge(p)
        return (m.n, m.mean, m.variance(), m.skewness(), m.kurtosis())

    def state(self) -> Any:
        return {"total": self.total, "recent": list(self.recent)}

    def restore(self, state: Mapping[str, Any]) -> None:
        self.total = state["total"]
        self.recent = deque(state["recent"])


moments = _stateful(
    _Moments(),
    name="moments",
    outputs={
        "count": _COUNT,
        "mean": _SCALAR,
        "var": _SCALAR,
        "skew": _SCALAR,
        "kurt": _SCALAR,
    },
)


# --------------------------------------------------------------------------- #
# outliers
# --------------------------------------------------------------------------- #

_MAD_SCALE = 1.4826  # MAD -> standard deviation, for normally distributed data


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[start, stop)`` index ranges of the True runs in ``mask``."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.view(np.int8), [0]])))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


class _Outliers(Stateful):
    """Tag the samples far from the typical value as ``outlier``.

    A sample's score is its distance from the median in units of the scaled
    median absolute deviation, a spread estimate the outliers themselves barely
    move (unlike the standard deviation): ``|x - median| / (1.4826 * MAD)``,
    about a z-score for normal data. Samples scoring above ``threshold`` are
    outliers; 3.5 is the usual choice.

    The median and MAD come from the frame itself, or with ``window=k`` from the
    last ``k`` frames, which suits short frames and frames that are mostly
    outliers. Each run of outlying samples gets an ``outlier`` tag over its time
    span, with its largest score as the value, so ``f.mask("outlier")`` marks
    them; a frame without a time axis gets one whole-frame tag instead. The
    data passes through unchanged, and so does the contract. A frame whose
    samples are almost all equal (MAD of 0) flags every sample that differs.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.recent: deque[np.ndarray] = deque()

    def __call__(
        self,
        f: Framed[Sig[float, N]],
        threshold: float = 3.5,
        window: int | None = None,
    ) -> Annotated[Sig[float, N], may_provide("outlier")]:
        x = np.asarray(f.data, dtype=float)
        if window is None:
            ref = x
        else:
            if window < 1:
                raise ValueError("window must be at least 1 frame")
            self.recent.append(x)
            while len(self.recent) > window:
                self.recent.popleft()
            ref = np.concatenate(list(self.recent))
        med = float(np.nanmedian(ref))
        mad = float(np.nanmedian(np.abs(ref - med))) * _MAD_SCALE
        with np.errstate(divide="ignore", invalid="ignore"):
            score = (
                np.abs(x - med) / mad if mad > 0 else np.where(x == med, 0.0, np.inf)
            )
        flagged = score > threshold
        if not flagged.any():
            return f  # type: ignore[return-value]
        span = f.time
        if span is None or not span.rate:
            return f.tag(outlier=float(np.max(score[flagged])))  # type: ignore[return-value]
        stamps = span.stamps()
        tags = [
            Tag(
                "outlier",
                float(np.max(score[i:j])),
                (float(stamps[i]), float(stamps[j - 1]) + 1 / span.rate),
            )
            for i, j in _runs(flagged)
        ]
        return f.evolve(tags=tags)  # type: ignore[return-value]

    def state(self) -> Any:
        return {"recent": list(self.recent)}

    def restore(self, state: Mapping[str, Any]) -> None:
        self.recent = deque(state["recent"])


outliers = _stateful(_Outliers(), name="outliers", preserve_type=True)


# --------------------------------------------------------------------------- #
# change_point
# --------------------------------------------------------------------------- #


class _ChangePoint(Stateful):
    """Tag the frame where the stream's mean or spread changes, by a two-sided
    CUSUM test.

    The first ``reference`` frames set what is normal. Each later frame gives a
    score that is about standard normal while nothing changes: its mean against
    the reference mean (``on="mean"``), or the log of its variance against the
    reference variance (``on="var"``). Two sums collect the scores beyond
    ``drift`` upward and downward, falling back to zero while the stream is
    normal; when one passes ``threshold`` the frame is tagged ``changed``, with
    the value ``"up"`` or ``"down"``, and its annotation ``change_onset`` is
    the start time (or index) of the frame where that sum last left zero, the
    likely start of the change. The test then starts again, learning the new
    normal from the next ``reference`` frames.

    ``drift`` is half the smallest shift worth detecting, in units of the
    score's standard deviation; ``threshold`` trades false alarms for delay:
    larger means fewer false alarms and later detection. For frames of 64
    normal samples and a 1.6-fold step in standard deviation, ``on="var"``
    with the defaults detected every change, typically within 2 frames, with a
    false alarm in about 1 run in 25 of 50 frames. The scores assume
    independent samples; correlated noise needs a larger ``threshold``.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.seen = 0
        self._relearn()

    def _relearn(self) -> None:
        self.ref = Moments()
        self.ref_frames = 0
        self.up = self.down = 0.0
        self.up_start: Any = None
        self.down_start: Any = None

    def __call__(
        self,
        f: Framed[Sig[float, None]],
        on: str = "var",
        reference: int = 10,
        drift: float = 0.25,
        threshold: float = 12.0,
    ) -> Annotated[Sig[float, None], may_provide("changed")]:
        if on not in ("mean", "var"):
            raise ValueError("on must be 'mean' or 'var'")
        part = Moments.of(f.data)
        here = f.t0 if f.t0 is not None else self.seen
        self.seen += 1
        if self.ref_frames < reference:
            self.ref = self.ref.merge(part)
            self.ref_frames += 1
            return f  # type: ignore[return-value]
        z = self._score(part, on)
        if math.isnan(z):
            return f  # type: ignore[return-value]
        if self.up == 0.0:
            self.up_start = here
        if self.down == 0.0:
            self.down_start = here
        self.up = max(0.0, self.up + z - drift)
        self.down = max(0.0, self.down - z - drift)
        if self.up <= threshold and self.down <= threshold:
            return f  # type: ignore[return-value]
        direction, onset = (
            ("up", self.up_start) if self.up > threshold else ("down", self.down_start)
        )
        self._relearn()
        return f.tag(changed=direction).annotate(change_onset=onset)  # type: ignore[return-value]

    def _score(self, part: Moments, on: str) -> float:
        n = part.n
        if on == "mean":
            sd = math.sqrt(self.ref.variance())
            if n < 1 or not sd > 0:
                return math.nan
            return (part.mean - self.ref.mean) / (sd / math.sqrt(n))
        if n < 3:
            return math.nan
        v, v0 = part.variance(), self.ref.variance()
        if not (v > 0 and v0 > 0):
            return math.nan
        return math.log(v / v0) / math.sqrt(2.0 / (n - 1))


change_point = _stateful(_ChangePoint(), name="change_point", preserve_type=True)
