"""Statistics across frames: moments, outliers, change points and fitted
distributions.

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
- :func:`fit` fits a distribution of ``scipy.stats`` to the samples, exactly
  from running sums for the distributions in :data:`EXACT_FITS`, over a window
  of frames for the others, with an optional goodness-of-fit test;
- :func:`threshold` tags the samples a fitted noise distribution makes
  improbable, for a chosen false-alarm probability.

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
from .dims import N, P
from .frame import Tag
from .graph import Block
from .persist import Stateful
from .types import Sig, may_provide

__all__ = [
    "EXACT_FITS",
    "Moments",
    "change_point",
    "fit",
    "moments",
    "outliers",
    "param_names",
    "threshold",
]


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
    them; a frame without a time axis gets one whole-frame tag instead, and a
    sample-level ``outlier`` annotation marking them. The data passes through
    unchanged, and so does the contract. A frame whose
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
        return _tag_runs(f, "outlier", flagged, score)  # type: ignore[no-any-return]

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


# --------------------------------------------------------------------------- #
# Fitting distributions
# --------------------------------------------------------------------------- #


def _dist(dist: Any) -> Any:
    if isinstance(dist, str):
        d = getattr(_scipy_stats, dist, None)
        known = isinstance(d, _scipy_stats.rv_continuous)
        if not known:  # a name scipy does not have: a bad value, not a bad type
            raise ValueError(f"scipy.stats has no continuous distribution {dist!r}")
        return d
    if not isinstance(dist, _scipy_stats.rv_continuous):
        raise TypeError(
            f"dist must be a continuous distribution of scipy.stats, not {dist!r}"
        )
    return dist


def param_names(dist: Any) -> tuple[str, ...]:
    """The names of a distribution's parameters, in the order :func:`fit` gives
    them and ``scipy.stats`` takes them: its shapes, then ``loc`` and ``scale``.
    ``param_names("gamma")`` is ``("a", "loc", "scale")``."""
    d = _dist(dist)
    shapes = [s.strip() for s in d.shapes.split(",")] if d.shapes else []
    return (*shapes, "loc", "scale")


def _gamma(ms: tuple[Moments, ...]) -> tuple[float, ...] | None:
    """Gamma maximum likelihood with ``loc=0`` from the means of x and log x:
    solve ``log(a) - digamma(a) = log(mean x) - mean(log x)``."""
    from scipy.optimize import brentq
    from scipy.special import digamma

    mx, ml = ms
    s = math.log(mx.mean) - ml.mean
    if not s > 1e-9:  # every sample equal: no finite shape
        return None
    a = float(brentq(lambda k: math.log(k) - digamma(k) - s, 1e-8, 1e12))  # type: ignore[arg-type]
    return (a, 0.0, mx.mean / a)


def _spread(m: Moments) -> float:
    return math.sqrt(m.m2 / m.n)  # maximum likelihood: divide by n


_LOC0 = {"loc": 0.0}
# The distributions whose maximum-likelihood fit depends only on running sums:
# name -> (the fixed parameters this needs, transforms of x whose Moments are
# kept, the parameters from those Moments)
_EXACT: dict[str, tuple[dict[str, float], tuple[Callable[..., Any], ...], Any]] = {
    "norm": ({}, (np.asarray,), lambda m: (m[0].mean, _spread(m[0]))),
    "expon": (_LOC0, (np.asarray,), lambda m: (0.0, m[0].mean)),
    "rayleigh": (_LOC0, (np.square,), lambda m: (0.0, math.sqrt(m[0].mean / 2))),
    "lognorm": (
        _LOC0,
        (np.log,),
        lambda m: (_spread(m[0]), 0.0, math.exp(m[0].mean)),
    ),
    "gamma": (_LOC0, (np.asarray, np.log), _gamma),
}

#: Distributions :func:`fit` fits exactly from running sums, without keeping
#: samples (with ``loc`` fixed at 0, except ``"norm"``).
EXACT_FITS = frozenset(_EXACT)


class _Fitter:
    """What a fit has seen of the stream, and the fit itself."""

    def __init__(
        self,
        dist: Any,
        window: int | None,
        fixed: Mapping[str, float] | None,
        every: int,
        keep_samples: bool,
    ) -> None:
        self.d = _dist(dist)
        self.names = param_names(self.d)
        if fixed is None:  # bounded below: start the support at 0
            fixed = dict(_LOC0) if np.isfinite(self.d.a) else {}
        unknown = set(fixed) - set(self.names)
        if unknown:
            raise ValueError(
                f"{self.d.name} has no parameter {', '.join(sorted(unknown))}; "
                f"its parameters are {', '.join(self.names)}"
            )
        self.fixed = {k: float(v) for k, v in fixed.items()}
        exact = _EXACT.get(self.d.name)
        # by name, not the functions: a checkpoint pickles this object
        self.family = (
            self.d.name if exact is not None and exact[0] == self.fixed else None
        )
        if window is not None and window < 1:
            raise ValueError("window must be at least 1 frame")
        if every < 1:
            raise ValueError("every must be at least 1")
        self.keep = keep_samples or self.exact is None
        if window is None and self.keep:
            raise ValueError(
                f"fitting {self.d.name} with fixed={self.fixed} needs the samples: "
                "give window= (frames). Exact over the whole stream: "
                + ", ".join(sorted(EXACT_FITS))
            )
        self.window, self.every = window, every
        # values at or below the support's lower end have no likelihood
        loc, scale = self.fixed.get("loc"), self.fixed.get("scale", 1.0)
        self.lower = (
            loc + self.d.a * scale if loc is not None and self.d.a > -np.inf else None
        )
        self.total: tuple[Moments, ...] | None = None
        self.recent: deque[tuple[tuple[Moments, ...] | None, np.ndarray | None]] = (
            deque()
        )
        self.frames = 0
        self.refits = 0
        self.refitted = False  # whether the last update refitted
        self.n = 0
        self.params = np.full(len(self.names), np.nan)
        self.error = ""

    @property
    def exact(self) -> Any:
        """How this fit is made from running sums, or None if it needs samples."""
        return _EXACT[self.family] if self.family is not None else None

    def update(self, x: Any) -> int:
        """Take a frame's samples, refit when due; the number of samples ignored
        for lying outside the support."""
        x = np.asarray(x, dtype=float).ravel()
        x = x[~np.isnan(x)]
        dropped = 0
        if self.lower is not None:
            inside = x > self.lower
            dropped = int(len(x) - inside.sum())
            x = x[inside]
        st = (
            tuple(Moments.of(t(x)) for t in self.exact[1])
            if self.exact is not None
            else None
        )
        if self.window is None:
            assert st is not None
            self.total = (
                st
                if self.total is None
                else tuple(a.merge(b) for a, b in zip(self.total, st))
            )
        else:
            self.recent.append((st, x if self.keep else None))
            while len(self.recent) > self.window:
                self.recent.popleft()
        self.frames += 1
        self.refitted = (self.frames - 1) % self.every == 0
        if self.refitted:
            self.refits += 1
            self._refit()
        return dropped

    def samples(self) -> np.ndarray:
        parts = [x for _, x in self.recent if x is not None]
        return np.concatenate(parts) if parts else np.empty(0)

    def _refit(self) -> None:
        self.error = ""
        params: Any = None
        if self.exact is not None:
            if self.window is None:
                st = self.total
            else:
                st = None
                for s, _ in self.recent:
                    assert s is not None
                    st = s if st is None else tuple(a.merge(b) for a, b in zip(st, s))
            self.n = st[0].n if st else 0
            if self.n >= 2:
                params = self.exact[2](st)
        else:
            x = self.samples()
            self.n = len(x)
            if self.n > len(self.names) - len(self.fixed):
                kw = {f"f{k}": v for k, v in self.fixed.items()}  # floc, fa...
                try:
                    params = self.d.fit(x, **kw)
                except Exception as e:  # an optimiser that fails gives no fit
                    self.error = f"{type(e).__name__}: {e}"
        self.params = (
            np.asarray(params, dtype=float)
            if params is not None
            else np.full(len(self.names), np.nan)
        )

    def frozen(self) -> Any:
        """The fitted distribution, or None while there is no fit."""
        if not np.all(np.isfinite(self.params)):
            return None
        return self.d(*self.params)


def _config_key(*config: Any) -> tuple[Any, ...]:
    return tuple(
        tuple(sorted(c.items())) if isinstance(c, Mapping) else c for c in config
    )


def _refine_fit(dist: Any = "norm", **_: Any) -> Any:
    from .types import SignalType

    return SignalType(float, (len(param_names(dist)),)).may_provide("poor_fit")


class _Fit(Stateful):
    """Fit a distribution of ``scipy.stats`` to the stream's samples, by maximum
    likelihood, and give its parameters.

    The output is a vector of the parameters in the order
    ``param_names(dist)`` gives and ``scipy.stats`` takes them (its shapes, then
    ``loc``, then ``scale``), so ``scipy.stats.gamma(*f.data)`` is the fitted
    distribution. The frame's ``dist`` and ``params`` stream annotations name
    them, and its ``fit_samples`` annotation is how many samples the fit used.

    ``fixed`` holds parameters at given values instead of fitting them. By
    default a distribution bounded below, such as ``"rayleigh"`` or
    ``"gamma"``, has ``loc`` fixed at 0: amplitudes and powers start at zero,
    and a free ``loc`` makes small-sample fits erratic and slow. Pass
    ``fixed={}`` to fit everything. Samples outside the support are ignored,
    with a warning issue.

    The distributions in :data:`EXACT_FITS` (``"norm"``, ``"expon"``,
    ``"rayleigh"``, ``"lognorm"``, ``"gamma"``, with the default ``fixed``)
    are fitted from running sums: exactly what ``dist.fit`` on every sample
    would give, without keeping the samples, so ``window=None`` (the default)
    fits the whole stream so far. Any other distribution, or other fixed
    parameters, keeps the samples of the last ``window`` frames and fits them
    with ``dist.fit``; ``every=k`` refits only every ``k`` frames (the frames
    between repeat the last fit), for distributions whose fit is slow.

    ``gof_every=m`` also tests the fit every ``m`` refits with
    :func:`scipy.stats.goodness_of_fit` (Anderson-Darling, ``draws`` Monte
    Carlo samples, seeded so that a replay agrees), which accounts for the
    parameters having been fitted to the same data: a plain KS test against a
    fitted distribution almost never rejects. The p-value is the
    ``fit_pvalue`` annotation, and while it is below ``alpha`` frames are
    tagged ``poor_fit``. The smallest p-value it can give is
    ``1 / (draws + 1)``, so ``alpha`` must be above that. It needs ``window``
    and takes tens of milliseconds per test.

    Until there are enough samples, or if the fit fails (an issue says why),
    the parameters are NaN.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.fitter: _Fitter | None = None
        self.key: tuple[Any, ...] | None = None
        self.tests = 0
        self.pvalue = math.nan

    def __call__(
        self,
        f: Framed[Sig[float, None]],
        dist: Any = "norm",
        window: int | None = None,
        fixed: Mapping[str, float] | None = None,
        every: int = 1,
        gof_every: int = 0,
        alpha: float = 0.05,
        draws: int = 99,
    ) -> Sig[float, P]:
        if gof_every > 0 and not 1 / (draws + 1) < alpha:
            raise ValueError(
                f"with {draws} draws no p-value is below {1 / (draws + 1):g}: "
                f"alpha={alpha:g} can never reject; raise draws"
            )
        key = _config_key(dist, window, fixed, every, gof_every > 0)
        if self.fitter is None or key != self.key:
            self.fitter = _Fitter(dist, window, fixed, every, gof_every > 0)
            self.key = key
            self.tests, self.pvalue = 0, math.nan
        fitter = self.fitter
        dropped = fitter.update(f.data)
        if gof_every > 0 and fitter.refitted and (fitter.refits - 1) % gof_every == 0:
            self.pvalue = self._test(fitter, draws)
        meta: dict[str, Any] = {"fit_samples": fitter.n}
        if gof_every > 0:
            meta["fit_pvalue"] = self.pvalue
        out = _params_frame(f, fitter, meta)
        if gof_every > 0 and self.pvalue < alpha:
            out = out.tag("poor_fit")
        if dropped:
            out = out.with_issue(
                f"{dropped} samples outside the support of {fitter.d.name} ignored",
                "warning",
                "fit",
            )
        if fitter.error:
            out = out.with_issue(f"fit failed: {fitter.error}", "warning", "fit")
        return out  # type: ignore[return-value]

    def _test(self, fitter: _Fitter, draws: int) -> float:
        x = fitter.samples()
        if not np.all(np.isfinite(fitter.params)) or len(x) < 3:
            return math.nan
        self.tests += 1
        try:
            res = _scipy_stats.goodness_of_fit(
                fitter.d,
                x,
                known_params=fitter.fixed or None,
                n_mc_samples=draws,
                rng=np.random.default_rng(self.tests),
            )
        except Exception:
            return math.nan
        return float(res.pvalue)


def _params_frame(f: Any, fitter: _Fitter, meta: Mapping[str, Any]) -> Any:
    """A frame of fitted parameters: ``f``'s annotations, but not those about its
    samples (rate, units, sample-level), and its whole-frame tags."""
    from .frame import Frame

    stream = {k: v for k, v in f.stream.items() if k not in ("rate", "units")}
    stream.update(dist=fitter.d.name, params=fitter.names)
    return Frame(
        fitter.params.copy(),
        {**f.meta, **meta},
        stream,
        {},
        f.issues,
        f.prov,
        tags=f.tags.whole(),
    )


fit = _stateful(_Fit(), name="fit", refine=_refine_fit)


class _Threshold(Stateful):
    """Tag the samples above a threshold set by a fitted noise distribution, for a
    chosen false-alarm probability.

    The block fits ``dist`` to the stream as :func:`fit` does (same ``window``,
    ``fixed`` and ``every``), and tests each frame against the fit of the
    frames *before* it: a sample is a detection when it exceeds the level that
    noise of that distribution exceeds with probability ``pfa``
    (``dist.isf(pfa, *params)``). Unlike a fixed level, the threshold follows
    the noise as it changes, and the false-alarm rate stays where it was set,
    as long as the noise follows ``dist``: pick it to suit the data (Rayleigh
    for the amplitude of complex Gaussian noise, exponential for its power,
    Weibull or lognormal for clutter with heavier tails).

    The first ``learn`` frames only train the fit. With ``exclude=True`` (the
    default) detected samples are left out of the fit, so strong targets do not
    raise the threshold that should find them.

    Each run of samples above the threshold gets a ``detection`` tag over its
    time span, with its largest sample as the value (a frame without a time axis
    gets one whole-frame tag and a sample-level ``detection`` annotation marking
    them), and the frame's ``threshold`` annotation is the level used. The data and its contract pass through unchanged.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.fitter: _Fitter | None = None
        self.key: tuple[Any, ...] | None = None

    def __call__(
        self,
        f: Framed[Sig[float, N]],
        dist: Any = "rayleigh",
        pfa: float = 1e-6,
        window: int | None = None,
        fixed: Mapping[str, float] | None = None,
        every: int = 1,
        learn: int = 5,
        exclude: bool = True,
    ) -> Annotated[Sig[float, N], may_provide("detection")]:
        if not 0 < pfa < 1:
            raise ValueError("pfa must be between 0 and 1")
        key = _config_key(dist, window, fixed, every)
        if self.fitter is None or key != self.key:
            self.fitter = _Fitter(dist, window, fixed, every, keep_samples=False)
            self.key = key
        fitter = self.fitter
        x = np.asarray(f.data, dtype=float)
        out: Any = f
        flagged = np.zeros(x.shape, dtype=bool)
        model = fitter.frozen() if fitter.frames >= learn else None
        if model is not None:
            level = float(model.isf(pfa))
            flagged = x > level
            out = f.annotate(threshold=level)
            if flagged.any():
                out = _tag_runs(out, "detection", flagged, x)
        fitter.update(x[~flagged] if exclude else x)
        return out  # type: ignore[return-value]


def _tag_runs(f: Any, name: str, flagged: np.ndarray, value: np.ndarray) -> Any:
    """Tag each run of flagged samples over its time span, with the largest
    ``value`` in the run. A frame without a time axis has no spans: it gets one
    whole-frame tag, and a sample-level annotation of the same name marking the
    flagged samples."""
    span = f.time
    if span is None or not span.rate:
        return f.tag(**{name: float(np.max(value[flagged]))}).annotate_samples(
            **{name: flagged}
        )
    stamps = span.stamps()
    return f.evolve(
        tags=[
            Tag(
                name,
                float(np.max(value[i:j])),
                (float(stamps[i]), float(stamps[j - 1]) + 1 / span.rate),
            )
            for i, j in _runs(flagged)
        ]
    )


threshold = _stateful(_Threshold(), name="threshold", preserve_type=True)
