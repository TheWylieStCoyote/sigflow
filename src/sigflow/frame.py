"""The :class:`Frame` — the immutable unit of data that flows through a graph.

A frame bundles four things:

* ``data``        – the payload (an array, a scalar, a tuple, any Python object)
* annotations     – at three levels:
    - ``stream``  – facts about the whole stream (rate, units, source name …)
    - ``meta``    – facts about this frame (timestamp, sequence number, SNR …)
    - ``samples`` – per-sample arrays aligned with ``data``'s leading axis
                    (e.g. a boolean "clipped" mask, a timestamp vector)
* ``issues``      – errors and warnings that have been attached along the way
* ``prov``        – provenance: which block produced it, from which parents

Frames are frozen; every transformation produces a new frame.  Blocks usually
return plain data and the runtime builds the new frame, carrying annotations,
issues and provenance forward automatically.  A block that needs control can
accept a ``Frame`` and return ``frame.evolve(data=...)``.
"""

from __future__ import annotations

import itertools
import types
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping

import numpy as np

__all__ = [
    "Frame",
    "Issue",
    "Provenance",
    "Missing",
    "MISSING",
    "frozen",
    "as_frame",
    "TimeSpan",
    "PROVENANCE_MODE",
]


@dataclass(frozen=True)
class TimeSpan:
    """When a frame's samples occur: start time ``t0`` (s), sample ``rate`` (Hz), ``n`` samples.
    Derived from the ``t0`` frame annotation and the ``rate`` stream annotation, which every
    source and re-framing block maintains, so blocks rarely touch it directly."""

    t0: float
    rate: float | None
    n: int

    @property
    def t1(self) -> float:
        """End time (exclusive): start of the next frame for contiguous streams."""
        return self.t0 + (self.n / self.rate if self.rate else 0.0)

    @property
    def duration(self) -> float:
        return self.t1 - self.t0

    def stamps(self) -> np.ndarray:
        """Per-sample timestamps."""
        return self.t0 + (
            np.arange(self.n) / self.rate if self.rate else np.zeros(self.n)
        )

    def index(self, t: float) -> int:
        """Sample index at absolute time ``t`` (clipped to the frame)."""
        if not self.rate:
            return 0
        return int(np.clip(round((t - self.t0) * self.rate), 0, max(self.n - 1, 0)))

    def overlaps(self, other: "TimeSpan", tolerance: float = 0.0) -> bool:
        return self.t0 < other.t1 + tolerance and other.t0 < self.t1 + tolerance

    def __repr__(self) -> str:
        return f"TimeSpan(t0={self.t0:.6g}s, rate={self.rate}, n={self.n})"


_ids = itertools.count(1)

import contextvars as _cv

PROVENANCE_MODE: "_cv.ContextVar[str]" = _cv.ContextVar(
    "sigflow_provenance", default="full"
)
"""How much lineage new frames keep: ``"full"`` (every node), ``"roots"`` (each node
keeps only the *source* frames it descends from — replay still works, memory stays
flat), ``"off"`` (no parents, no params). Set per run with ``run(g, provenance=...)``."""


def frozen(m: Mapping[str, Any] | None = None, **kw: Any) -> Mapping[str, Any]:
    """Read-only mapping used for annotations."""
    d = dict(m or {})
    d.update(kw)
    return types.MappingProxyType(d)


class Missing:
    """Sentinel payload of a frame whose computation failed upstream."""

    _inst: "Missing | None" = None

    def __new__(cls) -> "Missing":
        if cls._inst is None:
            cls._inst = super().__new__(cls)
        return cls._inst

    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        return False


MISSING = Missing()


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Provenance:
    """One node of the lineage DAG: *op* applied with *params* to *parents*."""

    op: str
    params: Mapping[str, Any] = field(default_factory=frozen)
    parents: tuple["Provenance", ...] = ()
    seq: int | None = None  # ordinal within the producing stream
    id: int = field(default_factory=lambda: next(_ids))

    def __post_init__(self) -> None:
        mode = PROVENANCE_MODE.get()
        if mode == "full" or not self.parents:
            return
        if mode == "roots":
            roots: list[Provenance] = []
            for p in self.parents:
                for r in p.roots() if p.parents else [p]:
                    if all(r is not q for q in roots):
                        roots.append(r)
            object.__setattr__(self, "parents", tuple(roots))
        elif mode == "off":
            object.__setattr__(self, "parents", ())
            object.__setattr__(self, "params", frozen())

    def walk(self) -> Iterator["Provenance"]:
        """Depth-first walk over the lineage, each node once."""
        seen: set[int] = set()
        stack = [self]
        while stack:
            node = stack.pop()
            if node.id in seen:
                continue
            seen.add(node.id)
            yield node
            stack.extend(reversed(node.parents))

    def chain(self) -> list["Provenance"]:
        """The main line of descent: this node, its first parent, and so on to a root."""
        out: list["Provenance"] = []
        node: "Provenance | None" = self
        while node is not None:
            out.append(node)
            node = node.parents[0] if node.parents else None
        return out

    def __getitem__(self, key: Any) -> Any:
        """Select along the lineage: ``prov[0]`` is this node, ``prov[2]`` two steps
        upstream, ``prov[-1]`` the root, ``prov["fft"]`` the nearest ancestor produced by
        a block called ``fft``, ``prov[1:3]`` a list of nodes along the chain."""
        chain = self.chain()
        if isinstance(key, str):
            for node in self.walk():
                if node.op == key:
                    return node
            raise KeyError(
                f"no ancestor named {key!r}; lineage: {' <- '.join(n.op for n in chain)}"
            )
        if isinstance(key, slice):
            return chain[key]
        if isinstance(key, int):
            try:
                return chain[key]
            except IndexError:
                raise IndexError(
                    f"lineage has {len(chain)} nodes; index {key} is out of range"
                ) from None
        raise TypeError("provenance keys are ints, block names or slices")

    def __len__(self) -> int:
        return len(self.chain())

    def __contains__(self, op: object) -> bool:
        return any(n.op == op for n in self.walk())

    def roots(self) -> list["Provenance"]:
        """Source nodes (no parents) this value ultimately derives from."""
        return [n for n in self.walk() if not n.parents]

    def ops(self) -> list[str]:
        return [n.op for n in self.walk()]

    def render(self, indent: str = "", _seen: set[int] | None = None) -> str:
        """Tree rendering of the lineage; a node reached by several paths is expanded
        once and marked ``(shared, see above)`` afterwards."""
        seen = _seen if _seen is not None else set()
        p = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        head = f"{indent}{self.op}({p})"
        if self.seq is not None:
            head += f"  #{self.seq}"
        if self.id in seen:
            return head + "  (shared, see above)"
        seen.add(self.id)
        lines = [head]
        for par in self.parents:
            lines.append(par.render(indent + "  ", seen))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Provenance({self.op!r}, seq={self.seq}, parents={len(self.parents)})"


# --------------------------------------------------------------------------- #
# Issues (errors / warnings that travel with data)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Issue:
    """An error or warning attached to a frame."""

    message: str
    level: str = "error"  # "error" | "warning" | "info"
    origin: str = ""  # block name that raised it
    exc: BaseException | None = None
    prov: Provenance | None = None

    @property
    def is_error(self) -> bool:
        return self.level == "error"

    def __str__(self) -> str:
        where = f" [{self.origin}]" if self.origin else ""
        return f"{self.level}{where}: {self.message}"


# --------------------------------------------------------------------------- #
# Frame
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Frame:
    data: Any = MISSING
    meta: Mapping[str, Any] = field(default_factory=frozen)  # frame-level
    stream: Mapping[str, Any] = field(default_factory=frozen)  # stream-level
    samples: Mapping[str, Any] = field(default_factory=frozen)  # sample-level
    issues: tuple[Issue, ...] = ()
    prov: Provenance | None = None
    parts: tuple["Frame", ...] = ()  # sub-frames of a tuple frame (fan-out / zip)

    # -- construction / evolution ------------------------------------------ #
    def __post_init__(self) -> None:
        object.__setattr__(self, "meta", frozen(self.meta))
        object.__setattr__(self, "stream", frozen(self.stream))
        object.__setattr__(self, "samples", frozen(self.samples))
        object.__setattr__(self, "issues", tuple(self.issues))

    def evolve(self, **changes: Any) -> "Frame":
        """Copy with fields replaced. ``meta=``/``stream=``/``samples=`` are *merged*."""
        for level in ("meta", "stream", "samples"):
            if level in changes and changes[level] is not None:
                changes[level] = frozen(getattr(self, level), **dict(changes[level]))
        from dataclasses import replace

        return replace(self, **changes)

    def annotate(self, **meta: Any) -> "Frame":
        """Add frame-level annotations."""
        return self.evolve(meta=meta)

    def annotate_stream(self, **stream: Any) -> "Frame":
        return self.evolve(stream=stream)

    def annotate_samples(self, **samples: Any) -> "Frame":
        n = self._leading()
        for k, v in samples.items():
            if n is not None and np.shape(v)[:1] != (n,):
                raise ValueError(
                    f"sample annotation {k!r} has length {np.shape(v)[:1]}, data has {n}"
                )
        return self.evolve(samples=samples)

    def with_issue(
        self,
        message: str,
        level: str = "warning",
        origin: str = "",
        exc: BaseException | None = None,
    ) -> "Frame":
        issue = Issue(message, level, origin, exc, self.prov)
        return self.evolve(issues=self.issues + (issue,))

    def without(self, *keys: str) -> "Frame":
        """Drop frame-level annotation keys."""
        meta = {k: v for k, v in self.meta.items() if k not in keys}
        from dataclasses import replace

        return replace(self, meta=frozen(meta))

    # -- state ------------------------------------------------------------- #
    @property
    def ok(self) -> bool:
        """True when the frame carries data and no error-level issues."""
        return self.data is not MISSING and not any(i.is_error for i in self.issues)

    @property
    def failed(self) -> bool:
        return not self.ok

    @property
    def errors(self) -> tuple[Issue, ...]:
        return tuple(i for i in self.issues if i.is_error)

    @property
    def warnings(self) -> tuple[Issue, ...]:
        return tuple(i for i in self.issues if i.level == "warning")

    def raise_if_failed(self) -> "Frame":
        for i in self.issues:
            if i.is_error:
                if i.exc is not None:
                    raise i.exc
                raise RuntimeError(str(i))
        return self

    # -- lineage ----------------------------------------------------------- #
    def lineage(self) -> Iterator[Provenance]:
        return iter(()) if self.prov is None else self.prov.walk()

    def sources(self) -> list[Provenance]:
        return [] if self.prov is None else self.prov.roots()

    def trace(self) -> str:
        return "(no provenance)" if self.prov is None else self.prov.render()

    # -- time -------------------------------------------------------------- #
    @property
    def time(self) -> TimeSpan | None:
        """The frame's :class:`TimeSpan`, or None when no ``t0`` annotation exists."""
        t0 = self.meta.get("t0")
        if t0 is None:
            return None
        n = self._leading() or 1
        return TimeSpan(float(t0), self.stream.get("rate"), n)

    @property
    def t0(self) -> float | None:
        return self.meta.get("t0")

    def at(self, t: float) -> Any:
        """The sample at absolute time ``t`` (nearest)."""
        span = self.time
        if span is None:
            raise KeyError("frame has no time axis (no 't0' annotation)")
        return np.asarray(self.data)[span.index(t)]

    def between(self, t_start: float, t_end: float) -> "Frame":
        """The part of the frame within [t_start, t_end) as a new frame (time kept)."""
        span = self.time
        if span is None:
            raise KeyError("frame has no time axis (no 't0' annotation)")
        i0, i1 = span.index(t_start), span.index(t_end)
        return self.slice(slice(i0, max(i1, i0)))

    # -- selection --------------------------------------------------------- #
    def __getitem__(self, key: Any) -> Any:
        """``frame["rate"]`` looks an annotation up (frame-level first, then stream-level,
        then sample-level); ``frame[1]`` is part 1 of a tuple frame (from a fan-out / zip)."""
        if isinstance(key, str):
            for level in (self.meta, self.stream, self.samples):
                if key in level:
                    return level[key]
            raise KeyError(
                f"no annotation {key!r}; have meta={list(self.meta)}, "
                f"stream={list(self.stream)}, samples={list(self.samples)}"
            )
        if isinstance(key, int) and (self.parts or isinstance(self.data, tuple)):
            if self.parts:
                return self.parts[key]
            return self.evolve(data=self.data[key], parts=())
        if isinstance(key, (int, slice, tuple)) or key is Ellipsis:
            return self.slice(key)
        raise TypeError(
            "frame keys are annotation names (str), part indices (int) or data slices"
        )

    def slice(self, key: Any) -> "Frame":
        """Slice the data; sample-level annotations are sliced along the leading axis in
        step (``frame[10:20]`` keeps ``samples`` aligned), and time (``t0``) is advanced."""
        data = np.asarray(self.data)[key]
        lead = key[0] if isinstance(key, tuple) else key
        samples = {}
        meta: dict[str, Any] = {}
        if isinstance(lead, slice):
            samples = {k: np.asarray(v)[lead] for k, v in self.samples.items()}
            start = lead.indices(len(self.data))[0] if self._leading() else 0
            rate = self.stream.get("rate")
            if "t0" in self.meta and rate and start:
                meta["t0"] = self.meta["t0"] + start / rate
        elif lead is Ellipsis or lead is None:
            samples = dict(self.samples)
        from dataclasses import replace

        return replace(
            self,
            data=data,
            samples=frozen(samples),
            meta=frozen(self.meta, **meta),
            parts=(),
        )

    def __contains__(self, key: object) -> bool:
        return key in self.meta or key in self.stream or key in self.samples

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    # -- convenience ------------------------------------------------------- #
    def _leading(self) -> int | None:
        if self.data is MISSING or np.ndim(self.data) == 0:
            return None
        try:
            return len(self.data)
        except TypeError:
            return None

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        return np.asarray(self.data, dtype=dtype)

    def _repr_html_(self) -> str:
        from .notebook import frame_html

        return frame_html(self)

    def __repr__(self) -> str:
        d = self.data
        if isinstance(d, np.ndarray):
            ds = f"ndarray{d.shape} {d.dtype}"
        elif d is MISSING:
            ds = "MISSING"
        else:
            ds = type(d).__name__
        extra = []
        if self.meta:
            extra.append(f"meta={dict(self.meta)}")
        if self.issues:
            extra.append(f"issues={len(self.issues)}")
        if self.prov is not None:
            extra.append(f"from={self.prov.op}")
        return f"Frame({ds}" + (", " + ", ".join(extra) if extra else "") + ")"


def as_frame(x: Any) -> Frame:
    return x if isinstance(x, Frame) else Frame(x)


def merge_annotations(frames: Iterable[Frame], *, keep_samples: bool) -> dict[str, Any]:
    """Combine annotations from several parent frames (used by the runtime).

    Frame-level and stream-level dicts are merged left-to-right (later parents
    win on conflicting keys).  Sample-level annotations are only preserved when
    the caller knows the leading axis is unchanged.
    """
    meta: dict[str, Any] = {}
    stream: dict[str, Any] = {}
    samples: dict[str, Any] = {}
    issues: list[Issue] = []
    frames = list(frames)
    for f in frames:
        meta.update(f.meta)
        stream.update(f.stream)
        for i in f.issues:  # an issue shared by several parents appears once
            if all(i is not j for j in issues):
                issues.append(i)
    if keep_samples and len(frames) == 1:
        samples.update(frames[0].samples)
    return {"meta": meta, "stream": stream, "samples": samples, "issues": tuple(issues)}
