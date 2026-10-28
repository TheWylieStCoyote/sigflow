"""The :class:`Frame` — the immutable unit of data that flows through a graph.

A frame bundles four things:

* ``data``        – the payload (an array, a scalar, a tuple, any Python object)
* annotations     – at four levels:
    - ``stream``  – facts about the whole stream (rate, units, source name …)
    - ``meta``    – facts about this frame (timestamp, sequence number, SNR …)
    - ``samples`` – per-sample arrays aligned with ``data``'s leading axis
                    (e.g. a boolean "clipped" mask, a timestamp vector)
    - ``tags``    – labels (``"speech"``, ``segment=3``) on the whole frame or
                    on a time span of it; a set, united at joins, that records
                    which block attached each label
* ``issues``      – errors and warnings that have been attached along the way
* ``prov``        – provenance: which block produced it, from which parents

Frames are frozen; every transformation produces a new frame.  Blocks usually
return plain data and the runtime builds the new frame, carrying annotations,
issues and provenance forward automatically.  A block that needs control can
accept a ``Frame`` and return ``frame.evolve(data=...)``.
"""

from __future__ import annotations

import enum
import itertools
import math
import types
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping

import numpy as np

from ._arrays import is_positional

__all__ = [
    "Frame",
    "Issue",
    "Provenance",
    "NO_PROV",
    "Missing",
    "MISSING",
    "frozen",
    "as_frame",
    "TimeSpan",
    "Tag",
    "Tags",
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
        stack: list[Provenance] = [self]
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

    # -- persistence (implemented in sigflow.persist) ----------------------- #
    def to_dict(self) -> dict:
        """The lineage DAG as a JSON-ready dict."""
        from .persist import prov_to_dict

        return prov_to_dict(self)

    def to_json(self, **kw: Any) -> str:
        import json

        return json.dumps(self.to_dict(), **kw)

    @staticmethod
    def from_dict(d: dict) -> "Provenance":
        from .persist import prov_from_dict

        return prov_from_dict(d)

    @staticmethod
    def from_json(s: str) -> "Provenance":
        import json

        return Provenance.from_dict(json.loads(s))

    def __bool__(self) -> bool:
        """False only for :data:`NO_PROV`, the lineage of a frame no block has produced."""
        return bool(self.op) or bool(self.parents)

    def __repr__(self) -> str:
        return f"Provenance({self.op!r}, seq={self.seq}, parents={len(self.parents)})"


NO_PROV: Provenance = Provenance("")
"""The empty lineage: what ``Frame(data).prov`` is before any block produced the frame.
It is falsy (``if f.prov:``), has no parents and renders as ``(no provenance)``, so
lineage code never needs an ``is None`` branch and ``f.prov.op`` always type-checks."""


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
    prov: Provenance = NO_PROV

    def __post_init__(self) -> None:
        if self.prov is None:  # accepted for compatibility; stored as the empty lineage
            object.__setattr__(self, "prov", NO_PROV)

    @property
    def is_error(self) -> bool:
        return self.level == "error"

    def __str__(self) -> str:
        where = f" [{self.origin}]" if self.origin else ""
        return f"{self.level}{where}: {self.message}"


# --------------------------------------------------------------------------- #
# Tags (labels on a frame or on a time span of it)
# --------------------------------------------------------------------------- #

_SCALARS = (type(None), bool, int, float, str)


def _tag_value(v: Any) -> Any:
    """Normalise a tag value: scalars, enums and tuples of those. Arrays belong in
    ``samples`` and objects in ``meta``; a tag has to hash, compare and round-trip."""
    if isinstance(v, enum.Enum):
        return v.value
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, _SCALARS):
        return v
    if isinstance(v, (tuple, list)):
        return tuple(_tag_value(x) for x in v)
    raise TypeError(
        f"tag values are scalars or tuples of scalars, not {type(v).__name__}; "
        "use meta for objects and samples for arrays"
    )


@dataclass(frozen=True, eq=False)
class Tag:
    """A label on a frame (``span=None``) or on ``[t0, t1)`` seconds of it.

    Identity is the name and the span: two tags that differ only in value or origin
    are the same tag, and the later one wins when frames merge. ``origin`` is the
    block that attached it, stamped by the runtime."""

    name: str
    value: Any = None
    span: tuple[float, float] | None = None
    origin: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise TypeError("a tag name is a non-empty string")
        object.__setattr__(self, "value", _tag_value(self.value))
        if self.span is not None:
            s0, s1 = self.span
            object.__setattr__(self, "span", (float(s0), float(s1)))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Tag) and (self.name, self.span) == (
            other.name,
            other.span,
        )

    def __hash__(self) -> int:
        return hash((self.name, self.span))

    def covers(self, t: float) -> bool:
        """True for a whole-frame tag, or a span containing ``t``."""
        if self.span is None:
            return True
        s0, s1 = self.span
        return s0 <= t < s1 or (s1 <= s0 and t == s0)

    def __repr__(self) -> str:
        out = self.name
        if self.value is not None:
            out += f"={self.value!r}"
        if self.span is not None:
            out += f"@[{self.span[0]:.6g}, {self.span[1]:.6g})"
        return out


class Tags(frozenset):
    """The tags of a frame: a frozenset of :class:`Tag` with lookup by name.

    ``"speech" in tags`` tests a name, ``tags["segment"]`` is a value (the whole-
    frame tag first, then any span), ``tags.at(t)`` are the tags covering time
    ``t``. :meth:`merge` unites several frames' tags, later ones winning on value.
    """

    def __new__(cls, items: Iterable[Any] = ()) -> "Tags":
        tags = [
            Tag(t) if isinstance(t, str) else t
            for t in (items if not isinstance(items, str) else [items])
        ]
        for t in tags:
            if not isinstance(t, Tag):
                raise TypeError(f"not a Tag: {t!r}")
        return super().__new__(cls, tags)

    def __contains__(self, item: object) -> bool:
        if isinstance(item, str):
            return any(t.name == item for t in self)
        return super().__contains__(item)

    def __getitem__(self, name: str) -> Any:
        found = [t for t in self if t.name == name]
        if not found:
            raise KeyError(f"no tag {name!r}; have {sorted(self.names)}")
        found.sort(key=lambda t: (t.span is not None, t.span or (0.0, 0.0)))
        return found[0].value

    def get(self, name: str, default: Any = None) -> Any:
        try:
            return self[name]
        except KeyError:
            return default

    @property
    def names(self) -> frozenset[str]:
        return frozenset(t.name for t in self)

    def at(self, t: float) -> "Tags":
        return Tags(tag for tag in self if tag.covers(t))

    def whole(self) -> "Tags":
        """Only the tags that apply to the whole frame (no span)."""
        return Tags(tag for tag in self if tag.span is None)

    def clip(self, t0: float, t1: float) -> "Tags":
        """Spanned tags cut to the window ``[t0, t1)``, those outside it dropped."""
        out = []
        for tag in self:
            if tag.span is None:
                out.append(tag)
                continue
            s0, s1 = tag.span
            if s1 <= s0:  # a point
                if t0 <= s0 < t1:
                    out.append(tag)
                continue
            lo, hi = max(s0, t0), min(s1, t1)
            if lo < hi:
                out.append(
                    tag if (lo, hi) == tag.span else _replace(tag, span=(lo, hi))
                )
        return Tags(out)

    @staticmethod
    def merge(*groups: Iterable[Any]) -> "Tags":
        """The union of several tag sets; on the same name and span the later wins."""
        by_key: dict[Tag, Tag] = {}
        for group in groups:
            for tag in Tags(group):
                by_key.pop(tag, None)
                by_key[tag] = tag
        return Tags(by_key.values())

    def __repr__(self) -> str:
        return "Tags({" + ", ".join(sorted(map(repr, self))) + "})"

    def __reduce__(self) -> Any:
        return (Tags, (tuple(self),))


def _replace(obj: Any, **changes: Any) -> Any:
    from dataclasses import replace

    return replace(obj, **changes)


# --------------------------------------------------------------------------- #
# Frame
# --------------------------------------------------------------------------- #


def _index(data: Any, key: Any) -> Any:
    """``data[key]`` for positional arrays (ndarray, DataArray, CuPy, dask...), so
    labels and device placement survive; lists and label-indexed objects (pandas)
    go through ``np.asarray`` as before."""
    if is_positional(data):
        return data[key]
    return np.asarray(data)[key]


@dataclass(frozen=True)
class Frame:
    data: Any = MISSING
    meta: Mapping[str, Any] = field(default_factory=frozen)  # frame-level
    stream: Mapping[str, Any] = field(default_factory=frozen)  # stream-level
    samples: Mapping[str, Any] = field(default_factory=frozen)  # sample-level
    issues: tuple[Issue, ...] = ()
    prov: Provenance = NO_PROV
    parts: tuple["Frame", ...] = ()  # sub-frames of a tuple frame (fan-out / zip)
    tags: Tags = field(default_factory=Tags)  # labels on the frame or a span of it

    # -- construction / evolution ------------------------------------------ #
    def __post_init__(self) -> None:
        object.__setattr__(self, "meta", frozen(self.meta))
        object.__setattr__(self, "stream", frozen(self.stream))
        object.__setattr__(self, "samples", frozen(self.samples))
        object.__setattr__(self, "issues", tuple(self.issues))
        if self.prov is None:  # accepted for compatibility; stored as the empty lineage
            object.__setattr__(self, "prov", NO_PROV)
        if self.tags or not isinstance(self.tags, Tags):
            tags = Tags(self.tags)
            # a spanned tag only means something inside this frame's window: cut it
            # to the window whenever the frame is made (slicing, reframing, evolving)
            span = self.time if tags else None
            if span is not None and span.rate and any(t.span for t in tags):
                tags = tags.clip(span.t0, span.t1)
            object.__setattr__(self, "tags", tags)

    def evolve(self, **changes: Any) -> "Frame":
        """Copy with fields replaced. ``meta=``/``stream=``/``samples=``/``tags=`` are
        *merged* (tags: the later wins on the same name and span)."""
        for level in ("meta", "stream", "samples"):
            if level in changes and changes[level] is not None:
                changes[level] = frozen(getattr(self, level), **dict(changes[level]))
        if changes.get("tags") is not None:
            changes["tags"] = Tags.merge(self.tags, changes["tags"])
        from dataclasses import replace

        return replace(self, **changes)

    def tag(
        self, *names: str, span: tuple[float, float] | None = None, **values: Any
    ) -> "Frame":
        """Attach tags: bare labels (``f.tag("speech")``), labels with a value
        (``f.tag(segment=3)``), on the whole frame or on ``span=(t0, t1)`` seconds."""
        tags = [Tag(n, None, span) for n in names]
        tags += [Tag(k, v, span) for k, v in values.items()]
        return self.evolve(tags=tags)

    def untag(self, *names: str) -> "Frame":
        """Drop every tag with these names, whatever their span."""
        return _replace(self, tags=Tags(t for t in self.tags if t.name not in names))

    def mask(self, name: str) -> np.ndarray:
        """A boolean array along the leading axis: True where a tag ``name`` applies
        (everywhere for a whole-frame tag, inside the spans otherwise)."""
        n = self._leading() or 0
        out = np.zeros(n, dtype=bool)
        found = [t for t in self.tags if t.name == name]
        if not found:
            return out
        if any(t.span is None for t in found):
            out[:] = True
            return out
        span = self.time
        if span is None or not span.rate:
            raise KeyError("frame has no time axis (no 't0' annotation or rate)")
        for t in found:
            s0, s1 = t.span  # type: ignore[misc]
            if s1 <= s0:  # a point: the nearest sample
                if n:
                    out[span.index(s0)] = True
                continue
            # membership, start <= t < stop: the first sample at or after each edge
            i0 = math.ceil(round((s0 - span.t0) * span.rate, 9))
            i1 = math.ceil(round((s1 - span.t0) * span.rate, 9))
            out[max(i0, 0) : min(i1, n)] = True
        return out

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
        return iter(()) if not self.prov else self.prov.walk()

    def sources(self) -> list[Provenance]:
        return [] if not self.prov else self.prov.roots()

    def trace(self) -> str:
        return "(no provenance)" if not self.prov else self.prov.render()

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
        return _index(self.data, span.index(t))

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
        then sample-level, then a tag's value); ``frame[1]`` is part 1 of a tuple frame
        (from a fan-out / zip)."""
        if isinstance(key, str):
            for level in (self.meta, self.stream, self.samples):
                if key in level:
                    return level[key]
            if key in self.tags:
                return self.tags[key]
            raise KeyError(
                f"no annotation {key!r}; have meta={list(self.meta)}, "
                f"stream={list(self.stream)}, samples={list(self.samples)}, "
                f"tags={sorted(self.tags.names)}"
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
        data = _index(self.data, key)
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
        return (
            key in self.meta
            or key in self.stream
            or key in self.samples
            or key in self.tags
        )

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    # -- convenience ------------------------------------------------------- #
    def _leading(self) -> int | None:
        if self.data is MISSING:
            return None
        try:
            if np.ndim(self.data) == 0:
                return None
            return len(self.data)
        except (TypeError, ValueError):  # scalars, ragged tuples
            return None

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        return np.asarray(self.data, dtype=dtype)

    # -- persistence (implemented in sigflow.persist) ----------------------- #
    def save(self, path: str) -> str:
        """Write the frame (data, annotations, issues, lineage) as ``.npz``."""
        from .persist import save_frame

        return save_frame(self, path)

    @staticmethod
    def load(path: str) -> "Frame":
        from .persist import load_frame

        return load_frame(path)

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
        if self.tags:
            extra.append(f"tags={sorted(map(repr, self.tags))}")
        if self.issues:
            extra.append(f"issues={len(self.issues)}")
        if self.prov:
            extra.append(f"from={self.prov.op}")
        return f"Frame({ds}" + (", " + ", ".join(extra) if extra else "") + ")"


def as_frame(x: Any) -> Frame:
    return x if isinstance(x, Frame) else Frame(x)


def merge_annotations(frames: Iterable[Frame], *, keep_samples: bool) -> dict[str, Any]:
    """Combine annotations from several parent frames (used by the runtime).

    Frame-level and stream-level dicts are merged left-to-right (later parents
    win on conflicting keys).  Tags are united (later parents win on the same
    name and span).  Sample-level annotations and spanned tags are only preserved
    when the caller knows the leading axis is unchanged.
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
    tags = (
        Tags.merge(*(f.tags for f in frames)) if any(f.tags for f in frames) else Tags()
    )
    if not keep_samples:
        tags = tags.whole()
    return {
        "meta": meta,
        "stream": stream,
        "samples": samples,
        "issues": tuple(issues),
        "tags": tags,
    }
