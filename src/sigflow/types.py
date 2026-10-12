"""Signal types: explicit, declarative descriptions of what flows along an edge.

A :class:`SignalType` describes the *data* carried by a frame: its element
dtype (a NumPy dtype or an arbitrary Python type), its shape (with symbolic
dimensions such as ``"N"`` that must agree across a graph), an optional sample
rate and physical units.  Types are written with a subscript syntax that reads
like a type annotation::

    Sig[np.complex64, "N"]             # 1-D complex vector of symbolic length N
    Sig[float, ("M", "N")]             # M x N real matrix
    Sig[float, (1024,), 48_000, "V"]   # fixed-length frame at 48 kHz, in volts
    Sig[dict]                          # any Python dict (non-array payload)
    Sig[Any]                           # anything at all

Compatibility between a producer's output type and a consumer's input type is
decided by :func:`unify`, which binds symbolic dimensions in an environment so
that ``"N"`` means the same thing everywhere in a graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "Any",
    "Sig",
    "SignalType",
    "DVar",
    "TupleType",
    "StreamType",
    "TypeMismatch",
    "unify",
    "Env",
    "Dim",
    "scoped",
    "slice_shape",
    "Scalar",
    "Vector",
    "Matrix",
    "Tensor",
    "CVector",
    "CMatrix",
    "is_signal_type",
    "describe",
]

# A dimension is a fixed int, a symbolic name (str), or None (= any length).
Dim = int | str | None


class TypeMismatch(TypeError):
    """Raised when two signal types cannot be unified."""


@dataclass(frozen=True)
class DVar:
    """A dtype variable: ``Sig["T", "N"] -> Sig["T", "N"]`` preserves the input dtype."""

    name: str

    def __str__(self) -> str:
        return self.name


# --------------------------------------------------------------------------- #
# Environment for symbolic dimensions
# --------------------------------------------------------------------------- #


class Env:
    """Union-find over symbolic dimensions and dtype variables.

    Symbols are *scoped to the block that declares them* (``N`` in one block is
    unrelated to ``N`` in another) and become equal only by meeting across an
    edge.  Concrete values are attached when a fixed size or a real frame is
    seen.  Bound symbols are shown with their scope stripped (``N=1024``).
    """

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._value: dict[str, Any] = {}  # root -> int (dims) or dtype (dtype vars)

    # -- union-find --------------------------------------------------------- #
    def find(self, name: str) -> str:
        self._parent.setdefault(name, name)
        while self._parent[name] != name:
            self._parent[name] = self._parent[self._parent[name]]
            name = self._parent[name]
        return name

    def value(self, name: str) -> Any:
        return self._value.get(self.find(name))

    def bind(self, name: str, value: int, where: str = "") -> None:
        root = self.find(name)
        prev = self._value.get(root)
        if prev is not None and prev != value:
            raise TypeMismatch(
                f"{where or 'edge'}: dimension {_plain(name)!r} is {prev} but here needs {value}"
            )
        self._value[root] = value

    def union(self, a: str, b: str, where: str = "") -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        va, vb = self._value.get(ra), self._value.get(rb)
        if va is not None and vb is not None and va != vb:
            raise TypeMismatch(
                f"{where or 'edge'}: dimension {_plain(a)!r}={va} cannot equal {_plain(b)!r}={vb}"
            )
        self._parent[rb] = ra
        if va is None and vb is not None:
            self._value[ra] = vb

    def bind_dtype(self, name: str, dt: Any, where: str = "") -> None:
        root = self.find(name)
        prev = self._value.get(root)
        if prev is None:
            self._value[root] = dt
        elif prev != dt and not _dtype_ok(dt, prev):
            raise TypeMismatch(
                f"{where or 'edge'}: dtype variable {_plain(name)!r} is {_dn(prev)} but here needs {_dn(dt)}"
            )

    def union_dtype(self, a: str, b: str, where: str = "") -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        va, vb = self._value.get(ra), self._value.get(rb)
        if va is not None and vb is not None and va != vb and not _dtype_ok(vb, va):
            raise TypeMismatch(
                f"{where or 'edge'}: dtype {_dn(va)} cannot equal {_dn(vb)}"
            )
        self._parent[rb] = ra
        if va is None and vb is not None:
            self._value[ra] = vb

    # -- views ------------------------------------------------------------- #
    @property
    def bindings(self) -> dict[str, Any]:
        """Bound symbols by plain name (scope stripped); for display and tests."""
        out: dict[str, Any] = {}
        for name in self._parent:
            v = self.value(name)
            if v is not None:
                out.setdefault(_plain(name), v)
        return out

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            "Env("
            + ", ".join(
                f"{k}={_dn(v) if not isinstance(v, int) else v}"
                for k, v in self.bindings.items()
            )
            + ")"
        )


def _plain(sym: str) -> str:
    return sym.split("#", 1)[0]


def scoped(t: Any, scope: str) -> Any:
    """Return a copy of ``t`` with its symbols made unique to ``scope``."""
    if isinstance(t, TupleType):
        return TupleType(tuple(scoped(i, scope) for i in t.items))
    if isinstance(t, StreamType):
        return StreamType(scoped(t.item, scope))
    if not isinstance(t, SignalType):
        return t
    dtype = t.dtype
    if isinstance(dtype, DVar) and "#" not in dtype.name:
        dtype = DVar(f"{dtype.name}#{scope}")
    shape = t.shape
    if shape is not None:
        shape = tuple(
            f"{d}#{scope}" if isinstance(d, str) and "#" not in d else d for d in shape
        )
    return t.with_(dtype=dtype, shape=shape)


# --------------------------------------------------------------------------- #
# SignalType
# --------------------------------------------------------------------------- #


def _norm_dtype(dt: Any) -> Any:
    """Normalise a dtype spec: NumPy dtypes stay dtypes, Python types stay types."""
    if dt is Any or dt is None:
        return Any
    if isinstance(dt, np.dtype):
        return dt
    if isinstance(dt, type) and issubclass(dt, np.generic):
        return np.dtype(dt)
    if dt in (float, complex, int, bool):
        return np.dtype(dt)
    if isinstance(dt, DVar):
        return dt
    if isinstance(dt, str):
        # Single letters are dtype *variables* (Sig["T", "N"]); longer names are
        # NumPy dtype names ("float32", "complex128", ...).
        if len(dt) == 1 and dt.isalpha():
            return DVar(dt)
        try:
            return np.dtype(dt)
        except TypeError:
            if dt.isidentifier():
                return DVar(dt)
            raise TypeError(f"unknown dtype string {dt!r}") from None
    if isinstance(dt, type):
        return dt  # arbitrary Python class, e.g. dict or a dataclass
    raise TypeError(f"cannot interpret {dt!r} as a dtype")


def _norm_shape(shape: Any) -> tuple[Dim, ...] | None:
    if shape is None or shape is Any:
        return None
    if isinstance(shape, (str, int)):
        return (shape,)
    out: list[Dim] = []
    for d in shape:
        if d is None or isinstance(d, (int, str)):
            out.append(d)
        else:
            raise TypeError(f"bad dimension {d!r}; use int, str (symbolic) or None")
    return tuple(out)


@dataclass(frozen=True)
class SignalType:
    """Description of the data carried along one edge of a graph.

    Attributes:
        dtype: ``Any``, a ``np.dtype`` for array payloads, or a Python ``type``
            for arbitrary object payloads.
        shape: ``None`` for *any* shape, else a tuple of :data:`Dim`.
            ``()`` denotes a scalar.
        rate: sample rate in Hz (``None`` = unspecified / don't care).
        units: free-form unit string (``None`` = unspecified).
        meta: required frame-level annotation keys (``None`` = no requirement).
    """

    dtype: Any = Any
    shape: tuple[Dim, ...] | None = None
    rate: float | None = None
    units: str | None = None
    meta: frozenset[str] = field(default_factory=frozenset)

    # -- construction ------------------------------------------------------ #
    def __class_getitem__(cls, params: Any) -> "SignalType":
        if not isinstance(params, tuple):
            params = (params,)
        dtype = params[0] if len(params) > 0 else Any
        shape = params[1] if len(params) > 1 else None
        rate = params[2] if len(params) > 2 else None
        units = params[3] if len(params) > 3 else None
        if len(params) > 4:
            raise TypeError(
                "Sig[dtype, shape, rate, units] takes at most four parameters"
            )
        return cls(_norm_dtype(dtype), _norm_shape(shape), rate, units)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", _norm_dtype(self.dtype))
        object.__setattr__(self, "shape", _norm_shape(self.shape))

    # -- derived views ----------------------------------------------------- #
    @property
    def is_array(self) -> bool:
        return isinstance(self.dtype, np.dtype)

    @property
    def is_any(self) -> bool:
        return (
            self.dtype is Any
            and self.shape is None
            and self.rate is None
            and self.units is None
        )

    @property
    def ndim(self) -> int | None:
        return None if self.shape is None else len(self.shape)

    def symbols(self) -> set[str]:
        return {d for d in (self.shape or ()) if isinstance(d, str)}

    def with_(self, **changes: Any) -> "SignalType":
        """Return a copy with some fields replaced (``shape`` is re-normalised)."""
        from dataclasses import replace

        return replace(self, **changes)

    def requires(self, *keys: str) -> "SignalType":
        """Require these frame-level annotation keys to be present."""
        return self.with_(meta=self.meta | frozenset(keys))

    def resolve(self, env: Env) -> "SignalType":
        """Substitute bound symbolic dimensions with their integers."""
        t = self
        if isinstance(self.dtype, DVar) and env.value(self.dtype.name) is not None:
            t = t.with_(dtype=env.value(self.dtype.name))
        if self.shape is None:
            return t
        shape = tuple(
            (env.value(d) if env.value(d) is not None else d)
            if isinstance(d, str)
            else d
            for d in self.shape
        )
        return t.with_(shape=shape)

    # -- runtime check ----------------------------------------------------- #
    def check(self, value: Any, env: Env | None = None, where: str = "") -> Env:
        """Verify a concrete value against this type. Returns the (possibly updated) env."""
        env = env if env is not None else Env()
        if self.dtype is Any and self.shape is None:
            return env
        if isinstance(self.dtype, DVar):
            arr = np.asarray(value)
            env.bind_dtype(self.dtype.name, arr.dtype, where)
            if self.shape is not None:
                self.with_(dtype=arr.dtype).check(value, env, where)
            return env
        if self.is_array:
            arr = np.asarray(value)
            if not np.can_cast(arr.dtype, self.dtype, casting="same_kind"):
                raise TypeMismatch(
                    f"{where}: value dtype {arr.dtype} is not {self.dtype}"
                )
            if self.shape is not None:
                if arr.ndim != len(self.shape):
                    raise TypeMismatch(
                        f"{where}: value has rank {arr.ndim}, type requires {len(self.shape)}"
                    )
                for i, (d, n) in enumerate(zip(self.shape, arr.shape)):
                    if isinstance(d, int) and d != n:
                        raise TypeMismatch(
                            f"{where}: axis {i} has length {n}, expected {d}"
                        )
                    if isinstance(d, str):
                        env.bind(d, int(n), where)
        elif isinstance(self.dtype, type):
            if not isinstance(value, self.dtype):
                raise TypeMismatch(
                    f"{where}: value of type {type(value).__name__} is not {self.dtype.__name__}"
                )
        return env

    # -- display ----------------------------------------------------------- #
    def __str__(self) -> str:
        if self.is_any:
            return "Sig[Any]"
        parts = [_dn(self.dtype)]
        if self.shape is not None or self.rate is not None or self.units is not None:
            if self.shape is None:
                parts.append("*")
            else:
                parts.append(
                    "("
                    + ", ".join(
                        "?" if d is None else _plain(str(d)) for d in self.shape
                    )
                    + ("," if len(self.shape) == 1 else "")
                    + ")"
                )
        if self.rate is not None or self.units is not None:
            parts.append(f"{self.rate}Hz" if self.rate is not None else "_")
        if self.units is not None:
            parts.append(repr(self.units))
        return "Sig[" + ", ".join(parts) + "]"

    __repr__ = __str__


Sig = SignalType


# --------------------------------------------------------------------------- #
# Composite types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TupleType:
    """Type of a frame whose data is a tuple of several signals (fan-out / zip)."""

    items: tuple[Any, ...]

    def __class_getitem__(cls, params: Any) -> "TupleType":
        if not isinstance(params, tuple):
            params = (params,)
        return cls(tuple(_as_type(p) for p in params))

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def resolve(self, env: Env) -> "TupleType":
        return TupleType(tuple(t.resolve(env) for t in self.items))

    def check(self, value: Any, env: Env | None = None, where: str = "") -> Env:
        env = env if env is not None else Env()
        if not isinstance(value, tuple) or len(value) != len(self.items):
            raise TypeMismatch(f"{where}: expected a {len(self.items)}-tuple")
        for i, (t, v) in enumerate(zip(self.items, value)):
            env = t.check(v, env, f"{where}[{i}]")
        return env

    def __str__(self) -> str:
        return "(" + ", ".join(str(t) for t in self.items) + ")"

    __repr__ = __str__


@dataclass(frozen=True)
class StreamType:
    """Marker used in annotations of *stream* blocks: ``Stream[Sig[...]]``.

    A stream block receives the whole iterator of frames rather than one frame
    at a time; its element type is ``item``.
    """

    item: Any

    def __class_getitem__(cls, item: Any) -> "StreamType":
        # Stream[Framed[...]] and Stream[Frame] deliver whole frames; keep those markers as-is
        if (
            isinstance(item, type)
            and item.__name__ == "Frame"
            or type(item).__name__ == "Framed"
        ):
            return cls(item)
        return cls(_as_type(item))

    def __str__(self) -> str:
        return f"Stream[{self.item}]"

    __repr__ = __str__


def is_signal_type(obj: Any) -> bool:
    return isinstance(obj, (SignalType, TupleType))


def _as_type(obj: Any) -> Any:
    """Coerce loose specs into a type object."""
    if isinstance(obj, (SignalType, TupleType, StreamType)):
        return obj
    if obj is Any or obj is None:
        return SignalType()
    if isinstance(obj, tuple):
        return TupleType(tuple(_as_type(o) for o in obj))
    if isinstance(obj, (type, np.dtype, str)):
        return SignalType(obj)
    raise TypeError(f"cannot interpret {obj!r} as a signal type")


as_type = _as_type


# --------------------------------------------------------------------------- #
# Unification
# --------------------------------------------------------------------------- #


def _unify_dtype(produced: Any, required: Any, env: Env, where: str) -> None:
    if isinstance(produced, DVar) and isinstance(required, DVar):
        env.union_dtype(produced.name, required.name, where)
        return
    if isinstance(required, DVar):
        if produced is not Any:
            env.bind_dtype(required.name, produced, where)
        return
    if isinstance(produced, DVar):
        v = env.value(produced.name)
        if v is None:
            env.bind_dtype(
                produced.name, required, where
            )  # the only thing known about it
            return
        produced = v
    if not _dtype_ok(produced, required):
        raise TypeMismatch(
            f"{where}: dtype {_dn(produced)} cannot feed {_dn(required)}"
        )


def _dtype_ok(produced: Any, required: Any) -> bool:
    if required is Any or produced is Any:
        return True  # gradual typing: Any is compatible both ways (checked at runtime)
    if isinstance(required, np.dtype):
        if not isinstance(produced, np.dtype):
            return False
        return bool(np.can_cast(produced, required, casting="safe"))
    # Python classes
    if isinstance(produced, np.dtype):
        return required is object
    return issubclass(produced, required)


def unify(produced: Any, required: Any, env: Env | None = None, where: str = "") -> Env:
    """Check that a *produced* type satisfies a *required* type.

    Symbolic dimensions are bound in ``env``; an already-bound symbol must
    agree.  Two symbols meeting each other are recorded as aliases by binding
    whichever side is bound.  Raises :class:`TypeMismatch` on failure.
    """
    env = env if env is not None else Env()
    produced, required = _as_type(produced), _as_type(required)

    if isinstance(required, TupleType) or isinstance(produced, TupleType):
        if not (isinstance(required, TupleType) and isinstance(produced, TupleType)):
            if isinstance(required, SignalType) and required.is_any:
                return env
            raise TypeMismatch(f"{where}: {produced} is not compatible with {required}")
        if len(required) != len(produced):
            raise TypeMismatch(
                f"{where}: {len(produced)} signals produced, {len(required)} required"
            )
        for i, (p, r) in enumerate(zip(produced, required)):
            env = unify(p, r, env, f"{where}[{i}]")
        return env

    _unify_dtype(produced.dtype, required.dtype, env, where)

    if required.shape is not None:
        if produced.shape is None:
            return env  # gradual: unknown shape is checked at runtime
        if len(produced.shape) != len(required.shape):
            raise TypeMismatch(
                f"{where}: rank {len(produced.shape)} produced, rank {len(required.shape)} required "
                f"({produced} -> {required})"
            )
        for i, (p, r) in enumerate(zip(produced.shape, required.shape)):
            _unify_dim(p, r, env, f"{where} axis {i}")

    if (
        required.rate is not None
        and produced.rate is not None
        and produced.rate != required.rate
    ):
        raise TypeMismatch(
            f"{where}: rate {produced.rate} Hz produced, {required.rate} Hz required"
        )
    if (
        required.units is not None
        and produced.units is not None
        and produced.units != required.units
    ):
        from .units import compatible

        hint = (
            f" (convertible: insert dsp.to_units({required.units!r}))"
            if compatible(produced.units, required.units)
            else " (different dimensions)"
        )
        raise TypeMismatch(
            f"{where}: units {produced.units!r} produced, {required.units!r} required{hint}"
        )
    return env


def _unify_dim(p: Dim, r: Dim, env: Env, where: str) -> None:
    if r is None or p is None:
        return
    if isinstance(p, str) and isinstance(r, str):
        env.union(p, r, where)
    elif isinstance(p, str) and isinstance(r, int):
        env.bind(p, r, where)
    elif isinstance(r, str) and isinstance(p, int):
        env.bind(r, p, where)
    elif p != r:
        raise TypeMismatch(f"{where}: length {p} produced, {r} required")


def _dn(dt: Any) -> str:
    if dt is Any:
        return "Any"
    if isinstance(dt, (np.dtype, DVar)):
        return _plain(dt.name) if isinstance(dt, DVar) else dt.name
    return dt.__name__


def slice_shape(shape: tuple | None, key: Any) -> tuple | None:
    """Shape after NumPy-style basic indexing: ints known statically are computed,
    symbolic dimensions become new symbols (``N`` -> ``N'``) unless the slice is ``[:]``."""
    if shape is None:
        return None
    keys = list(key) if isinstance(key, tuple) else [key]
    out: list = []
    i = 0
    for k in keys:
        if k is Ellipsis:
            rest = (
                len(shape)
                - i
                - sum(1 for kk in keys[keys.index(k) + 1 :] if kk is not None)
            )
            out.extend(shape[i : i + rest])
            i += rest
            continue
        if k is None:
            out.append(1)
            continue
        if i >= len(shape):
            raise TypeMismatch(
                f"too many indices for shape {tuple(_plain(str(x)) if isinstance(x, str) else x for x in shape)}"
            )
        d = shape[i]
        i += 1
        if isinstance(k, int):
            if isinstance(d, int) and not -d <= k < d:
                raise TypeMismatch(
                    f"index {k} out of range for dimension of length {d}"
                )
            continue  # axis removed
        if isinstance(k, slice):
            if k == slice(None):
                out.append(d)
            elif isinstance(d, int):
                out.append(len(range(*k.indices(d))))
            elif (
                isinstance(d, str)
                and k.start is not None
                and k.stop is not None
                and k.start >= 0
                and k.stop >= 0
                and (k.step or 1) > 0
            ):
                out.append(
                    len(range(k.start, k.stop, k.step or 1))
                )  # explicit bounds: known length
            elif isinstance(d, str):
                name, _, scope = d.partition("#")
                out.append(
                    f"{name}'#{scope}" if scope else name + "'"
                )  # a new, unrelated symbol
            else:
                out.append(None)
            continue
        raise TypeMismatch(f"unsupported index {k!r} (ints, slices, None and ... only)")
    out.extend(shape[i:])
    return tuple(out)


def describe(t: Any) -> str:
    return str(_as_type(t))


# --------------------------------------------------------------------------- #
# Handy aliases for linear-algebra primitives
# --------------------------------------------------------------------------- #

Scalar = Sig[float, ()]
Vector = Sig[float, ("N",)]
Matrix = Sig[float, ("M", "N")]
Tensor = Sig[float, None]
CVector = Sig[complex, ("N",)]
CMatrix = Sig[complex, ("M", "N")]
