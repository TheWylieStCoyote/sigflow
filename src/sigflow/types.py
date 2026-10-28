"""Signal types: explicit, declarative descriptions of what flows along an edge.

A :class:`SignalType` describes the *data* carried by a frame: its element
dtype (a NumPy dtype or an arbitrary Python type), its shape (with symbolic
dimensions such as ``"N"`` that must agree across a graph), an optional sample
rate and physical units.  Types are written with a subscript syntax that reads
like a type annotation::

    Sig[np.complex64, N]             # 1-D complex vector of symbolic length N
    Sig[float, tuple[M, N]]             # M x N real matrix
    Sig[float, tuple[Literal[1024]], Literal[48_000], Literal["V"]]   # fixed-length frame at 48 kHz, in volts
    Sig[dict]                          # any Python dict (non-array payload)
    Sig[Any]                           # anything at all

Compatibility between a producer's output type and a consumer's input type is
decided by :func:`unify`, which binds symbolic dimensions in an environment so
that ``"N"`` means the same thing everywhere in a graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, get_args, get_origin

import numpy as np
from ._arrays import array_metadata as _array_metadata
from .dims import M, N, T

__all__ = [
    "Any",
    "Sig",
    "Stream",
    "SignalType",
    "DVar",
    "TupleType",
    "StreamType",
    "TypeMismatch",
    "unify",
    "narrow",
    "resolve",
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
Dim = (
    int | str | type | None
)  # ``type``: a ``sigflow.dims`` symbol as a checker sees it


class TypeMismatch(TypeError):
    """Raised when two signal types cannot be unified."""


@dataclass(frozen=True)
class DVar:
    """A dtype variable: ``Sig[T, N] -> Sig[T, N]`` preserves the input dtype."""

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
        # Single letters are dtype *variables* (Sig[T, N]); longer names are
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
    if get_origin(shape) is tuple:  # tuple[M, N], tuple[()], tuple[Literal[1024]]
        args = get_args(shape)
        shape = () if args == ((),) else args
    shape = _unlit(shape)
    if isinstance(shape, (str, int)):
        return (shape,)
    out: list[Dim] = []
    for d in shape:
        d = _unlit(d)
        if d is None or isinstance(d, (int, str)):
            out.append(d)
        else:
            raise TypeError(f"bad dimension {d!r}; use int, str (symbolic) or None")
    return tuple(out)


def _unlit(x: Any) -> Any:
    """``Literal["V"]`` -> ``"V"``: the checker-readable spelling of a rate or unit."""
    return get_args(x)[0] if get_origin(x) is Literal else x


def _labelled_shape(value: Any, want: Any, shape: tuple[int, ...]) -> tuple[int, ...]:
    """A labelled array (``xarray.DataArray``) is matched to a contract that names
    all of its dimensions *by name*: ``Sig[float, tuple[CHANNEL, TIME]]`` accepts the
    array in any layout, and the sizes are read per name. Otherwise, by position."""
    sizes = getattr(value, "sizes", None)
    if (
        want is not None
        and sizes is not None
        and len(want) == len(shape)
        and all(isinstance(d, str) for d in want)
        and all(_plain(d) in sizes for d in want)
    ):
        return tuple(int(sizes[_plain(d)]) for d in want)
    return shape


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
        rate = _unlit(params[2]) if len(params) > 2 else None
        units = _unlit(params[3]) if len(params) > 3 else None
        if len(params) > 4:
            raise TypeError(
                "Sig[dtype, shape, rate, units] takes at most four parameters"
            )
        return cls(_norm_dtype(dtype), _norm_shape(shape), rate, units)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", _norm_dtype(self.dtype))
        object.__setattr__(self, "shape", _norm_shape(self.shape))
        object.__setattr__(self, "rate", _unlit(self.rate))
        object.__setattr__(self, "units", _unlit(self.units))

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
            dtype, _ = _array_metadata(value)
            env.bind_dtype(self.dtype.name, dtype, where)
            if self.shape is not None:
                self.with_(dtype=dtype).check(value, env, where)
            return env
        if self.is_array:
            dtype, shape = _array_metadata(value)
            shape = _labelled_shape(value, self.shape, shape)
            target: Any = self.dtype
            if not np.can_cast(dtype, target, casting="same_kind"):
                raise TypeMismatch(f"{where}: value dtype {dtype} is not {self.dtype}")
            if self.shape is not None:
                if len(shape) != len(self.shape):
                    raise TypeMismatch(
                        f"{where}: value has rank {len(shape)}, type requires {len(self.shape)}"
                    )
                for i, (d, n) in enumerate(zip(self.shape, shape)):
                    if isinstance(d, int) and d != n:
                        raise TypeMismatch(
                            f"{where}: axis {i} has length {n}, expected {d}"
                        )
                    if isinstance(d, str):
                        env.bind(d, int(n), where)
        elif isinstance(self.dtype, type):
            cls: Any = self.dtype
            if not isinstance(value, cls):
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


if TYPE_CHECKING:
    # ``Sig[...]`` is a *contract*: what a block body returns for it may be an array, a
    # NumPy scalar, a float, a dict or a Frame, so to a checker it is ``Any`` carrying the
    # DSL's parameters. Signatures then type-check as written (``Sig[float, N]``,
    # ``Sig[T, tuple[M, N]]``, ``Sig[float, N, None, Literal["V"]]``) and bodies are not
    # second-guessed; at runtime ``Sig`` is :class:`SignalType`, whose subscription
    # builds the contract. ``Stream[X]`` is an iterable of X to a checker.
    from typing import Generic

    from typing_extensions import TypeAliasType, TypeVarTuple, Unpack

    _Ts = TypeVarTuple("_Ts")

    Sig = TypeAliasType("Sig", Any, type_params=(_Ts,))

    class Stream(Generic[Unpack[_Ts]]):
        def __iter__(self) -> Any: ...
        def __aiter__(self) -> Any: ...
else:
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
        bound: Env = env if env is not None else Env()
        if not isinstance(value, tuple) or len(value) != len(self.items):
            raise TypeMismatch(f"{where}: expected a {len(self.items)}-tuple")
        for i, (t, v) in enumerate(zip(self.items, value)):
            bound = t.check(v, bound, f"{where}[{i}]")
        return bound

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
            or type(item).__name__ in ("Framed", "_FramedMarker")
        ):
            return cls(item)
        return cls(_as_type(item))

    def __str__(self) -> str:
        return f"Stream[{self.item}]"

    __repr__ = __str__


if not TYPE_CHECKING:
    Stream = StreamType


def is_signal_type(obj: Any) -> bool:
    return isinstance(obj, (SignalType, TupleType))


def _as_type(obj: Any) -> Any:
    """Coerce loose specs into a type object."""
    if isinstance(obj, (SignalType, TupleType, StreamType)):
        return obj
    if obj is Any or obj is None:
        return SignalType()
    if get_origin(obj) is tuple:  # tuple[Sig[...], Sig[...]]
        obj = get_args(obj)
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
    if produced is Any or required is Any:
        return  # gradual dtype information must not bind a variable to Any
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

    if required.shape is not None and produced.shape is not None:
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


def narrow(produced: Any, required: Any, where: str = "") -> Any:
    """The contract carried by a node that passes its input through unchanged
    (``required`` is its declared input) once ``produced`` is connected to it:
    ``produced`` wherever it says something, ``required`` where ``produced`` is
    gradual (``Any`` dtype, unknown axes, no rate or units). Unification bindings
    and repeated-variable relationships are retained across the entire contract.
    Raises :class:`TypeMismatch` when ``produced`` does not satisfy ``required``.
    """
    produced, required = _as_type(produced), _as_type(required)
    env = unify(produced, required, Env(), where)

    def dimension(d: Dim) -> Dim:
        if not isinstance(d, str):
            return d
        bound = env.value(d)
        return bound if bound is not None else env.find(d)

    def merge(p: Any, r: Any) -> Any:
        if isinstance(p, TupleType) and isinstance(r, TupleType):
            return TupleType(tuple(merge(a, b) for a, b in zip(p.items, r.items)))
        if not (isinstance(p, SignalType) and isinstance(r, SignalType)):
            return p
        dtype = r.dtype if p.dtype is Any else p.dtype
        if isinstance(dtype, DVar):
            bound = env.value(dtype.name)
            dtype = bound if bound is not None else DVar(env.find(dtype.name))
        if p.shape is None:
            shape = r.shape
        elif r.shape is None:
            shape = p.shape
        else:
            shape = tuple(a if a is not None else b for a, b in zip(p.shape, r.shape))
        if shape is not None:
            shape = tuple(dimension(d) for d in shape)
        return SignalType(
            dtype,
            shape,
            p.rate if p.rate is not None else r.rate,
            p.units if p.units is not None else r.units,
            p.meta | r.meta,
        )

    return merge(produced, required)


def resolve(t: Any, env: Env) -> Any:
    """``t`` with the symbols ``env`` has bound replaced by their values: a dtype
    variable by its dtype, a symbolic dimension by its size. Unbound symbols stay."""
    if isinstance(t, TupleType):
        return TupleType(tuple(resolve(i, env) for i in t.items))
    if isinstance(t, StreamType):
        return StreamType(resolve(t.item, env))
    if not isinstance(t, SignalType):
        return t
    dtype = t.dtype
    if isinstance(dtype, DVar):
        bound = env.value(dtype.name)
        if bound is not None:
            dtype = bound
    shape = t.shape
    if shape is not None:
        shape = tuple(
            env.value(d) if isinstance(d, str) and env.value(d) is not None else d
            for d in shape
        )
    if dtype is t.dtype and shape == t.shape:
        return t
    return t.with_(dtype=dtype, shape=shape)


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

Scalar = Sig[float, tuple[()]]
Vector = Sig[float, tuple[N]]
Matrix = Sig[float, tuple[M, N]]
Tensor = Sig[float, None]
CVector = Sig[complex, tuple[N]]
CMatrix = Sig[complex, tuple[M, N]]
