"""Labelled payloads: ``xarray.DataArray`` frames flow through blocks by contract.

A block body is an ordinary array function. When a block is *label-aware*
and an input arrives as a ``DataArray``, the wrapper installed by
:class:`~sigflow.block.BlockDef` (``defn.adapted``):

1. lines the array up with the input contract: when every dimension of
   ``Sig[float, tuple[CHANNEL, TIME]]`` names a dimension of the array, it is
   transposed into that order; otherwise axes match by position, as they do for
   a plain array;
2. hands the body the bare payload (``da.data``, so CuPy or dask arrays stay
   what they are) and remembers which *dimension variable* carried which
   label (``N`` was ``"time"``, with its coordinate);
3. rebuilds a ``DataArray`` around each output from the output contract: a
   variable seen on the way in keeps its name and coordinate, a new variable
   contributes its own name, fixed sizes become ``dim_<i>``; attributes and the
   array name come from the first labelled input. When the output contract
   names every dimension, the result is transposed back to the caller's
   layout; a positional contract keeps the contract's axis order, which is
   what its runtime checks expect.

Reductions (``tuple[M, N] -> tuple[M]``), broadcasts, several outputs and
tuple frames follow from the contract. A contract without a shape
(``Sig[float]``) keeps the input's dimensions when the rank is unchanged and
otherwise returns a plain array. Scalars (rank 0) are returned as they are.
Inputs that take the frame (``Framed[...]``) are passed untouched, so a body
that wants the labels reads them from ``frame.data``.

Calls without any ``DataArray`` take a fast path straight to the body, and
nothing here imports xarray: a ``DataArray`` can only exist once xarray has
been imported by the caller.

Whether a block is label-aware is its ``labels`` option (``@block(labels=True)``
or ``labels=False``). A block that leaves it unset follows the run:
``run(graph, labels=True)``, or, when the run does not say, the package default
set with :func:`default_labels`. A run reads the default once when it starts
(``RunOptions``), so changing it while a run is in progress, from another
thread or in a process-pool worker, does not change which body a block
executes: the resolved flag travels with the run. A block called directly
(``defn(x)``) reads the default at call time.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Iterable

from ._arrays import xarray_module
from .types import SignalType, TupleType

__all__ = ["default_labels", "current_default", "adapt"]

_DEFAULT = False


def default_labels(enabled: bool = True) -> bool:
    """Make every block whose ``labels`` option is unset label-aware (or not) in
    runs that do not pass ``labels=`` themselves and in direct calls.

    Returns the previous setting. Blocks declared with ``labels=True`` or
    ``labels=False`` are unaffected. A run captures the setting when it starts.
    """
    global _DEFAULT
    previous, _DEFAULT = _DEFAULT, bool(enabled)
    return previous


def current_default() -> bool:
    """The package default for blocks whose ``labels`` option is unset."""
    return _DEFAULT


class _Labels:
    """What a block's labelled inputs told us during one call."""

    __slots__ = ("dims", "refs", "order")

    def __init__(self) -> None:
        self.dims: dict[str, str] = {}  # dimension variable -> dimension name
        self.refs: list[Any] = []  # labelled inputs, in order of appearance
        self.order: tuple[str, ...] = ()  # dimension order of the first input as given

    @property
    def active(self) -> bool:
        return bool(self.refs)


# --------------------------------------------------------------------------- #
# unwrap: DataArray -> payload, recording labels
# --------------------------------------------------------------------------- #


def _unwrap(value: Any, t: Any, labels: _Labels, xr: Any) -> Any:
    if isinstance(value, xr.DataArray):
        return _unwrap_array(value, t, labels)
    if (
        isinstance(value, tuple)
        and isinstance(t, TupleType)
        and len(t.items) == len(value)
    ):
        return tuple(_unwrap(v, it, labels, xr) for v, it in zip(value, t.items))
    return value


def _unwrap_array(da: Any, t: Any, labels: _Labels) -> Any:
    shape = t.shape if isinstance(t, SignalType) else None
    if not labels.refs:
        labels.order = tuple(da.dims)
    if shape is not None and len(shape) == da.ndim:
        if (
            all(isinstance(d, str) for d in shape)
            and set(shape) <= set(da.dims)
            and tuple(shape) != tuple(da.dims)
        ):
            da = da.transpose(*shape)
        for d, name in zip(shape, da.dims):
            if isinstance(d, str):
                labels.dims.setdefault(d, name)
    labels.refs.append(da)
    return da.data


# --------------------------------------------------------------------------- #
# rewrap: payload -> DataArray, from the output contract
# --------------------------------------------------------------------------- #


def _rewrap(value: Any, t: Any, labels: _Labels, xr: Any) -> Any:
    if value is None or isinstance(value, xr.DataArray):
        return value
    if isinstance(value, tuple):
        items = (
            t.items if isinstance(t, TupleType) and len(t.items) == len(value) else None
        )
        return tuple(
            _rewrap(v, items[i] if items else None, labels, xr)
            for i, v in enumerate(value)
        )
    if not (hasattr(value, "shape") and hasattr(value, "dtype")):
        return value  # scalars, dicts, frames: not ours to label
    ndim = len(value.shape)
    if ndim == 0:
        return value
    dims = _output_dims(ndim, t, labels)
    if dims is None:
        return value
    ref = labels.refs[0]
    out = xr.DataArray(
        value,
        dims=dims,
        coords=_coords(dims, tuple(value.shape), labels),
        attrs=dict(ref.attrs),
        name=ref.name,
    )
    shape = t.shape if isinstance(t, SignalType) else None
    if dims != tuple(shape or ()):
        return out  # positional contract: keep the contract's axis order
    # a fully named contract is checked by name in any layout, so go back to the
    # caller's: dimensions the first input had keep their order, new ones follow
    order = [d for d in labels.order if d in dims] + [
        d for d in dims if d not in labels.order
    ]
    return out if tuple(order) == dims else out.transpose(*order)


def _output_dims(ndim: int, t: Any, labels: _Labels) -> tuple[str, ...] | None:
    shape = t.shape if isinstance(t, SignalType) else None
    if shape is None:
        ref = labels.refs[0]
        return tuple(ref.dims) if ref.ndim == ndim else None
    if len(shape) != ndim:
        return None
    dims: list[str] = []
    for i, d in enumerate(shape):
        if isinstance(d, str):
            name = labels.dims.get(d, d)
        else:
            name = f"dim_{i}"
        while name in dims:
            name = f"{name}_{i}"
        dims.append(name)
    return tuple(dims)


def _coords(dims: tuple[str, ...], shape: tuple[int, ...], labels: _Labels) -> dict:
    size = dict(zip(dims, shape))
    coords: dict[str, Any] = {}
    for ref in labels.refs:
        for key, coord in ref.coords.items():
            if key in coords or not set(coord.dims) <= set(dims):
                continue
            if all(coord.sizes[d] == size[d] for d in coord.dims):
                coords[key] = coord.variable
    return coords


# --------------------------------------------------------------------------- #
# the wrapper installed on a block definition
# --------------------------------------------------------------------------- #


def _has_labelled(values: Iterable[Any], xr: Any) -> bool:
    for v in values:
        if isinstance(v, xr.DataArray):
            return True
        if isinstance(v, tuple) and any(isinstance(x, xr.DataArray) for x in v):
            return True
    return False


def _stream(items: Any, t: Any, labels: _Labels, xr: Any) -> Any:
    """Unwrap the items of a stream input lazily. The labels follow the latest item:
    each item replaces the references the previous one left (the other, non-stream
    inputs were unwrapped before the body started and stay), so a long stream
    does not pin every array it has seen."""
    base: tuple[int, dict[str, str], tuple[str, ...]] | None = None

    def take(item: Any) -> Any:
        nonlocal base
        if base is None:
            base = len(labels.refs), dict(labels.dims), labels.order
        count, dims, order = base
        del labels.refs[count:]
        # Keep static inputs, but discard the preceding stream item's dimension
        # bindings and layout as well as its array reference. A renamed axis must
        # not inherit a stale symbol -> name association from the first item.
        labels.dims = dict(dims)
        labels.order = order
        return _unwrap(item, t, labels, xr)

    if hasattr(items, "__aiter__"):

        async def agen() -> Any:
            async for item in items:
                yield take(item)

        return agen()

    def gen() -> Any:
        for item in items:
            yield take(item)

    return gen()


def adapt(defn: Any) -> Callable[..., Any]:
    """Return ``defn.fn`` wrapped so labelled inputs are unwrapped and outputs
    rewrapped according to the block's contract (see the module docstring).
    Installed as ``defn.adapted``; ``defn.fn`` stays the raw body, and
    ``defn.resolve_call(default)`` picks one of the two for a block."""
    fn: Callable[..., Any] = defn.fn
    inputs = list(defn.inputs)
    if not inputs:
        return fn
    try:
        signature: inspect.Signature | None = inspect.signature(fn)
    except (TypeError, ValueError):
        signature = None
    by_name = {spec.name: spec for spec in inputs}
    has_stream = any(spec.stream for spec in inputs)

    def prepare(
        args: tuple, kwargs: dict, labels: _Labels, xr: Any
    ) -> tuple[tuple, dict]:
        if signature is not None:
            try:
                bound = signature.bind(*args, **kwargs)
            except TypeError:
                return args, kwargs  # let the body raise the real error
            for name, spec in by_name.items():
                if name in bound.arguments and not spec.framed:
                    v = bound.arguments[name]
                    bound.arguments[name] = (
                        _stream(v, spec.type, labels, xr)
                        if spec.stream
                        else _unwrap(v, spec.type, labels, xr)
                    )
            return bound.args, bound.kwargs
        new_args = list(args)
        for i, spec in enumerate(inputs[: len(new_args)]):
            if not spec.framed:
                v = new_args[i]
                new_args[i] = (
                    _stream(v, spec.type, labels, xr)
                    if spec.stream
                    else _unwrap(v, spec.type, labels, xr)
                )
        return tuple(new_args), kwargs

    def wanted(args: tuple, kwargs: dict) -> Any:
        """The xarray module when this call should be adapted, else None."""
        xr = xarray_module()
        if xr is None:
            return None
        if has_stream or _has_labelled(args, xr) or _has_labelled(kwargs.values(), xr):
            return xr
        return None

    def out(value: Any, labels: _Labels, xr: Any) -> Any:
        return _rewrap(value, defn.out_type, labels, xr) if labels.active else value

    if inspect.isasyncgenfunction(fn):

        @functools.wraps(fn, updated=())
        async def agen_adapted(*args: Any, **kwargs: Any) -> Any:
            xr = wanted(args, kwargs)
            if xr is None:
                async for item in fn(*args, **kwargs):
                    yield item
                return
            labels = _Labels()
            args, kwargs = prepare(args, kwargs, labels, xr)
            async for item in fn(*args, **kwargs):
                yield out(item, labels, xr)

        return agen_adapted

    if inspect.isgeneratorfunction(fn):

        @functools.wraps(fn, updated=())
        def gen_adapted(*args: Any, **kwargs: Any) -> Any:
            xr = wanted(args, kwargs)
            if xr is None:
                yield from fn(*args, **kwargs)
                return
            labels = _Labels()
            args, kwargs = prepare(args, kwargs, labels, xr)
            for item in fn(*args, **kwargs):
                yield out(item, labels, xr)

        return gen_adapted

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn, updated=())
        async def coro_adapted(*args: Any, **kwargs: Any) -> Any:
            xr = wanted(args, kwargs)
            if xr is None:
                return await fn(*args, **kwargs)
            labels = _Labels()
            args, kwargs = prepare(args, kwargs, labels, xr)
            return out(await fn(*args, **kwargs), labels, xr)

        return coro_adapted

    @functools.wraps(fn, updated=())
    def adapted(*args: Any, **kwargs: Any) -> Any:
        xr = wanted(args, kwargs)
        if xr is None:
            return fn(*args, **kwargs)
        labels = _Labels()
        args, kwargs = prepare(args, kwargs, labels, xr)
        return out(fn(*args, **kwargs), labels, xr)

    return adapted
