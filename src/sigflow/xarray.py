"""Adapt synchronous array functions to labeled DataArrays without coercing data."""

from __future__ import annotations

import inspect
import sys
from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any, Literal, ParamSpec, TypeVar, cast

from .block import BlockDef

__all__ = ["xarray_aware"]

P = ParamSpec("P")
R = TypeVar("R")


def _dimensions(groups: Sequence[Sequence[str]], name: str) -> tuple[tuple[str, ...], ...]:
    if not groups or isinstance(groups, str):
        raise ValueError(f"{name} must contain at least one dimension list")
    result = []
    for group in groups:
        if isinstance(group, str) or any(not isinstance(d, str) or not d for d in group):
            raise ValueError(f"{name} must contain lists of nonempty dimension names")
        if len(set(group)) != len(group):
            raise ValueError(f"{name} contains duplicate dimensions")
        result.append(tuple(group))
    return tuple(result)


def xarray_aware(
    *,
    input_core_dims: Sequence[Sequence[str]],
    output_core_dims: Sequence[Sequence[str]],
    vectorize: bool = False,
    keep_attrs: bool = True,
    join: Literal["exact", "inner", "outer", "left", "right"] = "exact",
    dask: Literal["forbidden", "allowed"] = "forbidden",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Let an array function accept xarray DataArrays via ``xr.apply_ufunc``.

    Apply *under* ``@block``. The first ``len(input_core_dims)`` positional
    parameters are array inputs; remaining parameters are configuration and are
    forwarded unchanged, including keyword-only and array-valued configuration.
    Calls without DataArrays go straight to the original function and need no
    xarray installation.

    Named core dimensions are moved to trailing axes inside the numerical
    function. Support leading batch axes, or use ``vectorize=True`` to call a
    NumPy-only function one core slice at a time. Outputs are transposed back to
    surviving input dimension order (first input first, then additional inputs).
    Coordinates are retained by xarray and attributes are kept by default.

    Same-size core outputs, reductions (``[]``), broadcasting and multiple
    outputs are supported. New output dimensions and resizing existing dimensions
    are not supported. Coordinate alignment defaults to ``join="exact"``.
    Dask arrays are rejected unless ``dask="allowed"`` explicitly delegates lazy
    execution to a compatible function. No CPU/GPU transfer is performed.
    """
    inputs = _dimensions(input_core_dims, "input_core_dims")
    outputs = _dimensions(output_core_dims, "output_core_dims")
    input_dimensions = {d for group in inputs for d in group}
    output_dimensions = {d for group in outputs for d in group}
    if output_dimensions - input_dimensions:
        raise ValueError("new output dimensions require a coordinate-aware wrapper")
    if join not in ("exact", "inner", "outer", "left", "right"):
        raise ValueError("unsupported coordinate join")
    if dask not in ("forbidden", "allowed"):
        raise ValueError("dask must be 'forbidden' or 'allowed'")
    if vectorize and dask == "allowed":
        raise ValueError("vectorize=True requires eager NumPy data, not dask='allowed'")

    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        if isinstance(fn, BlockDef):
            raise TypeError("place @block above @xarray_aware")
        if (
            inspect.isgeneratorfunction(fn)
            or inspect.isasyncgenfunction(fn)
            or inspect.iscoroutinefunction(fn)
        ):
            raise TypeError("xarray_aware supports synchronous array functions only")
        signature = inspect.signature(fn)
        positional = [
            p.name
            for p in signature.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional) < len(inputs):
            raise TypeError("input_core_dims exceeds the number of positional parameters")
        names = positional[: len(inputs)]

        @wraps(fn)
        def adapted(*args: P.args, **kwargs: P.kwargs) -> R:
            # A DataArray cannot exist without xarray already being imported. This
            # also recognizes subclasses without importing xarray for NumPy calls.
            xr: Any = sys.modules.get("xarray")
            if xr is None:
                return fn(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = [bound.arguments[name] for name in names]
            if any(isinstance(v, (xr.Dataset, xr.Variable)) for v in values):
                raise TypeError(
                    "xarray_aware accepts DataArray inputs, not Dataset or Variable"
                )
            labeled = [v for v in values if isinstance(v, xr.DataArray)]
            if not labeled:
                return fn(*args, **kwargs)
            if vectorize and any(
                hasattr(
                    v.data if isinstance(v, xr.DataArray) else v,
                    "__cuda_array_interface__",
                )
                for v in values
            ):
                raise TypeError("vectorize=True is NumPy-only; use a batched GPU function")

            def calculate(*arrays: Any) -> Any:
                arguments = {**bound.arguments, **dict(zip(names, arrays))}
                call = inspect.BoundArguments(signature, arguments)
                return fn(*call.args, **call.kwargs)

            result = xr.apply_ufunc(
                calculate,
                *values,
                input_core_dims=inputs,
                output_core_dims=outputs,
                vectorize=vectorize,
                keep_attrs=keep_attrs,
                join=join,
                dask=dask,
            )
            order = list(dict.fromkeys(d for value in labeled for d in value.dims))

            def restore(value: Any) -> Any:
                dims = [d for d in order if d in value.dims]
                dims += [d for d in value.dims if d not in dims]
                return value.transpose(*dims)

            if isinstance(result, tuple):
                return cast(R, tuple(restore(value) for value in result))
            return cast(R, restore(result))

        return adapted

    return decorate
