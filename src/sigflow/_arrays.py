"""One place to recognise array-likes, labelled and device arrays.

Nothing here imports an optional backend: a ``DataArray`` or a CuPy array can only
exist once its library has been imported by the caller, so ``sys.modules`` is
enough to find the class to test against.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import numpy as np


def is_array(value: Any) -> bool:
    """Anything with ``shape``, ``dtype`` and ``ndim`` (NumPy, CuPy, dask, DataArray...)."""
    return all(hasattr(value, attr) for attr in ("shape", "dtype", "ndim"))


def is_cuda_array(value: Any) -> bool:
    """A device array of any library that speaks the CUDA array interface."""
    return hasattr(value, "__cuda_array_interface__")


def array_metadata(value: Any) -> tuple[np.dtype, tuple[int, ...]]:
    """``(dtype, shape)`` of an array without materialising it (CUDA, dask, ...)."""
    if hasattr(value, "dtype") and hasattr(value, "shape"):
        try:
            return np.dtype(value.dtype), tuple(int(n) for n in value.shape)
        except (TypeError, ValueError):
            pass  # torch / pandas extension dtypes, unknown lazy sizes: materialise
    array = np.asarray(value)
    return array.dtype, array.shape


def xarray_module() -> Any:
    """The ``xarray`` module if the caller has imported it, else ``None``."""
    return sys.modules.get("xarray")


def is_data_array(value: Any) -> bool:
    xr = xarray_module()
    return xr is not None and isinstance(value, xr.DataArray)


def is_xarray_container(value: Any) -> bool:
    """A ``Dataset`` or ``Variable``: labelled, but not a single array payload."""
    xr = xarray_module()
    return xr is not None and isinstance(value, (xr.Dataset, xr.Variable))


def is_positional(value: Any) -> bool:
    """An array whose ``[]`` indexes by position (so ``value[key]`` can be used
    directly, keeping labels and device placement). pandas objects index by label."""
    return (
        is_array(value) and hasattr(value, "__getitem__") and not hasattr(value, "iloc")
    )


def map_payload(value: Any, convert: Callable[[Any], Any]) -> Any:
    """Apply ``convert`` to the bare array of ``value``: the payload of a ``DataArray``
    (dims, coords, name and attrs are kept), or the value itself."""
    if is_data_array(value):
        return value.copy(deep=False, data=convert(value.data))
    if is_xarray_container(value):
        raise TypeError(
            "device transfers accept xarray.DataArray, not Dataset or Variable"
        )
    return convert(value)
