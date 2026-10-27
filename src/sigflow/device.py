"""Explicit, synchronous CPU/CUDA transfers (CuPy is an optional dependency)."""

from __future__ import annotations

import sys
from importlib import import_module
from typing import Any, Callable

import numpy as np

from .block import block
from .dims import T
from .types import Sig

__all__ = ["to_device", "to_host"]


def _cupy() -> Any:
    try:
        return import_module("cupy")
    except ImportError as exc:
        raise ImportError(
            "CUDA transfers require CuPy. Install a CuPy package matching your CUDA "
            "installation, for example: pip install cupy-cuda12x"
        ) from exc


def _payload(x: Any, convert: Callable[[Any], Any]) -> Any:
    # Import xarray only when processing one of its objects; CPU/GPU arrays do
    # not require it. Copy the wrapper so dimensions, coordinates and attrs survive.
    xr = sys.modules.get("xarray")
    if xr is not None and isinstance(x, (xr.DataArray, xr.Dataset, xr.Variable)):
        if not isinstance(x, xr.DataArray):
            raise TypeError("device transfers accept xarray.DataArray, not Dataset")
        return x.copy(deep=False, data=convert(x.data))
    return convert(x)


@block(keep_samples=True)
def to_device(x: Sig[T], device: int | str = 0, copy: bool = False) -> Sig[T]:
    """Move sample data to a CUDA device, preserving dtype, shape and annotations.

    ``device`` is a nonnegative index or ``"cuda:N"``. Requires an independently
    installed CuPy package compatible with CUDA. An array already on the requested
    device may be reused unless ``copy=True``. The current CUDA stream is
    synchronized before returning. For xarray DataArrays only the payload moves;
    coordinates and frame sample annotations retain their existing placement.
    """
    if isinstance(device, str):
        if not device.startswith("cuda:") or not device[5:].isdigit():
            raise ValueError("device must be a nonnegative index or 'cuda:N'")
        device = int(device[5:])
    if not isinstance(device, int) or isinstance(device, bool) or device < 0:
        raise ValueError("device must be a nonnegative index or 'cuda:N'")
    cp = _cupy()
    with cp.cuda.Device(device):

        def convert(data: Any) -> Any:
            return cp.array(data, copy=True) if copy else cp.asarray(data)

        out = _payload(x, convert)
        cp.cuda.get_current_stream().synchronize()
    return out


@block(keep_samples=True)
def to_host(x: Sig[T], copy: bool = False) -> Sig[T]:
    """Move CUDA sample data to a NumPy array, blocking until the copy completes.

    CPU arrays are reused unless ``copy=True`` and do not require CuPy. xarray
    DataArray labels and attributes are preserved. Frame sample annotations are
    preserved without changing their device placement.
    """

    def convert(data: Any) -> np.ndarray:
        if hasattr(data, "__cuda_array_interface__"):
            cp = _cupy()
            if not isinstance(data, cp.ndarray):
                raise TypeError("to_host expects a CuPy array; convert to CuPy first")
            # Use the source device even when a different GPU is currently active.
            with cp.cuda.Device(data.device.id):
                return cp.asnumpy(data, blocking=True)
        array = np.asarray(data)
        return array.copy() if copy else array

    return _payload(x, convert)
