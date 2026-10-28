"""Explicit, synchronous CPU/CUDA transfers (CuPy is an optional dependency).

A transfer is a barrier: when it returns, the data it hands on is complete.
Work a producer launched on *any* stream of the source device has finished
before a device array is read (the device is synchronised as a whole, since a
producer's stream is not known here), and the copy itself has finished before
the block returns. Nothing is overlapped; group work on one device to keep the
number of transfers down.

Both blocks are ``preserve_type`` blocks: placed after a producer they carry its
contract (shape, rate, units, required keys), so what a transfer does not change
is still checked downstream and reframing ratios still resolve.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np

from ._arrays import is_cuda_array, map_payload
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


def _parse_device(device: int | str) -> int:
    if isinstance(device, str):
        if device.startswith("cuda:") and device[5:].isdecimal():
            return int(device[5:])
        raise ValueError("device must be a nonnegative index or 'cuda:N'")
    if isinstance(device, bool) or not isinstance(device, (int, np.integer)):
        # one error, one message, for every malformed device spec
        raise ValueError("device must be a nonnegative index or 'cuda:N'")  # noqa: TRY004
    if device < 0:
        raise ValueError("device must be a nonnegative index or 'cuda:N'")
    return int(device)


def _settled(cp: Any, data: Any) -> Any:
    """``data`` as a CuPy array (a view for other CUDA-array-interface objects:
    torch, numba...) after everything running on its device has finished."""
    source = data if isinstance(data, cp.ndarray) else cp.asarray(data)
    with cp.cuda.Device(source.device.id):
        cp.cuda.runtime.deviceSynchronize()
    return source


@block(preserve_type=True)
def to_device(x: Sig[T], device: int | str = 0, copy: bool = False) -> Sig[T]:
    """Move sample data to a CUDA device, preserving dtype, shape and annotations.

    ``device`` is a nonnegative index or ``"cuda:N"``. Requires an independently
    installed CuPy package compatible with CUDA. Device arrays of other libraries
    (anything with ``__cuda_array_interface__``) are accepted and become CuPy
    arrays. An array already on the requested device is reused unless
    ``copy=True``. Whatever was running on the source device has finished before
    its data is read, and the copy has finished before the block returns. For
    xarray DataArrays only the payload moves; coordinates and frame sample
    annotations retain their existing placement.
    """
    target = _parse_device(device)
    cp = _cupy()

    def convert(data: Any) -> Any:
        if is_cuda_array(data):
            data = _settled(cp, data)
            if not copy and data.device.id == target:
                return data  # already there and complete: nothing to copy or wait for
        with cp.cuda.Device(target):
            out = cp.array(data, copy=True) if copy else cp.asarray(data)
            cp.cuda.get_current_stream().synchronize()
        return out

    return map_payload(x, convert)


@block(preserve_type=True)
def to_host(x: Sig[T], copy: bool = False) -> Sig[T]:
    """Move CUDA sample data to a NumPy array, blocking until the copy completes.

    CPU arrays are reused unless ``copy=True`` and do not require CuPy. Device
    arrays of any library with ``__cuda_array_interface__`` are accepted. Whatever
    was running on the source device has finished before its data is read. xarray
    DataArray labels and attributes are preserved. Frame sample annotations are
    preserved without changing their device placement.
    """

    def convert(data: Any) -> np.ndarray:
        if is_cuda_array(data):
            cp = _cupy()
            data = _settled(cp, data)
            # Use the source device even when a different GPU is currently active.
            with cp.cuda.Device(data.device.id):
                return cp.asnumpy(data, blocking=True)
        array = np.asarray(data)
        return array.copy() if copy else array

    return map_payload(x, convert)
