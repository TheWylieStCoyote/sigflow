# Labelled arrays: ``xarray.DataArray`` frames flow through ordinary blocks

`examples/23_xarray_blocks.py`

Run ``python examples/23_xarray_blocks.py`` after installing ``sigflow[xarray]``.
Without xarray the script skips the labelled part. A block body is a NumPy
function written against its contract; when the input is a DataArray the block
lines it up with the contract (by dimension *name* when the contract names
dimensions, by position otherwise), hands the body the bare array and rebuilds a
labelled array around the result. Coordinates, attributes and the array name
survive, reductions keep the coordinates that still apply, and the caller's
dimension order is restored. Both functions remain directly callable with
ordinary NumPy arrays.

## Source

```python
"""Labelled arrays: ``xarray.DataArray`` frames flow through ordinary blocks.

Run ``python examples/23_xarray_blocks.py`` after installing ``sigflow[xarray]``.
Without xarray the script skips the labelled part. A block body is a NumPy
function written against its contract; when the input is a DataArray the block
lines it up with the contract (by dimension *name* when the contract names
dimensions, by position otherwise), hands the body the bare array and rebuilds a
labelled array around the result. Coordinates, attributes and the array name
survive, reductions keep the coordinates that still apply, and the caller's
dimension order is restored. Both functions remain directly callable with
ordinary NumPy arrays.
"""

from importlib import import_module
from typing import Literal

import numpy as np

from sigflow import Sig, block, dsp, from_iterable, run
from sigflow.dims import M, N

CHANNEL = Literal["channel"]
TIME = Literal["time"]


@block(labels=True)
def remove_dc(
    x: Sig[float, tuple[CHANNEL, TIME]], gain: float = 1.0
) -> Sig[float, tuple[CHANNEL, TIME]]:
    """Time is the trailing axis inside the body, whatever the labelled layout."""
    return gain * (x - np.mean(x, axis=-1, keepdims=True))


@block(labels=True)
def rms(x: Sig[float, tuple[CHANNEL, TIME]]) -> Sig[float, tuple[CHANNEL]]:
    """Reduce over time; the channel coordinate survives."""
    return np.sqrt(np.mean(x**2, axis=-1))


@block(labels=True)
def spectrum(x: Sig[float, tuple[M, N]]) -> Sig[complex, tuple[M, Literal[513]]]:
    """A positional contract: M keeps the name it had on the way in."""
    return np.fft.rfft(x, axis=1)


def main() -> None:
    # NumPy calls are unchanged: the caller arranges (channel, time) as the
    # contract says, and gets a plain array back.
    plain = np.arange(8.0)[None, :]
    np.testing.assert_allclose(remove_dc(plain), plain - 3.5)
    try:
        xr = import_module("xarray")
    except ImportError:
        print('Labelled example skipped: install with pip install -e ".[xarray]".')
        return

    time = np.arange(1024) / 1000
    samples = np.column_stack(
        [
            3 + np.sin(2 * np.pi * 25 * time),
            -2 + 2 * np.sin(2 * np.pi * 50 * time),
        ]
    )
    signal = xr.DataArray(
        samples,
        dims=("time", "channel"),
        coords={"time": time, "channel": ["left", "right"]},
        name="voltage",
        attrs={"units": "V"},
    )

    # --- direct calls: any layout in, the same layout out -------------------
    centered = remove_dc(signal)
    assert centered.dims == signal.dims
    xr.testing.assert_equal(centered.time, signal.time)
    xr.testing.assert_identical(
        remove_dc(signal.transpose("channel", "time")),
        centered.transpose("channel", "time"),
    )

    # --- in a graph, with runtime checks by dimension name ------------------
    graph = remove_dc(gain=2) >> rms
    output = graph(signal, check="all", on_error="raise")[0]
    expected = 2 * np.sqrt(np.mean((samples - samples.mean(axis=0)) ** 2, axis=0))
    np.testing.assert_allclose(output.data.data, expected)
    assert output.data.dims == ("channel",)
    assert output.data.attrs["units"] == "V"
    print("Input dimensions:", signal.dims)
    print("Output dimensions:", output.data.dims)
    for label, value in zip(output.data.channel.data, output.data.data):
        print(f"{label}: RMS = {value:.6f} V")

    # --- a positional contract keeps the labels it was given ----------------
    spec = spectrum(signal.transpose("channel", "time"))
    print("Spectrum dimensions:", spec.dims, "| coordinates:", list(spec.coords))

    # --- the run option covers the built-in library blocks ------------------
    # (``default_labels(True)`` sets the same thing package-wide)
    frame = run(
        from_iterable([signal.isel(channel=0)]) >> dsp.scale(gain=0.5),
        check="all",
        labels=True,
    )[0]
    print("dsp.scale on a DataArray:", frame.data.dims, frame.data.attrs)

    print("\nProvenance:")
    print(output.trace())


if __name__ == "__main__":
    main()
```
