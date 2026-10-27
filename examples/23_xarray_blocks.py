"""Ordinary numerical functions adapted to named xarray dimensions.

Run ``python examples/23_xarray_blocks.py`` after installing ``sigflow[xarray]``.
Without xarray the script skips the labeled example. The function bodies use
NumPy-style trailing-axis operations; the decorator maps a named ``time`` axis,
preserves coordinates, and restores the original dimension order. A reduction
then produces one RMS value per labeled channel. Both functions remain directly
callable with ordinary NumPy arrays.
"""

from importlib import import_module

import numpy as np

from sigflow import Sig, block, xarray_aware


@block
@xarray_aware(input_core_dims=[["time"]], output_core_dims=[["time"]])
def remove_dc(x: Sig[float], gain: float = 1.0) -> Sig[float]:
    """Last axis is time inside the function, regardless of the labeled layout."""
    return gain * (x - np.mean(x, axis=-1, keepdims=True))


@block
@xarray_aware(input_core_dims=[["time"]], output_core_dims=[[]])
def rms(x: Sig[float]) -> Sig[float]:
    """Reduce time; all broadcast dimensions and their labels survive."""
    return np.sqrt(np.mean(x**2, axis=-1))


def main() -> None:
    # NumPy calls are unchanged: there is no dimension information, so the
    # caller arranges time on the trailing axis as the function expects.
    np.testing.assert_allclose(remove_dc(np.arange(4.0)), [-1.5, -0.5, 0.5, 1.5])
    try:
        xr = import_module("xarray")
    except ImportError:
        print('Labeled example skipped: install with pip install -e ".[xarray]".')
        return

    time = np.arange(1000) / 1000
    samples = np.column_stack([
        3 + np.sin(2 * np.pi * 25 * time),
        -2 + 2 * np.sin(2 * np.pi * 50 * time),
    ])
    signal = xr.DataArray(
        samples,
        dims=("time", "channel"),
        coords={"time": time, "channel": ["left", "right"]},
        name="voltage",
        attrs={"units": "V"},
    )
    centered = remove_dc(signal)
    assert centered.dims == signal.dims
    xr.testing.assert_equal(centered.time, signal.time)
    xr.testing.assert_identical(
        remove_dc(signal.transpose("channel", "time")),
        centered.transpose("channel", "time"),
    )

    # Sig[float] checks dtype but leaves rank gradual: the labeled payload here
    # is 2-D even though the operation has only one core dimension (time).
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
    print("\nProvenance:")
    print(output.trace())


if __name__ == "__main__":
    main()
