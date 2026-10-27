"""Labeled adapters: dimension mapping, configuration, alignment and graph use."""

from __future__ import annotations

import inspect
import sys

import numpy as np
import pytest

from sigflow import Frame, Sig, block, from_iterable, run, xarray_aware


@pytest.fixture
def xr():
    return pytest.importorskip("xarray")


@block
@xarray_aware(input_core_dims=[["time"]], output_core_dims=[["time"]])
def center(x: Sig[float], gain: float = 1.0) -> Sig[float]:
    """Remove each channel's DC component along the last axis."""
    return (x - np.mean(x, axis=-1, keepdims=True)) * gain


@block
@xarray_aware(input_core_dims=[["time"]], output_core_dims=[[]])
def rms(x: Sig[float]) -> Sig[float]:
    return np.sqrt(np.mean(x**2, axis=-1))


def signal(xr):
    return xr.DataArray(
        np.arange(24, dtype=float).reshape(8, 3),
        dims=("time", "channel"),
        coords={"time": np.arange(8) / 1000, "channel": ["a", "b", "c"]},
        name="voltage",
        attrs={"units": "V"},
    )


def test_numpy_calls_without_xarray(monkeypatch):
    monkeypatch.setitem(sys.modules, "xarray", None)
    x = np.arange(8.0)
    np.testing.assert_array_equal(center(x, gain=2), (x - x.mean()) * 2)
    assert list(inspect.signature(center.fn).parameters) == ["x", "gain"]
    assert center.name == "center" and "Remove" in center.__doc__
    assert [spec.name for spec in center.inputs] == ["x"]
    assert list(center.config) == ["gain"]


@pytest.mark.parametrize("transposed", [False, True])
def test_named_axis_and_original_dimension_order(xr, transposed):
    x = signal(xr)
    if transposed:
        x = x.transpose("channel", "time")
    expected = (x - x.mean("time")) * 2
    expected.attrs = dict(x.attrs)
    xr.testing.assert_identical(center(x=x, gain=2), expected)
    assert center(x).dims == x.dims


def test_reduction_and_graph_annotations(xr):
    x = signal(xr)
    expected = np.sqrt(((x - x.mean("time")) ** 2).mean("time"))
    expected.attrs = dict(x.attrs)
    graph = center >> rms
    # Direct graph input is one DataArray, not a sequence of time samples.
    out = graph(x, check="all", on_error="raise")
    assert len(out) == 1
    xr.testing.assert_identical(out[0].data, expected)
    f = Frame(x, meta={"seq": 4}, stream={"rate": 1000}).with_issue("calibration")
    out = run(from_iterable([f]) >> graph, check="all", on_error="raise")[0]
    assert out.meta == f.meta and out.stream == f.stream and out.issues == f.issues
    assert "center" in out.prov and "rms" in out.prov


def test_configuration_binding_positional_only_and_keyword_only(xr):
    @xarray_aware(input_core_dims=[["time"]], output_core_dims=[["time"]])
    def weighted(x, /, weights, *, gain=1):
        return x * weights * gain

    x = signal(xr)
    weights = np.arange(8.0)
    expected = x * xr.DataArray(weights, dims="time") * 3
    expected.name = x.name
    expected.attrs = dict(x.attrs)
    xr.testing.assert_identical(weighted(x, weights, gain=3), expected)


def test_exact_alignment_and_multi_input_block(xr):
    @block
    @xarray_aware(input_core_dims=[["time"], ["time"]], output_core_dims=[["time"]])
    def add(x: Sig[float], y: Sig[float], gain: float = 1) -> Sig[float]:
        return x + y * gain

    x = signal(xr)
    out = run(
        from_iterable([(x, x)], type=add.in_type) >> add(gain=2),
        check="all", on_error="raise",
    )
    expected = x * 3
    expected.attrs = dict(x.attrs)
    xr.testing.assert_identical(out[0].data, expected)
    shifted = x.assign_coords(time=x.time + 1)
    with pytest.raises(ValueError, match="align|exact"):
        add(x, shifted)


def test_broadcasting_and_explicit_alignment(xr):
    @xarray_aware(input_core_dims=[[], []], output_core_dims=[[]], join="inner")
    def add(x, y):
        return x + y

    x = xr.DataArray([1.0, 2.0, 3.0], dims="time", coords={"time": [0, 1, 2]})
    y = xr.DataArray([4.0, 5.0, 6.0], dims="time", coords={"time": [1, 2, 3]})
    xr.testing.assert_identical(add(x, y), x + y)
    z = xr.DataArray([10.0, 20.0], dims="channel", coords={"channel": ["a", "b"]})
    xr.testing.assert_identical(add(x, z), x + z)


def test_vectorized_one_dimensional_function(xr):
    @xarray_aware(
        input_core_dims=[["time"]], output_core_dims=[["time"]], vectorize=True
    )
    def smooth(x):
        assert x.ndim == 1
        return np.convolve(x, np.ones(3) / 3, mode="same")

    x = signal(xr)
    expected = x.copy(data=np.apply_along_axis(smooth, 0, x.data))
    xr.testing.assert_identical(smooth(x), expected)


def test_multiple_outputs_and_attribute_policy(xr):
    @block
    @xarray_aware(
        input_core_dims=[["time"]], output_core_dims=[[], []], keep_attrs=False
    )
    def stats(x: Sig[float]) -> tuple[Sig[float], Sig[float]]:
        return np.mean(x, axis=-1), np.max(x, axis=-1)

    x = signal(xr)
    mean, peak = stats(x)
    expected_mean, expected_peak = x.mean("time"), x.max("time")
    expected_mean.attrs, expected_peak.attrs = {}, {}
    xr.testing.assert_identical(mean, expected_mean)
    xr.testing.assert_identical(peak, expected_peak)
    frame = run(from_iterable([x]) >> stats, check="all", on_error="raise")[0]
    xr.testing.assert_identical(frame.data[0], expected_mean)
    xr.testing.assert_identical(frame.data[1], expected_peak)


def test_missing_or_resized_dimensions_and_unsupported_payloads(xr):
    x = signal(xr)
    with pytest.raises(ValueError, match="time|tuple.index"):
        center(x.rename(time="sample"))
    with pytest.raises(TypeError, match="DataArray"):
        center(x.to_dataset())
    with pytest.raises(TypeError, match="DataArray"):
        center(x.variable)

    @xarray_aware(input_core_dims=[["time"]], output_core_dims=[["time"]])
    def shorten(x):
        return x[..., :-1]

    with pytest.raises(ValueError, match="size|dimension"):
        shorten(x)


def test_invalid_decorator_usage():
    with pytest.raises(ValueError, match="new output dimensions"):
        xarray_aware(input_core_dims=[["time"]], output_core_dims=[["frequency"]])
    with pytest.raises(ValueError, match="duplicate"):
        xarray_aware(input_core_dims=[["time", "time"]], output_core_dims=[[]])
    with pytest.raises(ValueError, match="lists"):
        xarray_aware(input_core_dims=["time"], output_core_dims=[[]])
    with pytest.raises(ValueError, match="eager NumPy"):
        xarray_aware(
            input_core_dims=[["time"]], output_core_dims=[[]],
            vectorize=True, dask="allowed",
        )
    with pytest.raises(TypeError, match="@block above"):
        xarray_aware(input_core_dims=[["time"]], output_core_dims=[[]])(center)
    with pytest.raises(TypeError, match="positional"):
        xarray_aware(input_core_dims=[["time"], ["time"]], output_core_dims=[[]])(lambda x: x)

    def generator(x):
        yield x

    with pytest.raises(TypeError, match="synchronous"):
        xarray_aware(input_core_dims=[["time"]], output_core_dims=[[]])(generator)


def test_lazy_arrays_when_available(xr):
    da = pytest.importorskip("dask.array")
    x = xr.DataArray(da.arange(8, chunks=4).astype(float), dims="time")
    with pytest.raises(ValueError, match="dask|chunked"):
        center(x)

    @xarray_aware(input_core_dims=[["time"]], output_core_dims=[[]], dask="allowed")
    def lazy_mean(x):
        return x.mean(axis=-1)

    result = lazy_mean(x)
    assert isinstance(result.data, da.Array)
    assert result.compute().item() == 3.5


def test_gpu_array_stays_on_device(xr):
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA unavailable")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(str(exc))

    @xarray_aware(input_core_dims=[["time"]], output_core_dims=[["time"]])
    def gpu_center(x):
        assert isinstance(x, cp.ndarray)
        return x - cp.mean(x, axis=-1, keepdims=True)

    x = signal(xr)
    result = gpu_center(x.copy(data=cp.asarray(x.data)))
    assert isinstance(result.data, cp.ndarray) and result.dims == x.dims
    np.testing.assert_allclose(cp.asnumpy(result.data), (x - x.mean("time")).data)

    vectorized = xarray_aware(
        input_core_dims=[["time"]], output_core_dims=[["time"]], vectorize=True
    )(lambda x: x)
    with pytest.raises(TypeError, match="NumPy-only"):
        vectorized(result)
