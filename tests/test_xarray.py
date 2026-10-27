"""Labelled payloads: DataArrays flow through blocks according to their contracts."""

from __future__ import annotations

import inspect
import sys
from typing import Iterator, Literal

import numpy as np
import pytest

import sigflow.xarray as sx
from sigflow import (
    Frame,
    Framed,
    Sig,
    Stream,
    block,
    default_labels,
    dsp,
    from_iterable,
    linalg,
    run,
)
from sigflow.dims import M, N, T

CHANNEL = Literal["channel"]
TIME = Literal["time"]


@pytest.fixture
def xr():
    return pytest.importorskip("xarray")


@pytest.fixture
def default_off():
    previous = default_labels(False)
    yield
    default_labels(previous)


@block(labels=True)
def center(
    x: Sig[float, tuple[CHANNEL, TIME]], gain: float = 1.0
) -> Sig[float, tuple[CHANNEL, TIME]]:
    """Remove each channel's DC component; time is the trailing axis by contract."""
    assert x.shape[-1] == 8  # the body sees (channel, time), whatever the layout
    return (x - np.mean(x, axis=-1, keepdims=True)) * gain


@block(labels=True)
def rms(x: Sig[float, tuple[CHANNEL, TIME]]) -> Sig[float, tuple[CHANNEL]]:
    return np.sqrt(np.mean(x**2, axis=-1))


def signal(xr):
    return xr.DataArray(
        np.arange(24, dtype=float).reshape(8, 3),
        dims=("time", "channel"),
        coords={"time": np.arange(8) / 1000, "channel": ["a", "b", "c"]},
        name="voltage",
        attrs={"units": "V"},
    )


def test_numpy_calls_take_the_fast_path(monkeypatch):
    monkeypatch.setitem(sys.modules, "xarray", None)
    x = np.zeros((3, 8))
    x[:, 0] = 1
    np.testing.assert_array_equal(center(x, gain=2), (x - x.mean(axis=-1)[:, None]) * 2)
    assert list(inspect.signature(center.fn).parameters) == ["x", "gain"]
    assert center.name == "center" and "DC" in center.__doc__
    assert [spec.name for spec in center.inputs] == ["x"]
    assert list(center.config) == ["gain"] and center.labels is True


@pytest.mark.parametrize("transposed", [False, True])
def test_named_contract_matches_any_layout(xr, transposed):
    x = signal(xr)
    if transposed:
        x = x.transpose("channel", "time")
    expected = (x - x.mean("time")) * 2
    expected.attrs = dict(x.attrs)
    out = center(x=x, gain=2)
    xr.testing.assert_identical(out, expected)
    assert out.dims == x.dims  # the caller's layout is restored
    assert out.name == "voltage" and out.attrs == {"units": "V"}


def test_reduction_keeps_surviving_coordinates(xr):
    x = signal(xr)
    out = rms(x)
    assert out.dims == ("channel",) and list(out.channel.values) == ["a", "b", "c"]
    np.testing.assert_allclose(out.values, np.sqrt((x**2).mean("time")).values)


def test_positional_contract_binds_letters_to_labels(xr):
    @block(labels=True)
    def scale(x: Sig[T, tuple[M, N]], gain: float = 2.0) -> Sig[T, tuple[M, N]]:
        return x * gain

    @block(labels=True)
    def rowsum(x: Sig[T, tuple[M, N]]) -> Sig[T, tuple[M]]:
        return x.sum(axis=1)

    @block(labels=True)
    def spectrum(x: Sig[float, tuple[M, N]]) -> Sig[complex, tuple[M, Literal[5]]]:
        return np.fft.rfft(x, axis=1)

    x = signal(xr)
    xr.testing.assert_identical(scale(x), (x * 2).assign_attrs(x.attrs))
    out = rowsum(x)  # M was "time": the name and its coordinate come back
    assert out.dims == ("time",) and out.time.equals(x.time)
    out = spectrum(x)  # a fixed size that no input dimension carried
    assert out.dims == ("time", "dim_1") and list(out.coords) == ["time"]
    assert run(from_iterable([x]) >> linalg.transpose(), check="all")[0].data.shape == (
        3,
        8,
    )


def test_graph_runs_check_by_name_and_keep_frame_fields(xr):
    x = signal(xr)
    graph = center(gain=2) >> rms
    out = graph([x, x.transpose("channel", "time")], check="all", on_error="raise")
    assert [f.data.dims for f in out] == [("channel",), ("channel",)]
    f = Frame(x, meta={"seq": 4}, stream={"rate": 1000}).with_issue("calibration")
    out = run(from_iterable([f]) >> graph, check="all", on_error="raise")[0]
    assert out.meta == f.meta and out.stream == f.stream and out.issues == f.issues
    assert "center" in out.prov and "rms" in out.prov


def test_mixed_labelled_and_plain_inputs_follow_the_contract(xr):
    @block(labels=True)
    def add(
        x: Sig[float, tuple[CHANNEL, TIME]], y: Sig[float, tuple[CHANNEL, TIME]]
    ) -> Sig[float, tuple[CHANNEL, TIME]]:
        return x + y

    x = signal(xr)
    plain = np.arange(24.0).reshape(3, 8)  # (channel, time), as the contract says
    out = run(
        from_iterable([(x, plain)], type=add.in_type) >> add,
        check="all",
        on_error="raise",
    )[0].data
    assert out.dims == ("time", "channel")
    np.testing.assert_array_equal(out.values, x.values + plain.T)


def test_tuple_outputs_and_shapeless_contracts(xr):
    @block(labels=True)
    def stats(
        x: Sig[float, tuple[M, N]],
    ) -> tuple[Sig[float, tuple[M]], Sig[float, tuple[M]]]:
        return x.mean(axis=1), x.max(axis=1)

    @block(labels=True)
    def negate(x: Sig[float]) -> Sig[float]:
        return -x

    @block(labels=True)
    def flatten(x: Sig[float]) -> Sig[float]:
        return x.reshape(-1)

    x = signal(xr)
    mean, peak = stats(x)
    assert mean.dims == peak.dims == ("time",)
    frame = run(from_iterable([x]) >> stats, check="all", on_error="raise")[0]
    xr.testing.assert_identical(frame.data[0], mean)
    assert negate(x).dims == x.dims  # same rank: the input's dimensions
    assert isinstance(flatten(x), np.ndarray)  # rank changed, nothing to go on


def test_stream_blocks_and_framed_inputs(xr):
    @block(labels=True)
    def pairs(
        xs: Stream[Sig[float, tuple[M, N]]],
    ) -> Iterator[Sig[float, tuple[Literal[2], M, N]]]:
        buf = []
        for item in xs:
            assert isinstance(item, np.ndarray)
            buf.append(item)
            if len(buf) == 2:
                yield np.stack(buf)
                buf = []

    @block(labels=True)
    def pick(f: Framed[Sig[float]]) -> Sig[float]:
        return f.data.sel(channel="a").values  # a framed body sees the labels

    x = signal(xr)
    out = run(from_iterable([x, x * 2, x * 3]) >> pairs, check="all", on_error="raise")
    assert len(out) == 1 and out[0].data.dims == ("dim_0", "time", "channel")
    assert list(out[0].data.coords) == ["time", "channel"]
    assert isinstance(pick.apply([x])[0].data, np.ndarray)


def test_default_switch_covers_library_blocks(xr, default_off):
    x = signal(xr).isel(channel=0)
    assert isinstance(
        run(from_iterable([x]) >> dsp.scale(gain=3))[0].data, xr.DataArray
    )
    previous = default_labels(True)
    assert previous is False
    out = run(from_iterable([x]) >> dsp.scale(gain=3) >> dsp.rms, check="all")
    assert isinstance(out[0].data, float)  # scalars stay scalars
    out = run(from_iterable([x]) >> dsp.scale(gain=3), check="all")[0].data
    assert out.dims == ("time",) and out.time.equals(x.time)

    @block(labels=False)
    def raw(x: Sig[float, tuple[N]]) -> Sig[float, tuple[N]]:
        assert isinstance(x, xr.DataArray)  # opted out: the body sees the DataArray
        return x

    assert isinstance(raw(x), xr.DataArray)


def test_bound_configuration_and_direct_calls(xr):
    x = signal(xr)
    assert center.bind(gain=3)(x).dims == x.dims
    assert center.call is not center.fn and center.fn.__name__ == "center"
    assert isinstance(center.fn(np.zeros((3, 8))), np.ndarray)  # raw body, no wrapper
    assert callable(sx.adapt(center))


def test_frame_slicing_and_persistence_keep_labels(xr, tmp_path):
    x = signal(xr)
    f = Frame(x, meta={"t0": 0.0}, stream={"rate": 1000})
    part = f[2:4]
    assert isinstance(part.data, xr.DataArray) and part.data.dims == x.dims
    assert part.data.time.values[0] == pytest.approx(0.002) and part["t0"] == 0.002
    assert isinstance(f.at(0.003), xr.DataArray)
    f.save(str(tmp_path / "labelled.npz"))
    back = Frame.load(str(tmp_path / "labelled.npz"))
    xr.testing.assert_identical(back.data, x)
    assert back["rate"] == 1000


def test_gpu_payload_stays_on_device(xr):
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA unavailable")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(str(exc))

    @block(labels=True)
    def gpu_center(
        x: Sig[float, tuple[CHANNEL, TIME]],
    ) -> Sig[float, tuple[CHANNEL, TIME]]:
        assert isinstance(x, cp.ndarray)
        return x - cp.mean(x, axis=-1, keepdims=True)

    x = signal(xr)
    result = gpu_center(x.copy(data=cp.asarray(x.data)))
    assert isinstance(result.data, cp.ndarray) and result.dims == x.dims
    np.testing.assert_allclose(cp.asnumpy(result.data), (x - x.mean("time")).data)
    f = Frame(cp.arange(8.0))
    assert isinstance(f[1:3].data, cp.ndarray)
