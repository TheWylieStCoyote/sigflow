"""Transfer contracts run on CPU; real CUDA round trips run when available."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sigflow import (
    Frame,
    Sig,
    SignalType,
    TypeMismatch,
    block,
    from_iterable,
    run,
    to_device,
    to_host,
)
from sigflow.dims import N, T


class DeviceArray:
    """Test payload which forbids implicit host conversion."""

    __cuda_array_interface__: dict = {}

    def __init__(self, data, device=0):
        self.data = data
        self.dtype, self.shape, self.ndim = data.dtype, data.shape, data.ndim
        self.device = SimpleNamespace(id=device)

    def __array__(self, *args, **kwargs):
        raise AssertionError("implicit host conversion")


@pytest.fixture
def fake_cupy(monkeypatch):
    state = SimpleNamespace(device=7, synchronized=[], downloads=[])

    class Device:
        def __init__(self, index):
            self.index = index

        def __enter__(self):
            self.previous = state.device
            state.device = self.index
            return self

        def __exit__(self, *args):
            state.device = self.previous

    def upload(data, copy=False):
        if isinstance(data, DeviceArray):
            if data.device.id == state.device and not copy:
                return data
            data = data.data
        return DeviceArray(np.array(data, copy=True), state.device)

    def download(data, *, blocking):
        assert blocking and state.device == data.device.id
        state.downloads.append(state.device)
        return data.data.copy()

    cp = SimpleNamespace(
        ndarray=DeviceArray,
        asarray=upload,
        array=upload,
        asnumpy=download,
        cuda=SimpleNamespace(
            Device=Device,
            get_current_stream=lambda: SimpleNamespace(
                synchronize=lambda: state.synchronized.append(state.device)
            ),
        ),
    )
    monkeypatch.setitem(sys.modules, "cupy", cp)
    return state


def test_array_contracts_do_not_transfer_data():
    value = DeviceArray(np.zeros((4, 2), dtype=np.float32))
    env = SignalType(T, (N, 2)).check(value)
    assert env.value("T") == np.dtype(np.float32) and env.value("N") == 4
    SignalType(np.float32, (4, 2)).check(value)
    with pytest.raises(TypeMismatch, match="rank"):
        SignalType(np.float32, (4,)).check(value)
    with pytest.raises(TypeMismatch, match="length"):
        SignalType(np.float32, (5, 2)).check(value)
    with pytest.raises(TypeMismatch, match="dtype"):
        SignalType(np.int32, (4, 2)).check(value)


def test_host_arrays_need_no_cupy(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    value = np.arange(6, dtype=np.float32).reshape(3, 2)
    assert to_host(value) is value
    copied = to_host(value, copy=True)
    assert not np.shares_memory(copied, value)
    np.testing.assert_array_equal(copied, value)
    with pytest.raises(ImportError, match="require CuPy"):
        to_device(value)


@pytest.mark.parametrize("device", [-1, True, "cpu", "cuda:-1", "cuda:x"])
def test_invalid_devices(device):
    with pytest.raises(ValueError, match="device must"):
        to_device(np.ones(2), device=device)


def test_transfer_graph_preserves_frames(fake_cupy):
    value = np.arange(8, dtype=np.float32).reshape(4, 2)
    frame = Frame(
        value,
        meta={"t0": 1.0},
        stream={"rate": 8000},
        samples={"clipped": np.zeros(4, dtype=bool)},
    ).with_issue("calibration warning")
    g = from_iterable([frame]) >> to_device(device="cuda:1") >> to_host()
    output = run(g, check="all", on_error="raise")[0]
    np.testing.assert_array_equal(output.data, value)
    assert output.data.dtype == value.dtype
    assert output.meta == frame.meta and output.stream == frame.stream
    assert output.issues == frame.issues
    assert output.samples["clipped"] is frame.samples["clipped"]
    assert "to_device" in output.prov and "to_host" in output.prov
    assert fake_cupy.synchronized == [1] and fake_cupy.downloads == [1]
    assert fake_cupy.device == 7


def test_device_reuse_copy_and_cross_device(fake_cupy):
    value = DeviceArray(np.arange(4), device=1)
    assert to_device(value, device=1) is value
    copied = to_device(value, device=1, copy=True)
    assert copied is not value
    moved = to_device(value, device=2)
    assert moved.device.id == 2
    np.testing.assert_array_equal(moved.data, value.data)


def test_direct_gpu_graph_input(fake_cupy):
    @block
    def host_identity(x: Sig[T, N]) -> Sig[T, N]:
        return x

    g = to_host() >> host_identity
    value = DeviceArray(np.arange(4, dtype=np.float32), device=2)
    output = g(value, check="all", on_error="raise")
    assert len(output) == 1
    np.testing.assert_array_equal(output[0].data, value.data)


def test_xarray_host_labels():
    xr = pytest.importorskip("xarray")
    value = xr.DataArray(
        np.arange(8, dtype=np.float32).reshape(4, 2),
        dims=("time", "channel"),
        coords={"time": np.arange(4) / 8000, "channel": ["left", "right"]},
        name="audio",
        attrs={"units": "V"},
    )
    output = (to_host() >> to_host(copy=True))(value, check="all")[0]
    xr.testing.assert_identical(output.data, value)


def test_cuda_roundtrip_when_available():
    cp = pytest.importorskip("cupy")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA unavailable: {exc}")
    if count == 0:
        pytest.skip("CUDA unavailable")
    value = np.arange(12, dtype=np.float32).reshape(4, 3)
    uploaded = to_device(value)
    assert isinstance(uploaded, cp.ndarray) and uploaded.device.id == 0
    assert to_device(uploaded) is uploaded
    assert to_device(uploaded, copy=True).data.ptr != uploaded.data.ptr
    output = run(from_iterable([uploaded]) >> to_host(), check="all", on_error="raise")
    np.testing.assert_array_equal(output[0].data, value)
    if count > 1:
        moved = to_device(uploaded, device=1)
        assert moved.device.id == 1
        with cp.cuda.Device(0):
            np.testing.assert_array_equal(to_host(moved), value)
            assert cp.cuda.Device().id == 0

    xr = pytest.importorskip("xarray")
    labeled = xr.DataArray(value, dims=("time", "channel"), name="audio")
    on_device = to_device(labeled)
    assert isinstance(on_device.data, cp.ndarray)
    xr.testing.assert_identical(to_host(on_device), labeled)
