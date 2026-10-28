"""Transfer contracts run on CPU; real CUDA round trips run when available."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sigflow import (
    Frame,
    Graph,
    RatioMismatch,
    Sig,
    SignalType,
    TypeMismatch,
    block,
    dsp,
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


class ForeignDeviceArray(DeviceArray):
    """Another library's CUDA array: speaks the interface, is not a CuPy array."""


@pytest.fixture
def fake_cupy(monkeypatch):
    """A CuPy stand-in that records every synchronisation as ``(kind, device)``."""
    state = SimpleNamespace(device=7, synchronized=[], downloads=[])

    class Device:
        def __init__(self, index=None):
            self.index = state.device if index is None else index

        @property
        def id(self):
            return self.index

        def __enter__(self):
            self.previous = state.device
            state.device = self.index
            return self

        def __exit__(self, *args):
            state.device = self.previous

    class CupyArray(DeviceArray):
        pass

    def upload(data, copy=False):
        if isinstance(data, CupyArray):
            if data.device.id == state.device and not copy:
                return data
            data = data.data
        elif isinstance(data, DeviceArray):  # a foreign array: a view on its device
            return CupyArray(data.data, data.device.id)
        return CupyArray(np.array(data, copy=True), state.device)

    def download(data, *, blocking):
        assert blocking and state.device == data.device.id
        assert state.synchronized[-1] == ("device", data.device.id)
        state.downloads.append(state.device)
        return data.data.copy()

    cp = SimpleNamespace(
        ndarray=CupyArray,
        asarray=upload,
        array=upload,
        asnumpy=download,
        cuda=SimpleNamespace(
            Device=Device,
            get_current_stream=lambda: SimpleNamespace(
                synchronize=lambda: state.synchronized.append(("stream", state.device))
            ),
            runtime=SimpleNamespace(
                deviceSynchronize=lambda: state.synchronized.append(
                    ("device", state.device)
                )
            ),
        ),
    )
    monkeypatch.setitem(sys.modules, "cupy", cp)
    state.CupyArray = CupyArray
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


@pytest.mark.parametrize(
    "device", [-1, True, "cpu", "cuda:-1", "cuda:x", "cuda:²", 1.0]
)
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
    # upload: the copy's stream is waited for; download: the source device settles
    assert fake_cupy.synchronized == [("stream", 1), ("device", 1)]
    assert fake_cupy.downloads == [1]
    assert fake_cupy.device == 7


def test_device_reuse_copy_and_cross_device(fake_cupy):
    value = fake_cupy.CupyArray(np.arange(4), device=1)
    assert to_device(value, device=np.int64(1)) is value
    assert fake_cupy.synchronized == [("device", 1)]  # settled once, nothing copied
    copied = to_device(value, device=1, copy=True)
    assert copied is not value
    assert fake_cupy.synchronized[1:] == [("device", 1), ("stream", 1)]
    moved = to_device(value, device=2)
    assert moved.device.id == 2
    np.testing.assert_array_equal(moved.data, value.data)
    assert fake_cupy.synchronized[3:] == [("device", 1), ("stream", 2)]


def test_foreign_cuda_arrays_are_accepted(fake_cupy):
    foreign = ForeignDeviceArray(np.arange(3.0), device=2)
    up = to_device(foreign, device=2)
    assert isinstance(up, fake_cupy.CupyArray) and up.device.id == 2
    assert fake_cupy.synchronized == [("device", 2)]
    np.testing.assert_array_equal(to_host(foreign), np.arange(3.0))
    assert fake_cupy.downloads == [2]


def test_direct_gpu_graph_input(fake_cupy):
    @block
    def host_identity(x: Sig[T, N]) -> Sig[T, N]:
        return x

    g = to_host() >> host_identity
    value = fake_cupy.CupyArray(np.arange(4, dtype=np.float32), device=2)
    output = g(value, check="all", on_error="raise")
    assert len(output) == 1
    np.testing.assert_array_equal(output[0].data, value.data)


@pytest.mark.parametrize("subclass", [False, True])
def test_xarray_host_labels(subclass):
    xr = pytest.importorskip("xarray")

    class LabeledSignal(xr.DataArray):
        __slots__ = ()

    array_type = LabeledSignal if subclass else xr.DataArray
    value = array_type(
        np.arange(8, dtype=np.float32).reshape(4, 2),
        dims=("time", "channel"),
        coords={"time": np.arange(4) / 8000, "channel": ["left", "right"]},
        name="audio",
        attrs={"units": "V"},
    )
    output = (to_host() >> to_host(copy=True))(value, check="all")[0]
    xr.testing.assert_identical(output.data, value)
    with pytest.raises(TypeError, match="DataArray"):
        to_host(value.to_dataset())


# --------------------------------------------------------------- contracts


@pytest.mark.parametrize("transfer", [to_device, to_host])
def test_transfer_preserves_complete_contract(transfer):
    contract = SignalType(np.float32, (8, 2), 48000, "V", frozenset({"calibrated"}))
    src = from_iterable([], type=contract)
    graph = src >> transfer()
    assert graph.out_type == src.out_type
    assert graph.stages[-1].in_type == src.out_type
    assert transfer.keep_samples


@pytest.mark.parametrize("transfer", [to_device, to_host])
def test_transfer_rejects_bad_shapes_and_preserves_ratios(transfer):
    needs_four = block(
        lambda x: x, inputs=SignalType(float, (4,)), outputs=SignalType(float, (4,))
    )
    with pytest.raises(TypeMismatch):
        _ = dsp.sine(n=8) >> transfer() >> needs_four
    assert str((dsp.sine(n=1024) >> transfer() >> dsp.chunk(256)).ratio) == "1:4"
    with pytest.raises(RatioMismatch):
        _ = dsp.sine(n=1024) >> transfer() >> (dsp.chunk(256) | dsp.rms)


@pytest.mark.parametrize("transfer", [to_device, to_host])
def test_transfer_carries_what_the_graph_has_pinned_down(transfer):
    # a generic block in between declares only symbols; the size and dtype the
    # source fixed reach the transfer through the pipeline's bindings
    pipeline = dsp.sine(n=1024) >> dsp.scale(gain=1.0) >> transfer() >> dsp.chunk(256)
    assert str(pipeline.ratio) == "1:4"
    assert pipeline.stages[2].out_type == SignalType(np.float64, (1024,))
    needs_four = block(
        lambda x: x, inputs=SignalType(float, (4,)), outputs=SignalType(float, (4,))
    )
    with pytest.raises(TypeMismatch):
        _ = dsp.sine(n=8) >> dsp.scale(gain=1.0) >> transfer() >> needs_four
    graph = Graph()
    src = graph.add(dsp.sine(n=1024, frames=1))
    scale = graph.add(dsp.scale(gain=1.0))
    moved = graph.add(transfer())
    chunk = graph.add(dsp.chunk(256))
    graph.connect(moved, chunk).connect(scale, moved).connect(src, scale).output(chunk)
    assert moved.out_type == SignalType(np.float64, (1024,))
    assert str(graph.ratio) == "1:4"


def test_transfer_contracts_in_reused_subgraphs():
    reusable = to_host() >> to_host.bind(copy=True)
    small = dsp.sine(n=8, frames=1) >> reusable
    large = dsp.sine(n=16, frames=1) >> reusable
    assert small.out_type.shape == (8,) and large.out_type.shape == (16,)
    assert run(small, check="all")[0].data.shape == (8,)
    assert run(large, check="all")[0].data.shape == (16,)


@pytest.mark.parametrize("reverse", [False, True])
def test_builder_transfer_contracts(reverse):
    graph = Graph()
    src = graph.add(dsp.sine(n=1024, frames=1))
    transfer = graph.add(to_host())
    chunk = graph.add(dsp.chunk(256))
    graph.output(transfer)
    if reverse:
        graph.connect(transfer, chunk).connect(src, transfer)
    else:
        graph.connect(src, transfer).connect(transfer, chunk)
    assert transfer.out_type.shape == (1024,)
    assert graph.out_type.shape == (1024,)
    graph.output(chunk)
    assert str(graph.ratio) == "1:4"
    assert len(run(graph, check="all", on_error="raise")) == 4


def test_builder_rechecks_downstream_and_rolls_back():
    graph = Graph()
    src = graph.add(dsp.sine(n=8))
    transfer = graph.add(to_host())
    sink = graph.add(
        block(
            lambda x: x, inputs=SignalType(float, (4,)), outputs=SignalType(float, (4,))
        )
    )
    graph.connect(transfer, sink)
    before = transfer.out_type
    with pytest.raises(TypeMismatch):
        graph.connect(src, transfer)
    assert len(graph.edges) == 1 and transfer.out_type == before


def test_reusable_builder_transfer_inherits_placement_contract():
    reusable = Graph("host")
    inlet = reusable.input(SignalType(float))
    transfer = reusable.add(to_host())
    reusable.connect(inlet, transfer).output(transfer)
    small = dsp.sine(n=8, frames=1) >> reusable
    large = dsp.sine(n=16, frames=1) >> reusable
    assert small.out_type.shape == (8,) and large.out_type.shape == (16,)
    assert reusable.out_type.shape is None  # the template is untouched
    assert run(small, check="all")[0].data.shape == (8,)
    assert run(large, check="all")[0].data.shape == (16,)
    assert str((dsp.sine(n=1024) >> reusable >> dsp.chunk(256)).ratio) == "1:4"
    needs_four = block(
        lambda x: x, inputs=SignalType(float, (4,)), outputs=SignalType(float, (4,))
    )
    with pytest.raises(TypeMismatch):
        _ = small >> needs_four


def test_reusable_graph_keeps_its_declared_input_contract():
    reusable = Graph("host")
    inlet = reusable.input(SignalType(np.float32, (4,)))
    transfer = reusable.add(to_host())
    reusable.connect(inlet, transfer).output(transfer)
    placed = (
        from_iterable([np.zeros(4, np.float32)], type=SignalType(np.float32))
        >> reusable
    )
    # what the producer leaves open, the declaration still pins down
    assert placed.out_type == SignalType(np.float32, (4,))
    needs_eight = block(
        lambda x: x, inputs=SignalType(float, (8,)), outputs=SignalType(float, (8,))
    )
    with pytest.raises(TypeMismatch):
        _ = placed >> needs_eight
    with pytest.raises(TypeMismatch):
        _ = dsp.sine(n=8, frames=1) >> reusable


def test_preserve_type_declaration_checked_at_runtime():
    @block(preserve_type=True)
    def bad(x: Sig[T]) -> Sig[T]:
        return x[:2]

    with pytest.raises(TypeMismatch, match="length"):
        run(dsp.sine(n=8, frames=1) >> bad, check="all", on_error="raise")
    with pytest.raises(TypeError, match="single-input map"):

        @block(preserve_type=True)
        def two(x: Sig[T], y: Sig[T]) -> Sig[T]:
            return x


# --------------------------------------------------------------- real CUDA


@pytest.fixture
def cuda():
    cp = pytest.importorskip("cupy")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA unavailable: {exc}")
    if count == 0:
        pytest.skip("CUDA unavailable")
    return cp


@pytest.mark.parametrize("subclass", [False, True])
def test_cuda_roundtrip_when_available(cuda, subclass):
    cp = cuda
    count = cp.cuda.runtime.getDeviceCount()
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

    class LabeledSignal(xr.DataArray):
        __slots__ = ()

    array_type = LabeledSignal if subclass else xr.DataArray
    labeled = array_type(
        value,
        dims=("time", "channel"),
        name="audio",
        coords={"time": np.arange(4), "channel": ["a", "b", "c"]},
        attrs={"units": "V"},
    )
    on_device = to_device(labeled)
    assert isinstance(on_device.data, cp.ndarray)
    xr.testing.assert_identical(to_host(on_device), labeled)


@pytest.mark.parametrize(
    "transfer,copy", [(to_host, False), (to_device, False), (to_device, True)]
)
def test_nondefault_producer_stream_is_ready(cuda, transfer, copy):
    cp = cuda
    kernel = cp.RawKernel(
        r"""
    extern "C" __global__ void delayed_write(float* output) {
        unsigned long long start = clock64();
        while (clock64() - start < 100000000ULL) {}
        output[0] = 42.0f;
    }
    """,
        "delayed_write",
    )
    kernel.compile()
    value = cp.zeros(1, dtype=cp.float32)
    cp.cuda.runtime.deviceSynchronize()
    producer = cp.cuda.Stream(non_blocking=True)
    with producer:
        kernel((1,), (1,), (value,))
    try:
        result = transfer(value, copy=copy)
        assert producer.done
        result = cp.asnumpy(result) if isinstance(result, cp.ndarray) else result
        np.testing.assert_array_equal(result, [42.0])
    finally:
        producer.synchronize()
