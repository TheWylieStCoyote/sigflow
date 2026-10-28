"""Labelled payloads: DataArrays flow through blocks according to their contracts."""

from __future__ import annotations

import gc
import inspect
import pickle
import sys
import weakref
from typing import Iterator, Literal

import numpy as np
import pytest

import sigflow.xarray as sx
from sigflow import (
    Frame,
    Framed,
    Sig,
    Stream,
    TypeMismatch,
    block,
    default_labels,
    dsp,
    from_iterable,
    linalg,
    parallel,
    run,
)
from sigflow.dims import M, N, T
from sigflow.runtime import RunOptions, _call_block

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


def test_named_source_contracts_compose_in_any_layout(xr):
    @block
    def needs3(x: Sig[float, tuple[Literal[3]]]) -> Sig[float, tuple[Literal[3]]]:
        return x

    x = signal(xr)  # dims (time, channel): 8 x 3
    # a source declared by name feeds (channel, time) blocks in any layout, and the
    # static check agrees with the runtime one
    named = from_iterable([x], type=Sig[float, tuple[TIME, CHANNEL]])
    out = run(named >> center >> rms >> needs3, check="all", on_error="raise")
    assert out[0].data.dims == ("channel",) and out[0].data.shape == (3,)
    # a positional declaration means what it says for a plain array: 8 channels
    positional = from_iterable([x], type=Sig[float, tuple[Literal[8], Literal[3]]])
    with pytest.raises(TypeMismatch, match="channel.*8.*needs 3"):
        _ = positional >> center >> rms >> needs3


def test_stream_blocks_keep_only_the_latest_item(xr):
    @block(labels=True)
    def double(
        xs: Stream[Sig[float, tuple[TIME]]],
    ) -> Iterator[Sig[float, tuple[TIME]]]:
        for item in xs:
            yield item * 2

    items = [
        xr.DataArray(
            np.full(4, float(i)), dims=("time",), coords={"time": np.arange(4)}
        )
        for i in range(6)
    ]
    alive = [weakref.ref(item) for item in items]

    def feed():
        while items:
            yield items.pop(0)  # the feeder keeps nothing behind

    for i, result in enumerate(double.call(feed())):
        assert result.dims == ("time",) and float(result[0]) == 2 * i
        gc.collect()
        # the wrapper holds the item being processed, not the ones before it
        assert all(ref() is None for ref in alive[:i])


def test_mismatched_return_shapes_reach_the_checker(xr):
    @block(labels=True)
    def two(x: Sig[float, tuple[M, N]]) -> Sig[float, tuple[M, N]]:
        return x * 2, x * 3

    @block(labels=True)
    def one(x: Sig[float, tuple[M, N]]) -> tuple[Sig[float], Sig[float]]:
        return x * 2

    x = signal(xr)
    direct = two(x)
    assert isinstance(direct, tuple) and direct[0].dims == x.dims
    assert one(x).dims == x.dims
    with pytest.raises(TypeMismatch, match="two output"):
        run(from_iterable([x]) >> two, check="all", on_error="raise")
    with pytest.raises(TypeMismatch, match="one output"):
        run(from_iterable([x]) >> one, check="all", on_error="raise")


@block
def follows_the_run(x: Sig[float, tuple[N]]) -> Sig[float, tuple[N]]:
    """No ``labels`` option: label-aware only when the run (or the default) says so."""
    return x * 2


def test_labels_are_a_run_option(xr, default_off):
    x = signal(xr).isel(channel=0)
    plain = run(from_iterable([x]) >> follows_the_run)[0].data
    assert isinstance(plain, xr.DataArray)  # untouched: xarray did the arithmetic
    labelled = run(from_iterable([x]) >> follows_the_run, labels=True)[0].data
    assert labelled.dims == ("time",) and labelled.time.equals(x.time)
    assert isinstance(
        run(from_iterable([x]) >> follows_the_run, labels=False)[0].data, xr.DataArray
    )
    # the package default is read when the run starts, not while it runs
    default_labels(True)
    opts = RunOptions()
    default_labels(False)
    assert opts.labels is True and RunOptions().labels is False
    assert RunOptions(labels=True).labels is True
    assert follows_the_run.resolve_call(True) is follows_the_run.adapted
    assert follows_the_run.resolve_call(False) is follows_the_run.fn
    assert center.resolve_call(False) is center.adapted  # its own option wins


def test_process_workers_get_the_resolved_flag(xr, default_off):
    x = signal(xr).isel(channel=0)
    # what a process-pool worker does: unpickle the definition, apply the run's flag
    defn = pickle.loads(pickle.dumps(follows_the_run))
    assert isinstance(_call_block(defn, x, {}, True), xr.DataArray)
    assert _call_block(defn, x, {}, True).dims == ("time",)
    assert isinstance(_call_block(defn, x.values, {}, True), np.ndarray)
    out = run(
        from_iterable([x, x])
        >> parallel(follows_the_run, workers=2, backend="process"),
        labels=True,
    )
    assert [f.data.dims for f in out] == [("time",), ("time",)]


def test_pandas_payloads_index_by_position():
    pd = pytest.importorskip("pandas")
    f = Frame(pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"]), stream={"rate": 1})
    assert f.slice(0).data == 1.0
    assert isinstance(f[0:2].data, np.ndarray) and list(f[0:2].data) == [1.0, 2.0]
