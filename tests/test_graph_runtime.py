import asyncio

import numpy as np
import pytest

from sigflow import (
    MISSING,
    Frame,
    Framed,
    Sig,
    Stream,
    TypeMismatch,
    block,
    from_iterable,
    merge,
    run,
    source,
    tap,
)
from sigflow import dsp


# --------------------------------------------------------------------------- #
# Blocks used by the tests
# --------------------------------------------------------------------------- #


@block
def scale(x: Sig["T", "N"], gain: float = 1.0) -> Sig["T", "N"]:
    return x * gain


@block
def energy(x: Sig["T", "N"]) -> Sig[float, ()]:
    return float(np.sum(np.abs(x) ** 2))


@block
def add(a: Sig["T", "N"], b: Sig["T", "N"]) -> Sig["T", "N"]:
    return a + b


@block
def boom(x: Sig[float, "N"]) -> Sig[float, "N"]:
    raise ValueError("kaboom")


@block
def pairs(xs: Stream[Sig[float, "N"]]) -> Sig[float, (2, "N")]:
    buf = []
    for x in xs:
        buf.append(x)
        if len(buf) == 2:
            yield np.stack(buf)
            buf = []


@source(total=lambda count, **_: count)
def ramp(count: int = 4, n: int = 3) -> Sig[float, "N"]:
    for i in range(count):
        yield np.full(n, float(i))


@source
async def aramp(count: int = 4, n: int = 3) -> Sig[float, "N"]:
    for i in range(count):
        await asyncio.sleep(0)
        yield np.full(n, float(10 + i))


# --------------------------------------------------------------------------- #
# Block definitions behave like functions
# --------------------------------------------------------------------------- #


def test_block_is_still_a_function():
    assert np.array_equal(scale(np.ones(3), gain=2), np.full(3, 2.0))
    assert scale.kind == "map" and pairs.kind == "stream" and ramp.kind == "source"
    assert [i.name for i in add.inputs] == ["a", "b"]
    assert list(scale.config) == ["gain"]


def test_wrapping_library_functions():
    fft = block(np.fft.fft, inputs=Sig[complex, "N"], outputs=Sig[complex, "N"])
    out = run(ramp(count=1) >> dsp.to_complex >> fft)
    assert out[0].data.shape == (3,)
    with pytest.raises(TypeError):
        scale(gainz=2)  # unknown configuration is rejected when placing a node


# --------------------------------------------------------------------------- #
# Composition and static validation
# --------------------------------------------------------------------------- #


def test_pipeline_runs_and_carries_provenance():
    out = run(ramp(count=3) >> scale(gain=2) >> energy)
    assert [f.data for f in out] == [0.0, 12.0, 48.0]
    f = out[2]
    assert f.prov.op == "energy"
    assert [p.op for p in f.lineage()] == ["energy", "scale", "ramp"]
    assert f.sources()[0].seq == 2
    assert f.prov.parents[0].params["gain"] == 2


def test_type_mismatch_is_raised_at_build_time():
    with pytest.raises(TypeMismatch, match="rank"):
        ramp() >> energy >> scale
    with pytest.raises(TypeMismatch):
        ramp() >> pairs >> scale  # (2, N) cannot feed (N,)
    with pytest.raises(TypeMismatch, match="source"):
        ramp() >> ramp()


def test_dimension_conflict_detected_before_running():
    @source
    def ten() -> Sig[float, (10,)]:
        yield np.ones(10)

    @source
    def twenty() -> Sig[float, (20,)]:
        yield np.ones(20)

    with pytest.raises(TypeMismatch, match="dimension 'N'"):
        (ten() & twenty()) >> add
    out = run((ten() & ten()) >> add)
    assert out[0].data.sum() == 20


def test_config_refines_shapes():
    with pytest.raises(TypeMismatch):
        (dsp.sine(n=64) & dsp.sine(n=128)) >> add
    g = (dsp.sine(n=64, frames=2) & dsp.noise(n=64, frames=2)) >> add
    assert g.env.bindings["N"] == 64


def test_fanout_and_multi_input():
    g = ramp(count=2, n=2) >> (scale(gain=2) | scale(gain=3)) >> add
    out = run(g)
    assert [list(f.data) for f in out] == [[0.0, 0.0], [5.0, 5.0]]
    assert out[1].prov.op == "add" and out[1].prov.parents[0].op == "fanout"


def test_zip_and_merge_mix_sync_and_async_sources():
    out = run((ramp(count=3) & aramp(count=3)) >> add)
    assert [f.data[0] for f in out] == [10.0, 12.0, 14.0]
    out = run(merge(ramp(count=2), aramp(count=2)) >> energy)
    assert sorted(f.data for f in out) == [0.0, 3.0, 300.0, 363.0]


def test_from_iterable_and_plain_functions():
    g = from_iterable([np.ones(2), np.ones(2) * 2], Sig[float, "N"]) >> (
        lambda x: x.sum()
    )
    assert [f.data for f in run(g)] == [2.0, 4.0]


def test_runtime_check_catches_lying_blocks():
    @block
    def liar(x: Sig[float, "N"]) -> Sig[float, "N"]:
        return np.zeros((2, 2))

    with pytest.raises(TypeMismatch, match="runtime check"):
        run(ramp() >> liar)


# --------------------------------------------------------------------------- #
# Annotations, issues and error propagation
# --------------------------------------------------------------------------- #


def test_annotations_propagate_through_blocks():
    src = from_iterable(
        [Frame(np.ones(4), meta={"seq": 7}, stream={"rate": 100.0})], Sig[float, "N"]
    )
    out = run(src >> scale(gain=2) >> scale(gain=2))
    assert out[0].meta["seq"] == 7 and out[0].stream["rate"] == 100.0


def test_sample_annotations_follow_the_leading_axis():
    @block
    def mark(f: Framed[Sig[float, "N"]]) -> Sig[float, "N"]:
        return f.annotate_samples(flag=f.data > 1)

    out = run(ramp(count=3) >> mark >> scale(gain=2))  # N -> N keeps samples
    assert "flag" in out[2].samples
    out = run(ramp(count=3) >> mark >> energy)  # N -> () drops them
    assert not out[2].samples


def test_errors_propagate_like_annotations():
    out = run(ramp(count=2) >> boom >> scale(gain=3) >> energy)
    assert all(f.failed and f.data is MISSING for f in out)
    assert out[0].errors[0].origin == "boom"
    assert isinstance(out[0].errors[0].exc, ValueError)
    assert [p.op for p in out[0].lineage()] == ["energy", "scale", "boom", "ramp"]
    with pytest.raises(ValueError):
        out[0].raise_if_failed()


def test_error_policies():
    with pytest.raises(ValueError):
        run(ramp() >> boom, on_error="raise")
    assert run(ramp(count=3) >> boom, on_error="drop") == []
    assert len(run(ramp(count=3) >> boom.node(), on_error="propagate")) == 3
    dropper = block(boom.fn, on_error="drop", name="dropper")
    assert run(ramp(count=3) >> dropper) == []


def test_warnings_do_not_stop_computation():
    out = run(
        dsp.sine(frames=1, n=64, amplitude=2.0)
        >> dsp.detect_clipping(limit=1.0)
        >> dsp.rms
    )
    assert out[0].ok and out[0].warnings and out[0].samples == {}


def test_failed_frames_bypass_stream_blocks():
    @block
    def failing_first(x: Sig[float, "N"]) -> Sig[float, "N"]:
        if x[0] == 0:
            raise RuntimeError("first")
        return x

    out = run(ramp(count=5) >> failing_first >> pairs)
    failed = [f for f in out if f.failed]
    good = [f for f in out if f.ok]
    assert len(failed) == 1 and len(good) == 2
    assert good[0].data.shape == (2, 3)
    assert len(good[0].prov.parents) == 2  # provenance: the two frames consumed


def test_filter_by_returning_none_and_taps():
    seen = []

    @block
    def only_even(x: Sig[float, "N"]) -> Sig[float, "N"]:
        return x if x[0] % 2 == 0 else None

    out = run(ramp(count=4) >> only_even >> tap(lambda x: seen.append(x[0])))
    assert [f.data[0] for f in out] == [0.0, 2.0] and seen == [0.0, 2.0]


# --------------------------------------------------------------------------- #
# Iteration APIs and progress
# --------------------------------------------------------------------------- #


def test_sync_and_async_iteration():
    g = ramp(count=3) >> scale(gain=2)
    assert [f.data[0] for f in g] == [0.0, 2.0, 4.0]

    async def collect():
        return [f.data[0] async for f in g]

    assert asyncio.run(collect()) == [0.0, 2.0, 4.0]


def test_arun_inside_running_loop_and_sync_run_from_loop():
    g = ramp(count=2) >> energy

    async def main():
        out = await g.arun()
        assert len(out) == 2
        assert len(g.run()) == 2  # run() from within a loop uses a helper thread

    asyncio.run(main())


def test_monitor_and_progress(capsys):
    g = ramp(count=5) >> scale(gain=2)
    g.run(progress="text")
    m = g.last_monitor
    assert m.percent == 100.0 and m.stats("scale").out == 5 and not m.running
    assert "scale" in m.table() and "100.0%" in capsys.readouterr().err


def test_stream_block_with_async_generator():
    @block
    async def running_sum(xs: Stream[Sig[float, "N"]]) -> Sig[float, "N"]:
        acc = None
        async for x in xs:
            acc = x if acc is None else acc + x
            yield acc

    out = run(aramp(count=3) >> running_sum)
    assert out[-1].data[0] == 33.0


def test_fanout_validates_every_branch():
    @block
    def wants_matrix(x: Sig[float, ("M", "N")]) -> Sig[float, ()]:
        return 0.0

    with pytest.raises(TypeMismatch):
        ramp() >> (energy | wants_matrix)  # second branch must be checked too
    g = dsp.sine(n=5, frames=2) >> (scale(gain=2) | pairs)
    env = g.validate()
    assert g.stages[-1].branches[1].out_type.resolve(env).shape == (
        2,
        5,
    )  # pairs' N bound too


def test_subgraphs_compose_and_can_be_reused():
    front = dsp.window >> dsp.to_complex >> dsp.fft  # sourceless sub-graph
    back = dsp.power_db >> dsp.rms
    assert len(run(dsp.sine(n=32, frames=2) >> front >> back)) == 2
    # the same sub-graph instance twice, with different sizes, in one graph
    g = (dsp.sine(n=32, frames=2) >> front) & (dsp.noise(n=16, frames=2) >> front)
    out = run(g)
    assert [x.shape for x in out[0].data] == [(32,), (16,)]
    assert {s.name for s in g.last_monitor.blocks} >= {"fft", "fft#2"}
    # as fan-out branches
    g3 = dsp.sine(n=8, frames=1) >> (front | front >> back)
    assert run(g3)[0].data[0].shape == (8,)


def test_background_run_pause_step_stop():
    import time

    g = dsp.ticker(period=0.02, count=200) >> (lambda d: d["tick"])
    r = g.start(paused=True)
    time.sleep(0.15)
    assert r.state == "paused" and r.frames == []
    r.step(2)
    time.sleep(0.2)
    assert len(r.frames) == 2 and r.state == "paused"
    r.resume()
    time.sleep(0.2)
    assert len(r.frames) > 2 and r.state == "running"
    r.pause()
    time.sleep(0.1)
    n = len(r.frames)
    r.stop()
    assert (
        r.state == "stopped" and r.finished and len(r.frames) == n
    )  # stop lets nothing more through
    done = (ramp(count=3) >> energy).start()
    assert len(done.wait()) == 3 and done.state == "done"
    bad = (ramp(count=2) >> boom).start(on_error="raise")
    with pytest.raises(ValueError):
        bad.wait()
    assert bad.state == "failed"


def test_source_exception_is_attributed_to_the_source():
    @source
    def broken(count: int = 3) -> Sig[float, "N"]:
        for i in range(count):
            if i == 1:
                raise RuntimeError("sensor unplugged")
            yield np.ones(2)

    out = run(broken() >> pairs)  # a stream block downstream must not get the blame
    assert len(out) == 1 and out[0].failed and out[0].errors[0].origin == "broken"
    with pytest.raises(RuntimeError):
        run(broken() >> scale, on_error="raise")


def test_stream_outputs_without_new_input_inherit_parents():
    out = run(dsp.from_array(np.arange(8.0), n=4, rate=10.0) >> dsp.overlap(n=2, hop=1))
    assert len(out) == 7
    assert all(f.prov.parents for f in out)  # every window has provenance
    assert all(f.stream["rate"] == 10.0 and "offset" in f.meta for f in out)


def test_port_selection():
    from sigflow import linalg

    @source
    def mats() -> Sig[float, (3, 3)]:
        yield np.diag([1.0, 2.0, 3.0])

    vals = run(mats() >> linalg.eigh[0])[0]
    assert (
        np.allclose(vals.data, [1, 2, 3])
        and vals.prov.op == "[0]"
        and vals.prov[1].op == "eigh"
    )
    with pytest.raises(TypeMismatch):
        mats() >> linalg.eigh[0] >> linalg.transpose  # selected port is typed
    with pytest.raises(TypeError):
        energy[0]  # single-output block has no ports
    # fan-out branches by name, index, negative index, multi-select and slice
    fan = ramp(count=1) >> (scale(gain=2) | energy | scale(gain=3))
    assert run(fan["energy"])[0].data == 0.0
    assert run(fan[-1])[0].data.tolist() == [0.0, 0.0, 0.0]
    both = run(fan[0, 2])[0]
    assert len(both.data) == 2 and both.parts[1].prov.op == "scale"
    assert (
        run(fan[1:])[0].prov.op == "[energy, 2]"
    )  # duplicate names fall back to indices
    with pytest.raises(KeyError):
        fan["scale"]  # ambiguous: two branches share the name
    # named output ports on a wrapped library function
    eigh = block(
        np.linalg.eigh,
        inputs=Sig["T", ("N", "N")],
        outputs={"vals": Sig[float, "N"], "vecs": Sig["T", ("N", "N")]},
    )
    assert run(mats() >> eigh["vecs"] >> linalg.transpose)[0].data.shape == (3, 3)
    assert run(mats() >> eigh["vals"])[0].prov.op == "[vals]"


def test_frame_and_provenance_selection():
    out = run(dsp.sine(n=16, frames=2) >> dsp.window >> dsp.to_complex >> dsp.fft)
    f = out[1]
    assert (
        f["rate"] == 48000.0 and f["seq"] == 1 and "rate" in f and f.get("nope", 0) == 0
    )
    with pytest.raises(KeyError):
        f["nope"]
    p = f.prov
    assert [n.op for n in p.chain()] == ["fft", "to_complex", "window", "sine"]
    assert (
        p[0] is p and p[2].op == "window" and p[-1].op == "sine" and p["sine"].seq == 1
    )
    assert (
        [n.op for n in p[1:3]] == ["to_complex", "window"]
        and "window" in p
        and "nope" not in p
    )
    with pytest.raises(KeyError):
        p["nope"]
    t = run(ramp(count=1) >> (scale(gain=2) | energy))[0]
    assert t[1].data == 0.0 and t[1].prov.op == "energy"


def test_selected_port_lineage_follows_the_branch():
    post = block(
        lambda e: e + 1.0, name="post", inputs=Sig[float, ()], outputs=Sig[float, ()]
    )
    f = run(ramp(count=1) >> (scale(gain=2) | energy >> post)["post"])[0]
    assert [n.op for n in f.prov.chain()] == ["[post]", "post", "energy", "ramp"]
    assert "fanout" in f.prov  # the tuple node is still in the lineage


def test_multi_input_framed_block_skips_failed_parts():
    @block
    def combine(
        a: Framed[Sig[float, "N"]], b: Framed[Sig[float, "N"]]
    ) -> Sig[float, ()]:
        return float(a.data.sum() + b.data.sum())

    out = run(ramp(count=3) >> (scale(gain=1) | boom) >> combine)
    assert all(f.failed for f in out) and all(
        len(f.errors) == 1 for f in out
    )  # no second error from combine
    assert out[0].errors[0].origin == "boom"


def test_knobs_tune_running_graphs_and_runs_keep_bounded_buffers():
    import time
    from sigflow import Knob

    gain = Knob(1.0, "gain")
    g = (
        dsp.ticker(period=0.01, count=10**6)
        >> (lambda d: np.ones(2))
        >> scale(gain=gain)
    )
    r = g.start(keep=5)
    time.sleep(0.15)
    assert r.latest.data[0] == 1.0
    gain.value = 4.0
    time.sleep(0.15)
    assert r.latest.data[0] == 4.0 and len(r.frames) <= 5
    batch = r.drain()
    assert batch and r.frames == [] or len(r.frames) < 5
    r.stop()

    # a source reads the knob itself
    @source
    def tone(level: Knob, frames: int = 10**6) -> Sig[float, ()]:
        for _ in range(frames):
            yield float(level.value)

    lvl = Knob(2.0)
    r = (tone(level=lvl) >> (lambda x: x)).start(keep=3)
    time.sleep(0.05)
    lvl.value = 7.0
    time.sleep(0.05)
    assert r.latest.data == 7.0
    r.stop()


def test_throttle_record_snapshot_and_knob_kinds():
    import time
    from sigflow import Knob

    win = Knob("hann", "window", choices=["hann", "hamming"])
    r = (dsp.sine(n=64, frames=10**6) >> dsp.window(kind=win)).start(keep=3)
    r.set_rate(20)
    time.sleep(0.5)
    n0 = r.monitor.produced
    time.sleep(1.0)
    assert 10 <= r.monitor.produced - n0 <= 40  # throttled to ~20 fps
    r.set_recording(True)
    time.sleep(0.3)
    r.set_recording(False)
    assert 2 <= len(r.recording) <= 12
    win.value = "hamming"
    time.sleep(0.2)
    assert (
        r.snapshot().prov["window"].params["kind"] == "hamming"
    )  # provenance holds the value used
    assert len(r.snapshots) == 1
    r.stop()


def test_knob_listeners_and_plot_options():
    from sigflow import Knob, plot

    seen = []
    k = Knob(1.0, "k").on_change(seen.append)
    k.value = 2.0
    assert seen == [2.0]
    ylim = Knob((-10.0, 10.0))
    p = plot.line(ylim=ylim)
    run(dsp.sine(n=16, frames=1) >> p)
    ylim.value = (-1.0, 1.0)
    p.redraw(p.frames[-1])
    assert tuple(p.ax.get_ylim()) == (-1.0, 1.0)
