"""per_key: one instance of a body per distinct key, made on first sight."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from sigflow import (
    Frame,
    Framed,
    Lineage,
    Sig,
    Stream,
    TypeMismatch,
    block,
    dsp,
    from_iterable,
    per_key,
    run,
    source,
)
from sigflow.dims import N
from sigflow.persist import Stateful

closed: list[str] = []


@source
def tracks(frames: int = 12) -> Iterator[Sig[float, N]]:
    """Two interleaved signals, "a" and "b"; "b" stops after frame 5."""
    for i in range(frames):
        key = "a" if i % 2 == 0 or i > 5 else "b"
        yield Frame(np.full(2, float(i)), meta={"seq": i, "track": key})


@block
def running_sum(xs: Stream[Framed[Sig[float, N]]]) -> Iterator[Sig[float, N]]:
    """Stateful per instance: the sum of everything this instance has seen."""
    acc = None
    try:
        for f in xs:
            acc = f.data if acc is None else acc + f.data
            yield f.evolve(data=acc)
    finally:
        closed.append("sum closed")


class Smoother(Stateful):
    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self.acc: float | None = None

    def __call__(self, x: Sig[float, N]) -> Sig[float, N]:
        v = float(x[0])
        self.acc = (
            v if self.acc is None else (1 - self.alpha) * self.acc + self.alpha * v
        )
        return np.full(2, self.acc)


smooth = block(Smoother(0.5), name="smooth")


def by_track(out):
    return {k: [f for f in out if f["track"] == k] for k in ("a", "b")}


def test_each_key_gets_its_own_stream_block_state():
    closed.clear()
    out = run(tracks() >> per_key("track", running_sum), on_error="raise")
    groups = by_track(out)
    # "a" sees 0, 2, 4, then 6..11: its own running sum, nothing of "b" in it
    assert [float(f.data[0]) for f in groups["a"]] == [0, 2, 6, 12, 19, 27, 36, 46, 57]
    assert [float(f.data[0]) for f in groups["b"]] == [1, 4, 9]  # not mixed with "a"
    assert {f.prov.op for f in out} == {"per_key(track)['a']", "per_key(track)['b']"}
    assert closed == ["sum closed", "sum closed"]  # both instances finished cleanly


def test_each_key_gets_its_own_stateful_object():
    out = run(tracks(frames=4) >> per_key("track", smooth), on_error="raise")
    groups = by_track(out)
    assert [float(f.data[0]) for f in groups["a"]] == [0.0, 1.0]  # 0, then (0+2)/2
    assert [float(f.data[0]) for f in groups["b"]] == [1.0, 2.0]  # 1, then (1+3)/2
    assert smooth.fn.acc is None  # the template was never touched


def test_ttl_closes_an_idle_instance_and_reopens_the_key():
    closed.clear()
    g = tracks() >> per_key("track", running_sum, ttl=3)
    out = run(g, on_error="raise")
    groups = by_track(out)
    # "b" is last seen at frame 5; three frames later its instance is closed
    assert [float(f.data[0]) for f in groups["b"]] == [1, 4, 9]
    assert closed.count("sum closed") == 2
    # a key that comes back after ttl starts a fresh instance: the sum restarts
    gap = (
        [Frame(np.ones(2), meta={"track": "x"})]
        + [Frame(np.ones(2), meta={"track": "y"}) for _ in range(4)]
        + [Frame(np.ones(2), meta={"track": "x"})]
    )
    out = run(
        from_iterable(gap) >> per_key("track", running_sum, ttl=3), on_error="raise"
    )
    xs = [float(f.data[0]) for f in out if f["track"] == "x"]
    assert xs == [1.0, 1.0]  # the second "x" did not continue the first instance


def test_callable_key_none_and_failed_frames_pass_through():
    @block
    def boom(x: Sig[float, N]) -> Sig[float, N]:
        if float(x[0]) == 2:
            raise ValueError("boom")
        return x

    g = (
        tracks(frames=5)
        >> boom
        >> per_key(
            lambda f: f["track"] if f["seq"] != 4 else None, dsp.scale(gain=10.0)
        )
    )
    out = run(g, on_error="propagate")
    by_seq = {f["seq"]: f for f in out}
    assert by_seq[2].failed and "boom" in by_seq[2].errors[0].origin  # passed through
    assert float(by_seq[4].data[0]) == 4.0  # key None: untouched
    assert float(by_seq[0].data[0]) == 0.0 and float(by_seq[1].data[0]) == 10.0
    assert all(f.prov.op.startswith("per_key(<lambda>)") for f in out)


def test_selector_errors_fail_the_frame_not_the_run():
    g = tracks(frames=3) >> per_key(lambda f: f["missing"], dsp.scale(gain=1.0))
    out = run(g)
    assert all(f.failed for f in out) and "KeyError" in out[0].errors[0].message


def test_types_describe_draw_and_reuse():
    body = dsp.scale(gain=2.0) >> dsp.rms
    node = per_key("track", body, ttl=10)
    g = tracks() >> node
    assert node.in_type == body.in_type and str(g.out_type) == str(body.out_type)
    assert "per_key(track), ttl=10" in g.describe() and "scale" in g.describe()
    assert "per key" in g.draw().svg
    with pytest.raises(TypeMismatch):
        _ = tracks() >> per_key(
            "track", dsp.chunk(4) >> dsp.fft >> per_key("x", dsp.rms) >> dsp.to_complex
        )
    twice = (tracks() >> node) & (
        tracks() >> node
    )  # the same node reused: fresh instances
    assert len(run(twice, on_error="raise")) == 12
    lineage = Lineage(run(g, on_error="raise"))
    assert len(lineage.through("rms")) == 12


def test_many_keys_and_order_within_a_key():
    @source
    def many(frames: int = 60) -> Iterator[Sig[float, N]]:
        for i in range(frames):
            yield Frame(np.full(1, float(i)), meta={"seq": i, "k": i % 7})

    out = run(many() >> per_key("k", running_sum), on_error="raise")
    assert len(out) == 60
    for k in range(7):
        seqs = [f["seq"] for f in out if f["k"] == k]
        assert seqs == sorted(seqs)  # a key's frames keep their order
