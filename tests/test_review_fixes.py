"""Regression tests for six findings of a code review (per_key, gather, loky, probes,
tag policies). Each test names the failure it guards against."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterator

import numpy as np
import pytest

from sigflow import (
    Frame,
    Framed,
    Graph,
    Sig,
    SignalType,
    Stream,
    block,
    dsp,
    from_iterable,
    gather,
    parallel,
    per_key,
    run,
    source,
    switch,
)
from sigflow.dims import M, N
from sigflow.persist import Stateful

FLOATS = SignalType(float, (N,))


def tagged(key: str, value: float = 1.0) -> Frame:
    return Frame(np.full(2, value), meta={"track": key})


def by_key(out: list[Frame], key: str = "track") -> dict:
    groups: dict = {}
    for f in out:
        groups.setdefault(f[key], []).append(float(f.data[0]))
    return groups


# --------------------------------------------------------------------------- #
# 1. per_key: an instance that finishes early must not stall the router
# --------------------------------------------------------------------------- #


def test_an_instance_that_finishes_early_retires_its_key():
    frames = [tagged("a", float(i)) for i in range(20)]
    g = from_iterable(frames) >> per_key("track", dsp.scale(gain=1.0).take(1))
    out = run(g, queue_size=1)  # used to wait for ever on the full queue
    assert [float(f.data[0]) for f in out] == [0.0]


def test_other_keys_go_on_after_one_retires():
    frames = [tagged("a" if i % 3 else "b", float(i)) for i in range(30)]
    out = run(
        from_iterable(frames) >> per_key("track", dsp.scale(gain=1.0).take(2)),
        queue_size=1,
    )
    assert by_key(out) == {"b": [0.0, 3.0], "a": [1.0, 2.0]}


def test_ttl_gives_a_retired_key_a_fresh_instance():
    frames = [tagged(k) for k in "aabbba"]
    g = from_iterable(frames) >> per_key("track", dsp.scale(gain=1.0).take(1), ttl=3)
    assert [f["track"] for f in run(g)] == ["a", "b", "a"]


@block
def refuses(x: Sig[float, N]) -> Sig[float, N]:
    raise ValueError("no")


def test_an_instance_that_raises_early_does_not_hang_the_run():
    frames = [tagged("a") for _ in range(20)]
    g = from_iterable(frames) >> per_key("track", refuses)
    with pytest.raises(ValueError, match="no"):
        run(g, queue_size=1, on_error="raise")
    out = run(g, queue_size=1, on_error="propagate")
    assert len(out) == 20 and all(f.failed for f in out)


@block
def slow(x: Sig[float, N]) -> Sig[float, N]:
    time.sleep(0.02)
    return x


@source
def endless() -> Iterator[Sig[float, N]]:
    while True:
        yield tagged("a")


def test_stopping_while_an_instance_queue_is_full():
    handle = (endless() >> per_key("track", slow)).start(queue_size=1)
    time.sleep(0.3)
    handle.stop()
    handle.wait(timeout=5)
    assert handle.state == "stopped"


# --------------------------------------------------------------------------- #
# 2. per_key: every kind of body gets state of its own, the template none
# --------------------------------------------------------------------------- #


class Counter(Stateful):
    def __init__(self) -> None:
        self.n = 0

    def __call__(self, x: Sig[float, N]) -> Sig[float, N]:
        self.n += 1
        return np.full(2, float(self.n))


@source
async def paced() -> AsyncIterator[Sig[float, N]]:
    """A, A, B, B with pauses: each key's first frame is processed before the next
    key opens, which is what exposed a shared template."""
    for key in "AABB":
        yield tagged(key, 0.0)
        await asyncio.sleep(0.02)


def graph_body(counter) -> Graph:
    g = Graph()
    i = g.input()
    c = g.add(counter)
    g.connect(i, c).output(c)
    return g


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(graph_body, id="builder-graph"),
        pytest.param(lambda c: dsp.scale(gain=1.0) >> c, id="pipeline"),
        pytest.param(
            lambda c: switch("track", {"A": c, "B": c}, default="pass"),
            id="switch",
        ),
    ],
)
def test_each_key_counts_from_one_and_the_template_is_untouched(make):
    counter = block(Counter(), name="count")
    body = make(counter)
    before = [leaf.defn for leaf in body.nodes()]
    out = run(paced() >> per_key("track", body), on_error="raise")
    assert by_key(out) == {"A": [1.0, 2.0], "B": [1.0, 2.0]}
    assert counter.fn.n == 0  # the template's object never ran
    assert [leaf.defn for leaf in body.nodes()] == before  # nor was it replaced


# --------------------------------------------------------------------------- #
# 3. gather(ordered=True): a key that completes twice while an earlier group waits
# --------------------------------------------------------------------------- #


def test_ordered_gather_keeps_every_group_of_a_repeated_key():
    seq = [("A", 0), ("B", 10), ("B", 11), ("B", 12), ("B", 13), ("A", 1)]
    frames = [Frame(np.full(1, float(v)), meta={"k": k}) for k, v in seq]
    out = run(from_iterable(frames) >> gather("k", count=2, ordered=True))
    assert [(f["group"], [float(p.data[0]) for p in f.parts]) for f in out] == [
        ("A", [0.0, 1.0]),
        ("B", [10.0, 11.0]),
        ("B", [12.0, 13.0]),
    ]


def test_unordered_gather_with_a_repeated_key():
    seq = [("B", 10), ("B", 11), ("A", 0), ("B", 12), ("A", 1), ("B", 13)]
    frames = [Frame(np.full(1, float(v)), meta={"k": k}) for k, v in seq]
    out = run(from_iterable(frames) >> gather("k", count=2))
    assert [f["group"] for f in out] == ["B", "A", "B"]


# --------------------------------------------------------------------------- #
# 4. a block made with .bind() runs on every pool
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("backend", ["thread", "process", "loky"])
def test_a_bound_block_runs_on_each_backend(backend):
    if backend == "loky":
        pytest.importorskip("loky")
    bound = dsp.scale.bind(gain=2.0)
    out = run(
        from_iterable([np.ones(2)] * 3, type=FLOATS)
        >> parallel(bound, workers=2, backend=backend),
        on_error="raise",
    )
    assert [f.data.tolist() for f in out] == [[2.0, 2.0]] * 3


def test_a_bound_block_pickles_as_its_block_bound():
    import pickle

    again = pickle.loads(pickle.dumps(dsp.scale.bind(gain=3.0)))
    assert again(np.ones(2)).tolist() == [3.0, 3.0]
    assert "gain" not in again.config


# --------------------------------------------------------------------------- #
# 5. closing a probe on the second of two blocks with the same name
# --------------------------------------------------------------------------- #


def test_closing_a_probe_by_node_detaches_it():
    second = dsp.scale(gain=2.0)
    g = from_iterable([np.ones(2)] * 20, type=FLOATS) >> dsp.scale(gain=1.0) >> second
    handle = g.start(paused=True)
    probe = handle.probe(second)  # shown as "scale#2", named "scale"
    handle.step(3)
    probe.wait(3, timeout=5)
    seen = probe.count
    probe.close()
    handle.resume()
    handle.wait()
    assert probe.count == seen
    assert all(not s.watchers for s in handle.monitor.blocks)


# --------------------------------------------------------------------------- #
# 6. a policy also governs inherited tags the block clipped or reframed
# --------------------------------------------------------------------------- #


def burst_frames(n: int = 1) -> list[Frame]:
    return [
        Frame(np.arange(8.0), meta={"t0": float(i), "seq": i}, stream={"rate": 8.0})
        .tag("burst", span=(i + 0.0, i + 0.5))
        .tag("calibrated")
        for i in range(n)
    ]


def spans_of(f: Frame) -> set:
    return {(t.name, t.span, t.origin) for t in f.tags}


def test_tags_none_drops_an_inherited_span_the_block_clipped():
    @block(tags="none")
    def crop(f: Framed[Sig[float, N]]) -> Sig[float, M]:
        return f[1:3]

    assert not run(from_iterable(burst_frames()) >> crop)[0].tags


def test_tags_none_keeps_what_the_block_attached_even_under_an_inherited_name():
    @block(tags="none")
    def crop_and_mark(f: Framed[Sig[float, N]]) -> Sig[float, M]:
        return f[1:3].tag("burst", span=(0.125, 0.25)).tag("mine")

    out = run(from_iterable(burst_frames()) >> crop_and_mark)[0]
    assert spans_of(out) == {
        ("burst", (0.125, 0.25), "crop_and_mark"),
        ("mine", None, "crop_and_mark"),
    }


def test_spans_drop_drops_a_clipped_span():
    @block(spans="drop")
    def crop(f: Framed[Sig[float, N]]) -> Sig[float, M]:
        return f[1:3]

    assert run(from_iterable(burst_frames()) >> crop)[0].tags.names == {"calibrated"}


def test_tags_none_on_a_reframing_stream_block():
    @block(tags="none")
    def halves(xs: Stream[Framed[Sig[float, N]]]) -> Iterator[Framed[Sig[float, M]]]:
        for f in xs:
            yield f[0:4]
            yield f[4:8]

    out = run(from_iterable(burst_frames(2)) >> halves)
    assert len(out) == 4 and not any(f.tags for f in out)
