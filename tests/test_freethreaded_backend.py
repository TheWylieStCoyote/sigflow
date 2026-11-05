"""parallel(block, backend="freethreaded" | "auto") and free-threaded Python.

Most tests stand in for the interpreter by patching ``sigflow.pools.free_threaded`` and
``gil_enabled``, so they run on any build; the last runs for real on python3.13t/3.14t.
"""

from __future__ import annotations

import threading
import warnings

import numpy as np
import pytest

from sigflow import SignalType, block, dsp, from_iterable, parallel, pools, run
from sigflow.dims import N

FLOATS = SignalType(float, (N,))
frames = [np.full(4, float(i)) for i in range(12)]


def source():
    return from_iterable(frames, type=FLOATS)


def doubled(fn=lambda x: 2 * x):
    return block(fn, inputs=FLOATS, outputs=FLOATS, name="doubled")


def values(out) -> list[float]:
    return [float(f.data[0]) for f in out]


@pytest.fixture
def no_gil(monkeypatch):
    monkeypatch.setattr(pools, "free_threaded", lambda: True)
    monkeypatch.setattr(pools, "gil_enabled", lambda: False)


@pytest.fixture
def gil_back_on(monkeypatch):
    monkeypatch.setattr(pools, "free_threaded", lambda: True)
    monkeypatch.setattr(pools, "gil_enabled", lambda: True)


def test_unknown_backend_lists_the_choices():
    with pytest.raises(ValueError, match="'process', 'thread', 'freethreaded', 'auto'"):
        parallel(dsp.rms, backend="gpu")


def test_freethreaded_is_refused_on_a_build_with_a_gil(monkeypatch):
    monkeypatch.setattr(pools, "free_threaded", lambda: False)
    with pytest.raises(RuntimeError, match="needs a free-threaded build"):
        parallel(dsp.rms, backend="freethreaded")


def test_freethreaded_runs_anything_in_order(no_gil):
    """Threads: a lambda is fine (a process pool could not pickle it)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = run(source() >> parallel(doubled(), workers=3, backend="freethreaded"))
    assert values(out) == [2.0 * i for i in range(12)]


def test_a_failing_frame_is_a_failed_frame(no_gil):
    def picky(x):
        if x[0] == 5:
            raise ValueError("five")
        return x

    out = run(source() >> parallel(doubled(picky), backend="freethreaded"))
    assert [len(f.issues) for f in out] == [0] * 5 + [1] + [0] * 6


def test_freethreaded_warns_when_the_gil_is_back_on(gil_back_on):
    with pytest.warns(RuntimeWarning, match="the GIL is enabled"):
        out = run(source() >> parallel(doubled(), backend="freethreaded"))
    assert values(out) == [2.0 * i for i in range(12)]


def test_auto_picks_threads_without_a_gil_and_processes_with_one(monkeypatch):
    monkeypatch.setattr(pools, "gil_enabled", lambda: False)
    assert pools.resolve("auto", "x") == "freethreaded"
    out = run(source() >> parallel(doubled(), backend="auto"))  # a lambda: threads
    assert values(out) == [2.0 * i for i in range(12)]

    monkeypatch.setattr(pools, "gil_enabled", lambda: True)
    assert pools.resolve("auto", "x") == "process"
    out = run(source() >> parallel(dsp.scale(gain=3.0), workers=2, backend="auto"))
    assert values(out) == [3.0 * i for i in range(12)]


@pytest.mark.skipif(
    not pools.free_threaded() or pools.gil_enabled(),
    reason="needs a free-threaded build with the GIL off",
)
def test_frames_run_at_the_same_time_on_a_free_threaded_build():
    workers = 4
    meet = threading.Barrier(workers, timeout=30)  # passes only when 4 calls overlap
    names = set()

    def together(x):
        names.add(threading.current_thread().name)
        meet.wait()
        return x

    out = run(source() >> parallel(doubled(together), workers=workers, backend="auto"))
    assert values(out) == [float(i) for i in range(12)]
    assert len(names) == workers
    assert not pools.gil_enabled()  # sigflow and its imports left the GIL off
