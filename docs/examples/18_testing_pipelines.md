# How to test a signal chain: blocks as functions, graphs on fixtures, contracts on

`examples/18_testing_pipelines.py`

Run with `pytest examples/18_testing_pipelines.py -q` or `python examples/18_testing_pipelines.py`.

What this exercises

- unit tests of blocks as plain functions — no graph, no runtime
- graph tests on fixed frames with `from_iterable` / `graph(frames)`: known input,
  assert on output _and_ on provenance / annotations
- `check="all"`: turn the runtime contract check into an assertion for every frame
- stream blocks tested by feeding an iterable and comparing to a reference
- a fake clock: an async source whose time you control, for time-based logic
- property-style checks across a parameter grid, error-policy behaviour, and
  a regression test pinned to a recorded golden output

## Source

```python
"""How to test a signal chain: blocks as functions, graphs on fixtures, contracts on.

Run with ``pytest examples/18_testing_pipelines.py -q`` or ``python examples/18_testing_pipelines.py``.

What this exercises
  * unit tests of blocks as plain functions — no graph, no runtime
  * graph tests on fixed frames with `from_iterable` / `graph(frames)`: known input,
    assert on output *and* on provenance / annotations
  * `check="all"`: turn the runtime contract check into an assertion for every frame
  * stream blocks tested by feeding an iterable and comparing to a reference
  * a fake clock: an async source whose time you control, for time-based logic
  * property-style checks across a parameter grid, error-policy behaviour, and
    a regression test pinned to a recorded golden output
"""
import asyncio
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from sigflow import Frame, Framed, MISSING, Sig, Stream, TypeMismatch, block, from_iterable, run, source  # noqa: E402
from sigflow import dsp  # noqa: E402

RATE = 1000.0


# ----------------------------------------------------------- the code under test

@block
def remove_dc(x: Sig[float, "N"]) -> Sig[float, "N"]:
    return x - x.mean()


@block
def peak_frequency(f: Framed[Sig[float, "N"]]) -> Sig[dict]:
    X = np.abs(np.fft.rfft(f.data * np.hanning(len(f.data))))
    k = int(np.argmax(X[1:])) + 1
    return {"freq_hz": float(k * f["rate"] / len(f.data)), "frame": f.get("seq", None)}


@block
def smooth(xs: Stream[Sig[float, "N"]], alpha: float = 0.5) -> Sig[float, "N"]:
    acc = None
    for x in xs:
        acc = x if acc is None else (1 - alpha) * acc + alpha * x
        yield acc


@block
def rate_limiter(events: Stream[Framed[Sig[dict]]], min_gap: float = 1.0) -> Sig[dict]:
    """Drop events that arrive within `min_gap` seconds of the last accepted one."""
    last = -np.inf
    for f in events:
        if f.data["t"] - last >= min_gap:
            last = f.data["t"]
            yield f.data


chain = remove_dc >> peak_frequency           # a source-less graph: the thing we ship


# ------------------------------------------------------------------ fixtures

def tone(freq: float, n: int = 256, offset: float = 0.0, seq: int = 0) -> Frame:
    t = np.arange(n) / RATE
    return Frame(np.sin(2 * np.pi * freq * t) + offset, meta={"seq": seq}, stream={"rate": RATE})


# ------------------------------------------------------- 1. blocks are functions

def test_remove_dc_as_a_plain_function():
    x = np.ones(8) + np.arange(8)
    assert abs(remove_dc(x).mean()) < 1e-12
    assert remove_dc.in_type is not None and str(remove_dc.out_type) == "Sig[float64, (N,)]"


def test_smooth_against_a_reference_implementation():
    xs = [np.full(4, v) for v in (1.0, 3.0, 3.0, 3.0)]
    out = list(smooth(iter(xs), alpha=0.5))                 # stream blocks take any iterable
    ref, acc = [], None
    for x in xs:
        acc = x if acc is None else 0.5 * acc + 0.5 * x
        ref.append(acc)
    assert all(np.allclose(a, b) for a, b in zip(out, ref))


# --------------------------------------------------------- 2. graphs on fixtures

def test_chain_on_known_frames_checks_values_and_lineage():
    frames = [tone(50, offset=2.0, seq=0), tone(125, seq=1)]
    out = chain(frames)                                    # apply the graph to in-memory frames
    assert [round(f.data["freq_hz"]) for f in out] == [51, 125]      # bin resolution is RATE/256 ~ 3.9 Hz
    assert out[1].data["frame"] == 1 and out[1]["rate"] == RATE      # annotations travelled
    assert [n.op for n in out[1].prov.chain()] == ["peak_frequency", "remove_dc", "input"]


def test_types_are_enforced_at_build_time():
    with pytest.raises(TypeMismatch):
        dsp.sine() >> peak_frequency >> remove_dc          # dict cannot feed a float vector
    with pytest.raises(TypeMismatch):
        from_iterable([np.zeros((2, 2))], Sig[float, ("M", "N")]) >> remove_dc


def test_every_frame_is_checked_against_the_contract():
    @block
    def lies(x: Sig[float, "N"]) -> Sig[float, "N"]:
        lies.n += 1
        return np.zeros(len(x) + (lies.n > 1))

    lies.n = 0
    assert len(run(from_iterable([np.ones(4)] * 3, Sig[float, "N"]) >> lies)) == 3   # default: first frame only
    lies.n = 0
    with pytest.raises(TypeMismatch, match="runtime check"):
        run(from_iterable([np.ones(4)] * 3, Sig[float, "N"]) >> lies, check="all")


# ------------------------------------------------------------- 3. a fake clock

class FakeClock:
    """Deterministic time for an async source: tests advance it explicitly."""

    def __init__(self) -> None:
        self.now = 0.0

    def tick(self, dt: float) -> None:
        self.now += dt


@source
async def events(clock: FakeClock, schedule: tuple = ()) -> Sig[dict]:
    """Emit one event per scheduled time, advancing the fake clock to it."""
    for t in schedule:
        clock.tick(t - clock.now)
        await asyncio.sleep(0)                               # yield to the loop, no real waiting
        yield {"t": clock.now}


def test_rate_limiter_with_a_fake_clock():
    clock = FakeClock()
    out = run(events(clock, schedule=(0.0, 0.2, 0.9, 1.0, 1.5, 2.1, 2.9, 3.0)) >> rate_limiter(min_gap=1.0))
    assert [f.data["t"] for f in out] == [0.0, 1.0, 2.1]
    assert clock.now == 3.0                                  # the test, not the wall clock, drove time


# ------------------------------------------------- 4. properties, policies, golden

@pytest.mark.parametrize("freq", [20, 60, 140, 310])
def test_peak_frequency_within_one_bin_for_any_tone(freq):
    out = chain(tone(freq))
    assert abs(out[0].data["freq_hz"] - freq) <= RATE / 256


def test_error_policies_are_what_the_docs_say():
    @block
    def boom(x: Sig[float, "N"]) -> Sig[float, "N"]:
        raise RuntimeError("no")

    src = lambda: from_iterable([np.ones(4)] * 3, Sig[float, "N"])   # noqa: E731
    out = run(src() >> boom >> remove_dc)
    assert all(f.failed and f.data is MISSING for f in out) and out[0].errors[0].origin == "boom"
    assert run(src() >> boom, on_error="drop") == []
    with pytest.raises(RuntimeError):
        run(src() >> boom, on_error="raise")


GOLDEN = [51.0, 125.0, 250.0]      # recorded once from a known-good version


def test_regression_against_golden_output():
    out = chain([tone(50), tone(125), tone(250)])
    assert [round(f.data["freq_hz"], 1) for f in out] == pytest.approx(GOLDEN, abs=RATE / 256)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
```
