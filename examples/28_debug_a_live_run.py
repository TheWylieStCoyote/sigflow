"""Debugging a running graph: probe a block, hold a frame, fix it, carry on.

Run ``python examples/28_debug_a_live_run.py``. A level meter reports one reading
far too high. Instead of adding prints, the script follows the debugging workflow on
the live graph:

1. run the graph and notice the bad reading, and which source frame it came from;
2. start it again, paused, and **probe** the calibration block to watch its output;
3. **hold** the frame that looks wrong with a breakpoint, before anything downstream
   sees it, and inspect its data, annotations and provenance;
4. **replace** it with a corrected frame (``drop_held()`` would drop it instead) and
   resume;
5. let a second breakpoint, ``break_on_error``, catch a frame whose block failed;
6. save a static **run report** (HTML) and close the probe and the run.

Writes ``28_debug_report.html`` next to where it runs.
"""

from collections.abc import Iterator

import numpy as np

from sigflow import Frame, Sig, block, run, source
from sigflow.dims import N

RATE = 1_000.0
GLITCH, BROKEN = 7, 11  # the frames that go wrong


@source
def sensor(frames: int = 16) -> Iterator[Sig[float, N]]:
    """A sensor's raw counts, 64 per frame. Frame 7 has a glitch: one huge spike.
    Frame 11 arrives empty."""
    rng = np.random.default_rng(3)
    for i in range(frames):
        x = 100 + rng.normal(0, 2, 64)
        if i == GLITCH:
            x[20] = 60_000.0
        if i == BROKEN:
            x = np.array([])
        yield Frame(x, meta={"seq": i, "t0": i * 64 / RATE}, stream={"rate": RATE})


@block
def calibrate(x: Sig[float, N], gain: float = 0.01) -> Sig[float, N]:
    """Counts to volts."""
    if x.size == 0:
        raise ValueError("empty frame")
    return x * gain


@block
def level(x: Sig[float, N]) -> Sig[float, tuple[()]]:
    """The frame's peak level in volts."""
    return float(np.max(np.abs(x)))


def graph():
    return sensor() >> calibrate >> level


def path(f: Frame) -> str:
    """A frame's lineage as block names, newest first."""
    return " <- ".join(p.op for p in f.prov.chain())


def main() -> None:
    # 1. notice the problem ------------------------------------------------------
    out = run(graph())
    readings = {f.prov.roots()[0].seq: f for f in out}
    bad = max((f for f in out if not f.failed), key=lambda f: f.data)
    print(f"1. peak readings around 1 V, but frame {bad['seq']} reads {bad.data:.0f} V")
    print(f"   it came from source frame {bad.prov.roots()[0].seq} by {path(bad)}")
    print(f"   and frame {BROKEN} failed: {readings[BROKEN].errors[0].message}")

    # 2. probe the calibration block on a paused run ------------------------------
    h = graph().start(paused=True, name="debug-level")
    probe = h.probe("calibrate", keep=4)
    h.step(3)
    probe.wait(3, timeout=5)
    print(
        f"\n2. probing calibrate: {probe.count} frames, last peak "
        f"{np.max(probe.latest.data):.2f} V"
    )

    # 3. hold the frame that looks wrong -----------------------------------------
    h.break_when("calibrate", lambda f: np.max(np.abs(f.data)) > 10, once=True)
    h.break_on_error()
    h.resume()
    held = h.wait_break(timeout=5)
    assert held is not None
    print(
        f"\n3. held at {h.held_at} (state {h.state}): seq {held['seq']}, "
        f"t0 {held['t0']:.3f} s, max {np.max(held.data):.0f} V at sample "
        f"{int(np.argmax(held.data))}"
    )
    print(f"   issues: {held.issues or 'none'}; lineage: {path(held)}")

    # 4. replace it with a corrected frame and resume ------------------------------
    fixed = np.clip(held.data, None, np.median(held.data) * 1.5)
    h.replace_held(held.evolve(data=fixed))
    print(f"\n4. replaced: max now {np.max(fixed):.2f} V; resumed")

    # 5. break_on_error holds the frame whose block failed -------------------------
    failed = h.wait_break(timeout=5)
    assert failed is not None and failed.failed
    print(f"\n5. held a failed frame at {h.held_at}: {failed.errors[0].message}")
    h.drop_held()  # nothing useful in it: drop it
    frames = h.wait(timeout=10)

    # 6. a static report, then clean up --------------------------------------------
    report = h.report()
    print(
        f"\n6. {len(frames)} readings after the fix, peak {max(f.data for f in frames):.2f} V"
    )
    print(f"   report written to {report.save('28_debug_report.html')}")
    probe.close()
    h.close()
    print(f"   run {h.name!r} is {h.state}; probe closed")


if __name__ == "__main__":
    main()
