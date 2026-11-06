"""Stop a long run part-way, save a checkpoint, and resume it later, exactly.

Run ``python examples/29_checkpoint_resume.py``. The graph integrates a sensor's
readings with a ``Stateful`` block, whose running total is its state. The script

1. runs the whole graph once, uninterrupted, as the reference;
2. starts it again in the background, lets 40 frames through, pauses, and saves a
   checkpoint: how many frames the source had emitted, and the block's state;
3. builds a *fresh* graph from the same function, as a new process would, and runs
   it with ``resume=`` the checkpoint file;
4. checks that the resumed outputs equal the rest of the reference run.

Two things make the resume exact: the graph is rebuilt by the same code (states are
matched to blocks by name), and the source is deterministic (seeded), because a
resumed source regenerates the frames it already emitted and skips them. The last
section shows what a checkpoint cannot capture: a generator stream block's state,
listed under ``skipped``.

Writes ``29_checkpoint.pkl`` next to where it runs.
"""

import time
from collections.abc import Iterator

import numpy as np

from sigflow import (
    Frame,
    Sig,
    Stateful,
    Stream,
    block,
    dsp,
    load_checkpoint,
    run,
    source,
)
from sigflow.dims import N

CUT = 40  # frames before the checkpoint


@source
def sensor(frames: int = 100, seed: int = 11) -> Iterator[Sig[float, N]]:
    """Deterministic readings: the same seed gives the same frames every run."""
    rng = np.random.default_rng(seed)
    for i in range(frames):
        yield Frame(rng.normal(1.0, 0.1, 8), meta={"seq": i})


class Integrate(Stateful):
    """A running total of every sample seen. ``Stateful`` gives it ``state()`` and
    ``restore()`` (its attributes), which is what a checkpoint saves."""

    def __init__(self) -> None:
        self.total = 0.0

    def __call__(self, x: Sig[float, N]) -> Sig[float, tuple[()]]:
        self.total += float(np.sum(x))
        return self.total


def make_graph():
    """A fresh graph with fresh state: a resumed run must not share the old objects."""
    return sensor() >> block(Integrate(), name="integrate")


def main() -> None:
    reference = [f.data for f in run(make_graph())]
    print(
        f"1. uninterrupted: {len(reference)} outputs, final total {reference[-1]:.3f}"
    )

    h = make_graph().start(paused=True)
    h.step(CUT)  # let CUT source frames through, then stay paused
    while h.monitor.produced < CUT:
        time.sleep(0.01)
    ck = h.checkpoint("29_checkpoint.pkl")
    h.stop()
    print(f"\n2. checkpoint after {ck.produced} outputs:")
    print(f"   source positions {ck.positions}")
    print(f"   block states     {ck.states}")

    resumed = [
        f.data for f in run(make_graph(), resume=load_checkpoint("29_checkpoint.pkl"))
    ]
    print(f"\n3. resumed run: {len(resumed)} outputs, final total {resumed[-1]:.3f}")

    same = np.allclose(resumed, reference[CUT:])
    print(
        f"\n4. resumed outputs equal the reference's last {len(reference) - CUT}: {same}"
    )
    assert same

    # what a checkpoint cannot capture -----------------------------------------------
    @block
    def smooth(xs: Stream[Sig[float, N]]) -> Iterator[Sig[float, N]]:
        """A generator stream block: its state lives in a suspended generator."""
        acc = None
        for x in xs:
            acc = x if acc is None else 0.9 * acc + 0.1 * x
            yield acc

    h = (sensor() >> smooth >> dsp.rms).start(paused=True)
    h.step(5)
    while h.monitor.produced < 5:
        time.sleep(0.01)
    ck = h.checkpoint()
    h.stop()
    print(
        f"\n5. with a generator stream block: states {ck.states}, skipped {ck.skipped}"
    )
    print(
        "   (a resumed run starts it fresh; write such a block as a Stateful map block)"
    )


if __name__ == "__main__":
    main()
