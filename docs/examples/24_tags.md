# Tags: labels on frames, or on a time span of them, that the runtime carries

`examples/24_tags.py`

Run ``python examples/24_tags.py``. A short synthetic recording goes through a
detector that tags every frame with its level, speech frames with ``speech`` and
overdriven stretches with a ``clipped`` span in seconds. A generic block in the
middle knows nothing about tags and carries them through. Frames tagged
``speech`` are routed to a recogniser that *requires* the tag in its contract
(``Annotated[Sig[...], requires("speech")]``) and clears it on the way out. The
script then shows what happens to spans when a frame is re-framed into smaller
chunks (they are clipped to each chunk), what a reduction keeps (whole-frame tags
only), that tags round-trip through ``Frame.save``, and that wiring the recogniser
after a producer that does not guarantee the tag is rejected before anything runs.

## Source

```python
"""Tags: labels on frames, or on a time span of them, that the runtime carries.

Run ``python examples/24_tags.py``. A short synthetic recording goes through a
detector that tags every frame with its level, speech frames with ``speech`` and
overdriven stretches with a ``clipped`` span in seconds. A generic block in the
middle knows nothing about tags and carries them through. Frames tagged
``speech`` are routed to a recogniser that *requires* the tag in its contract
(``Annotated[Sig[...], requires("speech")]``) and clears it on the way out. The
script then shows what happens to spans when a frame is re-framed into smaller
chunks (they are clipped to each chunk), what a reduction keeps (whole-frame tags
only), that tags round-trip through ``Frame.save``, and that wiring the recogniser
after a producer that does not guarantee the tag is rejected before anything runs.
"""

import os
import tempfile
from typing import Annotated

import numpy as np

from sigflow import (
    Frame,
    Framed,
    Sig,
    SignalType,
    TypeMismatch,
    block,
    clears,
    dsp,
    from_iterable,
    provides,
    requires,
    run,
    when,
)
from sigflow.dims import N

RATE = 8000.0
FRAME = 256
CONTRACT = SignalType(float, (FRAME,), RATE, "V")


def recording() -> list[Frame]:
    """Eight frames of 32 ms: a tone that comes and goes, one frame overdriven."""
    rng = np.random.default_rng(0)
    frames = []
    for i in range(8):
        t = (i * FRAME + np.arange(FRAME)) / RATE
        if i in (1, 2, 3, 6):
            x = 0.5 * np.sin(2 * np.pi * 440.0 * t)
        else:
            x = 0.01 * rng.standard_normal(FRAME)
        if i == 2:
            x[96:160] = 1.5  # the amplifier sits at its rail for 8 ms
        frames.append(
            Frame(
                np.clip(x, -1.0, 1.0),
                meta={"t0": i * FRAME / RATE, "seq": i},
                stream={"rate": RATE, "units": "V"},
            )
        )
    return frames


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Start and stop indices of each run of True."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(int), [0]))))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


# --------------------------------------------------------------------- blocks


@block
def detect(f: Framed[Sig[float, N]]) -> Annotated[Sig[float, N], provides("level")]:
    """Tag the level of every frame (a guarantee), ``speech`` when it is loud enough
    (conditional, so not promised) and each overdriven stretch as a span."""
    level = float(np.sqrt(np.mean(f.data**2)))
    out = f.tag(level=round(level, 3))
    if level > 0.1:
        out = out.tag("speech")
    span = f.time
    assert span is not None
    for start, stop in runs(np.abs(f.data) >= 1.0):
        out = out.tag("clipped", span=(span.t0 + start / RATE, span.t0 + stop / RATE))
    return out


@block
def transcribe(
    f: Framed[Annotated[Sig[float, N], requires("speech")]],
) -> Annotated[Sig[float, N], clears("speech")]:
    """A stand-in recogniser: it may only run on speech, and what it hands on is
    no longer "speech to be recognised". A clipped stretch is worth a word."""
    text = "la" * int(round(f["level"] * 10))
    if "clipped" in f:
        text += " (distorted)"
    return f.tag(text=text)


# ----------------------------------------------------------------------- main


def main() -> None:
    source = from_iterable(recording(), type=CONTRACT)
    graph = source >> detect >> dsp.scale(gain=2.0) >> when("speech", then=transcribe)

    print("Static view: what each stage guarantees")
    for stage in graph.stages[1:]:
        print(f"  {stage.name:<22} {stage.out_type}")

    print("\nFrames out, with their tags")
    out = run(graph, check="all", on_error="raise")
    for f in out:
        tags = ", ".join(sorted(map(repr, f.tags)))
        print(f"  #{f['seq']} t0={f.t0:.3f}s via {f.prov.op:<10} {tags}")

    distorted = next(f for f in out if "clipped" in f)
    first = next(t for t in distorted.tags if t.name == "clipped")
    print("\nA span is clipped to the frame it lives in, and remembers who set it:")
    print(f"  {first!r} origin={first.origin!r}")
    print(f"  samples covered: {int(distorted.mask('clipped').sum())} of {FRAME}")

    print("\nRe-framed into 64-sample chunks, the span follows each chunk:")
    chunks = run(source >> detect >> dsp.chunk(64), check="all", on_error="raise")
    for c in chunks:
        if "clipped" in c:
            print(f"  t0={c.t0:.4f}s  {sorted(map(repr, c.tags))}")

    print("\nA reduction keeps whole-frame tags and drops spans:")
    levels = run(source >> detect >> dsp.rms, check="all", on_error="raise")
    print(f"  {sorted(map(repr, levels[2].tags))}")

    with tempfile.TemporaryDirectory() as tmp:
        path = distorted.save(os.path.join(tmp, "frame.npz"))
        back = Frame.load(path)
        print("\nTags round-trip through Frame.save:", back.tags == distorted.tags)

    print("\nA requirement is checked before anything runs:")
    try:
        _ = source >> transcribe
    except TypeMismatch as e:
        print(f"  {e}")


if __name__ == "__main__":
    main()
```
