# Basics: declare typed blocks, compose with operators, run, inspect provenance

`examples/01_basics.py`



## Source

```python
"""Basics: declare typed blocks, compose with operators, run, inspect provenance."""

from typing import Iterator, Literal
import numpy as np

from sigflow import Frame, Framed, Sig, Stream, TypeMismatch, block, dsp, run, source
from sigflow.dims import N, T


# A block is a plain function with typed signal inputs/outputs and config kwargs.
@block
def scale(x: Sig[T, N], gain: float = 1.0) -> Sig[T, N]:
    return x * gain


@block
def energy(x: Sig[T, N]) -> Sig[float, tuple[()]]:
    return float(np.sum(np.abs(x) ** 2))


# A stream block is a generator: it sees the whole stream and may keep state.
@block
def pairs(xs: Stream[Sig[float, N]]) -> Iterator[Sig[float, tuple[Literal[2], N]]]:
    buf = []
    for x in xs:
        buf.append(x)
        if len(buf) == 2:
            yield np.stack(buf)
            buf = []


# A frame-aware block can read and write annotations and attach issues.
@block
def check_level(f: Framed[Sig[float, N]], limit: float = 0.9) -> Sig[float, N]:
    mask = np.abs(f.data) > limit
    out = f.annotate(peak=float(np.abs(f.data).max())).annotate_samples(hot=mask)
    return (
        out.with_issue(f"{mask.sum()} hot samples", "warning", "check_level")
        if mask.any()
        else out
    )


if __name__ == "__main__":
    # Blocks are still functions:
    print(scale(np.arange(4.0), gain=2))

    g = (
        dsp.sine(freq=440, rate=8000, n=64, frames=6, amplitude=1.2)
        >> check_level
        >> scale(gain=0.5)
        >> energy
    )
    print(g.describe(), "\n")
    g.draw().save("basics_graph.svg")

    frames = run(g, progress=True)
    f = frames[-1]
    print(f)  # Frame(float, meta={...}, issues=1, from=energy)
    print("warnings:", [str(w) for w in f.warnings])
    print("stream annotations:", dict(f.stream))
    print("lineage:\n" + f.trace())

    # Type errors are raised while *building* the graph:
    try:
        _ = dsp.sine() >> energy >> scale
    except TypeMismatch as e:
        print("\nrejected:", e)

    # Dimensions derived from configuration are checked too:
    try:
        _ = (dsp.sine(n=64) & dsp.sine(n=128)) >> (lambda ab: ab)
        print(((dsp.sine(n=64) & dsp.noise(n=64)) >> scale).describe())
    except TypeMismatch as e:
        print("rejected:", e)

    # Stream blocks and provenance of windowed outputs
    out = run(dsp.sine(n=8, frames=4) >> pairs)
    print("\npairs consumed:", [p.seq for p in out[0].prov.parents])
```
