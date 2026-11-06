# Your first graph

Five minutes: write two blocks, wire them, run the graph three ways, and read
what comes back.

## Blocks are typed functions

```python
import numpy as np
from sigflow import Sig, SignalType, block, from_iterable, run
from sigflow.dims import N


@block
def scale(x: Sig[float, N], gain: float = 2.0) -> Sig[float, N]:
    return x * gain


@block
def total(x: Sig[float, N]) -> Sig[float, tuple[()]]:
    return float(x.sum())
```

`Sig[float, N]` is a contract: a float vector of some length `N`.
`Sig[float, tuple[()]]`, a shape with no axes, is a single float. Parameters after the signal, such as `gain`, are the
block's configuration. A block is still a function: `scale(np.ones(3))` returns
`array([2., 2., 2.])`.

## Wire and run

```python
frames = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
graph = from_iterable(frames, type=SignalType(float, (N,))) >> scale(gain=10.0) >> total

out = run(graph)
print([f.data for f in out])  # [60.0, 150.0]
```

`from_iterable` turns a list into a source, `>>` connects blocks, and
`scale(gain=10.0)` configures a block in the graph. `run` returns the output
**frames**, one per input frame here.

The contracts are checked while you wire. Feeding the scalar `total` into
`scale`, which wants a vector, fails before anything runs:

```python
from_iterable(frames, type=SignalType(float, (N,))) >> total >> scale
# TypeMismatch: total -> scale: rank 0 produced, rank 1 required (Sig[float64, ()] -> Sig[float64, (N,)])
```

## What a frame holds

```python
f = out[1]
f.data  # 150.0: the payload
f.failed, f.issues  # False, (): nothing went wrong
f.meta  # {}: annotations (sources add "seq", "t0", rates...)
f.prov.chain()  # total <- scale <- list: how it was computed
f.prov["scale"].params  # {'gain': 10.0}: with which configuration
```

Every frame carries its **provenance**, the blocks and source frames it came
from, so any result can be traced and recomputed (see
[Lineage](../concepts/lineage.md)). A block that raises does not stop the run
by default: its frame comes out with `failed=True` and the error in `issues`
(see [Blocks](../concepts/blocks.md)).

## The same graph, three ways

```python
for f in graph:  # one frame at a time; break whenever you like
    print(f.data)


async def main():
    async for f in graph:  # the same inside an event loop
        print(f.data)


handle = graph.start()  # in the background: a Run handle
handle.wait()  # [Frame, Frame]
```

`run()` also works in a Jupyter cell. [Running graphs](../reference/running.md)
lists the options every entry point takes.

## Where next

- **Examples, in order:** `01_basics.py` (operators, provenance, stream
  blocks), then the [learning paths](../examples/index.md).
- **Concepts:** [Frames](../concepts/frames.md), [Blocks](../concepts/blocks.md),
  [Contracts](../concepts/contracts.md).
- **Graphs that operators cannot express:**
  [Composing graphs](../guide/composing.md).
