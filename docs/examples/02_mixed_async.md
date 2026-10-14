# Mixing synchronous streaming with asynchronous event sources

`examples/02_mixed_async.py`

A sampled signal (synchronous generator) is zipped with an asynchronous
telemetry ticker; a merged stream interleaves both as they arrive.

## Output

Files the script writes when run:

![mixed_async_graph.svg](assets/02_mixed_async/mixed_async_graph.svg)

*`mixed_async_graph.svg`*

![mixed_async_merge.svg](assets/02_mixed_async/mixed_async_merge.svg)

*`mixed_async_merge.svg`*

## Source

```python
"""Mixing synchronous streaming with asynchronous event sources.

A sampled signal (synchronous generator) is zipped with an asynchronous
telemetry ticker; a merged stream interleaves both as they arrive.
"""

import asyncio

import numpy as np

from sigflow import Sig, Stream, block, merge, run, source
from sigflow import dsp


@source(total=lambda count, **_: count)
async def gain_control(count: int = 5, period: float = 0.02) -> Sig[dict]:
    """Pretend AGC commands arriving over the network."""
    for i in range(count):
        await asyncio.sleep(period)
        yield {"gain": 1.0 + i}


@block
def apply_gain(x: Sig[float, "N"], cmd: Sig[dict]) -> Sig[float, "N"]:
    return x * cmd["gain"]


@block
async def smooth(xs: Stream[Sig[float, "N"]], alpha: float = 0.5) -> Sig[float, "N"]:
    """An *async* stream block: identical shape to a sync one, with `async for`."""
    acc = None
    async for x in xs:
        acc = x if acc is None else (1 - alpha) * acc + alpha * x
        yield acc


async def main() -> None:
    # zip: lock-step pairing of a sync source and an async source
    g = (
        (dsp.sine(n=16, frames=5) & gain_control(count=5))
        >> apply_gain
        >> smooth(alpha=0.3)
        >> dsp.rms
    )
    g.draw().save("mixed_async_graph.svg")
    async for f in g:  # graphs are async-iterable
        print(f"rms={f.data:.3f}  parents={[p.op for p in f.prov.parents]}")

    # merge: interleave heterogeneous events as they arrive
    events = merge(
        dsp.ticker(period=0.005, count=3), gain_control(count=3, period=0.004)
    )
    events.draw().save("mixed_async_merge.svg")
    out = await events.arun()
    print([f.data for f in out])


if __name__ == "__main__":
    asyncio.run(main())
    # ... and the very same graph runs from plain sync code:
    print(
        run((dsp.sine(n=16, frames=2) & gain_control(count=2)) >> apply_gain)[1].data[
            :3
        ]
    )
```
