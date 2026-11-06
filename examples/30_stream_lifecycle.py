"""Streams end, and get stopped: who closes what, and when.

Run ``python examples/30_stream_lifecycle.py``. The source logs when it opens and when
its ``finally`` runs, standing in for a file or a device it would release. The script
consumes it in each way a stream is consumed, and shows the log:

1. ``run()`` to the end;
2. a ``for`` loop that breaks early: the source is closed as the loop ends;
3. an iterator kept in a variable, closed with ``contextlib.closing``;
4. ``async for`` that breaks: without ``aclosing`` the source stays open until the
   event loop ends; with it, it is closed within a few turns of the loop (closing
   runs asynchronously, stage by stage), so a moment's ``await`` later it is done;
5. a background run with a bounded buffer (``start(keep=...)``, ``drain()``), then
   ``stop()``;
6. ``run(sink=..., collect=False)``: nothing kept, every frame seen;
7. a file sink, which writes its file when its ``with`` block closes it.

See the guide *Streams and their lifecycle*.
"""

import asyncio
import contextlib
import os
import time
from collections.abc import Iterator

import numpy as np

from sigflow import Sig, dsp, io, iterate, iterate_sync, run, source
from sigflow.dims import N

log: list[str] = []


@source
def device(frames: int = 1_000_000) -> Iterator[Sig[float, N]]:
    """Frames from a pretend device; the log shows when it is opened and released."""
    log.append("open")
    try:
        for i in range(frames):
            yield np.full(4, float(i))
    finally:
        log.append("released")


def show(step: str, extra: str = "") -> None:
    print(f"{step:44s} log {log}{'  ' + extra if extra else ''}")
    log.clear()


def main() -> None:
    run(device(frames=3))
    show("1. run() to the end")

    for f in device():
        if f.data[0] >= 4:
            break
    show("2. for ... break")

    with contextlib.closing(iterate_sync(device())) as frames:
        first = next(frames)
    show("3. closing(iterate_sync(...))", f"first frame {first.data[0]:.0f}")

    async def async_break() -> None:
        async for f in device():
            if f.data[0] >= 4:
                break
        show("4a. async for ... break (still open)")

    asyncio.run(async_break())
    show("    ... after the event loop ended")

    async def async_closing() -> None:
        async with contextlib.aclosing(iterate(device())) as frames:
            async for f in frames:
                if f.data[0] >= 4:
                    break
        show("4b. aclosing(iterate(...)) ... break")
        await asyncio.sleep(0.01)  # closing finishes within a few turns of the loop
        show("    ... a moment later")

    asyncio.run(async_closing())

    h = device().start(keep=50)
    time.sleep(0.2)
    kept = h.drain()
    h.stop()
    show("5. start(keep=50), drain(), stop()", f"drained {len(kept)}, state {h.state}")

    seen = []
    out = run(device(frames=500), sink=lambda f: seen.append(f.data[0]), collect=False)
    show("6. run(sink=..., collect=False)", f"kept {len(out)}, saw {len(seen)}")

    with io.wav_sink("30_tone.wav", rate=8_000) as sink:
        run(dsp.sine(freq=440.0, rate=8_000.0, n=800, frames=10) >> sink)
        written = os.path.exists("30_tone.wav")
    size = os.path.getsize("30_tone.wav")
    print(
        f"{'7. io.wav_sink in a with block':44s} file during run: {written}, after: {size} bytes"
    )


if __name__ == "__main__":
    main()
