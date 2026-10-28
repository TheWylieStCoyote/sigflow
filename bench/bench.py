"""Per-frame overhead of the runtime. Run: python bench/bench.py  (writes docs/performance.md)."""

import os
import sys
import time

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)
from sigflow import (
    Sig,
    Stream,
    block,
    dsp,
    run,
    source,
)

FRAMES = 5000
N = 1024


@source(total=lambda frames, **_: frames)
def src(frames: int = FRAMES) -> Sig[float, "N"]:
    x = np.ones(N)
    for _ in range(frames):
        yield x


@block
def ident(x: Sig[float, "N"]) -> Sig[float, "N"]:
    return x


@block
def passthrough(xs: Stream[Sig[float, "N"]]) -> Sig[float, "N"]:
    yield from xs


@block
async def apassthrough(xs: Stream[Sig[float, "N"]]) -> Sig[float, "N"]:
    async for x in xs:
        yield x


def timed(label, graph, frames=FRAMES, **kw):
    t0 = time.perf_counter()
    run(graph, collect=False, **kw)
    dt = time.perf_counter() - t0
    per = dt / frames * 1e6
    print(f"{label:46s} {dt:6.3f}s  {per:7.1f} us/frame  {frames / dt:9.0f} frames/s")
    return label, per


def main():
    rows = []
    rows.append(timed("source only", src()))
    rows.append(timed("source >> 1 map block", src() >> ident))
    rows.append(
        timed(
            "source >> 5 map blocks", src() >> ident >> ident >> ident >> ident >> ident
        )
    )
    rows.append(
        timed(
            "source >> 5 map blocks, check='all'",
            src() >> ident >> ident >> ident >> ident >> ident,
            check="all",
        )
    )
    rows.append(timed("source >> sync generator stream block", src() >> passthrough))
    rows.append(timed("source >> async generator stream block", src() >> apassthrough))
    rows.append(timed("source >> fan-out of 2 >> select", src() >> (ident | ident)[0]))
    rows.append(
        timed(
            "source >> fft >> power_db (real work, N=1024)",
            src() >> dsp.to_complex >> dsp.fft >> dsp.power_db,
        )
    )
    x = np.ones(N)
    t0 = time.perf_counter()
    for _ in range(FRAMES):
        np.fft.fft(x.astype(complex))
    base = (time.perf_counter() - t0) / FRAMES * 1e6
    print(f"{'(plain numpy fft of the same frames)':46s} {'':8s} {base:7.1f} us/frame")
    import platform

    with open(
        os.path.join(os.path.dirname(__file__), "..", "docs", "performance.md"), "w"
    ) as fh:
        fh.write(
            "# Runtime overhead\n\nMeasured with `python bench/bench.py` "
            f"(Python {platform.python_version()}, NumPy {np.__version__}, {FRAMES} frames of {N} float64).\n\n"
            "| configuration | us / frame |\n|---|---|\n"
        )
        fh.writelines(f"| {label} | {per:.1f} |\n" for label, per in rows)
        fh.write(
            f"| plain `np.fft.fft` on the same frames, for scale | {base:.1f} |\n\n"
        )
        fh.write(
            "Interpretation: the first map block pays for the scheduler hop; consecutive map "
            "blocks are fused into one scheduler member so each extra one costs a few microseconds "
            "(frame wrapping, annotation merge, a provenance node). A sync generator stream block "
            "runs inline on the event loop when `greenlet` is installed (otherwise in a worker "
            "thread, ~10x slower); `check='all'` adds the contract check per frame; "
            "`provenance='roots'` or `'off'` trims per-frame bookkeeping on long runs. The runtime "
            "is meant for frames of hundreds to millions of samples, not sample-by-sample streaming.\n"
        )


if __name__ == "__main__":
    main()
