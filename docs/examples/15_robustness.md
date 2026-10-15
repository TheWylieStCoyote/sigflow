# Robustness and recovery: restarting a source, degrading gracefully, routing by issue

`examples/15_robustness.py`

Example 10 attributed failures; this one keeps the pipeline *useful* while they
happen.

What this exercises
  * a watchdog source: wraps a flaky device, restarts it after it dies, and
    annotates frames with the restart count (`restarts` in stream annotations)
  * graceful degradation with `when()`: frames flagged low-SNR take a cheaper,
    more robust path; the rest take the precise one — both paths rejoin
  * routing by issue level: warnings go to a quarantine sink *and* carry on,
    errors are dropped, clean frames go to the main sink
  * `on_error="drop"` on a block vs. propagating by default
  * a block that catches its own exception and emits a warning instead of failing

## Source

```python
"""Robustness and recovery: restarting a source, degrading gracefully, routing by issue.

Example 10 attributed failures; this one keeps the pipeline *useful* while they
happen.

What this exercises
  * a watchdog source: wraps a flaky device, restarts it after it dies, and
    annotates frames with the restart count (`restarts` in stream annotations)
  * graceful degradation with `when()`: frames flagged low-SNR take a cheaper,
    more robust path; the rest take the precise one — both paths rejoin
  * routing by issue level: warnings go to a quarantine sink *and* carry on,
    errors are dropped, clean frames go to the main sink
  * `on_error="drop"` on a block vs. propagating by default
  * a block that catches its own exception and emits a warning instead of failing
"""

from typing import Iterator
import numpy as np

from sigflow import MISSING, Frame, Framed, Sig, Stream, block, dsp, run, source, when
from sigflow.dims import N

N_SAMPLES = 128


# --------------------------------------------------------- a flaky device


class Device:
    """Pretend hardware: works for a while, then dies and must be re-opened."""

    opened = 0

    def __init__(self, seed: int) -> None:
        Device.opened += 1
        self.rng = np.random.default_rng(seed + Device.opened)
        self.k = 0
        self.lifetime = int(self.rng.integers(8, 20))

    def read(self) -> np.ndarray:
        if self.k >= self.lifetime:
            raise IOError("device reset")
        self.k += 1
        snr = 20 if self.rng.random() > 0.3 else 0  # occasional low-SNR frames
        x = np.sin(2 * np.pi * 5 * np.arange(N_SAMPLES) / N_SAMPLES) + 10 ** (
            -snr / 20
        ) * self.rng.normal(size=N_SAMPLES)
        return x


@source(total=lambda frames, **_: frames)
def watchdog(
    frames: int = 60, max_restarts: int = 5, seed: int = 0
) -> Iterator[Sig[float, N]]:
    """Read from the device; when it dies, re-open it and continue. The restart count
    rides along as a stream annotation and the frame right after a restart carries a
    warning so downstream (and the lineage) know the stream was interrupted."""
    dev = Device(seed)
    restarts = 0
    k = 0
    while k < frames:
        try:
            x = dev.read()
        except IOError as e:
            if restarts >= max_restarts:
                raise  # give up: the stream ends with a failed frame
            restarts += 1
            dev = Device(seed)
            yield Frame(
                np.zeros(N_SAMPLES),
                meta={"seq": k, "gap": True},
                stream={"restarts": restarts},
            ).with_issue(f"{e}; restarted ({restarts})", "warning", "watchdog")
            k += 1
            continue
        yield Frame(x, meta={"seq": k, "gap": False}, stream={"restarts": restarts})
        k += 1


# ---------------------------------------------------- degrading gracefully


@block
def assess(f: Framed[Sig[float, N]]) -> Sig[float, N]:
    """Estimate SNR crudely and tag the frame; a gap frame is tagged as such."""
    x = f.data
    X = np.abs(np.fft.rfft(x))
    with np.errstate(divide="ignore"):
        snr = float(20 * np.log10(X[5] / (np.median(X) + 1e-9) + 1e-12))
    return f.annotate(snr_db=snr, low_snr=snr < 25 or f.meta.get("gap", False))


@block
def precise_frequency(x: Sig[float, N]) -> Sig[dict]:
    """Interpolated FFT peak: accurate, but fragile when the peak is buried."""
    X = np.abs(np.fft.rfft(x))
    k = int(np.argmax(X[1:-1])) + 1
    a, b, c = X[k - 1], X[k], X[k + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        delta = 0.5 * (a - c) / (a - 2 * b + c)
    if not np.isfinite(delta) or abs(delta) > 1:
        raise ValueError("interpolation failed")
    return {"bin": k + delta, "method": "precise"}


@block
def robust_frequency(x: Sig[float, N]) -> Sig[dict]:
    """Zero-crossing count: coarse, but never fails."""
    zc = np.count_nonzero(np.diff(np.signbit(x)))
    return {"bin": zc / 2, "method": "robust"}


@block
def guarded(f: Framed[Sig[float, N]]) -> Sig[dict]:
    """A block that handles its own exception: on failure it emits a warning frame with
    a fallback value rather than a failed frame."""
    try:
        return precise_frequency(f.data)
    except ValueError as e:
        return f.evolve(data=robust_frequency(f.data)).with_issue(
            f"fell back: {e}", "warning", "guarded"
        )


if __name__ == "__main__":
    # 1. Watchdog + graceful degradation: low-SNR frames take the robust path.
    estimate = when(
        lambda f: f["low_snr"],
        robust_frequency,
        otherwise=precise_frequency,
        name="pick_method",
    )
    g = watchdog(frames=60) >> assess >> estimate
    print(g.describe(), "\n")
    g.draw().save("robustness_graph.svg")
    out = run(g, progress=True)
    methods = [f.data["method"] for f in out if f.ok]
    print(
        f"\n{len(out)} frames: {methods.count('precise')} precise, {methods.count('robust')} robust, "
        f"{sum(f.failed for f in out)} failed; device restarted {out[-1]['restarts']} times"
    )
    gaps = [f for f in out if f.warnings]
    print("restart warnings:", [str(f.warnings[0]) for f in gaps][:2])

    # 2. Routing by issue level: clean -> main, warnings -> quarantine too, errors dropped.
    main, quarantine = dsp.collect(), dsp.collect()

    @block
    def drop_failed(f: Framed[Sig[dict]]) -> Sig[dict]:
        return None  # returning None drops the frame

    route = when(
        lambda f: f.failed,
        drop_failed,
        otherwise=when(lambda f: bool(f.warnings), quarantine >> main, otherwise=main),
    )
    # `guarded` turns estimator failures into warnings, the watchdog's restarts are warnings
    # already, and anything that still fails (none here) would be dropped by the first branch.
    g2 = watchdog(frames=40) >> assess >> guarded >> route
    g2.draw().save("routing_graph.svg")
    out2 = run(g2)
    print(
        f"\nrouting: {len(main.frames)} to main, {len(quarantine.frames)} of them also quarantined (warnings), "
        f"{40 - len(out2)} dropped"
    )

    # 3. A self-healing block vs. error policies on the same input.
    dropping = block(precise_frequency.fn, name="precise", on_error="drop")
    for label, graph in (
        ("guarded block", watchdog(frames=40) >> assess >> guarded),
        ("precise, on_error=drop", watchdog(frames=40) >> assess >> dropping),
        ("precise, propagate", watchdog(frames=40) >> assess >> precise_frequency),
    ):
        o = run(graph)
        print(
            f"{label:24s}: {len(o):2d} frames out, {sum(f.failed for f in o):2d} failed, "
            f"{sum(bool(f.warnings) for f in o):2d} with warnings"
        )
```
