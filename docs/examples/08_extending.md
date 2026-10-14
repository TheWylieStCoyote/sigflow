# Extending sigflow: your own typed blocks, library wrappers and policies

`examples/08_extending.py`

What this exercises
  * sample rate and units as part of the contract (Sig[dtype, shape, rate, units])
    — a 48 kHz block refuses a 44.1 kHz stream before anything runs
  * wrapping a stateful SciPy filter (sosfilt with carried state) in a few lines
  * a dataclass as a frame payload (Sig[MyType]) and a block that consumes it
  * the three error policies, side by side, on the same flaky block
  * runtime checking modes: check="first" (default) vs "all"
  * blocks are functions: unit-test them with plain asserts, no graph needed

## Output

Files the script writes when run:

![extending_graph.svg](assets/08_extending/extending_graph.svg)

*`extending_graph.svg`*

## Source

```python
"""Extending sigflow: your own typed blocks, library wrappers and policies.

What this exercises
  * sample rate and units as part of the contract (Sig[dtype, shape, rate, units])
    — a 48 kHz block refuses a 44.1 kHz stream before anything runs
  * wrapping a stateful SciPy filter (sosfilt with carried state) in a few lines
  * a dataclass as a frame payload (Sig[MyType]) and a block that consumes it
  * the three error policies, side by side, on the same flaky block
  * runtime checking modes: check="first" (default) vs "all"
  * blocks are functions: unit-test them with plain asserts, no graph needed
"""

from dataclasses import dataclass

import numpy as np
import scipy.signal as ss

from sigflow import Frame, Framed, Sig, Stream, TypeMismatch, block, lift, run, source
from sigflow import dsp

# ------------------------------------------------------------- rate and units

Audio48k = Sig[float, "N", 48_000, "V"]  # type aliases read well in signatures


@source(
    total=lambda frames, **_: frames,
    refine=lambda n, rate, **_: Sig[float, (n,), rate, "V"],
)
def adc(
    rate: float = 48_000, n: int = 480, frames: int = 20, seed: int = 0
) -> Sig[float, "N", None, "V"]:
    """An ADC whose declared output rate comes from its configuration (via refine)."""
    rng = np.random.default_rng(seed)
    for i in range(frames):
        yield Frame(
            rng.normal(0, 0.1, n)
            + np.sin(2 * np.pi * 1000 * (np.arange(n) + i * n) / rate),
            stream={"rate": rate, "units": "V"},
        )


@block
def de_emphasis(x: Audio48k, tau_us: float = 50.0) -> Audio48k:
    """Only meaningful at 48 kHz: the coefficient is baked for that rate."""
    a = np.exp(-1 / (48_000 * tau_us * 1e-6))
    return ss.lfilter([1 - a], [1, -a], x)


@block
def to_dbfs(
    x: Sig[float, "N", None, "V"], full_scale: float = 1.0
) -> Sig[float, (), None, "dBFS"]:
    return float(20 * np.log10(np.sqrt(np.mean(x**2)) / full_scale + 1e-12))


# ----------------------------------------------------- wrapping a SciPy filter


def sos_filter(sos: np.ndarray, name: str = "sos_filter"):
    """Factory: a streaming IIR block around scipy.signal.sosfilt with carried state."""

    def fn(xs: Stream[Sig[float, "N"]]) -> Sig[float, "N"]:
        zi = np.zeros((sos.shape[0], 2))
        for x in xs:
            y, zi = ss.sosfilt(sos, x, zi=zi)
            yield y

    return block(fn, name=name)


lowpass_2k = sos_filter(ss.butter(6, 2000, fs=48_000, output="sos"), name="lowpass_2k")

# ------------------------------------------------- a custom payload dataclass


@dataclass(frozen=True)
class Measurement:
    level_dbfs: float
    tone_hz: float
    frame: int


@block
def measure(f: Framed[Sig[float, "N"]], rate: float = 48_000) -> Sig[Measurement]:
    X = np.abs(np.fft.rfft(f.data * np.hanning(len(f.data))))
    tone = float(np.fft.rfftfreq(len(f.data), 1 / rate)[int(np.argmax(X[1:])) + 1])
    return Measurement(to_dbfs(f.data), tone, f.prov.seq if f.prov else -1)


@block
def report(m: Sig[Measurement]) -> Sig[str]:
    return f"frame {m.frame:2d}: {m.level_dbfs:6.1f} dBFS, tone {m.tone_hz:7.1f} Hz"


# ------------------------------------------------------------- error policies


@block
def flaky(x: Sig[float, "N"], every: int = 4) -> Sig[float, "N"]:
    flaky.calls += 1
    if flaky.calls % every == 0:
        raise IOError("DMA underrun")
    return x


flaky.calls = 0


if __name__ == "__main__":
    # 1. rate contract checked at build time
    try:
        adc(rate=44_100) >> de_emphasis
    except TypeMismatch as e:
        print("rejected:", e)
    g = adc(rate=48_000) >> de_emphasis >> lowpass_2k >> measure >> report
    print(g.describe(), "\n")
    g.draw().save("extending_graph.svg")
    for f in run(g)[:3]:
        print(f.data)

    # 2. the three error policies on identical input
    for policy in ("propagate", "drop", "raise"):
        flaky.calls = 0
        try:
            out = run(adc(frames=8) >> flaky >> to_dbfs, on_error=policy)
            print(
                f"{policy:9s}: {len(out)} frames, {sum(f.failed for f in out)} failed"
            )
        except IOError as e:
            print(f"{policy:9s}: run aborted with {e!r}")

    # 3. runtime checking: "first" only checks the first frame of each edge, "all" checks every frame
    @block
    def grows(x: Sig[float, "N"]) -> Sig[float, "N"]:
        grows.n += 1
        return np.zeros(
            len(x) + (grows.n > 2)
        )  # lies about its shape from the third frame

    grows.n = 0
    print("check=first:", len(run(adc(frames=4) >> grows)), "frames (unnoticed)")
    grows.n = 0
    try:
        run(adc(frames=4) >> grows, check="all")
    except TypeMismatch as e:
        print("check=all  :", e)

    # 4. blocks are plain functions: test them without a graph
    x = np.sin(2 * np.pi * 1000 * np.arange(480) / 48_000)
    assert abs(to_dbfs(x) - (-3.01)) < 0.05  # a full-scale sine is -3 dBFS rms
    assert measure(Frame(x)).tone_hz == 1000.0
    assert len(list(lowpass_2k([x, x]))) == 2  # stream blocks take any iterable
    # untyped library code can be dropped in with lift(); the contract is checked at run time
    assert run(adc(frames=1) >> lift(np.cumsum) >> to_dbfs)[0].ok
    print("unit checks passed")
```
