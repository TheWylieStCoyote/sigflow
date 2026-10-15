# Keyed routing with `switch()` and stream-rate contracts with `ratio=`

`examples/21_routing_and_rates.py`

A receiver sees bursts of different modulations on one channel. A classifier tags
each burst; `switch()` then sends every frame through the demodulator for *its*
modulation only. Unlike `when()`, the branches of a switch are real streams: they
may window, buffer or drop, keep state across the frames they see, and run
concurrently. The second half shows the stream-rate contract: blocks declare how
many frames they emit per frame they consume, pipelines multiply those ratios, and
a fan-out of branches that would run at different rates is refused when the graph
is *built* instead of silently stopping at the shortest branch when it runs.

What this exercises
  * switch(selector, {key: branch, ...}) with a string selector (an annotation key)
    and with a callable; default=node / "pass" / None (drop)
  * branches that are whole pipelines, including N:1 re-framers
  * ordered=True for strict input-order output (1:1 branches only)
  * provenance that names the branch each frame took, queried with Lineage
  * @block(ratio=...) declarations, g.ratio, describe() rate tags, RatioMismatch
  * run(strict=True) checking declared ratios and expected frame counts

## Source

```python
"""Keyed routing with `switch()` and stream-rate contracts with `ratio=`.

A receiver sees bursts of different modulations on one channel. A classifier tags
each burst; `switch()` then sends every frame through the demodulator for *its*
modulation only. Unlike `when()`, the branches of a switch are real streams: they
may window, buffer or drop, keep state across the frames they see, and run
concurrently. The second half shows the stream-rate contract: blocks declare how
many frames they emit per frame they consume, pipelines multiply those ratios, and
a fan-out of branches that would run at different rates is refused when the graph
is *built* instead of silently stopping at the shortest branch when it runs.

What this exercises
  * switch(selector, {key: branch, ...}) with a string selector (an annotation key)
    and with a callable; default=node / "pass" / None (drop)
  * branches that are whole pipelines, including N:1 re-framers
  * ordered=True for strict input-order output (1:1 branches only)
  * provenance that names the branch each frame took, queried with Lineage
  * @block(ratio=...) declarations, g.ratio, describe() rate tags, RatioMismatch
  * run(strict=True) checking declared ratios and expected frame counts
"""
# mypy: disable-error-code="valid-type, misc, syntax, return-value, union-attr, index"

import numpy as np

from sigflow import (
    Frame,
    Framed,
    Lineage,
    Monitor,
    Ratio,
    RatioMismatch,
    Sig,
    Stream,
    block,
    dsp,
    run,
    source,
    switch,
)
from sigflow.dims import K, N

RATE = 1e6
N_SAMPLES = 512
rng = np.random.default_rng(3)


# ------------------------------------------------------------------- the channel


@source(total=lambda bursts, **_: bursts)
def channel(bursts: int = 24, snr_db: float = 15.0) -> Sig[complex, N]:
    """Bursts of BPSK, QPSK or an unmodulated carrier, tagged with the truth so the
    classifier's mistakes can be counted."""
    for i in range(bursts):
        kind = ("bpsk", "qpsk", "carrier")[i % 3]
        if kind == "bpsk":
            sym = rng.choice([-1, 1], N_SAMPLES).astype(complex)
        elif kind == "qpsk":
            sym = (
                rng.choice([-1, 1], N_SAMPLES) + 1j * rng.choice([-1, 1], N_SAMPLES)
            ) / np.sqrt(2)
        else:
            sym = np.exp(1j * 2 * np.pi * 0.1 * np.arange(N_SAMPLES))
        noise = (
            (rng.standard_normal(N_SAMPLES) + 1j * rng.standard_normal(N_SAMPLES))
            / np.sqrt(2)
            * 10 ** (-snr_db / 20)
        )
        yield Frame(
            sym + noise,
            meta={"seq": i, "truth": kind, "t0": i * N_SAMPLES / RATE},
            stream={"rate": RATE},
        )


@block
def classify(f: Framed[Sig[complex, N]]) -> Sig[complex, N]:
    """Cheap modulation classifier (spectral line -> carrier, else the |m2| / |m4| moments);
    tags ``modulation``. Deliberately simple: a few misclassifications show the default branch."""
    x = f.data / np.sqrt(np.mean(np.abs(f.data) ** 2))
    spec = np.abs(np.fft.fft(x)) ** 2
    if spec.max() / spec.sum() > 0.5:
        mod = "carrier"
    else:
        m2, m4 = abs(np.mean(x**2)), abs(np.mean(x**4))
        mod = "bpsk" if m2 > 0.5 else "qpsk" if m4 > 0.5 else "unknown"
    return f.annotate(modulation=mod)


# ------------------------------------------------------------- the demodulators


@block
def demod_bpsk(x: Sig[complex, N]) -> Sig[int, N]:
    return (x.real > 0).astype(int)  # type: ignore[attr-defined]


@block
def demod_qpsk(x: Sig[complex, N]) -> Sig[int, N]:
    return ((x.real > 0).astype(int) << 1) | (x.imag > 0).astype(int)  # type: ignore[attr-defined]


@block(ratio="1:1")
def track_carrier(xs: Stream[Sig[complex, N]]) -> Sig[float, ()]:
    """Stateful: a running frequency estimate across the carrier bursts *it* sees."""
    est = None
    for x in xs:
        f = float(np.angle(np.mean(x[1:] * np.conj(x[:-1]))) / (2 * np.pi))
        est = f if est is None else 0.8 * est + 0.2 * f
        yield est


@block
def average_bursts(x: Sig[complex, (K, N)]) -> Sig[complex, N]:
    return x.mean(axis=0)


@block
def log_unknown(f: Framed[Sig[complex, N]]) -> Sig[str]:
    return f"unclassified burst #{f['seq']}"


# ---------------------------------------------------------------------- routing

demod = switch(
    "modulation",
    {
        "bpsk": demod_bpsk,
        "qpsk": demod_qpsk,
        "carrier": track_carrier,  # a stream block: only ever sees carrier bursts
    },
    default=log_unknown,
)

g = channel() >> classify >> demod
print(g.describe())
print()

out = run(g)
L = Lineage(out)
by_branch: dict[str, list] = {}
for f in out:
    by_branch.setdefault(f.prov.op, []).append(f)
for op, frames in sorted(by_branch.items()):
    print(
        f"{op:36} {len(frames):3} frames   e.g. seq={frames[0]['seq']}  truth={frames[0]['truth']}"
    )
wrong = [f for f in out if f["truth"] != f.get("modulation")]
print(f"\nmisclassified: {len(wrong)} of {len(out)}")

# `switch` branches may re-frame: average the QPSK symbols over 4 bursts at a time
demod4 = switch(
    "modulation",
    {
        "bpsk": demod_bpsk,
        "qpsk": dsp.buffer(count=4)
        >> average_bursts
        >> demod_qpsk,  # 4:1 inside this branch only
    },
    default="pass",
)
out4 = run(channel() >> classify >> demod4)
print(
    f"\nwith a 4:1 branch: {len(out4)} outputs from 24 bursts "
    f"({sum('qpsk' in f.prov.op for f in out4)} averaged QPSK frames, each from 4 bursts)"
)

# ordered=True keeps input order (and insists on 1:1 branches)
ordered = run(
    channel()
    >> classify
    >> switch(
        "modulation",
        {"bpsk": demod_bpsk, "qpsk": demod_qpsk},
        default="pass",
        ordered=True,
    )
)
assert [f["seq"] for f in ordered] == list(range(24))
try:
    switch("modulation", {"qpsk": dsp.buffer(count=4)}, ordered=True)
except RatioMismatch as e:
    print(f"\nordered switch refused: {e}")

# --------------------------------------------------------------- rate contracts

print("\n--- stream-rate contracts ---")
src = dsp.sine(freq=1000, rate=48_000, n=1024, frames=20)
for g2 in (
    src >> dsp.fft,
    src >> dsp.chunk(256),
    src >> dsp.overlap(256, hop=128) >> dsp.buffer(count=8),
    src >> dsp.decimate(every=4),
    (src >> dsp.rms).take(5),
):
    print(f"{str(g2.ratio):>6}   {g2.name}")

try:
    src >> (dsp.chunk(256) | dsp.rms)  # 1:4 next to 1:1: cannot be zipped
except RatioMismatch as e:
    print(f"\nrefused at build time: {e}")

ok = src >> (
    dsp.chunk(256) >> dsp.rms | dsp.overlap(512, hop=256) >> dsp.rms
)  # both 1:4
print(
    f"\nbuilt fine: {ok.ratio} -> {len(run(ok))} tuple frames from 20 inputs "
    "(the sliding window cannot complete its last frame: ratios are long-run averages)"
)

m = Monitor()
run(src >> dsp.chunk(256), monitor=m, strict=True)
print(
    f"expected {m.expected} frames from 20 source frames at {str((src >> dsp.chunk(256)).ratio)}; strict run agreed"
)


@block(ratio="1:1")
def misdeclared(xs: Stream[Sig[float, N]]) -> Sig[float, N]:
    for x in xs:
        yield x
        yield -x


try:
    run(src >> misdeclared, strict=True)
except RatioMismatch as e:
    print(f"strict run caught: {e}")
print(
    f"\n{Ratio.of(1, 4)} * {Ratio.of(8, 1)} = {Ratio.of(1, 4) * Ratio.of(8, 1)};  N:1 * 1:1 = {Ratio.DOWN * Ratio.ONE}"
)
```
