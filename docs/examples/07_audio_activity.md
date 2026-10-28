# Streaming audio analysis: re-framing, features, a hysteresis state machine,

`examples/07_audio_activity.py`

segment annotations and a waterfall — the shape of a VAD / event detector.

What this exercises
  * re-framing a stream with overlap (dsp.overlap), carrying rate/units
  * a fan-out that computes several features from the same frame and a block
    that consumes them together (tuple input)
  * a stateful stream block implementing a two-threshold state machine, which
    annotates frames with segment ids and raises warnings on suspicious input
  * sample-level annotations aligned with the audio, kept through a gain block
    (same leading dimension) and dropped by the FFT (different dimension)
  * graph.draw() to picture the topology; a waterfall plot of the result; the
    provenance of a segment lists exactly the frames that made it up

## Source

```python
"""Streaming audio analysis: re-framing, features, a hysteresis state machine,
segment annotations and a waterfall — the shape of a VAD / event detector.

What this exercises
  * re-framing a stream with overlap (dsp.overlap), carrying rate/units
  * a fan-out that computes several features from the same frame and a block
    that consumes them together (tuple input)
  * a stateful stream block implementing a two-threshold state machine, which
    annotates frames with segment ids and raises warnings on suspicious input
  * sample-level annotations aligned with the audio, kept through a gain block
    (same leading dimension) and dropped by the FFT (different dimension)
  * graph.draw() to picture the topology; a waterfall plot of the result; the
    provenance of a segment lists exactly the frames that made it up
"""

from collections.abc import Iterator

import numpy as np
import scipy.signal as ss

from sigflow import Frame, Framed, Sig, Stream, block, dsp, plot, run, source
from sigflow.dims import N

RATE = 16_000.0
FRAME, HOP = 512, 256


@source(total=lambda seconds, chunk, **_: int(seconds * RATE / chunk))
def microphone(
    seconds: float = 4.0, chunk: int = 1024, seed: int = 7
) -> Iterator[Sig[float, N]]:
    """Synthetic 'speech': tone bursts with harmonics over pink-ish noise, in irregular chunks."""
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    audio = 0.004 * rng.normal(size=n) + 0.004 * np.asarray(
        ss.lfilter([1.0], [1.0, -0.9], rng.normal(size=n))
    )
    for start, dur, f0 in [
        (0.4, 0.5, 180),
        (1.3, 0.3, 240),
        (1.75, 0.6, 200),
        (3.0, 0.4, 150),
    ]:
        i0, i1 = int(start * RATE), min(int((start + dur) * RATE), n)
        if i1 - i0 < 2:
            continue
        env = np.hanning(i1 - i0) ** 0.5
        voice = sum(np.sin(2 * np.pi * f0 * h * t[i0:i1]) / h for h in (1, 2, 3, 4))
        audio[i0:i1] += 0.3 * env * voice
    audio[int(0.6 * RATE) : int(0.6 * RATE) + 40] = 1.5  # a click that clips the ADC
    for i in range(0, n, chunk):
        yield Frame(
            audio[i : i + chunk],
            meta={"t0": i / RATE},
            stream={"rate": RATE, "units": "V"},
        )


@block
def short_time_energy(x: Sig[float, N]) -> Sig[float, tuple[()]]:
    return float(10 * np.log10(np.mean(x**2) + 1e-12))


@block
def zero_crossings(x: Sig[float, N]) -> Sig[float, tuple[()]]:
    return float(np.mean(np.abs(np.diff(np.signbit(x)))))


@block
def spectral_centroid(x: Sig[float, N], rate: float = RATE) -> Sig[float, tuple[()]]:
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1 / rate)
    return float(np.sum(f * X) / (np.sum(X) + 1e-12))


@block
def features(
    x: Framed[Sig[float, N]],
    energy: Sig[float, tuple[()]],
    zcr: Sig[float, tuple[()]],
    centroid: Sig[float, tuple[()]],
) -> Sig[float, N]:
    """Collect the three features into frame metadata; the audio passes through.
    `seq` was added by dsp.timestamp after re-framing, so it indexes 256-sample hops."""
    return x.annotate(
        energy_db=energy, zcr=zcr, centroid_hz=centroid, t0=x.meta["seq"] * HOP / RATE
    )


@block
def activity(
    frames: Stream[Framed[Sig[float, N]]],
    on_db: float = -30.0,
    off_db: float = -36.0,
    hang: int = 3,
) -> Iterator[Sig[float, N]]:
    """Two-threshold detector with hang-over. Annotates each frame with `active`
    and a running `segment` id, and marks clipped samples at sample level."""
    state, hold, segment = False, 0, 0
    for f in frames:
        e = f.meta["energy_db"]
        if not state and e > on_db:
            state, segment = True, segment + 1
        elif state and e < off_db:
            hold += 1
            if hold >= hang:
                state, hold = False, 0
        else:
            hold = 0
        clipped = np.abs(f.data) >= 0.99
        out = f.annotate(
            active=state, segment=segment if state else None
        ).annotate_samples(clipped=clipped)
        if clipped.any():
            out = out.with_issue(
                f"{int(clipped.sum())} clipped samples at t={f.meta['t0']:.3f}s",
                "warning",
                "activity",
            )
        yield out


@block
def segments(
    frames: Stream[Framed[Sig[float, N]]], rate: float = RATE, hop: int = HOP
) -> Iterator[Sig[dict]]:
    """Collapse per-frame activity into (start, end) segments, emitted when a segment closes."""
    current = None
    for f in frames:
        if f.meta["active"] and current is None:
            current = {
                "id": f.meta["segment"],
                "start": f.meta["t0"],
                "frames": 0,
                "centroids": [],
            }
        if current is not None:
            if f.meta["active"]:
                current["frames"] += 1
                current["centroids"].append(f.meta["centroid_hz"])
            else:
                yield {
                    "id": current["id"],
                    "start": round(current["start"], 3),
                    "end": round(f.meta["t0"], 3),
                    "centroid_hz": round(float(np.mean(current["centroids"])), 1),
                }
                current = None


if __name__ == "__main__":
    framing = (
        dsp.overlap(n=FRAME, hop=HOP) >> dsp.timestamp
    )  # overlap keeps rate/units; timestamp adds seq
    analysis = (
        framing
        >> (
            dsp.scale(gain=1.0) | short_time_energy | zero_crossings | spectral_centroid
        )
        >> features
        >> activity
    )

    fall = plot.waterfall(title="activity", rows=120)
    # The waterfall is a tap (one output per frame) and `segments` emits one dict per closed
    # segment, so they are chained rather than fanned out: fan-out pairs branches in lock step.
    g = microphone() >> analysis >> fall >> segments
    print(g.describe(), "\n")
    g.draw().save("audio_graph.svg")

    found = run(g, progress=True)
    frames = fall.frames  # every annotated audio frame the tap saw
    print(f"\n{len(frames)} frames; active in {sum(f.meta['active'] for f in frames)}")
    print("segments found:")
    for seg in found:
        print("  ", seg.data, " <- from", len(seg.prov.parents), "frames")
    warnings = [i for f in frames for i in f.warnings]
    print("warnings:", [str(w) for w in warnings][:2])
    clipped = [f for f in frames if f.samples["clipped"].any()]
    print(
        f"frames with clipped samples (sample-level mask kept through scale): {len(clipped)}"
    )
    print("saved audio_graph.svg,", fall.save("audio_waterfall.gif", interval=40))
```
