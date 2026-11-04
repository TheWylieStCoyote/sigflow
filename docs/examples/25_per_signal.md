# A varying number of signals, each processed by its own stateful graph

`examples/25_per_signal.py`

Run ``python examples/25_per_signal.py``. A spectrum estimate holds a changing set of
tones: one is always there, one comes and goes, one appears late. A detector block
yields one frame per detected tone (a stream block changes the cardinality of the
stream, and provenance remembers which spectrum frame each detection came from) and
gives each tone a stable track id by nearest frequency. ``per_key("track", ...)``
then runs a per-signal graph with memory, a running power average and a frequency
drift estimate, in one instance per track: a track that disappears has its
instance closed after ``ttl`` frames, and a track that comes back starts afresh.

## Source

```python
"""A varying number of signals, each processed by its own stateful graph.

Run ``python examples/25_per_signal.py``. A spectrum estimate holds a changing set of
tones: one is always there, one comes and goes, one appears late. A detector block
yields one frame per detected tone (a stream block changes the cardinality of the
stream, and provenance remembers which spectrum frame each detection came from) and
gives each tone a stable track id by nearest frequency. ``per_key("track", ...)``
then runs a per-signal graph with memory, a running power average and a frequency
drift estimate, in one instance per track: a track that disappears has its
instance closed after ``ttl`` frames, and a track that comes back starts afresh.
"""

from collections.abc import Iterator

import numpy as np

from sigflow import Frame, Framed, Lineage, Sig, Stream, block, per_key, run, source
from sigflow.dims import N

RATE = 8000.0
NFFT = 256
BINS = NFFT // 2 + 1
HZ_PER_BIN = RATE / NFFT

SCHEDULE = {  # tone -> the spectrum frames it is present in
    "steady at 1 kHz": range(40),
    "burst at 2.5 kHz": list(range(5, 15)) + list(range(28, 40)),
    "late at 3.2 kHz": range(22, 40),
}
FREQS = {
    "steady at 1 kHz": 1000.0,
    "burst at 2.5 kHz": 2500.0,
    "late at 3.2 kHz": 3200.0,
}


@source
def spectra(frames: int = 40, seed: int = 0) -> Iterator[Sig[float, N]]:
    """Power spectra (dB) with a noise floor and the scheduled tones, drifting slightly."""
    rng = np.random.default_rng(seed)
    for i in range(frames):
        power = 10 ** (rng.normal(-40.0, 1.0, BINS) / 10)
        for name, when in SCHEDULE.items():
            if i in when:
                f0 = FREQS[name] + 3.0 * np.sin(i / 5.0)  # a little drift
                k = f0 / HZ_PER_BIN
                power += 10 ** (-10.0 / 10) * np.exp(
                    -0.5 * ((np.arange(BINS) - k) / 1.2) ** 2
                )
        yield Frame(
            10 * np.log10(power),
            meta={"seq": i, "t0": i * NFFT / RATE},
            stream={"rate": RATE},
        )


@block
def detect(
    frames: Stream[Framed[Sig[float, N]]],
    threshold_db: float = -25.0,
    gate_hz: float = 150.0,
) -> Iterator[Sig[float, N]]:
    """One frame per detected tone: the spectrum around the peak, annotated with its
    frequency and a track id (nearest track within ``gate_hz``, else a new one)."""
    tracks: dict[int, float] = {}
    next_id = 0
    for f in frames:
        db = f.data
        peaks = [
            k
            for k in range(2, BINS - 2)
            if db[k] > threshold_db and db[k] >= db[k - 1] and db[k] > db[k + 1]
        ]
        for k in peaks:
            # parabolic interpolation of the peak: finer than one bin, so drift shows
            a, b, c = db[k - 1], db[k], db[k + 1]
            delta = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) else 0.0
            f0 = (k + delta) * HZ_PER_BIN
            near = [
                (abs(f0 - fc), tid)
                for tid, fc in tracks.items()
                if abs(f0 - fc) < gate_hz
            ]
            if near:
                tid = min(near)[1]
            else:
                tid, next_id = next_id, next_id + 1
            tracks[tid] = f0
            lo, hi = max(k - 4, 0), min(k + 5, BINS)
            yield f.evolve(data=db[lo:hi]).annotate(f0_hz=f0, track=tid, bins=(lo, hi))


@block
def track_signal(
    frames: Stream[Framed[Sig[float, N]]],
) -> Iterator[Sig[float, tuple[()]]]:
    """Per-signal state: a running mean of the peak power and the drift since first seen."""
    mean, first_f0 = 0.0, None
    for n, f in enumerate(frames, start=1):
        peak = float(f.data.max())
        mean += (peak - mean) / n
        first_f0 = f["f0_hz"] if first_f0 is None else first_f0
        yield f.evolve(data=np.float64(mean)).annotate(
            seen=n, drift_hz=f["f0_hz"] - first_f0
        )


def main() -> None:
    graph = spectra() >> detect >> per_key("track", track_signal, ttl=8)
    print(graph.describe(), "\n")
    out = run(graph, on_error="raise")

    print("Detections per spectrum frame (the count varies):")
    per_frame: dict[int, list[int]] = {}
    for f in out:
        per_frame.setdefault(f["seq"], []).append(f["track"])
    for seq in (0, 10, 20, 30, 39):
        print(f"  frame {seq:2d}: tracks {per_frame.get(seq, [])}")

    print("\nEach track kept its own running state (last frame of each):")
    last = {}
    for f in out:
        last[f["track"]] = f
    for tid, f in sorted(last.items()):
        print(
            f"  track {tid}: {f['f0_hz']:7.1f} Hz  mean peak {float(f.data):6.1f} dB"
            f"  seen {f['seen']:2d} frames  drift {f['drift_hz']:+5.1f} Hz  via {f.prov.op}"
        )

    print("\nThe burst left and came back: with ttl=8 its second visit is a new track,")
    print(
        "so its 'seen' count restarted:",
        [(tid, f["seen"]) for tid, f in sorted(last.items())],
    )

    lineage = Lineage(out)
    print(
        "\nLineage: every per-signal frame leads back to the spectrum frame it came from:"
    )
    print(" ", lineage.sources_of(out[-1:]))


if __name__ == "__main__":
    main()
```
