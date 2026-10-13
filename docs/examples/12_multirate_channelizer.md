# Multirate: a polyphase channelizer with N output ports at a lower rate, and a

`examples/12_multirate_channelizer.py`

rational resampler whose output rate is part of its type.

What this exercises
  * the `rate` field of Sig: a block's output rate derived from its configuration
    (`refine=...`), so `resample(up=3, down=4)` on 48 kHz *declares* 36 kHz, and a
    consumer that insists on 48 kHz is rejected before anything runs
  * a block with N output ports of a different rate than its input (one tuple
    frame = N channel frames); ports routed with `port(k)` to separate consumers
  * stateful polyphase filtering as a stream block
  * fan-out cardinality is 1:1 here because the channelizer emits one tuple per
    input frame — the right shape for multi-port, multi-rate outputs

## Source

```python
"""Multirate: a polyphase channelizer with N output ports at a lower rate, and a
rational resampler whose output rate is part of its type.

What this exercises
  * the `rate` field of Sig: a block's output rate derived from its configuration
    (`refine=...`), so `resample(up=3, down=4)` on 48 kHz *declares* 36 kHz, and a
    consumer that insists on 48 kHz is rejected before anything runs
  * a block with N output ports of a different rate than its input (one tuple
    frame = N channel frames); ports routed with `port(k)` to separate consumers
  * stateful polyphase filtering as a stream block
  * fan-out cardinality is 1:1 here because the channelizer emits one tuple per
    input frame — the right shape for multi-port, multi-rate outputs
"""
import numpy as np
import scipy.signal as ss

from sigflow import Frame, Sig, Stream, TypeMismatch, block, port, run, source
from sigflow import dsp, plot

FS = 48_000.0
M = 4                      # channels


def channel_type(M: int, rate: float, n: int):
    return tuple(Sig[complex, (n // M,), rate / M] for _ in range(M))


@block(refine=lambda M, rate, n, **_: channel_type(M, rate, n))
def channelizer(xs: Stream[Sig[complex, "N", None]], M: int = M, rate: float = FS, n: int = 1024,
                taps_per_phase: int = 16) -> tuple:
    """Polyphase (critically sampled) channelizer: N complex samples at `rate` in,
    M frames of N/M samples at rate/M out, one per channel, as one tuple frame."""
    h = ss.firwin(M * taps_per_phase, 1.0 / M)                    # prototype low-pass, cutoff fs/(2M)
    poly = h.reshape(taps_per_phase, M)                            # poly[j, rho] = h[j*M + rho]
    hist = np.zeros((taps_per_phase, M), dtype=complex)
    for x in xs:
        blocks = x.reshape(-1, M)[:, ::-1]                         # commutator: each row feeds the M phases
        out = np.empty((len(blocks), M), dtype=complex)
        for i, row in enumerate(blocks):
            hist = np.roll(hist, 1, axis=0)
            hist[0] = row
            out[i] = np.sum(hist * poly, axis=0)
        chans = np.fft.ifft(out, axis=1) * M                       # DFT across phases -> channels
        yield tuple(chans[:, k] for k in range(M))


@block(refine=lambda up, down, rate, **_: Sig[float, None, rate * up / down])
def resample(x: Sig[float, "N", None], up: int = 1, down: int = 1, rate: float = FS) -> Sig[float, None, None]:
    """Rational resampler whose declared output rate follows its configuration."""
    return ss.resample_poly(x, up, down)


@block
def needs_48k(x: Sig[float, None, 48_000]) -> Sig[float, ()]:
    """A consumer with a hard rate requirement in its type."""
    return float(np.sqrt(np.mean(x ** 2)))


@source(total=lambda frames, **_: frames)
def multitone(frames: int = 40, n: int = 1024, rate: float = FS) -> Sig[complex, "N"]:
    """Three carriers, each sitting in a different channelizer bin."""
    t0 = 0
    for k in range(frames):
        t = (np.arange(n) + t0) / rate
        t0 += n
        x = sum(a * np.exp(2j * np.pi * f * t) for a, f in ((1.0, 3_000), (0.5, 13_000), (0.25, -11_000)))
        x += 0.01 * (np.random.default_rng(k).normal(size=n) + 1j * np.random.default_rng(k + 1).normal(size=n))
        yield Frame(x, meta={"seq": k}, stream={"rate": rate})


@block
def tone_freq(c: Sig[complex, "K", None]) -> Sig[dict]:
    """Where is the energy within a channel (Hz relative to the channel centre)?"""
    X = np.fft.fftshift(np.abs(np.fft.fft(c)))
    f = np.fft.fftshift(np.fft.fftfreq(len(c), 1 / (FS / M)))
    return {"offset_hz": float(f[np.argmax(X)]), "power_db": float(20 * np.log10(X.max() / len(c) + 1e-12))}


if __name__ == "__main__":
    # --- rate contracts -----------------------------------------------------
    try:
        dsp.sine(rate=FS, n=1024) >> resample(up=3, down=4) >> needs_48k
    except TypeMismatch as e:
        print("rejected:", e)
    ok = dsp.sine(rate=FS, n=1024, frames=3) >> resample(up=3, down=4) >> dsp.rms
    print(ok.describe(), "\n")

    # --- channelizer: N ports at FS/M, each routed to its own consumer ----------
    bank = multitone() >> channelizer(M=M, rate=FS, n=1024)
    print(bank.describe())
    print("port types:", [str(t) for t in bank.out_type.items], "\n")

    per_channel = bank >> tuple(port(k) >> tone_freq for k in range(M))     # a fan-out of port selectors
    out = run(per_channel, progress=True)
    last = out[-1].data
    for k in range(M):
        centre = (k if k < M / 2 else k - M) * FS / M
        print(f"channel {k}: centre {centre:+7.0f} Hz  energy at {last[k]['offset_hz']:+7.0f} Hz "
              f"-> absolute {centre + last[k]['offset_hz']:+7.0f} Hz  ({last[k]['power_db']:.1f} dB)")

    # --- a single channel on its own, with a spectrum plot --------------------
    ch1 = plot.spectrum(title="channel 1 (centre +12 kHz, rate 12 kHz)", ylim=(-80, 10))
    run(bank[1] >> ch1)
    print("\nchannel 1 frames carry the decimated rate in their type:", bank[1].out_type)
    print("saved", ch1.save("channel1.gif", interval=100))
```
