"""A starter library of typed DSP blocks – sources, windows, FFTs, filters.

Most of these are thin, typed wrappers around NumPy / SciPy; they show how to
reuse existing library code with ``block(fn, inputs=..., outputs=...)`` and how
to write stateful stream blocks as plain generators.
"""

from __future__ import annotations

import asyncio
import time as _time
from typing import Any, Iterable, Iterator

import numpy as np
import scipy.signal as ss

complex_ = complex

from .block import Framed, block, source, tap
from .frame import Frame
from .graph import Block
from .types import Sig, Stream

__all__ = [
    "sine",
    "noise",
    "chirp",
    "impulse",
    "from_array",
    "ticker",
    "chunk",
    "overlap",
    "buffer",
    "decimate",
    "window",
    "fft",
    "rfft",
    "ifft",
    "magnitude",
    "power_db",
    "phase",
    "real",
    "imag",
    "to_complex",
    "scale",
    "offset",
    "mix",
    "lfilter",
    "fir",
    "hilbert",
    "resample",
    "energy",
    "rms",
    "peak",
    "running_mean",
    "clip",
    "detect_clipping",
    "printer",
    "collect",
    "timestamp",
    "to_units",
]

Real = Sig[float, "N"]
Complex = Sig[complex, "N"]


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #


@source(refine=lambda n, **_: Sig[float, (n,)], total=lambda frames, **_: frames)
def sine(
    freq: float = 1000.0,
    rate: float = 48_000.0,
    n: int = 1024,
    frames: int = 10,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> Sig[float, "N"]:
    """Continuous sine wave delivered in frames of ``n`` samples."""
    t0 = 0
    for i in range(frames):
        t = (np.arange(n) + t0) / rate
        yield Frame(
            amplitude * np.sin(2 * np.pi * freq * t + phase),
            meta={"t0": t0 / rate, "seq": i},
            stream={"rate": rate, "source": "sine"},
        )
        t0 += n


@source(
    refine=lambda n, complex, **_: Sig[complex_ if complex else float, (n,)],
    total=lambda frames, **_: frames,
)
def noise(
    n: int = 1024,
    frames: int = 10,
    sigma: float = 1.0,
    seed: int | None = None,
    rate: float = 48_000.0,
    complex: bool = False,
) -> Sig[float, "N"]:
    """White Gaussian noise frames."""
    rng = np.random.default_rng(seed)
    for i in range(frames):
        x = rng.normal(0, sigma, n)
        if complex:
            x = x + 1j * rng.normal(0, sigma, n)
        yield Frame(x, meta={"seq": i}, stream={"rate": rate, "source": "noise"})


@source(refine=lambda n, **_: Sig[float, (n,)], total=lambda frames, **_: frames)
def chirp(
    f0: float = 100.0,
    f1: float = 5000.0,
    rate: float = 48_000.0,
    n: int = 1024,
    frames: int = 10,
) -> Sig[float, "N"]:
    """Linear chirp sweeping f0 -> f1 across all frames (computed lazily, frame by frame)."""
    t1 = n * frames / rate
    for i in range(frames):
        t = (np.arange(n) + i * n) / rate
        yield Frame(
            ss.chirp(t, f0, t1, f1),
            meta={"seq": i, "t0": i * n / rate},
            stream={"rate": rate, "source": "chirp"},
        )


@source(refine=lambda n, **_: Sig[float, (n,)], total=lambda frames, **_: frames)
def impulse(n: int = 1024, frames: int = 1, at: int = 0) -> Sig[float, "N"]:
    for i in range(frames):
        x = np.zeros(n)
        if i == 0:
            x[at] = 1.0
        yield x


@source
def from_array(
    x: Any, n: int | None = None, rate: float | None = None
) -> Sig["T", "N"]:
    """Emit an array as one frame, or sliced into frames of ``n`` samples."""
    x = np.asarray(x)
    stream = {"rate": rate} if rate else {}
    if n is None:
        yield Frame(x, stream=stream)
        return
    for i in range(0, len(x) - n + 1, n):
        yield Frame(x[i : i + n], meta={"seq": i // n, "offset": i}, stream=stream)


@source(total=lambda count, **_: count)
async def ticker(period: float = 0.1, count: int = 10) -> Sig[dict]:
    """Asynchronous source: a dict event every ``period`` seconds (e.g. telemetry)."""
    for i in range(count):
        await asyncio.sleep(period)
        yield {"tick": i, "time": _time.time()}


# --------------------------------------------------------------------------- #
# Stream (stateful) blocks – plain generators
# --------------------------------------------------------------------------- #


class _Reframer:
    """Shared bookkeeping for re-framing stream blocks: absolute sample position and the
    time origin of the stream, so every output frame gets an exact ``t0``."""

    def __init__(self) -> None:
        self.buf: np.ndarray = np.empty(0)
        self.pos = 0  # absolute index of buf[0] in the stream
        self.origin: float | None = None  # t0 of the first frame seen (its sample 0)
        self.rate: float | None = None

    def push(self, f: Frame) -> None:
        x = np.asarray(f.data)
        if self.origin is None:
            self.origin = f.meta.get("t0", 0.0) or 0.0
            self.rate = f.stream.get("rate")
        self.buf = x if self.buf.size == 0 else np.concatenate([self.buf, x])

    def take(self, f: Frame, n: int, hop: int) -> Frame:
        """The next window of ``n`` samples as a frame stamped with its exact start time,
        then advance by ``hop``."""
        t0 = (self.origin or 0.0) + (self.pos / self.rate if self.rate else 0.0)
        out = f.evolve(data=self.buf[:n].copy(), meta={"t0": t0, "offset": self.pos})
        self.buf = self.buf[hop:]
        self.pos += hop
        return out


@block(refine=lambda n, **_: Sig["T", (n,)])
def chunk(xs: Stream[Framed[Sig["T", None]]], n: int = 256) -> Sig["T", "N"]:
    """Re-frame a stream of 1-D arrays into fixed-length frames of ``n`` samples (time kept)."""
    r = _Reframer()
    for f in xs:
        r.push(f)
        while len(r.buf) >= n:
            yield r.take(f, n, n)


@block(refine=lambda n, **_: Sig["T", (n,)])
def overlap(
    xs: Stream[Framed[Sig["T", "N"]]], n: int = 256, hop: int = 128
) -> Sig["T", "M"]:
    """Sliding windows of length ``n`` advanced by ``hop`` samples; each window's ``t0``
    is exact (computed from the sample position), not inherited from the input chunk."""
    r = _Reframer()
    for f in xs:
        r.push(f)
        while len(r.buf) >= n:
            yield r.take(f, n, hop)


@block(refine=lambda count, **_: Sig["T", (count, "N")])
def buffer(xs: Stream[Framed[Sig["T", "N"]]], count: int = 8) -> Sig["T", ("K", "N")]:
    """Stack ``count`` consecutive frames into a matrix (K x N); ``t0`` of the first row."""
    buf: list[Frame] = []
    for f in xs:
        buf.append(f)
        if len(buf) == count:
            yield f.evolve(
                data=np.stack([b.data for b in buf]),
                meta={"t0": buf[0].meta.get("t0")} if "t0" in buf[0].meta else {},
            )
            buf = []


@block
def decimate(xs: Stream[Sig["T", "N"]], every: int = 2) -> Sig["T", "N"]:
    """Keep one frame in ``every`` (frames keep their own time)."""
    x: Any
    for i, x in enumerate(xs):
        if i % every == 0:
            yield x


@block
def running_mean(xs: Stream[Sig[float, "N"]], alpha: float = 0.1) -> Sig[float, "N"]:
    """Exponential moving average across frames (stateful stream block)."""
    acc = None
    for x in xs:
        acc = (
            np.asarray(x, dtype=float) if acc is None else (1 - alpha) * acc + alpha * x
        )
        yield acc


@block
def timestamp(xs: Stream[Frame]) -> Frame:
    """Annotate frames with a running sequence number, wall-clock arrival time and — when
    the stream has a rate and the frame no ``t0`` yet — a ``t0`` from the sample count."""
    pos = 0
    f: Any
    for i, f in enumerate(xs):
        extra: dict = {}
        rate = f.stream.get("rate")
        if "t0" not in f.meta and rate and np.ndim(f.data) >= 1:
            extra["t0"] = pos / rate
        pos += len(f.data) if np.ndim(f.data) >= 1 else 0
        yield f.annotate(seq=i, t_arrival=_time.time(), **extra)


# --------------------------------------------------------------------------- #
# Map blocks – pure functions
# --------------------------------------------------------------------------- #


@block
def window(x: Sig["T", "N"], kind: str = "hann") -> Sig["T", "N"]:
    """Multiply by a window (any name scipy.signal.get_window accepts)."""
    return x * ss.get_window(kind, len(x))


fft = block(np.fft.fft, name="fft", inputs=Sig[complex, "N"], outputs=Sig[complex, "N"])
ifft = block(
    np.fft.ifft, name="ifft", inputs=Sig[complex, "N"], outputs=Sig[complex, "N"]
)
rfft = block(
    np.fft.rfft, name="rfft", inputs=Sig[float, "N"], outputs=Sig[complex, "M"]
)
hilbert = block(
    ss.hilbert, name="hilbert", inputs=Sig[float, "N"], outputs=Sig[complex, "N"]
)
magnitude = block(
    np.abs, name="magnitude", inputs=Sig[complex, "N"], outputs=Sig[float, "N"]
)
phase = block(np.angle, name="phase", inputs=Sig[complex, "N"], outputs=Sig[float, "N"])
real = block(np.real, name="real", inputs=Sig[complex, "N"], outputs=Sig[float, "N"])
imag = block(np.imag, name="imag", inputs=Sig[complex, "N"], outputs=Sig[float, "N"])


@block
def to_complex(x: Sig[float, "N"]) -> Sig[complex, "N"]:
    return x.astype(np.complex128)


@block
def power_db(x: Sig[complex, "N"], floor: float = 1e-12) -> Sig[float, "N", None, "dB"]:
    return 10 * np.log10(np.abs(x) ** 2 + floor)


@block
def scale(x: Sig["T", "N"], gain: float = 1.0) -> Sig["T", "N"]:
    return x * gain


@block(refine=lambda to, **_: Sig["T", "N", None, to])
def to_units(
    f: Framed[Sig["T", "N"]], to: str = "V", assume: str | None = None
) -> Sig["T", "N", None, None]:
    """Convert a stream to other units. The source units come from the frame's stream
    annotation ``units`` (or ``assume``); the output type and annotation carry ``to``."""
    from .units import factor

    src = assume or f.stream.get("units")
    if src is None:
        raise ValueError(
            "to_units: the stream carries no 'units' annotation; pass assume=..."
        )
    return f.evolve(data=f.data * factor(src, to), stream={"units": to})


@block
def offset(x: Sig[float, "N"], value: float = 0.0) -> Sig[float, "N"]:
    return x + value


@block
def mix(
    x: Sig[complex, "N"], freq: float = 0.0, rate: float = 48_000.0
) -> Sig[complex, "N"]:
    """Frequency-shift a complex frame by ``freq`` (stateless: phase restarts per frame)."""
    t = np.arange(len(x)) / rate
    return x * np.exp(2j * np.pi * freq * t)


@block
def lfilter(x: Sig[float, "N"], b: Any = (1.0,), a: Any = (1.0,)) -> Sig[float, "N"]:
    """Per-frame IIR/FIR filter (scipy.signal.lfilter; no state across frames)."""
    return ss.lfilter(np.asarray(b), np.asarray(a), x)


@block
def fir(xs: Stream[Sig[float, "N"]], taps: Any = (1.0,)) -> Sig[float, "N"]:
    """Streaming FIR filter that carries filter state across frames."""
    b = np.asarray(taps, dtype=float)
    zi = np.zeros(len(b) - 1)
    for x in xs:
        y, zi = ss.lfilter(b, [1.0], x, zi=zi)
        yield y


@block
def resample(f: Framed[Sig[float, "N"]], up: int = 1, down: int = 1) -> Sig[float, "M"]:
    """Rational resampling; the stream's ``rate`` annotation is scaled (time kept)."""
    rate = f.stream.get("rate")
    return f.evolve(
        data=ss.resample_poly(f.data, up, down),
        stream={"rate": rate * up / down} if rate else {},
    )


@block
def energy(x: Sig["T", "N"]) -> Sig[float, ()]:
    return float(np.sum(np.abs(x) ** 2))


@block
def rms(x: Sig["T", "N"]) -> Sig[float, ()]:
    return float(np.sqrt(np.mean(np.abs(x) ** 2)))


@block
def peak(x: Sig["T", "N"]) -> Sig[dict]:
    i = int(np.argmax(np.abs(x)))
    return {"index": i, "value": x[i]}


@block
def clip(x: Sig[float, "N"], limit: float = 1.0) -> Sig[float, "N"]:
    return np.clip(x, -limit, limit)


@block
def detect_clipping(f: Framed[Sig[float, "N"]], limit: float = 1.0) -> Sig[float, "N"]:
    """Frame-aware block: adds a sample-level mask and a warning issue when clipping occurs."""
    mask = np.abs(f.data) >= limit
    out = f.annotate_samples(clipped=mask)
    if mask.any():
        out = out.with_issue(
            f"{int(mask.sum())} samples clipped", "warning", "detect_clipping"
        )
    return out


# --------------------------------------------------------------------------- #
# Sinks / taps
# --------------------------------------------------------------------------- #


@tap
def printer(x: Any, prefix: str = "") -> None:
    print(prefix, x)


class collect(Block):
    """A tap node that appends frames to a list: ``sink = collect(); (g >> sink).run(); sink.data``."""

    def __init__(self) -> None:
        self.frames: list[Frame] = []
        super().__init__(tap(self.frames.append, name="collect", inputs=Frame))

    @property
    def data(self) -> list[Any]:
        return [f.data for f in self.frames]
