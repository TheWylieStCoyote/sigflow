"""Sources and sinks for files and sockets.

Sources (frames carry ``rate`` and ``source`` stream annotations where known):

    io.wav(path, n=1024)              WAV file in frames of n samples (scipy.io.wavfile)
    io.raw(path, dtype, n=1024)       raw binary stream (e.g. GNU Radio file sink: complex64)
    io.npy(path, n=1024)              a .npy array, memory-mapped and streamed in frames
    io.csv_rows(path)                 one dict per row of a CSV file
    io.udp(port, dtype, ...)          async UDP datagrams as arrays (loopback-friendly)
    io.zmq(address, dtype, ...)       async ZeroMQ SUB/PULL receiver (pyzmq) — the GNU Radio
                                      ZMQ PUB/PUSH sink blocks speak exactly this

Sinks (taps: the frame passes through unchanged):

    io.wav_sink(path, rate)           append frames to a WAV file (written on close)
    io.npy_sink(path)                 stack frames into a .npy array (written on close)
    io.csv_log(path, fields=...)      one CSV row per frame from meta / data dict keys
    io.udp_sink(host, port)           send each frame as one datagram
    io.zmq_sink(address, kind)        PUB/PUSH each frame as one message (GNU Radio ZMQ sources read it)

A sink object is a graph node; call ``sink.close()`` after the run (or use it as a
context manager) to flush files.
"""

from __future__ import annotations

import asyncio
import csv
import os
from typing import Any, AsyncIterator, Iterator, TypeVar

import numpy as np

from .block import source, tap
from .dims import N, T
from .frame import Frame
from .graph import Block
from .types import Sig, SignalType

__all__ = [
    "wav",
    "raw",
    "npy",
    "csv_rows",
    "udp",
    "zmq",
    "wav_sink",
    "npy_sink",
    "csv_log",
    "udp_sink",
    "zmq_sink",
]


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #


def _frames_of(
    x: np.ndarray, n: int, rate: float | None, name: str, path: str
) -> Iterator[Frame]:
    total = len(x)
    stream: dict[str, Any] = {"source": name, "path": os.fspath(path)}
    if rate:
        stream["rate"] = rate
    for i in range(0, total - n + 1, n):
        yield Frame(
            np.array(x[i : i + n]),
            meta={"seq": i // n, "offset": i, "t0": i / rate if rate else None},
            stream=stream,
        )


@source(
    total=lambda path, n, **_: _wav_frames(path, n),
    refine=lambda n, **_: SignalType(T, (n,)),
)
def wav(
    path: str, n: int = 1024, channel: int | None = None, float32: bool = True
) -> Iterator[Sig[T, N]]:
    """Stream a WAV file. Multi-channel files give (n, channels) frames unless ``channel`` picks one."""
    from scipy.io import wavfile

    rate, data = wavfile.read(path)
    if float32 and data.dtype.kind in "iu":
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    if channel is not None and data.ndim == 2:
        data = data[:, channel]
    yield from _frames_of(data, n, float(rate), "wav", path)


def _wav_frames(path: str, n: int) -> int | None:
    try:
        from scipy.io import wavfile

        rate, data = wavfile.read(path, mmap=True)
        return len(data) // n
    except Exception:
        return None


@source(
    total=lambda path, dtype, n, **_: _raw_frames(path, dtype, n),
    refine=lambda n, dtype, **_: SignalType(dtype, (n,)),
)
def raw(
    path: str, dtype: Any = np.complex64, n: int = 1024, rate: float | None = None
) -> Iterator[Sig[T, N]]:
    """Stream a raw binary file (what a GNU Radio file sink writes), memory-mapped."""
    data = np.memmap(path, dtype=np.dtype(dtype), mode="r")
    yield from _frames_of(data, n, rate, "raw", path)


def _raw_frames(path: str, dtype: Any, n: int) -> int | None:
    try:
        return os.path.getsize(path) // np.dtype(dtype).itemsize // n
    except OSError:
        return None


@source(refine=lambda n, **_: SignalType(T, (n,)))
def npy(path: str, n: int = 1024, rate: float | None = None) -> Iterator[Sig[T, N]]:
    """Stream a 1-D ``.npy`` array without loading it all (memory-mapped)."""
    data = np.load(path, mmap_mode="r")
    yield from _frames_of(data, n, rate, "npy", path)


@source
def csv_rows(path: str, numeric: bool = True) -> Iterator[Sig[dict]]:
    """One dict per row; numeric-looking fields are converted to float when ``numeric``."""
    with open(path, newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if numeric:
                row = {k: _maybe_float(v) for k, v in row.items()}
            yield Frame(
                row, meta={"seq": i}, stream={"source": "csv", "path": os.fspath(path)}
            )


def _maybe_float(v: Any) -> Any:
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


@source
async def udp(
    port: int,
    dtype: Any = np.float32,
    host: str = "127.0.0.1",
    count: int | None = None,
    timeout: float | None = None,
) -> AsyncIterator[Sig[T, None]]:
    """Asynchronous UDP receiver: each datagram becomes one frame (array of ``dtype``).
    Stops after ``count`` datagrams, or ``timeout`` seconds of silence."""
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    class Proto(asyncio.DatagramProtocol):
        def datagram_received(self, data: bytes, addr: Any) -> None:
            q.put_nowait((data, addr))

    transport, _ = await loop.create_datagram_endpoint(Proto, local_addr=(host, port))
    try:
        k = 0
        while count is None or k < count:
            try:
                data, addr = await asyncio.wait_for(q.get(), timeout)
            except asyncio.TimeoutError:
                return
            yield Frame(
                np.frombuffer(data, dtype=np.dtype(dtype)).copy(),
                meta={"seq": k, "from": f"{addr[0]}:{addr[1]}"},
                stream={"source": f"udp:{port}"},
            )
            k += 1
    finally:
        transport.close()


@source
async def zmq(
    address: str = "tcp://127.0.0.1:5555",
    dtype: Any = np.complex64,
    kind: str = "SUB",
    topic: bytes = b"",
    count: int | None = None,
    timeout: float | None = None,
    connect: bool = True,
) -> AsyncIterator[Sig[T, None]]:
    """Asynchronous ZeroMQ receiver (``pip install pyzmq``). ``kind`` is ``"SUB"`` (matches a
    GNU Radio *ZMQ PUB Sink*) or ``"PULL"`` (matches *ZMQ PUSH Sink*); each message becomes
    one frame of ``dtype``. ``connect=False`` binds instead of connecting."""
    import zmq as _zmq
    import zmq.asyncio as _zmqa

    ctx = _zmqa.Context.instance()
    sock = ctx.socket(_zmq.SUB if kind.upper() == "SUB" else _zmq.PULL)
    if kind.upper() == "SUB":
        sock.setsockopt(_zmq.SUBSCRIBE, topic)
    (sock.connect if connect else sock.bind)(address)
    try:
        k = 0
        while count is None or k < count:
            if timeout is not None:
                if not await sock.poll(int(timeout * 1000)):
                    return
            msg = await sock.recv()
            if topic and msg.startswith(topic):
                msg = msg[len(topic) :]
            yield Frame(
                np.frombuffer(msg, dtype=np.dtype(dtype)).copy(),
                meta={"seq": k},
                stream={"source": f"zmq:{address}"},
            )
            k += 1
    finally:
        sock.close(linger=0)


# --------------------------------------------------------------------------- #
# Sinks
# --------------------------------------------------------------------------- #


_SinkT = TypeVar("_SinkT", bound="_Sink")


class _Sink(Block):
    def __init__(self, fn: Any, name: str) -> None:
        super().__init__(tap(fn, name=name, inputs=Frame))

    def close(self) -> Any:  # pragma: no cover - overridden
        pass

    def __enter__(self: _SinkT) -> _SinkT:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class wav_sink(_Sink):
    """Collect frames and write them as one WAV file on ``close()``."""

    def __init__(
        self, path: str, rate: float | None = None, dtype: Any = np.int16
    ) -> None:
        self.path, self.rate, self.dtype = path, rate, np.dtype(dtype)
        self.chunks: list[np.ndarray] = []
        super().__init__(self._add, "wav_sink")

    def _add(self, f: Frame) -> None:
        self.chunks.append(np.asarray(f.data))
        if self.rate is None:
            self.rate = f.stream.get("rate")

    def close(self) -> str:
        from scipy.io import wavfile

        if not self.chunks:
            return self.path
        x = np.concatenate(self.chunks)
        if self.dtype.kind in "iu" and x.dtype.kind == "f":
            x = (np.clip(x, -1, 1) * np.iinfo(self.dtype).max).astype(self.dtype)
        wavfile.write(self.path, int(self.rate or 1), x)
        return self.path


class npy_sink(_Sink):
    """Stack frames (leading axis = frame) into a ``.npy`` file on ``close()``."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.chunks: list[np.ndarray] = []
        super().__init__(lambda f: self.chunks.append(np.asarray(f.data)), "npy_sink")

    def close(self) -> str:
        np.save(self.path, np.stack(self.chunks) if self.chunks else np.empty(0))
        return self.path


class csv_log(_Sink):
    """One CSV row per frame: ``fields`` are taken from the frame's dict data, then its
    annotations; scalars are logged under ``value``."""

    def __init__(self, path: str, fields: Any = None) -> None:
        self.path = path
        self.fields = list(fields) if fields else None
        self._fh = open(path, "w", newline="")
        self._writer: csv.DictWriter | None = None
        super().__init__(self._add, "csv_log")

    def _add(self, f: Frame) -> None:
        row: dict[str, Any] = {}
        if isinstance(f.data, dict):
            row.update(f.data)
        elif np.ndim(f.data) == 0:
            row["value"] = f.data
        row.update({k: v for k, v in f.meta.items() if np.ndim(v) == 0})
        if self.fields is not None:
            row = {k: row.get(k) for k in self.fields}
        if self._writer is None:
            self._writer = csv.DictWriter(self._fh, fieldnames=list(row))
            self._writer.writeheader()
        self._writer.writerow({k: row.get(k) for k in self._writer.fieldnames})

    def close(self) -> str:
        self._fh.close()
        return self.path


class udp_sink(_Sink):
    """Send each frame's array as one datagram."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9999) -> None:
        import socket

        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        super().__init__(
            lambda f: self.sock.sendto(
                np.ascontiguousarray(f.data).tobytes(), self.addr
            ),
            "udp_sink",
        )

    def close(self) -> None:
        self.sock.close()


class zmq_sink(_Sink):
    """Publish (``kind="PUB"``, read by a GNU Radio *ZMQ SUB Source*) or push (``"PUSH"``)
    each frame's array as one message; ``bind=True`` by default (the usual server side)."""

    def __init__(
        self,
        address: str = "tcp://127.0.0.1:5555",
        kind: str = "PUB",
        topic: bytes = b"",
        bind: bool = True,
    ) -> None:
        import zmq as _zmq

        self.ctx: Any = _zmq.Context.instance()
        self.sock = self.ctx.socket(_zmq.PUB if kind.upper() == "PUB" else _zmq.PUSH)
        (self.sock.bind if bind else self.sock.connect)(address)
        self.topic = topic
        super().__init__(
            lambda f: self.sock.send(
                self.topic + np.ascontiguousarray(f.data).tobytes()
            ),
            "zmq_sink",
        )

    def close(self) -> None:
        self.sock.close(linger=200)
