"""Real I/O boundaries: files in, files out, a socket, and a GNU Radio hand-off.

What this exercises
  * io.wav / io.npy / io.raw sources: memory-mapped, chunked, carrying rate + path
    annotations and a frame count for progress
  * sinks as taps: io.wav_sink, io.npy_sink, io.csv_log (context managers that flush
    on close) — a sink anywhere in a chain, not only at the end
  * a GNU Radio file-sink format (raw complex64) read back, and written for GR to read
  * an asynchronous UDP receiver fed by a sender graph: two graphs talking through a socket
  * provenance across the file boundary: the output frame knows the file and offset it
    came from
"""
# mypy: disable-error-code="valid-type, misc, syntax, return-value, union-attr, index"

import os
import tempfile
import time

import numpy as np
from scipy.io import wavfile

from sigflow import Sig, block, dsp, io, run

WORK = tempfile.mkdtemp(prefix="sigflow_io_")
RATE = 16_000


def make_fixture_wav(path: str, seconds: float = 2.0) -> None:
    t = np.arange(int(seconds * RATE)) / RATE
    x = 0.5 * np.sin(2 * np.pi * 440 * t) * (1 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
    wavfile.write(path, RATE, (x * 32767).astype(np.int16))


if __name__ == "__main__":
    wav_in = os.path.join(WORK, "input.wav")
    make_fixture_wav(wav_in)

    # 1. WAV -> process -> WAV + CSV log, all in one graph. Sinks are taps, so the
    #    envelope log sits *before* the de-emphasis and both files come out of one run.
    wav_out = io.wav_sink(os.path.join(WORK, "output.wav"))
    log = io.csv_log(os.path.join(WORK, "levels.csv"), fields=["seq", "t0", "value"])
    g = (
        io.wav(wav_in, n=512)
        >> dsp.scale(gain=0.8)
        >> (dsp.rms >> log | dsp.lfilter(b=[1, -0.95]) >> wav_out)[1]
    )
    print(g.describe(), "\n")
    g.draw().save(os.path.join(WORK, "real_io_graph.svg"))
    with wav_out, log:
        out = run(g, progress=True)
    rate, y = wavfile.read(wav_out.path)
    print(
        f"\nwrote {wav_out.path}: {len(y)} samples at {rate} Hz;  {log.path}: "
        f"{sum(1 for _ in open(log.path)) - 1} rows"
    )
    f = out[-1]
    print(
        "last frame came from",
        f["path"].split(os.sep)[-1],
        "offset",
        f["offset"],
        "| lineage:",
        [n.op for n in f.prov.chain()],
    )

    # 2. GNU Radio hand-off: write what a GR file sink writes (raw complex64), read it back.
    #    `io.raw` is memory-mapped, so gigabyte captures stream without loading.
    c64 = os.path.join(WORK, "capture.c64")
    with io.npy_sink(os.path.join(WORK, "capture.npy")) as cap:
        run(
            dsp.sine(freq=1000, rate=RATE, n=1024, frames=20)
            >> dsp.to_complex
            >> (lambda x: x.astype(np.complex64))
            >> cap
        )
    np.load(cap.path).reshape(-1).tofile(
        c64
    )  # GR file-sink layout: interleaved float32 I/Q
    back = run(
        io.raw(c64, np.complex64, n=2048, rate=RATE) >> dsp.fft >> dsp.power_db,
        progress=True,
    )
    peak_bin = int(np.argmax(back[0].data[:1024]))
    print(
        f"\n{len(back)} frames from {os.path.getsize(c64)} bytes; tone at {peak_bin * RATE / 2048:.0f} Hz "
        f"(source rate from annotation: {back[0]['rate']} Hz)"
    )

    # 3. Two graphs through a UDP socket: a receiver started in the background, a sender
    #    run in the foreground. Datagrams become frames with the sender address in meta.
    port = 9910
    receiver = (io.udp(port=port, dtype=np.float32, count=8) >> dsp.rms).start()
    time.sleep(0.2)  # let the socket bind
    sender = (
        dsp.sine(n=256, frames=8)
        >> (lambda x: x.astype(np.float32))
        >> io.udp_sink(port=port)
    )
    run(sender)
    received = receiver.wait(timeout=5)
    print(
        f"\nUDP: {len(received)} frames, rms {[round(f.data, 3) for f in received][:4]}..., "
        f"from {received[0]['from']}, stream {received[0].stream}"
    )

    # 4. A .npy capture streamed in chunks with a frame count for progress
    npy = os.path.join(WORK, "long.npy")
    np.save(npy, np.random.default_rng(0).normal(size=200_000))
    out = run(io.npy(npy, n=4096, rate=RATE) >> dsp.rms, progress=True)
    print(f"\n{len(out)} chunks of 4096 from {npy.split(os.sep)[-1]}")
    print("work dir:", WORK)
