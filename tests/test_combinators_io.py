from collections.abc import Iterator

import numpy as np
import pytest

from sigflow import (
    Frame,
    Framed,
    Sig,
    TypeMismatch,
    block,
    dsp,
    feedback,
    from_iterable,
    io,
    run,
    source,
    when,
)
from sigflow.dims import N


@source
def ramp(count: int = 6, n: int = 3) -> Iterator[Sig[float, N]]:
    for i in range(count):
        yield np.full(n, float(i))


@block
def accumulate(
    x: Sig[float, N], total: Sig[float, N]
) -> tuple[Sig[float, N], Sig[float, N]]:
    s = x + total
    return s, s


def test_feedback_carries_state_between_frames():
    loop = feedback(accumulate, init=np.zeros(3))
    out = run(ramp() >> loop)
    assert [f.data[0] for f in out] == [0.0, 1.0, 3.0, 6.0, 10.0, 15.0]
    assert "loop" in out[-1].prov and out[-1].prov[-1].op == "ramp"
    with pytest.raises(TypeMismatch):
        feedback(dsp.scale(gain=2), init=0.0)  # body must take (x, state)


def test_feedback_body_can_span_blocks():
    @block
    def step(
        x: Sig[float, N], s: Sig[float, tuple[()]]
    ) -> tuple[Sig[float, N], Sig[float, tuple[()]]]:
        return x * s, float(s + 1)

    @block
    def tag(
        y: Sig[float, N], s: Sig[float, tuple[()]]
    ) -> tuple[Sig[float, N], Sig[float, tuple[()]]]:
        return y + s, s

    out = run(ramp(count=4) >> feedback(step >> tag, init=1.0))
    assert [f.data[0] for f in out] == [0 * 1 + 2, 1 * 2 + 3, 2 * 3 + 4, 3 * 4 + 5]


def test_when_routes_and_can_drop():
    @block
    def odd_only(x: Sig[float, N]) -> Sig[float, N]:
        return x if int(x[0]) % 2 else None

    g = ramp() >> when(lambda f: f.data[0] < 4, odd_only, otherwise=dsp.scale(gain=10))
    assert [f.data[0] for f in run(g)] == [1.0, 3.0, 40.0, 50.0]
    nested = ramp() >> when(
        lambda f: True,
        when(lambda f: f.data[0] < 4, odd_only, otherwise=dsp.scale(gain=10)),
    )
    assert [f.data[0] for f in run(nested)] == [1.0, 3.0, 40.0, 50.0]
    assert run(nested)[0].prov.op.startswith("when")


def test_when_sees_failed_frames_for_routing():
    @block
    def boom(x: Sig[float, N]) -> Sig[float, N]:
        if x[0] == 2:
            raise ValueError("x")
        return x

    @block
    def drop(f: Framed[Sig[float, N]]) -> Sig[float, N]:
        return None

    out = run(ramp(count=4) >> boom >> when(lambda f: f.failed, drop))
    assert [f.data[0] for f in out] == [0.0, 1.0, 3.0] and all(f.ok for f in out)


def test_graphs_and_blocks_apply_to_memory():
    front = dsp.scale(gain=2) >> dsp.rms
    assert [round(f.data, 3) for f in front([np.ones(4), np.zeros(4)])] == [2.0, 0.0]
    assert front(np.ones(4))[0].prov[-1].op == "input"
    frames = [Frame(np.ones(4), meta={"k": 1})]
    assert dsp.rms.apply(frames)[0]["k"] == 1
    with pytest.raises(TypeError):
        (ramp() >> dsp.rms)([np.ones(3)])  # has its own source


def test_file_io_roundtrips(tmp_path):
    from scipy.io import wavfile

    wav = tmp_path / "t.wav"
    wavfile.write(wav, 8000, (np.sin(np.arange(4000) * 0.1) * 32000).astype(np.int16))
    out = run(io.wav(str(wav), n=1000) >> dsp.rms)
    assert len(out) == 4 and out[0]["rate"] == 8000.0 and "path" in out[0].stream
    with io.wav_sink(str(tmp_path / "o.wav")) as snk:
        run(io.wav(str(wav), n=1000) >> snk)
    assert wavfile.read(tmp_path / "o.wav")[1].shape == (4000,)
    np.arange(2000, dtype=np.complex64).tofile(tmp_path / "x.c64")
    out = run(io.raw(str(tmp_path / "x.c64"), np.complex64, n=500) >> dsp.magnitude)
    assert len(out) == 4 and out[0].data.dtype == np.float32
    np.save(tmp_path / "a.npy", np.arange(3000.0))
    assert len(run(io.npy(str(tmp_path / "a.npy"), n=1000))) == 3
    with io.csv_log(str(tmp_path / "l.csv")) as log:
        run(ramp(count=3) >> dsp.rms >> log)
    rows = run(io.csv_rows(str(tmp_path / "l.csv")))
    assert len(rows) == 3 and rows[2].data["value"] == pytest.approx(2.0)
    with io.npy_sink(str(tmp_path / "s.npy")) as s:
        run(ramp(count=3) >> s)
    assert np.load(tmp_path / "s.npy").shape == (3, 3)


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("float32", [False, True])
def test_wav_channel_shapes(tmp_path, channels, float32):
    from scipy.io import wavfile

    data = np.arange(16 * channels, dtype=np.int16)
    if channels > 1:
        data = data.reshape(16, channels)
    path = tmp_path / "channels.wav"
    wavfile.write(path, 8000, data)

    src = io.wav(str(path), n=4, float32=float32)
    assert src.out_type.shape == (4, *data.shape[1:])
    out = run(src, check="all")
    expected = data.astype(np.float32) / 32767 if float32 else data
    np.testing.assert_allclose(np.concatenate([f.data for f in out]), expected)
    assert [f["t0"] for f in out] == [i / 8000 for i in range(0, 16, 4)]

    if channels > 1:
        with pytest.raises(TypeMismatch):
            src >> dsp.window  # a one-dimensional block cannot consume stereo frames
        selected = io.wav(str(path), n=4, channel=1, float32=float32)
        assert selected.out_type.shape == (4,)
        out = run(selected, check="all")
        np.testing.assert_allclose(
            np.concatenate([f.data for f in out]), expected[:, 1]
        )


def test_stereo_wav_without_mmap_support(tmp_path):
    import wave

    data = np.arange(32, dtype="<i4").reshape(16, 2)
    pcm = data.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    path = tmp_path / "24bit.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(3)
        wav.setframerate(8000)
        wav.writeframes(pcm)

    src = io.wav(str(path), n=4, float32=False)
    assert src.out_type.shape is None
    out = run(src, check="all")
    np.testing.assert_array_equal(np.concatenate([f.data for f in out]), data << 8)


def test_unsigned_wav_decoding_and_roundtrip(tmp_path):
    from scipy.io import wavfile

    data = np.arange(256, dtype=np.uint8)
    path = tmp_path / "unsigned.wav"
    wavfile.write(path, 8000, data)
    out = run(io.wav(str(path), n=64), check="all")
    decoded = np.concatenate([f.data for f in out])
    assert decoded.dtype == np.float32
    np.testing.assert_array_equal(decoded, (data.astype(np.float32) - 128) / 128)
    assert decoded[128] == 0

    raw = run(io.wav(str(path), n=64, float32=False), check="all")
    np.testing.assert_array_equal(np.concatenate([f.data for f in raw]), data)

    output = tmp_path / "roundtrip.wav"
    with io.wav_sink(str(output), dtype=np.uint8) as sink:
        run(from_iterable(out) >> sink)
    rate, encoded = wavfile.read(output)
    assert rate == 8000
    np.testing.assert_array_equal(encoded, data)


def test_unsigned_wav_sink_clips_and_centers(tmp_path):
    from scipy.io import wavfile

    path = tmp_path / "clipped.wav"
    with io.wav_sink(str(path), rate=8000, dtype=np.uint8) as sink:
        run(from_iterable([np.array([-2.0, -1.0, 0.0, 1.0, 2.0])]) >> sink)
    np.testing.assert_array_equal(wavfile.read(path)[1], [0, 0, 128, 255, 255])


def test_udp_source_and_sink():
    import time

    rx = (io.udp(port=9931, dtype=np.float32, count=3) >> dsp.rms).start()
    time.sleep(0.2)
    run(ramp(count=3) >> (lambda x: x.astype(np.float32)) >> io.udp_sink(port=9931))
    out = rx.wait(timeout=5)
    assert [round(f.data, 1) for f in out] == [0.0, 1.0, 2.0] and out[0][
        "from"
    ].startswith("127.0.0.1")


def test_report_and_string_annotation_guard():
    g = ramp() >> dsp.scale(gain=2) >> dsp.rms
    html, md = g.report("t"), g.report("t", fmt="md")
    assert (
        "<svg" in html
        and "scale" in html
        and md.startswith("# t")
        and "| `rms` |" in md
    )
    with pytest.raises(TypeError, match="could not be evaluated"):

        @block  # a string annotation naming something that does not exist is rejected
        def bad(f: "Framedd[Sig[float, N]]") -> Sig[float, N]:  # type: ignore[name-defined]  # noqa: F821
            return f
