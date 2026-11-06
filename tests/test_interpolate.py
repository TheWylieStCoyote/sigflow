"""Interpolation across frame boundaries: a stream interpolated in frames matches
the whole signal interpolated at once, and invented samples are tagged."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from scipy import interpolate as si

from sigflow import Frame, Sig, dsp, run, source
from sigflow import interpolate as ip
from sigflow.dims import N

RATE = 1000.0


def signal(t: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * 37 * t) + 0.3 * np.sin(2 * np.pi * 110 * t)


@source
def frames(
    starts: tuple = (0, 1, 2, 3), n: int = 16, tags: tuple = ()
) -> Iterator[Sig[float, N]]:
    """Frames of ``signal`` starting at frame indices ``starts`` (a missing index is
    a hole); ``tags[i]`` is a dict of whole-frame tags for frame ``i``."""
    for i, s in enumerate(starts):
        t0 = s * n / RATE
        f = Frame(
            signal(t0 + np.arange(n) / RATE),
            meta={"t0": t0},
            stream={"rate": RATE},
        )
        if i < len(tags) and tags[i]:
            f = f.tag(**tags[i])
        yield f


def index(f: Frame) -> int:
    """Which 16-sample frame slot ``f`` starts at."""
    assert f.t0 is not None
    return round(f.t0 * RATE / 16)


def joined(out: list[Frame]) -> tuple[np.ndarray, np.ndarray]:
    data = np.concatenate([f.data for f in out])
    times = np.concatenate([f.time.stamps() for f in out])  # type: ignore[union-attr]
    return times, data


@pytest.mark.parametrize("kind", list(ip.KINDS))
@pytest.mark.parametrize("n", [1, 7, 64])
def test_resampling_in_frames_matches_the_whole_signal(kind, n):
    total = n * (640 // n)  # from_array drops a partial last frame
    x = signal(np.arange(total) / RATE)
    out = run(
        dsp.from_array(x, n=n, rate=RATE) >> ip.resample(rate=2400.0, kind=kind),
        on_error="raise",
    )
    t, y = joined(out)
    grid = np.arange(int((total - 1) / RATE * 2400) + 1) / 2400
    np.testing.assert_allclose(t, grid, atol=1e-12)
    whole = ip.interpolator(np.arange(total) / RATE, x, kind)(grid)
    tol = 1e-9 if kind in ("cubic", "spline") else 1e-12
    np.testing.assert_allclose(y, whole, atol=tol)
    assert all(f.stream["rate"] == 2400.0 for f in out)


def test_complex_samples_interpolate_as_real_and_imaginary_parts():
    t = np.arange(256) / RATE
    z = np.exp(2j * np.pi * 37 * t)
    out = run(
        dsp.from_array(z, n=32, rate=RATE) >> ip.resample(rate=1500.0, kind="akima"),
        on_error="raise",
    )
    tq, y = joined(out)
    re = si.Akima1DInterpolator(t, np.real(z))(tq)
    im = si.Akima1DInterpolator(t, np.imag(z))(tq)
    assert np.iscomplexobj(y)
    np.testing.assert_allclose(y, re + 1j * im, atol=1e-12)


def test_resampling_across_a_hole_tags_the_invented_samples():
    out = run(frames(starts=(0, 1, 3, 4)) >> ip.resample(rate=2000.0), on_error="raise")
    hole = (2 * 16 / RATE, 3 * 16 / RATE)  # frame 2 is missing
    for f in out:
        for t, y in zip(f.time.stamps(), f.data):  # type: ignore[union-attr]
            inside = hole[0] <= t < hole[1]
            assert ("interpolated" in f.tags.at(t)) == inside, t
            if not inside:  # real samples are on the curve
                assert abs(y - signal(np.array(t))) < 0.05


def test_resampling_keeps_whole_frame_tags_where_they_apply():
    out = run(
        frames(starts=(0, 1, 2, 3), tags=({"take": 1}, {"take": 1}, {}, {"take": 2}))
        >> ip.resample(rate=1000.0, kind="linear"),  # same grid: easy to read
        on_error="raise",
    )
    for f in out:
        for t in f.time.stamps():  # type: ignore[union-attr]
            frame_index = int(t * RATE // 16)
            expected = {0: 1, 1: 1, 2: None, 3: 2}[frame_index]
            assert f.tags.at(t).get("take") == expected, t


def test_filling_a_hole_inserts_tagged_frames_of_the_same_length():
    out = run(
        frames(starts=(0, 1, 4, 5), tags=({"ch": 3},) * 4) >> ip.fill_gaps(),
        on_error="raise",
    )
    assert [index(f) for f in out] == [0, 1, 2, 3, 4, 5]
    filled = [f for f in out if "interpolated" in f.tags]
    assert [index(f) for f in filled] == [2, 3]
    assert all(len(f.data) == 16 and f.tags["ch"] == 3 for f in out)
    t, _ = joined(out)
    np.testing.assert_allclose(t, np.arange(96) / RATE, atol=1e-12)
    real = [f for f in out if "interpolated" not in f.tags]
    for f in real:  # frames around the hole pass through untouched
        np.testing.assert_array_equal(f.data, signal(f.time.stamps()))  # type: ignore[union-attr]


def test_holes_longer_than_max_gap_are_left_alone():
    out = run(
        frames(starts=(0, 1, 2, 6, 8)) >> ip.fill_gaps(max_gap=2 * 16 / RATE),
        on_error="raise",
    )
    assert [index(f) for f in out] == [0, 1, 2, 6, 7, 8]


def test_a_partial_hole_gets_a_short_frame():
    @source
    def uneven() -> Iterator[Sig[float, N]]:
        for t0 in (0.0, 0.010, 0.025):  # 10-sample frames: a 5-sample hole
            yield Frame(np.zeros(10), meta={"t0": t0}, stream={"rate": RATE})

    out = run(uneven() >> ip.fill_gaps(kind="linear"), on_error="raise")
    assert [len(f.data) for f in out] == [10, 10, 5, 10]
    assert "interpolated" in out[2].tags


def test_frames_without_time_pass_through():
    out = run(dsp.from_array(np.arange(8.0), n=4) >> ip.fill_gaps(), on_error="raise")
    assert len(out) == 2


def test_bad_options_are_refused():
    with pytest.raises(ValueError, match="kind must be one of"):
        run(dsp.sine(n=8, frames=2) >> ip.resample(kind="quintic"), on_error="raise")
    with pytest.raises(ValueError, match="needs context >= 3"):
        run(
            dsp.sine(n=8, frames=2) >> ip.resample(kind="akima", context=2),
            on_error="raise",
        )
    with pytest.raises(ValueError, match="rate"):
        run(dsp.from_array(np.arange(8.0), n=4) >> ip.resample(), on_error="raise")


def test_short_context_trades_accuracy_for_latency_in_global_splines():
    x = signal(np.arange(256) / RATE)
    grid = np.arange(int(255 / RATE * 2400) + 1) / 2400
    whole = ip.interpolator(np.arange(256) / RATE, x, "cubic")(grid)
    errors = []
    for context in (2, 8):
        out = run(
            dsp.from_array(x, n=16, rate=RATE)
            >> ip.resample(rate=2400.0, kind="cubic", context=context),
            on_error="raise",
        )
        errors.append(np.abs(joined(out)[1] - whole).max())
    assert 1e-12 < errors[1] < errors[0] < 1e-2


def test_the_contract_says_the_tag_may_appear():
    from sigflow.types import maybe_names

    assert "interpolated" in maybe_names(ip.resample.out_type)
