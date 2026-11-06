"""Statistics across frames: moments match scipy on the whole stream, outliers and
change points are tagged where they are, and state is per node and checkpointed."""

from __future__ import annotations

import math
import time
from collections.abc import Iterator

import numpy as np
import pytest
from scipy import stats as ss

from sigflow import (
    Frame,
    Sig,
    as_node,
    dsp,
    iterate_sync,
    load_checkpoint,
    replay,
    run,
    source,
    stats,
)
from sigflow.dims import N
from sigflow.stats import Moments

RATE = 1000.0


def stream(x: np.ndarray, n: int = 64):
    # from_array called with an array runs at once: as_node makes it a graph node
    return as_node(dsp.from_array(np.asarray(x, dtype=float), n=n, rate=RATE))


# -- Moments ------------------------------------------------------------------ #


def test_merged_summaries_match_scipy_on_the_whole_sample():
    rng = np.random.default_rng(3)
    x = 1e7 + rng.gamma(2.0, 1.0, 5000)  # a large mean: where sum-of-squares fails
    cuts = sorted(rng.choice(np.arange(1, 5000), 30, replace=False))
    m = Moments()
    for part in np.split(x, cuts):
        m = m.merge(Moments.of(part))
    d = ss.describe(x)
    assert m.n == 5000
    assert m.mean == pytest.approx(d.mean, rel=1e-15, abs=1e-8)
    assert m.variance() == pytest.approx(d.variance, rel=1e-9)
    assert m.skewness() == pytest.approx(d.skewness, rel=1e-7)
    assert m.kurtosis() == pytest.approx(d.kurtosis, rel=1e-7)


def test_too_few_samples_give_nan_not_an_error():
    one = Moments.of([2.0])
    assert one.n == 1 and math.isnan(one.variance()) and math.isnan(one.skewness())
    assert Moments.of([np.nan, np.nan]).n == 0
    assert Moments.of([1.0, np.nan, 3.0]).mean == 2.0


# -- moments ------------------------------------------------------------------ #


@pytest.mark.parametrize("n", [1, 64])
def test_running_moments_match_describe_on_every_sample_so_far(n):
    rng = np.random.default_rng(1)
    x = rng.standard_t(5, 640)
    out = run(stream(x, n) >> stats.moments, on_error="raise")
    k = len(out) // 2
    count, mean, var, skew, kurt = out[k].data
    d = ss.describe(x[: (k + 1) * n])
    assert count == d.nobs
    np.testing.assert_allclose(
        [mean, var, skew, kurt], [d.mean, d.variance, d.skewness, d.kurtosis]
    )


def test_a_window_covers_the_last_frames():
    x = np.random.default_rng(2).normal(size=640)
    out = run(stream(x) >> stats.moments(window=3), on_error="raise")
    d = ss.describe(x[-192:])
    assert out[-1].data[0] == 192
    assert out[-1].data[2] == pytest.approx(d.variance)


def test_a_port_selects_one_statistic_and_scalars_count_once_each():
    rms = run(dsp.sine(n=32, frames=6) >> dsp.rms, on_error="raise")
    out = run(dsp.sine(n=32, frames=6) >> dsp.rms >> stats.moments["mean"])
    assert out[-1].data == pytest.approx(np.mean([f.data for f in rms]))


def test_each_use_has_its_own_state():
    ones = stream(np.ones(128))
    a = run(ones >> stats.moments["count"], on_error="raise")
    b = run(stream(np.ones(128)) >> stats.moments["count"], on_error="raise")
    assert [f.data for f in a] == [f.data for f in b] == [64, 128]
    both = run(
        stream(np.ones(128))
        >> [stats.moments["count"], stats.moments(window=1)["count"]],
        on_error="raise",
    )
    assert [f.data for f in both] == [(64, 64), (128, 64)]


def test_every_run_starts_afresh_and_replay_agrees():
    g = dsp.sine(n=8, frames=5) >> stats.moments["count"]
    first = run(g)
    assert [f.data for f in run(g)] == [f.data for f in first] == [8, 16, 24, 32, 40]
    assert [f.data for f in iterate_sync(g)] == [8, 16, 24, 32, 40]
    assert replay(g, first[2]).data == 24


def test_checkpoint_and_resume_carry_the_running_summary(tmp_path):
    x = np.random.default_rng(4).normal(size=64 * 40)

    def make():
        return stream(x) >> stats.moments

    h = make().start(paused=True)
    h.step(10)
    deadline = time.time() + 10
    while h.monitor.produced < 10 and time.time() < deadline:
        time.sleep(0.01)
    ck = h.checkpoint(str(tmp_path / "ck.pkl"))
    h.stop()
    assert "moments" in ck.states and "moments" not in ck.skipped
    rest = run(make(), resume=load_checkpoint(str(tmp_path / "ck.pkl")))
    full = run(make())
    assert rest[-1].data == pytest.approx(full[-1].data)
    # iterating resumes the same way, from the checkpoint or its path
    again = list(iterate_sync(make(), resume=str(tmp_path / "ck.pkl")))
    assert again[0].data[0] == full[10].data[0] == 64 * 11


# -- outliers ----------------------------------------------------------------- #


@source
def spiky(n: int = 64, frames: int = 3) -> Iterator[Sig[float, N]]:
    rng = np.random.default_rng(5)
    for i in range(frames):
        x = rng.normal(0, 1, n)
        if i == 1:
            x[10:13] = 40.0  # one run of three outliers
            x[50] = -30.0  # and a single one
        yield Frame(x, meta={"t0": i * n / RATE}, stream={"rate": RATE})


def test_outliers_are_tagged_over_their_time_span():
    out = run(spiky() >> stats.outliers, on_error="raise")
    assert [("outlier" in f.tags) for f in out] == [False, True, False]
    f = out[1]
    assert np.flatnonzero(f.mask("outlier")).tolist() == [10, 11, 12, 50]
    spans = sorted(t.span for t in f.tags if t.name == "outlier")  # type: ignore[type-var]
    assert len(spans) == 2
    assert all(t.value > 3.5 for t in f.tags if t.name == "outlier")
    np.testing.assert_array_equal(f.data, run(spiky())[1].data)  # data untouched


def test_a_window_judges_against_recent_frames():
    @source
    def mostly_high() -> Iterator[Sig[float, N]]:
        rng = np.random.default_rng(6)
        for i in range(4):
            x = rng.normal(0, 1, 16) + (50.0 if i == 3 else 0.0)  # a whole odd frame
            yield Frame(x, meta={"t0": i * 16 / RATE}, stream={"rate": RATE})

    alone = run(mostly_high() >> stats.outliers, on_error="raise")
    windowed = run(mostly_high() >> stats.outliers(window=4), on_error="raise")
    assert "outlier" not in alone[3].tags  # by itself the frame is consistent
    assert windowed[3].mask("outlier").all()


def test_a_frame_without_time_gets_a_whole_frame_tag():
    x = np.zeros(64)
    x[5] = 1.0
    out = run(dsp.from_array(np.r_[x], n=64) >> stats.outliers, on_error="raise")
    assert out[0].tags["outlier"] == math.inf


def test_the_contract_passes_through_and_names_the_tag():
    from sigflow.types import maybe_names

    g = dsp.sine(n=64, frames=1) >> stats.outliers
    assert "outlier" in maybe_names(g.out_type)
    assert g.out_type.shape == (64,)


# -- change_point ------------------------------------------------------------- #


def steps(seed: int, sd_after: float = 1.0, mean_after: float = 0.0):
    rng = np.random.default_rng(seed)
    before = rng.normal(0, 1, 64 * 40)
    after = rng.normal(mean_after, sd_after, 64 * 20)
    return stream(np.r_[before, after])


def changes(out: list[Frame]) -> list[tuple[int, str]]:
    return [(i, f.tags["changed"]) for i, f in enumerate(out) if "changed" in f.tags]


def test_a_step_in_spread_is_found_soon_after_it_happens():
    out = run(steps(7, sd_after=1.6) >> stats.change_point, on_error="raise")
    after = [(i, d) for i, d in changes(out) if i >= 40]  # the step is at frame 40
    assert after and after[0][1] == "up" and after[0][0] <= 45
    onset = out[after[0][0]].meta["change_onset"]  # a frame index: no t0 here
    assert 38 <= onset <= after[0][0]


def test_a_drop_in_the_mean_is_found_with_its_direction():
    g = steps(8, mean_after=-0.5) >> stats.change_point(on="mean")
    after = [(i, d) for i, d in changes(run(g, on_error="raise")) if i >= 40]
    assert after and after[0][1] == "down" and after[0][0] <= 45


def test_a_steady_stream_rarely_raises_an_alarm():
    alarms = sum(
        bool(changes(run(steps(seed) >> stats.change_point, on_error="raise")))
        for seed in range(40)
    )
    assert alarms <= 6  # about 1 in 25 runs of 50 tested frames, at the defaults


def test_bad_options_are_refused():
    with pytest.raises(ValueError, match="on must be"):
        run(steps(1) >> stats.change_point(on="median"), on_error="raise")
    with pytest.raises(ValueError, match="window"):
        run(steps(1) >> stats.moments(window=0), on_error="raise")
