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


def rvs(dist, *args, **kwargs) -> np.ndarray:
    """Random samples as a float array (scipy's stubs allow a scalar)."""
    return np.asarray(dist.rvs(*args, **kwargs), dtype=float)


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


# -- fit ---------------------------------------------------------------------- #


def timed(x: np.ndarray, n: int = 256):
    """Frames of ``x`` with start times, so tags have time spans."""
    frames = [
        Frame(x[i : i + n], meta={"t0": i / RATE}, stream={"rate": RATE})
        for i in range(0, len(x) - n + 1, n)
    ]
    return as_node(iter(frames))


SAMPLES = {
    "norm": lambda r: r.normal(3, 2, 3200),
    "expon": lambda r: r.exponential(2, 3200),
    "rayleigh": lambda r: rvs(ss.rayleigh, scale=2, size=3200, random_state=r),
    "lognorm": lambda r: r.lognormal(0.5, 0.4, 3200),
    "gamma": lambda r: r.gamma(2, 1.5, 3200),
}


@pytest.mark.parametrize("name", sorted(stats.EXACT_FITS))
def test_running_fits_match_fitting_every_sample(name):
    x = SAMPLES[name](np.random.default_rng(1))
    out = run(stream(x) >> stats.fit(dist=name), on_error="raise")
    expected = np.asarray(
        getattr(ss, name).fit(x, **({} if name == "norm" else {"floc": 0})), dtype=float
    )
    np.testing.assert_allclose(out[-1].data, expected, rtol=1e-9, atol=1e-12)
    assert out[-1].stream["params"] == stats.param_names(name)
    assert out[-1].stream["dist"] == name
    assert out[-1].meta["fit_samples"] == 3200
    assert "rate" not in out[-1].stream  # parameters are not samples


def test_a_window_fits_the_last_frames_and_every_spaces_the_refits():
    x = rvs(ss.weibull_min, 1.5, scale=2, size=64 * 30, random_state=2)
    out = run(
        stream(x) >> stats.fit(dist="weibull_min", window=10, every=5),
        on_error="raise",
    )
    assert len(out) == 30
    np.testing.assert_array_equal(out[25].data, out[29].data)  # one refit in 5
    expected = np.asarray(ss.weibull_min.fit(x[64 * 16 : 64 * 26], floc=0), dtype=float)
    np.testing.assert_allclose(out[25].data, expected, rtol=1e-6)


def test_the_output_contract_has_one_value_per_parameter():
    assert (dsp.sine(n=8, frames=1) >> stats.fit(dist="gamma")).out_type.shape == (3,)
    assert (dsp.sine(n=8, frames=1) >> stats.fit).out_type.shape == (2,)


def test_samples_outside_the_support_are_ignored_with_a_warning():
    x = np.r_[np.full(10, -1.0), rvs(ss.rayleigh, size=54, random_state=3)]
    out = run(stream(x) >> stats.fit(dist="rayleigh"), on_error="raise")
    assert out[0].meta["fit_samples"] == 54
    assert any("10 samples outside" in i.message for i in out[0].warnings)


def test_a_distribution_without_running_sums_needs_a_window():
    with pytest.raises(ValueError, match="needs the samples: give window="):
        run(stream(np.ones(64)) >> stats.fit(dist="weibull_min"), on_error="raise")
    with pytest.raises(ValueError, match="no continuous distribution"):
        run(stream(np.ones(64)) >> stats.fit(dist="nonsense"), on_error="raise")
    with pytest.raises(ValueError, match="no parameter c"):
        run(
            stream(np.ones(64)) >> stats.fit(dist="gamma", fixed={"c": 1}),
            on_error="raise",
        )


def test_goodness_of_fit_tells_a_wrong_family():
    rng = np.random.default_rng(4)
    for x, poor in [
        (rng.gamma(2, 1.5, 64 * 12), False),
        (rng.lognormal(0.5, 0.6, 64 * 12), True),
    ]:
        out = run(
            stream(x) >> stats.fit(dist="gamma", window=10, gof_every=11),
            on_error="raise",
        )
        assert (
            "poor_fit" in out[-1].tags
        ) == poor  # tests at refits 1 and 12: frame 11, 640 samples
    with pytest.raises(ValueError, match="can never reject"):
        run(
            stream(np.ones(64))
            >> stats.fit(dist="gamma", window=2, gof_every=1, alpha=0.01),
            on_error="raise",
        )


def test_a_fit_starts_afresh_each_run():
    g = dsp.noise(n=64, frames=10, seed=5) >> stats.fit
    first, second = run(g), run(g)
    assert first[-1].meta["fit_samples"] == second[-1].meta["fit_samples"] == 640
    np.testing.assert_array_equal(first[-1].data, second[-1].data)


# -- threshold ---------------------------------------------------------------- #


def test_the_false_alarm_rate_is_what_was_set():
    alarms = tested = 0
    for seed in range(8):
        x = rvs(ss.rayleigh, scale=1.7, size=256 * 60, random_state=seed)
        out = run(timed(x) >> stats.threshold(pfa=1e-2), on_error="raise")
        for f in out[5:]:
            tested += len(f.data)
            alarms += int(f.mask("detection").sum())
    assert 0.8e-2 < alarms / tested < 1.2e-2


def test_a_target_is_found_and_kept_out_of_the_noise_fit():
    x = rvs(ss.rayleigh, size=256 * 20, random_state=3)
    x[256 * 12 + 100 : 256 * 12 + 103] += 12.0
    out = run(timed(x) >> stats.threshold(pfa=1e-6), on_error="raise")
    found = [i for i, f in enumerate(out) if "detection" in f.tags]
    assert found == [12]
    assert np.flatnonzero(out[12].mask("detection")).tolist() == [100, 101, 102]
    assert out[12].tags["detection"] == pytest.approx(
        x[256 * 12 + 100 : 256 * 12 + 103].max()
    )
    assert out[13].meta["threshold"] == pytest.approx(
        out[12].meta["threshold"], rel=0.02
    )
    assert "threshold" not in out[4].meta  # the first 5 frames only learn


def test_the_threshold_follows_the_noise():
    level = np.repeat([1.0, 4.0], 256 * 10)
    x = level * rvs(ss.rayleigh, size=level.size, random_state=6)
    out = run(timed(x) >> stats.threshold(pfa=1e-3, window=3), on_error="raise")
    assert out[19].meta["threshold"] > 3 * out[9].meta["threshold"]


def test_frames_without_time_mark_detections_per_sample():
    x = rvs(ss.rayleigh, size=256 * 8, random_state=7)
    x[256 * 6 + 40] = 50.0
    frames = [
        Frame(x[i : i + 256], stream={"rate": RATE}) for i in range(0, len(x), 256)
    ]
    out = run(as_node(iter(frames)) >> stats.threshold(pfa=1e-6), on_error="raise")
    assert np.flatnonzero(out[6].samples["detection"]).tolist() == [40]


@pytest.mark.parametrize(
    "make_block",
    [
        lambda: stats.fit(dist="gamma"),
        lambda: stats.fit(dist="weibull_min", window=3),
        lambda: stats.threshold(dist="norm"),
    ],
)
def test_fits_checkpoint_to_a_file_and_resume(tmp_path, make_block):
    def make():
        return (
            dsp.noise(n=64, frames=30, seed=1) >> dsp.offset(value=10.0) >> make_block()
        )

    h = make().start(paused=True)
    h.step(8)
    deadline = time.time() + 10
    while h.monitor.produced < 8 and time.time() < deadline:
        time.sleep(0.01)
    h.checkpoint(str(tmp_path / "ck.pkl"))
    h.stop()
    rest = run(make(), resume=str(tmp_path / "ck.pkl"))
    full = run(make())
    np.testing.assert_array_equal(rest[-1].data, full[-1].data)
    assert rest[-1].meta.get("threshold") == full[-1].meta.get("threshold")


# -- review fixes: run-local state, calibration, support ends, zero spread ---- #


def test_a_reused_node_keeps_a_state_of_its_own():
    count = stats.moments["count"]
    out = run(dsp.sine(n=4, frames=2) >> [count, count], on_error="raise")
    assert [f.data for f in out] == [(4, 4), (8, 8)]


def test_runs_of_one_graph_side_by_side_do_not_disturb_each_other():
    import asyncio

    from sigflow import iterate

    g = dsp.sine(n=4, frames=4) >> stats.moments["count"]

    async def two():
        first = iterate(g)
        a = await first.__anext__()
        second = iterate(g)  # started, never consumed
        b = await first.__anext__()
        await first.aclose()
        await second.aclose()
        return a.data, b.data

    assert asyncio.run(two()) == (4, 8)


def test_a_bound_block_stays_run_local_and_checkpointed():
    bound = stats.moments.bind(window=2)
    g = dsp.sine(n=4, frames=3) >> bound["count"]
    assert [f.data for f in run(g)] == [f.data for f in run(g)] == [4, 8, 8]

    def make():
        return dsp.sine(n=4, frames=30) >> stats.moments.bind(window=50)["count"]

    h = make().start(paused=True)
    h.step(5)
    deadline = time.time() + 10
    while h.monitor.produced < 5 and time.time() < deadline:
        time.sleep(0.01)
    ck = h.checkpoint()
    h.stop()
    assert "moments" in ck.states
    assert run(make(), resume=ck)[0].data == 24


@pytest.mark.parametrize("backend", ["thread", "process", "loky"])
def test_statistics_are_refused_by_every_parallel_backend(backend):
    from sigflow import parallel

    with pytest.raises(TypeError, match="keeps state across frames"):
        parallel(stats.moments, backend=backend)


@pytest.mark.parametrize("pfa", [1e-1, 1e-2])
def test_noise_alone_keeps_the_false_alarm_rate_when_targets_are_excluded(pfa):
    alarms = tested = 0
    for seed in range(6):
        x = rvs(ss.rayleigh, size=256 * 60, random_state=seed)
        out = run(timed(x) >> stats.threshold(pfa=pfa, window=20), on_error="raise")
        for f in out[5:]:
            tested += len(f.data)
            alarms += int(f.mask("detection").sum())
    assert 0.85 < alarms / tested / pfa < 1.15  # leaving out every detection: 2.3x


def test_a_strong_target_is_still_kept_out_of_the_noise_fit():
    x = rvs(ss.rayleigh, size=256 * 30, random_state=1)
    x[256 * 12 : 256 * 12 + 60] += 25.0
    kept = run(timed(x) >> stats.threshold(pfa=1e-4, window=3), on_error="raise")
    raw = run(timed(x) >> stats.threshold(pfa=1e-4, window=3, exclude=None))
    assert kept[14].meta["threshold"] < 1.1 * kept[12].meta["threshold"]
    assert raw[14].meta["threshold"] > 3 * raw[12].meta["threshold"]


def test_samples_at_the_end_of_the_support_count_where_scipy_counts_them():
    zeros = np.array([0.0, 0.0, 1.0, 1.0])
    out = run(stream(zeros, n=4) >> stats.fit(dist="expon"), on_error="raise")
    np.testing.assert_allclose(
        out[0].data, np.asarray(ss.expon.fit(zeros, floc=0), dtype=float)
    )
    assert out[0].meta["fit_samples"] == 4 and not out[0].warnings
    x = np.r_[0.0, 0.0, np.random.default_rng(0).gamma(2, 1.5, 62)]
    out = run(stream(x) >> stats.fit(dist="weibull_min", window=1), on_error="raise")
    expected = np.asarray(ss.weibull_min.fit(x, floc=0), dtype=float)
    np.testing.assert_allclose(out[0].data, expected, rtol=1e-6)
    out = run(stream(x) >> stats.fit(dist="rayleigh"), on_error="raise")
    assert out[0].meta["fit_samples"] == 62  # no likelihood at 0, as scipy says
    assert any("lower end" in i.message for i in out[0].warnings)


def test_spread_vanishing_or_appearing_is_a_change():
    rng = np.random.default_rng(0)

    def found(x, **kw):
        out = run(stream(x) >> stats.change_point(**kw), on_error="raise")
        return [
            (i, f.tags["changed"]) for i, f in enumerate(out) if "changed" in f.tags
        ]

    assert found(np.r_[rng.normal(size=640), np.zeros(1280)]) == [(10, "down")]
    assert found(np.r_[np.zeros(640), rng.normal(size=1280)]) == [(10, "up")]
    assert found(np.zeros(1920)) == []
    assert found(np.r_[np.full(640, 2.0), np.full(640, 3.0)], on="mean") == [(10, "up")]


def test_a_goodness_of_fit_that_cannot_run_says_why(monkeypatch):
    from sigflow import stats as module

    def broken(*args, **kwargs):
        raise RuntimeError("no simulation today")

    monkeypatch.setattr(module._scipy_stats, "goodness_of_fit", broken)
    x = np.random.default_rng(1).gamma(2, 1.5, 64 * 3)
    out = run(stream(x) >> stats.fit(dist="gamma", window=2, gof_every=1))
    assert math.isnan(out[-1].meta["fit_pvalue"])
    assert any("no simulation today" in i.message for i in out[-1].warnings)


def test_scipy_calls_use_forms_every_supported_version_takes():
    import inspect

    from sigflow import stats as module

    params = inspect.signature(ss.goodness_of_fit).parameters
    assert module._GOF_SEED in params
    x = np.random.default_rng(2).normal(size=50)
    m = Moments.of(x)
    np.testing.assert_allclose(
        [m.m2, m.m3, m.m4], [ss.moment(x, k) * len(x) for k in (2, 3, 4)]
    )
