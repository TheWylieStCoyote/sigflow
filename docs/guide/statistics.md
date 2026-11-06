# Statistics across frames

`sigflow.stats` computes statistics of a stream rather than of one frame: a
frame of 64 samples is a small sample, and a statistic or a test over it alone
is noisy. These blocks keep what they need of the stream's history.

```python
from sigflow import dsp, run, stats

g = dsp.sine(n=64, frames=100) >> stats.moments
count, mean, var, skew, kurt = run(g)[-1].data  # over all 6400 samples
```

| block | gives | state it keeps |
|---|---|---|
| `moments` | count, mean, variance, skewness, kurtosis | a summary of every sample so far, or of the last `window` frames |
| `outliers` | the frame, with `outlier` tags on the outlying samples | the last `window` frames, if `window` is set |
| `change_point` | the frame, with a `changed` tag where the mean or spread changes | a reference summary and two running sums |
| `fit` | the parameters of a fitted `scipy.stats` distribution | running sums, or the last `window` frames' samples |
| `threshold` | the frame, with `detection` tags on samples above a level set by a fitted noise distribution | as `fit` |

## Moments

`moments` has five output ports: `count`, `mean`, `var`, `skew` and `kurt`.
Select one with `stats.moments["var"]`. The values are those of
`scipy.stats.describe` on the same samples: unbiased variance, biased
skewness, and excess (Fisher) kurtosis, which is 0 for a normal distribution.

- **Over the whole stream** by default; `moments(window=10)` covers the last
  10 frames.
- **Every sample counts.** A frame's array is flattened and its NaNs
  skipped. A stream of one value per frame, such as `dsp.rms`, gives the
  statistics of those values.
- **Exact for long streams.** Each frame is summarised with
  `scipy.stats.moment`, and the running summary (`sigflow.stats.Moments`) is
  combined frame by frame with the formulas for merging two summaries, which
  scipy does not provide. No samples are kept, and the summary does not lose
  precision the way the textbook sum-of-squares variance does when the mean is
  large. Two `Moments` combine
  with `a.merge(b)`, which also suits summaries computed separately.
- Too few samples (a variance of one sample) give NaN, not an error.

## Outliers

`outliers` scores each sample by its distance from the median, in units of
the scaled median absolute deviation (MAD): `|x - median| / (1.4826 * MAD)`.
For normal data this is about a z-score. Unlike the standard deviation, the
MAD barely moves when the outliers themselves are large. Samples scoring above
`threshold` (3.5 by default) are tagged.

```python
out = run(source >> stats.outliers(threshold=5))
for f in out:
    bad = f.mask("outlier")  # True on the outlying samples
```

- Each run of outlying samples gets one `outlier` tag over its time span,
  with its largest score as the value. A frame without a time axis gets one
  whole-frame tag.
- The median and MAD come from the frame itself, or with `window=k` from the
  last `k` frames. Use a window for short frames, or when a whole frame may be
  odd: judged by itself, a frame that is entirely off looks consistent.
- The data and its contract pass through unchanged.

## Change points

`change_point` tags the frame where the stream's spread (`on="var"`, the
default) or its mean (`on="mean"`) changes, with a two-sided CUSUM test.

1. The first `reference` frames (10) set what is normal.
2. Each later frame gets a score that is about standard normal while nothing
   has changed.
3. Two running sums collect the scores beyond `drift`, one upward and one
   downward, falling back to zero while the stream is normal.
4. When one passes `threshold`, the frame is tagged `changed` with the value
   `"up"` or `"down"`. Its `change_onset` annotation is the start time of the
   frame where that sum last left zero (the frame index, if frames have no
   `t0`), which is the likely start of the change. The test then learns the
   new normal from the next `reference` frames.

A spread of exactly zero on one side only is a change beyond doubt: a signal
stopping to a constant (`"down"`) or starting from silence (`"up"`) alarms on
the first frame. With `on="mean"`, a constant reference makes any other mean
a change.

Why a sequential test rather than testing each frame? On frames of 64
normal samples with a 1.6-fold step in standard deviation (200 trials of 100
frames, step at frame 60):

| detector | runs with a false alarm | change detected | median delay |
|---|---|---|---|
| KS test, each frame against the reference | 1% | 32% | 19 frames |
| KS test, last 5 frames against the reference | 0% | 98% | 6 frames |
| CUSUM on the log-variance (this block) | 4% | 100% | 2 frames |

A single frame has too few samples for a test to see a modest change. CUSUM
accumulates evidence across frames and reacts within a frame or two.

**Tuning.** `drift` is half the smallest shift worth detecting, in standard
deviations of the score. `threshold` trades false alarms for delay: raise it
for fewer false alarms and later detection. The scores assume independent
samples, so correlated noise (filtered, or oversampled) needs a larger
`threshold`.

## Fitting distributions

`fit` fits a continuous distribution of `scipy.stats` to the stream's samples
by maximum likelihood:

```python
last = run(source >> stats.fit(dist="gamma"))[-1]
a, loc, scale = last.data
model = scipy.stats.gamma(*last.data)  # the fitted distribution
```

The output is a vector of the parameters in scipy's order: the shapes, then
`loc`, then `scale`. `stats.param_names("gamma")` gives `("a", "loc",
"scale")`, and the frame's `params` annotation names them too.

**Fixed parameters.** `fixed={"loc": 0}` holds a parameter instead of
fitting it. A distribution bounded below (`rayleigh`, `gamma`, `weibull_min`,
`lognorm` and so on) has `loc` fixed at 0 by default, because amplitudes and
powers start at zero. Fitting `loc` as well is up to 100 times slower and much
less stable on small samples. Pass `fixed={}` to fit everything. Samples
outside the support (a negative amplitude) are ignored, with a warning. A
sample exactly at the lower end counts where scipy counts it: zeros are valid
for `expon` and `weibull_min`, while `rayleigh`, `gamma` and `lognorm` have no
likelihood at 0 and leave such samples out, with a warning.

**Exact running fits.** For `norm`, `expon`, `rayleigh`, `lognorm` and `gamma`
(`stats.EXACT_FITS`, with the default `fixed`), the fit depends only on a few
running sums, which merge across frames like `Moments`. These fits match
`dist.fit` on every sample, keep no samples, and with the default
`window=None` cover the whole stream so far.

**Windowed fits.** Any other distribution (`weibull_min`, `rice`, ...) is
fitted to the samples of the last `window` frames, and needs `window`. Its fit
takes milliseconds (Weibull about 2.5 ms and Rice about 4 ms on 640 samples),
so `every=k` refits only every `k` frames and repeats the last fit between.

**How much data.** One frame is seldom enough: a gamma shape of 2 fitted to
64 samples lands between 1.6 and 2.7 nine times in ten; over 20 frames (1280
samples), between 1.9 and 2.1.

**Is the fit any good?** `gof_every=m` tests the fit every `m` refits with
`scipy.stats.goodness_of_fit`, tags `poor_fit` while the p-value (the
`fit_pvalue` annotation) is below `alpha`, and needs `window`. A plain KS test
against the fitted distribution would be wrong: the parameters came from the
same data, so it almost never rejects (0 rejections in 200 correct fits at
α=5%, where 5% is expected). `goodness_of_fit` corrects for that by
simulation, at tens of milliseconds a test; its smallest p-value is
`1 / (draws + 1)`, so `alpha` must be above that. A test that cannot run (no
fit yet, too few samples, or a failure in scipy) leaves `fit_pvalue` NaN and
adds a warning issue saying why.

## Thresholds from a noise model

`threshold` sets a detection threshold from a fitted noise distribution, for a
false-alarm probability you choose, and tags the samples above it:

```python
g = amplitude >> stats.threshold(dist="rayleigh", pfa=1e-6, window=8)
for f in run(g):
    hits = f.mask("detection")
```

- Each frame is tested against the fit of the frames **before** it, at
  `dist.isf(pfa, *params)`. The first `learn` frames (5) only train the fit.
- **Samples no noise would reach are kept out of the fit**, so a strong
  target does not raise the threshold meant to find it. "No noise would
  reach" is beyond the level noise exceeds with probability `exclude`
  (1e-6), not the detection threshold itself: leaving out every detection
  would also leave out the noise above the threshold, and a fit to the
  truncated noise underestimates it. At `pfa=0.1` that more than doubled the
  false-alarm rate; leaving out only what lies beyond the 1e-6 level moves it
  by under 1%. `exclude=None` keeps every sample.
- Each run of detected samples gets a `detection` tag over its time span,
  with its largest sample as the value, and the frame's `threshold`
  annotation is the level used. Frames without a time axis get a whole-frame
  tag and a sample-level `detection` annotation.

**The false-alarm rate stays where it was set** while the noise follows the
distribution. On Rayleigh noise, `pfa=1e-3` measured 1.01e-3 with the whole
stream fitted, 1.04e-3 with a 4-frame window, and 1.14e-3 with a 1-frame
window of 256 samples (a small fit adds a little). A fixed level does not
adapt: set for the starting noise, it gave 4.9e-2 over a run in which the
noise level rose threefold.

**Rising noise raises the rate.** The fit trails the noise. With the noise
level rising threefold over 100 frames, `pfa=1e-3` measured 1.5e-3 with a
4-frame window and 4.2e-3 with a 20-frame window. Use a window as short as
the fit allows when the noise changes.

**Pick the distribution to suit the data:** Rayleigh for the amplitude of
complex Gaussian noise, exponential for its power, Weibull or lognormal for
clutter with heavier tails. A wrong family gives a wrong rate however well its
parameters fit: test it first with `fit(..., gof_every=)`.

## State, checkpoints and reuse

- **Each run, and each use, has its own state.** Every run makes its own
  copy of each statistics block's state for each place the block is used, so
  two graphs, one graph using a block (or a sub-graph) twice, and two runs of
  one graph at the same time do not share a summary. The block itself never
  changes. `stats.moments.bind(window=5)` presets configuration and keeps
  this.
- **Every run starts afresh.** Running a graph again, or `replay`, starts
  each statistic from nothing, unless the run resumes from a checkpoint.
- **Checkpoints capture the state.** These are `Stateful` blocks, so
  `Run.checkpoint()` saves their summaries, and `run(g, resume=ck)` (or
  `iterate`) continues them.
- **Name a second use.** Checkpoints match states to blocks by name, so in a
  graph with two `moments`, name one:
  `stats.moments.node(window=5).named("recent")`.
- **Not in `parallel()`.** Each worker would keep part of the state, so
  `parallel(stats.moments)` is refused when the graph is built, whatever the
  backend. `per_key` gives each key its own
  copy, which is usually what is wanted: `per_key("channel", stats.moments)`.
- **Replay.** A cumulative statistic depends on every frame before it, so
  `replay` of one output reruns the graph from the start up to it.

Your own `Stateful` blocks can opt in to the same behaviour: define a
`reset()` method, which every run calls before it starts (and before a
checkpoint is restored). Without one, a block's state lives on from one run to
the next.
