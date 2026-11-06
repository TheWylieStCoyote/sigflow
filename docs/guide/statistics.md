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

## State, checkpoints and reuse

- **Each use has its own state.** Two graphs using `stats.moments`, or one
  graph using it twice, do not share a summary.
- **Every run starts afresh.** Running a graph again, or `replay`, starts
  each statistic from nothing, unless the run resumes from a checkpoint.
- **Checkpoints capture the state.** These are `Stateful` blocks, so
  `Run.checkpoint()` saves their summaries, and `run(g, resume=ck)` (or
  `iterate`) continues them.
- **Name a second use.** Checkpoints match states to blocks by name, so in a
  graph with two `moments`, name one:
  `stats.moments.node(window=5).named("recent")`.
- **Not in `parallel()`.** Each worker would keep part of the state, so
  `parallel(stats.moments)` is refused. `per_key` gives each key its own
  copy, which is usually what is wanted: `per_key("channel", stats.moments)`.
- **Replay.** A cumulative statistic depends on every frame before it, so
  `replay` of one output reruns the graph from the start up to it.

Your own `Stateful` blocks can opt in to the same behaviour: define a
`reset()` method, which every run calls before it starts (and before a
checkpoint is restored). Without one, a block's state lives on from one run to
the next.
