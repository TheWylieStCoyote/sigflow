# Interpolation and resampling

`sigflow.interpolate` estimates a signal between its samples: at a new sample
rate, or where samples are missing. Interpolation needs samples on both sides
of the point it estimates, and a stream arrives in frames, so these blocks keep
the last few samples of the stream rather than treating each frame alone.

```python
from sigflow import dsp, run
from sigflow import interpolate as ip

g = dsp.sine(n=441, frames=20, rate=44_100.0) >> ip.resample(rate=48_000.0)
out = run(g)
```

## Why not interpolate each frame?

Interpolating each frame on its own treats its first and last samples as the
ends of the signal, and leaves a seam at every frame boundary.
`dsp.resample` works that way: it is exact in the middle of a frame and wrong
at its edges. Measured against resampling a whole sine of amplitude 1 at
once, up by 3 in frames of 64 samples, the error at the frame edges is 0.62.
`interpolate.resample` gives the whole-signal result whatever the frame
length, down to frames of one sample.

## Choosing a method

`kind=` picks the method; `interpolate.KINDS` lists them.

| kind | what it is | matches the whole signal |
|---|---|---|
| `"linear"` | straight lines between samples | exactly |
| `"pchip"` (default) | a cubic that does not overshoot: monotone data stays monotone | exactly |
| `"akima"`, `"makima"` | cubics that follow the local shape, with little overshoot (`makima` needs SciPy 1.13 or later) | exactly |
| `"cubic"` | a cubic spline with "not-a-knot" ends (scipy's `CubicSpline` default), smoothest | to about 1e-11 |
| `"spline"` | a cubic B-spline through the samples | to about 1e-11 |

The **local** methods (linear, pchip, akima, makima) depend only on a few
samples around each point, so keeping those across a frame boundary gives the
same answer as interpolating the whole signal. The **global** splines depend a
little on every sample. `context=` sets how many samples are kept on each
side: the error falls about tenfold for every two more, and the default of 16
makes it negligible. Output lags input by `context` samples.

For a whole array rather than a stream, `ip.interpolator(t, x, kind)` returns
the same function the blocks use: call it with the times to estimate.

Complex samples are interpolated as their real and imaginary parts. For an
IQ signal whose amplitude varies slowly, interpolating amplitude and
unwrapped phase separately can be more faithful; do that with two branches.

## Resampling

`resample(rate=)` works for any ratio of rates, including irrational ones.
Output samples fall on an exact grid, `first input time + k / rate`, and the
`rate` stream annotation is rewritten.

- **Frame lengths change.** Output frames are not one per input frame, and
  their lengths vary; the output contract says so (any length). Follow with
  `dsp.chunk(n=256)` for frames of a fixed length.
- **The output is floating point.** An integer or `float32` input gives
  `float64`, `complex64` gives `complex128`, and the output contract says so
  (`Sig[promoted(T), ...]`), so a block downstream that takes only integers is
  refused when the graph is built.
- **Interpolation does not filter.** Resampling *down* without a low-pass
  filter first lets content above the new Nyquist frequency alias. Filter
  first (`dsp.fir`), or use `dsp.resample` for integer ratios, where the
  polyphase filter does it.
- **Jittered frames are fine.** Each input frame's `t0` is used when present,
  so frames that start at uneven times are placed where they belong. A frame
  without `t0` follows on from the frame before it.
- **Memory stays bounded.** Only the samples the next output needs are kept,
  so resampling a fast stream to a very low rate does not accumulate input.
- **Holes are bridged.** Where whole samples are missing between frames, the
  output interpolates across them and tags those samples `interpolated` over
  their time span.

## Filling holes

`fill_gaps()` leaves the stream as it is and inserts frames of interpolated
samples where frames are missing: a dropped packet, or a frame that failed
upstream and was dropped. The inserted frames are on the input's sample
grid, as long as the frame before the hole, tagged `interpolated`, and carry
the whole-frame tags the frames on either side share.

```python
from sigflow import block, dsp, run, Sig, N
from sigflow import interpolate as ip


@block(on_error="drop")  # a failed frame leaves a hole instead of passing on
def decode(x: Sig[float, N]) -> Sig[float, N]: ...


g = source >> decode >> ip.fill_gaps(max_gap=0.5)
```

- `max_gap=` (seconds) leaves longer holes alone: past some length, an
  invented signal is worse than none.
- After a hole, the frames that follow are **held back** until there are
  `context` samples after it, so the interpolation sees both sides even when
  frames are shorter than `context`. If the stream ends (or another hole
  comes) before there are enough for `kind`, the hole is left as it is and
  the frame after it carries a warning issue saying so.
- A frame without `t0` follows on from the frame before it. A stream that
  never had times has no holes to find, and passes through.
- Frames that pass through are promoted to floating point, like the
  inserted ones, so the whole output has one dtype.
- Failed frames pass *around* stream blocks, so a frame that failed with the
  default `on_error="propagate"` reaches the output as a failed frame, and its
  time is not filled. Drop it (`on_error="drop"`) to have it filled.

Downstream, `"interpolated" in f.tags` tells invented frames from measured
ones, and `f.mask("interpolated")` marks the invented samples inside a frame
from `resample`.

## Limitations

- **Checkpoints.** The blocks keep their history in a generator, which a
  checkpoint cannot save: they are listed in `Checkpoint.skipped` and start
  fresh on resume, with a seam at the resume point.
- **Sample-level annotations** (`samples=`) are dropped: they belong to the
  old sample grid. Tags are carried over by time, and keep their origin.
- **One channel.** The blocks' contracts are for one-dimensional frames.
  `ip.interpolator` takes channels after the time axis, for whole arrays.
- **One stream.** Interpolating one stream at another's sample times (for
  sensor fusion) needs a join across two streams, which these blocks do not
  do yet; `align()` pairs frames by nearest time instead.
