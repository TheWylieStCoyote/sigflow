# Integration across frames

`sigflow.integrate` integrates a stream over time: velocity from
acceleration, charge from current, energy from power. A stream arrives in
frames, but the integral runs across them, so these blocks keep the last
samples of the stream and give what integrating the whole signal at once
would give.

```python
from sigflow import run
from sigflow import integrate as ig

velocity = run(acceleration >> ig.cumulative)  # the integral at every sample
energy = run(power >> ig.definite)  # one value per frame
```

## Why not integrate each frame?

Integrating each frame on its own restarts the integral every frame. Carrying
the running total over looks right, but still loses the interval between the
last sample of one frame and the first of the next: one sample period per
frame, and the error grows for as long as the stream runs. Over 50 frames of
64 samples of a 1000 Hz signal, against integrating the whole signal:

| approach | largest error |
|---|---|
| integrate each frame on its own | 1.56 (the final value 0.05 instead of 1.61) |
| carry the total, miss the interval between frames | 2.4e-2, and growing |
| `integrate.cumulative` | 3e-15 |

The same holds for per-frame totals: summing `scipy.integrate.trapezoid` of
each frame missed 49 intervals. The values of `integrate.definite` add up to
the integral of the whole stream.

## The running integral

`cumulative` has the input's shape and sample grid. Sample `i` of the output
is the integral from the start of the stream (where it is `initial`, 0 by
default) to sample `i`.

- `method="trapezoid"` (default) is exactly what
  `scipy.integrate.cumulative_trapezoid` gives on the whole signal.
- `method="simpson"` uses `scipy.integrate.cumulative_simpson`, which is
  far more accurate for smooth, well-sampled signals (3e-9 against 4e-6 for a
  7 Hz tone sampled at 1000 Hz). Scipy's value at a sample also uses the
  sample after it, which a stream does not have at the end of a frame, so the
  streamed result differs slightly: in the test above its error against the
  true integral was 7e-9.

## Drift

An integral of a real sensor drifts. Noise makes it wander, and a constant
bias makes it grow without bound: a still accelerometer with noise of 0.01
and a bias of 0.001 m/s², integrated twice for a minute, puts the position
1.83 m away, 1.80 m of it from the bias alone. Integration is exact
arithmetic and still gives nonsense on a real sensor without one of these:

- **A leak.** `cumulative(leak=tau)` integrates `y' = x - y / tau`: it
  forgets what happened more than a few `tau` seconds ago, follows changes
  faster than `tau`, and a constant bias settles at `bias * tau` instead of
  growing.
- **Resets from what else is known.** `cumulative(reset_on="still")` holds
  the integral at zero on the samples a `still` tag covers (the whole frame,
  or a span of it) and integrates afresh after them. These are zero-velocity
  updates, when another block knows the sensor is at rest.
- **Remove the bias first**, if it is known or can be estimated:
  `dsp.offset(value=-bias)`, or a high-pass filter.

## Holes in the time axis

Frame start times (`t0`) are used when present, so a frame that starts late is
a hole:

- `gaps="bridge"` (default) integrates across it as a straight line from the
  sample before to the sample after, and adds a warning issue saying how long
  the hole was;
- `gaps="reset"` restarts the integral after it, and tags that frame
  `integral_reset`.

Frames that overlap in time are treated as following on, with a warning. To
fill a hole with an interpolated signal instead, use
`interpolate.fill_gaps` before integrating, where that module is available.

## Per-frame integrals

`definite` gives one value per frame: the integral over the frame's samples
*and* the interval before its first sample, so the values add up to the
integral of the stream. `definite(window=k)` gives the integral over the last
`k` frames instead. The output drops the `rate` annotation, since it is one
value per frame, not samples.

## Units

The integral of a quantity is in its units times seconds, and the output's
`units` annotation says so, as simply as it can: `W` gives `J`, `A` gives
`C`, `m/s^2` gives `m/s`, `m/s` gives `m`, `V` gives `V*s` (see
[compound units](time-units-slicing.md#units)). It converts like any other
unit: `cumulative >> dsp.to_units("kJ")` on a power in `W`.

The output *contract* leaves the unit open, because a block's contract cannot
depend on its input's: downstream blocks that require `V` are not stopped by
the type checker on an integral of volts. The annotation is right; check it, or
convert with `to_units`, which reads it.

## Limitations

- **One channel.** The contracts are for one-dimensional frames. Integrate
  several channels with one block each, for now.
- **Checkpoints.** The blocks keep their state in a generator, which a
  checkpoint cannot save: they are listed in `Checkpoint.skipped` and start
  from zero on resume.
- **Failed frames** pass around stream blocks, so their time becomes a hole.
- **Models driven by the stream** (ordinary differential equations solved
  with `scipy.integrate.solve_ivp`) are not here yet. For a linear system,
  `scipy.signal.lfilter` on a stream block is far cheaper.
