# Time, units and slicing

## Time

Sources stamp frames with `t0`; the stream carries `rate`. `dsp.chunk`,
`dsp.overlap`, `dsp.buffer`, `dsp.resample` maintain exact stamps from sample
counts. `frame.time` is a `TimeSpan`; `frame.at(t)`, `frame.between(t0, t1)`.
`align(a, b, tolerance=...)` joins streams by time, recording each pair's
`skew`; sensors must stamp from one clock for that to mean anything (example 06).

## Units

The fourth slot of `Sig` is a unit string. `sigflow.units` knows SI prefixes on
the usual bases (V, A, W, Hz, s, m, rad, deg, dB-family …; `pint` for the rest
when installed). Different units are rejected at build time; convertible ones
come with the fix in the message — `dsp.to_units("V")` converts the data, the
`units` annotation and the type.

## Slicing

On array-typed nodes `[]` slices data with the shape tracked statically and
sample annotations and `t0` sliced in step: `fft[:512]`, `cube[:, 20:200]`,
`iq[::4]`; on frames `frame[10:20]`. Explicit bounds give concrete sizes,
symbolic ones become new symbols. Tuple-typed nodes keep port selection.
Stream slicing is by frame count: `take`, `skip`, `every`, `until`, `while_`.
