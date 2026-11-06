# Time, units and slicing

## Time

Sources stamp frames with `t0`; the stream carries `rate`. `dsp.chunk`,
`dsp.overlap`, `dsp.buffer`, `dsp.resample` maintain exact stamps from sample
counts. `frame.time` is a `TimeSpan`; `frame.at(t)`, `frame.between(t0, t1)`.
`align(a, b, tolerance=...)` joins streams by time, recording each pair's
`skew`; sensors must stamp from one clock for that to mean anything (example 06).

## Units

The fourth slot of `Sig` is a unit string. `sigflow.units` knows SI prefixes on
the usual symbols (V, A, W, J, N, C, Pa, ohm, Hz, s, min, h, m, g, rad, deg,
the dB family …; `pint` for any other symbol when installed). Different units
are rejected at build time; convertible ones come with the fix in the message —
`dsp.to_units("V")` converts the data, the `units` annotation and the type.

**Compound units** are products, quotients and powers of those symbols:
`V*s`, `m/s^2` (or `m/s**2`, `m·s^-2`), `kg*m^2/s^2`, `N m`, `V/Hz^0.5`.
`*`, `·` and `.` multiply, `/` divides from left to right (`a/b*c` is
`(a/b)*c`), `^` or `**` raise to a power, and parentheses group. Each unit
stands for powers of mass, length, time, current and temperature, so units
of the same quantity are convertible however they are written:

```python
from sigflow import units

units.parse("W*s")  # ("energy", 1.0), as for "J" and "N*m"
units.factor("km/h", "m/s")  # 0.2777...
units.compatible("Hz", "1/s")  # True
```

Angles, bits, bytes and symbols are quantities of their own (`rad` is not
dimensionless here, and `b` does not convert to `B`). Levels in dB do not
combine with other units.

`units.product`, `units.quotient` and `units.simplify` build the units of a
computed quantity. `simplify` combines repeated symbols and names a compound
that is exactly a joule, watt, coulomb or newton, without changing the value:

```python
units.simplify(units.product("W", "s"))  # "J"
units.simplify(units.product("m/s^2", "s"))  # "m/s"
units.simplify(units.product("V", "s"))  # "V*s"
```

A contract compares the unit strings, so `W*s` produced where `J` is required
is still a mismatch, convertible with `dsp.to_units("J")` at a factor of 1.

## Slicing

On array-typed nodes `[]` slices data with the shape tracked statically and
sample annotations and `t0` sliced in step: `fft[:512]`, `cube[:, 20:200]`,
`iq[::4]`; on frames `frame[10:20]`. Explicit bounds give concrete sizes,
symbolic ones become new symbols. Tuple-typed nodes keep port selection.
Stream slicing is by frame count: `take`, `skip`, `every`, `until`, `while_`.
