# Contracts

`Sig[dtype, shape, rate, units]` describes what flows along an edge:

```python
Sig[np.complex64, "N"]             # 1-D complex, symbolic length N
Sig[float, ("M", "N")]             # M x N real matrix
Sig[float, (1024,), 48_000, "V"]   # fixed length, 48 kHz, volts
Sig["T", "N"]                      # any dtype T (preserved in -> out), length N
Sig[dict]                          # a Python object
Sig[Any]                           # anything (gradual: checked against the first frame)
```

* **dtype**: a NumPy dtype, a Python class, `Any`, or a single-letter
  *variable* (`"T"`) that binds to whatever arrives and is preserved.
* **shape**: integers, `None` (any length) and **symbolic dimensions** (`"N"`).
  Symbols are scoped to the block that declares them — `N` in `fft` is unrelated
  to `N` in `scale` — and become equal only where edges meet.
* **rate** and **units**: equal or unspecified; units of the same dimension but a
  different scale are rejected with a hint to insert `dsp.to_units(...)`.

## Checked while you build

Every `>>`, `|`, `&`, `connect()` runs a union-find unification. A rank or
dimension conflict, an incompatible dtype, a rate or unit mismatch raises
`TypeMismatch` with both nodes named, before any data exists. Blocks can derive
concrete sizes from their configuration (`refine=`), so
`(sine(n=64) & sine(n=128)) >> dot` fails at build time too.

## Checked at run time, lightly

The first frame crossing every edge is verified against its contract
(`check="first"`, the default); `check="all"` verifies every frame; `"none"`
turns it off. A block whose output does not match its declaration raises a
`TypeMismatch` tagged *runtime check failed*.

## Ports

A block returning several signals, a fan-out and a zip have **tuple outputs**
with ports. `node[0]`, `node["vals"]`, `node[0, 2]`, `node[1:]` select them;
`port("vals")` is a selector you can place inside a fan-out so one producer feeds
different consumers. A single-output array node slices data instead:
`fft[:512]`, `cube[:, 20:200]`, with the resulting shape computed statically.
