# Contracts

`Sig[dtype, shape, rate, units]` describes what flows along an edge:

```python
Sig[np.complex64, N]             # 1-D complex, symbolic length N
Sig[float, tuple[M, N]]             # M x N real matrix
Sig[float, tuple[Literal[1024]], Literal[48_000], Literal["V"]]   # fixed length, 48 kHz, volts
Sig[T, N]                      # any dtype T (preserved in -> out), length N
Sig[dict]                          # a Python object
Sig[Any]                           # anything (gradual: checked against the first frame)
```

- **dtype**: a NumPy dtype, a Python class, `Any`, or a single-letter
  _variable_ (`T`) that binds to whatever arrives and is preserved.
- **shape**: integers, `None` (any length) and **symbolic dimensions** (`N`).
  Symbols come from `sigflow.dims` (`from sigflow import N, M`); at runtime they
  are the one-letter strings, so `Sig[float, "N"]` still works, but only the
  symbol spelling type-checks.
  Symbols are scoped to the block that declares them — `N` in `fft` is unrelated
  to `N` in `scale` — and become equal only where edges meet.
- **rate** and **units**: equal or unspecified; units of the same dimension but a
  different scale are rejected with a hint to insert `dsp.to_units(...)`.

## Checked while you build

Every `>>`, `|`, `&`, `connect()` runs a union-find unification. A rank or
dimension conflict, an incompatible dtype, a rate or unit mismatch raises
`TypeMismatch` with both nodes named, before any data exists. Blocks can derive
concrete sizes from their configuration (`refine=`), so
`(sine(n=64) & sine(n=128)) >> dot` fails at build time too.

## Stream rates

Besides _what_ a block emits, its contract says _how many_: the **ratio** of
output frames to input frames. Map blocks are `1:1`; stream blocks declare
`ratio="N:1"` (fewer or equal), `"1:N"`, an exact `(in, out)` pair or a
callable that receives the configuration and the input frame length
(`n_in`), so `dsp.chunk(256)` fed 1024-sample frames is `1:4`. Pipelines
multiply ratios; the branches of a fan-out or zip must agree, or
`RatioMismatch` is raised at build time. `run(strict=True)` verifies declared
ratios against the counts a run actually produced.

## Checked at run time, lightly

The first frame crossing every edge is verified against its contract
(`check="first"`, the default); `check="all"` verifies every frame; `"none"`
turns it off. A block whose output does not match its declaration raises a
`TypeMismatch` tagged _runtime check failed_.

## Ports

A block returning several signals, a fan-out and a zip have **tuple outputs**
with ports. `node[0]`, `node["vals"]`, `node[0, 2]`, `node[1:]` select them;
`port("vals")` is a selector you can place inside a fan-out so one producer feeds
different consumers. A single-output array node slices data instead:
`fft[:512]`, `cube[:, 20:200]`, with the resulting shape computed statically.
