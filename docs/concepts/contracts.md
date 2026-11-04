# Contracts

`Sig[dtype, shape, rate, units]` describes what flows along an edge:

```python
Sig[np.complex64, N]  # 1-D complex, symbolic length N
Sig[float, tuple[M, N]]  # M x N real matrix
Sig[
    float, tuple[Literal[1024]], Literal[48_000], Literal["V"]
]  # fixed length, 48 kHz, volts
Sig[T, N]  # any dtype T (preserved in -> out), length N
Sig[dict]  # a Python object
Sig[Any]  # anything (gradual: checked against the first frame)
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

## Tags

A contract can speak about [tags](frames.md#tags), the labels frames carry.
In an annotation the markers go inside `typing.Annotated`, so checkers keep
reading the signature:

```python
from typing import Annotated
from sigflow import clears, provides, requires


@block
def vad(f: Framed[Sig[float, N]]) -> Annotated[Sig[float, N], provides("speech")]:
    return f.tag("speech") if active(f.data) else f


@block
def asr(
    x: Annotated[Sig[float, N], requires("speech")],
) -> Annotated[Sig[str, ()], clears("speech")]: ...
```

- `requires(name)` on an input: frames must carry the tag. It is checked when
  the graph is built, against what the producer *guarantees*, and at run time
  with the other checks.
- `accepts(name)` on an input: the block reads the tag when it is there. It
  is never required, so a producer that does not provide it is fine.
- `provides(name)` on an output: the block attaches the tag. The promise is
  checked at run time like the output's dtype and shape.
- `may_provide(name)` on an output: the block sometimes attaches the tag. It
  guarantees nothing, but it says what downstream blocks may look for.
- `clears(name)` on an output: the tag is not carried on; the runtime drops
  it from the block's output frames.

Each of the first four also takes **value types** by keyword:

```python
@block
def demod(
    x: Annotated[Sig[complex, N], requires("synced", track=int), accepts(snr_db=float)],
) -> Annotated[Sig[int, N], provides(bits=int), may_provide(crc_fail=bool)]: ...
```

A type is a class or a tuple of classes, checked like `isinstance`, with
numeric widening (an `int` satisfies `float`) and `Enum` classes matched by
member value, since a tag stores an enum member's value. When the graph is
built, a type the producer declares for a required or accepted tag must fit
the consumer's (`tag 'track' is str, needs int`); a tag guaranteed without a
type is accepted and checked at run time. At run time every typed tag on the
frame is checked, on the input for `requires` and `accepts` and on the output
for `provides` and `may_provide`, whenever it is present. Guaranteed types
travel downstream like guaranteed names, so a block that knows nothing of
`track` still passes its type on.

As values (`inputs=`, `outputs=`, `from_iterable(type=...)`) the same is
spelled on the type object: `SignalType(float, (N,)).provides("speech")`.

Guarantees and requirements are about **whole-frame** tags. A spanned tag is
never guaranteed: a frame need not overlap any span, and blocks that change
the leading axis drop spans. At run time a guarantee is checked on the
producer's output, so a frame that lacks a tag it was declared to carry is
caught before the next block runs.

A tag that a block attaches only sometimes is not a guarantee. Route on it
instead: `when("speech", then=transcribe)` sends the frames carrying the tag
on the whole frame through `transcribe`, which may require it, because the
predicate establishes it for that branch. A block's requirements are checked
once more on the frame it receives, so a frame that reaches it some other way
without the tag is still caught.

Guarantees flow at placement, like dimensions: a block that says nothing about
tags carries its producer's guarantees through, so `vad >> scale >> asr` is
accepted while `sine >> asr` is rejected with `needs tag(s) speech, producer
guarantees none`. This holds through fan-outs, zips, port selection,
`when`/`switch` branches, reused subgraphs and `Graph` wiring. A source
declares what its frames carry with
`from_iterable(..., type=SignalType(float, (N,)).provides("speech"))`.

## Tag propagation

By default a block passes on every tag its input frames carry, minus what it
clears. A block can choose otherwise:

```python
@block(tags="none")                     # pass on nothing it inherited
@block(tags="declared")                 # only tags its input contract requires or accepts
@block(tags=keep("calibrated", "seq"))  # only these
@block(tags=drop("raw"))                # all but these
@block(spans="drop")                    # inherited spanned tags stop here
@block(merge="intersection")            # several input frames: only tags all of them carry
```

`merge` decides how the frames feeding one output combine: the parts of a
tuple input, or the frames a stream block consumed for one output. It is
`"union"` by default, `"intersection"` or `"first"`. The policy shapes what an
output guarantees when the graph is built and what an output frame carries at
run time, in the same way. Tags the block attaches itself are never filtered
by it, and `clears(...)` still applies on top.

One port of a tuple output can have a policy of its own, overriding the
block's for that port:

```python
@block
def split(x: Sig[float, N]) -> tuple[Annotated[Sig[float, N], carries("none")], Sig[float, N]]:
    return x, x
```

`g.describe()` shows a block's policy next to its contract, the diagram's
tooltips show the same, and `run.report()` has a "Tag contracts" table: what
each block requires, accepts, provides, may provide and clears, and what it
passes on.

## Tag warnings

Some wiring is not wrong but cannot do what it says. When the graph is built,
a `TagWarning` names it:

- a block **accepts** a tag that nothing upstream provides or may provide;
- `when("tag", ...)` **routes on** such a tag, so its branch never runs;
- `per_key("tag", ...)` **keys on** such a tag, or on one that is only
  possible, so some or all frames pass through without a key.

`may_provide(...)` travels downstream the way guarantees do, under each
block's policy, so a tag a block sometimes attaches is known to be possible
several blocks later. A producer that says nothing about its tags, such as a
source declared without them, is gradual and draws no warning. Filter the
warning with the `warnings` module like any other.

## Ports

A block returning several signals, a fan-out and a zip have **tuple outputs**
with ports. `node[0]`, `node["vals"]`, `node[0, 2]`, `node[1:]` select them;
`port("vals")` is a selector you can place inside a fan-out so one producer feeds
different consumers. A single-output array node slices data instead:
`fft[:512]`, `cube[:, 20:200]`, with the resulting shape computed statically.
