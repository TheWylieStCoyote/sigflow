# Blocks

A block is an ordinary Python function with a typed signature:

```python
@block
def scale(x: Sig[T, N], gain: float = 1.0) -> Sig[T, N]:
    return x * gain
```

Parameters annotated with a signal type (`Sig[...]`, a tuple of them, `Frame`,
`Framed[...]`, `Stream[...]`) are **inputs**; every other parameter is
**configuration**, fixed when the block is placed in a graph (`scale(gain=2)`).
The return annotation is the output **contract**. The decorated object is still a
function — `scale(np.ones(4), gain=2)` runs it — so blocks are testable and
reusable on their own.

## Three kinds, inferred

- **map** blocks are called once per frame and should be pure.
- **stream** blocks are generator functions (sync or async) that receive the
  whole stream of items and may keep state: windows, filters with memory,
  decimators, trackers. A generator that yields several items for one input, or
  none, changes the cardinality of the stream — and the runtime records exactly
  which input frames each output consumed.
- **sources** have no signal inputs and yield the stream. `@source(threaded=True)`
  runs a _blocking_ source (hardware reads, wall-clock pacing) in a worker thread.

## Reusing existing code

```python
fft   = block(np.fft.fft, inputs=Sig[complex, N], outputs=Sig[complex, N])
eigh  = block(np.linalg.eigh, inputs=Sig[T, tuple[N, N]],
              outputs={"vals": Sig[float, N], "vecs": Sig[T, tuple[N, N]]})   # named ports
anything = lift(some_function)                     # Any -> Any, checked at run time
```

## Options worth knowing

`on_error="propagate" | "raise" | "drop"`, `keep_samples=`, `refine=lambda n, **_:
Sig[float, (n,)]` (derive concrete dimensions from configuration, so they are
checked before running), `total=lambda frames, **_: frames` (expected frame
count, for progress), `threaded=True`. A `Knob` passed as configuration makes a
map block re-read it on every call, which is how notebook sliders tune a running
graph. A block written as a `Stateful` object (callable with `state()` /
`restore()`) can be checkpointed.

## Positional arguments

Calling a block definition with an array, Frame, list or iterator calls the
function; with strings, numbers or other scalars it configures a node
(`to_units("V")`, `io.wav("x.wav", n=512)`). `block.fn(...)` always calls the
function.
