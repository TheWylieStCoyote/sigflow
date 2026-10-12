# sigflow

A small Python framework for building signal-processing applications out of
typed, pure functions, composed into graphs with operators and executed lazily
with generators. It depends only on NumPy, SciPy and matplotlib.

```python
import numpy as np
from sigflow import block, Sig, Stream, run
from sigflow import dsp, plot


@block
def scale(x: Sig["T", "N"], gain: float = 1.0) -> Sig["T", "N"]:
    return x * gain


@block
def pairs(
    xs: Stream[Sig[float, "N"]],
) -> Sig[float, (2, "N")]:  # a stateful stream block
    buf = []
    for x in xs:
        buf.append(x)
        if len(buf) == 2:
            yield np.stack(buf)
            buf = []


spec = plot.spectrum(title="1 kHz tone")
g = (
    dsp.sine(freq=1000, rate=48_000, n=1024, frames=50)
    >> scale(gain=0.5)
    >> spec
    >> dsp.to_complex
    >> dsp.fft
)

frames = run(
    g, progress=True
)  # or: await g.arun(),  for f in g: ...,  async for f in g: ...
frames[-1].trace()  # full lineage back to the source frame
spec.save("tone.gif")  # the plot remembered every frame -> animation
```

```
python -m pip install -e .          # from this directory, with the Python you will use
python -m pytest                    # 68 tests (also work uninstalled: tests/conftest.py adds src/)
python examples/01_basics.py
jupyter notebook examples/sigflow_tour.ipynb
```

If the notebook reports `ModuleNotFoundError: sigflow`, its kernel is a different
Python than the one you installed into. Check with `import sys; sys.executable`
in a cell, then either install with that interpreter
(`/path/to/that/python -m pip install -e .`) or register the Python you
installed into as a kernel: `python -m ipykernel install --user --name sigflow`
and select it from the Kernel menu. The tour notebook also falls back to the
`src/` folder next to it, so it runs from the unpacked archive either way.

## What is in the box

| module             | contents |
|--------------------|----------|
| `sigflow.types`    | `Sig[dtype, shape, rate, units]` signal types, symbolic dimensions, dtype variables, unification |
| `sigflow.frame`    | immutable `Frame`: data + three annotation levels + issues + provenance |
| `sigflow.block`    | `@block`, `@source`, `@tap`, `lift` — turn functions (yours or NumPy/SciPy's) into blocks |
| `sigflow.graph`    | operators `>>`, `\|`, `&`, `merge()`; compile-time validation |
| `sigflow.runtime`  | asyncio-native execution with transparent adapters for sync generators |
| `sigflow.progress` | `Monitor`: per-block counters, percent complete, text/Jupyter display |
| `sigflow.control`  | `g.start()` background runs: pause / step / stop / throttle / record, `Knob` live parameters |
| `sigflow.ui`       | ipywidgets control panels (`run.controls()`, `dashboard()`), knob widgets |
| `sigflow.io`       | WAV / raw (GNU Radio) / npy / CSV / UDP sources and sinks |
| `sigflow.builder`  | `Graph`: explicit DAG builder (add / connect / output) |
| `sigflow.units`    | SI-prefixed unit registry; compatibility and conversion |
| `sigflow.persist`  | provenance JSON, frame save/load, `replay()`, `Stateful`, checkpoints |
| `sigflow.dsp`      | sources (sine, noise, chirp, async ticker…), reframing, FFT, filters, taps |
| `sigflow.linalg`   | shape-checked matmul / solve / eig / svd / covariance … |
| `sigflow.plot`     | line / spectrum / waterfall / constellation taps, animations, live notebook plots |
| `sigflow.draw`     | `g.draw()` SVG diagrams and `g.to_dot()` Graphviz export |
| `sigflow.notebook` | HTML representations for graphs, frames and monitors |

## Design notes

### Blocks are functions, and stay functions

The unit of work is an ordinary Python function. `@block` reads its signature:
parameters annotated with a signal type (`Sig[...]`, a tuple of them, `Frame`,
`Framed[...]` or `Stream[...]`) are inputs, every other parameter is
configuration. The return annotation is the output type. Nothing else is
required, and the function remains directly callable (`scale(x, gain=2)`), so
blocks are unit-testable and reusable outside any graph.

Three kinds are inferred, never declared:

* **map** blocks are called once per frame and must be pure.
* **stream** blocks are generator functions (sync or async) that receive the
  whole stream and may hold state — windows, filters with memory, decimators.
* **sources** have no signal inputs and yield the stream.

Existing library code is wrapped, not rewritten:
`fft = block(np.fft.fft, inputs=Sig[complex, "N"], outputs=Sig[complex, "N"])`.
`lift(fn)` wraps anything as `Any -> Any` when you do not care about types yet.

Calling a block definition with keyword configuration (`scale(gain=2)`) creates
a graph node; the bare name uses the defaults. Configuration is validated
against the signature (unknown or missing parameters fail at build time).

### Types are explicit, and checked before anything runs

`Sig[dtype, shape, rate, units]` describes an edge. `dtype` is a NumPy dtype, a
Python class (`Sig[dict]`) or `Any`; `shape` mixes integers, `None` (any
length) and **symbolic dimensions** (`"N"`); `"T"`-style single letters in the
dtype slot are **dtype variables**, so `Sig["T", "N"] -> Sig["T", "N"]` says
"same dtype and length in as out" without committing to either.

Symbols are scoped to the block that declares them — `N` in `fft` is unrelated
to `N` in `scale` — and become equal only where edges meet. Validation is a
union-find unification run while the graph is built, so

```python
(dsp.sine(n=64) & dsp.sine(n=128)) >> linalg.dot
# TypeMismatch: (sine & sine) -> dot[1] axis 0: dimension 'N' is 64 but here needs 128
```

fails before a single frame exists. Blocks can derive concrete shapes from their
configuration with `refine=lambda n, **_: Sig[float, (n,)]`, which is how the
example above knows the sizes. dtypes are compatible when NumPy can cast them
safely (`float64` may feed `complex128`, not the other way round); `Any` is
gradual — it is accepted at build time and verified against the first real
frame at run time (`check="all"` verifies every frame, `"none"` disables it).
A block whose output does not match its declaration raises a `TypeMismatch`
tagged `runtime check failed`.

### Frames carry annotations at three levels, and errors travel the same way

A `Frame` is frozen; every transformation makes a new one. Alongside `data` it
holds `stream` annotations (facts about the whole stream: `rate`, `units`,
`source`), `meta` (facts about this frame: timestamp, sequence number, SNR) and
`samples` (arrays aligned with the leading axis of `data`, such as a `clipped`
mask). The runtime merges and forwards all three automatically; sample-level
annotations are kept only through blocks whose declared output keeps the
leading dimension (or that say `keep_samples=True`), because they would be
meaningless otherwise.

Errors are first-class data, not control flow. When a map block raises, the
runtime attaches an error `Issue` (with the exception and the provenance node
where it happened) and emits a frame whose data is `MISSING`. Downstream map
blocks pass such frames through without computing (a frame-aware single-input
map block, `Framed[...]`, does see them and can inspect `f.failed`; multi-input
blocks never compute on a tuple with a failed part); stream blocks never
see them, because a generator that raised cannot be resumed; sinks receive them with their full history, so a failure at the end of a
long pipeline can be traced to the exact source frame that caused it. Warnings
(`frame.with_issue(..., "warning")`) travel the same way without stopping
computation. The policy is per block or per run: `on_error="propagate"`
(default), `"raise"` or `"drop"`. Returning `None` from a map block drops a
frame, which is how filters are written.

### Everything is traceable

Each frame has a `Provenance` node: the block name, its configuration, the
sequence number within that block's output, and the parent nodes. Map blocks
record one parent; zip and fan-out record one per input; stream blocks record
exactly the frames consumed since their previous output (so a window of five
frames lists five parents). `frame.trace()` renders the tree, `frame.lineage()`
walks it, `frame.sources()` returns the root frames. Provenance is a linked
structure shared between frames, so it costs one small object per computation.

### Loops, routing and applying graphs to data

Two combinators go beyond feed-forward chains. `feedback(body, init)` closes a
loop: `body` is any 1:1 node mapping a tuple frame `(x, state) -> (y, new_state)`
and the runtime feeds `new_state` back for the next frame, so an AGC is
`feedback(agc_step, init=0.1)` and an LMS canceller is
`feedback(split >> lms_filter >> lms_update, init=zeros)` — the loop spans
blocks, and the loop state shows up in provenance. `when(pred, then,
otherwise=...)` routes each frame by a predicate on the *frame* (so it can look
at `f.failed`, `f.warnings` or annotations) through one of two 1:1 branches,
preserving order; a branch that returns `None` drops the frame. Both keep the
rest of the graph type-checked.

A source-less graph is also a function: `front_end(x)`, `front_end([x1, x2])`
or `front_end(frames)` run it over in-memory data and return frames (lineage
starts at a node called `input`); `block.apply(frames)` does the same for one
block when you want annotations kept rather than a plain function call.

### An explicit graph when operators are not enough

`Graph` builds any DAG: add nodes, connect them (by position or by input name,
selecting ports on the way), declare outputs. Edges are type-checked as they
are made, cycles are rejected (use `feedback()`), a producer may feed any
number of consumers, and multi-input members receive their inputs zipped.

```python
g = Graph("diamond")
cov = g.add(src >> linalg.covariance, "cov")
e = g.add(eigh)  # named ports vals / vecs
s = g.add(summarize)  # takes (vals, vecs, raw)
g.connect(cov, e).connect(e["vals"], s, "vals").connect(e["vecs"], s, "vecs").connect(
    cov, s, "raw"
)
g.output(s)  # several outputs -> a tuple
```

A graph with `g.input(type)` instead of sources is a reusable node; `g.draw()`
lays it out in layers with long edges arced over the row, and `g.report()`
documents it.

### Slicing: data and stream

On an array-typed node, `[]` slices the data with the shape tracked statically
and sample-level annotations and time sliced in step: `fft[:512]`,
`cube[:, 20:200]`, `iq[::4]`, `frame[10:20]`. Explicit bounds give concrete
sizes (`Sig[T, (N,)]` sliced `[10:60]` is `(50,)`), symbolic ones become new
symbols. On a tuple-typed node `[]` still selects ports. Stream slicing has its
own words: `g.take(n)`, `g.skip(n)`, `g.every(k)`, `g.until(pred)`,
`g.while_(pred)` — identity-typed, so `g.take(2) >> fft` is checked as before.

### Time as an axis

Every frame with a `t0` annotation and a stream `rate` has `frame.time`, a
`TimeSpan` with `t0`, `t1`, `duration`, `stamps()`, `index(t)`;
`frame.at(t)` and `frame.between(t0, t1)` select by time. Sources stamp their
frames; re-framing blocks (`chunk`, `overlap`, `buffer`, `resample`) maintain
exact `t0` and `rate` from sample counts rather than inheriting them; `dsp.timestamp`
adds `t0` from the sample count when a stream lacks it. `align(a, b,
tolerance=0.02)` joins streams by time instead of by count — the right join for
sensors with different rates and jitter — recording the `skew` of each pair.

### Units

The fourth slot of `Sig` is a unit. `sigflow.units` knows SI prefixes on the
usual bases (V, A, W, Hz, s, m, rad, deg, dB, dBm, dBV, dBFS …; `pint` is used
for anything else when installed). Unification rejects different units; when
they are *convertible* the error says which `dsp.to_units("V")` to insert, and
that block converts the data, the stream annotation and the type.

### Saving, replaying, checkpointing

`frame.prov.to_json()` / `Provenance.from_json()` persist a lineage (shared
ancestors kept); `frame.save("f.npz")` / `Frame.load()` persist a frame with
its annotations, issues and lineage. `replay(graph, frame)` recomputes exactly
that output from the same graph — it checks that the sources are configured as
recorded (`ReplayMismatch` otherwise) and returns the frame whose lineage ends
in the same source frames. For long runs, a block written as a `Stateful`
object (`state()` / `restore()`, defaulting to its `__dict__`) is
checkpointable: `run.checkpoint("ck.pkl")` records source positions and block
states, and `run(g, resume="ck.pkl")` / `g.start(resume=...)` continues from
there (generator stream blocks cannot be captured and are listed as `skipped`).

### Selecting ports, annotations and ancestors with `[]`

A node with a tuple output — a fan-out, a zip, or a block that returns several
signals — exposes **output ports**, selected with the subscript operator:

```python
cov >> linalg.eigh[0] >> plot.line()  # eigenvalues only
src >> (fft >> power_db | rms | peak)["rms"]  # a fan-out branch by name
src >> (fft >> power_db | rms | peak)[0, 2]  # a tuple of two ports; [1:] also works
eigh = block(
    np.linalg.eigh,
    inputs=Sig["T", ("N", "N")],
    outputs={"vals": Sig[float, "N"], "vecs": Sig["T", ("N", "N")]},
)
mats >> eigh["vecs"]  # named ports on a wrapped library function
```

When several ports of one producer must reach *different* consumers, fan out
port selectors — `port(key)` is a selection that resolves to a typed `Select`
once it is placed after a tuple-typed node, including inside a branch:

```python
(
    eigh >> (port("vecs") | port("vals") >> model_order) >> noise_subspace
)  # (vecs, order) -> block
(a & b & c) >> (port(0) >> validate | port(1) >> validate | port(2) >> validate)
```

Branch ports are named after their blocks when the names are unique (set a
label with `node.named("x")` otherwise); the selected port keeps its own type,
so `eigh[0] >> transpose` is rejected at build time, and its provenance records
the selection (`[vals]`) above the producing block. A single-output node has
no ports and `[]` raises, which keeps the door open for data slicing later.

The same operator reads frames and lineage:

```python
f["rate"]  # annotation lookup: frame-level, then stream-level, then sample-level
f.get("snr", None)  # with a default;  "rate" in f
f[1]  # part 1 of a tuple frame (from a fan-out / zip), with its own provenance
f.prov[0]  # this node;  f.prov[2] two steps upstream;  f.prov[-1] the root
f.prov[
    "fft"
]  # nearest ancestor produced by a block called fft (its params, seq, parents)
f.prov[1:3]  # nodes along the main line of descent;  "fft" in f.prov
```

### Functional by construction — graphs connect to graphs

Graphs are immutable values built from operators, and a graph is a node like
any other, so graphs connect exactly the way blocks do:

```python
front = dsp.window >> dsp.to_complex >> dsp.fft  # no source: a reusable sub-graph
back = dsp.power_db >> dsp.rms

dsp.sine(n=256) >> front >> back  # chain graphs
(rx >> front) & (ref >> front) >> compare  # zip two graphs
src >> (front | front >> back)  # a graph as a fan-out branch
from_iterable(g1, type=g1.out_type) >> g2  # feed one run's output into another
```

The join is type-checked like any edge. A sub-graph may be used several times
in one graph, with different sizes on each use: leaves that appear twice are
cloned with fresh symbol scopes, and the monitor shows them as `fft`, `fft#2`.
Stateful nodes (`plot.*`, `collect()`) keep sharing their state when reused.
`a >> b` chains, `a | b` fans out (every frame goes to each branch, the result
is a tuple frame), `a & b` zips streams in lock step, `merge(a, b)` interleaves
sources by arrival. A block with several signal parameters consumes a tuple
frame, so `(x & y) >> add` and `src >> (f | g) >> combine` read like the data
flow they are.

### Synchronous, asynchronous, and both at once

The runtime is asyncio-native: every node compiles to an async transformation
of frame streams, which is what lets a network ticker and a sampled signal live
in one graph. Plain generators, iterables and functions are adapted
automatically; a synchronous generator stream block runs in a worker thread
behind bounded queues so it cannot stall async sources. From sync code, `run(g)`
works everywhere (including inside a Jupyter kernel, where it uses a helper
thread); from async code, `await g.arun()` or `async for f in g`.

### Seeing the graph

`g.draw()` returns an SVG diagram that displays inline in Jupyter (and
`.save("g.svg")` writes it): pipelines flow left to right, fan-out / zip / merge
stack vertically and converge on a join node, boxes are coloured by kind
(source, map, stream, tap/plot), show the non-default configuration and the
output type with dimensions resolved, and carry the full signature as a hover
tooltip. `g.to_dot()` gives Graphviz source for `dot` or the `graphviz`
package. `g` on its own still renders the typed node table.

### Progress and status

`run(g, progress=True)` shows a live bar in the terminal or an updating HTML
table in Jupyter. `g.last_monitor` (or a `Monitor` you pass in) exposes per-block
in/out/error counts and time, `percent` (when a source declares its length via
`total=lambda frames, **_: frames` or is a sized iterable) and `elapsed`. Graphs,
frames and monitors all have HTML representations, so evaluating one in a cell
shows the typed node table, the annotation/provenance card, or the status table.

### Controlling long-running graphs

`g.start()` runs a graph in the background and returns a `Run` handle:
`run.pause()`, `run.resume()`, `run.step(n)` (let *n* source frames through,
then stay paused), `run.stop()`, `run.wait()`; `run.frames` holds the outputs so
far and `run.state` is `running`, `paused`, `stopped`, `done` or `failed`.
`g.start(paused=True)` arms a graph without releasing any data. In a notebook,
`run.controls()` shows a panel: Pause / Resume / Step (with a count) / Stop,
Snapshot and Record buttons, a throttle slider (max source frames per second,
also `run.set_rate(fps)`), a progress bar, one widget per `Knob`, live figures
and the status table; `dashboard({...})` stacks panels for several runs with
Pause-all / Resume-all / Stop-all. The panel needs **ipywidgets in the kernel**
(`python -m pip install ipywidgets`, then restart the kernel); `require_widgets()`
tells you whether it will work, and without it `controls()` shows a status line
plus the same instructions.
Pausing gates the sources, so frames already in flight drain through the graph
and plots and sinks settle before everything idles.

Three things make long-running graphs practical in a notebook. `Knob(value)`
is a tunable parameter: a map block declared with one (`scale(gain=Knob(1.0))`)
reads its value on every call, sources and stream blocks receive the Knob and
read `.value` themselves, and `knob.widget()` (or `controls(knobs=[...])`)
binds it to the natural ipywidgets control — a slider for numbers, a toggle for
booleans, a dropdown for `Knob("hann", choices=[...])`, a text box otherwise —
so the graph is tuned while it runs. Plot options (`ylim`, `lim`, `db`, `rows`)
accept Knobs too, `knob.on_change(fn)` chains knobs, and provenance records the
values actually used on each call.
`controls(plots=[...])` refreshes plot figures as images inside the panel —
the way to watch a plot from a background run, since a thread cannot draw
into the notebook itself. `start(keep=N)` bounds the frame buffer for endless
streams and `run.drain()` hands over what arrived since the last call;
`start(sink=fn)` reacts to every frame (and may call `run.stop()`).
`examples/live_control.ipynb` walks through all of it.

### Files and sockets

`sigflow.io` has memory-mapped, chunked sources — `io.wav(path, n)`,
`io.raw(path, np.complex64, n)` for GNU Radio file-sink captures, `io.npy`,
`io.csv_rows` — an async `io.udp(port, dtype)` receiver, and sinks that are taps
(`io.wav_sink`, `io.npy_sink`, `io.csv_log`, `io.udp_sink`; context managers
that flush on close), so a file can be written from the middle of a chain.
Sources take positional configuration (`io.wav("x.wav")` makes a node).

### Documenting a graph

`g.report()` returns a self-contained HTML page (diagram, typed table with
resolved dimensions and configuration, each block's docstring); `fmt="md"`
gives Markdown. Together with `g.draw().save()` and `g.to_dot()` that is enough
for a small generated docs site (example 19).

### Plotting and animation

Beyond `line`, `spectrum`, `waterfall` and `constellation`: `plot.history(key)`
draws a scalar (the frame's data, an annotation, or `key(frame)`) against time;
`plot.multi([...], cols=2)` tiles several plot blocks into one figure that
animates, saves and sits in a control panel as one unit. With the `ipympl`
backend (`%matplotlib widget`) `live=True` updates the interactive canvas in
place, so zoom and pan keep working while the graph runs.

Plot blocks are taps built on matplotlib. They draw each frame they see, keep
the frames, and expose `figure`, `animate()` (a `FuncAnimation`), `save(path)`
(GIF via Pillow, MP4 via ffmpeg) and `html()` (a JavaScript player for
notebooks). `live=True` updates the figure in the notebook while the graph
runs. `plot.custom(draw)` takes a `draw(ax, data, frame)` function for anything
the built-in line, spectrum, waterfall and constellation plots do not cover, and
`plot.animate(frames)` animates any list of frames after the fact.

## Performance

`python bench/bench.py` measures per-frame overhead and writes
`docs/performance.md`. On this machine: ~8 µs per source frame, ~10 µs per map
block, ~20 µs for an async generator stream block, ~260 µs for a *sync*
generator stream block (it runs in a worker thread with a queue hop each way —
prefer `async def` stream blocks on hot paths), ~15 µs for a plain 1024-point
NumPy FFT for scale. The runtime suits frames of hundreds to millions of
samples; it is not for sample-by-sample streaming. `run(..., strict=True)`
raises when fan-out / zip branches produce different frame counts instead of
silently stopping at the shortest.

## Conventions worth knowing

* Config lambdas (`refine=`, `total=`) receive the full configuration as keyword
  arguments; accept `**_` for the ones you do not use.
* Fan-out and zip stop at the shortest branch; if branches change cardinality
  (a window on one side), pair them deliberately.
* A source that blocks (hardware reads, `time.sleep` pacing) should be declared
  `@source(threaded=True)` so it cannot stall async sources beside it.
* A stream block that yields several outputs from one input gives them all the
  same parents (provenance and annotations are inherited); a source that raises
  mid-stream ends its stream with a failed frame in its own name.
* `Frame.evolve(meta=...)` merges annotation dicts; `Frame.without(*keys)`
  removes keys.
* A tap sees the frame (`inputs=Frame`) and passes it on unchanged; a map block
  annotated `Framed[...]` can return an evolved frame to control annotations and
  issues precisely.

## Examples

`examples/` holds twenty runnable scripts and two notebooks (see
`examples/README.md` for the index): signal chains (QPSK modem, pulse-Doppler
radar, IMU/GPS fusion, audio activity, subspace direction finding, a polyphase
channelizer), framework properties (ports and lineage, feedback loops, routing
and recovery, real file/socket I/O, parameter sweeps in processes, 2-D video
frames), and workflow (testing pipelines, analysing recordings interactively,
generating documentation, controlling long runs from a notebook), and one
that ties the 0.2 features together (units, slicing, time, Graph, replay,
checkpoints).

## Status and next steps

This is version 0.2 (see `CHANGELOG.md`) (68 tests, twenty example scripts, a tour
notebook). Natural extensions: a process-based executor inside one graph (sweeps already
use processes per configuration), inline execution of sync generator stream
blocks (removing the thread hop), a provenance depth cap for very long runs,
identity-typed taps so `Any` stops leaking through fan-outs, a `Sig` that
static type checkers can read (block bodies are not type-checked today), and a
richer set of scipy.signal wrappers.
