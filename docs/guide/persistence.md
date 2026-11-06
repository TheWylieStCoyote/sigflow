# Persistence, replay and checkpoints

Three operations share the word "again". They answer different questions:

| operation | answers | needs |
|---|---|---|
| `sigflow.replay(graph, frame)` | *how was this one output computed?* It recomputes exactly that frame. | the graph, configured as when the frame was made; a deterministic graph |
| `run(graph, resume=checkpoint)` | *carry on where a stopped run left off.* | a checkpoint of that graph's sources and stateful blocks |
| `Run.replay(timeline)` | *what if I turn the knobs the same way again?* It replays recorded knob changes on a new run. | a run's `timeline` (or its records) and a graph with knobs of the same names |

`examples/29_checkpoint_resume.py` runs the second end to end and compares it
with an uninterrupted run; `examples/10_lineage_forensics.py` and
`examples/20_builder_time_replay.py` use the first.

## Saving frames and lineage

```python
f.save("frame.npz")  # data, annotations, tags, issues, provenance
g = Frame.load("frame.npz")
text = f.prov.to_json()  # lineage alone, shared ancestors kept shared
prov = Provenance.from_json(text)
```

A labelled `DataArray` payload round-trips with its dims, coordinates and
attributes; a device array is fetched to the host when saved.

## Replaying one output

```python
out = run(g)
again = replay(g, out[42])  # the frame whose lineage ends in the same source frames
```

`replay` runs the graph from the start until it reaches an output descending
from the same source frames, so it costs as much as running up to that point.
Before it starts it checks that each source is configured as recorded in the
frame's provenance; a difference raises `ReplayMismatch` (pass `strict=False`
to skip the check). `provenance="roots"` keeps enough for replay;
`provenance="off"` does not.

**Replay is only as exact as the graph is deterministic.** A source that draws
random numbers must be seeded: `dsp.noise(seed=7)` replays to the same data,
`dsp.noise()` does not, though the check passes because the configuration
matches. The same holds for blocks with hidden randomness. For an **external
recording** (a file read with `io.wav`, `io.raw`, `io.npy`), replay is exact as
long as the file is unchanged; it is not when the source is a live socket or
device, which cannot be re-read. Record such data to a file first (the file
sinks), then analyse the recording.

## Checkpoints and resuming

A block whose function is a `Stateful` object (`state()` / `restore()`, which
default to its `__dict__`) can be checkpointed:

```python
from sigflow import Stateful, block, load_checkpoint, run


class Accumulate(Stateful):
    def __init__(self):
        self.total = 0.0

    def __call__(self, x: Sig[float, N]) -> Sig[float, N]:
        self.total += float(x[0])
        return np.full(2, self.total)


h = make_graph().start(paused=True)
h.step(100)  # pause first for a clean cut
...
h.checkpoint("ck.pkl")  # source positions + block states
h.stop()

rest = run(make_graph(), resume=load_checkpoint("ck.pkl"))  # or resume="ck.pkl"
```

A checkpoint records how many frames each source had emitted and the state of
each stateful block, **by name**. Resuming restores the states and has each
source regenerate the frames it already emitted without passing them on, so
resuming is cheap for file sources and exact for deterministic ones.

What a checkpoint does not capture:

- **Generator stream blocks.** Their state lives in a suspended generator,
  which cannot be saved. They are listed in `checkpoint.skipped` and start
  fresh on resume. Write a block that must survive a resume as a `Stateful`
  map block instead.
- **`per_key` instances.** Each key's instance is a copy made during the run;
  the checkpoint records the template's state, which never changes, so a
  resumed run starts every key's instance fresh. The block is not listed in
  `skipped`.
- **Frames in flight.** Frames between nodes when the checkpoint is taken are
  not saved; pause first so there are none.
- **Anything outside the graph:** file positions in sinks, socket state, the
  random state of an unseeded source.

**Compatibility.** States are matched to blocks by name and positions to
sources by name, so resume into the *same* graph: a fresh instance built by the
same code (a `make_graph()` function). A renamed block's state is not restored.
A checkpoint is a pickle: load only checkpoints you made, with the same version
of the code that made them.

## Recording and replaying knob changes

A background run watches the knobs in its graph's configuration; `h.timeline`
lists every change with the frame number it happened at, and
`h.timeline_records()` gives JSON-ready records to keep. `Run.replay(records)`
applies them to a new run when it reaches the same frame (`by="frame"`) or the
same time (`by="time"`). See [Notebooks and control](notebooks.md).
