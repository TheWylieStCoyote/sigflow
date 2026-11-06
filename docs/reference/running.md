# Running graphs

There are four ways to run a graph. They share one engine and one set of core
options; the table below says which entry point takes which. An option an entry
point does not take is a `TypeError`.

| entry point | returns | use it when |
|---|---|---|
| `run(g, **options)` | `list[Frame]` | the graph ends and you want every output |
| `await arun(g, **options)` | `list[Frame]` | the same, inside a running event loop |
| `for f in g`, `iterate_sync(g, **options)` | an iterator | you consume frames one by one, or only some of them |
| `async for f in g`, `iterate(g, **options)` | an async iterator | the same, inside an event loop |
| `g.start(**options)` | a [`Run`](api/running.md) handle | the graph runs in the background: pause, step, probe, stop |

`run()` works in a notebook cell too: inside a running loop it runs on a private
one. Leaving a `for` loop early, closing the iterator or dropping it stops the
graph and closes its sources.

## Options

| option | values | default | `run` / `arun` | `start` | iterate |
|---|---|---|:-:|:-:|:-:|
| `on_error` | `"propagate"`, `"raise"`, `"drop"` | each block's own | ✓ | ✓ | ✓ |
| `check` | `"first"`, `"all"`, `"none"` | `"first"` | ✓ | ✓ | ✓ |
| `strict` | `True` / `False` | `False` | ✓ | ✓ | ✓ |
| `provenance` | `"full"`, `"roots"`, `"off"` | `"full"` | ✓ | ✓ | ✓ |
| `queue_size` | queue capacity between nodes | `64` | ✓ | ✓ | ✓ |
| `labels` | `True` / `False` | `sigflow.default_labels()` | ✓ | ✓ | ✓ |
| `monitor` | a `Monitor` to fill | a new one | ✓ | ✓ | ✓ |
| `resume` | a `Checkpoint` or its path | none | ✓ | ✓ | ✓ |
| `sink` | `fn(frame)`, may be async | none | ✓ | ✓ | |
| `collect` | `True` / `False` | `True` (`run`) | ✓ | ✓ | |
| `progress` | `True`, `"text"`, `"notebook"` | off | ✓ | ✓ | |
| `total` | expected source frames | unknown | ✓ | ✓ | |
| `paused` | start paused | `False` | | ✓ | |
| `keep` | frames `Run.frames` retains | all | | ✓ | |
| `name` | the run's name in `runs()` | the graph's | | ✓ | |
| `replace` | stop a run of the same name first | `False` | | ✓ | |

What they do:

- **`on_error`** overrides every block's policy for this run. `"propagate"`
  attaches the error to the frame and passes a failed frame on, `"raise"` stops
  the run, `"drop"` drops the frame. See [Robustness](../concepts/blocks.md).
- **`check`** decides how often outputs are checked against their contracts:
  each block's first output, every output, or never.
- **`strict`** turns three silent outcomes into errors: the branches of a
  fan-out or zip produce different numbers of frames (otherwise the join stops
  at the shortest), a `switch` meets a key with no branch and no default
  (otherwise the frame is dropped), and a stream block with an exact declared
  ratio does not keep to it.
- **`provenance`** sets how much lineage frames carry: all of it, only the
  source frames (replay still works), or none (fastest for very long runs).
- **`queue_size`** bounds the frames waiting between two nodes. A slow block
  makes its producers wait, so memory stays bounded however fast a source is.
- **`labels`** decides whether blocks with an unset `labels` option receive
  `xarray.DataArray` payloads. See [Labelled arrays](../guide/xarray.md).
- **`monitor`** is filled with per-block statistics while the run goes;
  `graph.last_monitor` keeps the last one either way.
- **`resume`** restores the stateful blocks from a checkpoint and skips the
  source frames already emitted. See [Persistence](../guide/persistence.md).
- **`sink`** is called with every output frame as it arrives; with
  `collect=False` nothing is kept, which suits endless streams.
- **`keep`** bounds `Run.frames` to the newest *N* frames of a background run.

## The rest of the API

[Running graphs (API)](api/running.md) has the signatures and docstrings of
`run`, `arun`, `iterate`, `iterate_sync`, `Run`, `runs`, `stop_all` and
`Monitor`. See also [Runs](../concepts/runs.md) and
[Streams and their lifecycle](../guide/streaming.md).
