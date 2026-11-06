# Changelog

## Unreleased

New

- `sigflow.interpolate`: interpolation that works across frame boundaries,
  with the methods of `scipy.interpolate` (linear, PCHIP, Akima, modified
  Akima, cubic spline, B-spline). `resample(rate=)` resamples a stream to any
  rate, matching interpolation of the whole signal (exactly for the local
  methods); `fill_gaps()` inserts interpolated frames where frames are
  missing, waiting for samples after a hole when frames are short. Invented
  samples carry an `interpolated` tag; inherited tags keep their origin. The
  output is promoted to floating point and varies in length, as its contract
  says; memory stays bounded at any rate. `makima` needs SciPy 1.13. Guide:
  *Interpolation and resampling*.
- `promoted(T)`: a dtype in an output contract that is the input's promoted to
  floating point (int64 and float32 give float64, complex64 gives
  complex128), for blocks that compute in floating point.
- `sigflow.integrate`: integration across frame boundaries. `cumulative` gives
  the running integral at every sample (trapezoid, exactly as on the whole
  signal, or Simpson), with a leak (`leak=tau`) and tag-driven resets
  (`reset_on=`) against drift; `definite` gives the integral over each frame
  (or the last few), and the values add up to the integral of the stream.
  Holes in the time axis are bridged with a straight line (whatever the
  method) and a warning, or restart the integral; a leak decays over the time
  that actually passes. The output is promoted to floating point, as its
  contract says, and its units are the input's times seconds (`W` gives `J`). Guide:
  *Integration across frames*.
- Compound units: `sigflow.units` reads products, quotients and powers of its
  symbols (`V*s`, `m/s^2`, `kg*m^2/s^2`, `V/Hz^0.5`), and units measuring the
  same quantity are convertible however they are written (`J`, `W*s`, `N*m`;
  `Hz`, `1/s`). New symbols: J, N, C, ohm, min and h. `units.product`,
  `units.quotient` and `units.simplify` build the units of computed
  quantities. With `pint` installed, symbols only pint knows (`inch`, `rpm`)
  are read in the same terms, so `inch/s` converts to `m/s`.
- `sigflow.stats`: statistics across frames. `moments` gives count, mean,
  variance, skewness and kurtosis over the whole stream or the last frames,
  matching `scipy.stats.describe` (`Moments` is the mergeable summary behind
  it); `outliers` tags outlying samples by a robust (median / MAD) score;
  `change_point` tags the frame where the mean or spread changes (two-sided
  CUSUM), with the likely onset. Each use of a block has its own state, every
  run starts afresh, and checkpoints capture it. Guide: *Statistics across
  frames*.
- A `Stateful` block may define `reset()`: every run calls it before starting
  (and before restoring a checkpoint), so the block starts afresh. Without it,
  state lives on between runs as before.
- Documentation: a Getting started section (installation with a table of the
  optional extras, a first graph); an API reference generated from the code;
  a *Running graphs* page with every option of every entry point; guides on
  parallel backends, streams and their lifecycle, debugging a running graph,
  and a full persistence guide. Example pages now show what to install, the
  command, a diagram of the graph, the expected output, the figures and
  reports the example writes, design decisions and limitations, and the
  index groups them into learning paths. `CONTRIBUTING.md` says which pages
  are generated and how.
- Examples 27-32: every `parallel()` backend on one workload, debugging a live
  run, checkpoint and resume, the stream lifecycle, empty / failed / late
  per-signal detections, and tag policies and ports.
- Free-threaded Python (3.13t, 3.14t): `parallel(block,
  backend="freethreaded")` runs the block on a thread pool that insists on a
  build without a GIL, where pure-Python blocks use several cores; it warns if
  an extension module has turned the GIL back on. `backend="auto"` picks it
  when the GIL is off and `"process"` otherwise. `sigflow.pools` holds the
  backends, with `free_threaded()` and `gil_enabled()`. CI runs the suite on
  3.14t.
- On a free-threaded build, synchronous generator stream blocks run in a
  worker thread instead of inline via greenlet, which crashes there
  (greenlet 3.5 on 3.14t).
- Breakpoints on two branches no longer trample each other: holds take turns,
  so each held frame is shown, and released, one at a time. Before, a frame
  held right after another was released could lose its hold state, and the
  run then hung on a frame nothing could release.
- `parallel(block, backend="interpreter")` (Python 3.14+): the block runs in
  sub-interpreters through `InterpreterPoolExecutor`, so pure-Python work
  (decoding, parsing) uses several cores in one process. The function must
  come from a plain-Python module and the frames must carry plain values,
  since sub-interpreters cannot load NumPy; `parallel` refuses lambdas,
  main-script functions and NumPy-importing modules when the graph is built,
  and a job a worker cannot start stops the run with an explanation.
- `sigflow.dims`: dimension and dtype-variable symbols (`N`, `M`, `T`, the
  whole alphabet; also exported from `sigflow`). `Sig[float, N]` builds the
  same contract as `Sig[float, "N"]`, which keeps working, but checkers read
  `N` as a type instead of an undefined name.
- Checker-readable spellings for the rest of the DSL, all equivalent at
  runtime to the old ones: shapes `tuple[M, N]`, `tuple[()]`,
  `tuple[Literal[1024]]`; rate and units `Literal[48_000]`, `Literal["V"]`;
  several outputs `tuple[Sig[...], Sig[...]]`; generator blocks
  `Iterator[Sig[...]]` / `AsyncIterator[...]`; a type built from configuration
  (`refine=`) as a call, `SignalType(float, (n,), rate)`.
- `NO_PROV`: every frame carries a `Provenance`; the empty one is falsy and
  renders as `(no provenance)`.
- Labelled arrays (`sigflow[xarray]`): an `xarray.DataArray` can be a frame's
  payload. `@block(labels=True)` lines the array up with the input contract
  (by dimension *name* when the contract names every dimension, e.g.
  `Sig[float, tuple[Literal["channel"], Literal["time"]]]`; by position
  otherwise), hands the body the bare array (`da.data`, so CuPy stays CuPy)
  and rebuilds a labelled array around each output from the output contract:
  a dimension variable seen on an input keeps its name and coordinate, new
  ones get their own name, fixed sizes become `dim_<axis>`. Runtime checks
  read a labelled array's sizes by name. A block with an unset `labels`
  option follows the run: `run(g, labels=True)` switches every such block on,
  including the built-in `dsp` / `linalg` blocks, and
  `sigflow.default_labels(True)` sets the package default a run reads once
  when it starts (so the choice is per run and reaches process-pool workers);
  `labels=False` opts out. `Framed` inputs and `DataArray` results pass
  through untouched. A stream block keeps only the item it is processing,
  and that item's own labels: dimension names, coordinates, name and
  attributes follow the latest labelled item, a plain-array item gets none,
  and the block's static inputs keep theirs.
- `@block(preserve_type=True)`: a single-input map block that carries its
  producer's complete contract (`types.narrow`) as the graph has pinned it
  down so far (`types.resolve`), so shape, rate and unit checks and reframing
  ratios see through it, in pipelines, reused subgraphs and `Graph` wiring in
  any order. The carried contract is merged axis by axis: a declared `(4,)`
  survives a producer that only says `(None,)`, repeated variables keep their
  equality across tuple elements, and required keys from both sides are kept.
- `Graph.connect` is transactional: any failure (type or rate mismatch, cycle,
  bad port name or index) restores the members, edges, bindings and naming
  counter, including nodes the failed call added implicitly.
- CPU / CUDA transfers (`sigflow.to_device`, `sigflow.to_host`; CuPy is
  optional): explicit moves of a frame's payload that keep dtype, shape,
  annotations and, for a `DataArray`, its labels, and carry the producer's
  contract (`preserve_type`). A transfer is a barrier: the source device is
  synchronised before its data is read, so work a producer launched on any
  stream is complete, and the copy is finished before the block returns.
  Device arrays of other libraries (`__cuda_array_interface__`) are accepted.
  A manually triggered GPU CI workflow (`.github/workflows/gpu.yml`) runs the
  device and xarray tests on a self-hosted runner; the docs gained a backend
  support matrix (**Guide → Array backends and GPU**).
- `per_key(key, body, ttl=)`: dynamic keyed routing. One instance of `body`
  per distinct key, created when the key is first seen, each a real stream
  with state of its own (stream blocks and `Stateful` objects alike), closed
  after `ttl` input frames without its key. Frames whose key is `None`, and
  failed frames, pass through. Outputs carry provenance `per_key(...)[key]`.
  `examples/25_per_signal.py` detects a varying number of tones in a spectrum
  and processes each through its own stateful graph.
- `gather(by, count= | lag= | ttl=, order=, ordered=, incomplete=)`: a stream
  block that folds the frames sharing a key back into one tuple frame with
  the members as parts, annotations and tags merged as at a join, a warning
  on a group released before it was complete. The mirror of a detector plus
  `per_key`; the example gathers every track's result per spectrum frame.
- Notebook inputs: knobs pick their widget from their value (entry fields for
  unbounded numbers, `widget="log"` / `"stepper"`, range sliders for pairs,
  `Enum` dropdowns, `multiple=True` multi-selects); `parse=` and `pattern=`
  validate input and a refused value marks the widget and leaves the knob
  unchanged; `group=` sections the panel; `Button(label, fn)` actions;
  `commit()` changes several knobs atomically for the blocks reading them;
  `run.controls(debug=True)` adds probe and breakpoint pickers.
- Notebook runs: a registry (`sigflow.runs()`, `stop_all()`), `start(name=,
  replace=True)` and `run.close()`, so re-running a cell leaves no run, panel,
  diagram or probe behind; `run.timeline` records knob changes with the frame
  they happened at, `run.replay()` applies a recorded timeline to another run,
  `run.report()` is a static summary that survives saving the notebook, and
  `display=False` builds panels and diagrams headless. CI executes the
  live-control notebook. A knob set from code with a widget attached no
  longer notifies its listeners twice.
- Breakpoints: `run.break_when("fft", pred)`, `break_at(name, seq)`,
  `tag=`, `every=`, `once=` and `run.break_on_error()` hold the graph on a
  frame at the block that produced it, before the next block receives it;
  `run.held`, `held_at`, `state == "break"`, `resume()`, `step()` (the next
  frame of the same block), `replace_held()`, `drop_held()` and
  `wait_break()`. The panel shows the held frame and the live diagram marks the
  block. Built on `Control.hold()` / `release()` and `Monitor.hold_at()`.
- Probes: `run.probe("fft")` watches one block of a background run without
  editing the graph: `latest`, `history`, `count`, `on(fn)`, `plot(block)`,
  `wait(n)`, `show()` and `close()`. Built on `Monitor.watch(name, fn)`, which
  also works with a blocking `run(g, monitor=mon)`.
- `parallel(..., backend="loky")` (`pip install sigflow[parallel]`): a process
  pool that needs no `if __name__ == "__main__":` guard and runs blocks defined
  in a script or a notebook cell, lambdas included, by sending their definition
  by value; workers are reused between runs. Backends live in `sigflow.pools`,
  and an unknown or uninstalled backend is refused when the graph is built.
- Every Python sample in the README runs as written, and `tests/test_readme.py`
  runs each one as its own script (a sample that shows an error must raise
  exactly that error). The zip sample was wrong as written: `>>` binds tighter
  than `&`, so it now parenthesises the zip.
- A `parallel(..., backend="process")` whose worker dies now stops the run
  with an error naming the usual cause: since workers start from a fork
  server they import the main module, so a script must run its graph under
  `if __name__ == "__main__":` (the README says so too).
- `examples/26_graph_checks.py` walks through what `Graph.connect` checks on
  the spot and what `validate()` checks later; one `connect` now emits a
  `TagWarning` once instead of once per re-check pass.
- Tag contracts in signatures: `accepts(...)` declares tags an input reads when
  present, `may_provide(...)` tags an output sometimes attaches, and
  `requires` / `accepts` / `provides` / `may_provide` take value types by
  keyword (`requires(track=int)`), checked when the graph is built against the
  producer's declarations and at run time on the frame. A block's propagation
  policy is `@block(tags="carry" | "none" | "declared" | keep(...) | drop(...),
  spans="carry" | "drop", merge="union" | "intersection" | "first")`, applied
  statically and at run time. `carries(...)` gives one port of a tuple output
  a policy of its own. `may_provide` travels downstream, and a `TagWarning`
  names an `accepts`, a `when("tag")` or a `per_key("tag")` that nothing
  upstream provides or may provide. `describe()` shows a block's policy and
  `run.report()` a table of every block's tag contract. The plan is in
  `design/tag-contracts.md`.
- Live diagram: `run.diagram()` displays the graph's diagram refreshed while a
  background run goes, with frames in and out on every block, errors in red, a
  state colour on the border and frame counts on the edges;
  `run.controls(diagram=True)` adds it to the control panel and
  `g.draw(stats=run.monitor)` renders the same still after any run
  (`Monitor.lookup(node)` finds a node's stats). Edges with a runtime queue
  show its depth, a full queue turns the edge amber (backpressure), blocks
  show their throughput while running, and a still shows each queue's peak
  (`Monitor.queue_between(a, b)`). `g.draw(frame=f)` and
  `live.highlight(f)` draw the path one frame took, matched from its lineage,
  with the router branch it took and, for a failed frame, the block where it
  broke.
- Tags, a fourth annotation level: `frame.tag("speech", segment=3)`,
  `frame.tag("burst", span=(1.2, 1.5))`, `frame.untag(...)`, `frame.mask(name)`,
  `frame.tags.at(t)`; `"speech" in frame` and `frame["segment"]` look tags up
  too. A `Tag` is a name, a scalar value, an optional time span and the block
  that attached it (stamped by the runtime). Tags are united at joins, spanned
  tags are clipped to each frame's window and dropped by blocks that change
  the leading axis, and `Frame.save` / `Frame.load` round-trip them. Contracts
  declare them: `Annotated[Sig[...], requires("speech")]` on an input is
  checked when the graph is built against the producer's guarantees and at
  run time; `provides("bits")` on an output is checked at run time;
  `clears("raw")` stops a tag from being carried on (as values:
  `SignalType(...).requires(...)` / `.provides` / `.clears`).
  `when("speech", then=...)` routes frames by tag and lets the branch require
  it; `when` / `switch` keep the tags every branch guarantees even when the
  payload type becomes gradual. Guarantees mean whole-frame tags. `examples/24_tags.py`
  walks through all of it. Guarantees flow at placement through
  generic blocks, fan-outs, zips, port selection, `when` / `switch`, reused
  subgraphs and `Graph` wiring. (`SignalType.meta`, the dormant "required
  annotation keys", is what `requires()` now fills.)
- `Frame.slice` / `frame[a:b]` / `at` / `between` index array-like payloads
  directly, so labelled and device arrays keep their labels and placement;
  `Frame.save` / `Frame.load` round-trip a `DataArray` (values, dims, coords,
  attrs, name) and fetch device arrays to the host.

Changed

- `units.compatible` and `units.factor` no longer treat a unit the registry
  cannot read as convertible to itself; `factor` raises `UnitError` for it.
- Iteration (`iterate`, `iterate_sync`, `for f in graph`) resumed from a
  checkpoint now restores the states of stateful blocks, as `run` does, and
  takes a checkpoint path as well as a `Checkpoint`; it used to skip the
  sources' emitted frames only.
- `on_error` must be `"propagate"`, `"raise"` or `"drop"`, in `run()` and in
  `@block(...)`; any other value is a `ValueError`. Before, a misspelt policy
  such as `"mark"` or `"skip"` silently behaved like `"propagate"`.
- `iterate()` / `iterate_sync()` / `for f in graph` refuse an option they do
  not take (`TypeError`), as `run()` does; before, they ignored it.
  `start()` checks its options when called, not later in its thread.
- `iterate()` and `iterate_sync()` are annotated as the generators they
  return, so `contextlib.aclosing` / `closing` type-check.
- Declared rates and units are checked when a graph is built even when the
  produced shape is unknown; before, an unknown shape skipped those checks
  until run time.
- Tuple frames keep one story: `frame.evolve(data=...)` drops parts that
  described the old data (so `frame[0]` and port selection no longer return
  stale frames), a plain tuple returned for a tuple frame whose parts carry
  different tags gets parts of its own so each port keeps its tags, and
  `Frame.save` / `Frame.load` round-trip per-port tags. Stopping a lowered
  graph early no longer logs an unretrieved cancellation.
- Stopping a synchronous iteration early (`iterate_sync`, `replay`, a `break`
  or an exception in the consumer) now stops the graph: the helper thread is
  told to stop and joined, and the sources' generators are closed at once
  rather than at garbage collection. A tuple-to-tuple block carries tag
  guarantees port by port, and clearing a tag on one port removes it from the
  tuple frame unless another part still carries it.
- The `zmq` extra pulls in `tornado` on Windows, which pyzmq's asyncio support
  needs on the default proactor event loop.
- `parallel(..., backend="process")` starts its workers from a fork server
  (a fresh interpreter where there is none) instead of forking the running,
  multi-threaded runtime, which Python 3.12+ warns about. Blocks sent to a
  worker are re-imported there, as before.
- `Frame.prov` and `Issue.prov` are `Provenance`, never `None` (passing `None`
  still works and stores `NO_PROV`); test with `if f.prov:` rather than
  `is None`.
- Under `TYPE_CHECKING`, `Sig` is a variadic type alias to `Any` rather than an
  `ndarray` subclass: block bodies are no longer checked as NumPy code, and
  returning an array, scalar or dict for a `Sig` output is no longer an error.
- The package, tests, examples, notebooks and docs use the new spellings. No
  mypy error codes are disabled anywhere; `mypy src/sigflow tests examples`
  and `pyright` (standard mode) gate CI. `Frame.save` / `Frame.load` and
  `Provenance.to_dict` / `to_json` / `from_dict` / `from_json` are declared on
  the classes instead of attached at import time, so editors see them. The
  `draw(ax, data, frame)` callback of `plot.custom` may return a value.

Fixed

- `sigflow.__version__` comes from the installed package, so it matches
  `pyproject.toml` (it said 0.4.0).
- `per_key`: an instance that finishes by itself (`take(1)`, an error) retires
  its key, and later frames of that key are dropped until `ttl` frames pass
  without it. Before, the router filled the finished instance's queue and the
  run waited for ever.
- `per_key`: a builder `Graph` or `switch` body is copied whole for each key.
  Before, keys could share a stateful block's state, and the template's blocks
  were replaced.
- `gather(ordered=True)`: a key that completed a second group while an earlier
  group was still waiting overwrote the first; every group now goes out.
- A block made with `.bind()` runs under `parallel()` with the process and loky
  backends; it pickles as the original block, bound again.
- `Probe.close()` detaches a probe placed on a node shown under another name
  (`scale#2`); a closed probe ignores frames already on their way.
- `tags="none"`, `keep(...)` and `spans="drop"` also govern inherited tags the
  block clipped (a slice) or shifted: only tags the block attached itself are
  kept regardless of the policy.

## 0.4.0

New

- Stream-rate contracts (`sigflow.ratio`): `@block(ratio=...)` declares frames
  out per frame in (`"1:1"`, `"N:1"`, `"1:N"`, `(4, 1)`, or a callable of the
  configuration and `n_in`); map blocks are `1:1` by definition. Ratios
  multiply along pipelines (`node.ratio`, `describe()` tags), fan-out / zip
  branches and `Graph` edges at incompatible rates raise `RatioMismatch` at
  build time, `run(strict=True)` verifies declared ratios, and the progress
  monitor scales source totals by the graph ratio. `dsp.chunk`, `overlap`,
  `buffer`, `decimate`, `running_mean`, `fir`, `timestamp` declare theirs.
- `switch(selector, {key: branch, ...}, default=..., ordered=...)`: keyed
  routing through real stream branches (windowing, state, drops allowed) that
  run concurrently; `default` is a node, `"pass"` or drop; `ordered=True`
  keeps input order for 1:1 branches; provenance names the branch taken.
- Example 21 (modulation-switched demodulators, rate contracts).

Changed

- A fan-out whose branches have different _declared_ rates (for example
  `dsp.rms | dsp.buffer(count=2)`) is now a build-time error rather than a
  run-time `strict=True` one.

## 0.3.0

New

- One runtime: operator trees (`>>`, `|`, `&`, `feedback`, `when`, `align`,
  slicing) are lowered to the `Graph` scheduler, which is the only executor.
  Linear runs of map blocks are fused into a single scheduler member; the
  output edge is iterated directly; errors and cancellation propagate through
  a shared failure future.
- Synchronous generator stream blocks run inline on the event loop when
  `greenlet` is installed (`pip install sigflow[perf]`), ~10x less overhead
  than the thread hop; `threaded=True` still forces a thread.
- `run(..., provenance="full" | "roots" | "off")` controls per-frame lineage
  bookkeeping; `"roots"` keeps replay and checkpoints working.
- `parallel(block, workers, backend="process" | "thread")`: a map block on a
  pool inside one graph, in order, with provenance and failed frames as usual.
  Blocks pickle by import path.
- `io.zmq` / `io.zmq_sink` (`pip install sigflow[zmq]`): PUB/SUB and
  PUSH/PULL, wire-compatible with GNU Radio's ZMQ blocks.
- `Lineage`: query provenance across a run — `failed`, `through(op, **params)`,
  `descendants_of`, `sources_of`, `blame`, `configs`, `diff`, `table`.
- Static typing: `Sig` / `Stream` / `Framed` have `TYPE_CHECKING` stubs so
  block bodies type-check as NumPy code; `py.typed`; `mypy src/sigflow` clean.
- Docs site (`mkdocs-material`): concepts, guides, generated example pages;
  `docs/build_docs.py`; CI builds it with `--strict`.

Changed

- `plot` taps: the `every` parameter is now `stride`.
- Benchmarks refreshed (`docs/performance.md`); README gained sections on the
  scheduler, `parallel`, lineage queries, typing and the docs site.

## 0.2.0

New

- `Graph` builder (`sigflow.builder`): explicit DAGs with `add` / `connect` (by
  position or input name, with port selection on edges) / `output`; edge-time
  type checking, cycle rejection, layered drawing, reports.
- Slicing: `node[a:b]` on array-typed nodes with static shape propagation and
  sample annotations / time sliced in step; `frame[a:b]`; stream slicing
  `take` / `skip` / `every` / `until` / `while_` (identity-typed).
- Time axis: `Frame.time` (`TimeSpan`), `frame.at(t)`, `frame.between()`;
  re-framing blocks keep exact `t0`/`rate`; `align()` joins streams by time.
- Units: `sigflow.units` registry, compatibility in unification with a
  conversion hint, `dsp.to_units`.
- Persistence: provenance to/from JSON, `frame.save` / `Frame.load`,
  `replay(graph, frame)`, `Stateful` blocks, `run.checkpoint()` and
  `resume=` on `run` / `start`.
- Plotting: `plot.history`, `plot.multi`, ipympl-native live figures.
- `run(..., strict=True)` raises on fan-out / zip cardinality mismatch.
- `bench/bench.py` and `docs/performance.md`; GitHub Actions CI matrix
  (Python 3.10–3.13, NumPy 1.x/2.x, Linux/macOS/Windows); `py.typed` and a
  clean `mypy` run (block bodies excluded: `Sig[...]` is a runtime DSL).

Changed

- Positional arguments to a block definition: arrays / Frames / lists call the
  function; strings, numbers and other scalars are configuration
  (`to_units("V")` makes a node). `block.fn(...)` always calls the function.
- A block that returns a `Frame` always gets fresh provenance for the node.
- `requires-python` lowered to 3.10.

Fixed

- Overlap/chunk windows carried the input chunk's `t0`; now exact.
- `step()` before the run loop attached could leave a paused run stuck.

## 0.1.0

Initial version: typed blocks, operators, asyncio runtime with sync adapters,
annotations and error propagation, provenance, plotting, notebook controls,
ports, `feedback` / `when`, `sigflow.io`, examples 01–19.
