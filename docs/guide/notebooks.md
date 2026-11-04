# Notebooks and control

Graphs, frames, diagrams, monitors and plots all render in Jupyter. For long
runs, `r = g.start()` keeps the kernel free and `r.controls(plots=[...],
knobs=[...])` shows a panel: Pause / Resume / Step (with a count) / Stop,
Snapshot and Record, a throttle slider, a progress bar, one widget per `Knob`
(slider, toggle, dropdown or text), the plot blocks' figures refreshed live,
and the status table. `dashboard({...})` stacks panels for several runs.

`run.diagram()` shows the graph's diagram live: frames in and out on every
block, errors in red, the state as the border colour (grey idle, blue active,
amber paused, red errors, green finished), frame counts on the edges and a
status line. It needs no ipywidgets. `run.controls(diagram=True)` adds the same
picture to the panel, and `g.draw(stats=run.monitor)` is a still of it after a
run.

Where the runtime keeps a queue between two blocks, the edge carries a bar of
its depth against its capacity, and a full queue turns the edge amber: that is
backpressure, and the block after the edge is the slow one. Queues sit where a
graph branches or joins (fan-outs, zips, `Graph` wiring); a straight chain of
map blocks is fused and passes frames along without a queue, so its edges show
counts only. A queue only backs up when the slow block yields while it works:
an `async` block, a threaded one, or a process pool. A block that blocks the
event loop holds everything back with it, and no queue fills. While a run goes
every block also shows its throughput in frames per second, and a still after
a run shows each queue's peak depth.

`g.draw(frame=out[7])` draws the path one frame took instead: the blocks and
edges it passed through stand out with its sequence number at each, everything
else fades, a router shows the branch it went down, and for a failed frame the
block where it broke is red. `live.highlight(frame)` does the same on a live
diagram and `live.highlight(None)` goes back to the counts. The path is found
by matching the frame's lineage against the graph, so two blocks with the same
name are told apart by where they sit. With `provenance="roots"` lineage keeps
only the sources, so the path shows the source and the final block.

`run.probe("fft")` watches one block without editing the graph: `probe.latest`
is the newest frame the block put out, `probe.history` the recent ones,
`probe.on(fn)` runs a callback per frame, `probe.plot(plot.spectrum())` feeds
a plot block as if it sat after the block, `probe.wait(n)` blocks until more
frames arrive and `probe.show()` keeps the latest frame displayed. A second
block of the same name is `"fft#2"`, a fan-out or zip is `"fanout"` / `"zip"`,
as in `run.monitor.table()`; a node object works too.

`run.break_when("fft", lambda f: f["snr"] < 3)` sets a breakpoint: when a
frame `fft` puts out matches, the graph holds it there, before the next block
receives it, and upstream stalls on its queues. `run.held` is the frame,
`run.held_at` the block, `run.state` is `"break"`. `run.resume()` lets it go,
`run.step()` lets it go and holds the next frame of the same block,
`run.replace_held(frame)` and `run.drop_held()` change what goes on. Match on
a tag (`tag="clipped"`), a sequence number (`run.break_at("fft", seq=143)`),
every n-th match (`every=50`) or any failure anywhere (`run.break_on_error()`).
The panel shows the held frame with Resume and Step wired up, the live
diagram marks the block, and `run.wait_break(timeout)` blocks a script until a
breakpoint hits.

The panel needs **ipywidgets in the kernel** (`python -m pip install
ipywidgets`, then restart the kernel); `require_widgets()` tells you whether it
will work. Live figures inside the panel are refreshed images; with the
`ipympl` backend (`%matplotlib widget`) a plot block with `live=True` updates
its interactive canvas in place during a blocking `run()`.

`Knob(value, name, min, max, step)`, `Knob(True)`, `Knob("hann",
choices=[...])`; a map block declared with one re-reads it on every call,
sources read `knob.value` themselves; `knob.on_change(fn)` chains knobs; plot
options accept knobs. Provenance records the values actually used.

`examples/live_control.ipynb` walks through all of it; run it top to bottom in
a live kernel.
