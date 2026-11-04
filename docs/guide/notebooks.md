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

## Inputs

A knob picks its widget from its value:

| knob | widget |
|---|---|
| a number with `min` and `max` | a slider (`widget="log"`: a log slider) |
| a number without bounds | an entry field (`widget="stepper"`: an integer entry with arrows) |
| a pair of numbers, `Knob((300.0, 3400.0), min=0, max=24000)` | a two-handle range slider |
| `choices=[...]` | a dropdown; with `multiple=True` a multi-select, the value a tuple |
| an `Enum` member | a dropdown of its members; the block gets the member |
| a bool | a toggle |
| anything else, or any knob with `parse=` | a text field |

`parse=` turns typed text into a value (`Knob("48k", parse=to_hz)`) and may
raise to refuse it; `pattern=` is a regular expression a text value must
match. A refused value leaves the knob as it was, gives the widget a red
border with the reason as its tooltip, and sets `knob.error`; setting a bad
value from code raises `InvalidValue`. `group="filter"` puts knobs in a
labelled section of the panel. `Button("reset", fn)` is an action:
`run.controls(knobs=[..., Button(...)])`. `commit((low, 300.0), (high, 3400.0))`
changes several knobs at once: a block reads all of the new values or none of
them, and a refused value changes nothing.

`run.controls(debug=True)` adds pickers: choose a block and **Probe** it (its
latest frame shows below), or **Break** on it with a condition written in
Python over `f`, the frame (`f["snr"] < 3`), and an optional tag picked from
the tags seen so far; **Clear** removes those breakpoints.

## Re-running cells

Every run is registered. `g.start(name="demo", replace=True)` stops and
closes a run still going under that name first, so re-running a cell does
not leave the previous run, its panel, its diagram or its probes behind.
`sigflow.runs()` lists the runs still going, `sigflow.stop_all()` stops them,
and `run.close()` stops one and releases everything watching it. Showing
`run.controls()` or `run.diagram()` again replaces the previous one, and runs
still going are stopped when the kernel exits.

## Keeping a session

`run.timeline` records every change of a knob in the graph during the run:
when, at which frame, which knob, and the value (`run.timeline_records()` is
JSON-ready). `other_run.replay(records)` applies them to another run's knobs
at the same frames (`by="time"`: at the same times), which turns an
interactive session into something repeatable. `run.report()` is a static
summary for after the run: the diagram with every block's counts and each
queue's peak, the block table, the knobs and their changes, breakpoints and
probes, and the latest frame. It is plain HTML, so unlike live widgets it
survives saving and sharing the notebook; `report.save("run.html")` writes it.

`run.controls(display=False)` and `run.diagram(display=False)` build the
panel or diagram without showing it or refreshing it on a timer
(`panel.render()` refreshes it), for tests and scripts. CI executes
`examples/live_control.ipynb` end to end.

`examples/live_control.ipynb` walks through all of it; run it top to bottom in
a live kernel.
