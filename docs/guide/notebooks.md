# Notebooks and control

Graphs, frames, diagrams, monitors and plots all render in Jupyter. For long
runs, `r = g.start()` keeps the kernel free and `r.controls(plots=[...],
knobs=[...])` shows a panel: Pause / Resume / Step (with a count) / Stop,
Snapshot and Record, a throttle slider, a progress bar, one widget per `Knob`
(slider, toggle, dropdown or text), the plot blocks' figures refreshed live,
and the status table. `dashboard({...})` stacks panels for several runs.

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
