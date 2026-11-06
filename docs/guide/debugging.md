# Debugging a running graph

A wrong number at the end of a graph rarely says where it went wrong. sigflow
lets you look inside a running graph without editing it: watch a block's
output, hold the frame that looks wrong before anything downstream sees it,
inspect it, fix or drop it, and carry on. `examples/28_debug_a_live_run.py`
follows this page step by step; its output is quoted below.

The workflow:

1. notice the unexpected result, and trace it back;
2. **probe** an intermediate block;
3. **hold** a matching frame with a breakpoint;
4. inspect its data, issues, tags and provenance;
5. **replace** or **drop** it, and resume;
6. save a static **run report**;
7. close what you opened.

## 1. Trace the result back

Every output frame knows how it was made, so start from the bad one:

```python
out = run(graph())
bad = max(out, key=lambda f: f.data)
bad.prov.roots()  # the source frames behind it
bad.prov.chain()  # the blocks it went through, newest first
bad.prov["calibrate"]  # that block's record: its configuration, its sequence number
```

```text
1. peak readings around 1 V, but frame 7 reads 600 V
   it came from source frame 7 by level <- calibrate <- sensor
   and frame 11 failed: ValueError: empty frame
```

For questions across many outputs, which failed, and from which source frames,
index them: `Lineage(out).blame()`. See [Lineage](../concepts/lineage.md).

## 2. Probe a block

Start the graph in the background, paused, and probe the block you suspect. A
probe sees every frame the block puts out, as the next block would receive it:

```python
h = graph().start(paused=True)
probe = h.probe("calibrate", keep=4)  # a block's name; "calibrate#2" for a second one
h.step(3)  # let three source frames through
probe.wait(3, timeout=5)
probe.latest, probe.history, probe.count, probe.failed
```

A probe can also feed a plot (`probe.plot(plot.line())`) or call a function per
frame (`probe.on(fn)`); in a notebook, `probe.show()` keeps the latest frame on
screen. `h.monitor.table()` lists the names to probe.

## 3. Hold the frame that looks wrong

A breakpoint holds a frame **at the block that produced it**: nothing
downstream sees it until you let it go.

```python
h.break_when("calibrate", lambda f: np.max(np.abs(f.data)) > 10, once=True)
h.break_on_error()  # also hold any frame whose block failed
h.resume()
held = h.wait_break(timeout=5)
h.held_at, h.state  # "calibrate", "break"
```

Other conditions: `break_when(block, tag="speech")`, `seq=7`, `every=100`, or
`break_at(block, seq)` for one frame. While a frame is held the run's `state`
is `"break"`; blocks on other branches go on until they need something from the
held block, or until the queues between them fill.

## 4. Inspect it

```python
held.data, held.meta, held.stream  # payload and annotations
held.issues, held.errors, held.failed
held.tags  # spanned and whole-frame tags
held.prov.chain()  # how it was made
```

```text
3. held at calibrate (state break): seq 7, t0 0.448 s, max 600 V at sample 20
   issues: none; lineage: calibrate <- sensor
```

## 5. Replace or drop it, and resume

```python
h.replace_held(held.evolve(data=fixed))  # go on with a corrected frame
h.drop_held()  # or go on without it
h.step()  # or go on, and hold this block's next frame
h.resume()  # or just go on
```

A replaced frame keeps the held frame's provenance, so the outputs still trace
back to the source frame they came from.

## 6. Save a static report

Live widgets do not survive saving a notebook. `h.report()` is plain HTML: the
run's state and totals, the diagram with every block's counts and each queue's
peak, the block table, knobs and their changes, the breakpoints and probes and
what they saw, and the latest frame.

```python
h.report().save("debug_report.html")
```

In a notebook, evaluating `h.report()` shows it inline. The page of
[example 28](../examples/28_debug_a_live_run.md) links the report its run
saved, so you can see one without running anything.

## 7. Clean up

```python
probe.close()  # stop watching
h.close()  # stop the run; detach its panel, diagram, probes and breakpoints
```

`sigflow.stop_all()` stops every run still going, and
`start(name=..., replace=True)` stops an earlier run of the same name first.

## In a notebook

`h.controls()` gives a panel with Pause, Resume, Step and Stop and shows the
held frame; `h.controls(debug=True)` adds a row to probe a block or set a
breakpoint on it by picking it from a list. `h.diagram()` draws the graph live,
with the held frame's block highlighted. Replacing or dropping a held frame is
done from code, as above. See [Notebooks and control](notebooks.md).
