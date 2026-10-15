"""Generates examples/live_control.ipynb (run: python make_live_notebook.py).

The notebook is written WITHOUT outputs on purpose: its widgets only exist while a
kernel runs the cells, so open it in Jupyter and run it top to bottom.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Controlling long-running graphs from a notebook

Run this notebook **top to bottom in a live kernel** — the panels are ipywidgets and only exist while
the kernel runs. Every graph here is started in the background with `g.start()`, so the kernel stays
free while it runs: press buttons, drag sliders, inspect `run.frames`, start more graphs.

| in the panel | what it does |
|---|---|
| **Pause / Resume** | hold or release the *sources*; in-flight frames drain through plots and sinks first |
| **Step** `n` | release exactly *n* source frames, then stay paused |
| **Stop** | end the run; frames kept so far stay available |
| **Snapshot** | keep the latest output frame in `run.snapshots` |
| **Record** | collect every output frame into `run.recording` while on |
| **throttle / max fps** | slow the sources down to at most that many frames per second (log scale) |
| progress bar | percent complete when the source declares a length; an activity sweep otherwise |
| knob widgets | sliders / toggles / dropdowns / text bound to `Knob` parameters of running blocks |
| figures | plot blocks refreshed live inside the panel |
| status table | per-block in / out / errors / time |

The same things are available from code: `run.pause()`, `run.step(5)`, `run.set_rate(20)`,
`run.set_recording(True)`, `run.snapshot()`, `knob.value = ...`.""")

code("""import sys, pathlib, time
try:
    import sigflow
except ModuleNotFoundError:
    sys.path.insert(0, str(pathlib.Path.cwd().resolve().parent / "src")); import sigflow
import numpy as np
%matplotlib inline
from sigflow import Sig, Stream, Frame, Framed, Knob, block, source, run, port, dashboard, require_widgets
from sigflow.dims import N
from sigflow import dsp, plot
require_widgets()          # green = panels will work; red = what to install""")

md("""## 1. Transport: Pause, Resume, Step, Stop

An endless spectrum chain (`frames=10**9`), started **paused**. Press **Step** a few times (change the
count box next to it to release more per press), then **Resume**, then **Pause**. The status table
underneath updates while it runs.""")
code("""tone = dsp.sine(freq=1000, rate=48_000, n=1024, frames=10**9) >> dsp.window >> dsp.to_complex >> dsp.fft >> dsp.power_db
run1 = tone.start(paused=True, keep=200)
run1.controls()""")
code("""# the buttons are plain method calls, so the same works from code:
run1.step(5); time.sleep(0.3)
print(run1.state, len(run1.frames), "frames;", run1.latest)
run1.resume(); time.sleep(0.5); run1.pause()
print(run1.status())""")

md("""## 2. Throttle, progress and live figures

A tone drifts across the band. Tick **throttle** and drag **max fps** down to ~5 to watch it move
frame by frame; the spectrum and waterfall are refreshed inside the panel. This source declares its
length (`total=`), so the progress bar shows percent.""")
code("""@source(total=lambda frames, **_: frames)
def drifting_tone(rate: float = 48_000, n: int = 1024, frames: int = 3000) -> Sig[float, N]:
    t0 = 0
    for k in range(frames):
        f = 6000 + 4000 * np.sin(2 * np.pi * k / 150)                 # sweeps 2-10 kHz every 150 frames
        t = (np.arange(n) + t0) / rate
        t0 += n
        yield Frame(np.sin(2 * np.pi * f * t) + 0.05 * np.random.default_rng(k).normal(size=n),
                    meta={"seq": k, "freq_hz": f}, stream={"rate": rate})

spec = plot.spectrum(title="spectrum", ylim=(-70, 10), figsize=(5.5, 2.6))
fall = plot.waterfall(title="waterfall", rows=80, figsize=(5.5, 2.6))
run2 = (drifting_tone() >> spec >> dsp.to_complex >> fall).start(keep=50)
run2.set_rate(20)                                   # start throttled; the slider changes it live
run2.controls(plots=[spec, fall], refresh=0.3)""")
code("""time.sleep(2); run2.pause()
print(run2.status(), "| last tone at %.0f Hz" % run2.latest["freq_hz"])""")

md("""## 3. Knobs: sliders, dropdowns, toggles — and plot options — on a running graph

`Knob(value, name, min, max, step)` makes a float/int slider, `Knob(True)` a toggle,
`Knob("hann", choices=[...])` a dropdown, and a Knob of a tuple can drive a plot's `ylim`.
A **map** block declared with a Knob reads it on every call; **sources** receive the Knob and read
`.value` themselves. Drag the sliders and watch the spectrum.""")
code("""freq   = Knob(3000.0, "freq_hz", min=200.0, max=20_000.0, step=50.0)
gain   = Knob(1.0, "gain", min=0.0, max=4.0, step=0.05)
window = Knob("hann", "window", choices=["hann", "hamming", "blackman", "boxcar"])
notch  = Knob(True, "notch 5 kHz")
floor  = Knob(-70.0, "plot floor dB", min=-120.0, max=-10.0, step=5.0)

@source
def tunable_tone(freq: Knob, rate: float = 48_000, n: int = 1024, frames: int = 10**9) -> Sig[float, N]:
    t0 = 0
    for k in range(frames):
        t = (np.arange(n) + t0) / rate; t0 += n
        yield Frame(np.sin(2 * np.pi * freq.value * t) + 0.02 * np.random.default_rng(k).normal(size=n),
                    meta={"seq": k}, stream={"rate": rate})

@block
def notch_5k(x: Sig[float, N], on: bool = True, rate: float = 48_000) -> Sig[float, N]:
    if not on:
        return x
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / rate)
    X[(f > 4500) & (f < 5500)] = 0
    return np.fft.irfft(X, n=len(x))

ylim = Knob((-70.0, 10.0))
floor.on_change(lambda v: setattr(ylim, "value", (v, 10.0)))      # one knob driving another

live_spec = plot.spectrum(title="tune me", ylim=ylim, figsize=(6, 2.8))
run3 = (tunable_tone(freq=freq) >> dsp.window(kind=window) >> notch_5k(on=notch) >> dsp.scale(gain=gain) >> live_spec).start(keep=20)
run3.set_rate(30)
run3.controls(plots=[live_spec], knobs=[freq, gain, window, notch, floor], table=False)""")
code("""# knobs are ordinary objects too: a scripted sweep while it runs (the sliders follow)
for f in (2000, 5000, 9000):
    freq.value = f; gain.value = 0.5 + f / 9000
    time.sleep(0.6)
    print(f"freq={freq.value:.0f} Hz gain={gain.value:.2f} window={window.value} -> rms {dsp.rms(run3.latest.data):.3f}")
print("provenance records the values actually used:", run3.latest.prov["scale"].params, run3.latest.prov["window"].params)""")

md("""## 4. Snapshot and Record

**Snapshot** keeps the latest frame; **Record** collects everything while it is on. Both are also
methods. Snapshots keep their provenance, so each one knows the knob settings it was made with.""")
code("""run3.snapshot(); gain.value = 3.0; time.sleep(0.3); run3.snapshot()
print([f"gain={s.prov['scale'].params['gain']}" for s in run3.snapshots])
run3.set_recording(True); time.sleep(0.5); run3.set_recording(False)
print(len(run3.recording), "frames recorded;", "rms range %.2f-%.2f" % (
      min(dsp.rms(f.data) for f in run3.recording), max(dsp.rms(f.data) for f in run3.recording)))
plot.animate(run3.recording[:10], interval=120)   # a FuncAnimation of what was recorded""")

md("""## 5. Several runs: a dashboard

`dashboard({...})` stacks a panel per run with Pause-all / Resume-all / Stop-all on top.""")
code("""runs = {
    "tone":  (dsp.sine(n=512, frames=10**9) >> dsp.rms).start(keep=1),
    "noise": (dsp.noise(n=512, frames=10**9, seed=1) >> dsp.rms).start(keep=1),
    "chirp": (dsp.chirp(n=512, frames=10**9) >> dsp.rms).start(keep=1),
}
for r in runs.values():
    r.set_rate(200)
panels = dashboard(runs)""")
code("""time.sleep(0.5)
for name, r in runs.items():
    print(f"{name:6s} {r.state:8s} {r.monitor.produced:6d} frames  latest rms {r.latest.data:.3f}")""")

md("""## 6. Endless streams without unbounded memory: `keep` and `drain()`

`keep=N` makes `run.frames` a ring buffer; `run.drain()` returns what arrived since the last call and
clears it — the natural way to summarise a live stream periodically.""")
code("""levels = dsp.noise(n=256, frames=10**9, seed=3) >> dsp.detect_clipping(limit=3.0) >> dsp.rms
run4 = levels.start(keep=100_000)
for i in range(4):
    time.sleep(0.25)
    batch = run4.drain()
    print(f"batch {i}: {len(batch):5d} frames, mean rms {np.mean([f.data for f in batch]):.3f}, "
          f"{sum(bool(f.warnings) for f in batch)} with clipping warnings")
run4.stop()""")

md("""## 7. Stopping on a condition

`start(sink=fn)` calls `fn` for every frame on the runtime thread; it may call `run.stop()`.
`run.wait(timeout)` blocks the cell until the run ends.""")
code("""@source
def random_walk(frames: int = 10**9, seed: int = 0) -> Sig[float, ()]:
    rng = np.random.default_rng(seed); x = 0.0
    for _ in range(frames):
        x += rng.normal(); yield x

holder = {}
def stop_when_far(f):
    if abs(f.data) > 25:
        holder["run"].stop()

# start paused: the sink can fire before `start()` even returns, so wire the handle first
holder["run"] = (random_walk() >> (lambda x: x)).start(sink=stop_when_far, keep=5, paused=True)
holder["run"].resume()
frames = holder["run"].wait(timeout=30)
print(holder["run"].state, "after", holder["run"].monitor.stats("random_walk").out, "steps; last values",
      [round(f.data, 1) for f in frames])""")

md("""## 8. Tidy up and look back

A stopped run keeps its frames, snapshots, recording, monitor and lineage.""")
code("""for r in (run1, run2, run3, *runs.values()):
    r.stop()
print({k: r.state for k, r in {"run1": run1, "run2": run2, "run3": run3, **runs}.items()})
f = run2.latest
print(f, "| tone was at %.0f Hz" % f["freq_hz"], "| lineage:", [n.op for n in f.prov.chain()])
run2.monitor""")

nb["cells"] = cells
nb.metadata["kernelspec"] = {
    "name": "python3",
    "display_name": "Python 3",
    "language": "python",
}
nbf.write(nb, "live_control.ipynb")
print("wrote live_control.ipynb (no outputs: run it in Jupyter)")
