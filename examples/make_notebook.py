"""Generates examples/sigflow_tour.ipynb (run: python make_notebook.py)."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# sigflow tour

Typed, functional, generator-based signal-processing graphs. This notebook walks through
blocks, composition, compile-time type checking, annotations, error propagation,
provenance, mixed sync/async streams, live progress and plotting.""")

code("""# If `sigflow` was installed into a different Python than this kernel uses,
# fall back to the package source that sits next to this notebook (../src).
import sys, pathlib
try:
    import sigflow
except ModuleNotFoundError:
    sys.path.insert(0, str(pathlib.Path.cwd().resolve().parent / "src"))
    import sigflow
print("sigflow", sigflow.__version__, "from", pathlib.Path(sigflow.__file__).parent, "| kernel:", sys.executable)""")
code("""import numpy as np
import matplotlib
%matplotlib inline
from sigflow import Sig, Stream, Framed, Frame, TypeMismatch, RatioMismatch, block, source, run
from sigflow.dims import M, N, T
from sigflow import dsp, linalg, plot""")

md("""## 1. Blocks are plain functions with typed signatures
Parameters annotated with `Sig[...]` are inputs; everything else is configuration fixed when the block
is placed in a graph. `"T"` is a dtype variable, `"N"` a symbolic dimension.""")
code("""@block
def scale(x: Sig[T, N], gain: float = 1.0) -> Sig[T, N]:
    return x * gain

@block
def energy(x: Sig[T, N]) -> Sig[float, ()]:
    return float(np.sum(np.abs(x) ** 2))

scale(np.arange(4.0), gain=2)   # still an ordinary function""")

md("""## 2. Compose with operators; the graph is type-checked as you build it
A graph renders as a table in Jupyter.""")
code("""g = dsp.sine(freq=1000, rate=48_000, n=1024, frames=20) >> dsp.window >> dsp.to_complex >> dsp.fft >> dsp.power_db
g""")
md(
    """`g.draw()` renders the topology as an SVG (hover a box for its full signature); `g.to_dot()` exports Graphviz."""
)
code("""@block
def add(a: Sig[T, N], b: Sig[T, N]) -> Sig[T, N]:
    return a + b

front = ((dsp.sine(n=256, frames=8) >> dsp.window) & dsp.noise(n=256, frames=8)) >> add >> dsp.to_complex
try:                       # branches must run at compatible rates: buffer(4) is 4:1, the others 1:1
    front >> (dsp.fft >> dsp.power_db | dsp.rms | dsp.buffer(count=4) >> linalg.norm)
except RatioMismatch as e:
    print("rejected:", e)
big = front >> (dsp.fft >> dsp.power_db | dsp.rms | dsp.buffer(count=1) >> linalg.norm)
big.draw()""")
code("""try:
    dsp.sine() >> energy >> scale      # scalar cannot feed a vector
except TypeMismatch as e:
    print("rejected:", e)

try:
    (dsp.sine(n=64) & dsp.sine(n=128)) >> linalg.dot   # dimensions derived from config
except TypeMismatch as e:
    print("rejected:", e)""")

md("""## 3. Run with live progress; inspect frames, annotations and provenance""")
code("""frames = run(g, progress=True)
f = frames[-1]
f""")
code("""print(f.trace())
print(dict(f.stream), dict(f.meta))""")
code("""g.last_monitor""")

md("""## 4. Annotations at three levels, and errors that travel like annotations
A frame-aware block (`Framed[...]`) can add frame-level metadata, per-sample masks and warnings.
A block that raises produces a *failed* frame which flows downstream untouched, keeping its lineage.""")
code("""@block
def flaky(f: Framed[Sig[float, N]], every: int = 3) -> Sig[float, N]:
    if f.meta["seq"] % every == 0:
        raise RuntimeError("sensor glitch")
    return f.data

g2 = (dsp.sine(n=64, frames=6, amplitude=1.5)
      >> dsp.detect_clipping(limit=1.0)        # sample-level mask + warning issue
      >> flaky
      >> scale(gain=0.5)
      >> energy)
for fr in run(g2):
    print(fr.ok, fr.data, [str(i) for i in fr.issues][:1], "samples:", list(fr.samples))""")

md("""## 5. Stream blocks are generators; async sources mix freely with sync ones""")
code("""import asyncio

@block
def pairs(xs: Stream[Sig[float, N]]) -> Sig[float, (2, N)]:
    buf = []
    for x in xs:
        buf.append(x)
        if len(buf) == 2:
            yield np.stack(buf); buf = []

@source(total=lambda count, **_: count)
async def commands(count: int = 4) -> Sig[dict]:
    for i in range(count):
        await asyncio.sleep(0.01)
        yield {"gain": i + 1}

@block
def apply(x: Sig[float, (2, N)], cmd: Sig[dict]) -> Sig[float, (2, N)]:
    return x * cmd["gain"]

g3 = ((dsp.sine(n=8, frames=8) >> pairs) & commands()) >> apply >> linalg.norm
out = await g3.arun(progress=True)     # top-level await works in Jupyter; run() works too
print([round(o.data, 2) for o in out])
print(out[0].trace())""")

md("""## 5b. Long-running graphs: start in the background and control them with buttons
`g.start()` returns immediately with a `Run` handle; `run.controls()` shows Pause / Resume / Step / Stop
and a live status table (needs `ipywidgets`; without it you get a self-updating status line and call
`run.pause()` etc. from a cell). Outputs accumulate in `run.frames`.""")
code("""live = dsp.ticker(period=0.1, count=600) >> (lambda d: d["tick"])   # ~1 minute of events
run_ = live.start(paused=True)
run_.controls()""")
code("""run_.step(3)          # let three events through
import time; time.sleep(0.5)
run_.frames[-3:]""")
code("""run_.stop()
run_""")
md("""## 6. Plots are taps; every plot remembers its frames and can be animated""")
code("""spec = plot.spectrum(title="chirp", ylim=(-80, 10))
fall = plot.waterfall(rows=32)
run(dsp.chirp(f0=200, f1=6000, rate=16_000, n=512, frames=24) >> dsp.window >> spec >> dsp.to_complex >> fall)
fall.figure""")
code("""spec.html(interval=80)      # JavaScript animation player""")

md("""## 6b. `[]` selects ports, annotations and ancestors
Tuple outputs (fan-out, zip, multi-output blocks) expose ports; frames and provenance are indexable too.""")
code("""g = dsp.sine(n=256, frames=3) >> (dsp.rms | dsp.energy | dsp.peak)["energy"]
f = run(g)[-1]
print(f.data, "| rate:", f["rate"], "| ancestors:", [n.op for n in f.prov.chain()], "| source seq:", f.prov["sine"].seq)
g.draw()""")
code("""@source
def spd() -> Sig[float, (3, 3)]:
    a = np.random.default_rng(0).normal(size=(3, 3)); yield a @ a.T

eigh = block(np.linalg.eigh, inputs=Sig[T, (N, N)],
             outputs={"vals": Sig[float, N], "vecs": Sig[T, (N, N)]})
try:
    spd() >> eigh["vals"] >> linalg.transpose      # a vector is not a matrix: rejected at build time
except TypeMismatch as e:
    print("rejected:", e)
run(spd() >> eigh["vals"])[0]""")
md("""## 7. Linear algebra with explicit dimension contracts""")
code("""@source
def snapshots(channels: int = 4, samples: int = 128, frames: int = 2) -> Sig[complex, (M, N)]:
    rng = np.random.default_rng(0)
    steer = np.exp(1j * np.pi * np.arange(channels) * np.sin(np.deg2rad(20)))
    for _ in range(frames):
        s = np.exp(2j * np.pi * 0.05 * np.arange(samples))
        yield np.outer(s, steer) + 0.3 * (rng.normal(size=(samples, channels)) + 1j * rng.normal(size=(samples, channels)))

cov = snapshots() >> linalg.covariance >> linalg.eigh
for fr in run(cov):
    print(np.round(fr.data[0], 2))
cov""")

nb["cells"] = cells
nb.metadata["kernelspec"] = {
    "name": "python3",
    "display_name": "Python 3",
    "language": "python",
}
nbf.write(nb, "sigflow_tour.ipynb")
print("wrote sigflow_tour.ipynb")
