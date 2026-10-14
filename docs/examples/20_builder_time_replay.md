# 0.2 features together: units, slicing, the time axis, an explicit Graph, replay, checkpoints

`examples/20_builder_time_replay.py`

What this exercises
  * units in the type: a millivolt ADC refuses a volts-only block until dsp.to_units converts
  * data slicing `node[a:b]` with static shapes; stream slicing `take/skip/every/until`
  * Frame.time / TimeSpan through re-framing; align() joining two sources by timestamp
  * Graph builder: a diamond where a covariance feeds eigh AND a later combiner
  * frame.save / Frame.load / replay(): reproduce exactly one output frame from disk
  * Stateful blocks with run.checkpoint() and resume=
  * plot.multi and plot.history in one figure

## Source

```python
"""0.2 features together: units, slicing, the time axis, an explicit Graph, replay, checkpoints.

What this exercises
  * units in the type: a millivolt ADC refuses a volts-only block until dsp.to_units converts
  * data slicing `node[a:b]` with static shapes; stream slicing `take/skip/every/until`
  * Frame.time / TimeSpan through re-framing; align() joining two sources by timestamp
  * Graph builder: a diamond where a covariance feeds eigh AND a later combiner
  * frame.save / Frame.load / replay(): reproduce exactly one output frame from disk
  * Stateful blocks with run.checkpoint() and resume=
  * plot.multi and plot.history in one figure
"""
import os
import tempfile
import time

import numpy as np

from sigflow import (Frame, Graph, Sig, Stateful, TypeMismatch, align, block, replay, run, source)
from sigflow import dsp, linalg, plot

WORK = tempfile.mkdtemp(prefix="sigflow_v02_")
RATE = 1000.0


@source(total=lambda frames, **_: frames)
def adc(frames: int = 40, n: int = 100) -> Sig[float, "N", None, "mV"]:
    """A sensor in millivolts with a timestamp on every frame."""
    for k in range(frames):
        t = (np.arange(n) + k * n) / RATE
        x = 800 * np.sin(2 * np.pi * 7 * t) + 20 * np.random.default_rng(k).normal(size=n)
        yield Frame(x, meta={"t0": k * n / RATE, "seq": k}, stream={"rate": RATE, "units": "mV"})


@source
async def gps_fixes(count: int = 10) -> Sig[dict]:
    """Slow, jittery fixes with their own timestamps."""
    rng = np.random.default_rng(3)
    for j in range(count):
        yield Frame({"fix": j}, meta={"t0": j * 0.4 + rng.uniform(-0.01, 0.01)})


@block
def volts_rms(x: Sig[float, "N", None, "V"]) -> Sig[float, ()]:
    return float(np.sqrt(np.mean(x ** 2)))


class Integrator(Stateful):
    """Checkpointable running sum of frame energies."""

    def __init__(self) -> None:
        self.total = 0.0

    def __call__(self, x: Sig[float, "N"]) -> Sig[float, ()]:
        self.total += float(np.sum(x ** 2))
        return self.total


if __name__ == "__main__":
    # 1. units in the contract
    try:
        adc() >> volts_rms
    except TypeMismatch as e:
        print("rejected:", e)
    print("rms (V):", round(run(adc(frames=1) >> dsp.to_units("V") >> volts_rms)[0].data, 3))

    # 2. data + stream slicing, with static shapes and time kept
    chain = adc(frames=10) >> dsp.to_units("V")
    print("sliced type:", chain[10:60].out_type, "| first 3 of every 2nd frame:",
          [f["seq"] for f in run(chain[10:60].every(2).take(3))],
          "| t0 of a slice:", run(chain[10:60])[1]["t0"])

    # 3. the time axis through re-framing, and a timestamp join
    windows = run(adc(frames=5) >> dsp.overlap(n=80, hop=40))
    print("window times:", [round(f["t0"], 3) for f in windows[:4]], "->", windows[1].time)
    joined = run(align(adc(frames=40), gps_fixes(), tolerance=0.05) >> (lambda pair: (pair[0].shape, pair[1]["fix"])))
    print("aligned pairs:", [f.data for f in joined][:4], "| skews:", [round(f["skew"], 3) for f in joined][:4])

    # 4. an explicit graph: the covariance feeds eigh and the combiner (a diamond)
    eigh = block(np.linalg.eigh, name="eigh", inputs=Sig["T", ("N", "N")],
                 outputs={"vals": Sig[float, "N"], "vecs": Sig["T", ("N", "N")]})

    @block
    def summarize(vals: Sig[float, "N"], vecs: Sig["T", ("N", "N")], raw: Sig[float, ("N", "N")]) -> Sig[dict]:
        return {"top": round(float(vals[-1]), 1), "trace": round(float(np.trace(raw)), 1), "n": vecs.shape[0]}

    g = Graph("diamond")
    cov = g.add(adc(frames=16) >> dsp.to_units("V") >> dsp.buffer(count=4) >> linalg.covariance, "cov")
    e = g.add(eigh)
    s = g.add(summarize)
    g.connect(cov, e).connect(e["vals"], s, "vals").connect(e["vecs"], s, "vecs").connect(cov, s, "raw")
    g.output(s)
    print(g.describe())
    out = run(g)
    print("graph output:", out[-1].data, "| lineage roots:", {r.op for r in out[-1].prov.roots()})
    g.draw().save(os.path.join(WORK, "diamond.svg"))

    # 5. save a frame, load it elsewhere, replay it from the same graph
    path = os.path.join(WORK, "frame7.npz")
    chain = adc(frames=12) >> dsp.to_units("V") >> dsp.detect_clipping(limit=0.75) >> dsp.scale(gain=2)
    f7 = run(chain)[7]
    f7.save(path)
    loaded = Frame.load(path)
    again = replay(chain, loaded)
    print("\nreplayed frame equals the saved one:", np.allclose(again.data, loaded.data),
          "| lineage from JSON:", [n.op for n in loaded.prov.chain()],
          "| warnings kept:", [str(w) for w in loaded.warnings][:1])

    # 6. checkpoint a stateful block mid-run, resume in a fresh process-like run
    def integ():    # a Stateful block is an object: build a fresh one per run
        return adc(frames=30) >> dsp.to_units("V") >> block(Integrator(), name="integrate")

    r = integ().start(paused=True)
    r.step(12); time.sleep(0.3)
    ck = r.checkpoint(os.path.join(WORK, "ckpt.pkl"))
    r.stop()
    print(f"\ncheckpoint after {ck.positions['adc']} source frames, integrator total {ck.states['integrate']['total']:.3f}")
    rest = run(integ(), resume=ck)
    print(f"resumed: {len(rest)} frames remained, final total {rest[-1].data:.3f} "
          f"(uninterrupted run: {run(integ())[-1].data:.3f})")

    # 7. several views in one figure
    panel = plot.multi([plot.line(title="volts"), plot.spectrum(title="spectrum"),
                        plot.history(key=lambda f: float(np.abs(f.data).max()), title="peak over time")], cols=3)
    run(adc(frames=20) >> dsp.to_units("V") >> panel)
    print("saved", panel.save(os.path.join(WORK, "panel.gif"), interval=120), "| work dir:", WORK)
```
