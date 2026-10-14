# Analysing a recording interactively: graphs without sources, applied to slices

`examples/16_recorded_analysis.py`

The live examples push data *through* graphs. Analysis is the other way round:
load a capture once, build a source-less graph, and apply it to whichever
slice you are looking at — `front_end(frames[100:120])` — as a plain call.
In a notebook, bind the slice index to a slider and the plot updates as you
scrub (the last section shows the pattern; it works headless too).

What this exercises
  * pipelines as functions on in-memory data: `g(x)`, `g([x1, x2])`, `g(frames)`,
    and `block.apply(frames)` for a single block (plain `block(x)` calls the function)
  * re-using the *same* graph for a live run and for post-hoc analysis
  * slicing a recording into windows with `dsp.overlap` applied offline
  * provenance on offline results points at the in-memory source and offset
  * a scrubber built from `Knob` + `on_change` driving a plot block

## Source

```python
"""Analysing a recording interactively: graphs without sources, applied to slices.

The live examples push data *through* graphs. Analysis is the other way round:
load a capture once, build a source-less graph, and apply it to whichever
slice you are looking at — `front_end(frames[100:120])` — as a plain call.
In a notebook, bind the slice index to a slider and the plot updates as you
scrub (the last section shows the pattern; it works headless too).

What this exercises
  * pipelines as functions on in-memory data: `g(x)`, `g([x1, x2])`, `g(frames)`,
    and `block.apply(frames)` for a single block (plain `block(x)` calls the function)
  * re-using the *same* graph for a live run and for post-hoc analysis
  * slicing a recording into windows with `dsp.overlap` applied offline
  * provenance on offline results points at the in-memory source and offset
  * a scrubber built from `Knob` + `on_change` driving a plot block
"""
import numpy as np

from sigflow import Frame, Knob, Sig, block, from_iterable, run, source
from sigflow import dsp, plot

RATE = 8_000.0
N = 512


# ------------------------------------------------------------ a recording

def record(seconds: float = 6.0, seed: int = 0) -> np.ndarray:
    """A capture: a tone that hops frequency, with a burst of interference."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    hops = np.where(t < 2, 500, np.where(t < 4, 1200, 900))
    x = np.sin(2 * np.pi * np.cumsum(hops) / RATE) + 0.02 * rng.normal(size=len(t))
    i0, i1 = int(3.2 * RATE), int(3.6 * RATE)
    x[i0:i1] += 0.8 * np.sin(2 * np.pi * 2500 * t[i0:i1])
    return x


# ------------------------------------------------------- the analysis graph

@block
def dominant(x: Sig[float, "N"], rate: float = RATE) -> Sig[dict]:
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1 / rate)
    k = int(np.argmax(X[1:])) + 1
    second = np.sort(X)[-6]                                  # rough "second peak" level
    return {"freq_hz": float(f[k]), "clean": bool(X[k] > 4 * second)}


front_end = dsp.window >> dsp.to_complex >> dsp.fft >> dsp.power_db     # no source: a function of frames
features = front_end >> (lambda p: {"peak_db": float(p.max()), "mean_db": float(p.mean())})


if __name__ == "__main__":
    x = record()
    windows = run(dsp.from_array(x, n=N, rate=RATE) >> dsp.overlap(n=N, hop=N // 2))   # offline re-framing
    print(f"{len(windows)} windows of {N} from a {len(x) / RATE:.0f} s recording\n")

    # 1. apply graphs as functions to any slice of the recording
    print("one window      :", front_end(windows[10].data)[0].data.shape)
    print("a list of arrays:", [round(f.data["peak_db"], 1) for f in features([w.data for w in windows[:3]])])
    print("frames in, frames out (annotations kept):", features(windows[:2])[1].meta)

    # 2. the same analysis block used live and offline
    live = run(dsp.from_array(x, n=N, rate=RATE) >> dominant)
    offline = dominant.apply(windows)                         # a block over frames (keeps annotations)
    print("\nlive  :", [round(f.data["freq_hz"]) for f in live[::12]])
    print("offline:", [round(f.data["freq_hz"]) for f in offline[::24]])

    # 3. ask questions of the recording
    hops = [(round(w["t0"] if "t0" in w else i * N / 2 / RATE, 2), d.data["freq_hz"])
            for i, (w, d) in enumerate(zip(windows, offline))]
    changes = [(t, f) for (t0, f0), (t, f) in zip(hops, hops[1:]) if abs(f - f0) > 50]
    print("\nfrequency changes at (s, Hz):", changes[:6])
    dirty = [i for i, d in enumerate(offline) if not d.data["clean"]]
    print(f"windows with interference: {dirty[:3]}... ({len(dirty)} total) "
          f"= {dirty[0] * N / 2 / RATE:.2f}-{dirty[-1] * N / 2 / RATE:.2f} s")
    print("offline result provenance:", [n.op for n in offline[dirty[0]].prov.chain()],
          "offset", offline[dirty[0]]["offset"] if "offset" in offline[dirty[0]] else "(from overlap)")

    # 4. a scrubber: a Knob selects the window, a plot block redraws on change.
    #    In a notebook: `display(idx.widget())` gives the slider and `spec.figure` the picture
    #    (or put both in a VBox); here we just move it programmatically.
    idx = Knob(0, "window", min=0, max=len(windows) - 1, step=1)
    spec = plot.spectrum(title="scrub", ylim=(-80, 10))

    def show(i: int) -> None:
        spec.redraw(windows[int(i)].annotate(seq=int(i), t0=int(i) * N / 2 / RATE))
        spec.frames.append(windows[int(i)])

    idx.on_change(show)
    for i in (5, 40, 55, 80):
        idx.value = i
    print("\nscrubbed", len(spec.frames), "windows; saved", spec.save("scrub.gif", interval=400))
```
