# Documenting a graph for a colleague: `g.report()`

`examples/19_graph_report.py`

"What does this pipeline do?" answered from the graph itself: the diagram, the
typed node table with resolved dimensions and configuration, and every block's
docstring — as HTML (self-contained, diagram inline) or Markdown. Nothing is
written by hand; docstrings and type annotations on the blocks *are* the docs.

What this exercises
  * `g.report()` / `g.report(fmt="md")` and `g.draw().save()` / `g.to_dot()`
  * docstrings on blocks (including wrapped library functions via `doc=`)
  * reports for sub-graphs, feedback loops and routed graphs
  * a tiny "docs site": one HTML page per graph plus an index, from a script

## Source

```python
"""Documenting a graph for a colleague: `g.report()`.

"What does this pipeline do?" answered from the graph itself: the diagram, the
typed node table with resolved dimensions and configuration, and every block's
docstring — as HTML (self-contained, diagram inline) or Markdown. Nothing is
written by hand; docstrings and type annotations on the blocks *are* the docs.

What this exercises
  * `g.report()` / `g.report(fmt="md")` and `g.draw().save()` / `g.to_dot()`
  * docstrings on blocks (including wrapped library functions via `doc=`)
  * reports for sub-graphs, feedback loops and routed graphs
  * a tiny "docs site": one HTML page per graph plus an index, from a script
"""

from typing import Literal
import os
import tempfile

import numpy as np
import scipy.signal as ss

from sigflow import Framed, Sig, Stream, block, dsp, feedback, linalg, port, run, when
from sigflow.dims import N


@block
def deemphasis(
    x: Sig[float, N], tau_us: float = 50.0, rate: float = 48_000
) -> Sig[float, N]:
    """FM de-emphasis: first-order low-pass with time constant ``tau_us`` microseconds.

    Applied after demodulation to undo the transmitter's pre-emphasis.
    """
    a = np.exp(-1 / (rate * tau_us * 1e-6))
    return ss.lfilter([1 - a], [1, -a], x)


@block
def squelch(f: Framed[Sig[float, N]], threshold_db: float = -40.0) -> Sig[float, N]:
    """Mute frames whose level is below ``threshold_db`` dBFS (tagged ``muted`` in meta)."""
    level = 20 * np.log10(np.sqrt(np.mean(f.data**2)) + 1e-12)
    return f.annotate(muted=level < threshold_db).evolve(
        data=f.data if level >= threshold_db else np.zeros_like(f.data)
    )


hilbert = block(
    ss.hilbert,
    name="hilbert",
    inputs=Sig[float, N],
    outputs=Sig[complex, N],
    doc="Analytic signal via scipy.signal.hilbert (positive-frequency half of the spectrum).",
)


@block
def agc_step(
    x: Sig[float, N], gain: Sig[float, tuple[()]], target: float = 0.3
) -> tuple[Sig[float, N], Sig[float, tuple[()]]]:
    """One AGC iteration: apply ``gain`` and nudge it toward ``target`` rms."""
    y = x * gain
    return y, float(
        gain
        * (
            1
            + 0.5
            * (target - np.sqrt(np.mean(y**2)))
            / max(np.sqrt(np.mean(y**2)), 1e-6)
        )
    )


if __name__ == "__main__":
    out_dir = tempfile.mkdtemp(prefix="sigflow_docs_")

    receiver = (
        dsp.sine(freq=1000, rate=48_000, n=1024, frames=4)
        >> feedback(agc_step(target=0.3), init=0.1, name="agc")
        >> deemphasis(tau_us=75)
        >> when(
            lambda f: f.data.std() < 0.01,
            dsp.scale(gain=0.0),
            otherwise=squelch(threshold_db=-35),
            name="mute_silence",
        )
        >> (dsp.window >> hilbert >> dsp.fft >> dsp.power_db | dsp.rms)["rms"]
    )
    analysis = dsp.buffer(count=8) >> linalg.covariance >> linalg.eigh[0]

    graphs = {"receiver": receiver, "analysis (sub-graph, no source)": analysis}
    index = ["<h1>sigflow graphs</h1><ul>"]
    for name, g in graphs.items():
        html = g.report(title=name)
        md = g.report(title=name, fmt="md")
        slug = name.split(" ")[0]
        with open(os.path.join(out_dir, f"{slug}.html"), "w") as fh:
            fh.write("<!doctype html><meta charset='utf-8'>" + html)
        with open(os.path.join(out_dir, f"{slug}.md"), "w") as fh:
            fh.write(md)
        g.draw().save(os.path.join(out_dir, f"{slug}.svg"))
        with open(os.path.join(out_dir, f"{slug}.dot"), "w") as fh:
            fh.write(g.to_dot())
        index.append(
            f"<li><a href='{slug}.html'>{name}</a> ({len(g.nodes())} blocks)</li>"
        )
        print(
            f"--- {name}: {len(html)} bytes of HTML, {md.count(chr(10))} lines of Markdown"
        )
    with open(os.path.join(out_dir, "index.html"), "w") as fh:
        fh.write("<!doctype html><meta charset='utf-8'>" + "".join(index) + "</ul>")

    print("\nMarkdown report for the receiver (first 40 lines):\n")
    print("\n".join(receiver.report(title="receiver", fmt="md").splitlines()[:40]))
    print(f"\nwritten to {out_dir}: {sorted(os.listdir(out_dir))}")
    print("sanity: the receiver still runs ->", round(run(receiver)[-1].data, 3))
```
