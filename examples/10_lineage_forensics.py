"""Lineage forensics: answering "where did this bad number come from?" with `[]`.

A small monitoring pipeline fuses three sensor channels. One channel has a
calibration that drifts, another drops out intermittently, and a downstream
block occasionally fails. Afterwards we use frame and provenance indexing to
attribute every anomaly to its source frame, block and configuration —
without having instrumented anything up front.

What this exercises
  * `f["key"]` annotation lookup across levels, `f.get`, `"key" in f`
  * `f.prov["name"]` to reach a specific ancestor and read its `.params` / `.seq`
  * `f.prov[-1]`, `f.prov.chain()`, `f.prov.roots()` on frames that merge several sources
  * `f[i]` to open a tuple frame's parts and compare their lineages
  * failed frames: their provenance and the issue's `prov` point at the exact spot
  * a helper that groups anomalies by (block, config, source frame) — a report
    built purely from lineage
"""
from collections import Counter

import numpy as np

from sigflow import MISSING, Frame, Framed, Sig, block, port, run, source
from sigflow import dsp


@source(total=lambda frames, **_: frames)
def sensor(name: str = "s", frames: int = 40, scale: float = 1.0, drift: float = 0.0,
           dropout_every: int = 0, seed: int = 0) -> Sig[float, (64,)]:
    """A sensor with per-instance configuration that will show up in provenance."""
    rng = np.random.default_rng(seed)
    for k in range(frames):
        if dropout_every and k % dropout_every == dropout_every - 1:
            raise_on = True
        else:
            raise_on = False
        x = scale * (1 + drift * k) * (np.sin(2 * np.pi * 5 * np.arange(64) / 64) + 0.05 * rng.normal(size=64))
        if raise_on:
            x = np.full(64, np.nan)                  # a dropout: the hardware returned garbage
        yield Frame(x, meta={"k": k, "sensor": name}, stream={"sensor": name, "rate": 1000.0})


@block
def validate(f: Framed[Sig[float, (64,)]]) -> Sig[float, (64,)]:
    if np.isnan(f.data).any():
        raise ValueError(f"{f['sensor']} frame {f['k']}: NaN samples")
    return f.annotate(level=float(np.sqrt(np.mean(f.data ** 2))))


@block
def fuse(a: Framed[Sig[float, (64,)]], b: Framed[Sig[float, (64,)]], c: Framed[Sig[float, (64,)]]) -> Sig[dict]:
    """Combine three channels; flag when one disagrees with the other two."""
    levels = np.array([a["level"], b["level"], c["level"]])
    med = float(np.median(levels))
    off = [name for name, lv in zip(("a", "b", "c"), levels) if abs(lv - med) / med > 0.25]
    return {"k": a["k"], "level": med, "disagree": off}


@block
def alarm(d: Sig[dict], limit: float = 0.5) -> Sig[dict]:
    if d["level"] > limit * 2:
        raise RuntimeError("level out of range")      # a downstream failure to attribute later
    return {**d, "alarm": bool(d["disagree"])}


def attribute(frame: Frame) -> str:
    """One line saying what happened and which source frame/configuration is behind it."""
    if frame.failed:
        issue = frame.errors[0]
        where = issue.prov            # the provenance node *at* the failure
        root = where[-1]              # ... and the source frame it descends from
        return (f"FAILED in {issue.origin!r} ({issue.message}) <- {root.op}(name={root.params.get('name')!r}, "
                f"drift={root.params.get('drift')}, dropout_every={root.params.get('dropout_every')}) frame #{root.seq}")
    d = frame.data
    if d["alarm"]:
        # roots() reaches every source frame behind this output; pick the disagreeing sensors'
        # configuration straight out of their provenance
        culprits = [r for r in frame.prov.roots() if r.params.get("name") in d["disagree"]]
        cfg = ", ".join(f"{r.params['name']}(drift={r.params['drift']}, frame #{r.seq})" for r in culprits)
        return f"ALARM k={d['k']}: {cfg} disagrees with the others"
    return f"ok    k={d['k']} level={d['level']:.3f}"


if __name__ == "__main__":
    channels = (sensor(name="a", seed=1)
                & sensor(name="b", seed=2, drift=0.02)              # calibration drifting 2 % per frame
                & sensor(name="c", seed=3, dropout_every=9))        # hardware dropout every 9th frame
    # Each channel is validated independently: a fan-out would send the *tuple* to all three,
    # so select each port and validate it, then zip the validated channels back together.
    checked = channels >> (port(0) >> validate | port(1) >> validate | port(2) >> validate)
    g = checked >> fuse >> alarm
    print(g.describe(), "\n")

    out = run(g, progress=True)

    print("\n--- per-frame attribution (first 14) ---")
    for f in out[:14]:
        print(attribute(f))

    print("\n--- anomalies grouped by cause ---")
    causes: Counter = Counter()
    for f in out:
        if f.failed:
            issue = f.errors[0]
            root = issue.prov[-1]                                   # source behind the failure
            causes[(issue.origin, root.params.get("name"), issue.message.split(":")[0])] += 1
        elif f.data["alarm"]:
            causes[("fuse", tuple(f.data["disagree"]), "disagreement")] += 1
    for (where, who, what), n in causes.most_common():
        print(f"{n:3d} x  {what:<14} at {where:<9} sensor {who}")

    # Opening a tuple frame: the validated channels before fusion, each with its own lineage
    first = run(checked)[0]
    for i in range(3):
        part = first[i]
        print(f"\nport {i}: sensor {part['sensor']!r} level {part['level']:.3f}; "
              f"lineage {' <- '.join(n.op for n in part.prov.chain())}; "
              f"source config {dict(part.prov[-1].params)}")

    # Which source frames fed a given output?  roots() walks every path, not just the main line.
    f = out[5]
    print("\noutput k=5 descends from:", [(r.params["name"], r.seq) for r in f.prov.roots()])
    print("the 'b' sensor's drift setting at that point:", f.prov.roots()[1].params["drift"])
    print("ancestors on the main line:", [n.op for n in f.prov.chain()], "| 'fuse' in lineage:", "fuse" in f.prov)
