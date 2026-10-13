# Sensor fusion: a Kalman filter fed by a fast synchronous IMU stream merged with

`examples/06_sensor_fusion.py`

slow, irregular, asynchronous GPS fixes — plus a flaky sensor to show error flow.

What this exercises
  * merge() of a blocking synchronous source (threaded=True) and an async
    source into one event stream of heterogeneous Python objects (Sig[dict])
  * a stateful stream block (the filter) that dispatches on the event kind and
    uses the linear-algebra primitives directly (predict / update)
  * error propagation: the GPS parser raises on a corrupt fix; the failed frame
    travels on with its provenance and the filter simply never sees it
  * frame annotations (innovation per update) and an animation built after
    the run from the recorded frames (plot.animate)

## Source

```python
"""Sensor fusion: a Kalman filter fed by a fast synchronous IMU stream merged with
slow, irregular, asynchronous GPS fixes — plus a flaky sensor to show error flow.

What this exercises
  * merge() of a blocking synchronous source (threaded=True) and an async
    source into one event stream of heterogeneous Python objects (Sig[dict])
  * a stateful stream block (the filter) that dispatches on the event kind and
    uses the linear-algebra primitives directly (predict / update)
  * error propagation: the GPS parser raises on a corrupt fix; the failed frame
    travels on with its provenance and the filter simply never sees it
  * frame annotations (innovation per update) and an animation built after
    the run from the recorded frames (plot.animate)
"""
import asyncio
import math
import time

import numpy as np

from sigflow import Frame, Framed, Sig, Stream, block, merge, run, source
from sigflow import plot

IMU_RATE = 50.0          # Hz, synchronous
GPS_PERIOD = 0.4         # s, asynchronous and jittery
DURATION = 6.0           # s


def true_state(t: float) -> tuple[float, float, float, float]:
    """Ground truth: a vehicle on a gentle S-curve. Returns x, y, vx, vy."""
    x, y = 4.0 * t, 2.0 * math.sin(0.6 * t)
    vx, vy = 4.0, 1.2 * math.cos(0.6 * t)
    return x, y, vx, vy


SPEEDUP = 20.0           # run the simulation 20x faster than real time
_clock = {"t0": None}


def sim_time() -> float:
    """One shared clock for every sensor (sim seconds). Sensors that stamp from
    independent counters drift apart and bias the fused velocity — try it."""
    now = time.perf_counter()
    if _clock["t0"] is None:
        _clock["t0"] = now
    return (now - _clock["t0"]) * SPEEDUP


@source(total=lambda duration, rate, **_: int(duration * rate), threaded=True)
def imu(duration: float = DURATION, rate: float = IMU_RATE, sigma: float = 0.3, seed: int = 1) -> Sig[dict]:
    """Synchronous accelerometer stream, paced like a blocking device read.
    ``threaded=True`` keeps its time.sleep() from stalling the async GPS source."""
    rng = np.random.default_rng(seed)
    for k in range(int(duration * rate)):
        time.sleep(1 / rate / SPEEDUP)
        t = sim_time()
        ax, ay = 0.0, -0.72 * math.sin(0.6 * t)          # d/dt of the true velocity
        yield {"kind": "imu", "t": t, "acc": np.array([ax, ay]) + rng.normal(0, sigma, 2)}


@source
async def gps(duration: float = DURATION, period: float = GPS_PERIOD, sigma: float = 0.8,
              corrupt_every: int = 5, seed: int = 2) -> Sig[dict]:
    """Asynchronous position fixes with jitter; every ``corrupt_every``-th message is garbage."""
    rng = np.random.default_rng(seed)
    k = 0
    t = 0.0
    while t < duration:
        await asyncio.sleep(period * rng.uniform(0.7, 1.3) / SPEEDUP)   # irregular arrival
        t = sim_time()
        x, y, *_ = true_state(t)
        raw = "GARBAGE" if (k + 1) % corrupt_every == 0 else f"{x + rng.normal(0, sigma):.3f},{y + rng.normal(0, sigma):.3f}"
        k += 1
        yield {"kind": "gps", "t": t, "raw": raw}


@block
def parse(ev: Sig[dict]) -> Sig[dict]:
    """Decode raw GPS text; IMU events pass through. Raises on corrupt input."""
    if ev["kind"] == "gps":
        x, y = (float(v) for v in ev["raw"].split(","))      # ValueError on "GARBAGE"
        return {"kind": "gps", "t": ev["t"], "pos": np.array([x, y])}
    return ev


@block
def kalman(events: Stream[Framed[Sig[dict]]], q: float = 1.0, r_gps: float = 0.8) -> Sig[dict]:
    """Constant-velocity Kalman filter. IMU events predict; GPS events update.
    Emits the state after every event, annotated with the innovation on updates."""
    x = np.zeros(4)                                       # [px, py, vx, vy]
    P = np.eye(4) * 10.0
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    R = np.eye(2) * r_gps ** 2
    t_last = None
    for f in events:
        ev = f.data
        t = ev["t"]
        if ev["kind"] == "imu":
            dt = 0.0 if t_last is None else t - t_last    # the IMU clock drives prediction;
            t_last = t                                    # GPS fixes are applied as they arrive
            F = np.eye(4); F[0, 2] = F[1, 3] = dt
            B = np.array([[0.5 * dt ** 2, 0], [0, 0.5 * dt ** 2], [dt, 0], [0, dt]])
            Q = np.zeros((4, 4))                          # white-acceleration process noise
            for i in range(2):
                Q[i, i], Q[i + 2, i + 2] = dt ** 3 / 3, dt
                Q[i, i + 2] = Q[i + 2, i] = dt ** 2 / 2
            x = F @ x + B @ ev["acc"]
            P = F @ P @ F.T + Q * q ** 2
            yield f.evolve(data={"t": t, "pos": x[:2].copy(), "vel": x[2:].copy(), "step": "predict"})
        else:
            innov = ev["pos"] - H @ x
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.solve(S, np.eye(2))     # P Hᵀ S⁻¹ without an explicit inverse
            x = x + K @ innov
            P = (np.eye(4) - K @ H) @ P
            yield f.evolve(data={"t": t, "pos": x[:2].copy(), "vel": x[2:].copy(), "step": "update"},
                           meta={"innovation": float(np.linalg.norm(innov))})


@block
def error_vs_truth(s: Sig[dict]) -> Sig[dict]:
    tx, ty, *_ = true_state(s["t"])
    return {**s, "err": float(math.hypot(s["pos"][0] - tx, s["pos"][1] - ty))}


def draw_track(ax, s, f: Frame) -> None:
    """Draw the estimate path up to this frame (used after the run, via plot.animate)."""
    hist = draw_track.history
    hist.append((s["pos"][0], s["pos"][1], s["step"]))
    ts = np.linspace(0, DURATION, 200)
    truth = np.array([true_state(t)[:2] for t in ts])
    ax.plot(truth[:, 0], truth[:, 1], "k--", lw=1, label="truth")
    ax.plot([h[0] for h in hist], [h[1] for h in hist], color="#2563eb", lw=1.2, label="estimate")
    ups = [(h[0], h[1]) for h in hist if h[2] == "update"]
    if ups:
        ax.scatter(*zip(*ups), s=18, color="#dc2626", zorder=3, label="GPS update")
    ax.set_xlim(-1, 26); ax.set_ylim(-4, 4); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"t = {s['t']:.2f} s   error = {s['err']:.2f} m")


draw_track.history = []


if __name__ == "__main__":
    _clock["t0"] = None
    g = merge(imu(), gps()) >> parse >> kalman >> error_vs_truth
    print(g.describe(), "\n")

    # The sources pace themselves on the wall clock, so keep the graph lean while it
    # runs (no drawing inside); animate the recorded frames afterwards.
    out = run(g, progress=True)
    good = [f for f in out if f.ok]
    failed = [f for f in out if f.failed]
    print(f"\n{len(good)} state estimates, {len(failed)} failed events")
    if failed:
        print("example failure:", failed[0].errors[0], "\n   lineage:", " <- ".join(failed[0].prov.ops()))
    updates = [f for f in good if f.data["step"] == "update"]
    print(f"{len(updates)} GPS updates; mean innovation {np.mean([f.meta['innovation'] for f in updates]):.2f} m")
    print(f"position error: first second {np.mean([f.data['err'] for f in good if f.data['t'] < 1]):.2f} m, "
          f"last second {np.mean([f.data['err'] for f in good if f.data['t'] > DURATION - 1]):.2f} m")

    shown = [f for i, f in enumerate(good) if i % 3 == 0 or f.data["step"] == "update"]
    anim = plot.animate(shown, draw_track, interval=60, figsize=(7, 3.6))
    anim.save("fusion_track.gif", writer="pillow")
    print("saved fusion_track.gif")
```
