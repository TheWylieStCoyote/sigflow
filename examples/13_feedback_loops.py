"""Feedback across blocks: an AGC and an LMS echo canceller built with `feedback()`.

Everything earlier was feed-forward. `feedback(body, init)` closes a loop: `body`
maps a tuple frame `(x, state) -> (y, new_state)` and the runtime feeds
`new_state` back as `state` for the next input frame. The body can be a single
block or a whole pipeline, so the loop can span several blocks — here the LMS
weights are produced by an *update* block and consumed by a *filter* block.

What this exercises
  * feedback() with a one-block body (AGC) and a two-block body (LMS)
  * state typed like any other signal (a scalar gain, a weight vector of symbolic length)
  * loop state in provenance: each output's lineage includes the previous iteration
  * a loop inside a larger graph, zipped inputs (microphone + far-end reference)
"""
import numpy as np

from sigflow import Frame, Framed, Sig, block, feedback, run, source
from sigflow import dsp, plot

RATE = 8_000
N = 256
TAPS = 32


# ------------------------------------------------------------------ AGC

@block
def agc_step(x: Sig[float, "N"], gain: Sig[float, ()], target: float = 0.3,
             attack: float = 0.5) -> (Sig[float, "N"], Sig[float, ()]):
    """One AGC iteration: apply the gain, then move it toward the target level."""
    y = x * gain
    level = float(np.sqrt(np.mean(y ** 2)))
    new_gain = gain * (1 + attack * (target - level) / max(level, 1e-6))
    return y, float(np.clip(new_gain, 1e-3, 1e3))


# ---------------------------------------------------------- LMS canceller

@source(total=lambda frames, **_: frames)
def far_end(frames: int = 60, seed: int = 1) -> Sig[float, "N"]:
    """The far-end (loudspeaker) signal: coloured noise."""
    rng = np.random.default_rng(seed)
    for k in range(frames):
        yield Frame(np.convolve(rng.normal(size=N + 4), np.ones(5) / 5, mode="valid"), meta={"seq": k})


@block
def room(ref: Sig[float, "N"], near_speech: bool = False) -> Sig[float, "N"]:
    """What the microphone hears: the far-end signal through an echo path (plus near-end
    speech now and then). Carries filter state across frames."""
    y, room.zi = __import__("scipy.signal").signal.lfilter(room.h, [1.0], ref, zi=room.zi)
    if near_speech:
        y = y + 0.3 * np.sin(2 * np.pi * 440 * np.arange(N) / RATE)
    return y


room.h = np.array([0.0, 0.0, 0.6, 0.0, -0.3, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 0.05] + [0.0] * (TAPS - 12))
room.zi = np.zeros(TAPS - 1)


@block
def lms_filter(mic: Sig[float, "N"], ref: Sig[float, "N"], w: Sig[float, "W"]) -> (Sig[float, "N"], Sig[dict]):
    """Estimate the echo with the current weights and subtract it. Passes what the
    update block needs (the regressor matrix and the error) along as a dict."""
    R = np.lib.stride_tricks.sliding_window_view(np.concatenate([lms_filter.tail, ref]), len(w))[:N][:, ::-1]
    lms_filter.tail = ref[-(len(w) - 1):]
    echo_hat = R @ w
    err = mic - echo_hat
    return err, {"R": R, "err": err, "w": w}


lms_filter.tail = np.zeros(TAPS - 1)


@block
def lms_update(err: Sig[float, "N"], aux: Sig[dict], mu: float = 0.02) -> (Sig[float, "N"], Sig[float, "W"]):
    """Normalised LMS: new weights from the regressors and the error — the state that
    flows back around the loop to lms_filter on the next frame."""
    R, e, w = aux["R"], aux["err"], aux["w"]
    norm = np.sum(R ** 2, axis=1) + 1e-6
    w_new = w + mu * (R.T @ (e / norm))
    lms_update.last_w = w_new                 # keep the latest weights visible from outside the loop
    return err, w_new


lms_update.last_w = None


@block
def split_inputs(pair: (Sig[float, "N"], Sig[float, "N"]), w: Sig[float, "W"]) -> (Sig[float, "N"], Sig[float, "N"], Sig[float, "W"]):
    """The loop input is ((mic, ref), w); flatten it for lms_filter's three parameters."""
    mic, ref = pair
    return mic, ref, w


@block
def erle(err: Sig[float, "N"], mic: Sig[float, "N"]) -> Sig[float, ()]:
    """Echo return loss enhancement in dB: how much of the microphone signal was removed."""
    return float(10 * np.log10(np.mean(mic ** 2) / (np.mean(err ** 2) + 1e-12)))


if __name__ == "__main__":
    # --- AGC: a one-block loop -----------------------------------------------
    agc = feedback(agc_step(target=0.3, attack=0.5), init=0.05)
    g = dsp.sine(n=N, frames=25, amplitude=2.0) >> agc >> dsp.rms
    print(g.describe(), "\n")
    levels = [round(f.data, 3) for f in run(g)]
    print("AGC output level per frame:", levels[:4], "...", levels[-3:])
    f = run(g)[-1]
    print("loop state in lineage:", [n.op for n in f.prov.chain()[:4]], "| state 3 iterations back:",
          f.prov["loop"].parents[1].op if len(f.prov["loop"].parents) > 1 else "(init)")

    # --- LMS echo canceller: a two-block loop (filter -> update -> back to filter) ------
    weights0 = np.zeros(TAPS)
    canceller = feedback(split_inputs >> lms_filter >> lms_update(mu=0.05), init=weights0, name="lms_loop")
    print(canceller.describe(), "\n")

    mic = far_end() >> room                                  # what the microphone hears
    ref = far_end()                                          # the same far-end signal, as the reference
    g = (mic & ref) >> (canceller | (lambda pair: pair[0])) >> erle
    improvement = plot.custom(lambda ax, v, f: (ax.bar([0], [v]), ax.set_ylim(0, 40), ax.set_ylabel("ERLE [dB]"),
                                                ax.set_title(f"frame {f.prov[-1].seq}")),
                              name="erle_plot", in_type=Sig[float, ()], figsize=(3, 3))
    out = run(g >> improvement, progress=True)
    e = [round(f.data, 1) for f in out]
    print("ERLE per frame (dB):", e[:5], "...", e[-5:])
    print("learned taps :", np.round(lms_update.last_w[:8], 2))
    print("room response:", np.round(room.h[:8], 2))
    print("lineage of the last ERLE value:", [n.op for n in out[-1].prov.chain()])
    print("saved", improvement.save("erle.gif", interval=100))
