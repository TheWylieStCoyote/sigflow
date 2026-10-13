"""Pulse-Doppler radar: pulses -> range compression -> Doppler FFT -> CFAR -> tracks.

What this exercises
  * 2-D frames with symbolic dimensions ("P" pulses x "R" range cells), checked
    across the whole chain at build time
  * reusing NumPy FFT routines on matrices via block(...) wrappers
  * a frame-aware block that attaches detections as frame metadata and a
    sample-level mask over the range-Doppler map
  * a custom plot (plot.custom) drawing the map with detections overlaid
  * fan-out: the same map feeds a plot, a detector and a summary statistic, and
    `[1:]` selects the ports worth keeping
"""

import numpy as np

from sigflow import Frame, Framed, Sig, Stream, block, run, source
from sigflow import plot

P, R = 64, 256  # pulses per coherent interval, range cells
PRF = 5000.0  # Hz  (unambiguous velocity +-37.5 m/s)
RANGE_RES = 15.0  # metres per cell
WAVELENGTH = 0.03  # 10 GHz


@source(total=lambda cpis, **_: cpis, refine=lambda **_: Sig[complex, (P, R)])
def radar_returns(
    cpis: int = 12, snr_db: float = 12.0, seed: int = 3
) -> Sig[complex, ("P", "R")]:
    """Raw (pulse x range) data cubes with two moving targets and clutter at zero Doppler."""
    rng = np.random.default_rng(seed)
    targets = [
        dict(r=60, v=18.0, a=1.0),
        dict(r=140, v=-26.0, a=0.6),
    ]  # cell, m/s, amplitude
    noise = 10 ** (-snr_db / 20)
    t_cpi = P / PRF
    for k in range(cpis):
        cube = (
            noise
            * (rng.normal(size=(P, R)) + 1j * rng.normal(size=(P, R)))
            / np.sqrt(2)
        )
        cube[:, 20:40] += 3.0 * np.exp(
            1j * rng.uniform(0, 2 * np.pi, (1, 20))
        )  # stationary ground clutter
        for tg in targets:
            cell = int(
                round(tg["r"] + tg["v"] * k * t_cpi / RANGE_RES)
            )  # target migrates
            fd = 2 * tg["v"] / WAVELENGTH
            cube[:, cell] += tg["a"] * np.exp(2j * np.pi * fd * np.arange(P) / PRF)
        yield Frame(
            cube,
            meta={"cpi": k, "t": k * t_cpi},
            stream={"prf": PRF, "range_res": RANGE_RES},
        )


@block
def pulse_compress(cube: Sig[complex, ("P", "R")]) -> Sig[complex, ("P", "R")]:
    """Stand-in for matched filtering along range (here a 3-cell smoothing kernel)."""
    k = np.array([0.25, 0.5, 0.25])
    return np.apply_along_axis(lambda row: np.convolve(row, k, mode="same"), 1, cube)


@block
def mti(cube: Sig[complex, ("P", "R")]) -> Sig[complex, ("P", "R")]:
    """Two-pulse canceller: removes stationary clutter (zero Doppler)."""
    out = np.zeros_like(cube)
    out[1:] = cube[1:] - cube[:-1]
    return out


# Doppler processing: window along pulses, FFT across pulses, log power.
doppler_fft = block(
    lambda c: np.fft.fftshift(np.fft.fft(c * np.hanning(P)[:, None], axis=0), axes=0),
    name="doppler_fft",
    inputs=Sig[complex, ("P", "R")],
    outputs=Sig[complex, ("P", "R")],
)


@block
def power_db(x: Sig[complex, ("P", "R")]) -> Sig[float, ("P", "R"), None, "dB"]:
    return 20 * np.log10(np.abs(x) + 1e-9)


@block
def cfar(
    f: Framed[Sig[float, ("P", "R")]],
    guard: int = 2,
    train: int = 8,
    margin_db: float = 12.0,
) -> Sig[float, ("P", "R")]:
    """Cell-averaging CFAR along range with peak picking. Returns the map unchanged, with
    detections in meta and a boolean 'detected' mask over the map (sample level)."""
    m = f.data
    lin = 10 ** (m / 10)
    k = np.ones(2 * (guard + train) + 1)
    k[train : train + 2 * guard + 1] = 0
    noise = np.apply_along_axis(
        lambda r: np.convolve(r, k / k.sum(), mode="same"), 1, lin
    )
    mask = m > 10 * np.log10(noise) + margin_db
    mask[:, : train + guard] = mask[:, -(train + guard) :] = False
    # keep only local maxima so one target yields one detection, not a blob
    pad = np.pad(m, 1, constant_values=-np.inf)
    neigh = np.stack(
        [
            pad[1 + dp : 1 + dp + P, 1 + dr : 1 + dr + R]
            for dp in (-1, 0, 1)
            for dr in (-1, 0, 1)
        ]
    )
    mask &= m >= neigh.max(axis=0)
    dets = []
    for p, r in zip(*np.nonzero(mask)):
        fd = (p - P / 2) * f.stream["prf"] / P
        dets.append(
            {
                "range_m": r * f.stream["range_res"],
                "velocity_mps": fd * WAVELENGTH / 2,
                "snr_db": float(m[p, r] - 10 * np.log10(noise[p, r])),
            }
        )
    out = f.annotate(detections=dets, n_detections=len(dets)).annotate_samples(
        detected=mask
    )
    if len(dets) > 20:
        out = out.with_issue(
            f"{len(dets)} detections: CFAR threshold probably too low",
            "warning",
            "cfar",
        )
    return out


@block
def strongest(f: Framed[Sig[float, ("P", "R")]]) -> Sig[dict]:
    dets = f.meta.get("detections", [])
    best = max(dets, key=lambda d: d["snr_db"]) if dets else None
    return {"cpi": f.meta["cpi"], "count": len(dets), "best": best}


@block
def tracker(reports: Stream[Sig[dict]], gate_mps: float = 5.0) -> Sig[dict]:
    """Nearest-neighbour association of detections into tracks across CPIs (stateful)."""
    tracks: list[dict] = []
    for rep in reports:
        for d in rep["dets"]:
            for t in tracks:
                if (
                    abs(t["velocity_mps"] - d["velocity_mps"]) < gate_mps
                    and abs(t["range_m"] - d["range_m"]) < 60
                ):
                    t.update(
                        range_m=d["range_m"],
                        velocity_mps=0.8 * t["velocity_mps"] + 0.2 * d["velocity_mps"],
                        hits=t["hits"] + 1,
                    )
                    break
            else:
                tracks.append({**d, "hits": 1, "id": len(tracks) + 1})
        yield {"cpi": rep["cpi"], "tracks": [dict(t) for t in tracks if t["hits"] >= 2]}


@block
def reports(f: Framed[Sig[float, ("P", "R")]]) -> Sig[dict]:
    return {"cpi": f.meta["cpi"], "dets": f.meta["detections"]}


def draw_map(ax, m, f: Frame) -> None:
    prf = f.stream["prf"]
    ext = (
        0,
        m.shape[1] * f.stream["range_res"] / 1000,
        -prf / 2 * WAVELENGTH / 2,
        prf / 2 * WAVELENGTH / 2,
    )
    ax.imshow(
        m, aspect="auto", origin="lower", extent=ext, cmap="magma", vmin=-20, vmax=25
    )
    ax.set_xlabel("range [km]")
    ax.set_ylabel("radial velocity [m/s]")
    if "detected" in f.samples:
        p, r = np.nonzero(f.samples["detected"])
        ax.scatter(
            r * f.stream["range_res"] / 1000,
            (p - P / 2) * prf / P * WAVELENGTH / 2,
            s=60,
            facecolors="none",
            edgecolors="cyan",
        )
    ax.set_title(
        f"CPI {f.meta.get('cpi')}  ({f.meta.get('n_detections', 0)} detections)"
    )


if __name__ == "__main__":
    rd_map = plot.custom(
        draw_map, name="range_doppler", in_type=Sig[float, ("P", "R")], figsize=(7, 4)
    )

    processing = (
        radar_returns() >> pulse_compress >> mti >> doppler_fft >> power_db >> cfar
    )
    g = (
        processing >> (rd_map | strongest | reports >> tracker)[1:]
    )  # keep the two report ports
    print(g.describe(), "\n")

    out = run(g, progress=True)
    for f in out:
        best, trk = f.data
        best_s = (
            f"{best['best']['range_m']:.0f} m @ {best['best']['velocity_mps']:+.1f} m/s"
            if best["best"]
            else "none"
        )
        print(
            f"CPI {best['cpi']:2d}: {best['count']} detections, strongest {best_s:>22}, "
            f"tracks: "
            + ", ".join(
                f"#{t['id']} {t['range_m']:.0f}m {t['velocity_mps']:+.1f}m/s"
                for t in trk["tracks"]
            )
        )
    print("\nwarnings:", [str(i) for i in out[-1].warnings] or "none")
    print("saved", rd_map.save("range_doppler.gif", interval=250))
