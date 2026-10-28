# Direction finding with a sensor array: MUSIC from the eigen-decomposition's ports

`examples/09_subspace_doa.py`

What this exercises
  * output ports: `port("vals")` / `port("vecs")` route the two results of one
    decomposition to different consumers that later rejoin, each port with its
    own type contract; `decomposition["vals"]` for a direct selection
  * named ports declared on a wrapped NumPy function (`outputs={...}`)
  * port selection inside a fan-out (`port(1)` re-exports the model order)
  * shape symbols flowing through: (M snapshots x N sensors) -> (N x N) -> (N, K)
  * provenance of a result that merges several ports: `f.prov["eigh"]`, `f[1]`

## Source

```python
"""Direction finding with a sensor array: MUSIC from the eigen-decomposition's ports.

What this exercises
  * output ports: `port("vals")` / `port("vecs")` route the two results of one
    decomposition to different consumers that later rejoin, each port with its
    own type contract; `decomposition["vals"]` for a direct selection
  * named ports declared on a wrapped NumPy function (`outputs={...}`)
  * port selection inside a fan-out (`port(1)` re-exports the model order)
  * shape symbols flowing through: (M snapshots x N sensors) -> (N x N) -> (N, K)
  * provenance of a result that merges several ports: `f.prov["eigh"]`, `f[1]`
"""

from collections.abc import Iterator

import numpy as np

from sigflow import Frame, Framed, Sig, block, linalg, plot, port, run, source
from sigflow.dims import G, K, M, N, T

N_SENSORS, N_SNAP = 8, 200
TRUE_DOAS = (-30.0, 10.0, 45.0)  # degrees
GRID = np.linspace(-90, 90, 721)


def steering(theta_deg: float, n: int = N_SENSORS) -> np.ndarray:
    return np.exp(1j * np.pi * np.arange(n) * np.sin(np.deg2rad(theta_deg)))


@source(total=lambda frames, **_: frames)
def array_snapshots(
    frames: int = 6, snr_db: float = 10.0, seed: int = 4
) -> Iterator[Sig[complex, tuple[M, N]]]:
    """M snapshots of an N-element uniform linear array with three sources; the
    middle source fades out over the frames so the model order changes."""
    rng = np.random.default_rng(seed)
    noise = 10 ** (-snr_db / 20)
    for k in range(frames):
        gains = [1.0, max(0.0, 1.0 - k / 3), 0.8]
        X = sum(
            g * np.outer(np.exp(2j * np.pi * rng.uniform(size=N_SNAP)), steering(th))
            for g, th in zip(gains, TRUE_DOAS)
        )
        X += (
            noise
            * (
                rng.normal(size=(N_SNAP, N_SENSORS))
                + 1j * rng.normal(size=(N_SNAP, N_SENSORS))
            )
            / np.sqrt(2)
        )
        yield Frame(X, meta={"frame": k, "active_sources": sum(g > 0 for g in gains)})


# eigh returns (values ascending, vectors as columns): declare both ports with names
eigh = block(
    np.linalg.eigh,
    name="eigh",
    inputs=Sig[T, tuple[N, N]],
    outputs={"vals": Sig[float, N], "vecs": Sig[T, tuple[N, N]]},
)


@block
def model_order(vals: Sig[float, N], gap_db: float = 10.0) -> Sig[int]:
    """Count signal eigenvalues: those more than `gap_db` above the smallest (noise) one."""
    db = 10 * np.log10(vals / vals.min())
    return int(np.sum(db > gap_db))


@block
def noise_subspace(vecs: Sig[T, tuple[N, N]], order: Sig[int]) -> Sig[T, tuple[N, K]]:
    """Eigenvectors spanning the noise subspace: all but the `order` largest."""
    return vecs[:, : vecs.shape[1] - order]


@block
def music_spectrum(En: Sig[T, tuple[N, K]]) -> Sig[float, G]:
    A = np.stack([steering(th) for th in GRID], axis=1)  # N x G
    proj = En.conj().T @ A  # K x G
    return 1.0 / np.maximum(np.sum(np.abs(proj) ** 2, axis=0), 1e-12)


@block
def peaks(f: Framed[Sig[float, G]], order: Sig[int]) -> Sig[dict]:
    p = f.data
    idx = [i for i in range(1, len(p) - 1) if p[i] > p[i - 1] and p[i] >= p[i + 1]]
    best = sorted(idx, key=lambda i: -p[i])[:order]
    return {
        "doa_deg": sorted(round(float(GRID[i]), 1) for i in best),
        "order": order,
        "frame": f["frame"],
        "truth": TRUE_DOAS[:1]
        + (TRUE_DOAS[1:] if f["active_sources"] == 3 else TRUE_DOAS[2:]),
    }


def draw_spectrum(ax, p, f: Frame) -> None:
    ax.plot(GRID, 10 * np.log10(p / p.max()))
    for th in TRUE_DOAS:
        ax.axvline(th, color="#dc2626", lw=0.8, ls="--")
    ax.set_xlabel("angle [deg]")
    ax.set_ylabel("MUSIC [dB]")
    ax.set_ylim(-40, 2)
    ax.set_title(f"frame {f['frame']}: {f['active_sources']} active sources")


if __name__ == "__main__":
    decomposition = array_snapshots() >> linalg.covariance >> eigh

    # Both ports of `eigh` are needed downstream: the eigenvalues give the model order,
    # the eigenvectors (with that order) give the noise subspace. A fan-out of port
    # selectors does the routing, and the second fan-out re-exports the order so the
    # peak picker can use it alongside the spectrum.
    spectrum = plot.custom(
        draw_spectrum, name="spectrum", in_type=Sig[float, G], figsize=(7, 3.2)
    )
    g = (
        decomposition
        >> (port("vecs") | port("vals") >> model_order)  # (vecs, order)
        >> (noise_subspace >> music_spectrum >> spectrum | port(1))  # (spectrum, order)
        >> peaks
    )
    print(g.describe(), "\n")
    g.draw().save("doa_graph.svg")

    for f in run(g, progress=True):
        d = f.data
        print(
            f"frame {d['frame']}: order {d['order']}  DoA {d['doa_deg']}  truth {d['truth']}"
        )

    # Direct selection when only one port matters:
    vals = run(decomposition["vals"])
    print(
        "\neigenvalue spread per frame (dB):",
        [round(float(10 * np.log10(f.data.max() / f.data.min())), 1) for f in vals],
    )

    # Provenance of a merged result: both ports lead back to the same decomposition.
    last = run(g)[-1]
    print(
        "\nthe peak picker's inputs came from:",
        [n.op for n in last.prov.parents[0].parents],
    )
    print(
        "nearest eigh in the lineage:",
        last.prov["eigh"],
        "| source frame:",
        last.prov[-1].seq,
    )
    print("saved doa_graph.svg,", spectrum.save("doa_spectrum.gif", interval=400))
```
