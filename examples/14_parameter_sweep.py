"""Parameter sweeps: the same graph over a grid of configurations, in parallel.

The QPSK link from example 04 is swept over SNR to produce a BER curve, two ways:

  * processes — each configuration builds and runs its graph in a worker process
    (true parallelism for CPU-bound chains; graphs are built inside the worker
    from a plain factory function, so nothing needs to be pickled but the config)
  * threads   — `start()` several runs at once in this process and watch them
    together (NumPy releases the GIL in the heavy parts; handy in a notebook)

Provenance tags every result with the configuration that produced it, and a
summary plot is made after the sweep.

What this exercises
  * building graphs from a factory, cloning sub-graphs per configuration
  * `run(...)` in worker processes; progress across the whole sweep
  * `start()`-based concurrency and polling several Run handles
  * `f.prov["channel"].params` to recover the configuration from a result
"""
import importlib.util
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_qpsk():
    """Import examples/04_qpsk_link.py as a module (its name starts with a digit)."""
    spec = importlib.util.spec_from_file_location("qpsk_link", os.path.join(HERE, "04_qpsk_link.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def link(snr_db: float, frames: int = 24, cfo: float = 0.002):
    """Graph factory: a complete link at one SNR. Called in whichever process runs it."""
    q = load_qpsk()
    rx = (q.bits(frames=frames) >> q.qpsk_map >> q.pulse_shape >> q.channel(snr_db=snr_db, cfo=cfo)
          >> q.matched_filter >> q.symbol_sync >> q.coarse_cfo >> q.carrier_recovery >> q.qpsk_demap)
    return (rx & q.bits(frames=frames)) >> q.ber


def ber_at(snr_db: float) -> dict:
    """One sweep point (runs in a worker process)."""
    from sigflow import run
    out = run(link(snr_db))
    good = out[3:]                                       # skip loop acquisition
    errors = sum(f.data["errors"] for f in good)
    bits = sum(2 * 256 for _ in good)
    cfg = good[-1].prov["channel"].params                # the configuration, from provenance
    return {"snr_db": cfg["snr_db"], "cfo": cfg["cfo"], "ber": errors / bits, "bits": bits}


def theory(snr_db: np.ndarray) -> np.ndarray:
    from scipy.special import erfc
    return 0.5 * erfc(np.sqrt(10 ** (snr_db / 10) / 2))     # QPSK per-bit BER with Eb/N0 = Es/N0 / 2


if __name__ == "__main__":
    grid = [3, 5, 7, 9, 11]

    # --- processes -----------------------------------------------------------
    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
        futures = {pool.submit(ber_at, snr): snr for snr in grid}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            print(f"[{i}/{len(grid)}] SNR {r['snr_db']:4.1f} dB  BER {r['ber']:.4f}  ({r['bits']} bits)")
    results.sort(key=lambda r: r["snr_db"])
    print(f"processes: {time.perf_counter() - t0:.1f}s for {len(grid)} points\n")

    # --- threads: several background runs, polled together -------------------
    from sigflow import Monitor
    t0 = time.perf_counter()
    runs = {snr: link(snr).start() for snr in grid}
    snapshots = []
    while not all(r.finished for r in runs.values()):
        snapshots.append("  ".join(f"{snr}dB:{r.monitor.percent or 0:3.0f}%" for snr, r in runs.items()))
        time.sleep(0.25)
    for line in snapshots[:3]:
        print("   progress:", line)
    print(f"threads:   {time.perf_counter() - t0:.1f}s; "
          + ", ".join(f"{snr}dB={sum(f.data['errors'] for f in r.frames[3:]) / (2 * 256 * len(r.frames[3:])):.4f}"
                      for snr, r in runs.items()))
    print("(at 3 dB the carrier loop cycle-slips, so simulated BER sits above the AWGN theory curve)")

    # --- summary plot --------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    snr = np.array([r["snr_db"] for r in results])
    ber = np.array([r["ber"] for r in results])
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.semilogy(snr, np.maximum(ber, 1e-5), "o-", label="simulated (processes)")
    ax.semilogy(snr, np.maximum(theory(snr), 1e-5), "k--", label="theory")
    ax.set_xlabel("Es/N0 [dB]"); ax.set_ylabel("BER"); ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.savefig("ber_sweep.png", dpi=100, bbox_inches="tight")
    print("\nsaved ber_sweep.png")
