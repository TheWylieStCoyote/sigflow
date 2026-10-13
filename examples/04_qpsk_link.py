"""A complete QPSK link: modulator -> channel -> receiver -> bit error rate.

What this exercises
  * stateful stream blocks written as plain generators (pulse shaping with
    filter memory, a 4th-power frequency estimator, a phase-tracking loop)
  * a stream block that annotates frames (the CFO estimate rides along as meta)
  * wrapping scipy.signal functions as typed blocks
  * zip (`&`) to compare received bits with the transmitted ones
  * constellation plots before and after carrier recovery, exported as GIFs
  * provenance: which transmitted frame a BER figure came from
"""

import numpy as np
import scipy.signal as ss

from sigflow import Frame, Framed, Sig, Stream, block, plot, run, source

SPS = 8  # samples per symbol
N_SYM = 256  # symbols per frame
FRAMES = 30

# ---------------------------------------------------------------- transmitter


@source(total=lambda frames, **_: frames, refine=lambda n, **_: Sig[np.int8, (2 * n,)])
def bits(n: int = N_SYM, frames: int = FRAMES, seed: int = 1) -> Sig[np.int8, "B"]:
    rng = np.random.default_rng(seed)
    for i in range(frames):
        yield Frame(rng.integers(0, 2, 2 * n).astype(np.int8), meta={"seq": i})


@block
def qpsk_map(b: Sig[np.int8, "B"]) -> Sig[complex, "N"]:
    """Gray-coded QPSK: two bits -> one unit-energy symbol."""
    i, q = 1 - 2 * b[0::2].astype(float), 1 - 2 * b[1::2].astype(float)
    return (i + 1j * q) / np.sqrt(2)


def rrc_taps(beta: float = 0.35, span: int = 8, sps: int = SPS) -> np.ndarray:
    t = np.arange(-span * sps, span * sps + 1) / sps
    with np.errstate(divide="ignore", invalid="ignore"):
        num = np.sin(np.pi * t * (1 - beta)) + 4 * beta * t * np.cos(
            np.pi * t * (1 + beta)
        )
        den = np.pi * t * (1 - (4 * beta * t) ** 2)
        h = num / den
    h[t == 0] = 1 - beta + 4 * beta / np.pi
    sing = np.isclose(np.abs(t), 1 / (4 * beta))
    h[sing] = (
        beta
        / np.sqrt(2)
        * (
            (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
            + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
        )
    )
    return h / np.sqrt(np.sum(h**2))


@block
def pulse_shape(syms: Stream[Sig[complex, "N"]], sps: int = SPS) -> Sig[complex, "M"]:
    """Upsample and root-raised-cosine filter; filter state carries across frames."""
    h = rrc_taps(sps=sps)
    zi = np.zeros(len(h) - 1, dtype=complex)
    for s in syms:
        up = np.zeros(len(s) * sps, dtype=complex)
        up[::sps] = s
        y, zi = ss.lfilter(h, [1.0], up, zi=zi)
        yield y


# -------------------------------------------------------------------- channel


@block
def channel(
    x: Stream[Sig[complex, "M"]],
    snr_db: float = 15.0,
    cfo: float = 0.002,
    phase: float = 0.7,
    seed: int = 2,
) -> Sig[complex, "M"]:
    """AWGN + carrier frequency offset (cycles/sample) + phase, continuous across frames."""
    rng = np.random.default_rng(seed)
    n0 = 0
    # Unit-energy RRC pulses + unit-energy matched filter: Es = 1 at the symbol instant and
    # N0 = 2 sigma^2, so sigma^2 = 1 / (2 Es/N0) per real/imaginary component.
    sigma = np.sqrt(10 ** (-snr_db / 10) / 2)
    for frame in x:
        n = np.arange(n0, n0 + len(frame))
        rot = np.exp(1j * (2 * np.pi * cfo * n + phase))
        noise = sigma * (rng.normal(size=len(frame)) + 1j * rng.normal(size=len(frame)))
        n0 += len(frame)
        yield frame * rot + noise


# ------------------------------------------------------------------- receiver


@block
def matched_filter(x: Stream[Sig[complex, "M"]], sps: int = SPS) -> Sig[complex, "M"]:
    h = rrc_taps(sps=sps)
    zi = np.zeros(len(h) - 1, dtype=complex)
    for frame in x:
        y, zi = ss.lfilter(h, [1.0], frame, zi=zi)
        yield y


@block
def symbol_sync(
    x: Stream[Sig[complex, "M"]],
    sps: int = SPS,
    n: int = N_SYM,
    delay: int = 2 * 8 * SPS,
) -> Sig[complex, "N"]:
    """Compensate the two filters' group delay, pick one sample per symbol and
    re-frame to exactly ``n`` symbols so output frame k lines up with transmitted frame k."""
    samples = np.zeros(0, dtype=complex)
    symbols = np.zeros(0, dtype=complex)
    skipped = 0
    for frame in x:
        samples = np.concatenate([samples, frame])
        if skipped < delay:  # drop the filter delay once
            take = min(delay - skipped, len(samples))
            samples, skipped = samples[take:], skipped + take
        usable = len(samples) // sps * sps
        symbols = np.concatenate([symbols, samples[:usable:sps]])
        samples = samples[usable:]
        while len(symbols) >= n:
            yield symbols[:n]
            symbols = symbols[n:]


@block
def coarse_cfo(
    syms: Stream[Framed[Sig[complex, "N"]]], nfft: int = 8192
) -> Sig[complex, "N"]:
    """4th-power frequency estimate per frame (QPSK modulation is removed by s**4), applied
    with a phase accumulator that stays continuous across frames.  Annotates each frame
    with the estimate (cycles / symbol) so it is visible downstream and in provenance."""
    phase = 0.0
    for f in syms:
        s = f.data
        spectrum = np.abs(np.fft.fft(s**4, nfft))
        est = np.fft.fftfreq(nfft)[int(np.argmax(spectrum))] / 4  # cycles per symbol
        n = np.arange(len(s))
        out = s * np.exp(-1j * (2 * np.pi * est * n + phase))
        phase = (phase + 2 * np.pi * est * len(s)) % (2 * np.pi)
        yield f.evolve(data=out, meta={"cfo_est": float(est)})


@block
def carrier_recovery(
    syms: Stream[Sig[complex, "N"]], bw: float = 0.02
) -> Sig[complex, "N"]:
    """Decision-directed PLL for QPSK (second order loop), state kept across frames."""
    phase, freq = 0.0, 0.0
    alpha, beta = bw, bw * bw / 4
    for s in syms:
        out = np.empty_like(s)
        for k, z in enumerate(s):
            y = z * np.exp(-1j * phase)
            out[k] = y
            dec = (np.sign(y.real) + 1j * np.sign(y.imag)) / np.sqrt(2)
            err = np.angle(y * np.conj(dec))
            freq += beta * err
            phase += alpha * err + freq
        yield out


@block
def qpsk_demap(s: Sig[complex, "N"]) -> Sig[np.int8, "B"]:
    b = np.empty(2 * len(s), dtype=np.int8)
    b[0::2] = s.real < 0
    b[1::2] = s.imag < 0
    return b


@block
def ber(rx: Framed[Sig[np.int8, "B"]], tx: Framed[Sig[np.int8, "B"]]) -> Sig[dict]:
    """Bit error rate with the QPSK 4-fold phase ambiguity resolved (try the 4 rotations)."""
    r, t = rx.data, tx.data
    best = min(_rotations(r), key=lambda cand: np.count_nonzero(cand != t))
    errors = int(np.count_nonzero(best != t))
    return {"frame": tx.meta.get("seq"), "errors": errors, "ber": errors / len(t)}


def _rotations(b: np.ndarray):
    i, q = b[0::2], b[1::2]
    for ii, qq in ((i, q), (1 - q, i), (1 - i, 1 - q), (q, 1 - i)):
        out = np.empty_like(b)
        out[0::2], out[1::2] = ii, qq
        yield out


# ---------------------------------------------------------------------- graph

if __name__ == "__main__":
    tx = bits() >> qpsk_map
    before = plot.constellation(title="after matched filter (CFO + phase)", lim=1.6)
    after = plot.constellation(title="after carrier recovery", lim=1.6)

    rx = (
        tx
        >> pulse_shape
        >> channel(snr_db=12, cfo=0.002)
        >> matched_filter
        >> symbol_sync
        >> before
        >> coarse_cfo
        >> carrier_recovery
        >> after
        >> qpsk_demap
    )

    link = (
        rx & bits()
    ) >> ber  # the receiver's bits zipped with a second, identical bit source
    print(link.describe(), "\n")

    results = run(link, progress=True)
    total_err = sum(
        f.data["errors"] for f in results[3:]
    )  # skip frames while the loop pulls in
    total_bits = sum(2 * N_SYM for _ in results[3:])
    print(
        f"\nBER after acquisition: {total_err / total_bits:.4f} over {total_bits} bits"
    )
    print("first frames (loop acquiring):", [f.data["ber"] for f in results[:4]])
    print(
        "CFO estimate carried in meta: %.5f cycles/symbol (true: %.5f)"
        % (results[-1].meta["cfo_est"], 0.002 * SPS)
    )

    print("\nprovenance of the last BER figure:\n" + results[-1].trace())
    print(
        "\nsaved",
        before.save("qpsk_before.gif", interval=120),
        after.save("qpsk_after.gif", interval=120),
    )
