"""Tag contracts end to end: required and optional tags, typed values, policies, ports.

Run ``python examples/32_tag_policies_and_ports.py``. A small receiver chain shows what
a signature can say about tags and what the graph does with it:

1. **required, typed tags**: ``demod`` requires the label ``synced`` and a ``track``
   tag whose value is an ``int``, and accepts an optional ``snr_db``. A producer that
   guarantees them is accepted; one without them, or whose ``track`` is a ``str``,
   is refused when the graph is built;
2. **a warning, not an error**: nothing upstream provides ``snr_db``, so the
   ``accepts`` can never be met, and building the graph says so (``TagWarning``);
3. **produced tags**: ``demod`` ``provides`` a ``bits`` count on every frame and
   ``may_provide`` a ``crc_fail`` flag on some; ``when("crc_fail", ...)`` routes on
   the possible tag, and ``clears`` removes a tag once it is handled;
4. **propagation policies**: ``tags="none"``, ``keep(...)`` and ``spans="drop"`` decide
   what a block passes on of what it inherited; tags it attaches itself always pass;
5. **per-port policies**: a block with two outputs carries tags to one port and not
   the other (``carries``), and the tuple frame's own tags are the union of its
   parts' tags.

``describe()`` shows each block's contract and policy; the end of the output prints it.
"""

import warnings
from collections.abc import Iterator
from typing import Annotated

import numpy as np

from sigflow import (
    Frame,
    Framed,
    Sig,
    SignalType,
    TypeMismatch,
    accepts,
    block,
    carries,
    clears,
    from_iterable,
    keep,
    may_provide,
    provides,
    requires,
    run,
    source,
    when,
)
from sigflow.dims import N
from sigflow.types import TagWarning

RATE = 1_000.0


@source
def synced_bursts(
    frames: int = 6,
) -> Iterator[Annotated[Sig[float, N], provides("synced", track=int)]]:
    """Bursts after a synchroniser: each carries the label ``synced``, its ``track``
    id, a ``calibrated`` label, and a spanned ``preamble`` tag over its first 16
    samples."""
    rng = np.random.default_rng(5)
    for i in range(frames):
        x = rng.choice([-1.0, 1.0], 64)
        yield (
            Frame(x, meta={"seq": i, "t0": i * 64 / RATE}, stream={"rate": RATE})
            .tag("synced", "calibrated", track=i % 2)
            .tag("preamble", span=(i * 64 / RATE, (i * 64 + 16) / RATE))
        )


@block
def demod(
    f: Framed[
        Annotated[Sig[float, N], requires("synced", track=int), accepts(snr_db=float)]
    ],
) -> Annotated[Sig[float, N], provides(bits=int), may_provide(crc_fail=bool)]:
    """Hard decisions; every third burst fails its CRC."""
    bits = (f.data > 0).astype(float)
    out = f.evolve(data=bits).tag(bits=int(bits.size))
    return out.tag(crc_fail=True) if f["seq"] % 3 == 2 else out


@block
def retry(
    x: Annotated[Sig[float, N], requires("crc_fail")],
) -> Annotated[Sig[float, N], clears("crc_fail")]:
    """Handles a failed burst; the flag stops here."""
    return x


@block(tags="none")
def anonymise(x: Sig[float, N]) -> Sig[float, N]:
    """Passes on none of the tags it inherited."""
    return x


@block(tags=keep("synced"), spans="drop")
def keep_sync(x: Sig[float, N]) -> Sig[float, N]:
    """Passes on ``synced`` only, and no spanned tags."""
    return x


@block
def split(
    x: Sig[float, N],
) -> tuple[Annotated[Sig[float, N], carries("none")], Sig[float, N]]:
    """Two copies: port 0 for a consumer that must not see the tags, port 1 with them."""
    return x, x


def names(f: Frame) -> list[str]:
    return sorted(f.tags.names)


def main() -> None:
    # 1. required, typed tags -------------------------------------------------------
    plain = from_iterable([np.zeros(64)], type=SignalType(float, (N,)))
    wrong = from_iterable(
        [np.zeros(64)], type=SignalType(float, (N,)).provides("synced", track=str)
    )
    for label, producer in (("no tags", plain), ("track id a str", wrong)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", TagWarning)  # section 2 shows it
                producer >> demod
        except TypeMismatch as e:
            print(f"1. {label:15s} refused: {e}")

    # 2. the accepted tag nothing provides ------------------------------------------
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", TagWarning)
        graph = synced_bursts() >> demod >> when("crc_fail", retry)
    for message in dict.fromkeys(str(w.message) for w in caught):  # each one once
        print(f"\n2. warned: {message}")
    warnings.simplefilter("ignore", TagWarning)  # shown once; run() would warn again

    # 3. produced tags, routing on a possible tag, clearing it ------------------------
    out = run(graph, on_error="raise")
    print("\n3. after demod and when('crc_fail', retry):")
    for f in out:
        retried = any(p.op == "retry" for p in f.prov.chain())
        how = "crc_fail -> retry, cleared" if retried else "passed"
        print(f"   seq {f['seq']}: bits={f.tags['bits']}, {how:26s} tags {names(f)}")

    # 4. propagation policies ---------------------------------------------------------
    before = run(synced_bursts(frames=1))[0]
    print(f"\n4. a source frame carries {names(before)}")
    for blk in (anonymise, keep_sync):
        after = run(synced_bursts(frames=1) >> blk)[0]
        print(f"   after {blk.name:9s} it carries {names(after)}")

    # 5. per-port policies and the tuple frame's tags ----------------------------------
    pair = run(synced_bursts(frames=1) >> split)[0]
    print(
        f"\n5. split port 0 carries {names(pair[0])}, port 1 carries {names(pair[1])}"
    )
    print(f"   the tuple frame itself carries {names(pair)} (the union of its parts)")

    print("\nThe contracts and policies, as describe() shows them:")
    print((synced_bursts() >> demod >> keep_sync >> split).describe())


if __name__ == "__main__":
    main()
