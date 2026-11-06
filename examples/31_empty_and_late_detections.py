"""Per-signal processing when detections are missing, failing, late or intermittent.

Run ``python examples/31_empty_and_late_detections.py``. Example 25 builds the routing
model on a scene that always holds a signal; real scenes do not. A detector yields
one frame per detection, each signal is measured by its own stateful instance
(``per_key``), and the measurements are gathered back into one frame per scan
(``gather``). This script feeds that graph the awkward cases and shows what comes out:

- **a scan with no detections**: the detector would yield nothing, and the scan would
  vanish from the output. It yields one *marker* frame instead (``track=None``),
  which ``per_key`` passes through untouched, so every scan gets an output;
- **a measurement that fails**: the failed frame travels on as its own output, and
  its scan's group is released *incomplete*, with a warning, by ``ttl`` or at the end;
- **late results**: track A's instance answers slowly, so measurements finish out of
  order and ``gather`` emits groups as they complete. ``ordered=True`` follows the
  order groups *started* (their first member's arrival), which is not scan order
  either when results are late: sort by ``group`` when you need scan order;
- **a signal that disappears and comes back**: after ``ttl`` scans without it, its
  instance is closed; when it returns, a fresh instance starts (its count restarts);
- ``incomplete="drop"``: incomplete groups are dropped instead of emitted;
- **a ``ttl`` shorter than the lateness** splits a group: one member is released
  alone, and the late one starts a second group with the same key.
"""

import asyncio
from collections.abc import Iterator

from sigflow import (
    Frame,
    Framed,
    Sig,
    Stateful,
    Stream,
    block,
    gather,
    per_key,
    run,
    source,
)

# scan -> the detections in it, as (track, level in dB); a level below zero is a bad
# detection that the measurement rejects
SCANS = [
    [("A", 12.0), ("B", 9.0)],
    [],  # nothing detected
    [("A", 12.5)],
    [("A", 13.0), ("B", -1.0)],  # B's measurement fails
    [("A", 12.0)],
    [("A", 12.2)],
    [("A", 11.8)],
    [("A", 12.1), ("B", 8.5)],  # B is back after four scans without it
]
DELAY = {"A": 0.03, "B": 0.0}  # A's instance is slow: its results arrive late


@source
def scans() -> Iterator[Sig[list]]:
    for i, detections in enumerate(SCANS):
        yield Frame(detections, meta={"seq": i})


@block
def detect(scans: Stream[Framed[Sig[list]]]) -> Iterator[Framed[Sig[float]]]:
    """One frame per detection, tagged with its scan (``seq``), its ``track`` and how
    many detections the scan had (``n``). A scan with none yields one marker frame
    with ``track=None``, so the scan is not lost downstream."""
    for scan in scans:
        found = scan.data
        if not found:
            yield scan.evolve(data=0.0, meta={**scan.meta, "track": None, "n": 1})
            continue
        for track, level in found:
            yield scan.evolve(
                data=level, meta={**scan.meta, "track": track, "n": len(found)}
            )


class Measure(Stateful):
    """One instance per track: counts the detections it has measured."""

    def __init__(self) -> None:
        self.count = 0

    async def __call__(self, f: Framed[Sig[float]]) -> Sig[float]:
        await asyncio.sleep(DELAY[f["track"]])
        if f.data < 0:
            raise ValueError(
                f"track {f['track']}: level {f.data} dB is not a detection"
            )
        self.count += 1
        return f.evolve(data=f.data, meta={**f.meta, "instance_count": self.count})


measure = block(Measure(), name="measure")


def describe(group: Frame) -> str:
    parts = []
    for p in group.parts:
        if p.failed:
            parts.append("failed")
        elif p["track"] is None:
            parts.append("no detections")
        else:
            parts.append(f"{p['track']} {p.data:.1f} dB (#{p['instance_count']})")
    warn = f"  [{group.issues[0].message}]" if group.issues else ""
    return f"scan {group['group']}: {', '.join(parts)}{warn}"


def collect(ttl: int = 10, **options) -> list[Frame]:
    graph = (
        scans()
        >> detect
        >> per_key("track", measure, ttl=3)
        >> gather("seq", count="n", order=lambda f: str(f["track"]), ttl=ttl, **options)
    )
    return run(graph)


def groups(out: list[Frame]) -> list[Frame]:
    return [f for f in out if "group" in f.meta]


def main() -> None:
    print("Groups as they complete (A's late results reorder them):")
    for f in collect():
        print(
            "  "
            + (
                describe(f)
                if "group" in f.meta
                else f"failed member: {f.errors[0].message}"
            )
        )

    started = [f["group"] for f in groups(collect(ordered=True))]
    by_scan = sorted(groups(collect()), key=lambda f: f["group"])
    print(f"\nordered=True, the order groups started: scans {started}")
    print(
        f"sorted by group, scan order:          scans {[f['group'] for f in by_scan]}"
    )

    kept = [f["group"] for f in groups(collect(incomplete="drop"))]
    print(f"\nincomplete='drop' keeps complete scans only: {kept}")

    print("\nA gather ttl shorter than the lateness splits scan 7 in two:")
    for f in groups(collect(ttl=2)):
        if f["group"] == 7:
            print("  " + describe(f))


if __name__ == "__main__":
    main()
