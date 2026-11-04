# What an explicit Graph checks while you wire it, and what it checks later

`examples/26_graph_checks.py`

Run ``python examples/26_graph_checks.py``. Every ``Graph.connect`` is checked on the
spot against the consumer's input contract: dtype, shape and the sizes symbols are
bound to, rate, units, tags and the types of their values. A connection that fails
leaves the graph exactly as it was, so the next attempt starts clean. Earlier edges
are re-checked too, so wiring in any order finds the same problems. Rates are
checked where branches meet, ports by name, and cycles are refused.

One thing is checked later: a member that is itself a pipeline is checked at its
entry when connected, and the rest of it when the graph is validated, which ``run``
does before any frame moves. The last section shows that case.

## Source

```python
"""What an explicit Graph checks while you wire it, and what it checks later.

Run ``python examples/26_graph_checks.py``. Every ``Graph.connect`` is checked on the
spot against the consumer's input contract: dtype, shape and the sizes symbols are
bound to, rate, units, tags and the types of their values. A connection that fails
leaves the graph exactly as it was, so the next attempt starts clean. Earlier edges
are re-checked too, so wiring in any order finds the same problems. Rates are
checked where branches meet, ports by name, and cycles are refused.

One thing is checked later: a member that is itself a pipeline is checked at its
entry when connected, and the rest of it when the graph is validated, which ``run``
does before any frame moves. The last section shows that case.
"""

import warnings
from typing import Annotated

from sigflow import (
    Graph,
    RatioMismatch,
    Sig,
    SignalType,
    TypeMismatch,
    accepts,
    block,
    dsp,
    from_iterable,
    requires,
    run,
)
from sigflow.dims import M, N
from sigflow.types import TagWarning

RATE = 48_000.0


def check(label: str, attempt) -> None:
    """Run one wiring attempt and say how the graph took it."""
    try:
        attempt()
    except (TypeMismatch, RatioMismatch, KeyError) as e:
        print(f"  refused   {label}\n            {type(e).__name__}: {e}")
    else:
        print(f"  accepted  {label}")


# --------------------------------------------------------------------- blocks

needs_256 = block(
    lambda x: x,
    inputs=SignalType(float, (256,)),
    outputs=SignalType(float, (256,)),
    name="needs_256",
)
volts_only = block(
    lambda x: x,
    inputs=SignalType(float, (N,), None, "V"),
    outputs=SignalType(float, (N,), None, "V"),
    name="volts_only",
)


@block
def combine(x: Sig[float, N], y: Sig[float, M]) -> Sig[float, N]:
    """Two inputs that must arrive frame for frame."""
    return x


@block
def demod(
    x: Annotated[Sig[float, N], requires("synced", track=int), accepts("snr_db")],
) -> Sig[float, N]:
    """Needs a synced frame with an integer track id; reads an SNR when there is one."""
    return x


def tagged(**provided):
    """A source whose contract guarantees tags (``name=type`` for a typed one)."""
    names = [k for k, v in provided.items() if v is None]
    typed = {k: v for k, v in provided.items() if v is not None}
    return from_iterable([], type=SignalType(float, (N,)).provides(*names, **typed))


def tag_checks() -> None:
    g = Graph("tags")
    use = g.add(demod)
    check(
        "a source guaranteeing nothing -> demod",
        lambda: g.connect(g.add(from_iterable([])), use),
    )
    g = Graph("tags")
    use = g.add(demod)
    check(
        "track guaranteed as str -> requires track: int",
        lambda: g.connect(g.add(tagged(synced=None, track=str)), use),
    )
    g = Graph("tags")
    use = g.add(demod)
    check(
        "synced and track: int -> demod",
        lambda: g.connect(g.add(tagged(synced=None, track=int)), use),
    )


def main() -> None:
    print("1. A shape that cannot fit is refused, and the graph is left as it was")
    g = Graph("shapes")
    src = g.add(dsp.sine(n=1024, rate=RATE, frames=2), name="src")
    fit = g.add(needs_256)
    check("sine(n=1024) -> needs_256", lambda: g.connect(src, fit))
    print(f"            edges in the graph afterwards: {len(g.edges)}")
    framed = g.add(dsp.chunk(256), name="chunk")
    g.connect(src, framed).connect(framed, fit).output(fit)
    print(f"  fixed with chunk(256) between them: {len(run(g))} frames of 256")

    print("\n2. Wiring in any order finds the same problem")
    g = Graph("reverse")
    src = g.add(dsp.sine(n=1024, rate=RATE), name="src")
    gain = g.add(dsp.scale(gain=2.0))
    fit = g.add(needs_256)
    g.connect(gain, fit)  # scale passes its input's size on: fine so far
    check("then sine(n=1024) -> scale", lambda: g.connect(src, gain))

    print("\n3. Units are part of the contract")
    g = Graph("units")
    adc = g.add(from_iterable([], type=SignalType(float, (N,), RATE, "mV")), name="adc")
    volts = g.add(volts_only)
    check("an mV source -> a V-only block", lambda: g.connect(adc, volts))
    convert = g.add(dsp.to_units("V"))
    check(
        "mV -> to_units('V') -> V-only",
        lambda: g.connect(adc, convert).connect(convert, volts),
    )

    print("\n4. Branches that meet must keep pace")
    g = Graph("rates")
    src = g.add(dsp.sine(n=1024, rate=RATE), name="src")
    framed = g.add(dsp.chunk(256), name="chunk")
    both = g.add(combine)
    g.connect(src, framed).connect(src, both, "x")
    check(
        "1 frame per input and 4 per input into one block",
        lambda: g.connect(framed, both, "y"),
    )

    print("\n5. Ports by name, each connected once")
    g = Graph("ports")
    a = g.add(dsp.sine(n=16), name="a")
    b = g.add(dsp.sine(n=16), name="b")
    both = g.add(combine)
    check("slot='z' on combine(x, y)", lambda: g.connect(a, both, "z"))
    g.connect(a, both, "x")
    check("a second edge into combine.x", lambda: g.connect(b, both, "x"))

    print("\n6. Loops need feedback()")
    g = Graph("cycle")
    one = g.add(dsp.scale(gain=1.0), name="one")
    two = g.add(dsp.scale(gain=2.0), name="two")
    g.connect(one, two)
    check("two -> one, closing a loop", lambda: g.connect(two, one))

    print("\n7. Tags: required, typed, and read if present")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", TagWarning)
        tag_checks()
    for w in {str(w.message) for w in caught}:
        print(f"            warning: {w}")

    print("\n8. Checked later: a member that is itself a pipeline")
    g = Graph("composite")
    cplx = g.add(dsp.sine(n=256, frames=1) >> dsp.to_complex, name="cplx")
    inner = g.add(dsp.scale(gain=1.0) >> needs_256, name="inner")
    check("complex samples -> (scale >> needs_256)", lambda: g.connect(cplx, inner))
    print("            connect saw only the member's entry: scale takes any dtype")
    g.output(inner)
    check("g.validate(), which run() does before any frame moves", g.validate)


if __name__ == "__main__":
    main()
```
