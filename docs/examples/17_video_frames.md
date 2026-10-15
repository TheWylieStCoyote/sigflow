# 2-D signals: a video stream — frame differencing, 2-D FFT, masks as sample annotations

`examples/17_video_frames.py`

What this exercises
  * frames whose data is an (H, W) image, with symbolic 2-D shapes checked end to end
  * a stateful stream block (frame differencing needs the previous frame)
  * sample-level annotations on 2-D data: a motion mask aligned with the image,
    kept through image->image blocks (same leading dimension) and dropped by a
    block that changes it (the 2-D FFT's shifted spectrum keeps H, so it survives!
    — the example shows exactly when masks are kept)
  * a per-row annotation (motion energy per scan line) as a second sample-level array
  * fan-out to a 2-D spectrum, a centroid tracker and an image plot with the mask overlaid

## Source

```python
"""2-D signals: a video stream — frame differencing, 2-D FFT, masks as sample annotations.

What this exercises
  * frames whose data is an (H, W) image, with symbolic 2-D shapes checked end to end
  * a stateful stream block (frame differencing needs the previous frame)
  * sample-level annotations on 2-D data: a motion mask aligned with the image,
    kept through image->image blocks (same leading dimension) and dropped by a
    block that changes it (the 2-D FFT's shifted spectrum keeps H, so it survives!
    — the example shows exactly when masks are kept)
  * a per-row annotation (motion energy per scan line) as a second sample-level array
  * fan-out to a 2-D spectrum, a centroid tracker and an image plot with the mask overlaid
"""

from typing import Iterator, Literal
import numpy as np

from sigflow import Frame, Framed, Sig, Stream, block, plot, run, source
from sigflow.dims import H, W
from sigflow import SignalType

HEIGHT, WIDTH = 64, 96


@source(
    total=lambda frames, **_: frames,
    refine=lambda **_: SignalType(float, (HEIGHT, WIDTH)),
)
def camera(frames: int = 40, seed: int = 0) -> Iterator[Sig[float, tuple[H, W]]]:
    """A bright blob drifting diagonally over a static textured background, with noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    background = 0.3 + 0.1 * np.sin(2 * np.pi * xx / 16) * np.cos(2 * np.pi * yy / 12)
    for k in range(frames):
        cx, cy = 10 + 1.8 * k, 8 + 1.1 * k
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 4.0**2))
        img = background + 0.8 * blob + 0.03 * rng.normal(size=(HEIGHT, WIDTH))
        yield Frame(img, meta={"frame": k, "true_xy": (cx, cy)}, stream={"fps": 25})


@block
def difference(
    frames: Stream[Framed[Sig[float, tuple[H, W]]]], threshold: float = 0.15
) -> Iterator[Sig[float, tuple[H, W]]]:
    """|frame - previous| with a motion mask (sample level, 2-D) and per-row motion energy."""
    prev = None
    for f in frames:
        img = f.data
        if prev is None:
            prev = img
            continue
        d = np.abs(img - prev)
        prev = img
        mask = d > threshold
        yield f.evolve(data=d).annotate_samples(motion=mask, row_energy=d.sum(axis=1))


@block
def blur(img: Sig[float, tuple[H, W]]) -> Sig[float, tuple[H, W]]:
    """3x3 box blur — same shape in and out, so sample-level masks are kept."""
    p = np.pad(img, 1, mode="edge")
    return sum(p[i : i + HEIGHT, j : j + WIDTH] for i in range(3) for j in range(3)) / 9


@block
def spectrum2d(
    img: Sig[float, tuple[H, W]],
) -> Sig[float, tuple[H, W], None, Literal["dB"]]:
    """Shifted 2-D power spectrum. Also (H, W): the runtime would keep masks here too,
    which is wrong for a spectrum — so this block opts out with keep_samples=False."""
    return 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft2(img))) + 1e-6)


spectrum2d = block(spectrum2d.fn, name="spectrum2d", keep_samples=False)


@block
def centroid(f: Framed[Sig[float, tuple[H, W]]]) -> Sig[dict]:
    """Centre of the motion mask: where the moving object is."""
    mask = f.samples["motion"]
    if mask.sum() < 5:
        return {"frame": f["frame"], "xy": None, "pixels": int(mask.sum())}
    yy, xx = np.nonzero(mask)
    return {
        "frame": f["frame"],
        "xy": (float(xx.mean()), float(yy.mean())),
        "pixels": int(mask.sum()),
        "true_xy": f["true_xy"],
    }


def draw_frame(ax, img, f: Frame) -> None:
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    if "motion" in f.samples:
        overlay = np.zeros((HEIGHT, WIDTH, 4))
        overlay[f.samples["motion"]] = (1, 0.2, 0.2, 0.5)
        ax.imshow(overlay)
    ax.set_title(
        f"frame {f['frame']}  motion pixels: {int(f.samples['motion'].sum()) if 'motion' in f.samples else 0}"
    )
    ax.set_axis_off()


if __name__ == "__main__":
    view = plot.custom(
        draw_frame, name="view", in_type=Sig[float, tuple[H, W]], figsize=(4.5, 3.2)
    )
    g = camera() >> difference(threshold=0.15) >> blur >> (view | centroid | spectrum2d)
    print(g.describe(), "\n")
    g.draw().save("video_graph.svg")

    out = run(g, progress=True)
    f = out[-1]
    img, track, spec = f.parts
    print(
        "\nafter blur, masks survive :",
        list(img.samples),
        "| row_energy shape",
        img.samples["row_energy"].shape,
    )
    print("after spectrum2d, dropped :", list(spec.samples))
    print("centroid vs truth (last 4 frames):")
    for t in out[-4:]:
        d = t.data[1]
        print(
            f"  frame {d['frame']:2d}: est {tuple(round(v, 1) for v in d['xy'])}  true {tuple(round(v, 1) for v in d['true_xy'])}  ({d['pixels']} px)"
        )
    err = [
        np.hypot(
            t.data[1]["xy"][0] - t.data[1]["true_xy"][0],
            t.data[1]["xy"][1] - t.data[1]["true_xy"][1],
        )
        for t in out
        if t.data[1]["xy"]
    ]
    print(f"mean centroid error {np.mean(err):.2f} px over {len(err)} frames")
    print(
        "busiest scan line in the last frame:",
        int(np.argmax(img.samples["row_energy"])),
    )
    print("saved", view.save("motion.gif", interval=80))
```
