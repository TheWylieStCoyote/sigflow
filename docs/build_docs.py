"""Generate the example pages (and copy README/CHANGELOG) for the mkdocs site.

Each example script is run in a scratch directory; the plots, animations and graph
drawings it writes (``.png``, ``.svg``, ``.gif``) are copied to
``docs/examples/assets/<example>/`` and embedded in its page.

Run: python docs/build_docs.py && mkdocs build
     python docs/build_docs.py --no-run   # reuse assets from a previous build
"""

import ast
import glob
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")
ASSETS = os.path.join(HERE, "examples", "assets")
sys.path.insert(0, SRC)

IMAGE_EXT = (".png", ".svg", ".gif")
MAX_GIF_BYTES = 8 * 1024 * 1024  # larger animations are shown as a still of the last frame
TIMEOUT = 600  # seconds per example


def _warn(msg: str) -> None:
    print(f"build_docs: {msg}", file=sys.stderr)


def _images(root: str) -> list[str]:
    out = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMAGE_EXT):
                out.append(os.path.join(dirpath, f))
    return sorted(out, key=lambda p: (os.path.basename(p), p))


def _still(gif: str, png: str) -> None:
    """Write the last frame of ``gif`` as ``png``."""
    from PIL import Image

    with Image.open(gif) as im:
        im.seek(getattr(im, "n_frames", 1) - 1)
        im.convert("RGB").save(png)


def run_example(path: str, name: str) -> list[tuple[str, str]]:
    """Run one example in a scratch directory; copy its images into the assets folder.
    Returns ``(filename, caption)`` pairs for the page."""
    dest_dir = os.path.join(ASSETS, name)
    shutil.rmtree(dest_dir, ignore_errors=True)
    work = tempfile.mkdtemp(prefix=f"sigflow_docs_{name}_")
    env = dict(os.environ, MPLBACKEND="Agg", TMPDIR=work, TEMP=work, TMP=work)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (SRC, env.get("PYTHONPATH")) if p)
    try:
        proc = subprocess.run(
            [sys.executable, path],
            cwd=work,
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        _warn(f"{name}: timed out after {TIMEOUT}s; page has no output section")
        shutil.rmtree(work, ignore_errors=True)
        return []
    if proc.returncode:
        tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
        _warn(f"{name}: exited with {proc.returncode}; page has no output section\n{tail}")
        shutil.rmtree(work, ignore_errors=True)
        return []

    items: list[tuple[str, str]] = []
    used: set[str] = set()
    for src in _images(work):
        base = os.path.basename(src)
        if base in used:  # same filename from two scratch sub-directories
            base = f"{os.path.basename(os.path.dirname(src))}_{base}"
        used.add(base)
        os.makedirs(dest_dir, exist_ok=True)
        size = os.path.getsize(src)
        if base.lower().endswith(".gif") and size > MAX_GIF_BYTES:
            still = os.path.splitext(base)[0] + ".png"
            _still(src, os.path.join(dest_dir, still))
            items.append(
                (still, f"`{base}` (last frame; the {size / 2**20:.0f} MB animation is not copied)")
            )
        else:
            shutil.copy(src, os.path.join(dest_dir, base))
            items.append((base, f"`{base}`"))
    shutil.rmtree(work, ignore_errors=True)
    return items


def existing_assets(name: str) -> list[tuple[str, str]]:
    """Assets left by a previous build (``--no-run``)."""
    dest_dir = os.path.join(ASSETS, name)
    if not os.path.isdir(dest_dir):
        return []
    return [(f, f"`{f}`") for f in sorted(os.listdir(dest_dir)) if f.lower().endswith(IMAGE_EXT)]


def example_pages(run: bool = True) -> None:
    index = [
        "# Examples",
        "",
        "Every script under `examples/` runs on its own (`python examples/NN_name.py`). "
        "The notebooks `sigflow_tour.ipynb` and `live_control.ipynb` cover the interactive side. "
        "The plots, animations and graph drawings on each page were written by the script "
        "when the docs were built.",
        "",
        "| example | what it exercises | output |",
        "|---|---|---|",
    ]
    for path in sorted(glob.glob(os.path.join(ROOT, "examples", "[0-9]*.py"))):
        name = os.path.basename(path)[:-3]
        src = open(path).read()
        doc = ast.get_docstring(ast.parse(src)) or ""
        title = doc.splitlines()[0].rstrip(".") if doc else name
        body = "\n".join(doc.splitlines()[1:]).strip()
        assets = run_example(path, name) if run else existing_assets(name)
        page = [f"# {title}", "", f"`examples/{name}.py`", "", body, ""]
        if assets:
            page += ["## Output", "", "Files the script writes when run:", ""]
            for fname, caption in assets:
                page += [f"![{fname}](assets/{name}/{fname})", "", f"*{caption}*", ""]
        page += ["## Source", "", "```python", src.rstrip(), "```", ""]
        with open(os.path.join(HERE, "examples", f"{name}.md"), "w") as fh:
            fh.write("\n".join(page))
        kinds = ", ".join(sorted({os.path.splitext(f)[1][1:] for f, _ in assets})) or "—"
        index.append(f"| [{name}]({name}.md) | {title} | {kinds} |")
    with open(os.path.join(HERE, "examples", "index.md"), "w") as fh:
        fh.write("\n".join(index) + "\n")


def copies() -> None:
    shutil.copy(os.path.join(ROOT, "CHANGELOG.md"), os.path.join(HERE, "changelog.md"))
    readme = open(os.path.join(ROOT, "README.md")).read()
    with open(os.path.join(HERE, "index.md"), "w") as fh:
        fh.write(readme)


if __name__ == "__main__":
    example_pages(run="--no-run" not in sys.argv[1:])
    copies()
    print("docs generated")
