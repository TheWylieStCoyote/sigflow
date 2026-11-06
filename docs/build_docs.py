"""Generate the parts of the docs site that come from the code.

Run: ``python docs/build_docs.py [--run-examples] && mkdocs build``

- ``docs/index.md`` and ``docs/changelog.md``: copies of ``README.md`` and
  ``CHANGELOG.md``;
- ``docs/reference/api/*.md``: the API reference, by introspection (``apiref.py``);
- ``docs/examples/*.md``: one page per ``examples/NN_*.py`` and an index by learning
  path, from each script's docstring and ``examples/catalog.toml``.

With ``--run-examples`` every example is run (in a scratch directory, with a time
limit) and its page gains the expected output, a diagram of the first graph it runs,
and the images and reports it writes. CI builds the site that way; without the flag
the pages are generated in seconds, without those sections.

Everything written here is generated: it is not checked in (see ``.gitignore``).
"""

from __future__ import annotations

import argparse
import ast
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXAMPLES = os.path.join(ROOT, "examples")
OUT = os.path.join(HERE, "examples")
MEDIA = os.path.join(OUT, "media")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

IMAGES = (".svg", ".gif", ".png")
REPORTS = (".html",)
MAX_OUTPUT_LINES = 60
MAX_IMAGES = 6


def scripts() -> list[str]:
    return sorted(glob.glob(os.path.join(EXAMPLES, "[0-9]*.py")))


def catalog() -> dict[str, Any]:
    with open(os.path.join(EXAMPLES, "catalog.toml"), "rb") as fh:
        return tomllib.load(fh)


# --------------------------------------------------------------------------- #
# running an example
# --------------------------------------------------------------------------- #


def run_example(path: str, name: str, timeout: float = 300) -> dict[str, Any]:
    """Run one example in a scratch directory; keep its output, the diagram of the
    first graph it ran and the files it wrote that a page can show."""
    media = os.path.join(MEDIA, name)
    shutil.rmtree(media, ignore_errors=True)
    with tempfile.TemporaryDirectory() as work:
        diagram = os.path.join(work, "_graph.svg")
        env = dict(os.environ, MPLBACKEND="Agg", SIGFLOW_DOCS_DIAGRAM=diagram)
        env["PYTHONPATH"] = os.pathsep.join(
            filter(
                None,
                [
                    os.path.join(HERE, "_hooks"),
                    os.path.join(ROOT, "src"),
                    env.get("PYTHONPATH"),
                ],
            )
        )
        try:
            done = subprocess.run(
                [sys.executable, path],
                cwd=work,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,  # a failing example is reported on its page
            )
            ok, stdout, stderr = done.returncode == 0, done.stdout, done.stderr
        except subprocess.TimeoutExpired as e:
            ok, stdout, stderr = (
                False,
                (e.stdout or ""),
                f"timed out after {timeout:.0f} s",
            )
            stdout = stdout.decode() if isinstance(stdout, bytes) else stdout
        found = []
        for dirpath, _, files in os.walk(work):
            for f in sorted(files):
                if f.endswith(IMAGES + REPORTS) and f != "_graph.svg":
                    found.append(os.path.join(dirpath, f))
        os.makedirs(media, exist_ok=True)
        kept: dict[str, list[str]] = {"images": [], "reports": []}
        if os.path.exists(diagram):
            shutil.copy(diagram, os.path.join(media, "graph.svg"))
            kept["graph"] = ["graph.svg"]
        for src in found:
            kind = "images" if src.endswith(IMAGES) else "reports"
            if kind == "images" and len(kept["images"]) >= MAX_IMAGES:
                continue
            rel = os.path.relpath(src, work).replace(os.sep, "__")
            shutil.copy(src, os.path.join(media, rel))
            kept[kind].append(rel)
    lines = stdout.rstrip().splitlines()
    if len(lines) > MAX_OUTPUT_LINES:
        lines = [
            *lines[:MAX_OUTPUT_LINES],
            f"... ({len(lines) - MAX_OUTPUT_LINES} more lines)",
        ]
    return {
        "ok": ok,
        "output": "\n".join(lines),
        "error": stderr.strip()[-1500:],
        **kept,
    }


# --------------------------------------------------------------------------- #
# pages
# --------------------------------------------------------------------------- #


def example_page(
    name: str, src: str, entry: dict[str, Any], paths: dict, ran: dict | None
) -> tuple[str, str]:
    from apiref import markdown

    doc = ast.get_docstring(ast.parse(src)) or ""
    title = doc.splitlines()[0].rstrip(".").replace("``", "`") if doc else name
    body = markdown("\n".join(doc.splitlines()[1:]))
    needs = entry.get("needs") or []
    lines = [
        f"# {title}",
        "",
        (
            f"`examples/{name}.py` · learning path: "
            f"[{paths[entry['path']].split(':')[0]}](index.md#{entry['path']})"
        ),
        "",
        "## Goal",
        "",
        f"{title}.",
        "",
        body,
        "",
        "## Prerequisites",
        "",
        "The base install, plus " + "; ".join(needs) + "."
        if needs
        else "The base install, nothing more.",
        "",
        "## Run it",
        "",
        "```bash",
        f"python examples/{name}.py",
        "```",
        "",
    ]
    if ran is not None:
        if ran.get("graph"):
            lines += [
                "## The graph",
                "",
                "The first graph the script runs:",
                "",
                f"![graph](media/{name}/graph.svg)",
                "",
            ]
        lines += ["## Expected output", ""]
        if ran["ok"]:
            lines += [
                "Captured when these pages were built; timings and some orderings vary.",
                "",
                "```text",
                ran["output"] or "(prints nothing)",
                "```",
                "",
            ]
        else:
            lines += [
                "The script did not finish when these pages were built (often an optional",
                "dependency is missing on the build machine):",
                "",
                "```text",
                ran["error"] or "(no message)",
                "```",
                "",
            ]
        if ran["images"] or ran["reports"]:
            lines += ["## What it produces", ""]
            lines += [f"![{img}](media/{name}/{img})" for img in ran["images"]]
            lines += [f"- [{rep}](media/{name}/{rep})" for rep in ran["reports"]]
            lines.append("")
    for heading, key in (
        ("Design decisions", "decisions"),
        ("Limitations", "limitations"),
    ):
        if entry.get(key):
            lines += [f"## {heading}", "", *[f"- {item}" for item in entry[key]], ""]
    lines += ["## Source", "", "```python", src.rstrip(), "```", ""]
    return title, "\n".join(lines)


def example_pages(run: bool) -> None:
    cat = catalog()
    paths, entries = cat["paths"], cat["examples"]
    os.makedirs(OUT, exist_ok=True)
    grouped: dict[str, list[tuple[str, str]]] = {key: [] for key in paths}
    for path in scripts():
        name = os.path.basename(path)[:-3]
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        ran = None
        if run:
            print(f"running {name} ...", flush=True)
            ran = run_example(path, name)
            if not ran["ok"]:
                print(
                    f"  {name} did not finish: {ran['error'].splitlines()[-1:] or ''}"
                )
        title, page = example_page(name, src, entries[name], paths, ran)
        with open(os.path.join(OUT, f"{name}.md"), "w", encoding="utf-8") as fh:
            fh.write(page)
        grouped[entries[name]["path"]].append((name, title))
    index = [
        "# Examples",
        "",
        "Every script under `examples/` runs on its own (`python examples/NN_name.py`).",
        "They are grouped below into learning paths; within a path, read them in order.",
        "The notebooks `sigflow_tour.ipynb` and `live_control.ipynb` cover the",
        "interactive side.",
        "",
    ]
    for key, label in paths.items():
        heading, _, description = label.partition(":")
        index += [
            f"## {heading} {{#{key}}}",
            "",
            description.strip().capitalize() + ".",
            "",
        ]
        index += ["| example | what it shows |", "|---|---|"]
        index += [f"| [{name}]({name}.md) | {title} |" for name, title in grouped[key]]
        index.append("")
    with open(os.path.join(OUT, "index.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(index))


def copies() -> None:
    for src, dst in (("CHANGELOG.md", "changelog.md"), ("README.md", "index.md")):
        with open(os.path.join(ROOT, src), encoding="utf-8") as fh:
            text = fh.read()
        with open(os.path.join(HERE, dst), "w", encoding="utf-8") as fh:
            fh.write(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run-examples",
        action="store_true",
        help="run every example and add its output, graph and images to its page",
    )
    args = parser.parse_args()
    import apiref

    copies()
    apiref.write(os.path.join(HERE, "reference", "api"))
    example_pages(run=args.run_examples)
    print("docs generated")


if __name__ == "__main__":
    main()
