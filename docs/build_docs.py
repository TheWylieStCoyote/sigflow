"""Generate the example pages (and copy README/CHANGELOG) for the mkdocs site.
Run: python docs/build_docs.py && mkdocs build"""

import ast
import glob
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))


def example_pages() -> None:
    index = [
        "# Examples",
        "",
        "Every script under `examples/` runs on its own (`python examples/NN_name.py`). "
        "The notebooks `sigflow_tour.ipynb` and `live_control.ipynb` cover the interactive side.",
        "",
        "| example | what it exercises |",
        "|---|---|",
    ]
    for path in sorted(glob.glob(os.path.join(ROOT, "examples", "[0-9]*.py"))):
        name = os.path.basename(path)[:-3]
        src = open(path).read()
        doc = ast.get_docstring(ast.parse(src)) or ""
        title = doc.splitlines()[0].rstrip(".") if doc else name
        body = "\n".join(doc.splitlines()[1:]).strip()
        page = [
            f"# {title}",
            "",
            f"`examples/{name}.py`",
            "",
            body,
            "",
            "## Source",
            "",
            "```python",
            src.rstrip(),
            "```",
            "",
        ]
        with open(os.path.join(HERE, "examples", f"{name}.md"), "w") as fh:
            fh.write("\n".join(page))
        index.append(f"| [{name}]({name}.md) | {title} |")
    with open(os.path.join(HERE, "examples", "index.md"), "w") as fh:
        fh.write("\n".join(index) + "\n")


def copies() -> None:
    shutil.copy(os.path.join(ROOT, "CHANGELOG.md"), os.path.join(HERE, "changelog.md"))
    readme = open(os.path.join(ROOT, "README.md")).read()
    with open(os.path.join(HERE, "index.md"), "w") as fh:
        fh.write(readme)


if __name__ == "__main__":
    example_pages()
    copies()
    print("docs generated")
