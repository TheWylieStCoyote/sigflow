"""The generated docs keep up with the code: the API reference lists every public
name, every example has a catalog entry (and every entry an example), and the
options table names every option the entry points take."""

from __future__ import annotations

import glob
import os
import re
import sys
import tomllib
from pathlib import Path

import sigflow
from sigflow.runtime import RUN_OPTIONS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs"))

import apiref  # docs/ is not a package: imported from the path above


def test_the_reference_lists_every_public_name():
    covered = apiref.covered()
    missing = [
        name
        for name in sigflow.__all__
        if name != "io"  # a module, with a page of its own
        and f"sigflow.{name}" not in covered
        and f"sigflow.dims.{name}" not in covered
    ]
    assert not missing, f"add these to docs/apiref.py PAGES: {missing}"


def test_every_reference_page_renders(tmp_path):
    written = apiref.write(str(tmp_path))
    for stem in written:
        text = (tmp_path / f"{stem}.md").read_text(encoding="utf-8")
        assert text.startswith("# ") and "###" in text, stem


def test_docstrings_become_markdown():
    doc = """Run it with :func:`run`, or ``iterate``.

    ``on_error``   what to do
                   with errors
    """
    md = apiref.markdown(doc)
    assert "Run it with `run`, or `iterate`." in md
    assert "```text\non_error   what to do" in md


def test_every_example_has_a_catalog_entry_and_back():
    with open(ROOT / "examples" / "catalog.toml", "rb") as fh:
        cat = tomllib.load(fh)
    scripts = {
        os.path.basename(p)[:-3]
        for p in glob.glob(str(ROOT / "examples" / "[0-9]*.py"))
    }
    assert set(cat["examples"]) == scripts
    for name, entry in cat["examples"].items():
        assert entry["path"] in cat["paths"], name
        assert entry.get("decisions") and entry.get("limitations"), name


def test_the_options_table_names_every_option():
    table = (ROOT / "docs" / "reference" / "running.md").read_text(encoding="utf-8")
    rows = set(re.findall(r"^\| `(\w+)` \|", table, re.MULTILINE))
    expected = (set(RUN_OPTIONS) - {"control"}) | {"paused", "keep", "name", "replace"}
    assert rows == expected


def test_the_version_is_the_packages():
    from importlib.metadata import version

    assert sigflow.__version__ == version("sigflow")


def test_every_nav_page_exists():
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    generated = ("index.md", "changelog.md", "examples/index.md", "reference/api/")
    for page in re.findall(r":\s+(\S+\.md)\s*$", nav, re.MULTILINE):
        if page.startswith(generated) or page in generated:
            continue
        assert (ROOT / "docs" / page).exists(), page
