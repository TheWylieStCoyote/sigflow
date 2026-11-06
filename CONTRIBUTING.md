# Contributing

## Set up

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]" pyright
```

`dev` brings the test, type-checking, notebook and docs tools and every optional
dependency the tests use. Python 3.11 or later.

## The checks CI runs

```bash
.venv/bin/ruff check src tests examples
.venv/bin/ruff format --check src tests examples
.venv/bin/mypy src/sigflow tests examples
.venv/bin/pyright --pythonpath .venv/bin/python
.venv/bin/python -m pytest -q tests examples/18_testing_pipelines.py
for f in examples/[0-9]*.py; do .venv/bin/python "$f" > /dev/null || echo "FAILED $f"; done
```

- **pyright** must be pointed at the virtual environment (`--pythonpath`);
  otherwise it may pick the system Python and report the optional imports as
  missing.
- **Notebooks are not formatted with ruff.** They are generated: edit
  `examples/make_notebook.py` or `examples/make_live_notebook.py` and run it,
  rather than the `.ipynb` files.
- **README samples are tests.** `tests/test_readme.py` runs every ```` ```python ````
  block of `README.md` as its own script. A sample that shows an error ends
  with a comment naming it, `# TypeMismatch: ...`, and must fail with exactly
  that.
- **Some tests need hardware or other interpreters** and skip without them:
  CuPy and a CUDA device; Python 3.14 for `backend="interpreter"`; a
  free-threaded build (`python3.14t`) for `backend="freethreaded"`. CI runs
  3.14 and 3.14t.
- **Two tests use fixed local ports** (UDP 9931 and ZeroMQ 5599): two test runs
  on one machine at the same time can collide on them.

## Changes

Add an entry to the `## Unreleased` section of `CHANGELOG.md` under New,
Changed or Fixed. Public API needs a docstring: the reference pages are built
from them.

## The docs site

`docs/` is an MkDocs Material site. Some pages are written by hand, the rest
are **generated and not checked in** (they are in `.gitignore`):

| page | source | made by |
|---|---|---|
| `docs/concepts/`, `docs/guide/`, `docs/getting-started/`, `docs/reference/running.md`, `docs/performance.md` | written by hand | you |
| `docs/index.md` | `README.md` | `docs/build_docs.py` |
| `docs/changelog.md` | `CHANGELOG.md` | `docs/build_docs.py` |
| `docs/reference/api/*.md` | the package's docstrings and signatures | `docs/apiref.py` |
| `docs/examples/*.md` | each script's docstring and `examples/catalog.toml` | `docs/build_docs.py` |

Build and preview:

```bash
.venv/bin/python docs/build_docs.py                 # seconds: pages without example output
.venv/bin/python docs/build_docs.py --run-examples  # about a minute: runs every example
.venv/bin/mkdocs serve                               # http://127.0.0.1:8000
.venv/bin/mkdocs build --strict                      # what CI checks
```

With `--run-examples` each example runs in a scratch directory and its page
gains the printed output, a diagram of the first graph it runs, and the images
and HTML reports it writes (`docs/examples/media/`). The published site is
built that way by the `docs` workflow; the CI check builds without it.

### A new example

1. Add `examples/NN_name.py`. Its docstring's first line is the page title and
   goal; the rest is the page's introduction. It must run on its own
   (`python examples/NN_name.py`), and anything it needs beyond the base install
   must be optional: skip that part with a message when it is missing.
2. Add an `[examples.NN_name]` entry to `examples/catalog.toml`: its learning
   `path`, what it `needs`, its `decisions` and `limitations`.
   `tests/test_docs.py` fails until every script has an entry and every entry a
   script.
3. Files it writes (`.svg`, `.gif`, `.png`, `.html`) next to where it runs
   appear on its page.
4. Add a row to the table in `examples/README.md`.

### New public API

Add the name to a page in `docs/apiref.py` (`PAGES`); `tests/test_docs.py`
fails while a name in `sigflow.__all__` is on no page. A new option of `run()`
or `start()` also goes in the table in `docs/reference/running.md`, which the
same test checks against the code.
