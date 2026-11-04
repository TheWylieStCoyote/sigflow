"""Every Python sample in README.md runs as written (each as its own script).

A sample that demonstrates an error ends with a comment naming it, such as
``# TypeMismatch: ...``; it must fail with that error and that message. A sample that
uses a GPU (``to_device``) is skipped without CuPy and a CUDA device.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
BLOCKS = [
    (README[: m.start()].count("\n") + 1, m.group(1))
    for m in re.finditer(r"^```python\n(.*?)^```", README, re.DOTALL | re.MULTILINE)
]


def _has_gpu() -> bool:
    try:
        import cupy  # type: ignore[import-not-found]

        return bool(cupy.cuda.runtime.getDeviceCount())
    except Exception:  # no CuPy, no driver, no device: all mean no GPU
        return False


EXPECTED_ERROR = re.compile(r"^# (\w+(?:Mismatch|Error)): (.+)$", re.MULTILINE)


@pytest.mark.parametrize("line,code", BLOCKS, ids=[f"line {n}" for n, _ in BLOCKS])
def test_readme_sample_runs(line, code, tmp_path):
    if "to_device(" in code and not _has_gpu():
        pytest.skip("needs CuPy and a CUDA device")
    script = tmp_path / "sample.py"
    script.write_text(code, encoding="utf-8")
    env = dict(os.environ, MPLBACKEND="Agg")
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(ROOT / "src"), env.get("PYTHONPATH")])
    )
    done = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,  # the exit status is what the test examines
    )
    expected = EXPECTED_ERROR.search(code)
    if expected is None:
        assert done.returncode == 0, f"README line {line}:\n{done.stderr[-2000:]}"
        return
    kind, message = expected.groups()
    assert done.returncode != 0, f"README line {line}: expected {kind}, it ran"
    last = done.stderr.strip().splitlines()[-1]
    assert last.split(":", 1)[0].endswith(kind), f"README line {line}: {last}"
    assert message.rstrip(" .") in last, f"README line {line}:\n{last}"


def test_every_sample_is_found():
    assert len(BLOCKS) >= 13
