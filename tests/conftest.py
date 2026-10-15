"""Make the tests runnable without installing the package (adds src/ to sys.path)
and keep matplotlib headless so plot tests never try to open a window."""
# mypy: disable-error-code="valid-type, misc, syntax, return-value, union-attr, index"

import os
import sys

import matplotlib

matplotlib.use("Agg")
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)
