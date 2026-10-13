# Testing a signal chain

Blocks are functions: test them with plain asserts. Graphs apply to fixtures:
`chain(frames)` returns output frames to assert on, including their provenance
and annotations. `check="all"` turns the contract check into an assertion for
every frame. A fake clock is an async source whose time the test advances.
Parametrise over a grid, pin a golden output. `examples/18_testing_pipelines.py`
is a complete, pytest-runnable template; `tests/` in the repository shows the
same patterns at larger scale.
