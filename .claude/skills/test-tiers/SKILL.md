---
name: test-tiers
description: Run the aggregate test suite correctly. Three tiers (a narrowed
  edit loop, the full fast suite, everything including slow), the load-bearing
  flags each tier needs, and the two environment traps that make a green run a
  lie.
  Use when running tests, checking whether an edit broke anything, gating a
  version bump, choosing a pytest invocation, or when a run behaves oddly:
  selects nothing, selects everything, or a package that was installed has
  vanished.
argument-hint: [tier | path | -k expr]
---

# Running the aggregate suite

**The suite is not bloated. Measure before proposing a trim.** As of `1.0.0a396`
it is 5,413 fast cases in about 70 seconds, roughly 13 ms per test on FFT/numpy
bound work. The case count has more than doubled since a161 while wall clock
fell, so the whole fast suite is now a reasonable thing to run. If it *feels*
slow, measure first: `uv run pytest -m 'slow or not slow' --durations=40` before
proposing a cull.

## Two environment traps, check these first

A green run in the wrong environment proves nothing.

**1. There are two virtualenvs and a bare `uv run` may pick the wrong one.**
`.venv` is the development environment, the one to work in. `.doc-venv` builds
the docs only. An ambient `UV_PROJECT_ENVIRONMENT=.doc-venv` is set in this
shell, so a bare `uv run ...` silently resolves to `.doc-venv`. Both are
editable installs onto the same `src/`, so test *results* agree, but packages
installed into one are invisible to the other. That is exactly how a tool can
look installed and still be missing. When it matters, be explicit:

```
.venv/Scripts/python.exe -m pytest             # unambiguous on Windows
UV_PROJECT_ENVIRONMENT=.venv uv run pytest     # or force uv to the dev env
UV_PROJECT_ENVIRONMENT=.venv uv pip install <pkg>
```

**2. Sync with `--all-extras`, never a single extra.** `uv sync` is an *exact*
sync: it prunes anything outside the selected extras. The optional deps are
split across five extras (`dev`, `notebook`, `numba`, `massive`, `viz`), so
`uv sync --extra dev` **deletes** `zarr`, `holoviews`, `datashader` and `numba`,
which breaks the bivariate suites. It once broke `zarr`'s pytest plugin by
pruning `donfig` and `google-crc32c` mid-run.

```
uv sync --all-extras
```

Add a one-off tool with `uv pip install <pkg>`. `uv sync --inexact` keeps
extraneous packages if you ever need that fallback.

`UV_LINK_MODE=copy` is required whenever invoking `uv` here. The repo's
`.claude/settings.local.json` sets it for the harness already, so this is only a
concern in a plain shell.

## The three tiers

Use the first tier that covers the change.

### Tier 1, the edit loop

Narrow the selection by hand to the tests the edit can plausibly touch:

```
UV_PROJECT_ENVIRONMENT=.venv uv run pytest tests/test_pnl.py
```

One file, or one case with `::test_name`. Also `-k "pnl and not engine"` to
select by expression, `--lf` to rerun last failures, `--ff` for failures first,
`-x` to stop at the first. There is no setup and nothing to keep in sync, and if
the selection misses something the full fast suite behind it is under 90 s.

Add `-n0 --dist no` when you want a breakpoint or deterministic single process
ordering; those override `-n auto --dist loadgroup` in `addopts`. `-p no:xdist`
does **not** work, it makes those `addopts` unparseable.

### Tier 2, pre-commit

```
UV_PROJECT_ENVIRONMENT=.venv uv run pytest
```

The full fast suite, parallel via `-n auto`, with `slow` deselected. This is the
gate for "am I done", **not** an edit-loop tool. Running it eight times in one
session is the mistake this skill exists to prevent.

Always finish with tier 2 before declaring a change done. Do not rely on
eyeballing blast radius as the only check.

### Tier 3, the version-bump gate

```
UV_PROJECT_ENVIRONMENT=.venv uv run pytest -m 'slow or not slow'
```

Everything, about six minutes. Run it **once**, at the commit boundary. See the
`version-bump` skill, which calls this tier as its gate.

## Two extra gates, run at the matching bump

**Numerics.** At a bump that touches numerics:

```
uv run pytest -m 'slow or not slow' -W error::RuntimeWarning
```

The library emits no `RuntimeWarning` of its own as of `1.0.0a220`
(`[RuntimeWarning-Census]`), so a new one is a real finding. Either it is
arithmetic whose result is discarded, which wants an `np.errstate` guard and a
`Notes` paragraph saying why, or it is a `NaN` reaching an answer, which wants a
fix. This is deliberately **not** in `addopts`: a third-party release could
break the everyday loop, and one bivariate case flakes under it
(`[Bivariate-Gate-Flake]`).

**Grammar snapshot.** After any grammar or transformer change:

```
uv run python tests/capture_spec_snapshot.py
```

The snapshot is not frozen and re-capturing is routine. It is a change detector,
not a correctness oracle: it cannot prove the parser right, since it is captured
from the parser it checks. What it gives you is an edit's blast radius as a diff
you must read and accept deliberately. **A line you did not expect to move is a
finding.** Do not regenerate and commit without reading it.

## Markers

- **`slow`, fast by default.** `addopts` carries `-m 'not slow'`. The three
  bleeding-edge bivariate suites (`test_bivariate.py`,
  `test_massive_bivariate.py`, `test_reins_bivariate.py`) are tagged at module
  level via `pytestmark = pytest.mark.slow`. They hold the two multi-minute
  monsters that set the suite's wall-clock floor. Run just those with `-m slow`.
  To quarantine a new expensive case, tag it `@pytest.mark.slow`, or add
  `pytestmark` for a whole module. Find candidates with
  `uv run pytest -m 'slow or not slow' --durations=20`.
- **`xdist_group`, about memory, not time.** `addopts` carries
  `--dist loadgroup`. Ungrouped tests distribute as before, but every test
  sharing an `xdist_group` name goes to one worker and so runs sequentially
  against its group-mates. `test_bivariate.py` uses it: each case allocates a 2-D
  FFT grid, and several running concurrently once exhausted memory, surfacing as
  a numpy allocation failure in a test whose own grid was only 64x64. The
  pressure was its neighbours. Reach for `xdist_group` when tests are
  individually fine but collectively too large; reach for `slow` when a test is
  simply long. Costs about 10% on the gate.
- `--strict-markers` is on, so every marker must be registered in
  `[tool.pytest.ini_options].markers`.

## Reporting a run

State the tier used and the actual command. If a tier was skipped, say so. A
failing run is reported with its output, never summarized as "mostly green".
