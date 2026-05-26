# Plan A — `aggregate.style` module

**Status:** ready to execute.
**Depends on:** nothing.
**Unblocks:** every plot in the apiweb (Plan C), shares a single source of truth with the docs build.

## Goal

Extract the plotting style currently defined inline as `knobble_fonts(color=True)` in `docs/conf.py` into a first-class submodule `aggregate.style`. The same style should drive:

1. The Sphinx docs build (replaces `knobble_fonts` call in `conf.py`).
2. Any plotting in the future apiweb (via a scoped `rc_context`).
3. User notebooks that want the house look (`import aggregate.style; aggregate.style.use()`).

Color-only — the legacy B&W branch in `knobble_fonts` is dropped (paperless commitment).

## Deliverables

- New: `src/aggregate/style.py`
- New: `src/aggregate/data/aggregate.mplstyle`
- Modified: `src/aggregate/__init__.py` — no top-level re-export (per CLAUDE.md "submodule access only" convention); just ensure the submodule is importable.
- Modified: `pyproject.toml` — extend `[tool.setuptools.package-data]` to include the `.mplstyle` file.
- Modified: `docs/conf.py` — remove inline `knobble_fonts`, call `aggregate.style.use()` instead.
- New: `tests/test_style.py` — minimal smoke tests.

## Module API

Resource discovery uses `importlib.resources` (not `pathlib.Path` arithmetic) so the bundled `.mplstyle` is located through the package metadata, not the filesystem layout. Works correctly for editable installs, wheel installs, and zip-imported installs.

Rather than read the `.mplstyle` file on every call, parse it once at module load into a frozen `rcParams` dict, then apply that dict via `mpl.rcParams.update(...)` (for `use`) or `mpl.rc_context(...)` (for `context`). This sidesteps the wheel-vs-zip-install path question entirely — we only need a real path during the one-time load.

```python
# src/aggregate/style.py
from contextlib import contextmanager
from importlib.resources import files, as_file
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

_RESOURCE = files("aggregate").joinpath("data/aggregate.mplstyle")

def _load_rc_params() -> dict:
    """Parse the bundled .mplstyle once into an rcParams dict."""
    with as_file(_RESOURCE) as p:
        return mpl.rc_params_from_file(
            str(p),
            fail_on_error=True,
            use_default_template=False,
        )

_STYLE_PARAMS = _load_rc_params()

# Pandas display options that travel with the style (not expressible in .mplstyle).
_PANDAS_OPTIONS = {
    "display.width": 120,
}

def use(pandas: bool = True) -> None:
    """Apply aggregate's house style globally.

    Intended for notebook / docs / interactive use. Mutates global state
    (``plt.rcParams`` and optionally ``pd.options``).

    Parameters
    ----------
    pandas : bool, default True
        Also set pandas display options. Pass ``False`` to leave pandas
        configuration untouched.
    """
    mpl.rcParams.update(_STYLE_PARAMS)
    if pandas:
        for key, value in _PANDAS_OPTIONS.items():
            pd.set_option(key, value)

@contextmanager
def context(**overrides):
    """Scoped style for server-side / library code.

    Restores prior ``rcParams`` on exit. Does not touch ``pd.options``
    (deliberately — server code shouldn't mutate global pandas state).

    Parameters
    ----------
    **overrides
        Optional ``rcParams`` overrides layered on top of the base style.
        Used by the apiweb to bump up ``figure.figsize`` / down ``figure.dpi``
        for screen rendering without forking the .mplstyle file.

    Yields
    ------
    None

    Examples
    --------
    >>> with aggregate.style.context():
    ...     fig, ax = plt.subplots()
    ...     fig.savefig(buf)

    >>> with aggregate.style.context(**{"figure.figsize": (5.5, 3.5),
    ...                                  "figure.dpi": 100}):
    ...     ...
    """
    params = {**_STYLE_PARAMS, **overrides}
    with mpl.rc_context(params):
        yield

def rc_params() -> dict:
    """Return a copy of the bundled style as an ``rcParams`` dict.

    Useful for downstream tooling that wants to compose with other styles
    or inspect the values programmatically.
    """
    return dict(_STYLE_PARAMS)
```

Notes:
- `use()` mutates global state — that's the point. Documented.
- `context()` is the apiweb path. Per-request scope, no leak. Accepts overrides so the apiweb can bump figsize up / dpi down without forking the file.
- `rc_params()` is for inspection / composition (replaces the earlier `path()` helper — returning a path doesn't compose well when the install might be zip-based).
- No `aggregate.style` import in `aggregate/__init__.py` body — keep it lazy. Users do `import aggregate.style; aggregate.style.use()`.

## Apiweb usage pattern (forward reference)

In Plan C the plot endpoint will do:

```python
WEB_OVERRIDES = {
    "figure.figsize": (5.5, 3.5),    # wider than docs default
    "figure.dpi": 100,               # screen, not print
    "savefig.dpi": 100,              # screen, not print
}

with aggregate.style.context(**WEB_OVERRIDES):
    fig = obj.plot(...)
    fig.savefig(buf, format="png")
```

Docs canonical values stay in `.mplstyle`; the web is the one who adjusts.

## `aggregate.mplstyle` content

Translation of `knobble_fonts(color=True)` with enhancements. Color-only, no cycler munging.

```ini
# aggregate house style — color, screen/PDF rendering
# Source of truth used by docs build and aggregate.apiweb.

# --- fonts ---
font.size:           9
font.family:         serif
font.serif:          STIX Two Text, Times New Roman, DejaVu Serif
font.sans-serif:     Myriad Pro, Segoe UI, DejaVu Sans
font.monospace:      Ubuntu Mono, Cascadia Mono, DejaVu Sans Mono
mathtext.fontset:    stixsans

# --- legend ---
legend.fontsize:     x-small
legend.facecolor:    lightsteelblue
legend.edgecolor:    lightsteelblue
legend.frameon:      True

# --- axes / figure background ---
axes.facecolor:      lightsteelblue
figure.facecolor:    aliceblue
axes.edgecolor:      black
axes.linewidth:      0.6
axes.grid:           True
grid.color:          white
grid.linewidth:      0.5
grid.alpha:          0.8

# --- figure sizing & resolution (docs canonical; apiweb overrides at render time) ---
figure.figsize:      3.5, 2.45
figure.dpi:          300
figure.constrained_layout.use: True

# --- savefig ---
savefig.dpi:         300
savefig.bbox:        tight
savefig.pad_inches:  0.05
savefig.facecolor:   inherit
savefig.transparent: False

# --- lines ---
lines.linewidth:     1.2
lines.markersize:    4

# --- ticks ---
xtick.direction:     out
ytick.direction:     out
xtick.labelsize:     small
ytick.labelsize:     small
```

Open knobs you may want to tune before execution:
- **`figure.constrained_layout.use: True`** — modern replacement for `tight_layout` calls. New behavior vs current docs. If any existing figure is custom-laid-out, this may shift it slightly. Worth turning on but flag as a watch-item during the docs rebuild.
- **`axes.prop_cycle`** — current code sets a cycler in the B&W branch only. Color mode leaves the default mpl `tab10` cycler. Leaving as-is unless you want a curated palette.

(Resolved: `figure.figsize: 3.5, 2.45` and `figure.dpi: 300` are docs canonical; the apiweb overrides up/down at render time via `aggregate.style.context(**overrides)`.)

## Migration of `docs/conf.py`

Current (lines 1–80):
```python
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
sys.path.insert(0, os.path.abspath('../src'))
import aggregate as agg

FIG_W = 3.5
FIG_H = 2.45
FONT_SIZE = 9
LEGEND_FONT = 'x-small'
PLOT_FACE_COLOR = 'lightsteelblue'
FIGURE_BG_COLOR = 'aliceblue'
VALIDATION_EPS = 1e-4
RECOMMEND_P = 0.99999

def knobble_fonts(color=False):
    ...    # ~50 lines

knobble_fonts(True)
plt.rcParams['figure.dpi'] = 300
```

After:
```python
sys.path.insert(0, os.path.abspath('../src'))
import aggregate as agg
import aggregate.style

# constants still referenced elsewhere in conf.py / by extensions
VALIDATION_EPS = 1e-4
RECOMMEND_P = 0.99999

aggregate.style.use()
```

(Sizing/color constants `FIG_W`, `FIG_H`, `PLOT_FACE_COLOR`, etc. — grep `docs/` first to confirm they're not referenced elsewhere. If they are, leave them as module-level constants pointing at the same values; otherwise delete.)

## Verification steps

1. `uv sync` — no-op, but confirms env is current.
2. `uv run python -c "import aggregate.style; aggregate.style.use(); import matplotlib.pyplot as plt; print(plt.rcParams['axes.facecolor'])"` → should print `lightsteelblue`.
3. `uv run python -c "import aggregate.style, matplotlib.pyplot as plt; plt.rcdefaults(); pre = plt.rcParams['axes.facecolor']; ctx = aggregate.style.context();
   ctx.__enter__(); inside = plt.rcParams['axes.facecolor']; ctx.__exit__(None, None, None); after = plt.rcParams['axes.facecolor']; print(pre, inside, after)"` → should show `white lightsteelblue white` confirming scoped restore.
4. `uv run pytest tests/test_style.py` — unit tests pass.
5. **Docs rebuild** (manual, outside iteration loop per CLAUDE.md): `cd docs && uv run make.bat html`. Spot-check 3–4 figures across chapters for visual continuity. *Watch item*: the `constrained_layout` change may shift figure spacing slightly versus today's `tight_layout`-style rendering.

## Test sketch (`tests/test_style.py`)

```python
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import pytest

import aggregate.style

def test_rc_params_returns_dict():
    params = aggregate.style.rc_params()
    assert isinstance(params, dict)
    assert "axes.facecolor" in params

def test_use_sets_facecolor():
    aggregate.style.use(pandas=False)
    assert plt.rcParams["axes.facecolor"] == "lightsteelblue"
    assert plt.rcParams["figure.facecolor"] == "aliceblue"

def test_use_pandas_toggle():
    pd.reset_option("display.width")
    default = pd.get_option("display.width")
    aggregate.style.use(pandas=False)
    assert pd.get_option("display.width") == default
    aggregate.style.use(pandas=True)
    assert pd.get_option("display.width") == 120

def test_context_scopes_rcparams():
    plt.rcdefaults()
    before = plt.rcParams["axes.facecolor"]
    with aggregate.style.context():
        inside = plt.rcParams["axes.facecolor"]
    after = plt.rcParams["axes.facecolor"]
    assert inside == "lightsteelblue"
    assert before == after  # restored

def test_context_overrides():
    plt.rcdefaults()
    with aggregate.style.context(**{"figure.figsize": (5.5, 3.5)}):
        assert tuple(plt.rcParams["figure.figsize"]) == (5.5, 3.5)
        assert plt.rcParams["axes.facecolor"] == "lightsteelblue"  # base still applies
```

## `pyproject.toml` change

```toml
[tool.setuptools.package-data]
aggregate = ["*.lark", "agg/*.agg", "data/*.mplstyle"]
```

## Out of scope (explicit non-goals)

- Migrating other docs-build helpers (the `FIG_W`/`FIG_H` constants if used in figure code) — handled if grep shows references, otherwise deleted.
- B&W / print-mode variant — dropped per the paperless decision; can be re-added as `aggregate-print.mplstyle` later if a publisher requirement returns.
- A `set_pandas()` function — the `use(pandas=...)` toggle is enough.
- Updating `README.rst` references to `knobble_fonts` — those are historical changelog entries describing the removal, not active docs; leave them.

## File-by-file checklist (for execution)

1. Create `src/aggregate/data/aggregate.mplstyle` with the content above.
2. Create `src/aggregate/style.py` with the API above.
3. Edit `pyproject.toml` package-data line to include `data/*.mplstyle`.
4. Grep `docs/` for `FIG_W`, `FIG_H`, `FONT_SIZE`, `LEGEND_FONT`, `PLOT_FACE_COLOR`, `FIGURE_BG_COLOR` references outside `conf.py`; preserve any that are used.
5. Edit `docs/conf.py` per the migration above.
6. Create `tests/test_style.py`.
7. Run `uv run pytest tests/test_style.py -v`.
8. Run the four-line verification snippet from "Verification steps".
9. Hand-off note: tell the user docs need a manual rebuild to visually confirm the figure continuity.

## Recovery / rollback

If the docs rebuild reveals a visual regression: revert `docs/conf.py` only (restore `knobble_fonts` and its call). The new `aggregate.style` module can stay — it'll just be unused by docs until the regression is diagnosed. Low-risk, easy to back out.
