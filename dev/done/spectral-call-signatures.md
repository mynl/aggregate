# Plan — Distortion constructor reform (spectral.py)

## Context

The `Distortion` base class in `src/aggregate/spectral.py` currently has a generic constructor `Distortion(name, shape, r0=0.0, df=None, col_x='', col_y='', display_name='')`. Every kind squeezes its natural parameters into these slots:

- `shape` is a float for ph/wang/dual/tvar/ccoc/etc., a list for wtdtvar (the ps), a list for beta (a, b), and a list-of-distortions for minimum/mixture.
- `df` is `[p0, p1]` for bitvar, weights for wtdtvar, a DataFrame for convex, `[x0, x1]` for power. The name comes from the now-removed t-Student kind ("degrees of freedom") and is a vestige.
- `r0` is only meaningful for the five mass-at-zero kinds (ccoc, cll, clin, lep, ly) but appears in the universal signature.
- `col_x`, `col_y` are convex-only but appear in the universal signature.

Goal: give each subclass a natural-named `__init__` while preserving:

- The factory call `Distortion('ph', a=0.7)` and `Distortion('ph', 0.7)`.
- The static convenience factories `Distortion.ph(0.7)`, `Distortion.tvar(p)`, etc.
- Direct subclass construction `PHDistortion(a=0.7)`.
- The internal contract that `self.shape` is the scalar Newton iterates against for the calibratable kinds (ph, wang, dual, tvar, ccoc, ly, clin, lep, cll). Calibration code stays untouched.

Final target version bump: 1.0.0a12 → 1.0.0a13.

## Decisions made (consolidating prior + this round)

1. **`shape` stays as the load-bearing calibration variable** for scalar-param kinds. Natural names (`a`, `lam`, `p`, `b`, `r`, etc.) are exposed as read/write properties aliased to `self.shape`.
2. **`df` is eliminated** as a constructor parameter. Each multi-param kind owns its `__init__` with named structural kwargs.
3. **`r0` lives only on the five mass-at-zero kinds** (ccoc, cll, clin, lep, ly), declared per their `__init__`.
4. **DecL parser translates** `(shape, df)` to per-kind kwargs in `parser.py`'s transformer, so `underwriter.py:422`'s `Distortion(**spec)` still works without changes.
5. **No backwards-compat shims** per CLAUDE.md. Old `Distortion('bitvar', w, df=[p0, p1])` calls break — every call site is updated.
6. **Static factories preserved.** They become one-line delegators.
7. **CCoC: kwarg-only `*, d=None, r=None`.** Both default to `None`; pass exactly one. Positional raises `TypeError`. Zen of Python: explicit > implicit. Members: `self.d`, `self.r`, `self.v = 1 - d`, `self.shape = r`. Migration: `Distortion('ccoc', xx)` callers passed return → rewrite as `Distortion('ccoc', r=xx)`. `Distortion.ccoc(d)` factory unchanged.
8. **`ConvexDistortion` is REMOVED.** Per user: "convex is not a distortion, it's a constructor." Replaced by module-level `convex_distortion(s, gs, *, mass=0, display_name='')` that takes pre-computed s and g(s) **arrays** (caller handles any DataFrame / column extraction pre-call) and returns a `WtdTVaRDistortion`. `bagged_distortion` and `convex_example` are also relocated to module level (not `Distortion` staticmethods) and rewired to use `convex_distortion`. `s_gs_distortion` is merged into `convex_distortion` — same signature, same behavior. `average_distortion` is **deleted entirely** (user doesn't recall a use case). `_plot_decorations` is also deleted (only convex used it). All module-level helpers live in a clearly-delineated block at the bottom of `spectral.py` with a banner comment.
9. **wtdtvar normalization: tight.** `np.isclose(sum(wts), 1.0)` → normalize silently (clean up FP noise). Otherwise `ValueError` with the sum value in the message. No warning path, no auto-fix of grossly-wrong inputs.
10. **`power` is NOT calibratable.** Drop `_calibration_init_shape` and `strict_pricing=True`. Calibration code skips it.
11. **`ly` parameter is `r`, not `consumption`.** Docstring credits Don Mango and explains `r` is the consumption rate.
12. **Newton-Raphson: keep hand-rolled.** Considered `scipy.optimize.newton`/`root_scalar` and declined. See "Considered, declined" section below.
13. **Docstrings: NumPy style for every docstring touched.** No invented content. Empty `:param x:` lines deleted, with the list of dropped params surfaced separately for user triage. Docstrings on methods NOT modified by this refactor are left alone (deferred to the broader sweep).
14. **Hash: per-subclass `_id_fields()`** method returning the tuple to hash. Each kind declares its own identifying state.

## Per-kind parameter names

| Kind | Natural params | Storage | Notes |
|---|---|---|---|
| `ccoc` | kwarg-only `*, d=, r=` | `self.d`, `self.r`, `self.v=1-d`, `self.shape=r` | Pass exactly one of `d=` or `r=`; positional raises TypeError |
| `ph` | `a=` | `self.shape=a`; property `.a` | was `rho`; user renames to `a` |
| `wang` | `lam=` | `self.shape=lam`; property `.lam` | ASCII, scientific-Python convention (cf. `scipy.stats.poisson(lam=)`). Unicode `λ` alias dropped — not PEP-style |
| `dual` | `b=` | `self.shape=b`; property `.b` | gives nice `a`/`b` symmetry with ph |
| `tvar` | `p=` | `self.shape=p`; property `.p` | matches existing `Distortion.tvar(p)` factory |
| `bitvar` | `p0=, p1=, w1=` | `self._p0`, `self._p1`, `self._w1`, `self.shape=w1` | `w1` stresses pairing with `p1` |
| `wtdtvar` | `ps=, wts=` | `self._ps`, `self._wts`, `self.shape=ps` | enforce `len(ps)==len(wts)`; normalize if `np.isclose(sum(wts), 1)` else `ValueError` |
| `beta` | `a=, b=` | `self._a`, `self._b`, `self.shape=[a,b]` | not calibratable; constraints 0<a≤1, b≥1 |
| `power` | `x0=, x1=, alpha=` | `self._x0`, `self._x1`, `self.shape=alpha` | x0<x1; NOT calibratable (no `_calibration_init_shape`, `strict_pricing=False`) |
| `cll` | `r0=, b=` | `self.r0`, `self.shape=b`; property `.b` | b is shape (Newton iterates on it); r0 is mass |
| `clin` | `r0=, slope=` | `self.r0`, `self.shape=slope`; property `.slope` | r0 is mass at 0 |
| `lep` | `r0=, r=` | `self.r0`, `self.shape=r`; property `.r` | r is target return; r0 is rental rate |
| `ly` | `r0=, r=` | `self.r0`, `self.shape=r`; property `.r` | r0 is occupancy; **docstring credits Don Mango** and notes that `r` is the consumption rate |
| `minimum` | `distortions=` (list) | `self._distortions`, `self.shape=distortions` | list of Distortion instances |
| `mixture` | `distortions=, wts=None` | `self._distortions`, `self._wts`, `self.shape=distortions` | wts defaults to uniform |

Property aliases are read/write where the underlying is `self.shape`, read-only where the underlying is a private `_attr` (multi-param kinds).

## Convex distortion: removal + module-level helpers block

Remove `ConvexDistortion` entirely from `spectral.py`. The replacement is a clearly-delineated **module-level helpers block** at the bottom of the file containing **three** functions (none on `Distortion`):

```python
# ===========================================================================
# Distortion construction helpers (module-level functions, NOT subclasses)
# ===========================================================================
# Functions that build a Distortion from sample data. They RETURN a
# Distortion instance (typically WtdTVaRDistortion) rather than being
# distortion kinds in their own right — kept out of the class so the
# Distortion namespace stays focused on actual kinds.

def convex_distortion(s, gs, *, mass=0, display_name=''):
    """
    Construct a distortion as the upper convex envelope of (s, gs) points.

    Returns a ``WtdTVaRDistortion`` whose piecewise-linear g matches the
    convex hull of the supplied points. Caller does any DataFrame /
    column-name pre-extraction; this function just takes two arrays.

    Parameters
    ----------
    s : array_like
        x-coordinates. Padded with 0 and 1 if missing.
    gs : array_like
        Corresponding g(s) values, same length as ``s``.
    mass : float, optional
        Point mass at zero. Default 0.
    display_name : str, optional
        Label override.

    Returns
    -------
    WtdTVaRDistortion
    """
    # 1. Pad (0,0) and (1,1) if missing.
    # 2. ConvexHull on the (s, gs) point set; extract upper-hull knots
    #    ordered by s ascending.
    # 3. Convert ordered knots → (ps, wts) by slope decomposition.
    # 4. Return WtdTVaRDistortion(ps=ps, wts=wts, display_name=...).

def bagged_distortion(data, proportion, samples, *,
                      el_col='EL', spread_col='Spread', display_name=''):
    """Bootstrap-averaged convex distortion from (EL, Spread) tabular data.

    Repeatedly subsamples ``data``, builds the convex envelope of each
    sample via ``convex_distortion``, averages g(s) across samples on a
    uniform grid, returns the final averaged distortion (wtdtvar-shaped).
    """
    # body: for each sample, extract (s_pts, gs_pts) arrays, call
    # convex_distortion, evaluate on uniform grid, average, final call to
    # convex_distortion(s_grid, avg_gs, display_name=display_name)

def convex_example(source='bond'):
    """Example convex distortion built from bundled BIS yield-curve or
    cat-bond ROL-vs-EL data. ``source ∈ {'bond', 'cat'}``."""
    # extracts (s, gs) from the hardcoded example data, returns
    # convex_distortion(s, gs, display_name=...)
```

`s_gs_distortion` (existing staticmethod) is **merged into `convex_distortion`** — same signature, same purpose; no separate function. The `display_name` kwarg makes the merge transparent.

Knock-on changes:

- `ConvexDistortion` class and its `'convex'` entry in `_registry`: **gone**.
- `Distortion.bagged_distortion`, `Distortion.s_gs_distortion`, `Distortion.convex_example`: **deleted** from the class; replaced by module-level functions above.
- `Distortion.average_distortion`: **deleted entirely** (user doesn't recall a use case; not worth migrating).
- `_plot_decorations` hook on the base: **deleted** (only convex used it; presentation concern, not core behavior).
- `Distortion.available_distortions()`: no longer lists `'convex'` — desired outcome.

The slopes-to-(ps,wts) conversion: for ordered upper-hull knots `(s_0=0, gs_0=0), (s_1, gs_1), …, (s_K=1, gs_K=1)`, slopes `m_i = (gs_{i+1} - gs_i)/(s_{i+1} - s_i)` are non-increasing (concavity). Weights solve a linear system relating slopes to TVaR component contributions; reuses the existing `Distortion.tvar_terms` machinery.

## Base class design

```python
class Distortion:
    _registry: dict[str, type] = {}
    kind: str = ''
    # ... existing class-level metadata ...
    param_name: str | None = None   # natural name of the scalar param; None for multi-param

    def __new__(cls, name=None, *args, **kwargs):
        # unchanged: factory dispatch
        ...

    def __init__(self, name, shape=None, *, display_name='', **natural):
        """Scalar-shape constructor. Multi-param kinds override entirely."""
        if name == 'roe':
            name = 'ccoc'
        pn = type(self).param_name
        if pn is not None and pn in natural:
            if shape is not None:
                raise TypeError(f"pass {pn}= or positional shape, not both")
            shape = natural.pop(pn)
        if natural:
            raise TypeError(f"unexpected kwargs: {list(natural)}")
        self._name = name
        self.shape = shape
        self.display_name = display_name
        self._common_init()
        self._build()

    def _common_init(self):
        """Common audit/state defaults."""
        self.has_mass = False
        self.mass = 0.0
        self.standard_shape = np.nan
        self.error = 0.0
        self.premium_target = 0.0
        self.assets = 0.0

    def _id_fields(self):
        """Tuple of identifying attribute values for ``id()`` hashing.
        Subclasses override to declare their own structural state."""
        return (self._name, self.shape, self.display_name)
```

Scalar kinds (ph, wang, dual, tvar) need **no `__init__` override** — they declare `param_name` and a property alias. Mass-at-zero scalar kinds (cll, clin, ly, lep) and ccoc override `__init__` to handle their extra kwargs (`r0`, `d`/`r` for ccoc). Multi-param kinds (bitvar, wtdtvar, beta, power, minimum, mixture) override `__init__` entirely and call `self._common_init()` + `self._build()` themselves.

### `_id_fields()` per subclass

- Scalar kinds: `(self._name, self.shape, self.display_name)` — inherited default.
- Scalar+r0 kinds (cll, clin, ly, lep): `(self._name, self.shape, self.r0, self.display_name)`.
- ccoc: `(self._name, self.r, self.d, self.display_name)`.
- bitvar: `(self._name, self._p0, self._p1, self._w1, self.display_name)`.
- wtdtvar: `(self._name, tuple(self._ps), tuple(self._wts), self.display_name)`.
- beta: `(self._name, self._a, self._b, self.display_name)`.
- power: `(self._name, self._x0, self._x1, self.shape, self.display_name)`.
- minimum, mixture: `(self._name, tuple(d.id() for d in self._distortions), …)`.

The existing `Distortion.id()` method updates to use `_id_fields()`:

```python
def id(self):
    return _short_hash(str(self._id_fields()))
```

## DecL parser translation

The DSL grammar (`decl.lark`) parses distortion clauses to either:
- short form: `("distortion", name, {"name": kind_id, "shape": expr})`
- long form: `("distortion", name, {"name": kind_id, "shape": expr, "df": numberl})`

`underwriter.py:422` does `Distortion(**spec)`. After the refactor, spec dicts must contain natural kwargs.

**Translation point:** in `parser.py`'s `distortion_out_short` / `distortion_out_long` methods, add a per-kind dispatch:

```python
def _distortion_spec(kind_id, shape, df=None):
    """Translate DecL (kind, shape, df?) into natural-kwarg spec dict."""
    spec = {"name": kind_id}
    if kind_id == 'ph':
        spec['a'] = shape
    elif kind_id == 'wang':
        spec['lam'] = shape
    elif kind_id == 'dual':
        spec['b'] = shape
    elif kind_id == 'tvar':
        spec['p'] = shape
    elif kind_id == 'ccoc':
        # legacy DSL semantics: shape is return (r). Preserve by passing r=.
        spec['r'] = shape
    elif kind_id == 'bitvar':
        if df is None or len(df) != 2:
            raise ValueError("DecL bitvar requires [p0 p1]; got df=" + repr(df))
        spec['p0'], spec['p1'], spec['w1'] = df[0], df[1], shape
    elif kind_id == 'wtdtvar':
        spec['ps'], spec['wts'] = shape, df
    elif kind_id == 'cll':
        spec['b'] = shape
        # r0 not currently DSL-expressible — DSL form takes only b
    # ... etc for kinds the DSL exposes
    else:
        raise ValueError(f"DecL: unknown distortion kind {kind_id!r}")
    return spec
```

The grammar itself does NOT need to change — the existing 2-arg `kind shape [df]` form is sufficient. Kinds not exposed via DSL (lep, lyd with custom r0, beta, power, minimum, mixture) are constructed in Python only.

## Other call sites (per Phase 1 reconnaissance)

| File | Lines | Change |
|---|---|---|
| `src/aggregate/portfolio.py` | 2352-2353, 2420-2421 | `Distortion(name=, shape=, r0=, df=)` → per-kind natural kwargs. `_calibration_init_shape` is read by the calibration loop and stays unchanged on each subclass |
| `src/aggregate/pedagogy.py` | 161, 162, 165, 175, 1045 | `Distortion('roe', roe)` → `Distortion('ccoc', r=roe)`; `Distortion('tvar', p_star)` → `Distortion('tvar', p=p_star)`; `Distortion('wtdtvar', [pl, pu], df=[1-w, w])` → `Distortion('wtdtvar', ps=[pl, pu], wts=[1-w, w])` |
| `src/aggregate/bounds.py` | 393 | `Distortion('bitvar', w, df=[pl, pu])` → `Distortion('bitvar', p0=pl, p1=pu, w1=w)` |
| `src/aggregate/distributions.py` | 4214 | `Distortion(**g)` — confirm during implementation. If `g` is parser-emitted, the parser translation covers it; if from elsewhere, translate at call site |
| `tests/test_distortion_calibrate.py` | 47, 71, 88, 103 | Update keyword calls to natural kwargs; `Distortion('ccoc', shape=0.25)` → `Distortion('ccoc', r=0.25)` |

## Newton-Raphson: considered, declined

The current `_newton_iterate(self, f, shape, max_iter=50, tol=1e-5)` is hand-rolled. Considered switching to `scipy.optimize.newton` (or `scipy.optimize.root_scalar(method='newton', ...)`).

**Why decline:**

- scipy's API requires `func` and `fprime` as separate callables, each evaluated on its own. Our pricing kinds compute value and derivative **together** in a single pass (e.g. PH calibration computes `S^ρ` once and uses it for both `ex = sum(trho)*bs` and `ex_prime = sum(trho*lS)*bs`). Wrapping naively doubles the cost.
- The hand-rolled loop is 8 lines, well-understood, and matches the docstring contract.
- scipy's failure modes (RuntimeError on non-convergence) differ from the current logger.warning behavior. Switching would change observable behavior for users with marginal calibrations.
- TVaR's `max_iter=200` quirk is harder to express cleanly through scipy's API.

**Recommendation:** keep hand-rolled. Revisit if calibration robustness becomes a complaint.

## Docstring style sweep (in-scope, light)

Every docstring on a function or method modified by this refactor — every subclass `__init__`, every modified `_build`, the base class `__init__`/`_common_init`/`_id_fields`, the new `convex_distortion`, the rewired static factories — gets converted to **NumPy style** (Parameters / Returns / Notes / References sections). No invented content.

Empty `:param x:` lines that survived in modified docstrings are deleted and reported as a list at the end of the implementation PR for user triage. Docstrings on methods NOT modified by this refactor are left alone (deferred to the broader sweep noted in CLAUDE.md TODO).

## Testing strategy

**One-time pre-refactor snapshot.** New script `tests/capture_distortion_snapshot.py` evaluates `g(s)` and `g_inv(s)` on `s = np.linspace(0, 1, 101)` for representative parameters across all kinds, writes to `tests/data/distortion_g_snapshot.csv`. Run ONCE on the pre-refactor tip and commit the CSV.

Parameter grid (one row per kind):

```
ph         a=0.7
wang       lam=0.3
dual       b=2.0
tvar       p=0.95
ccoc       d=0.0909             # equivalently r=0.1
bitvar     p0=0.95, p1=0.99, w1=0.5
wtdtvar    ps=[0.5, 0.9, 1.0], wts=[0.3, 0.3, 0.4]
cll        r0=0.05, b=0.9
clin       r0=0.05, slope=2.0
lep        r0=0.03, r=0.15
ly         r0=0.05, r=1.25
beta       a=0.7, b=1.5
power      x0=0.01, x1=1.0, alpha=2.0
```

CSV columns: `s, ph_g, ph_g_inv, wang_g, wang_g_inv, …`. ~13 kinds × 2 columns + 1 s column = ~27 columns × 101 rows.

**Post-refactor regression test.** New `tests/test_distortion_snapshot.py` reads the CSV, reconstructs each distortion via the new constructor, asserts each column matches within `rtol=1e-10`. This catches numerical regressions invisible to API tests.

**Construction surface test.** New `tests/test_distortion_construction.py` parametrizes over (kind, construction_form) where `construction_form ∈ {factory_positional, factory_kwarg, direct_subclass, static_method}`. For each, assert the result has working `g(0)=0`, `g(1)=1`, `g_inv(g(0.5))≈0.5`, and the natural-name property reads correctly. Covers basic five (ph, wang, dual, tvar, ccoc) + bitvar + wtdtvar + minimum + mixture (per user instruction).

**Convex regression.** Quick check that `convex_distortion(...)` on the bond and cat-bond example data produces a `WtdTVaRDistortion` whose `g(s)` matches the old `ConvexDistortion('convex', ...).g(s)` snapshot to within `rtol=1e-10`.

**Existing tests:** `tests/test_distortion_calibrate.py` updates to natural kwargs. Other tests should pass unchanged.

## Critical files

| File | Action |
|---|---|
| `src/aggregate/spectral.py` | Base class refactor; per-subclass `__init__` rewrites; property aliases; static-factory rewires; **remove** `ConvexDistortion` + `_plot_decorations` + `Distortion.average_distortion`; **add** module-level helpers block at bottom with `convex_distortion`, `bagged_distortion`, `convex_example`; **delete** `Distortion.bagged_distortion`, `Distortion.s_gs_distortion`, `Distortion.convex_example` staticmethods; add `_id_fields()` per subclass |
| `src/aggregate/parser.py` | Add `_distortion_spec()` helper; update `distortion_out_short`/`distortion_out_long` |
| `src/aggregate/portfolio.py` | Lines 2352, 2420: natural kwargs |
| `src/aggregate/pedagogy.py` | 5 call sites |
| `src/aggregate/bounds.py` | Line 393 |
| `src/aggregate/distributions.py` | Line 4214: verify and update if needed |
| `tests/test_distortion_calibrate.py` | Natural kwargs |
| `tests/capture_distortion_snapshot.py` | NEW — one-time generator |
| `tests/data/distortion_g_snapshot.csv` | NEW — golden file |
| `tests/test_distortion_snapshot.py` | NEW — regression vs snapshot |
| `tests/test_distortion_construction.py` | NEW — construction-form matrix |
| `pyproject.toml` | Version 1.0.0a12 → 1.0.0a13 |
| `README.rst` | New 1.0.0a13 release-note block |

## Reused existing symbols

- `_calibration_init_shape` (per-subclass class attr) — unchanged; calibration code reads it directly.
- `Distortion._registry` and `__init_subclass__` — unchanged.
- `Distortion.__new__` factory dispatch — unchanged.
- `_newton_iterate` and `_finalize_calibration` — unchanged; they read/write `self.shape`.
- `Distortion.tvar_terms` — reused by the new `convex_distortion` to map slopes → (ps, wts).
- All `g`, `g_inv`, `g_prime`, `calibrate` method bodies — unchanged; they read `self.shape` and (for some) `self.r0`.

## Verification

1. **Run snapshot capture on pre-refactor tip**: `uv run python tests/capture_distortion_snapshot.py`. Commit the CSV.
2. **Implement and run full pytest**: `uv run pytest`. All existing 453+ tests should pass plus the new construction and snapshot tests.
3. **DecL smoke checks**:
   ```python
   from aggregate import build
   build('distortion d1 ph 0.5')                   # PHDistortion
   build('distortion d2 bitvar 0.5 [0.95 0.99]')   # BiTVaRDistortion
   build('distortion d3 wtdtvar [0.5 0.9] [0.5 0.5]')  # WtdTVaRDistortion
   ```
4. **Direct construction smoke**:
   ```python
   from aggregate.spectral import Distortion, PHDistortion
   Distortion('ph', 0.7)              # factory positional
   Distortion('ph', a=0.7)            # factory kwarg
   PHDistortion(a=0.7)                # direct
   Distortion.ph(0.7)                 # static
   Distortion('ccoc', d=0.1)          # explicit discount
   Distortion('ccoc', r=0.111)        # explicit return
   try: Distortion('ccoc', 0.1)       # positional should raise
   except TypeError: pass
   ```
5. **Portfolio.calibrate_distortion** end-to-end on a fixture portfolio — Newton converges, `self.shape` updates.
6. **Snapshot regression**: `uv run pytest tests/test_distortion_snapshot.py -v` — every column matches.
7. **Convex regression**: confirm `convex_example('bond').g(np.linspace(0,1,11))` (now a module-level function) matches pre-refactor `Distortion.convex_example('bond').g(...)` values within `rtol=1e-10`.

## Things explicitly out of scope

- The `info` / `describe` / `stats_df` / `density_df` quartet — separate plan (was discussed as "stats_df first" sequencing; both plans are independent).
- Full docstring style sweep beyond the touched surface — deferred per CLAUDE.md TODO.
- Renaming `r0` more evocatively — stays as `r0`.
- `MixtureDistortion.g_inv` (currently `NotImplementedError`) — pre-existing limitation, not part of this refactor.
- DecL grammar changes — existing `kind shape [df]` form is sufficient.
- Switching to scipy Newton-Raphson — declined (see above).

## Iteration expected

User has stated: "I'm sure we will need to iterate." Treat per-kind decisions as defaults; flag any item that needs further discussion before implementing.
