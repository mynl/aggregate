# Plan — Distortion quartet: info / describe / stats_df / density_df

**Final file destination (on approval):** `dev/spectral-quartet.md` (alongside the
existing `spectral-call-signatures.md`). This plan is staged in the plan-mode file
during planning; on ExitPlanMode + approval, content is copied to `dev/`.

## Context

`Aggregate` and `Portfolio` each expose an `info` / `describe` / `stats_df` /
`density_df` quartet that gives users a quick read on the object. `Distortion`
currently has none of these — only `__repr__` (returns the name) and `plot`.

A distortion `g : [0,1] → [0,1]` with `g(0)=0`, `g(1)=1`, monotone non-decreasing,
is structurally a CDF on `[0,1]`. So `g` and `g_inv` induce two natural
distributions:

| Distribution | Sample as     | CDF       |
|--------------|---------------|-----------|
| `D_g`        | `g_inv(U)`    | `g`       |
| `D_g_inv`    | `g(U)`        | `g_inv`   |

Convenient sanity identity: `E[D_g] + E[D_g_inv] = 1` (the two areas partition the
unit square). The dual `g_dual = 1 - g(1-·)` gives a second pair surfaced in
`density_df` columns; no separate distribution rows for the dual in describe.

All four properties are **lazy** — zero cost when not accessed; cached after first
read; invalidated by `_build()` so calibrate-then-read sees fresh tables.

## Decisions

1. **Four `@cached_property` properties on the base class.** `info`, `describe`,
   `stats_df`, `density_df`. Each delegates to a `_compute_*` method. The cached
   value lands in `self.__dict__['<name>']`; `_invalidate_cache()` pops the four
   keys.

2. **Invalidation hook in `_build()`.** Appended to the existing base `_build()`
   no-op (spectral.py line 440). Every subclass `_build()` already calls
   `super()._build()` indirectly through `_common_init` semantics, OR sets state
   then exits — verify and add a `self._invalidate_cache()` call at the end of
   each subclass `_build`. Property setters already call `_build()`, so
   `d.a = 0.5; d.stats_df` returns the recomputed table.

3. **`cached_property` over manual `_attr` properties.** Both achieve the same
   first-call-wins behaviour. `cached_property` is the stdlib idiom — on first
   access it calls the function then writes the result to `self.__dict__[name]`;
   subsequent reads bypass the descriptor protocol because instance `__dict__`
   wins. Invalidation = `self.__dict__.pop(name, None)`. Pickle-safe (the cache
   is in `__dict__`, so it pickles and restores). Manual `_cache` attrs add
   boilerplate (~3 lines × 4 = 12 lines) for no functional gain. Use uniformly.

4. **No entropy in v1.** Add a `# TODO: entropy` note in `_compute_stats_df`.
   Mass-at-zero kinds (ccoc/cll/clin) genuinely diverge; piecewise-linear kinds
   require per-kind exact summation. Defer until a user reaches for it.

5. **Default grid: 101 evenly spaced points + spliced knots.** Class-level
   `_density_n_points = 101` on the base. Per-subclass `_density_knots()` hook
   returns extra x-values to splice (TVaR kink at `1-p`, BiTVaR at `{1-p0,1-p1}`,
   WtdTVaR at `{1-p for p in ps}`, CCoC at the mass step, PowerD at `{x0, x1}`,
   CLL/CLin at the cap point if it falls in `(0,1)`). Default returns `[]`.

6. **Broad closed-form coverage.** Per-subclass `_describe_closed_form()` returns
   `dict[row_name → float]` for `mean, var, std, cv, skew`. Implement on PH,
   Wang, Dual, TVaR, CCoC, Beta, Power, CLL, CLin, LEP, LY (11). BiTVaR, WtdTVaR,
   Min, Mix return `{}` (— in the table). Each formula is ~3–8 lines; for kinds
   where moments factor through scipy (Wang via `scipy.stats.norm`, Beta via
   `scipy.stats.beta`), use scipy directly.

7. **`p_equiv` as the single "where on the TVaR scale" row.** `p_equiv = 2·∫g − 1`.
   Equals the inversion of `p_to_parameters` for TVaR-equivalent kinds (tvar→p,
   ph→(1-a)/(1+a), wang→2·Φ(λ/√2)-1, dual→(b-1)/(b+1), ccoc→d). Drop the
   redundant "gini/concavity" row name (user's positive convention is the same
   number). Single canonical row.

8. **`loading` row.** `loading = ∫g − 0.5`. Linear in p_equiv (=`p_equiv/2`), but
   actuaries read it as "how much g loads a U[0,1] above the actuarial price";
   keep as a separate row.

9. **Moments via tail integrals, not g'.** `mean = 1 - ∫g`; `E[X²] = ∫2x(1-g) dx`;
   `var = E[X²] - mean²`. Skewness similarly via third central moment using
   tail-integral form. Robust to kinks (no g' on the grid is needed for the
   primary moments). For D_g_inv, swap `g` for `g_inv`.

10. **MixtureDistortion.g_inv currently `NotImplementedError`.** For the
    `D_g_inv` column in describe/stats_df, fall back to numerical inversion via
    `brentq(lambda u: g(u)-q, 0, 1)` evaluated on the density_df grid. Do **not**
    fix the actual `g_inv` method — out of scope.

11. **`checks` block at the bottom of describe.** Three rows appended below
    moments:
    - `E[D_g] + E[D_g_inv]` (target 1.0)
    - `g(g_inv(0.5))` (target 0.5)
    - `g(0), g(1)` (target 0, 1)

    Implementation: separate small DataFrame `pd.concat([moments, checks])` with
    `checks` rows visually offset (blank row or section label).

12. **density_df columns.** `x, g, g_inv, g_dual, g_dual_inv, g_prime,
    g_dual_prime, kusuoka`. Name order matters: `g_dual_prime` = derivative of
    the dual, computed obviously from `g_prime` as `g_prime(1-x)` (not `g_prime`
    of the dual). The `kusuoka` column reuses the existing
    `weights_function(d)` machinery (spectral.py line ~2430), which already
    branches per kind for ph/wang/dual/tvar and falls back to a numerical
    `(1-p)·∇g'(1-p)`. The column is the (approximation to the) mixed
    distribution function of the Kusuoka spectral measure μ on `[0,1]`. For
    kinds with discrete μ (tvar, bitvar, wtdtvar) the splice of knots into
    the grid ensures the atoms land on grid points.

12a. **Kusuoka-atoms stats_df row.** Three rows summarising the spectral
    measure μ:
    - `mass_at_0` — numeric: atom of g at x=0 (= μ's atom at p=0; the existing
      `self.mass` for ccoc/cll/clin/lep/ly with `r0>0`). 0 otherwise.
    - `mass_at_1` — numeric: atom of g at x=1 (mass of g's discontinuity at 1,
      = w·1[1 ∈ supp(μ)] form). 0 unless tvar with p=1, bitvar with p1=1,
      wtdtvar with 1 ∈ ps, ccoc (always — μ has Dirac at p=1), or min/mix
      containing such.
    - `kusuoka_interior_atoms` — bool: True iff μ has atoms in the open
      interval `(0, 1)`. True for tvar, bitvar, wtdtvar (depending on the
      ps), and min/mix containing these. False for ph, wang, dual, beta,
      power, lep, ly, cll, clin (continuous μ).

    Atoms in μ at interior p correspond to kinks in g at `x = 1-p`. The
    density_df grid already splices these kinks in (per decision 5), so the
    `kusuoka` column visually picks them out.

13. **`plot()` reads from `density_df`.** Existing `Distortion.plot()` (spectral.py
    line 622) rewires to consume `density_df` instead of evaluating g/g_dual
    inline. Side benefit: plot inherits knot splicing automatically and stays
    consistent with the table.

14. **`stats_df` is single-column "D_g"** (per user notes) with `closed_form`
    and `error` side columns where analytics exist. `describe` shows both
    `D_g` and `D_g_inv` (the symmetric pair) plus `closed_form` / `error`.

15. **`info` is a multi-line `str`** mirroring Aggregate/Portfolio. For
    multi-param kinds (WtdTVaR/Min/Mix) it shows summary stats (knot count,
    weight total) and points at `stats_df` for the full breakdown. Includes the
    `id` hash from `_id_fields()`.

## Module-level constants

```python
# In spectral.py, near the top:
_DISTORTION_DENSITY_N = 101   # default grid for density_df / stats integrations
```

## Base-class additions

```python
import functools

class Distortion:
    _density_n_points = _DISTORTION_DENSITY_N

    @functools.cached_property
    def info(self): return self._compute_info()

    @functools.cached_property
    def describe(self): return self._compute_describe()

    @functools.cached_property
    def stats_df(self): return self._compute_stats_df()

    @functools.cached_property
    def density_df(self): return self._compute_density_df()

    def _invalidate_cache(self):
        for k in ('info', 'describe', 'stats_df', 'density_df', '_grid_moments'):
            self.__dict__.pop(k, None)

    def _density_knots(self):
        """Subclass hook: extra x-values to splice into the uniform grid."""
        return []

    def _describe_closed_form(self):
        """Subclass hook: dict of row_name → analytic value."""
        return {}

    # Base `_build()` (line 440) gets `self._invalidate_cache()` appended.
```

## Private compute helpers (base class, on the base file)

- `_compute_density_df()`: build sorted unique grid, eval columns, return df.
- `_grid_moments` (cached): dict of integrals from a single trapz pass over the
  density grid (`int_g`, `int_g_inv`, `int_x2_S`, etc.) used by both
  `_compute_describe` and `_compute_stats_df` — avoids double integration.
- `_compute_stats_df()`: assemble single-column DataFrame from `_grid_moments`
  and `_describe_closed_form()`.
- `_compute_describe()`: smaller pair-column DataFrame plus the checks block.
- `_compute_info()`: format string from `_id_fields()`, `param_name`, `has_mass`,
  etc.

## Per-subclass overrides

| Subclass    | `_describe_closed_form` | `_density_knots`             | mass_at_1                  |
|-------------|-------------------------|------------------------------|----------------------------|
| PH          | ✓                       | []                           | 0                          |
| Wang        | ✓ (scipy.stats.norm)    | []                           | 0                          |
| Dual        | ✓                       | []                           | 0                          |
| TVaR        | ✓ (piecewise)           | [1-p]                        | 1 if p==1 else 0           |
| CCoC        | ✓                       | [d] (mass step)              | 0                          |
| BiTVaR      | ✓ (linear combo)        | [1-p0, 1-p1]                 | w1 if p1==1 else 0         |
| WtdTVaR     | {}                      | [1-p for p in ps]            | sum(wts where ps==1)       |
| Beta        | ✓ (scipy.stats.beta)    | []                           | 0                          |
| Power       | ✓                       | [x0, x1]                     | 0                          |
| CLL         | ✓                       | [cap if cap ∈ (0,1)]         | 0                          |
| CLin        | ✓                       | [cap if cap ∈ (0,1)]         | 0                          |
| LEP         | ✓                       | []                           | 0                          |
| LY          | ✓                       | []                           | 0                          |
| Minimum     | {}                      | union of members'            | max of members'            |
| Mixture     | {}                      | union of members'            | sum(wts · members')        |

A new attribute `mass_at_1` (default 0) is set in each subclass `_build` where
applicable. Currently only `has_mass` (mass at 0) is tracked; adding `mass_at_1`
is a one-line addition per relevant kind.

## Critical files

| File | Action |
|------|--------|
| `src/aggregate/spectral.py` | All implementation: base class, per-subclass overrides, `plot()` rewire to read `density_df` |
| `tests/test_distortion_quartet.py` | NEW — see Verification |
| `pyproject.toml` | Version 1.0.0a13 → 1.0.0a14 |
| `README.rst` | New 1.0.0a14 release-note block |
| `dev/spectral-quartet.md` | Final home for this plan (copied on approval) |

## Reused existing symbols

- `weights_function(d)` (spectral.py ~2430) — produces the kusuoka column.
- `p_to_parameters(p)` (spectral.py 2456) — inversion gives `p_equiv` formulas.
- `_id_fields()` — supplies the `id` hash for `info`.
- `_build()`, `_finalize_calibration` — already trigger invalidation chain;
  only addition is `_invalidate_cache()` at the end of base `_build()`.

## Verification

1. **New tests** in `tests/test_distortion_quartet.py`:
   - Each kind: `d.info` is a str; `d.describe`/`d.stats_df`/`d.density_df` are
     DataFrames with expected shape.
   - `mean(D_g) + mean(D_g_inv) ≈ 1` for every kind (rtol 1e-6 at n=101 grid for
     smooth g; rtol relaxes for piecewise-linear kinks unless knots present).
   - `p_equiv` self-consistency: `Distortion('tvar', p=0.7).stats_df.loc['p_equiv'] == 0.7`.
     Same closed-form check for ph/wang/dual/ccoc.
   - Cache invalidation: build d, read `d.stats_df.loc['mean']`, set `d.a = 0.5`,
     read again — value changes.
   - `density_df` contains the spliced knots (e.g. `1-p ∈ d.density_df.index` for
     TVaR).
   - Closed-form vs numeric `error` column < 1e-3 at n=101 for smooth kinds.

2. **Smoke**:
   ```python
   from aggregate import Distortion
   d = Distortion('ph', a=0.7)
   print(d.info); print(d.describe); print(d.stats_df); d.density_df.head(); d.plot()
   ```

3. **Full pytest** — none of the existing 527 tests should break (additive only).

## Out of scope

- Entropy (TODO row in stats_df docstring).
- Fixing `MixtureDistortion.g_inv` properly (we numerically fall back for the
  describe column; the method itself stays `NotImplementedError`).
- Per-call grid override (`describe(n=)`, `stats_df(n=)`) — class-level
  `_density_n_points` is the knob for v1.
- Any change to Aggregate/Portfolio quartets.

## Open follow-ups (record, defer until tables are seen)

- Whether to surface the dual distribution as additional describe columns
  (currently we only show D_g / D_g_inv, with the dual visible in density_df).

## Iteration expected

User noted "I'm sure we will need to iterate." Treat the column lists, row
names, and the checks-block layout as defaults — flag any item that needs
discussion before implementing.
