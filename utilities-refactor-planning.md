# `utilities.py` refactor — planning

## Context

`aggregate/utilities.py` is the catch-all module that grew alongside
the rest of the package. After the spectral / parser / distributions /
underwriter / portfolio sweeps it's the last load-bearing module
without a focused cleanup pass. Current state: **3,753 lines, 74
top-level definitions**.

The goal is twofold:

1. **Delete dead code + inline single-call helpers + privatise
   internal-only helpers** — net ~1,300+ LOC reduction.
2. **Break utilities into themed modules** — small focused files
   beat a grab-bag. After this sweep `utilities.py` shrinks to
   ~12 genuinely cross-cutting helpers (~500–700 LOC).

No new behaviour, no signature changes on live functions, no
numerical changes. PEG regression stays bit-identical at `rtol=1e-10`.

## Final shape

| Module | Holds |
|---|---|
| `aggregate/utilities.py` | ~10 cross-cutting helpers: ft, ift, round_bucket, pprint_ex, tweedie_convert, tweedie_density, make_ceder_netter, subsets, nice_multiple, make_var_tvar, kaplan_meier, kaplan_meier_np, agg_help, explain_validation, decl_pprint |
| `aggregate/moments.py` *(new)* | MomentAggregator, MomentWrangler, xsden_to_meancv, xsden_to_meancvskew |
| `aggregate/iman_conover.py` *(new)* | iman_conover, ic_*, block_iman_conover, make_corr_matrix, random_corr_matrix, rearrangement_algorithm_max_VaR |
| `aggregate/distributions.py` | **Public fitting cluster** (PUBLIC, exported): `lognorm_fit` (renamed from `mu_sigma_from_mean_cv`), `gamma_fit`, `beta_fit`, `invgamma_fit`, `invgauss_fit`, `sln_fit`, `sgamma_fit`, `approximate_from_mcvsk` (renamed from `approximate_work`), `lognorm_approx`, `lognorm_lev`. **Private single-module helpers**: `_estimate_agg_percentile`, `_picks_work`, `_moms_analytic` + `_partial_e` + `_partial_e_numeric`, `_integral_by_doubling`, `_logarithmic_theta`. |
| `aggregate/portfolio.py` | absorbs (public) `make_comonotonic_allocations_work` — module-level function paired with `Portfolio.make_comonotonic_allocations` method; re-exported as `make_comonotonic_allocations` |
| `aggregate/underwriter.py` | absorbs (private) the merged `_parse_note` |
| `aggregate/spectral.py` | absorbs (private) `_short_hash` |

## Tasks (execution order)

Each task ends with `uv run pytest -q` green and PEG regression
bit-identical.

### Phase A — Deletes

#### A.1 — Confirmed-dead bodies

| Symbol | Location | LOC | Why dead |
|---|---|---|---|
| `MomentAggregator.stats_series` | L1115 | ~20 | Retired by Stage 1c+ |
| `frequency_examples` | L1373 | ~328 | No callers |
| `test_var_tvar` | L3390 | ~80 | No callers |
| `axiter_factory` | L549 | ~20 | Only consumer was frequency_examples |
| `AxisManager` | L568 | ~180 | Only consumer was frequency_examples |
| `html_title` | L781 | ~14 | **Dead-import** — listed in `portfolio.py` import line but never called |
| `suptitle_and_tight` | L809 | ~10 | **Dead-import** — same |
| `ln_fit` alias | L198 | 1 | Only user is `estimate_agg_percentile`, which moves to distributions.py and gains direct access to `mu_sigma_from_mean_cv` |
| Timer cruft (`last_time`, `first_time`, `timer_active`) | L48–50 | 3 |  |

#### A.2 — Dead public helpers

Zero in-tree callers anywhere:

| Symbol | Notes |
|---|---|
| `mv` | docs-only |
| `qdp` | docs-only |
| `introspect` | "match" in distributions.py:4714 is a comment |
| `get_fmts` | zero callers |
| `sensible_jump` | zero callers |
| `GCN` (namedtuple) | zero callers |

Delete body + `__init__` re-export for each.

#### A.3 — Plotting / formatting stack

Aggregate should not touch the user's matplotlib settings. Delete:

| Symbol | LOC |
|---|---|
| `GreatFormatter` | ~10 |
| `make_mosaic_figure` | ~38 |
| `knobble_fonts` | ~65 |
| `easy_formatter` | ~40 |
| `style_df` | ~40 |
| `friendly` | ~35 |
| `FigureManager` | ~210 |
| `sEngFormatter` | ~170 |
| `show_fig` | ~20 |

**KEEP** `nice_multiple` (user: useful elsewhere later).

#### A.4 — Caller rewrites (plain matplotlib)

- `aggregate/distributions.py:2838-2852` — `Aggregate.plot()`:
  - `make_mosaic_figure('ABC', **kwargs)` → `plt.subplot_mosaic('ABC', layout='constrained')`.
  - Drop `xfmt='great' / yfmt='great'` (GreatFormatter styling).
  - **Keep** `span = nice_multiple(mx)`.
- `aggregate/bounds.py:13-14, 879` — drop commented + live `FigureManager`
  imports; rewrite the `smfig = FigureManager(...)` site + its
  `smfig.<method>` uses to plain `plt.subplots()`.
- `aggregate/underwriter.py:18` — drop `show_fig` if unused.
- `aggregate/__init__.py` — drop re-exports for everything in A.1+A.2+A.3
  except `nice_multiple`. Drop the commented `# knobble_fonts(True)` at L54.

#### A.5 — `extensions/case_studies.py` mpl conversion

Per user: don't break for silly reasons.

- `case_studies.py:8` — drop `make_mosaic_figure`, `FigureManager` from import.
- `case_studies.py:246` — `FigureManager(...)` → `plt.subplots()` block.
- `case_studies.py:1238, 1565` — `make_mosaic_figure(...)` → `plt.subplot_mosaic(...)`.

Mechanical; drop formatter-styling; accept matplotlib defaults.

### Phase B — Extractions

#### B.1 — Create `aggregate/moments.py`

Move from `utilities.py`:
- `MomentAggregator` class (~320 LOC)
- `MomentWrangler` class (~95 LOC)
- `xsden_to_meancv` (~30 LOC)
- `xsden_to_meancvskew` (~30 LOC)

Module docstring (one sentence): *Moment accumulation across mixture
components and conversion between central / non-central / factorial
representations.*

Update imports in `distributions.py`, `portfolio.py`,
`extensions/tweedie.py` to source from `.moments`. Update
`__init__.py` re-exports.

#### B.2 — Create `aggregate/iman_conover.py`

Move from `utilities.py`:
- `ic_noise`, `ic_t_noise`, `ic_rank`, `ic_reorder` (private helpers)
- `iman_conover` (main entry)
- `block_iman_conover`
- `make_corr_matrix`, `random_corr_matrix`
- `rearrangement_algorithm_max_VaR` (per user — family resemblance)

Module docstring: *Iman–Conover dependence imposition and related
correlation utilities.*

Update imports in `portfolio.py` + `__init__.py`.

### Phase C — Moves into `distributions.py`

#### C.1 — Public `*_fit` family (the headline change)

All `*_fit` functions move to `distributions.py` and stay public,
forming a symmetric `(m, cv[, skew])` → distribution-parameters
family. The current asymmetry (`mu_sigma_from_mean_cv` not following
the `*_fit` pattern; `approximate_work`'s gamma branch inlining the
math while its lognorm branch uses a helper) gets cleaned up.

| Symbol | Was | After | Notes |
|---|---|---|---|
| `lognorm_fit(m, cv)` | `mu_sigma_from_mean_cv` in utilities | distributions.py | **Rename** to match the `*_fit` pattern; matches scipy's `scipy.stats.lognorm` name |
| `gamma_fit(m, cv)` | utilities | distributions.py | Already exists; just moves |
| `beta_fit(m, cv)` | utilities | distributions.py | Already exists; **NOT inlined** (kept symmetric with siblings) |
| `invgamma_fit(cv)` | utilities | distributions.py | Already exists; **NOT inlined** |
| `invgauss_fit(cv)` | utilities | distributions.py | Already exists; **NOT inlined** |
| `sln_fit(m, cv, skew)` | utilities | distributions.py | Already exists; moves |
| `sgamma_fit(m, cv, skew)` | utilities | distributions.py | Already exists; moves |
| `approximate_from_mcvsk(...)` | `approximate_work` in utilities | distributions.py | **Renamed** — the function takes (m, cv, skew) and dispatches to fitting helpers; the new name says so. Gamma branch **refactored** to use `gamma_fit(m, cv)` instead of inlining `shape = cv**-2; scale = m/shape` (now symmetric with the lognorm branch using `lognorm_fit`). Methods `Aggregate.approximate` (`distributions.py:3466`) and `Portfolio.approximate` (`portfolio.py:1170`) updated to call by new name. |
| `lognorm_approx(ser)` | utilities | distributions.py | Clusters with the lognorm family |
| `lognorm_lev(mu, sigma, n, limit)` | utilities | distributions.py | Clusters with the lognorm family (rst-documented public actuarial helper) |

Drop the `ln_fit = mu_sigma_from_mean_cv` alias entirely — `lognorm_fit`
is now the canonical name.

All exports preserved via `aggregate/__init__.py` (sourced from
`.distributions` instead of `.utilities`).

**Update `Portfolio.approximate`** (`portfolio.py:1170`): change the
import line at L33 to source `approximate_from_mcvsk` from
`.distributions` instead of `approximate_work` from `.utilities`.
Update the call site. (Portfolio already imports from distributions
at L25.)

**Inside `approximate_from_mcvsk` body** (was `approximate_work`):
replace the inline gamma math at L309-311 with
`shape, scale = gamma_fit(m, cv)` — symmetric with the lognorm
branch that uses `lognorm_fit`.

#### C.2 — Private single-module helpers

These are used only in `distributions.py`; move + privatise:

| Symbol | LOC | Callers | Action |
|---|---|---|---|
| `estimate_agg_percentile` | ~53 | 3× | move as `_estimate_agg_percentile` |
| `picks_work` | ~158 | 1× | move as `_picks_work` (too big to inline) |
| `moms_analytic` | ~53 | 1× | move as `_moms_analytic` |
| `partial_e` | ~85 | called by `_moms_analytic` | move as `_partial_e` |
| `partial_e_numeric` | ~14 | called by `_partial_e` | move as `_partial_e_numeric` |
| `integral_by_doubling` | ~34 | 2× | move as `_integral_by_doubling` |
| `logarithmic_theta` | ~12 | 2× | move as `_logarithmic_theta` (keep `@lru_cache`) |

Drop their `__init__.py` re-exports.

#### C.3 — Into `portfolio.py` (public)

| Symbol | LOC | Callers | Action |
|---|---|---|---|
| `make_comonotonic_allocations` | ~30 | 1× method | Move to portfolio.py at module level. Keep public. Renamed locally to `make_comonotonic_allocations_work` to avoid clashing with the `Portfolio.make_comonotonic_allocations(...)` method that wraps it. Re-export from `__init__.py` under the clean name `make_comonotonic_allocations`. (This codifies the existing `as ..._work` alias convention — the user has a public function and a public method sharing one logical name.) |

#### C.4 — Into `underwriter.py` (private)

After D.1 (`parse_note` + `parse_note_ex` merge):

| Symbol | Callers | Action |
|---|---|---|
| `parse_note` | 2× in underwriter.py | move as `_parse_note` |

#### C.5 — Into `spectral.py` (private)

| Symbol | LOC | Callers | Action |
|---|---|---|---|
| `short_hash` | ~10 | 1× | could inline (5-line hash op), but a name carries intent — **move as `_short_hash`** |

### Phase C.6 — Update `mu_sigma_from_mean_cv` callers (the rename)

7 occurrences across 3 rst files + 1 py site:

| File | Lines | Change |
|---|---|---|
| `aggregate/distributions.py` | L3952 | `mu, sigma = mu_sigma_from_mean_cv(1.0, cv)` → `mu, sigma = lognorm_fit(1.0, cv)`. Also drop `mu_sigma_from_mean_cv` from the utilities import (gone), import `lognorm_fit` from local module (no import needed — same file). |
| `aggregate/__init__.py` | re-export | `mu_sigma_from_mean_cv` → `lognorm_fit`; source from `.distributions` |
| `docs/2_user_guides/2_x_actuary_student.rst` | L103, 108, 111, 134 | Prose + `from aggregate import` + 2 call sites |
| `docs/2_user_guides/2_x_re_pricing.rst` | L603, 648, 789, 1033, 1042 | Import line + 4 call sites |
| `docs/5_technical_guides/5_x_rearrangement_algorithm.rst` | L129 | `agg.mu_sigma_from_mean_cv(10, i)` → `agg.lognorm_fit(10, i)` |

### Phase D — Renames + merges

#### D.1 — Merge `parse_note` + `parse_note_ex`

- Inline `txt.split(';')` parser block at top of `parse_note_ex` body.
- Rename merged function `parse_note(txt, log2, bs, recommend_p, kwargs)`.
- Delete original `parse_note(txt)`.
- Update `underwriter.py:562,598` from `parse_note_ex(...)` to `parse_note(...)`.
- Then move per C.4.

#### D.2 — Rename `pprint` → `decl_pprint`

Avoid stdlib collision; clarify purpose. Per user: keep public, keep
in utilities.

- `utilities.py` — rename function definition.
- `__init__.py` — export `decl_pprint` (drop `pprint`).
- `docs/2_user_guides/2_x_10mins.rst` — update the one example.
- `pprint_ex` stays unchanged (multi-py-used).

### Phase F — Docstring refresh + final audit

#### F.1 — Docstring refresh

Opportunistic pass. Targets:
- Empty `:return:` / `:param:` blocks — strip or fill in one line.
- Any stale `statistics_df` / `report_ser` / `audit_df` refs the
  housekeeping pass missed.
- Author's own `# TODO` notes — address or remove if obsolete.

#### F.2 — Module-top import sweep

After all deletions, audit `utilities.py` top-of-file imports. Many
become unused:
- `from collections import namedtuple` (after GCN deletion — keep
  only if a survivor still uses it; probably not).
- Anything else dead-after-deletions.

#### F.3 — Verification

```
uv run pytest -q                                # → 430 passed
rg "FigureManager|make_mosaic_figure|easy_formatter|knobble_fonts|sEngFormatter|style_df|GreatFormatter|show_fig|parse_note_ex|frequency_examples|axiter_factory|AxisManager|MomentAggregator\.stats_series|html_title|suptitle_and_tight|ln_fit|\bmv\b|qdp|introspect|get_fmts|sensible_jump|\bGCN\b|mu_sigma_from_mean_cv|approximate_work" aggregate/ tests/
# → empty (except where intended: rg may catch `mv` matches inside scipy's `.stats('mv')` — those are fine)
uv run python -c "from aggregate import build, qd, lognorm_fit, gamma_fit, beta_fit, invgamma_fit, invgauss_fit, sln_fit, sgamma_fit; print('fit family OK:', lognorm_fit(100, 0.5), gamma_fit(100, 0.5))"
uv run python -c "from aggregate import build, qd; a = build('agg X 1 claim sev lognorm 100 cv 1 poisson'); qd(a); a.plot()"
uv run python -c "from tests.peg import build_peg; p = build_peg(); p.plot()"
```

#### F.4 — README + commit

README `1.0.0a8 (in progress)` block gets a utilities-refactor block.
No version bump.

## Out of scope

- Numerical-behaviour changes.
- Signature changes on anything that survives.
- Renaming `pprint_ex` (could mirror as `decl_pprint_ex` later).
- Restructuring `qd` — 29 rst references, separate plan.
- Touching docs beyond the one `pprint` → `decl_pprint` example.

## Commit

Single commit. Suggested message:

```
Utilities refactor — delete ~1,300 LOC, extract themed modules,
privatise single-module helpers

Phase A (deletes):
- frequency_examples / axiter_factory / AxisManager (~530 LOC,
  pedagogical, no consumers)
- MomentAggregator.stats_series (retired by Stage 1c+)
- test_var_tvar (internal scaffold)
- Plotting / formatting subsystem: FigureManager,
  make_mosaic_figure, easy_formatter, knobble_fonts, style_df,
  friendly, GreatFormatter, sEngFormatter, show_fig (~630 LOC).
  Aggregate no longer touches the user's matplotlib settings.
- Dead-imports caught: html_title, suptitle_and_tight, ln_fit alias.
- Dead public helpers: mv, qdp, introspect, get_fmts,
  sensible_jump, GCN.
- Timer cruft + commented knobble_fonts(True) line.

Phase B (extractions):
- New aggregate/moments.py: MomentAggregator, MomentWrangler,
  xsden_to_meancv, xsden_to_meancvskew.
- New aggregate/iman_conover.py: iman_conover + ic_* +
  block_iman_conover + make_corr_matrix + random_corr_matrix +
  rearrangement_algorithm_max_VaR.

Phase C (moves into distributions.py):
- The *_fit family lives in distributions.py as a public, symmetric
  cluster: lognorm_fit (renamed from mu_sigma_from_mean_cv),
  gamma_fit, beta_fit, invgamma_fit, invgauss_fit, sln_fit,
  sgamma_fit. Also: approximate_from_mcvsk (renamed from
  approximate_work), lognorm_approx, lognorm_lev. ln_fit alias
  dropped (lognorm_fit is canonical).
- approximate_from_mcvsk's gamma branch refactored to call
  gamma_fit(m, cv) — symmetric with its lognorm branch using
  lognorm_fit.
- Private single-module helpers move + privatise:
  _estimate_agg_percentile, _picks_work,
  _moms_analytic + _partial_e + _partial_e_numeric,
  _integral_by_doubling, _logarithmic_theta.
- portfolio.py absorbs make_comonotonic_allocations as a public
  module-level function (named locally make_comonotonic_allocations_work
  to avoid clashing with the Portfolio method; re-exported clean as
  make_comonotonic_allocations).
- underwriter.py absorbs the merged _parse_note.
- spectral.py absorbs _short_hash.
- Update mu_sigma_from_mean_cv → lognorm_fit across the 3 rst doc
  files (2_x_actuary_student, 2_x_re_pricing,
  5_x_rearrangement_algorithm).

Phase D (renames + merges):
- parse_note + parse_note_ex merged into a single parse_note.
- pprint renamed to decl_pprint (DecL syntax highlighter; avoids
  stdlib name collision).

Phase F:
- Aggregate.plot() and bounds.py's FigureManager site → plain mpl.
- extensions/case_studies.py mpl call sites converted.
- aggregate/__init__.py re-exports trimmed and resourced.
- Stale docstrings refreshed.
- Module-top imports cleaned up post-deletions.

Keep public per user: tweedie_density, kaplan_meier(_np),
nice_multiple, decl_pprint. PEG regression bit-identical.
```
