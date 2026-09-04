# Plan: approximate punch-up ([Approximate-Punchup])

Status: DRAFT v2 for final review, not executed. Written 2026-09-04 from
the investigation conversation of 2026-09-03/04; v2 incorporates the
author's in-file and message rulings of the same day. All questions are
now ruled: the DecL `approximate` keyword accepts the unshifted families;
the reflected (left-skew) fit stops being a special case; the emitted
severity clause type mirrors the input on **every** path, the DecL
keyword included (`sev` in, `sev` out, accepting the clamp mass at 0;
`ssev` in, `ssev` out); `approximation_df` takes the `stats_df` shape
(families as columns, a `meta` / `stats` / `quantiles` row MultiIndex);
the standalone `oep` chart is dropped; and the work lands in two bumps,
one per scope, not four.

## Goal

Finish the method-of-moments `approximate` surface so that every family the
fit core supports is reachable from DecL, every output form of
`Aggregate.approximate` / `Portfolio.approximate` is a first-class citizen
(the returned object carries a program and decompiles), and the reflected
fit rides the ordinary DecL reflection syntax instead of raising.

## Current state (for a reviewer with no context)

The fit core is `_approximate_sev_kwargs` in `src/aggregate/_fits.py`: given
`(m, cv, skew)` it returns `sev_*` kwargs for any of five families, `norm` /
`lognorm` / `gamma` (unshifted; `lognorm` and `gamma` are two-moment fits,
skew unmatched) and `sgamma` / `slognorm` (shifted, three-moment). It owns
the guards: a near-symmetric input degenerates a shifted family to the
normal, and a left-skew input is fitted by reflection (`sev_reflect=True`,
`sev_signed=True`). The output adapter `approximate_from_mcvsk` (same file,
line ~176) formats the core's kwargs into the requested `output` form.
Everything below is plumbing around that one core; the fits themselves do
not change.

- **DecL keyword**: `approximate KIND` parses as `APPROXIMATE ID` in
  `decl.lark` (~line 106), so the grammar accepts any word; the restriction
  is two hard-coded lists, `UnderwritingParser._APPROX_KINDS = ('exact',
  'sgamma', 'slognorm')` (`parser.py:1453`) and the constructor mirror check
  in `Aggregate.__init__` (`_aggregate.py:2402`). `norm`, `lognorm` and
  `gamma` are not reserved words, so no grammar change is needed to accept
  them. The constructor rewrite (`_aggregate.py:2406` on) calls the shared
  core, which already handles all five. See `dev/done/plan-approximate.md`;
  it restricted the keyword to the shifted pair without recording a reason
  to exclude the unshifted three.
- **`Aggregate.approximate(approx_type, output)`** (`_aggregate.py:6923`)
  and **`Portfolio.approximate`** (`_portfolio.py:2239`) both delegate to
  `approximate_from_mcvsk`. Outputs: `'scipy'` (frozen rv), `'sev_kwargs'`,
  `'sev_decl'`, `'agg_decl'`, and **any other string** returns an
  `Aggregate` built by direct construction, `Aggregate(**kwargs)`.
- **Confirmed bug, `norm` DecL fragment**: `_sev_kwargs_to_decl`
  (`_fits.py`, norm branch) emits pre-refactor syntax
  `'{scale} @ norm 1 # {loc}'`, which today's grammar rejects
  (`Unexpected character '@'`). `approximate('norm', output='agg_decl')`
  returns an unparseable program. The modern spelling is
  `{scale} * norm + {loc}` (verified: parses, and under plain `sev` clamps
  the sub-zero tail to an atom at 0).
- **Reflected fits raise today**: `_sev_kwargs_to_scipy`,
  `_sev_kwargs_to_decl` and therefore `output='scipy'` / `'sev_decl'` /
  `'agg_decl'` all raise `ValueError` for a left-skew (reflected) fit,
  claiming "no one-line DecL form". That claim is stale:
  `dev/done/plan-reflected-loss-severity.md` gave DecL full reflection
  syntax (`shift - X` via `sev2_rsub`, negative multipliers, unary minus;
  `parser.py:1875` on). Verified numerically: for a target
  `(m, cv, skew) = (100, 0.2, -1)` the reflected slognorm kwargs
  `(loc L, scale s, sigma)` render as
  `ssev {L} - {s} * lognorm {sigma}` and the built aggregate reproduces
  `(100, 0.2, -1)` exactly, identical to the direct-construction object.
- **The direct-constructed object is not first class**: `ob.program` is
  `''` (it never met the parser) and `spec_to_decl(ob.spec, 'agg', name)`
  crashes, because a constructor-shaped spec carries `label_map=None`
  (`decl_writer.py:499` calls `.get` on it) and `exp_attachment=None`
  (`_fmt_num` rejects `None`). Any hand-built `Aggregate` has the same gap;
  `approximate` is just where the author met it. Building the same program
  through `build(...)` yields a fully first-class object.
- **Signedness plumbing**: `Aggregate._signed()` (`_aggregate.py:3610`) is
  True when the severity reaches below 0 (an `ssev` continuous severity or
  a `dsev` with a negative atom); `Portfolio._signed()` asks each unit.
  The `ssev` keyword sets `sev_signed=True` in the spec; plain `sev` clamps
  a signed law at 0 with the clamp mass as an atom (verified:
  `sev 223.6 * norm + 500` builds with mean 500.99, the 1.3% sub-zero mass
  clamped to 0; the `ssev` twin is exact at 500).

## Changes

### [Approximate-DecL-All-Families] `approximate norm | lognorm | gamma` in DecL

Widen the two lists to
`('exact', 'norm', 'lognorm', 'gamma', 'sgamma', 'slognorm')`:
`_APPROX_KINDS` in `parser.py` and the constructor check in
`_aggregate.py:2402` (and its docstring at 2306). No grammar production
changes; update the comment block in `decl.lark` (~100 to 106) and the
kinds lists in `docs/4_agg_language_reference/ref_include.rst:103`,
`docs/2_aggregate_overview/info-strings.rst:60` and
`docs/2_aggregate_overview/features.rst:688`. Two behavior sentences for
the keyword docs: `lognorm` and `gamma` match mean and cv only (the
declared aggregate's skew is not reproduced), and `norm` matches mean and
cv with zero skew. The `_check_approx` occurrence-reinsurance rejection
and the degenerate-to-normal guard are family-independent and unchanged.

### [Approximate-Norm-Decl-Fix] the norm fragment speaks today's DecL

`_sev_kwargs_to_decl`'s norm branch emits `{scale} * norm + {loc}` (drop
the dead `@` / `#` spelling). Straight bug fix; the fragment is reachable
today through `approximate('norm', output='sev_decl' | 'agg_decl')` and
returns an unparseable string.

### [Approximate-Reflected-One-Line] reflected fits render as ordinary DecL

Ruled: not a special case. A reflected fit
(`sev_reflect=True`, kwargs `sev_name`, `sev_a`, `sev_scale=s`,
`sev_loc=L`) renders as the rsub form `{L} - {s} * {name} {a}` (for
`norm`, `{L} - {s} * norm`). Delete the reflected `ValueError` in
`_sev_kwargs_to_decl`; `output='sev_decl'` / `'agg_decl'` and the object
mode then serve every fit. `_sev_kwargs_to_scipy` keeps its refusal:
scipy has no frozen reflected rv, so `output='scipy'` for a reflected fit
still raises, with the message updated to point at the now-working
`'agg_decl'` and object outputs (and `approx_type='norm'`). The
`approximate('all')` survey keeps skipping families whose requested
output form cannot be built, which after this change is only
scipy-with-reflection.

### [Approximate-Sev-Type-Mirrors-Input] `sev` in, `sev` out; `ssev` in, `ssev` out

Ruled, and ruled to apply on **every** path, the DecL `approximate`
keyword / constructor rewrite included (see the ruled question below).
The emitted severity keyword follows the **source object's** own
signedness, not the fit's: `ssev` when `self._signed()` (for
`Portfolio.approximate`, `Portfolio._signed()`, any unit signed), else
plain `sev`. On the constructor path the rewrite sets
`sev_signed = self._signed_severity()` of the exact throwaway copy
rather than the fit's flag, and the `plan-approximate` regression tests
adjust their expectations accordingly. Consequences, stated in the
docstrings:

- A signed fit emitted under plain `sev` (the normal approximation of an
  ordinary loss aggregate, or a reflected fit of a left-skew loss
  aggregate) clamps its sub-zero tail to an atom at 0. The clamp is the
  point of the ruling (a loss stays a loss); it costs a small moment
  drift, which `validation_df` of the built object reports like any other
  discretization error.
- `output='sev_kwargs'` mirrors the same rule through `sev_signed`
  (`True` only for a signed input), so the kwargs and DecL forms describe
  the same distribution.
- The fit core is untouched; it keeps returning its own
  `sev_signed` / `sev_reflect` flags and the adapters decide the keyword.
  One fit core, per `plan-approximate.md`'s standing rule.

### [Approximate-Object-Via-Build] the object mode is `build(agg_decl)`

The "any other string" branch of `approximate_from_mcvsk` becomes: compose
the `agg_decl` program (which after [Approximate-Reflected-One-Line]
exists for every fit), append the note as a ` note{...}` trailer, and
return `build(program)` (module-level import inside the function, as the
`Aggregate` import is today). The returned object is parser-born: it
carries `program`, decompiles, round-trips, and lands in the underwriter's
knowledge under its fit name (`slog.T` style, overwriting on repeat calls,
which is the ordinary `build` behavior). Direct construction disappears
from this path entirely; there is no program-less fallback left.
Formalize `output='agg'` as the documented spelling for this mode in both
docstrings (any other unrecognized string keeps returning the object, as
today, so nothing breaks).

## Ruled question: mirroring on the DecL keyword path

Asked in v1: does [Approximate-Sev-Type-Mirrors-Input] extend to the DecL
keyword / constructor path, where today the surrogate's `sev_signed`
comes from the fit so the matched moments survive the grid
(`plan-approximate.md`, the signedness-test paragraph)?

RULING (author, in file, 2026-09-04): mirror sev/ssev. approximation_df
(below) will report achieved moments. This teaches that a shifted fit can
"fail" and introduce a mass at zero. Easy to change later if that's not
what we really want.

Consequence: the keyword path clamps like every other path (one rule
everywhere), the constructor rewrite reads the input's signedness, and
the moment drift a clamp introduces is not hidden but **displayed**, in
the achieved-moment rows of [Approximation-Frame] and in the surrogate's
own `validation_df`.

## Name vetting

No new public names. `output='agg'` is a new documented **value** of an
existing kwarg, not a name; `rg` shows no `output='agg'` caller today
(callers use `'scipy'`, `'sev_decl'`, `'agg_decl'`, `'sev_kwargs'`, or ad
hoc strings, all preserved). Private helpers keep their names.

## Files touched

- `src/aggregate/_fits.py`: norm fragment fix, reflected DecL rendering,
  keyword-mirroring, object-via-build; docstrings.
- `src/aggregate/parser.py`: `_APPROX_KINDS` widened.
- `src/aggregate/_aggregate.py`: constructor kind check + docstring
  (2306, 2402); `approximate` docstring (output vocabulary, reflected note
  now scipy-only).
- `src/aggregate/_portfolio.py`: `approximate` docstring likewise.
- `src/aggregate/decl.lark`: comment block only (~100 to 106).
- `src/aggregate/agg/decl-testers.agg`: `approximate` lines for the three
  new kinds (corpus round-trip; the unparser clause at
  `decl_writer.py:716` is kind-agnostic and needs no change).
- `tests/data/expected_specs.json` if new lines go to `test_suite.agg`
  instead; the snapshot recapture is routine and its diff read.
- `tests/test_approximate.py`: new cases per below.
- Docs: the three `.rst` kind lists; note in commit that the docs build is
  pending, per standing rule.
- `CHANGELOG.md`, `pyproject.toml`, `dev/TODO.md` (log
  `[Spec-Decompile-Robustness]`, below).

## Tests

- DecL: `approximate norm` / `lognorm` / `gamma` build; `norm` surrogate
  matches mean and cv; an unknown kind still raises naming all six.
- `approximate('norm', output='agg_decl')` parses and builds (the bug).
- Reflected: a left-skew input's `agg_decl` builds and reproduces
  `(m, cv, skew)` to grid tolerance (the `(100, 0.2, -1)` fixture above);
  `output='scipy'` for it still raises.
- Typing: a plain `sev` input yields a program spelled `sev ...` whose
  build carries an atom at 0 under a normal fit; an `ssev` input yields
  `ssev ...` and `_signed()` True; `sev_kwargs` mirrors via `sev_signed`.
- Object mode: `ob = a.approximate('slognorm', output='agg')` has
  `ob.program` nonempty, `build(ob.program)` round-trips, and
  `spec_to_decl(ob.spec, 'agg', ob.name)` succeeds; works for the
  reflected fixture too.
- `approximate('all')` returns all five families for a right-skew input on
  the object output, and skips only scipy-with-reflection on `'scipy'`.
- Portfolio: `Portfolio.approximate` object mode likewise parser-born.

## Commit slicing (RULED: one bump per scope, no small bumps)

Two bumps for the whole plan (the author's in-file ruling, "Do in one
bump. Do not need these small bumps.", applied to each scope):

1. The whole first scope, one unit: [Approximate-Norm-Decl-Fix],
   [Approximate-Reflected-One-Line], [Approximate-Sev-Type-Mirrors-Input]
   (constructor path included per the ruled question),
   [Approximate-Object-Via-Build] and [Approximate-DecL-All-Families].
2. The whole second scope, one unit: [Approximation-Frame],
   [Approximation-Density-Frame], [Approximation-Exhibit],
   [Approximation-Chart].

## Acceptance checks

- `uv run pytest` green per bump (against the known pre-existing baseline
  failures, if still present); tier 3 at the final bump.
- `build('agg X 10 claims sev lognorm 50 cv 1 poisson approximate norm')`
  builds; its `info` reports the approximation; mean and cv match theory.
- `a.approximate('slognorm', output='agg')` returns an object whose
  `program` builds the same distribution; same for a left-skew input.
- `qd` of a clamped normal surrogate shows the atom at 0 doing what the
  ruling says (mass at 0, small mean lift), and `validation_df` reports
  the drift rather than hiding it.

## Deferred (noted, not in scope)

- `[Spec-Decompile-Robustness]` (to `dev/TODO.md`): `spec_to_decl` assumes
  parser-shaped specs; a constructor-shaped spec (`label_map=None`,
  `exp_attachment=None`) crashes it (`decl_writer.py:499`, `_fmt_num`).
  After this plan no library path constructs an `Aggregate` it hands to a
  user without a program, but a user's own `Aggregate(**kwargs)` still
  cannot decompile. Harden the writer (treat `None` as absent) as its own
  small pass.
- Frozen scipy form for reflected fits: would need a reflected-rv wrapper
  class; not worth it while the object and DecL outputs cover the case.

## Second scope (added 2026-09-04): the approximation frames, exhibit, chart

Added after discussion with the API agent, aimed at the aLL teaching
application. Ruled by the author: both frames are **on-demand properties,
no options, always all five families plus the exact row**; the chart draws
the five approximations plus the sub-exponential implied tail.

### [Approximation-Frame] `approximation_df` on Aggregate and Portfolio

RULED shape (author, 2026-09-04): like `stats_df`. **Columns** are the
approximations, `exact | norm | gamma | lognorm | sgamma | slognorm`
(`exact` first as the anchor column). **Rows** are a two-level MultiIndex
whose first level is the block, `meta` / `stats` / `quantiles`:

- **`meta` block**: the fitted distribution. Row
  `('meta', 'distribution')` carries the DecL severity fragment as a
  string, self-describing across families with different parameter counts
  (the reflected form reads `L - s * lognorm sigma`; the emitted keyword,
  `sev` or `ssev`, rides in the fragment's home program rather than
  here). Rows `('meta', 'shape')`, `('meta', 'loc')`, `('meta', 'scale')`
  hold the numeric parameters, `NaN` where a family has none. The `exact`
  column is `''` / `NaN`. (One string row makes the frame object-dtyped;
  greater-tables types per cell, so this costs nothing downstream.)
- **`stats` block**: rows `mean`, `cv`, `skew`, `ks`. The moments are the
  **achieved** moments of the **emitted** law, clamp included (author's
  ruling): under a plain `sev` input the sub-zero tail sits as an atom at
  0, so even a three-parameter (shifted) family can differ from the exact
  column, which is exactly what the frame teaches; the two-parameter
  `lognorm` / `gamma` differ in `skew` by construction (`norm` at zero
  pre-clamp). The `exact` column is the realized grid moments. `ks` is
  the Kolmogorov distance `sup |F - G|`, `F` the realized cumulative (one
  `cumsum` of the existing density) and `G` the family's emitted-law cdf
  vectorized over the same grid (the Berry-Esseen quantity module 2 of
  the teaching application currently measures in a Python workbook
  because the browser cannot); cheap by construction, one cumsum and one
  cdf call per family, no new grid. `ks` is 0 in the `exact` column.
- **`quantiles` block**: one row per non-exceedance probability of the
  same symmetric ladder `tail_df` uses (both `1/T` and `1 - 1/T` per
  rung of :data:`DEFAULT_RETURN_PERIODS`, so `P` runs 0.001 to 0.999).
  The `exact` column reads the grid (`q(p)`); family columns evaluate the
  emitted law analytically (the reflected fit as `L - s * q_base(1 - p)`,
  a `sev`-typed clamp as `max(0, ppf)`; no frozen scipy object needed).

No separate error block: with families as columns, an error is a column
subtraction against `exact`, the same reading `stats_df` gives its
`independent` / `total` columns, and the exhibit stays one screen wide.

Implementation notes. Everything a family column reports is the law of
the **emitted program** (mirrored keyword, clamp included), one law per
column, no mixing of the fit target and the fit result. The cdf and ppf
come from the family's closed forms with the reflect and clamp transforms
applied; the achieved moments come from the same grid-mass evaluation
that feeds `ks` and [Approximation-Density-Frame] (one dot product per
moment), so the `stats` and `quantiles` blocks cannot disagree about
which law they describe. Evaluate `G` at the bucket convention
`F(x_k) = P(X <= x_k)` so the two cumulatives are compared at the same
points; say the convention in the docstring.

Portfolio: the same frame on the **total** (its realized grid and grid
quantiles; the fits come from `Portfolio.approximate('all')`, which shares
the core). Both on demand, computed each call like the other derived
frames; no caching, no arguments, always all five families plus exact.

### [Approximation-Density-Frame] `approximation_density_df`

Index: the existing output grid `xs` (the total's grid on Portfolio).
Columns: the exact realized density (named `exact`), then one column per
family holding that family's density expressed in grid mass terms
(`pdf(x) * bs`, so the columns overlay the discrete density directly; the
reflected fit through the same closed form). This is the plotting feed for
[Approximation-Chart] and for the class `plot` route; also on demand, no
options.

### [Approximation-Exhibit] one registration line

```
register_simple_exhibit(
    'approximation', 'Approximation', 'approximation_df',
    [Aggregate, Portfolio], predicate=_perspectives_updated,
    caption=...)
```

`'approximation'` collides with no registered exhibit name, and the two
new property names collide with nothing (`rg` clean; the existing
`Aggregate.approximation` noun attribute is a different name and stays).
Through the existing exhibit route this is a formatted, greater-tables
rendered, CSV-exportable table in aLL with one nav leaf as the entire app
cost; per the API agent it covers teaching modules 1, 2 and half of 4 with
no chart work. Exhibit snapshots recapture (new keys, additive).

### [Approximation-Chart] `register_chart('approximation', ...)`

Upstream chart, own registered name per the existing `agg` / `reins` /
`kappa` separate-names pattern (so the API's `_chart_options` needs no
change and it gets its own nav leaf). `primary=Aggregate`, charts-module
`_updated` predicate. Draws, from the subject's own computations only:

- the exact density / tail (the subject),
- the five moment-matched families off
  [Approximation-Density-Frame] / the fit core,
- the **sub-exponential implied tail** `E[N] * S_X(x)` via the exact
  severity functions (`Aggregate.sev.sf`; exact, not gridded, which the
  `sev` property docstring at `_aggregate.py` ~6807 says is the right
  call for exceedance questions).

The implied-tail curve needs a single severity, so the chart registers on
`Aggregate` only; `Portfolio` keeps the frames and exhibit. Presentation
(a survival-tail panel where the Berry-Esseen story is visible, windows,
log depth) follows the shared two-panel conventions in
`charts/_two_panel.py`; the emitter design is implementation detail for
its bump.

### Ruled point: one chart, oep dropped

The API-side sketch proposed **three** chart names, `approximation`,
`asymptotic` and `oep`, each a separate leaf. RULED (author,
2026-09-04): one chart. The asymptotic curve (`E[N] * S_X(x)`) folds
into the approximation chart, which covers the `asymptotic` leaf's
content, and the standalone `oep` chart is **dropped** (not deferred, not
logged; :func:`aggregate.utilities.oep` stays as it is, upstream and
unexposed).

### Additions to files, tests, slicing

- Files: `_aggregate.py` and `_portfolio.py` (the two properties),
  `exhibits/__init__.py` (one registration plus snapshot recapture),
  `charts/_emit_aggregate.py` or a new `charts/_emit_approximation.py`
  (whichever the registry's one-module-per-chart convention prefers),
  format sheets if the new columns need typing
  (`aggregate/formats/formats-raw.yaml`).
- Tests: `approximation_df` has the six family columns and the
  `meta` / `stats` / `quantiles` row blocks with the tail-ladder `P`
  values; on an `ssev` fixture (no clamp) the shifted columns' achieved
  `mean` / `cv` / `skew` equal the exact column's to fit tolerance while
  the two-parameter columns differ in `skew`; on a plain `sev` fixture
  with a normal fit the achieved mean shows the clamp lift and the `meta`
  fragment builds under `sev`; `ks` is 0 in the exact column, positive
  elsewhere, and decreases from `norm` to the shifted families on a
  right-skew fixture; `approximation_density_df` columns each sum to ~1
  over the grid (the clamped and reflected cases included); the exhibit
  builds and snapshots; the chart doc emits with the expected series
  count.
- Slicing: the whole second scope is bump 2 of the ruled two-bump plan
  (see Commit slicing above).

### Additions to acceptance checks

- `a.approximation_df` on an updated lognormal-severity aggregate: six
  family columns over the `meta` / `stats` / `quantiles` blocks, the
  `slognorm` and `sgamma` columns near the exact moments (differing by
  the clamp mass where the `sev` mirror clamps), and the `ks` row
  reproducing the workbook's Berry-Esseen measurements to grid
  tolerance.
- `p.approximation_df` on an updated Portfolio reads the total.
- The `approximation` exhibit serves through
  `build_exhibit(a, 'approximation')` and renders in greater-tables.
- `charts.build_chart_doc(a, 'approximation')` emits the subject, five
  family curves, and the implied-tail series.

## Execution log

Executed from this plan per the execute-plan skill; one bump per scope as
ruled. Divergences recorded here at the moment they were made.

### Bump 1, `1.0.0a331` (scope 1, 2026-09-04)

Landed: [Approximate-DecL-All-Families], [Approximate-Norm-Decl-Fix],
[Approximate-Reflected-One-Line], [Approximate-Sev-Type-Mirrors-Input]
(constructor path included), [Approximate-Object-Via-Build].

Divergences:

1. **`approximate_from_mcvsk` signature.** The plan left the mirroring
   plumbing unnamed. Implemented as a new `signed_input=False` keyword, and
   the `agg_str` argument now excludes the trailing severity keyword
   (callers pass `'agg NM 1 claim '`; the adapter chooses `sev` / `ssev`).
   Both callers are in-repo; the function is exported through
   `aggregate.distributions` but is not in the stable tier. Noted in the
   CHANGELOG.
2. **The `spec_to_decl(ob.spec, ...)` test assertion was dropped.** The
   plan's Tests section asks that the object mode's
   `spec_to_decl(ob.spec, 'agg', ob.name)` succeed, but that call fails for
   **every** built object today, parser-born included: `Aggregate._spec`
   captures all constructor arguments, so `label_map` is present-but-`None`
   (the library's own decompile route, `underwriter.py` `recipe`, uses the
   sparse parser spec, never `_spec`). The dense-spec gap is already
   tracked as `[Unparse-Dense-Spec-Guard]` in `dev/TODO.md`, whose recorded
   disposition (a guard, not `None`-tolerance) contradicts the hardening
   sketched under this plan's deferred `[Spec-Decompile-Robustness]`; that
   deferred item therefore folds into the existing entry rather than
   becoming a duplicate. First-classness is tested through
   `build(ob.program)` round-trips instead.
3. **Constructor rewrite loc default.** The unshifted `lognorm` / `gamma`
   fits carry no `sev_loc`; the keyword-path rewrite read
   `_fit['sev_loc']` and now reads `_fit.get('sev_loc', 0.0)`.
4. **Test expectation updates per the mirroring ruling** (anticipated by
   the plan): `test_left_skew_reflect_matches_exact` became
   `test_left_skew_reflect_clamps_under_sev` (unsigned input now clamps;
   measured drift ~5e-5 mean / ~2e-3 cv / ~3e-2 skew relative on the
   beta(5, 1.3) fixture) with a new exact `ssev` twin;
   `test_symmetric_uses_normal_limit`'s realized-skew tolerance widened to
   1e-3 (the clamped ~1e-5 sub-zero mass lifts skew to ~1.3e-4).
5. **Pre-existing baseline failures** (verified present with this plan's
   edits stashed): 8 `pricing.stand_alone` / `pricing.allocate` exhibit
   snapshot cases and `test_split_limit_policy_prices_the_per_accident_limit`
   (the author's in-flight `library.agg` Capstone picks edit). Not touched.

### Bump 2, `1.0.0a332` (scope 2, 2026-09-04)

Landed: [Approximation-Frame], [Approximation-Density-Frame],
[Approximation-Exhibit], [Approximation-Chart].

Divergences:

1. **The shared frame engine lives in `_aggregate.py` at module level**
   (`approximation_frame`, `approximation_density_frame`,
   `_approximation_laws`, `APPROXIMATION_FAMILIES`), the
   `return_period_frame` arrangement, imported by `_portfolio.py`; the
   plan's files list named only the two property hosts.
2. **The chart registers with no `primary`.** The plan's
   [Approximation-Chart] says `primary=Aggregate`, but `register_chart`'s
   `primary` means "the object's own picture" (`primary_chart` returns the
   first claimant in registration order) and the `agg` chart already
   claims Aggregate; a second claimant would be order-fragile and
   semantically wrong. Registered like `reins` / `kappa`: dispatch on
   Aggregate, `_updated` predicate, no primary. `primary_chart` verified
   unchanged.
3. **Inadmissible families are served as NaN columns and skipped by the
   chart.** On a negative-mean (signed) subject the unshifted `gamma` /
   `lognorm` fits have no valid law; the frame keeps their NaN columns
   (self-describing) and the chart drops the non-finite curves, since NaN
   cannot travel in canonical JSON. The implied-tail curve is likewise
   trimmed to where `E[N] * S_X(x) <= 1` and above `SURVIVAL_FLOOR`.
4. **Exhibit snapshots merged additively.** The committed
   `exhibit_snapshots.json` carries drifted `pricing.stand_alone` /
   `pricing.allocate` entries (8 pre-existing failures, see bump 1 note
   5); a full recapture would have silently absorbed that drift into this
   commit, so only the 8 new `approximation/*` keys were computed and
   merged, leaving every existing entry byte-identical.
5. **`dev/FEATURES.csv` is hand-curated** (`regen_features.py` is an
   auditor, not a generator); the two property rows were added by hand and
   the auditor passes.

Findings for the author (not acted on):

- **Half-bucket bias on coarse grids.** The plan's grid-mass convention
  (`F(x_k) = P(X <= x_k)`, family mass `diff(G(xs))`) reads a continuous
  law's mass at the bucket's right edge, a systematic `+bs/2` mean shift
  visible on coarse discrete fixtures (the dice portfolio reads family
  mean 8.5 against exact 8.0 at `bs=1`). Negligible on production grids
  and it is the convention the plan specifies; flagging in case the
  teaching frame meets a discrete example.
- The 8 pricing exhibit snapshot failures and the 2 `library.agg`
  Capstone failures pre-exist this plan (verified by stashing); they ride
  the author's in-flight work.

RuntimeWarning gate (numerics-touching bump): after guarding the frame's
deliberate probe of inadmissible families with `np.errstate` (divergence
3's companion; Notes paragraph in `_approximation_laws`), the plan's code
is green under `-W error::RuntimeWarning`. One pre-existing emission
surfaced, a third finding for the author: **`moments.py:500` sqrt of a
negative on `agg:RenewalDeterministicWait`** (an a327 library entry;
nothing in this plan touches moments or renewal), which fails
`test_every_library_entry_builds` under the gate only.
