# Plan: bivariate punch-up ([Bivariate-Punchup])

Status: FINAL DRAFT (v3), approved in review conversation, not yet executed.
v2 incorporated the author's rulings: the gate table goes private and its
downstream is re-sourced (no public `checks_df`), the in-core and massive
containers converge on one API, `total` is a cached property, and `stats_df`
is restructured to parallel `Portfolio.stats_df` (now in scope). v3 closes
the last three open points: `report=` is in, `stats_df` omits Portfolio's
`empirical` / `error` columns, and the small residual API difference on
`pushforward` (massive requires an explicit `bs`) is accepted and deferred
as `[Bivariate-Pushforward-Parity]`.

## Goal

Bring `BivariateAggregate` (and its distribution containers) up to the
first-class-citizen reporting pattern the rest of the library settled on, and
add the probability accessors an analyst reaches for first: marginal,
conditional, and total, each returned as a `GridDistribution`.

The pieces:

1. `.marginal(axis)` returning a `GridDistribution`.
2. `.conditional(kind, value)` returning a `GridDistribution`, with
   `kind` in `x | y | x+y | x-y`.
3. `.total`: cached property, the realized X + Y law as a `GridDistribution`.
4. A new `summary_df`: the at-a-glance Mean / SD / CV / Skew / P01 / Median /
   P99 frame, rows = the two marginals by name plus `total`, matching
   `Aggregate.summary_df` and `Portfolio.summary_df`.
5. `validation_df` becomes the eight-column theory-vs-realized moment audit
   (today's `summary_df` body), matching the other FCC. The current gate
   table goes private; its consumers are re-sourced.
6. `stats_df` restructured to parallel `Portfolio.stats_df`: moment-block
   rows, columns per marginal plus `independent` plus `total`.
7. One API across `BivariateDistribution` and
   `MassiveBivariateDistribution` for the probability surface.
8. Fix the CV to SD switch: a component declared with `ssev` (signed
   severity) does not flip the audit frame to SD today. Confirmed bug.

`info` is fine and untouched.

## Current state (for a reviewer with no context)

All in `src/aggregate/bivariate.py` unless noted.

- `BivariateAggregate` (line ~1209) is the joint-aggregate engine. Three
  construction modes: copula (two units + copula), netceded (the view pairs
  `netceded` / `grossnet` / `grossceded` built from one gross `Aggregate`,
  via DecL or `Aggregate.occ_bivariate()`), and discrete (`dbvsev`). All
  three produce the same object, so every change below rolls through to
  net/ceded views automatically.
- `.bivariate` (property) wraps the density as a `BivariateDistribution`
  (in-core) or `MassiveBivariateDistribution` (disk-backed zarr). Both share
  `JointBandsMixin`, whose `_row_bands` yields row bands uniformly across the
  two routes, and whose `slice(x=, y=)` already returns the normalized
  conditional on one axis as a `GridDistribution` (snap to bucket,
  renormalize, refuse a zero-mass slice).
- `MassiveBivariateDistribution.marginal(i)` already returns a
  `GridDistribution` (line ~4065). The in-core `BivariateDistribution` has
  only `marginals()` returning raw arrays, even though `slice`'s error
  message says "see `.marginal()`". Container API drift, item 7's motivation.
- `BivariateAggregate.summary_df` (line ~2756) is today the
  Portfolio-validation-shaped frame: `EX | Est EX | Err EX | CV | Est CV |
  Err CV | Sk | Est Sk` over Freq / Sev / Agg blocks, with a frame-wide CV or
  SD spread choice via `self._signed()`.
- `BivariateAggregate.validation_df` (line ~3061) is today a **check table**:
  one row per gate (marginal mean per axis, tail deficit) with `Est / Ref /
  Err / Gate / Pass` columns. It is the single computation behind the
  `validation` row of `info`, `validation_explanation`, `_explain_oneline`,
  and the `validation.insurer` exhibit registration in
  `src/aggregate/exhibits/_bivariate.py`. This shape does not match the other
  FCC and is retired from the public surface here.
- `BivariateAggregate.stats_df` (line ~2703) is theoretical vs empirical
  marginal moments per component: columns = the two unit names, rows =
  `(basis, stat)` with basis theoretical/empirical. Does not parallel
  `Aggregate.stats_df` (columns `mixed`/`independent`/`empirical`/...) or
  `Portfolio.stats_df` (rows `_PORT_STATS_ROW_INDEX`, a `(component,
  measure)` MultiIndex of meta / freq / sev / agg moment blocks; columns
  per-unit + `total` + `empirical` + `error`, see
  `_portfolio.py:_build_stats_df`).
- The FCC naming convention elsewhere: `summary_df` is the daily-driver
  headline (`Mean | SD | CV | Skew | P01 | Median | P99`; see
  `Aggregate.summary_df` at `_aggregate.py:5655` and `Portfolio.summary_df`
  at `_portfolio.py:1504`), `validation_df` is the moment-error audit, and
  validation *gates* live behind the frame (Aggregate: `Validation` flags +
  `explain_validation()`; the exhibit emphasis comes from
  `_moment_validation_emphasis` reading those flags, not from a public check
  frame). The bivariate predates that settlement and has the names crossed.
- `dev/TODO.md` item `[Bivariate-Total-Exeqa]` records the deferred design
  for conditioning on the total: where the axes share a `bs` the
  anti-diagonal is lattice aligned and exact; where they differ, route mass
  through `_scatter_1d` onto the total grid. The `x+y` conditional and
  `.total` below are the first consumers of exactly that primitive.

## Changes

### [Marginal-Accessor] `.marginal(axis)` returns a GridDistribution

New method on `BivariateAggregate` and on `BivariateDistribution` (the
massive container already has it; signatures align under
[Container-API-Parity]).

- Signature: `marginal(self, axis=0)`. Accept `0 | 1 | 'x' | 'y'` and an
  axis name from `unit_names` (the netceded views make `'net'` / `'ceded'`
  the natural spelling). One `_resolve_axis` helper shared with
  `conditional`.
- Returns `GridDistribution(self.axis_xs[i], marginal_density, bs=self.bs[i],
  name=<axis label>, is_loss_value=...)`. `is_loss_value=False` for a `pnl`
  axis (`_axis_kind(i) == 'pnl'`), `True` otherwise, so return periods read
  the correct tail.
- The `BivariateAggregate` method delegates through `.bivariate` so the
  in-core and massive routes share one entry point. The existing `marginals`
  (raw arrays) stays as the zero-copy primitive.
- Fixes the dangling `.marginal()` reference in `slice`'s error message.

### [Conditional-Accessor] `.conditional(kind, value, report=0)`

New method on `BivariateAggregate` (delegating to the container), `kind` in
`{'x', 'y', 'x+y', 'x-y'}`.

- `kind='x'`: the law of Y given X in the bucket containing `value`.
  Delegates to the existing `JointBandsMixin.slice(x=value)`; `kind='y'` is
  the transpose. Same snap-to-bucket and zero-mass refusal semantics.
  `slice` remains as the container-level primitive.
- `kind='x+y'`: the law of one axis given X + Y in the bucket containing
  `value`. The conditioning event is the total-grid bucket: mask cells with
  `round((x_i + y_j - value) / bs_total) == 0`, fold the masked mass onto
  the reported axis's grid, renormalize. Where `bs[0] == bs[1]` this is the
  exact anti-diagonal (`bs_total = bs`); where they differ, `bs_total =
  max(bs)` and the event is honest (a bucket of the total), consistent with
  how `slice` conditions on a bucket rather than a measure-zero line. This
  is the single-value version of the `[Bivariate-Total-Exeqa]` sweep and is
  written so that item can later reuse the band extraction.
- `kind='x-y'`: identical with `x_i - y_j`; the conditional support may be
  signed, which `GridDistribution` handles.
- `report=` (`0 | 1 | 'x' | 'y'` | axis name): which axis's law the returned
  1-D distribution is expressed in, default axis 0. Rationale: the diagonal
  band is indexed by both variables; the two readings are affine images of
  each other (`y = value - x`), but `GridDistribution` has no affine
  transform, so without `report=` a caller wanting Y's law would rebuild it
  by hand. The masked fold is symmetric, so supporting it is a one-line
  choice of fold axis. For `kind='x'` / `'y'` the reported axis is
  determined (the non-conditioning axis); a contradictory `report=` raises.
- Massive route: `x` conditionals read one row band (cheap); `y` and the
  diagonal kinds sweep `_row_bands` accumulating the masked fold, bounded
  memory. No new storage.
- Docstring cross-references `exeqa_df`: that is the conditional **mean**
  swept over the whole conditioning grid; `conditional` is the full law at
  one point. The per-row `GridDistribution` construction inside `exeqa_df`'s
  quantile sweep and `conditional('x', v)` should share the row extraction.

### [Total-Distribution] `.total`: cached property

Ruled in. The new `summary_df` needs P01 / Median / P99 of the realized
total, which requires the distribution of X + Y, not just its moments
(`_total_agg_empirical` is moments-only).

- Cached property on `BivariateAggregate` (and a container method under
  [Container-API-Parity]): fold the joint onto the total grid. Equal `bs`:
  exact anti-diagonal fold onto the shared lattice. Unequal `bs`:
  value-weighted `_scatter_1d` routing per `[Bivariate-Total-Exeqa]`
  (equivalently the existing `pushforward(x + y)`; the implementation picks
  whichever is cleaner, but the exact lattice path must be taken when
  available). The fold is O(nm); cache invalidation follows the density
  (recompute on `update`, same discipline as the other cached frames).
- Sanity anchor: in netceded mode the total has an exact reference, the
  gross aggregate (`ceded + net = gross` per occurrence), which becomes a
  test and a validation cross-check for free.
- Name vetting per house rule: `rg` confirms no existing `total` attribute,
  method, or spec key on `BivariateAggregate` or the containers
  (`total_key='total'` is only a `pushforward` kwarg default). `marginal`
  coexists with the `marginals` property; `conditional` is fresh.

### [Container-API-Parity] one probability API, in-core and massive

Ruled in: the two containers should present the same surface. Audit of the
drift:

| member | in-core | massive |
|---|---|---|
| `marginal(i)` | missing | present |
| `marginals()` | present | present |
| `slice(x=, y=)` | mixin | mixin |
| `moments` / `corr` / `transformed_moments` | present | present |
| `pushforward` | `(function, *, bs=None, ...)` auto-sizing | `(functions, bs, *, ...)` `bs` required |
| `contour` | present | missing (has `plot` / `explore`) |

- In scope: `marginal(i)`, `conditional(...)`, `total()` land on
  `JointBandsMixin` (implemented over `_row_bands` + the axis accessors both
  containers already expose), so both containers get them from one
  implementation and `BivariateAggregate` delegates uniformly. Add an FCC
  style parity test asserting the shared probability surface exists on both
  containers.
- Flagged, not in scope (YELL): unifying `pushforward` signatures is real
  work. The massive route requires an explicit `bs` because auto-sizing
  needs a min/max sweep over the on-disk density before scattering; giving
  it the in-core signature means either a paid pre-sweep or a refusal path.
  Proposal: record as `[Bivariate-Pushforward-Parity]` in `dev/TODO.md` and
  do it separately; this plan only aligns what the new accessors need.
  Plot-surface parity (`contour` vs `plot`/`explore`) likewise stays as is.

### [Summary-Validation-Swap] the FCC rename, gates re-sourced

- **New `summary_df`** (replacing the current one): rows = the two marginals
  by resolved label plus `total`; columns `Mean | SD | CV | Skew | P01 |
  Median | P99`, exactly the `Aggregate` / `Portfolio` layout (reuse
  `SUMMARY_PERCENTILES`, `_summary_pct_label`, `Aggregate._cv_or_nan`,
  `_snap_noise`). Marginal rows read moments and percentiles off
  `self.marginal(i)`; the `total` row reads `_total_agg_empirical()` for
  moments (exact from mixed moments) and `.total` for percentiles. Route the
  row axis through `self._relabel`. `CV` blanks per row via `_cv_or_nan`
  (so a near-zero-mean `pnl` axis blanks its own CV; `SD` never blanks).
- **`validation_df`**: the current `summary_df` body moves here verbatim
  (Freq / Sev / Agg blocks, eight audit columns, frame-wide CV or SD via
  `_signed()`), which makes it match `Aggregate.validation_df` and
  `Portfolio.validation_df` in shape and role. Docstrings updated to point
  each frame at its new sibling.
- **The gate table goes private and its downstream is re-sourced**, the way
  the other FCC manage gates (Aggregate: `Validation` flags behind
  `explain_validation()`; no public check frame):
  - The computation becomes a private helper (`_gate_checks()`, returning
    the same small frame it builds today). Public `checks_df` is NOT added.
  - `info`'s validation row, `validation_explanation`, and
    `_explain_oneline` read `_gate_checks()`; verdict text unchanged.
  - `exhibits/_bivariate.py`'s `validation.insurer` registration now
    receives the **new** `validation_df` (the moment audit) as its source
    frame, matching every other class, and derives its row emphasis from
    `_gate_checks()`: a failing marginal-mean gate emphasizes that
    component's `Agg` row, a failing tail deficit emphasizes the `total`
    `Agg` row. This mirrors `_validation_insurer_aggregate`, which
    emphasizes rows from the object's flags rather than from the frame.
  - `exhibits/_core.py`'s `validation` docstring ("`Pass == False` check
    rows on Distortion and BivariateAggregate") updates: only `Distortion`
    keeps a check-shaped `validation_df` after this plan. A later
    `[FCC-Validation-Uniformity]` pass for `Distortion` goes to
    `dev/TODO.md`.
- `qd()` (`utilities.py` ~352) prints `x.summary_df` for `Aggregate` /
  `Portfolio`; verify which branch the bivariate takes and leave the HTML
  repr (`.bivariate._repr_html_()`) alone. Visible effect of the swap: any
  headline that showed the audit frame now shows the at-a-glance frame.

### [Stats-Frame-Parallel] `stats_df` mirrors Portfolio's

Ruled in (was deferred in v1). The parallel with `Portfolio.stats_df`:

- **Rows**: the `(component, measure)` MultiIndex blocks of
  `_PORT_STATS_ROW_INDEX`: `meta` (limit, attachment, el, prem, lr, ...),
  `freq` / `sev` / `agg`, each with `ex1..ex3`, `mean`, `cv`, `skew`. Meta
  rows populate where the mode carries them (`_resolve_en` handles el /
  premium / lr in copula mode; netceded views inherit the gross terms) and
  hold NaN otherwise, exactly as Portfolio leaves non-applicable cells.
- **Columns**: one per marginal (each component's theoretical moments, the
  analog of Portfolio copying each unit's `stats_df['mixed']`), then
  `independent`, then `total`:
  - `independent`: the moments X + Y would have were the axes independent,
    from the marginal theory (means and variances and third central moments
    add). Cheap, purely analytic, and the natural benchmark: `total` against
    `independent` reads off the dependence lift, which is the whole point of
    building a joint.
  - `total`: the realized dependent total, from the joint mixed moments
    (`_total_agg_empirical` extended to fill the `ex1..ex3` raw rows). In
    netceded mode this column has an exact analytic reference (the gross
    aggregate), which the tests pin.
- The current frame's theoretical-vs-empirical *per-marginal* comparison is
  not lost: it is exactly what the new `validation_df` carries. `stats_df`
  returns to its FCC role, the canonical moment store.
- Portfolio's extra columns (`empirical`, `error`, `after_occ`, the reins
  impact columns) are not copied: the first two duplicate `validation_df`
  here, the rest are reinsurance plumbing the bivariate does not have.
  Ruled: omitted (author, v3).

### [Signed-Spread-Fix] ssev flips the audit frame to SD

Confirmed bug. `BivariateAggregate._signed()` (line ~2894) returns True only
when a `pnl` affine is present (`self._affine`), and hard-returns False in
netceded mode. `Portfolio._signed()` (`_portfolio.py:2304`) checks each
component's `Aggregate._signed()`, which covers `ssev` and negative-atom
`dsev`. Fix:

```
copula/discrete mode:  any pnl affine  OR  any(a._signed() for a in self.units)
netceded mode:         self._nc_agg._signed()
```

The netceded arm matters because an occurrence view pair built on a signed
gross aggregate has signed views. The fix lands in the (renamed)
`validation_df`, the only frame-wide CV/SD consumer; the new `summary_df`
blanks CV per row instead. Regression test: a copula bivariate with one
`ssev` component reports `SD | Est SD | Err SD` columns.

## Files touched

- `src/aggregate/bivariate.py`: everything above.
- `src/aggregate/exhibits/_bivariate.py`: validation registration re-sourced
  to the new `validation_df` with gate-derived emphasis; the stats exhibit
  picks up the reshaped `stats_df`.
- `src/aggregate/exhibits/_core.py`: the `validation` exhibit docstring line
  about check-shaped frames.
- `tests/test_bivariate.py` (~226, 239, 282-300, 477, 740, 790),
  `tests/test_bv_discrete.py` (~106, 113), `tests/test_create_pnl.py` (~43,
  110-112): readers of the old `summary_df` audit columns switch to
  `validation_df`; `stats_df` readers switch to the new layout.
- `tests/test_fcc_surface.py` (~711-752): the check-table tests become
  `_gate_checks()` behavior tests (gates still fire, verdicts still reach
  `info` / `validation_explanation`); the required-surface test passes
  unchanged since `summary_df` / `validation_df` / `stats_df` all still
  exist.
- New tests: `marginal` reproduces the standalone aggregate's quantiles;
  `conditional('x', v)` agrees with `slice`; `conditional('x+y', v,
  report=1)` is the affine mirror of `report=0`; an independent discrete
  pair matches hand arithmetic; equal-`bs` `.total` mean/SD agree with
  `_total_agg_empirical` to grid tolerance; netceded `.total` matches the
  gross aggregate; `stats_df` netceded `total` column matches gross theory;
  the ssev SD regression; container parity (in-core and massive expose the
  same probability surface); netceded objects answer all new accessors
  (roll-through).
- `tests/data/exhibit_snapshots.json`: recapture
  (`tests/capture_exhibit_snapshots.py`); the validation and stats blocks
  both change shape.
- `CHANGELOG.md`, `pyproject.toml`, `dev/TODO.md`: annotate
  `[Bivariate-Total-Exeqa]` (band primitive now exists), add
  `[Bivariate-Pushforward-Parity]` and `[FCC-Validation-Uniformity]`
  (Distortion) as new items.

## Suggested commit slicing (one bump per commit)

1. `[Marginal-Accessor]` + `[Conditional-Accessor]` + `[Total-Distribution]`
   + `[Container-API-Parity]`: pure additions on the containers and engine.
2. `[Summary-Validation-Swap]` + `[Signed-Spread-Fix]`: the breaking rename
   and the spread fix land together, since the fix lands in the renamed
   frame. CHANGELOG flags the rename in its one extra sentence.
3. `[Stats-Frame-Parallel]`: the reshape, separately revertible.

## Acceptance checks

- `uv run pytest` green per bump; tier 3 (`-m 'slow or not slow'`) at the
  final bump (the three bivariate suites are `slow`-marked and must see
  this).
- `qd(bv)` on a copula bivariate and on `a.occ_bivariate()` shows the new
  at-a-glance frame; `bv.validation_df` shows the eight-column audit;
  `bv.info`'s validation row and `bv.validation_explanation` unchanged in
  verdict; the validation exhibit emphasizes the right rows on a failing
  fixture.
- A bivariate with an `ssev` component shows SD columns in `validation_df`.
- `bv.marginal(0).q(0.99)` matches the standalone component's `q(0.99)`
  within grid tolerance; `bv.conditional('x+y', bv.total.q(0.99))` returns a
  normalized law supported inside axis 0's grid; netceded `bv.total` matches
  the gross aggregate's distribution within grid tolerance.
- `bv.stats_df` has `(component, measure)` rows and `unit0 | unit1 |
  independent | total` columns; the netceded `total` column equals gross
  theory.

## Resolved decisions (author rulings, v2 and v3)

- No public `checks_df`: gates go private, downstream re-sourced (above).
- `total` is a cached property.
- Container API parity is in scope for the probability surface; full
  `pushforward` signature unification is deferred, and the residual
  difference (massive requires an explicit `bs`) is accepted as
  understandable, not papered over.
- `stats_df` reshape is in scope, Portfolio-parallel, and omits Portfolio's
  `empirical` / `error` columns (they would duplicate `validation_df`).
- `conditional` gains `report=` (default axis 0) rather than axis-0-only.

No open questions remain; the plan is ready for a fresh-instance
implementation pass via the usual `/execute-plan` route.

## Deferred (noted, not in scope)

- `[Bivariate-Pushforward-Parity]`: unify the two `pushforward` signatures
  (massive needs an explicit `bs` today because auto-sizing wants a
  pre-sweep of the on-disk density).
- `[FCC-Validation-Uniformity]`: `Distortion.validation_df` is still a check
  table after this plan.
- `tail_df` / `tail_periods_df` on the **total**: with `.total` in hand the
  bivariate could serve a real return-period table for X + Y. Natural
  follow-on.
- `[Bivariate-Total-Exeqa]`: the full conditional-mean sweep on the total's
  grid stays deferred; this plan builds its band primitive.

## Execution log

Executed in this repo (LIB only), `/execute-plan`, starting 2026-09-03 at
`1.0.0a327`. One phase per bump per the commit slicing above. Baseline before
any edit: the fast suite showed 9 pre-existing failures (8 pricing exhibit
snapshot cases and `test_split_limit_policy_prices_the_per_accident_limit`),
consistent with the author's uncommitted `library.agg` edits and unrelated to
this plan; the gate for each bump is "no failures beyond that baseline set".

### Phase 1 (`1.0.0a328`): accessors and container parity

Divergences, each what the plan specified, what the code does, and why:

1. **`conditional` defaults `report=None`, not `report=0`.** The plan's
   signature reads `conditional(kind, value, report=0)`, but for `kind='x'`
   the reported axis is determined to be axis 1 and the plan itself says a
   contradictory `report=` raises, so a literal default of 0 would make
   `conditional('x', v)` raise. `None` means the natural axis: the non
   conditioning axis for `'x'` / `'y'`, axis 0 for the diagonal kinds, which
   preserves the plan's default-axis-0 ruling where a choice exists.
2. **The massive container's own `marginal(i)` was removed, not kept
   alongside.** The mixin implementation reads `marginals()`, which on the
   massive container returns the free pass-3 accumulators, so behavior and
   cost are identical and the one-implementation goal is met literally (the
   parity test asserts the three accessors are the same function objects).
3. **Axis-name matching is case insensitive.** Netceded `unit_names` are
   capitalized (`'Net'` / `'Ceded'`); the plan's natural spelling `'net'`
   would otherwise miss. `_resolve_axis` lowercases both sides.
4. **`.total` orientation**: the plan is silent on `is_loss_value` for the
   total; it is a loss unless both axes are `pnl` (payoff) axes.
5. **`exeqa_df` row-extraction sharing was not refactored.** The plan's
   soft ask ("should share the row extraction"): `slice` already is the
   shared one-row primitive behind `conditional('x', v)`, and `exeqa_df`'s
   sweep needs the whole band loop, so there was nothing left to factor.
6. **The dyadic grid pads the discrete fixture.** The hand-arithmetic test
   compares the live prefix of each density (the grid is a power of two, the
   padding is exact zeros).

New DecL program `bivariate MV.DBVIndep ...` added to
`src/aggregate/agg/decl-testers.agg` (DBVSEV block) per the corpus rule; it
round-trips (corpus suites green).

### Phase 2 (`1.0.0a329`): the summary / validation swap and the signed spread

Divergences:

1. **`qd()` gained a bivariate branch.** The plan said to verify which
   branch the bivariate takes and leave the HTML repr alone; it took the
   fall-through (`print(repr)`), which contradicts the acceptance check
   ("`qd(bv)` shows the new at-a-glance frame"). Resolved toward the
   acceptance check: `qd` now prints the repr line then `summary_df`,
   mirroring the `PnL` branch, with a pre-update guard (repr only, no
   density to summarize). The HTML repr (`.bivariate._repr_html_()`) is
   untouched.
2. **`tests/test_create_pnl.py` needed no change.** The plan lists lines
   ~43 and ~110 to 112 as readers of the old audit columns; they are PnL
   tests reading `PnL.summary_df` / `economic_df` and never touch the
   bivariate. Left alone.
3. **Two readers the plan's file list missed**, found by the suites and
   switched: `tests/test_massive_bivariate.py` (the 7-row `summary_df`
   shape assertion, now 3-row summary plus 7-row `validation_df`) and
   `tests/test_reins_bivariate.py::test_describe_and_info_netceded` (tuple
   indexing moved to `validation_df`).
4. **The exhibit snapshot was recaptured selectively.** A full recapture
   would also have re-baselined the eight pricing blocks that fail at
   baseline for reasons in the author's uncommitted scope, silently
   absorbing that drift. A merge script (session scratchpad) ran the
   capture and copied across only the four `*/BivariateAggregate` keys;
   the committed file differs from `HEAD` in exactly those four, verified
   by key-level comparison, and each was read: `summary` is the 3-row
   headline, `validation` the 7-row audit with the gate-derived emphasis
   caption on the insurer view.
