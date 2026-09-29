# plan-a368: the kappa chart moves to the Palm route, per layer, and lights for a P&L

**Status:** drafted 2026-09-29; the author's rulings on the two open
questions recorded the same day (see Rulings at the end). Awaiting author
review before execution.
**Repo:** `V:\worktrees\aggregate_REFACTOR`, the `aggregate` library, two
bumps `1.0.0a368` and `1.0.0a369`, phases in order below. One small
`aggregate-api` commit at the end (nav residue and one test; version
mechanics per that repo's rules, checked at execution). The same session
executes both sides.

## Goal

The app's Pricing, Plot tab is the `kappa` chart: the conditional cession
curves behind the natural allocation. Today an `Aggregate` with an
occurrence program serves it off the 2-D `occ_joint`, showing only the
**total** ceded curve with a percentile band. Three changes, per the
author's request of 2026-09-29:

1. **The default route becomes the Palm conditional-mean identity** (the
   a362 kernel), computed on the fine model grid with no joint: mean curves
   only, but now **one curve per occurrence layer** beside the total, the
   net mirror, and the identity. The percentile band is a property of the
   conditional *distribution*, which only a 2-D joint knows, so it is kept
   exactly where a joint is the input: a `BivariateAggregate` serves the
   band chart unchanged, and a new `bands=` option on the aggregate route
   opts back in (author's ruling on question 1).
2. **A frequency without a pgf derivative falls back to today's joint band
   chart**, so chart availability does not move for any object.
3. **The chart registers for `PnL` through its wrapped engine**, so a
   reinsured P&L lights Pricing, Plot, including the multi-layer peel whose
   per-layer curves the 2-D route cannot produce at all. The conditioning
   subject stays **gross** (author's ruling on question 2): Pricing, Plot
   remains the premium-splitting picture, and the net-conditioned story
   already lives in the waterfall.

## Background: the current chart (verified 2026-09-29)

- `chart_kappa` lives in `charts\_emit_bivariate.py`. Constants
  `KAPPA_CDF_RANGE = (1e-3, 0.999)` (`:473`) and `KAPPA_LEVELS =
  (0.01, 0.99)` (`:478`). Three registrations:
  - `BivariateAggregate` (`_kappa_band`, `:553-697`): two panels over one
    gross axis. **cession** carries `E[ceded | gross]`, its percentile band
    (a `y2` series), the mirrored net curve and band (`g - kappa`,
    identity-derived), and the identity line; **share** carries the total
    ceded share and the single-layer ceiling (`_kappa_ceiling`, `:483`).
    Refuses a joint with no gross axis (`_kappa_frame`, `:531`).
  - `Aggregate` (`_kappa_from_aggregate`, `:700-710`): a thin delegate to
    `_kappa_band(agg.occ_joint(views=('gross', 'ceded')))`, i.e. one 2-D
    FFT on the budget grid (memoized, but built if absent).
  - `Portfolio` (`charts\_emit_portfolio.py:233`): the independence-based
    kappa panel. **Untouched by this plan.**
- Availability is `_kappa_available` (`:713-732`), duck-typed: a book with
  units and a density; a netceded joint with a gross axis; an aggregate
  with `occ_reins` and a realized grid. No `PnL` branch.
- The chart is **occurrence-only** by construction: `occ_bivariate` ignores
  aggregate reinsurance, and this plan keeps that scope. A per-layer
  aggregate-tier curve conditioned on gross would need the joint of the
  dependent pair (retained, gross), a genuinely 2-D object, and is out of
  scope.
- Tests: `tests\test_chart_kappa.py` (band structure, identity footing,
  reflected band, ceiling comb, lattice payload, hash round trip, render,
  disk-backed joint). There is no global chart snapshot file; each chart
  test pins its own doc.
- The SPA (`V:\dev\aggregate-api\web\src\nav.js:216-219`): Pricing, Plot
  gates on chart `kappa`, `why: 'needs a book of units or an occurrence
  cession'`. The tab lights from the capability payload, so the PnL
  registration is LIB work; the API owes the `why` hint, comments, and a
  capability test. `test_capability.test_every_listed_chart_serves`
  asserts every listed chart serves, which is why availability and the
  fallback must agree exactly.

## Background: the Palm machinery this builds on (landed a362 to a363)

- `Aggregate.palm_kappa(target, conditioning=None)` returns
  `(kappa, conditioning_density)` on the model grid;
  `conditioning=None` conditions on the **gross** aggregate, which is
  exactly this chart's axis. `NaN` where the conditioning density is
  negligible; raises `NotImplementedError` for a frequency without
  `freq_pgf_prime` or on a signed or windowed grid.
- Per-layer occurrence ceders come from
  `make_ceder_netter([clause])[0]` per layer; the validator rejects
  overlapping cessions, so per-layer ceders sum identically to the program
  ceder and the per-layer kappas sum to the total by linearity of the
  moment measure (the exact footing the tests pin).
- Layer display labels: the declared `as` label from the engine's
  `label_map`, falling back to the DecL descriptor
  (`_pnl_builders._reins_layer_label`, `:383`). That helper lives in the
  P&L builders; the chart module gets its own small copy or the helper is
  hoisted, decided at execution by import hygiene (charts must not import
  `_pnl_builders`; prefer a hoist to `_reinsurance.py`, which both may
  import).
- `charts\_emit_structure.py:112` has `_engine(obj)`: the wrapped
  `Aggregate` of a `PnL`, `None` for anything else, the pattern a367 used
  for the reins chart.

## Names introduced (vetted 2026-09-29; re-run `rg` at execution)

- `Frequency.supports_pgf_prime`: read-only property, `True` when the
  family can evaluate the pgf derivative (the `_panjer_ab0` flag, or a
  subclass override of `freq_pgf_prime`). Sits beside the existing
  `supports_zm` class attribute in name and spirit. Needed because chart
  availability and the emitter's fallback must agree without a try/except
  at predicate time. No existing `supports_*` or `*_pgf_prime` name
  collides.
- `bands` keyword on the aggregate `chart_kappa` route (default `False`):
  `True` builds (or reuses, via the `occ_joint` memo) the budget-grid
  joint and overlays the total ceded and net percentile band series on the
  Palm chart. No emitter in `charts\` takes a `bands` kwarg today.
- Emitter-internal `_kappa_palm(agg, ...)` beside `_kappa_band`; the
  registered `Aggregate` function keeps its name
  (`_kappa_from_aggregate`) and becomes the route switch.

## Phase [Palm-Kappa-Chart], bump 1.0.0a368

`Frequency.supports_pgf_prime` in `_frequency.py`, with a docstring noting
it is the capability the kappa chart and the Palm ladder key on.

In `charts\_emit_bivariate.py`:

- `_kappa_palm(agg, cdf_range=KAPPA_CDF_RANGE, ceiling=True)`: the Palm
  chart. One `palm_kappa` call per occurrence layer (target the layer
  ceder, conditioning `None`, so the axis is gross); the total ceded curve
  is the **sum** of the layer curves (exact, and asserting it foots is a
  test, not a recomputation); net is `g - total`; identity as today. The
  gross axis is windowed to `cdf_range` on the conditioning density the
  kernel returns (the gross compound), mirroring the band chart's window
  on the gross marginal, which also discards the NaN guard cells. Series
  roles: layers and total carry role `ceded` with distinct names (the
  layer label helper above; the total named `E[ceded | gross]` exactly as
  today so readers and downstream styling see a familiar name), net and
  identity as today. The **share** panel carries the total ceded share and
  the ceiling only; per-layer share curves would clutter a panel whose
  reading is one number against the ceiling. A single-layer program serves
  one layer curve that coincides with the total; serve the total only in
  that case rather than two identical lines. `meta` records
  `route: 'palm'` and `band: 'none'`.
- `_kappa_from_aggregate(agg, bands=False, **options)` becomes the route
  switch: `bands=True`, or a frequency with
  `not agg.frequency.supports_pgf_prime`, takes the joint; otherwise the
  Palm route. With `bands=True` the Palm chart is drawn and the two band
  series (ceded and net, `y2` form, budget-grid lattice payload) are
  appended from `occ_joint(views=('gross', 'ceded')).exeqa_df`; with an
  unsupported frequency the served chart is exactly today's
  `_kappa_band(agg.occ_joint(...))`, so nothing regresses for the exotic
  families (`meta` records `route: '2d-fallback'`). The
  `BivariateAggregate` registration is untouched.
- `_kappa_available` is unchanged in this phase (aggregate availability
  still reads `occ_reins` plus a realized grid; the fallback is what keeps
  that true).

Tests (`tests\test_chart_kappa.py`, extending the existing file):

- The Palm doc's structure: two panels, one gross axis, per-layer series
  present on a two-layer tower with role `ceded`, no band series, meta
  `route: 'palm'`.
- Footing: layer curves sum to the total, and total plus net equals the
  identity, cell by cell (`1e-9`).
- Cross-route agreement: the Palm total-ceded curve agrees with the 2-D
  `exeqa` curve at the joint's grid points within a tolerance set at
  implementation (budget-grid coarseness; record the observed gap, expect
  percent-level as in `tests\test_palm.py`).
- `bands=True`: the band series appear and match the values the pure band
  chart serves.
- Fallback: a sichel (or logarithmic) frequency with an occurrence program
  still answers `kappa` in `available_charts` and serves the band chart,
  meta `route: '2d-fallback'`.
- Single-layer program serves the total without a duplicate layer curve;
  ceiling still present.
- Hash round trip and render for the Palm doc, mirroring the existing
  band-doc tests.
- `supports_pgf_prime` truth table over the family registry (True for the
  ten supported families, False for logarithmic, sig, beta, sichel,
  neymana, pascal).

`dev/FEATURES.csv` gains the `supports_pgf_prime` row (frequency-api
group). CHANGELOG entry states the default route change and that the band
moved behind `bands=` and the `BivariateAggregate` input; a caller reading
the aggregate chart's band series must now opt in, which is breaking for
that (provisional-tier) consumer and said plainly.

## Phase [PnL-Kappa-Chart], bump 1.0.0a369

- `chart_kappa.register(PnL)`: a thin delegate through
  `_emit_structure._engine`, exactly the a367 `chart_reins` pattern; the
  engine takes the same route switch, so a P&L over an exotic frequency
  serves the fallback band chart and a peel serves per-layer Palm curves.
  The doc title reads the P&L's own label (decided at execution: pass a
  title override or accept the engine's label, matching whatever
  `chart_reins` ended up doing for consistency between the two tabs).
- `_kappa_available` gains the engine look-through: an object with an
  `engine` attribute answers for its engine when that engine is an
  `Aggregate` (a P&L over a `Portfolio`, and a hand-built ledger with no
  engine, stay dark). Duck-typed like the rest of the predicate, so the
  module still imports no classes.
- Conditioning is gross (ruling 2): no new mathematics, the engine's own
  chart.

Tests: `available_charts` on the reinsured-tower P&L and on the two-layer
peel P&L include `kappa`, on the plain (uncessioned) P&L do not; the peel
doc carries one curve per layer; the tower P&L doc equals the engine's own
doc apart from any title difference; hash round trip and render. Placed in
`tests\test_chart_kappa.py` beside the rest, with the builds inline as in
`test_chart_pnl.py`.

The plan moves to `dev/done/` with this bump.

## Phase [API-Kappa-Residue], aggregate-api commit (no LIB bump)

In `V:\dev\aggregate-api`: the Pricing, Plot leaf's `why` extends to the
P&L ('needs a book of units, an occurrence cession, or a reinsured P&L'
or similar wording settled at execution); the leaf's comment block and the
Pricing group commentary note the 2026-09-29 supersession and the route
change (means on the fine grid by default, band on a joint input). A
capability test asserts `kappa` appears in a reinsured P&L's chart list
and not in the plain P&L's, beside the existing structure-chart test at
`tests\test_capability.py:183`. `test_every_listed_chart_serves` then
covers serving automatically. Version and CHANGELOG per that repo's rules
(next is `1.0.0a177` as of drafting; read at execution), committed by the
same session, separately from the LIB bumps.

## Acceptance checks

1. `uv run --no-sync pytest tests/test_chart_kappa.py tests/test_palm.py`
   green at each phase; the full fast suite once per bump; tier 3 at each
   commit boundary per the version-bump skill.
2. On a two-layer occurrence tower (`Aggregate`): per-layer curves foot to
   the total and the identity; the Palm total agrees with the 2-D `exeqa`
   curve at the joint's grid points within the recorded tolerance;
   `bands=True` restores the band.
3. An occurrence program under sichel still lists and serves `kappa`
   (fallback), byte-identical to today's chart.
4. The peel P&L serves `kappa` with one curve per layer; the plain P&L
   does not list it.
5. In the SPA, a reinsured `xpnl` session lights Pricing, Plot and the
   panel draws the per-layer curves (verified by the LIB and API tests;
   eyeballed once in the app).
6. Chart availability moves for no existing object: the a368 diff changes
   what the aggregate chart *contains*, never whether it exists.

## Bookkeeping

One bump, one commit, per phase, house format:

```
[Palm-Kappa-Chart] a368: kappa chart serves Palm per-layer curves, bands opt in
[PnL-Kappa-Chart] a369: kappa chart lights for a reinsured P&L through the engine
```

Each carries its code, tests, `pyproject.toml`, `CHANGELOG.md` section,
and `dev/FEATURES.csv` where the member surface moves
(`supports_pgf_prime`). The a368 CHANGELOG entry states the route change
and the band opt-in as breaking for a consumer of the aggregate chart's
band series (provisional tier, `aggregate.charts`); the a369 entry states
the new P&L capability. The plan moves to `dev/done/` with the final LIB
bump. Not pushed. The author pushes.

## Rulings (author, 2026-09-29)

1. **[bands]** Palm is the default on `Aggregate` and `PnL`; `bands=`
   opts back into the joint-derived percentile band; a
   `BivariateAggregate` input keeps the full band chart. (Chosen over
   strictly-by-input-type and over the always-hybrid option.)
2. **[conditioning]** The P&L chart conditions on **gross**, matching the
   aggregate chart, the allocation, and the panel title; the
   net-conditioned reading stays the waterfall's.

## Execution log

### [Palm-Kappa-Chart], 1.0.0a368 (2026-09-29)

Facts re-verified against a367 before touching anything: every named path,
symbol and line range held. Names re-vetted: no collision for
`supports_pgf_prime`, `_kappa_palm`, or a `bands` kwarg anywhere in
`src/aggregate`. Divergences and execution-time decisions, none plan
breaking:

- **[label-helper-hoist]** The plan's preferred route taken: `_SITE_PREFIX`,
  `_layer_descriptor` and `_reins_layer_label` moved from `_pnl_builders.py`
  to `_reinsurance.py` (a leaf both sides may import); `_pnl_builders`
  imports them back at its top import block, call sites unchanged. The
  `decl_writer` import inside `_layer_descriptor` stays function local, so
  no import cycle.
- **[zero-cell-guard]** New-in-execution fix, recorded as a divergence: the
  fallback test build (logarithmic frequency) exposed a latent band-chart
  edge in which the joint's budget grid holds at least the window floor of
  mass in its zero bucket, the `cdf_range` window then includes `g = 0`,
  the share series carries NaN, and `canonical_json` (``allow_nan=False``)
  refuses the document. Both routes now start the drawn window strictly
  above zero (`_kappa_band` drops the `g = 0` row after `_kappa_frame`;
  `_kappa_palm` filters it beside the NaN guard cells). No existing pinned
  doc included the zero cell, so no test moved.
- **[fallback-meta]** Acceptance check 3's "byte-identical to today's
  chart" is read as identical modulo the `route: '2d-fallback'` meta key
  the same plan mandates; the test pins series, axes and panels equal to
  the direct joint chart's.
- **[grid-fallback]** The route switch also falls back to the joint on a
  signed or windowed grid (`i0` / `x_min`), mirroring `palm_kappa`'s own
  refusal, so the emitter cannot raise where availability answered yes.
- **[palm-signature]** `_kappa_palm` takes
  `(agg, levels, cdf_range, ceiling, bands)` rather than the plan's
  sketched `(agg, cdf_range, ceiling)`: the band overlay is built inline
  (one series list, one `ChartDoc`), which is simpler than post-hoc
  document surgery in the route switch.
- **[delegate-tests]** The two pre-existing delegate tests updated for the
  new default: the band-equality test is superseded by the Palm suite plus
  `test_bands_opt_back_in`; the held-joint memo test now exercises the memo
  through `bands=True`, the route that actually builds a joint.
- **[observed-gap]** Cross-route agreement on the two-layer tower: max
  0.20% relative, median 0.017%, over the 353 budget-grid cells clearing a
  5% magnitude floor (recorded in the test docstring; tolerance 5%).
- **[no-todo-entry]** `dev/TODO.md` carries no tracked item for this plan
  (drafted and executed the same day), so there is nothing to tick.

Gate: tier 3 (`uv run pytest -m 'slow or not slow'`) green, 5,435 passed,
142 s. FEATURES audit OK with the new `supports_pgf_prime` row.

### [PnL-Kappa-Chart], 1.0.0a369 (2026-09-29)

As planned, no surprises. Decisions the plan deferred to execution:

- **[title]** The P&L doc accepts the engine's label, exactly what
  `chart_reins` does at a367, so the two tabs title consistently; the
  tower P&L doc is hash-identical to the engine's own.
- **[availability]** The engine look-through in `_kappa_available` is the
  final (aggregate) branch only: a P&L reaches `occ_reins` plus
  `agg_density` on whatever it wraps, so a P&L over a `Portfolio` (no
  occurrence program on the wrapped object) and a stitched ledger with no
  engine both answer False, and the book and joint branches are never
  reached through a look-through.
- **[peel-fixture]** The peel P&L test build appends `peel top-down` to a
  two-layer occurrence tower (the `tests/test_pnl_peel.py` idiom); its
  engine carries the program, so the per-layer curves come through the
  route switch with no peel-specific code.

Gate: tier 3 green, 5,439 passed, 142 s. Plan retired to `dev/done/` with
this bump. No `dev/TODO.md` entry existed to tick (same-day plan). The
[API-Kappa-Residue] phase follows in `V:\dev\aggregate-api` as a separate
commit under that repo's rules.
