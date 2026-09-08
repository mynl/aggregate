# plan-pk-tab, LIB execution notes

Execution record for the LIB half of `dev/plan-pk-tab.md` (the Pricing >
Pr Ruin pill). Started 2026-09-05 at `1.0.0a338`. The plan's canonical copy
is in the API repo; the LIB copy is a plain identical file by ruling. This
file records the review findings, the author's execution rulings, and every
divergence, phase by phase.

## Review findings (2026-09-05, at a338)

- All plan facts verified against the code: `pollaczeck_khinchine`,
  `wiener_hopf`, `RuinFunction` and `_ruin_find_u` in `_aggregate.py`,
  `ruin_cepstral` in `_renewal.py`, `pedagogy.ruin_example`, the AD ruin
  section of `decl-testers.agg`, the chart registry with predicates and
  `**options`, and the `charts/_payload.py` lattice machinery.
- `plan-payload` exists in neither repo yet; this plan's downsampling rules
  are specified inline, so nothing blocks on it.
- Name vetting: `ruin` collides with nothing in the exhibit registry, the
  chart registry, or on `Aggregate`; `_ruin_paths` and `eventual_ruin` are
  free.
- One plan claim the code contradicts, resolved by ruling below: the
  exhibit route carries no options (`build_exhibit(obj, name, perspective,
  *, max_rows)`), so an exhibit "on Aggregate taking form inputs" cannot
  work as written. The form-input pattern for exhibits is pricing's result
  object.

## Execution rulings (author, 2026-09-05, via question batch)

1. **Serving mechanics**: a new `RuinResult` object in `results.py`
   returned by a new `Aggregate` method; the `ruin` exhibit registers on
   `RuinResult`, the pricing pattern. The `ruin` chart stays on
   `Aggregate` with `**options` so the generic chart route needs nothing
   new app side.
2. **Method name**: `Aggregate.eventual_ruin(...)`.
3. **Bump grouping**: separate bumps, one per phase: a339 [Ruin-Engine],
   a340 [Ruin-Exhibit], a341 [Ruin-Chart] with [Ruin-Downsampling] inside.

## Phase [Ruin-Engine], a339

What landed: the simulation core of `pedagogy.ruin_example` lifted into
`Aggregate._ruin_paths`, returning the `_RuinPaths` named tuple (exact
solver dispatch, moments, the full simulated check, the drawable paths,
trend and LIL funnel arrays). `ruin_example` is now a thin matplotlib
consumer. New test `test_ruin_paths_deterministic`.

Divergences:

- **Private method, not a module-level helper.** The plan suggested
  `_ruin_paths(...)` in `_renewal.py` or `_aggregate.py`; it landed as a
  private *method* on `Aggregate` in `_aggregate.py`, directly after
  `wiener_hopf`, which is what "beside the solvers" means literally and
  reads naturally (`self` replaces the `agg` argument).
- **Seed semantics.** `_RUIN_SEED = 20260905` module constant; the
  helper's default. `seed=None` (the Sample action) materializes a fresh
  integer seed and reports it in the returned `seed` field, per ruling 2.
  `pedagogy.ruin_example` keeps its public `seed=None` default, passed
  through, so the docs figure's behavior is unchanged (random unless
  seeded), while the served documents will default deterministic.
- **Dispatch duplicated, deliberately.** The helper carries its own
  eight-line frequency dispatch rather than calling
  `pedagogy._ruin_function`, because the core cannot import `pedagogy`
  (import direction). The raise messages match `_ruin_function`'s so the
  tests cover both.
- **The rug's ruined mask** is re-derived in `pedagogy` from finite
  `ruin_time` (NaN marks survivors) rather than shipping the boolean
  array in `_RuinPaths`; a `np.where` guard keeps NaN out of the
  comparison.
- Path records resolve the window question in the helper: `ruined_at` is
  the ruin index within the plot horizon, 0 for a window survivor, so both
  consumers draw from the same reading.

Gate: tier 3 (`uv run pytest -m 'slow or not slow'`), 5104 passed, 1
failed: `test_library_entries.py::test_split_limit_policy_prices_the_per_accident_limit`,
verified pre-existing by stashing this phase's three edited files and
re-running (fails identically at the a338 baseline; `library.agg` and
`test_library_entries.py` are the author's in-flight modifications).
`tests/test_ruin.py` 15 passed including the new determinism case.

## Phase [Ruin-Exhibit], a340

What landed: `RuinResult` in `results.py`; `Aggregate.eventual_ruin(rho
| lr, p | u, log2, n_sims=1000, seed)` returning it; the `ruin` exhibit
registered on `RuinResult` as a manifest passthrough of `ruin_df`;
`_lundberg_exponent` (module level, beside `_ruin_find_u`);
`_ruin_paths` gains `p=` as the alternative to `u0`; a `dev/FEATURES.csv`
row and the `a340` table-version stamp; four new tests in `test_ruin.py`.

Divergences:

- **The frequency predicate moved into the method.** The plan gated the
  exhibit on `Aggregate` with "updated and frequency in {poisson,
  renewal}". Keyed on the result, the gate is `eventual_ruin` itself
  raising on any other frequency (and on a stale grid, through the
  solvers); `available_exhibits` on a `RuinResult` always lists `ruin`,
  matching the pricing exhibits' mechanics. The app-side `can_ruin`
  capability flag still derives from the frequency, unchanged.
- **`p` resolves inside `_ruin_paths`**, not in the caller, so the ruin
  function is solved once rather than twice; `find_u` uses the `'index'`
  kind, which is what "the achieved psi at the grid point" asks for.
- **Lundberg "if cheap"** is a `brentq` root of the discretized
  adjustment equation under a `700 / max(x)` overflow guard; when the
  bracket fails (a heavy tail on a wide grid) the two rows are simply
  omitted, and the renewal path never carries them.
- **The strip omits the plot horizons** (`t_plot` is a figure fact, not a
  reading); it carries the simulation horizon in claims.
- The exhibit joins no snapshot corpus; its coverage is direct assertions
  in `test_ruin.py`.

Gate: tier 3, 5108 passed, 1 failed: the same pre-existing
`test_library_entries` failure as a339, the author's in-flight
`library.agg` work. `test_ruin.py` 19 passed; `test_exhibits.py`,
`test_exhibit_formats.py`, `test_pricing_results.py` 290 passed.

## Phase [Ruin-Chart] with [Ruin-Downsampling], a341

What landed: `charts/_emit_ruin.py` registering chart `ruin` on
`Aggregate` (predicate: updated and frequency poisson or renewal), the
`_resolve_margin` helper shared with `eventual_ruin`, four chart tests,
and the `AD.Ruin.Negbin` tester line.

Divergences:

- **Bare-call defaults `rho=0.2, p=0.05`.** The plan lists the options
  but no defaults; the registry's own contract ("`available_charts`
  answers what *can* be drawn") and two existing sweeps
  (`test_load_chart_doc_round_trips_every_emitted_document`,
  `test_chart_marks`) call every available chart with no options, so a
  chart that raises bare would break the contract and both tests.
- **No `Mark`s.** `test_no_emitter_marks_a_percentile` constrains mark
  roles to `mean` / `break_even` across every emitted document. The
  resolved `(u, psi(u))` reading rides as the one-point `marker` series
  plus the `meta` scalars; pedagogy's faint guide lines are renderer
  sugar, not document meaning.
- **`detail` defaults to 192, not the plan's "about 256 to 1024".**
  Measured: 256 points per path puts the bare poisson document at 204 kB
  canonical JSON, over the plan's own 200 kB acceptance; 192 lands both
  fixture books at about 155 kB. The two plan numbers conflict and the
  acceptance won.
- **Path coordinates are rounded to six significant figures** (the
  psi(u) law is not), roughly halving byte weight at far below drawing
  resolution. The plan is silent on rounding; recorded here because
  `charts/_payload.py`'s philosophy is "nothing rounds", and sample
  paths are illustrative draws rather than the law.
- **`n_sims` is pinned at the module constant `RUIN_SIMS = 1000`**, not
  exposed as an option, reading ruling 4 as a serving cap.
- The trend is a two-point line (it is straight); the LIL funnel is one
  band series (`y` / `y2`) at 64 samples; the rug thins to at most 256
  ticks at ``y = 0``. Roles `sample` / `mean` / `band` / `survival` /
  `marker` / `rug` use the IR's open vocabulary.

Gate: tier 3 combined with the RuntimeWarning gate
(`uv run pytest -m 'slow or not slow' -W error::RuntimeWarning`), this
being a numerics-touching phase: 5112 passed, 2 failed, both the
author's in-flight `library.agg` work and both verified at baseline with
this phase's edits stashed. One is the same split-limit pricing failure
as a339/a340; the other,
`test_every_library_entry_builds[agg:RenewalDeterministicWait]`, is a
`RuntimeWarning: invalid value encountered in sqrt` from
`moments.py` `mcvsk` (a slightly negative numerical variance) that only
the `-W error::RuntimeWarning` gate surfaces, worth the author's
attention against the `[RuntimeWarning-Census]` policy.
`tests/test_ruin.py` 23 passed; the two registry sweeps
(`test_charts_ir.py`, `test_chart_marks.py`) and the DecL corpus
(`test_decl_parser.py`, `test_decl_unparser.py`, 984 cases) pass with
the new chart and tester line.

## LIB half status

COMPLETE at `a341`, three bumps `a339` to `a341`, one per phase as
ruled. The plan and this notes file stay in `dev/` until the API half
([Ruin-Route], [Ruin-Pane], [Ruin-Sample]) lands, the
plan-pricing-natural-allocation precedent for a cross-repo plan whose
LIB half finishes first. Owed to the API side: nothing beyond the plan
itself; the chart route needs no reader change (`load_chart_doc`
round-trips the ruin document, covered by the registry sweep).
