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
