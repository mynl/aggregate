# Plan — finish the concern modules (the deferred bodies)

> **Status: APPROVED — execute all three phases.** The closing step of the
> god-module refactor track (`plan-README.md`, H6). P3 1b and P4 §3 **birthed**
> three concern modules — `_reinsurance.py`, `_validation.py`, `_bucket_window.py`
> — but left them as **thin shells**: only the leaf math + narrative + constants
> moved; the per-class **orchestration bodies** stayed on `Aggregate` /
> `Portfolio`. To understand reinsurance / validation / sizing you read the class
> file (the bulk) *plus* a near-empty sidecar — the worst of both worlds. This
> plan moves the bodies in so each concern **lives in one place** (the README
> end-state, and the actual payoff of the split), and **retires the legacy
> `recommend_bucket` / `best_bucket` sizers** (the old W10 item, folded in — the
> new `_bs_window` system is proven and already the only live path).
>
> **Behaviour-frozen.** No numbers change. `uv run pytest` green and the frozen
> baseline (`test_baseline.py`) **unmoved** before every commit — reinsurance
> apply math and validation noise logic are among the most author-sensitive
> numerics in the library, so any drift is a relocation bug, not a feature. The
> legacy-sizer retirement is confirmed off the live path (both classes' update
> route bs==0 through `_bs_window`), so it too is behaviour-neutral.

---

## 0. Cold-start context (read first if starting fresh)

- **Environment.** `uv` for everything, with `UV_LINK_MODE=copy` (the harness
  `.claude/settings.local.json` sets it; in a shell `$env:UV_LINK_MODE='copy'`).
  Files are **CRLF** — for large block deletions use a small Python script
  (`open(path, newline='')` → `splitlines(keepends=True)` → drop the 1-based index
  range → rejoin) to preserve line endings; the `Edit` tool is fine for small
  changes.
- **Baseline.** `uv run pytest -q` currently reports **`3 failed, 1883 passed`**.
  The 3 failures are **pre-existing** in `tests/test_style.py`
  (`test_use_sets_facecolor`, `test_context_scopes_rcparams`,
  `test_context_overrides` — rcParams/facecolor, unrelated to this work). That is
  the green reference: any *new* failure is a regression. (Pass count will shift
  as this plan removes `test_best_bucket_retained_for_comparison` and adds the
  Phase-A/C pin tests.) A full run is ~6 min; use a targeted subset for fast
  feedback, full suite before declaring a phase done.
- **Façade discipline (the pattern every move follows).** `distributions.py` and
  `portfolio.py` are thin re-export façades; the classes live in `_aggregate.py`
  / `_portfolio.py`. The concern modules (`_reinsurance`, `_validation`,
  `_bucket_window`) are **leaves**: a moved body becomes a free function **taking
  the object** (`agg` / `port`); the method becomes a one-line delegator. The
  module must **never import `_aggregate` / `_portfolio`** (no cycle). These are
  method *bodies*, not public names, so the façade `__all__` / re-exports do not
  change — but confirm with `rg` that no test imports the moved private helper by
  its old qualified path.
- **Naming vet (CLAUDE.md).** `rg` each new free-function name against the surface
  before fixing it; object-first signatures (`apply_reins_work(agg, …)`,
  `bs_window(agg, …)`, `validate(port, …)`) avoid colliding with existing
  method/property names. Call the chosen names out for review.
- **The same play, every phase:** cut body → free function taking the object in
  the concern module → method delegates → targeted tests → full suite + baseline
  unmoved → next.

**Recommended order: A (reinsurance) → B (bucket/window + retirement) → C
(validation, most sensitive, last).** The phases are independent; this order does
the biggest payoff first and the most delicate last.

---

## 1. The three thin shells today

| Module | In it now (the leaves) | Still on the class(es) — this plan moves it | Sharing |
|---|---|---|---|
| `_reinsurance.py` (152 ln) | `make_ceder_netter`, `_validate_reins_layers` | `_apply_reins_work`, `apply_occ_reins`, `apply_agg_reins`, `reins_density_df`, `reins_stats_df`, `reins_summary_df`, `_reins_view_stats`, the `_reins_*6` moment helpers, `reins_description`/`_reins_description`, `reins_kinds`, `_reins_describe_block`, the `_describe`/`_describe_signed` reins-label logic (`_reins_after_label`) | **Agg-only** |
| `_validation.py` (~25 ln) | `explain_validation`, `VALIDATION_NOISE`, `ALIASING_RATIO` | `Aggregate.valid`/`validation_explanation`, `Portfolio.valid`/`validation_explanation` | **Agg + Port, side by side** |
| `_bucket_window.py` (~388 ln) | `estimate_agg_window`, `_estimate_agg_percentile`, `bs_describe`/`bs_explain`, 7 constants | `Aggregate._bs_window`, `Portfolio._bs_window`/`best_window`/`bs_window_df`/`_single_big_jump_window`/`_build_bs_window_df` (+ retire the legacy `recommend_bucket`/`best_bucket`) | **Agg + Port, side by side** |

---

## 2. Phase A — reinsurance bodies → `_reinsurance` (Agg-only; biggest payoff)

`_reinsurance.py` 152 → ~750 lines; `_aggregate.py` shrinks by the same. Move, as
free functions taking `agg`, the coherent reinsurance subsystem; methods delegate:

- **Apply engine:** `_apply_reins_work` (`_aggregate.py:3892`, the ceder/netter
  convolution) → `apply_reins_work(agg, reins_list, base_density, debug=False)`,
  with `apply_occ_reins` / `apply_agg_reins` as thin methods calling it on the
  occ / agg densities. (P3 1b explicitly parked `_apply_reins_work` "for now" —
  this is the now.)
- **Reporting frames:** `reins_density_df`, `reins_stats_df`, `reins_summary_df`,
  `_reins_view_stats`, and the moment kernels `_reins_moments6_from_raw` /
  `_reins_exact_image_raw` / `_reins_agg6_from_sev_raw` / `_reins_density6`.
- **Narrative:** `reins_description` / `_reins_description`, `reins_kinds`,
  `_reins_describe_block`, `_reins_after_label`, and the reins-label branches of
  `_describe` / `_describe_signed`. Keep beside `make_ceder_netter` so the whole
  reinsurance story is one file.
- **Plot:** `reins_occ_plot` (`_aggregate.py:1037`) is a matplotlib method —
  route it through `plots/_aggregate` (P2), **not** `_reinsurance`
  (`tests/test_plots_boundary.py` forbids matplotlib in compute modules).
- **Guard (new test):** the reins lines of `test_suite.agg` + the reins
  case-study tests exercise this end-to-end; add a focused test calling
  `apply_reins_work(agg, …)` directly on a small hand-checkable layer program.
- **Watch:** `_apply_reins_work` reads `agg.sev_density` / `agg.agg_density` and
  several spec attrs — pass `agg` and read through it; do not import the class.

---

## 3. Phase B — bucket/window → `_bucket_window`, and retire the legacy sizers

### B0 — retire `recommend_bucket` / `best_bucket` (the folded-in W10)

Confirmed **off the live path** — `Aggregate.update`/`Portfolio.update` route
bs==0 through `_bs_window` (Agg `_aggregate.py:3008`; Port `_portfolio.py:2044`
forwards to `best_window`), never the legacy sizers. So this is behaviour-neutral
(baseline must not move). Enumerated call sites to clean:

- **Delete** `Aggregate.recommend_bucket` (`_aggregate.py:5491`),
  `Portfolio.recommend_bucket` (`_portfolio.py:1628`), `Portfolio.best_bucket`
  (`_portfolio.py:1651`).
- **`aggregate_error_analysis`** (`_aggregate.py:5527`) is `recommend_bucket`'s
  only remaining caller (a diagnostic; no tests, no other callers). Decision:
  either repoint its default-`bs` guess to `estimate_agg_window` /
  `round_bucket` (keep the diagnostic) **or** delete the method too. Lean: repoint
  (cheap, keeps the tool) — flag the choice in the PR.
- **Tests:** remove `tests/test_bucket_sizing.py::test_best_bucket_retained_for_comparison`;
  fix `tests/peg.py:69` (`bs = port.best_bucket(log2)`) to a fixed bs or
  `best_window`. `tests/test_allocation_bounds.py:116` is a comment only — update
  wording if touched.
- **Doc lockstep:** the stale underwriter comment (`underwriter.py:1246`,
  "best_bucket (non-signed)") → `best_window`; the `recommend_bucket` mentions in
  `underwriter.py:1170/1179/1274`, `config.py:123`,
  `data/config.default.toml:60`, and `_bucket_window.py:62` → reword to the
  `_bs_window` / `estimate_agg_window` sizer.
- **Public-API removal** (`recommend_bucket` is a public method) → its own
  `CHANGELOG.md` line.

### B1/B2 — move the surviving sizers in

- **B1 Aggregate:** `Aggregate._bs_window` (`_aggregate.py`) body →
  `bs_window(agg, …)` in `_bucket_window.py`; method delegates. Co-locates the
  orchestrator with its `estimate_agg_window` / `_estimate_agg_percentile` leaves.
- **B2 Portfolio:** `Portfolio._bs_window` (forwarder), `best_window`,
  `bs_window_df`, `_single_big_jump_window`, `_build_bs_window_df` → free
  functions taking `port`; methods/properties delegate.
- **B3:** the two `bs_window` orchestrators now sit beside each other and beside
  the shared estimators — "different approaches, shared module".

---

## 4. Phase C — validation bodies → `_validation` (author-sensitive)

The intricate, `stats_df`-coupled noise logic P3 1b flagged. **Byte-identical.**

- **C1 Aggregate:** `Aggregate.valid` (`_aggregate.py:3571`) body → `validate(agg)`
  (or `valid_aggregate`); `validation_explanation` (`:2631`) → beside
  `explain_validation`. The thresholds it reads (`VALIDATION_NOISE`,
  `ALIASING_RATIO`, the config floors) already live in `_validation` — co-location
  removes the cross-module read.
- **C2 Portfolio:** the same for `Portfolio.valid` (`_portfolio.py:2422`) /
  `validation_explanation` (`:2528`). README: "similar-not-identical,
  co-located" — **do not merge**; place side by side so the differences show.
- **C3 guard (new tests):** pin bit-identical flags/explanations vs a pre-move
  capture on representative builds (clean, aliased, defective-tail, signed).
- **`remove_fuzz`:** lift to a shared helper only if the Agg and Port forms are
  identical (vet vs `remove_fuzz_util` in `utilities.py`); else leave local.

---

## 5. Guardrails

- **Two-file rule (retrospective).** After the dust settles, a routine reinsurance
  / sizing / validation change should land in **one** file. If it still forces
  editing the class *and* the concern module together every time, that seam is
  wrong — re-inline it. (We're committing to all three; this is the check that
  tells us if any one should be reverted.)
- **Leaf discipline.** Concern modules never import `_aggregate` / `_portfolio`;
  they take the object. Watch for cycles when a body references a sibling concern.
- **No parallel paths.** Accessors → `GridDistribution` (P1); single-GD pricing →
  `_pricing` (P3). This plan touches only reinsurance, sizing, validation.
- **Behaviour-frozen.** No numbers change; full suite + baseline unmoved after
  each phase.

---

## 6. Release / housekeeping (CLAUDE.md)

- **Bump `1.0.0a*` + `CHANGELOG.md`** at completion (the relocations are tidy
  moves, but Phase B0 is a breaking public-API removal and A/C add pin tests, so
  the iteration bumps). One section covering: reinsurance now lives in
  `_reinsurance`; sizing/validation bodies in `_bucket_window` / `_validation`;
  `recommend_bucket` / `best_bucket` removed (breaking).
- **Move this plan to `dev/done/`** when all three phases land; update
  `dev/TODO.md` (close the H6 structural item and the W10 entry) and
  `plan-README.md` (the track is then fully closed bar the post-beta/conditional
  4B / 2B / sample-review / Pass B).
- **Do not build docs** in the loop (CLAUDE.md); keep `.rst`/doc references in
  lockstep and note a pending rebuild.
