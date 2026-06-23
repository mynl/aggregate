# Plan P3 — split `distributions.py` (kind-keyed module set)

> **Status: Phases 1 + 1b + 1c DONE — Phase 2A pending.**
> `distributions.py` is now a thin façade over `_fits` / `_frequency` / `_severity`
> / `_aggregate`, and the shared concerns are born as leaf/near-leaf modules
> `_bucket_window` / `_reinsurance` / `_validation` / `_pricing` (each takes plain
> data or an `agg` object, never imports `_aggregate`/`_portfolio`, so P4 reuses
> them). `explain_validation` was relocated out of `utilities.py` (back-compat
> re-import kept). **Phase 1c (a93, bumps):** distortion-set calibration factored
> onto `Distortion.calibrate_set`; `Aggregate.calibrate_distortions` + `price_ccoc`
> added (Agg/Port parity, verified bit-for-bit vs the 1-unit-Portfolio path); the
> unused singular `Portfolio.calibrate_distortion` dropped. The full suite matches
> the pre-split baseline at each step (registry dispatch made the kind split
> dispatch-safe; function-local imports break the `_fits`/`_severity` → `_aggregate`
> cycles). **Deferred within 1b** (revisit in P4 / 2A): extracting the
> `Aggregate.valid` body and `_apply_reins_work` into their concern modules.
> **Known caveat surfaced in 1c:** `GridDistribution.lev` undercounts on the
> Aggregate's filtered (`p_total>0`) grid, so the Agg calibration sources `el` from
> the full contiguous `density_df['p_total']` instead — worth a look during 2A.
> **Paused before Phase 2A** (Aggregate compute extraction). See `plan-README.md`.
> This is a *structural* refactor: it moves code and adds a façade, with **zero
> behaviour change** and **zero public import-path breakage**. The frozen numeric
> baseline must not move.
>
> **Dependencies.** Sequenced **after P1** (`plan-grid-distribution.md`, accessors
> delegated) and **P2** (`plan-plots-subsystem.md`, plotting lifted out) — so
> `_aggregate` is already smaller and matplotlib-free when moved. Portfolio is a
> separate plan, **P4** (`plan-split-portfolio.md`).
>
> **Why now.** `distributions.py` is ~9,950 lines; `Aggregate` alone is ~5,900
> (lines 2309–8253). The import paths it defines freeze at 1.0. Doing the split
> pre-1.0 behind a re-export façade costs zero compatibility; doing it after is a
> breaking-change conversation. This is the natural window.
>
> **Release mechanics (CLAUDE.md).** Pure-move phases (the kind split, Phase 1b) are
> tidying and need no version bump; the capability phases — **Phase 1c** (distortion
> calibration on a GD) and **Phase 2A** (compute extraction) — touch behaviour / add
> tests, so they bump `1.0.0a*` + add a `CHANGELOG.md` section. `uv run pytest` green
> before every commit. Move this plan to `dev/done/` when its phases land.

---

## -1. Considerations / Warnings (the regret guard)

**Bottom line:** the kind split (Phase 1) and the plots subsystem (P2) cut along
the grain — those you'll be glad of, and the façade kills the re-join cost that
makes normalisation hurt. The genuine over-splitting risk is concentrated in
2A/2B, which is exactly why the plan quarantines them as narrow-and-conditional.
Do Phase 1, write the architecture map, live with it briefly, and let the
**two-file rule** — not dogma — decide how much further to go (if a typical change
routinely forces editing 2+ of the new files together, the seam is wrong, merge it
back). That's the opposite of the path that leads to regret.

## 0. Goals and non-goals

**Goals**
- One class taxonomy per file, split on the existing `Base<Kind>` axis.
- **Birth the cross-cutting shared concerns** — `_validation`, `_bucket_window` (bucket /
  window sizing), and `_pricing` (single-GD: pentagon completion + distortion
  calibration), plus the Agg-only `_reinsurance` — as standalone modules, seeded
  here from the `Aggregate` implementations. P4 then drops the `Portfolio`
  implementations into the *same* `_validation`/`_bucket_window`/`_pricing` files, so the
  two are read **side by side** (they are similar, not identical — co-locating
  surfaces the commonality without forcing a premature merge).
- No file over ~6k lines after Phase 1; the real win (shrinking `Aggregate`) is
  staged separately in Phase 2A.
- Public surface unchanged: `aggregate.Aggregate`, `aggregate.Severity`,
  `aggregate.Frequency`, and `aggregate.distributions.<X>` all keep working.
- Make the FFT/discretization math reachable and testable **without** a full
  `Aggregate.update()` — serves both testing and the "see what's going on"
  intuition-tool goal.

**Non-goals**
- No algorithm changes, no renames of public methods/attributes, no DecL changes,
  no new features.
- **No attempt to reduce import time here** — that is P2's matplotlib defer. This
  split is for human maintainability (see §4).
- Not committing to mixin-vs-composition for `Aggregate` beyond Phase 2A; Phase 2B
  is explicitly deferred and re-decided after 2A lands.

---

## 1. Naming convention decision

**Implementation modules are underscore-prefixed; the public names are exposed
through a façade.** This is the `scipy.stats` pattern (`_continuous_distns.py`,
`_discrete_distns.py`, `_distn_infrastructure.py` re-exported from
`scipy/stats/__init__.py`) and the `scikit-learn` 0.22 pattern (every impl module
`_`-prefixed, public classes re-exported from the package). It is the dominant
convention for "split the implementation, keep one public namespace," and it
reinforces the centralise-the-public-surface goal: the underscore answers "may I
import from here?" with "no — import from the package."

Three layers, no breakage:
- `aggregate.Aggregate` — blessed public path (unchanged; via top-level `__init__`).
- `aggregate.distributions.Aggregate` — still works (façade re-export below).
- `aggregate._aggregate.Aggregate` — implementation; the `_` says "don't."

**`distributions.py` becomes a thin public façade — no logic:**

```python
# distributions.py  — public façade; defines no classes/functions of its own
from ._fits import *        # noqa: F401,F403
from ._frequency import *   # noqa: F401,F403
from ._severity import *    # noqa: F401,F403
from ._aggregate import *   # noqa: F401,F403
```

**The façade localises import ordering.** Today `__init__.py` sequences the package
by hand. After the split, `__init__.py` still imports from `distributions`
unchanged; the façade's internal import order
(`_fits` → `_frequency`/`_severity` → `_aggregate`) is the only place the
intra-`distributions` ordering lives, and it is visible in four lines.

> Pre-1.0 the no-aliases stance would *permit* dropping the `distributions` path
> entirely and exposing only top-level. **Recommendation: keep the façade through
> beta**, revisit at 1.0. A pure re-export façade is one module, not a "second way
> to do things."

---

## 2. Target layout and line-range mapping

Source ranges below are from the current `distributions.py` (will drift as the file
is edited — the `Base<Kind>` class boundaries are the real split lines, not the
numbers).

| New module | Current source | Approx lines | Notes |
|---|---|---|---|
| **`_fits.py`** | 184–385 (`lognorm_fit`, `sln_fit`, `sgamma_fit`, `gamma_fit`, `beta_fit`, `invgamma_fit`, `invgauss_fit`, `lognorm_lev`, `lognorm_approx`, `approximate_from_mcvsk`) + the `_sev_kwargs_*` / `_approximate_sev_kwargs` helpers 386–551 | ~370 | **True leaf** (numpy/scipy only). Consumed by `_severity` and by `Aggregate.approximate`. |
| **`_frequency.py`** | 1429–2268 (`Frequency` + 21 subclasses), plus `_logarithmic_theta` (1219), `_normalize_freq_name` (1275) | ~900 | Near-leaf: `moments` + small numeric helpers only. |
| **`_severity.py`** | 8832–9954 (`Severity` + subclasses, `_DiscreteRV`*), **plus** the severity support functions: `make_conditional_*`/`make_layer_attachment_*` (8254–8505), `_classify_sev`/`_cv_to_shape`/`_mean_to_scale`/`_safe_integrate`/`_numerical_moms` (8506–8830), `_partial_e*`/`_moms_analytic` (874–1025), `validate_discrete_distribution` (1231) | ~1,900 | `Severity(ss.rv_continuous)` keeps its scipy coupling. Imports `_fits`. *`_DiscreteRV` may have moved to the grid-distribution module in P1 — check. |
| **`_aggregate.py`** | 2309–8253 (`Aggregate`) minus the shared-concern blocks below, plus its genuinely Agg-private support: `_picks_work` (1026), `_flat_col_to_stats_index` (2269) | ~5,200 | The monolith; Phase 2A shrinks it further (and P1 already thinned its accessors). Imports `_frequency`, `_severity`, `_fits`, `_validation`, `_bucket_window`, `_pricing`, `_reinsurance`, `spectral.Distortion`, and the P1 `GridDistribution`. |
| **`_bucket_window.py`** *(shared)* | bucket / window sizing `estimate_agg_window`/`bs_describe`/`bs_explain`/`_bs_grid_top`/`_estimate_agg_percentile` (552–873) | ~320 | Bucket-and-window-selection concern. Seeded from `Aggregate`; P4 adds the `Portfolio` sizers (`recommend_bucket`/`best_bucket`/`best_window`/`bs_window_df`) here so the two sit side by side. Near-leaf (numpy/scipy). |
| **`_reinsurance.py`** *(Agg-only)* | reins helpers `_validate_reins_layers`/`make_ceder_netter` (1288–1428) + the Aggregate ceder/netter application | ~200 | Agg-only carve-out (Portfolio has no reinsurance). The 2B `ReinsuranceProgram` collaborator, if it ever lands, grows from here. |
| **`_validation.py`** *(shared)* | the per-class `valid` checks + aliasing/moment-match machinery (exact symbols resolved by `rg` at execution) | ~250 | Validation concern. **The shared core already exists** — `explain_validation` is a free function in `utilities.py:669` (in `utilities.__all__`), already called by *both* classes' `validation_explanation`. So `_validation.py` **pulls `explain_validation` out of `utilities`** (a public-surface move, exactly like `make_var_tvar` in P1) and houses the per-class `valid` checks beside it; the classes keep `validation_explanation` as thin delegators. P4 adds `Portfolio.valid` here. Similar-not-identical — co-located to expose the commonality. |
| **`_pricing.py`** *(shared)* | the single-GD pricing surface: `Aggregate.price`, `price_ccoc`, pentagon completion, and the flavor-(b) **pentagon→target glue** + new `Aggregate.calibrate_distortions` (Phase 1c) | ~300 | Single-GD pricing concern (pentagon completion + distortion calibration). `price_ccoc` is genuinely single-GD — it acts on **one** GD (a `Portfolio` total *or* an `Aggregate` unit) — but **today exists only on `Portfolio` (`portfolio.py:4508`)**; relocating it here is what gives `Aggregate` the same entry. Thin orchestration — pentagon algebra stays in `pentagon.py`; the **distortion-set loop is `Distortion.calibrate_set` in `spectral.py`** (Phase 1c — `_pricing` only resolves the pentagon target and calls it, no set-loop lives here); per-distortion Newton stays on `Distortion`; **result dataclasses (`PricingResult`/`AnalyzeDistortion*Result`) stay in `results.py`** (reuse, don't reinvent). P4 routes `Portfolio`-total pricing through it. See `plan-README.md` pricing↔allocation boundary. |
| **`distributions.py`** | — | ~10 | Façade only (§1). |
| **shared tiny helpers** | `value_type_role`/`value_type_label` (111–162), `max_log2` (163) | small | Put in whichever module uses them; if used by >1, a `_dist_common.py` leaf. Decide at execution. |

**Prior extractions already exist — the line ranges above are net of them.**
`moments.py` (`MomentAggregator`/`MomentWrangler` + the `xsden_*` helpers),
`tail.py` (`TailClass*`), and `results.py` (`PricingResult` /
`AnalyzeDistortion*Result`) were split out of the god files earlier and are imported
by both. `_severity`/`_aggregate` keep importing them unchanged, and `_pricing`
**reuses `results.py`'s dataclasses** rather than minting new ones. Do not
re-extract or duplicate these.

**Dependency direction is clean.** `spectral` does **not** import `distributions`
(verified one-way), so `_aggregate.py` importing `Distortion` from `spectral`
creates no cycle. The split DAG is
`_fits` → `_severity`; `_frequency`; `_validation`, `_bucket_window`, `_reinsurance`
(near-leaves: numpy/scipy + `GridDistribution`); `_pricing` → (`spectral`,
`pentagon`, `GridDistribution`); then
(`_severity`, `_frequency`, `_validation`, `_bucket_window`, `_reinsurance`, `_pricing`,
`spectral`, `GridDistribution`) → `_aggregate` → façade. The shared concern modules
must **not** import `_aggregate`/`_portfolio` (they take a GD and plain data), which
is exactly what lets P4 reuse them — same leaf discipline as `GridDistribution`. No
new circular risk.

Split axis = the `Base<Kind>` convention: all `Frequency*` together, all
`Severity*` together, so the alphabetical-sort rationale carries to file level.

---

## 3. Phased execution (commit per pure move)

Each numbered step is a standalone commit that **only relocates code** and leaves
`uv run pytest` green. No "improve while moving."

### Phase 1 — kind split (mechanical, low risk; pure move, no bump)

- **1.1** Extract `_fits.py` (leaf first). Update `_severity`/`Aggregate`
  references via the façade or direct import.
- **1.2** Extract `_frequency.py`.
- **1.3** Extract `_severity.py` (class block + its support functions).
- **1.4** Extract `_aggregate.py` (remaining class + its support functions); reduce
  `distributions.py` to the façade.
- **1.5** Add `__all__` to each new module; façade re-exports the union. Confirm
  `aggregate.Aggregate`, `aggregate.distributions.Aggregate`, and a representative
  `from aggregate.distributions import Severity` all resolve.

After Phase 1: four readable files, nothing over ~5.2k, public surface identical.
**Shippable unit.** Phase 1b can follow later.

### Phase 1b — birth the shared concerns (pure move, no bump)

Lift the cross-cutting blocks out of `_aggregate.py` into their own modules,
**seeded from the `Aggregate` code only** (P4 adds the `Portfolio` side later). Each
is a standalone commit, `uv run pytest` green, no logic change:

- **1b.1** `_bucket_window.py` — the bucket / window sizing helpers. The
  `bs_describe`/`bs_explain` *worker functions* it relocates (`distributions.py:743`
  / `779`) are the subject of open TODO **B5** (verb workers shadowing the
  `bs_description`/`bs_explanation` noun properties). The public properties are
  already correctly named; this step is a **pure relocation of the workers only** —
  do not entrench them, and cross-reference B5 so the eventual rename knows their
  new home.
- **1b.2** `_reinsurance.py` — the ceder/netter helpers + Aggregate application
  (Agg-only; no Portfolio counterpart will join it).
- **1b.3** `_validation.py` — the `explain_validation` / aliasing-and-moment-match
  machinery (`rg` the exact symbols at execution).
- **1b.4** `_pricing.py` — birth the single-GD pricing surface by relocating
  `Aggregate`'s **existing** single-GD pricing methods (`price`, `price_pentagon`,
  pentagon completion) as the seed. This step is a **pure move** — the distortion-
  calibration capability (`Distortion.calibrate_set`, the pentagon→target glue,
  `price_ccoc` on `Aggregate`, `Aggregate.calibrate_distortions`) is **not** here; it
  bumps and lands in **Phase 1c** below.

These modules are leaves/near-leaves (they take a GD + plain data, never import
`_aggregate`), which is what lets **P4 consume them unchanged**. After 1b:
`_aggregate.py` is ~5.2k and the shared concerns are visible in one place each.
**Shippable unit.** Phase 1c can follow later.

### Phase 1c — distortion calibration on a GD (the former P1 Phase L; bumps)

A **capability** phase, not a pure move: it drops a public method, adds new ones,
and re-sources calibration data — so it bumps `1.0.0a*` + adds a `CHANGELOG.md`
section. It depends on `_pricing.py` (1b.4) and the P1 GD accessors (`sf`/`lev`).
**Goal: calibrate a distortion set on an `Aggregate` as well as a `Portfolio`**
(today only `Portfolio` can).

**The finding.** `Portfolio.calibrate_distortions` (plural, `portfolio.py:3582`) is
the only public entry the author uses; it computes assets `a` and premium target `P`
(from `coc`), then loops `['ccoc','ph','wang','dual','tvar']` calling the **singular**
`calibrate_distortion` (`portfolio.py:3473`) once per name — and the singular
*re-resolves the whole S-vector / `ess_sup` / `el` from `density_df` on every call*
even though `a` and `P` are already fixed. The per-name dispatch
(`portfolio.py:3562–3577`: read `subclass._calibration_init_shape` / `param_name`,
build the `Distortion`, call `subclass.calibrate`) reads **only `Distortion` class
state** plus `(S, bs, premium_target, ess_sup)` — i.e. GD data plus one pricing
scalar. It contains no Portfolio knowledge.

**The change.**
- **`Distortion.calibrate_set(...)` — the family loop's permanent home.** Factor the
  per-name dispatch out of `Portfolio.calibrate_distortion` into a `Distortion`
  classmethod, beside the singular `calibrate` (`spectral.py:1866`+) and
  `available_distortions` — it is pure `Distortion` knowledge (the family registry,
  `_calibration_init_shape`, `param_name`). It takes GD-derived data
  (`S`/`lev`/`bs`/`ess_sup`) + `(premium_target, assets)` + the name list and returns
  the calibrated set. **It stays on `Distortion` permanently** — no
  author-in-`spectral`-then-move-to-`_pricing` relay (the double-move the old P1
  Phase L carried is gone). **GD → Distortion**: the caller hands the GD's data in;
  `GridDistribution` never imports `Distortion`. (`rg`-vet the name `calibrate_set`
  against `Distortion`'s surface before fixing.)
- **`_pricing` flavor (b) = the thin pentagon glue.** `_pricing` resolves
  `(premium_target P, assets a)` from pentagon state and calls
  `Distortion.calibrate_set` with the caller's GD. That glue is all of calibration
  that lives in `_pricing`; the loop itself is on `Distortion`.
- **Drop the public singular `Portfolio.calibrate_distortion`.** The author never
  calls it; reimplement the plural over `Distortion.calibrate_set`, resolving the GD
  data **once** before the loop. The singular's unused `S_column`/`S_calc`/`r0`/`kind`
  options (the plural always used defaults) are dropped. *(Public removal —
  `CHANGELOG.md` line.)*
- **Bring `price_ccoc` to `Aggregate`.** Move `price_ccoc` (today Portfolio-only,
  `portfolio.py:4508`) into `_pricing` as a single-GD method so an `Aggregate` gets
  it too.
- **Add `Aggregate.calibrate_distortions`.** Because the glue needs only a GD + a
  premium target, an `Aggregate` (or a unit, or a distorted density) calibrates
  directly — **no more 1-unit-Portfolio wrap.** This is the headline: `Aggregate` /
  `Portfolio` parity. (Calibration to the **total** only; per-unit allocation stays a
  Portfolio concern in `_portfolio_common`.)

**Tests.** `tests/test_distortion_calibrate.py` currently targets the singular —
re-point it at the plural / `Distortion.calibrate_set`, pinning identical calibrated
shapes (the Newton iterations are unchanged; only their inputs are sourced
differently). Add an `Aggregate.calibrate_distortions` case asserting it matches the
old 1-unit-Portfolio path.

**Shippable unit.** Phase 2A can follow later.

### Phase 2A — extract pure functions out of `Aggregate` (high value, low risk; bumps)

Pull methods that take `self` only to read arrays into module-level functions. Two
payoffs: the math becomes **directly testable against brute force** (matches the
`np.convolve` cross-check idiom in `test_numerics2_objective.py`, which today can
only reach the kernel through a full `update()`), and a free function
`freq_sev_convolution(sev_pmf, pgf, ...)` is notebook-inspectable in a way a buried
method is not.

- **2A.1** `_aggregate_compute.py` — discretization math, the FFT convolution core,
  validation/aliasing calculations, as pure functions. `Aggregate` keeps thin
  methods that call them.
- **2A.2** Add focused unit tests calling the extracted kernels directly with small
  hand-checkable inputs (augments, does not replace, the end-to-end golden tests).

> Plotting extraction is **not** here — it is library-wide; see **P2**
> (`plan-plots-subsystem.md`). Distribution accessors are **not** here — they are
> delegated to `GridDistribution` in **P1** (`plan-grid-distribution.md`).

### Phase 2B — composition for genuine collaborators (DEFERRED; re-decide after 2A)

Only where a real stateful collaborator exists. The one strong candidate is a
**`ReinsuranceProgram`** object (it owns the layer list, the ceder/netter
functions, and the gross/ceded/net views — cohesive state, not just methods).
`Aggregate` would hold one and delegate. The remaining stateful methods (update
orchestration, pricing, reporting) stay as thin methods on a now-much-smaller
class.

**Explicitly not doing:** mixin-splitting for its own sake. Mixins relocate lines
without reducing coupling and `self`-sharing across mixin files can read worse than
one sectioned file. Reach for them only if, after 2A, file size itself is still the
pain. Decide at that point, not now.

---

## 4. Why this split is *not* for import time

The module split is for **human** maintainability, not import speed. Lazy-loading
your own pure-Python code (the kind subclasses, reinsurance, pricing) buys nothing
— it binds at class-definition time and you have already paid for it once
`import aggregate` runs. The one real import-time win is deferring **matplotlib**,
and that is **P2's** job (`plan-plots-subsystem.md` §4), where centralising every
matplotlib use into one `plots` subsystem makes the defer trivial and global. Do
not conflate the two: this plan moves code between files; P2 changes when a heavy
dependency loads.

---

## 5. Guardrails and risks

- **Safety net.** The golden-baseline (`test_baseline.py`) + brute-force
  convolution (`test_numerics2_objective.py`) suite is what makes this a mechanical
  refactor. Run `uv run pytest` green after **every** step; the numeric baseline
  must not regenerate.
- **One concern per commit.** Relocation commits contain no logic edits. Any genuine
  cleanup spotted mid-move is noted, not done inline.
- **`__all__` hygiene.** Each new module owns its `__all__`; the façade re-exports
  the union; top-level `__init__.py` is untouched (it imports from `distributions`,
  which now resolves through the façade).
- **Import-cycle watch.** Confirm at each step that `_aggregate` → `spectral` (and
  `_aggregate` → `GridDistribution`) stay one-way; no new back-edge into
  `distributions`/`_aggregate`.
- **Risk: circular import via the façade re-export ordering.** Mitigated by the fixed
  façade order in §1 and by extracting leaves first (`_fits`).
- **Risk: `from aggregate.distributions import _privatehelper`.** If any *internal*
  code reaches a now-moved private helper by its old `distributions._x` path, grep
  and fix at the moving step. (External users importing privates is unsupported;
  internal ones are caught by the test run.)

---

## 6. TODO / sequencing

- This is **P3** in the four-plan track (`plan-README.md`): P1 GridDistribution →
  P2 plots subsystem → **P3 split distributions** → P4 split portfolio.
- Within P3: **Phase 1** (kind split, pure move) is independently shippable;
  **Phase 1b** (birth the shared concerns `_validation`/`_bucket_window`/`_pricing` +
  Agg-only `_reinsurance`, pure move) follows and is what P4 consumes; **Phase 1c**
  (distortion calibration on a GD — `Distortion.calibrate_set` +
  `Aggregate.calibrate_distortions` for Agg/Port parity; the former P1 Phase L;
  bumps) next; **Phase 2A** (Aggregate compute extraction, bumps) next; **Phase 2B**
  (composition) is post-beta and conditional, gated by the two-file rule.
- File a `dev/TODO.md` entry pointing at this plan and the README.
