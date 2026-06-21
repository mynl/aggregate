# Plan P3 — split `distributions.py` (kind-keyed module set)

> **Status: DRAFT — not executed.** See `plan-README.md` for the four-plan map.
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
> **Release mechanics (CLAUDE.md).** Pure-move phases (the kind split) are tidying
> and need no version bump; the compute-extraction phase (new tests, touched
> behaviour) bumps `1.0.0a*` + adds a `CHANGELOG.md` section. `uv run pytest` green
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
| **`_aggregate.py`** | 2309–8253 (`Aggregate`), plus its private support: bucket sizing `estimate_agg_window`/`bs_describe`/`bs_explain`/`_bs_grid_top`/`_estimate_agg_percentile` (552–873), `_picks_work` (1026), reins helpers `_validate_reins_layers`/`make_ceder_netter` (1288–1428), `_flat_col_to_stats_index` (2269) | ~6,300 | The monolith; Phase 2A shrinks it (and P1 already thinned its accessors). Imports `_frequency`, `_severity`, `_fits`, `spectral.Distortion`, and the P1 `GridDistribution`. |
| **`distributions.py`** | — | ~10 | Façade only (§1). |
| **shared tiny helpers** | `value_type_role`/`value_type_label` (111–162), `max_log2` (163) | small | Put in whichever module uses them; if used by >1, a `_dist_common.py` leaf. Decide at execution. |

**Dependency direction is clean.** `spectral` does **not** import `distributions`
(verified one-way), so `_aggregate.py` importing `Distortion` from `spectral`
creates no cycle. The split DAG is
`_fits` → `_severity`; `_frequency`; (`_severity`, `_frequency`, `spectral`,
`GridDistribution`) → `_aggregate` → façade. No new circular risk.

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

After Phase 1: four readable files, nothing over ~6.3k, public surface identical.
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
  **Phase 2A** (Aggregate compute extraction, bumps) follows; **Phase 2B**
  (composition) is post-beta and conditional, gated by the two-file rule.
- File a `dev/TODO.md` entry pointing at this plan and the README.
