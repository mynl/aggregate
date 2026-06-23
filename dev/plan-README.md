# Refactor planning — README / reorientation

> **Read this first when you come back.** It is the map for a set of structural
> refactor plans drafted June 2026. None are executed yet. The detail lives in the
> four `plan-*.md` files below; this page is just the bird's-eye view and the
> *why*, so you can decide with fresh eyes whether it is still a good idea before
> any code moves.

---

## The core bet (re-evaluate this honestly)

Two modules are god modules: `distributions.py` (~9,950 lines; `Aggregate` alone
~5,900) and `portfolio.py` (~4,965; `Portfolio` ~4,600). The bet is that splitting
them — **behind re-export façades, with zero behaviour change and zero public
import-path breakage** — pays for itself in maintainability, contributor
onboarding, and localised testing, and that **pre-1.0 is the only cheap window**
because the import paths freeze at 1.0.

**The honest risk** (see `plan-split-distributions.md` §-1, the normalisation /
"PITA to re-join" worry): over-splitting can scatter the code so the bird's-eye
view is lost. The mitigations baked in: (1) a façade so the *public* view never
fragments — `from aggregate import X` is unchanged; (2) split **along the grain**
(whole classes, plotting, the distribution primitive) and quarantine the
across-the-grain work (composition) as deferred/conditional; (3) a written
architecture map; (4) the **two-file rule** — if a typical change routinely forces
editing 2+ of the new files together, the seam is wrong, merge it back.

If, reading fresh, the façade + grain-aligned split still feels right, proceed
with P1. If it feels like premature normalisation, the cheapest thing that still
captures most of the value is **P1 alone** (GridDistribution) — it is a pure
addition, not a split, and stands on its own.

---

## End-state architecture (the target both classes converge to)

Designing `Aggregate` and `Portfolio` *together* (not splitting each god module in
isolation) makes the real shape visible: both reduce to a **spec → theoretic
moments → update → a held `GridDistribution`**, plus **thin delegations to a small
tier of shared-concern modules**. `Portfolio` is largely an **orchestrator that
combines its unit `Aggregate`s**; its only genuinely own compute is the
independent-sum combine + `add_exa` allocation.

**Lifecycle (both classes), in order — validation runs *after* update:**

| # | Stage | Aggregate | Portfolio | Sharing |
|---|---|---|---|---|
| 1 | Construct (spec → components) | yes | yes | parallel |
| 2 | Pre-update **theoretic** moments → `stats_df` | freq×sev | Σ units (shares a method-of-moments; else combines Aggs) | parallel |
| 3 | Bucket / window sizing | yes | yes | **shared functions, different approaches** → `_bucket_window` |
| 4 | **Update** → `density_df` | freq×sev FFT | combine the unit Aggs | parallel (own `_compute` each) |
| 5 | **Validation** | yes | yes | **shared** → `_validation` |
| 6 | Revisit `stats_df` → **empirical** columns | yes | yes | **shared** → `GridDistribution` |
| 7 | Distribution view (`q/tvar/cdf/sf/lev/snap`) | yes | yes | **shared** → `GridDistribution` (P1) |
| 8 | **Reinsurance** | yes | — | **Agg-only** carve-out → `_reinsurance` |
| 9 | **Pricing** (calibrate · apply distortion · pentagon) | on its GD | on the **total** GD | **shared** → `_pricing` (single-GD: pentagon completion + distortion calibration) |
| 10 | **Allocation** (linear / lifted, given a distortion on the total) | — | yes | **Port-only** → `_portfolio_common` (exeqa-based) |
| 11 | Reporting / narrative | yes | yes | parallel (shared `_description`/`_explanation` + pprogram helpers) |
| 12 | Sampling / dependence | minimal | yes | **Port-only** → `_portfolio_sample` (extract now, review later) |
| 13 | Plotting | yes | yes | **shared** → `plots/` (P2) |

**The pricing ↔ allocation boundary (replaces the vague "generic pricing"):**
- **Pricing** = everything you can do to **one** distribution — an `Aggregate`, or
  a `Portfolio` *total* (units are `Aggregate`s). It is a **shared concern module,
  `_pricing.py`** (born in P3, consumed by P4) — co-located so the Aggregate and
  Portfolio-total pricing paths are read side by side. It works on a single GD and
  has **two flavors**:
	- **(a) pentagon completion** — GD + cost of capital + p-level (or other
	  pentagon variables) → the completed `Pentagon`, including premium. *(Today we
	  require `a` or `p` to be supplied; relaxing that is a future enhancement, not a
	  now-step.)*
	- **(b) distortion calibration** — GD + pentagon pricing targets → a calibrated
	  `Distortion` (set). **GD → Distortion** (the *caller* hands its GD to the
	  `Distortion`), so `GridDistribution` stays a numpy/pandas **leaf** and never
	  imports `Distortion`.
  `_pricing.py` is **thin orchestration only**: the pentagon algebra stays on
  `Pentagon` (`pentagon.py`), the Newton calibration stays on `Distortion`
  (`spectral.py`); `_pricing` just drives the two flavors over a GD. Both
  `Aggregate` and `Portfolio`-total price through it.
- **Allocation** = splitting the total's distorted price across **units** (linear α
  / lifted β). **Portfolio-only**; lives in the Portfolio **common-numerics** module
  (`_portfolio_common`, exeqa-based) — *the distinguishing feature of a portfolio*,
  shared by the FFT and sample construction paths but **not** shared with
  `Aggregate`, and **not** part of `_pricing`.

**GD is the common currency threading the whole pipeline.** The discretised
**Severity → a GD before the Agg update**; the **Agg FFT** (sev-GD raised to the
freq PGF) → the **Agg's GD**; the **Portfolio independent-sum FFT** over the unit
Agg-GDs → the **Port's GD**. (And **Bivariate** reads already-updated Aggs' GDs to
size its bucket/window — the `balanced_window`/`focus` "measure the support" path.)
Each stage's output GD is the next stage's input — which is why P1 is the keystone.

**Shared tier (Aggregate + Portfolio):** `GridDistribution`, `_validation`,
`_bucket_window`, `_pricing` (single-GD: pentagon completion + distortion calibration),
`plots/`. **Agg-only:** `_reinsurance`. **Port-only:** the three-subsystem split —
`_portfolio_density` (works with densities — independent-sum FFT combine), `_portfolio_sample` (works with samples — sample /
switcheroo / dependence), and `_portfolio_common` (the exeqa-based numerics —
augmented df, apply distortion, **allocation** — shared by both construction paths).
**Parallel-not-shared (resist a common base class):** construct, theoretic moments,
the update math. **Removed:** `pollaczeck_khinchine` → `pedagogy.py` (never used in
anger; not a distortion).

This end-state is delivered **through the existing four plans** — P3 *births* the
shared concerns (`_validation`, `_bucket_window`, `_pricing`, plus the Agg-only
`_reinsurance`) as it splits `distributions.py`; P4 *consumes* them as it splits
`portfolio.py` into its three subsystems (`_portfolio_density` / `_portfolio_sample` /
`_portfolio_common`) and keeps allocation in the common-numerics tier. No separate
"shared-concerns" plan; the README is the map and P3/P4 are kept in sync with it.

---

## The four plans, in dependency order

The order is **strict bottom-up**: build the shared primitive, lift the
cross-cutting concern out, *then* reorganise the (now smaller) modules. Each layer
shrinks the next.

| # | Plan | One line | Risk | Independently shippable? |
|---|---|---|---|---|
| **P1** | `plan-grid-distribution.md` | New spacing-agnostic `GridDistribution` value type owns the marginal-vector accessors (`q/var/tvar/cdf/sf/pmf/lev/…`); the `make_var_tvar` kernel **moves into it** from `utilities.py`; `_DiscreteRV` becomes an adapter over it; it feeds `Distortion` calibration (GD→Distortion); every consumer adopts it | low (pure add + guarded swaps) | yes |
| **P2** | `plan-plots-subsystem.md` | One `plots/` package, per-class modules, `_style.py`, **global matplotlib defer** (the real import-time win) | low (mechanical moves) | yes |
| **P3** | `plan-split-distributions.md` | Kind split → `_fits`/`_frequency`/`_severity`/`_aggregate` + façade; **births the shared concerns** (`_validation`, `_bucket_window`, `_pricing`, and Agg-only `_reinsurance`); Aggregate pure-compute extraction; `pollaczeck_khinchine`→`pedagogy` | low→med | yes (Phase 1 alone) |
| **P4** | `plan-split-portfolio.md` | `portfolio.py` façade + **three-subsystem split** of the one `Portfolio` class — `_portfolio_density` (independent-sum combine), `_portfolio_sample` (sample/switcheroo/dependence; extracted now, reviewed later), `_portfolio_common` (exeqa-based augmented-df / distortion / allocation); **consumes** the P3 shared concerns (`_validation`/`_bucket_window`/`_pricing`) | low→med | yes |

**Deferred bucket** (TODO entries, *not* plans yet — promote only when the seam is
proven by the two-file rule, post-beta):
- Aggregate **2B** `ReinsuranceProgram` collaborator.
- **Sample subsystem review.** P4 *extracts* `_portfolio_sample` structurally
  (behaviour-frozen, tests green); the substantive review/improvement of the
  sample / switcheroo / dependence machinery — the answer to "what about portfolios
  with correlation" — is its own later plan. Coordinate with the separate
  shared-mixing-across-units design; do not pre-empt it.
- **Relaxing pentagon completion** to no longer require `a` or `p` (future pricing
  enhancement; `_pricing` flavor (a)).
- **Bounds** structural class-split (already isolated; low ROI).
- **Bivariate** structural split (newer god class; cost/benefit still open).

```
P1 GridDistribution ──┐ (shrinks Agg, Port, Severity, Bounds, Bivariate)
                      ├─► P3 split distributions ─► P4 split portfolio
P2 plots subsystem  ──┘    (births shared concerns:  (consumes them; splits into
                            _validation / _pricing /    _portfolio_density /
                            _bucket_window; Agg-only    _sample / _common;
                            _reinsurance)               allocation in _common)
   deferred: 2B / sample-review / pentagon-relax / Bounds-split / Bivariate-split
             (post-beta, conditional)
```

---

## Why P1 (GridDistribution) is the keystone

Both god classes — *and* Severity, Bounds, Bivariate — wrap a discrete
distribution on a grid: a probability vector over a loss index plus `bs`.
Everything downstream (`q`, `var`, `tvar`, `tvar_threshold`, `cdf`, `sf`, `pmf`,
`percentiles`, `snap`) is a function of that vector. Today the *kernel*
(`make_var_tvar`) is already shared in `utilities.py`, but the **plumbing around
it is duplicated and has already drifted**: each class carries its own
`_var_tvar_function` cache, its own `_make_var_tvar` wrapper (Aggregate *returns*,
Portfolio *mutates* — different shapes), its own invalidation, and Aggregate has
the pattern *twice* (`_var_tvar_function` + `_sev_var_tvar_function`). Bounds
constructs `make_var_tvar` by hand at `bounds.py:753`.

`GridDistribution` is a small **read-only value type** holding `(x, p, bs)` that
owns the lazy cache and the accessors. It depends only on numpy/pandas — a true
**leaf**, so it can be built before any split, and adopting it *removes* the
duplicated plumbing now, shrinking every consumer before they are moved.

The payoff is bigger than dedup: a Portfolio unit's allocation, a distorted
density, a discretised severity, and a Bounds object all become *the same type* —
which is how you already think about them. That is why it is the keystone and goes
first.

Resolved in P1 (was an open question): `GridDistribution` is the **spacing-agnostic
cumulative core** (no equal-spacing assumption in the risk measures; `bs` optional,
needed only by `pdf`/`snap`); the existing `_DiscreteRV(xs, ps)` becomes a thin
scipy-naming adapter over it rather than a parallel abstraction (one kernel, two
faces — the `pdf` semantics differ, so no forced merge). P1 also folds in three
payoffs the audit surfaced: a `lev(a)` accessor (limited expected value), a
`limited_tvar(p, a)` accessor (capped TVaR `TVaR_p(min(X, a))`, which `Bounds`
delegates to instead of its hand-rolled `_tvar_x_a`), and
**feeding distortion calibration from a GD** — the Newton calibration stays on
`Distortion` (`spectral.py`); the singular `Portfolio.calibrate_distortion` is
dropped and inlined into the plural, whose set-calibration helper now takes a
`GridDistribution` (GD→Distortion, GD stays a leaf), giving
`Aggregate.calibrate_distortions` with no 1-unit-Portfolio wrap. That set-helper is
the **seed of the shared `_pricing` concern** (flavor (b)); P1 authors it next to
`Distortion`, and P3 relocates it into `_pricing.py` when it births that module. The
name itself still needs the usual `rg`-against-the-surface vetting (incl.
`lev`/`limited_tvar`/`cap`) before it is fixed.

---

## Conventions that apply across all four plans

- **Façade pattern** (`scipy.stats` / `scikit-learn` 0.22): implementation modules
  are underscore-prefixed (`_aggregate.py`, `_portfolio.py`, …); public names are
  re-exported from a thin façade (`distributions.py`, `portfolio.py`) so every
  existing import path keeps working. Underscore answers "import from here?" → no.
- **Commit per pure move**, `uv run pytest` green after each; the frozen numeric
  baseline (`test_baseline.py`) must not move. GridDistribution adoption is
  *behaviour-guarded* — same kernel, so numbers are identical; any drift is a swap
  bug, not a feature.
- **Release mechanics (CLAUDE.md):** new-primitive / compute-extraction phases bump
  `1.0.0a*` + add a `CHANGELOG.md` section; *pure-move* phases (file relocations,
  no behaviour change) are tidying and need no bump. Keep `dev/TODO.md` current;
  move a finished plan to `dev/done/`.
- **Composition over mixins, and only when state is real.** Every plan defers its
  composition step (2B/4B) and explicitly rejects mixin-soup for its own sake.

---

## Suggested first session back

1. Re-read `plan-split-distributions.md` §-1 (the regret guard) and decide if the
   bet still holds.
2. If yes: execute **P1** (`plan-grid-distribution.md`) — it is the lowest-risk,
   highest-concept piece and is valuable even if you never do P2–P4.
3. Reassess after P1 with the two-file rule before committing to the splits.
