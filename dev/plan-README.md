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

## The four plans, in dependency order

The order is **strict bottom-up**: build the shared primitive, lift the
cross-cutting concern out, *then* reorganise the (now smaller) modules. Each layer
shrinks the next.

| # | Plan | One line | Risk | Independently shippable? |
|---|---|---|---|---|
| **P1** | `plan-grid-distribution.md` | New `GridDistribution` value type owns the marginal-vector accessors (`q/var/tvar/cdf/sf/pmf/…`); the `make_var_tvar` kernel **moves into it** from `utilities.py`; every consumer adopts it | low (pure add + guarded swaps) | yes |
| **P2** | `plan-plots-subsystem.md` | One `plots/` package, per-class modules, `_style.py`, **global matplotlib defer** (the real import-time win) | low (mechanical moves) | yes |
| **P3** | `plan-split-distributions.md` | Kind split → `_fits`/`_frequency`/`_severity`/`_aggregate` + façade; then Aggregate pure-compute extraction | low→med | yes (Phase 1 alone) |
| **P4** | `plan-split-portfolio.md` | `portfolio.py` façade + `_portfolio_compute` extraction (one god class, not a taxonomy) | low→med | yes |

**Deferred bucket** (TODO entries, *not* plans yet — promote only when the seam is
proven by the two-file rule, post-beta):
- Aggregate **2B** `ReinsuranceProgram` collaborator.
- Portfolio **4B** `PricingEngine` collaborator + the dependence/sampling unit
  (coordinate with the separate shared-mixing-across-units design, not pre-empt).
- **Bounds** structural class-split (already isolated; low ROI).
- **Bivariate** structural split (newer god class; cost/benefit still open).

```
P1 GridDistribution ──┐ (shrinks Agg, Port, Severity, Bounds, Bivariate)
                      ├─► P3 split distributions ─► P4 split portfolio
P2 plots subsystem  ──┘ (lifts plotting out of every class)
        deferred: 2B / 4B / Bounds-split / Bivariate-split   (post-beta, conditional)
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

Open question carried in P1: there is already a `_DiscreteRV(xs, ps)` in the
severity code — decide whether `GridDistribution` generalises it rather than
adding a parallel abstraction. And the name itself needs the usual
`rg`-against-the-surface vetting before it is fixed.

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
