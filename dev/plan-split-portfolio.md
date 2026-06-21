# Plan P4 — split `portfolio.py` (one god class, not a taxonomy)

> **Status: DRAFT — not executed.** See `plan-README.md`. Sequenced **after** P1
> (GridDistribution adopted), P2 (plotting lifted out), and P3 (distributions
> split) — by then `Portfolio` already delegates its accessors and has no plotting,
> so the split moves a smaller, cleaner class.
>
> **Release mechanics (CLAUDE.md).** The façade/relocation phase is a *pure move*
> (tidying, no bump). The compute-extraction phase adds tests / touches behaviour
> indirectly → bump + `CHANGELOG.md`. `uv run pytest` green before every commit;
> the frozen baseline must not move.

---

## 0. The structural difference from `distributions.py`

`portfolio.py` is ~4,965 lines, ~104 methods, but it is **one class**
(`Portfolio`) — not a `Frequency`/`Severity`-style taxonomy. There are no sibling
classes to peel into kind-files, only a few module-level free functions
(`make_comonotonic_allocations_work`, `swap_density_df`, `check01`, `make_array`,
`convex_points`, `make_awkward`). So **Portfolio has no kind-split**; its
decomposition is the *concern-extraction* play (P3's Aggregate-compute pattern),
its share of the P2 plotting subsystem, and the GridDistribution adoption already
done in P1.

---

## 1. Concern clusters inside `Portfolio`

| Cluster | Representative methods | Disposition |
|---|---|---|
| **Combine engine** | `__init__`, `update`, `add_exa`, `add_exa_sample`, `trim_density_df` | Stays the class core; pure-math helpers extracted (§2). |
| **Distribution accessors** | `q`, `cdf`, `sf`, `pdf`, `pmf`, `var`, `tvar`, `tvar_threshold`, `snap`, `percentiles`, `as_severity`, `approximate` | **Delegated to `GridDistribution` in P1** — already thin by the time this plan runs. |
| **Bucket / window sizing** | `recommend_bucket`, `best_bucket`, `best_window`, `bs_window_df` | Mirrors `Aggregate`'s; co-locate the shared math with the bucket helpers extracted in P3. |
| **Reporting / DataFrames / narratives** | `info`, `summary_df`, `tail_df`, `tail_description`/`_explanation`, `bs_description`/`_explanation`, `unit_density*`, `aligned_unit_density_df`, `pprogram*`, `nice_program`, `unit_renamer`, `spec`, `json`, `save` | Pure formatting; candidates for a `_portfolio_report` compute module. |
| **Pricing / distortion / allocation** | `calibrate_distortion(s)`, `apply_distortion`, `augmented_df(s)`, `pricing_at`, `pentagon_at`, `allocation_diagnostics`, `var_dict`, `price`, `price_stand_alone`, `price_pentagon`, `price_ccoc`, `analyze_distortion(s)`, `bodoff` | **Largest, most cohesive cluster** — the strongest composition seam (§3, deferred). |
| **Bounds** | `allocation_bounds`, `pricing_bounds` | Already delegate to `bounds.py`; leave. |
| **Dependence / sampling** | `sample`, `create_from_sample`, `make_comonotonic_allocations`, `scatter`, `sample_compare`, `sample_density_compare` | Ties into the pre-beta dependence work (Iman–Conover / switcheroo); second composition seam (§3, deferred). |
| **Plotting** | `plot`, `scatter`, `sample_compare`, `sample_density_compare` | → `plots/_portfolio.py` in **P2**. |
| **Validation** | `valid`, `validation_explanation`, `remove_fuzz` | Thin; stays on the class. |

---

## 2. Phase 4A — façade + pure-function extraction (mirror of P3's 2A)

Same play as `Aggregate`: pull the methods that take `self` only to read arrays
into a `_portfolio_compute.py` of module-level functions —

- the independent-sum combine math,
- the `add_exa` conditional-expectation / allocation kernels,
- the comonotonic majorisation (`make_comonotonic_allocations_work` already *is*
  free — co-locate it),
- the convex-hull helpers (`check01`, `make_array`, `convex_points`).

Payoffs as in P3: the allocation/combine math becomes directly testable (today
reachable only through a full `update()`) and notebook-inspectable.

`portfolio.py` becomes a thin **façade** re-exporting `_portfolio` (the class) +
`_portfolio_compute` (free functions), exactly the underscore-module + façade
pattern of P3 §1. `aggregate.Portfolio` and `aggregate.portfolio.Portfolio` both
keep working.

Add focused unit tests calling the extracted kernels directly (small
hand-checkable inputs; augments, does not replace, the end-to-end baseline).

---

## 3. Phase 4B — composition seams (DEFERRED; re-decide after 4A)

Two genuine collaborators, mirroring P3's deferred `ReinsuranceProgram`:

- **`PricingEngine`** — the pricing/distortion/allocation cluster is large and
  cohesive (`apply_distortion` → `augmented_df` → `pricing_at` → `pentagon_at` →
  `price`/`analyze_distortions`) and owns real state (the applied distortion, the
  `self._augmented_dfs` cache). A strong extract-to-collaborator candidate.
- **Dependence / sampling** — `sample`, `create_from_sample`, Iman–Conover, and
  comonotonic form the dependence story being firmed up pre-beta. **Coordinate this
  with the separate shared-mixing-across-units design — do not pre-empt it.**
  Whether it becomes a collaborator depends on where that work lands.

As with P3's 2B: **no mixin-soup for its own sake**; extract a collaborator only
when it owns real state, re-decide after 4A, and apply the two-file rule
(`plan-README.md`) before committing.

---

## 4. Guardrails

- Same as P3 §5: commit per pure move, `__all__` hygiene, façade re-exports the
  union, baseline must not move, import-cycle watch.
- The distribution-accessor duplication with `Aggregate` is **already resolved by
  P1** (held `GridDistribution`); this plan must *not* re-introduce a parallel
  accessor path.

---

## 5. TODO / sequencing

This is **P4**, last of the four. Depends on P1 (accessors), benefits from P2
(plotting out) and P3 (the Aggregate-compute pattern to mirror). 4B is post-beta
and conditional.
