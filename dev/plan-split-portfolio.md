# Plan P4 — split `portfolio.py` (three subsystems under one `Portfolio` class)

> **Status: DRAFT — not executed.** See `plan-README.md`. Sequenced **after** P1
> (GridDistribution adopted), P2 (plotting lifted out), and P3 (distributions split
> **and the shared concerns `_validation`/`_bucket_window`/`_pricing` born**) — by then
> `Portfolio` already delegates its accessors, has no plotting, and the shared
> concern modules already exist for it to drop its implementations into.
>
> **Release mechanics (CLAUDE.md).** The façade / relocation phases are *pure moves*
> (tidying, no bump). The compute-extraction phase adds tests / touches behaviour
> indirectly → bump + `CHANGELOG.md`. `uv run pytest` green before every commit; the
> frozen baseline must not move.

---

## 0. The structural difference from `distributions.py`

`portfolio.py` is ~4,965 lines, ~104 methods, but it is **one class**
(`Portfolio`) — not a `Frequency`/`Severity`-style taxonomy. So there is **no
kind-split**. But `Portfolio` *does* have a natural internal split — not by class
kind, but by **how the joint loss distribution is built**:

1. **Density-based** (independence) — works with **densities**: the independent-sum
   combine over the unit `Aggregate`s (FFT convolution is the mechanic). This is
   most of what the refactor has worked on to date. The distinguishing feature is
   *densities*, not the FFT — hence the module name `_portfolio_density`, not
   `_portfolio_fft`.
2. **Sample-based** (dependence / correlation) — build from a sample, the
   switcheroo, Iman–Conover, comonotonic allocations. This is the **answer to the
   "what about portfolios with correlation?" critique** — important, VIP, and **not
   yet reviewed during this refactor** (it is still on the agenda; see §4).
3. **Common exeqa-based numerics** — once *either* path has produced an augmented
   `density_df` carrying the `exeqa_*` columns, everything downstream (apply a
   distortion, allocate the distorted price across units, diagnostics) is
   **identical**. That identity is the **magic of the switcheroo**: the allocation
   numerics work off `exeqa` (conditional expectations), not the raw losses, so they
   do not care how the joint distribution was built.

So P4 keeps the single public `Portfolio` class and splits its body into **three
implementation modules behind a façade** — `_portfolio_density`, `_portfolio_sample`,
`_portfolio_common` — **plus** it *consumes* the P3 shared concerns
(`_validation`/`_bucket_window`/`_pricing`) and the P1 `GridDistribution` adoption. One
class, three subsystems; you still instantiate the same `Portfolio`, so behaviour is
frozen and the tests stay green.

```python
# portfolio.py — public façade; defines no logic of its own
from ._portfolio import *            # noqa: F401,F403  (the Portfolio class)
from ._portfolio_common import *     # noqa: F401,F403  (free helpers)
```

The `Portfolio` class lives in `_portfolio.py` and is composed from the three
subsystem modules (free functions it calls, plus the FFT/sample construction
bridges). `aggregate.Portfolio` and `aggregate.portfolio.Portfolio` both keep
working — the `scipy.stats` / `scikit-learn` underscore-module + façade pattern of
P3 §1.

---

## 1. The three subsystems + concern map

| Cluster | Representative methods | Disposition |
|---|---|---|
| **Density-based** | `__init__`, `update`, `add_exa`, `trim_density_df`, the independent-sum combine math | **`_portfolio_density`** — works with densities; the FFT combine engine; produces an augmented `density_df` with `exeqa_*` columns via `add_exa`. Pure-math helpers extracted as free functions. |
| **Sample-based** | `sample`, `create_from_sample`, `add_exa_sample`, `make_comonotonic_allocations`(`_work`), `swap_density_df` (switcheroo), Iman–Conover | **`_portfolio_sample`** — the dependence path; produces the *same* augmented `density_df`/`exeqa_*` from a sample. **Extracted now, reviewed later** (§4). |
| **Common exeqa numerics** | `apply_distortion`, `augmented_df(s)`, `allocation_diagnostics`, `var_dict`, the linear/lifted **allocation**, `bodoff`, the per-unit breakdown in `pricing_at`/`pentagon_at`/`analyze_distortion(s)` | **`_portfolio_common`** — operates on the `exeqa_*` columns regardless of how they were built; this is where **allocation** lives (Port-only, *not* shared with Aggregate). |
| **Single-GD total pricing** | `calibrate_distortions`, the total-side of `price`/`price_stand_alone`/`price_pentagon`/`price_ccoc`/`pricing_at`/`pentagon_at` | **→ shared `_pricing` (P3).** `Portfolio` resolves `a`/`P` from its pentagon state and hands its **total** GD to `_pricing`; the per-unit allocation of the result stays in `_portfolio_common`. The singular `calibrate_distortion` was already removed in P1 Phase L. |
| **Distribution accessors** | `q`, `cdf`, `sf`, `pdf`, `pmf`, `var`, `tvar`, `tvar_threshold`, `snap`, `as_severity`, `approximate` | **Delegated to `GridDistribution` (P1)** — already thin. **`percentiles` is dropped, not delegated** — the deprecated interpolated per-unit table (only `percentiles` in the library; no `Aggregate` analogue), superseded by vector-`q`. Remove it in P4. |
| **Bucket / window sizing** | `recommend_bucket`, `best_bucket`, `best_window`, `bs_window_df` | **→ shared `_bucket_window` (P3).** Drop these into the `_bucket_window.py` born from `Aggregate`, so the two sizers sit side by side. |
| **Validation** | `valid`, `validation_explanation`, `remove_fuzz` | **→ shared `_validation` (P3).** Drop alongside the `Aggregate` validation machinery (similar-not-identical, co-located). `remove_fuzz` may stay local if Port-specific. |
| **Reporting / DataFrames / narratives** | `info`, `summary_df`, `tail_df`, `tail_description`/`_explanation`, `bs_description`/`_explanation`, `unit_density*`, `aligned_unit_density_df`, `pprogram*`, `nice_program`, `unit_renamer`, `spec`, `json`, `save` | Pure formatting; fold into `_portfolio_common` (or a `_portfolio_report` sub-file if it grows past the two-file rule). |
| **Plotting** | `plot`, `scatter`, `sample_compare`, `sample_density_compare` | **→ `plots/_portfolio.py` (P2).** |
| **Bounds** | `allocation_bounds`, `pricing_bounds` | Already delegate to `bounds.py`; leave. |

---

## 2. Phase 4A — façade + three-subsystem extraction

Same play as P3's 2A (pull `self`-only-reads-arrays methods into module-level
functions), but organised into the three subsystem modules. Each numbered step is a
standalone **pure-move** commit, `uv run pytest` green, baseline unmoved:

- **4A.1 — `_portfolio_density`.** Extract the independent-sum combine math and the
  `add_exa` conditional-expectation / `exeqa_*`-building kernels as free functions;
  `Portfolio.update`/`add_exa` become thin callers.
- **4A.2 — `_portfolio_common`.** Extract the exeqa-based numerics —
  `apply_distortion`, `augmented_df` construction, the linear/lifted **allocation**
  kernels, `allocation_diagnostics`, the convex-hull helpers (`check01`,
  `make_array`, `convex_points`), `bodoff`. These take the augmented `density_df`
  (or its `exeqa_*` columns) + a distortion and are **agnostic to how the joint was
  built** — the switcheroo payoff made explicit.
- **4A.3 — `_portfolio_sample`.** Move `sample`, `create_from_sample`,
  `add_exa_sample`, `make_comonotonic_allocations`(`_work`), `swap_density_df`, and
  the Iman–Conover machinery. **Behaviour-frozen** — this is a relocation, *not* the
  review (§4).
- **4A.4 — façade.** Reduce `portfolio.py` to the re-export façade (§0); confirm
  `aggregate.Portfolio` and `aggregate.portfolio.Portfolio` both resolve.

Payoffs as in P3: the combine, allocation, and sample math become **directly
testable** (today reachable only through a full `update()`/`sample()`) and
notebook-inspectable. Add focused unit tests calling the extracted kernels directly
(small hand-checkable inputs; augments, does not replace, the end-to-end baseline).

---

## 3. Consuming the P3 shared concerns

P4 does **not** re-derive validation, bucket/window, or single-GD pricing — those
modules already exist from P3 §1b, seeded from `Aggregate`. P4 **drops the
`Portfolio` implementations into the same files** so the two are read side by side:

- **`_bucket_window`** ← `recommend_bucket`/`best_bucket`/`best_window`/`bs_window_df`.
- **`_validation`** ← `valid`/`validation_explanation`.
- **`_pricing`** ← `calibrate_distortions` resolves `a`/`P` from pentagon state and
  hands the **total** GD to `_pricing`; the per-unit allocation of the distorted
  price stays in `_portfolio_common`.

Each drop is a pure move; the shared modules stay leaves (they take a GD + plain
data, never import `_portfolio`/`_aggregate`), so no cycle. The
distribution-accessor duplication with `Aggregate` is **already resolved by P1**
(held `GridDistribution`) — this plan must **not** re-introduce a parallel accessor
path.

---

## 4. The sample subsystem — extract now, review later

`_portfolio_sample` is the **answer to "what about portfolios with correlation"** —
the dependence story (sample-build, switcheroo, Iman–Conover, comonotonic). It is
VIP and **has not been reviewed during this refactor**. The staging decision:

- **Now (Phase 4A.3):** extract it structurally, **behaviour-frozen**, tests green —
  a relocation only, no improvement while moving (the same rule that governs every
  other move here).
- **Later (its own plan):** the substantive review / improvement of the
  sample / switcheroo / dependence machinery. **Coordinate with the separate
  shared-mixing-across-units design — do not pre-empt it.** Whether the sample path
  ever becomes a stateful collaborator (vs. a free-function module) is decided
  *then*, against the two-file rule, not now.

Extracting now means the review starts from one clean, self-contained module instead
of code tangled through the god class — worth doing first even though the review
waits.

---

## 5. Phase 4B — composition seams (DEFERRED; re-decide after 4A)

Mirroring P3's deferred `ReinsuranceProgram`, only where a real stateful
collaborator exists:

- **The sample-subsystem review (§4)** — the primary deferred item; its own plan.
- **A `Portfolio` pricing/allocation collaborator** — *only if* `_portfolio_common`
  + the applied-distortion / `self._augmented_dfs` cache prove to want a stateful
  owner after 4A. Note the **single-GD pricing is already the shared `_pricing`
  module** (not a Portfolio-internal engine); what could still become a collaborator
  is the *allocation* state, not pricing. Re-decide after 4A.

As with P3's 2B: **no mixin-soup for its own sake**; extract a collaborator only when
it owns real state, and apply the two-file rule (`plan-README.md`) before committing.

---

## 6. Guardrails

- Same as P3 §5: commit per pure move, `__all__` hygiene, façade re-exports the
  union, baseline must not move, import-cycle watch.
- **Subsystem leaf discipline.** `_portfolio_common` must operate on the augmented
  `density_df`/`exeqa_*` columns + a distortion, **agnostic to FFT-vs-sample
  origin** — if a common-numerics function branches on how the joint was built, the
  seam is wrong. That agnosticism is the switcheroo invariant and the whole reason
  the three-way split is honest.
- **Behaviour-frozen extraction.** 4A (incl. the sample move) changes no numbers;
  the review that *does* change behaviour is a separate, later, version-bumping plan.
- **No parallel accessor / pricing path.** Accessors → `GridDistribution` (P1);
  single-GD pricing → `_pricing` (P3). Do not let P4 grow a second copy of either.

---

## 7. TODO / sequencing

This is **P4**, last of the four. Depends on P1 (accessors), P2 (plotting out), and
P3 (the Aggregate-compute pattern to mirror **and** the shared concerns to consume).
Within P4: **Phase 4A** (façade + three-subsystem extraction, pure move then a
bump for the new kernel tests) is the deliverable; **the sample review** and **4B**
composition are post-beta, conditional, each its own later plan. File a `dev/TODO.md`
entry pointing at this plan and the README.
