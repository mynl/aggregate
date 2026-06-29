# Plan — bivariate legs: a generic leg kernel, with insurance as a View

> **Status: DRAFT — not executed.** Replaces the earlier
> `dev/plan-pnl-legs-unification.md` draft (never executed). The leg engine is
> **generic**: a leg is a function of the two coordinates of a *bivariate
> distribution*, pushed forward. Gross / ceded / net reinsurance is one **special
> case**; the insurance vocabulary lives in a **View** on top, never in the kernel.
> Author insight (2026-06-29): "a leg is just two legs of a bivariate — the special
> case where the bivariate is gross-and-ceded. Define in terms of a generic
> bivariate (`freq × coupled bivariate severity`) and legs fall out for free; do not
> lock into insurance-specific language."

---

## Consolidation note — plans moved to `dev/done/` (2026-06-29)

This plan is the canonical reference for the leg model; it supersedes the
`[analysis]`/`[exhibit]` workstreams of the shipped Phase-2/3 plans and **replaces**
the intermediate `dev/plan-pnl-legs-unification.md` (deleted — never executed; fully
absorbed here). Moved to `dev/done/` (design realized in code a116–a120 and/or
subsumed here; **nothing unique lost** — the legs-model essentials are in §1–§3):

| Moved | Now |
|---|---|
| `pre-plan-expense-variable-rating.md` | Pre-plan; realized Phase 1 + Phase 3. |
| `pre-plan-reinstatements.md` | Pre-plan; realized Phase 2. Future nugget (event-date / pro-rata-as-to-time terms) in `dev/TODO.md`. |
| `plan-variable-rating.md` | Executed a118–a120. |
| `plan-variable-rating-appendix.md` | Legs-model / dimensionality / engine-seam absorbed into §1–§3. |
| `plan-pnl-gcn-reins.md` | Executed Phase 1 (GCN + reins economics). |

**Kept in `dev/`** (not moved): `plan-pnl-first-class.md` (the reporting basis this
plan reshapes); `reinstatements.md` (worked arithmetic, referenced by
`reinstatement.py`); `plan-pnl-portfolio.md` (deferred book-level P&L — unblocked by
this plan's kernel); `plan-display-mode.md`, `plan-signed-bounded-window-overflow.md`,
`plan-plotting-punchups.md` (independent).

---

## Why

The pushforward engine — `BivariateDistribution.pushforward(φ)`, `pushforward_1d`,
`transformed_moments` — already operates on **a bivariate law + an arbitrary
vectorized φ**. It never needed "gross" or "ceded." Today, though, two analysis
classes each define their own legs *and* their own `gcn_df`, with insurance names
(`perspective ∈ {gross, ceded, net}`) baked into the leg state — domain leaking into
the kernel. The fix is to name the kernel generically and push the insurance
vocabulary into a consumer layer:

```
bivariate (X, Y)  →  legs = f(X, Y)  →  cashflows  →  reporting
                              └── kernel (domain-free) ──┘   └─ View (insurance) ─┘
```

Payoffs: the **1-D/2-D split dissolves** (a 1-D source is a *degenerate* bivariate);
**"net" stops being a third axis** (it is the leg `f = X − Y`; "any two of three" is
just the insurance affine identity `G = C + N`); and **future couplings come for
free** — parametric / index covers with basis risk (`net = loss − payout(index)`,
copula-coupled), clash between correlated lines, reinsurer credit risk, stop-loss on
correlated portfolios — none expressible in gross/ceded terms.

---

## The three layers

1. **Kernel (domain-free).** A `BivariateDistribution` + label-free **legs**
   `f(X, Y)` pushed forward; per-leg moments + `GridDistribution`; leg algebra. No
   insurance terms. → `legs.py`.
2. **Source.** A bivariate law, built either as a **true bivariate**
   (`freq × coupled bivariate severity`: independent / copula / comonotone /
   clash; occurrence reinsurance) **or** as a **univariate + map** (`Y = κ(X)`,
   the aggregate-reinsurance / cession case — a *degenerate* bivariate). One
   protocol; the degenerate case takes the 1-D fast path. → reuse / extend
   `BivariateDistribution` / `BivariateAggregate`.
3. **View (insurance).** Accepts a cession-style bivariate, **assembles** the
   standard leg set, **owns the labels** (`gross/ceded/net`, `premium/loss/expense`,
   `consideration/obligation`) and the **grouping** (`gcn_df`, fixed `summary_df`).
   Reinsurance, variable rating, and reinstatement are Views. **The View is
   *composition* over a `LegSet`, not a mixin on `Leg`** — a mixin on `Leg` would
   re-attach domain semantics per leg (the coupling we are removing); if anything it
   mixes onto the *analysis object* that owns the leg set.

---

## §1 The leg (kernel)

**A leg is one cash-flow stream** — a function of the bivariate's coordinates. The
kernel `Leg` is label-free:

| field | values | role |
|---|---|---|
| `name` | e.g. `gross_loss`, `ceded_premium` | identity (the View maps name → labels) |
| `map` | a vectorized `f(X, Y)` returning the leg's **signed** value | the amount |
| `is_value` | orientation flag for the resulting `GridDistribution` | evaluation detail |

No `perspective` / `category` / `kind` on the leg — those are **View labels** (Alt B,
locked 2026-06-29). Signs live in the maps, so cashflow algebra is just summation:
any combination (e.g. a margin, or `net = X − Y`) is itself a leg, a map of `(X, Y)`.

`Margin = Σ legs` (signs baked in). **Rows add and columns add — in expectation.**

## §2 The source (a bivariate, possibly degenerate)

Ceded loss is the second coordinate; everything pivots on whether it is a genuine
random axis or a function of the first:

- **True bivariate → full 2-D.** `(X, Y)` from `freq × coupled bivariate severity`
  (copula / clash / occurrence reins). The leg pushes over the joint
  (`BivariateDistribution.pushforward`).
- **Univariate + map → degenerate (1-D fast path).** `Y = κ(X)` deterministic (the
  ceder `κ`; aggregate reinsurance). All mass lies on the curve `Y = κ(X)`; the leg
  `f(X, κ(X)) = ψ(X)` pushes over the univariate density (`pushforward_1d`).

**One protocol.** The source exposes `pushforward(f)` and `moments(f)`; a degenerate
source advertises "I am a graph `Y = κ(X)`" so the kernel takes the cheap 1-D path.
The kernel never branches on insurance, and **"1-D vs 2-D is a source swap."**

(Self-consistency: even a *per-claim comonotone* severity yields a *non-*comonotone
aggregate joint, because the random claim count decouples the margins — which is
exactly why reinstatement's `(L, R)` is genuinely 2-D.)

## §3 How a leg is computed (the internals)

> Doc-ready: graduates to a user/dev docs page (alongside `dev/pipeline-*.rst` — a
> "bivariate leg model / pipeline" page) when `legs.py` lands. Keep the `X` / `Y`
> generic notation, with `κ` the cession map.

### The one idea: a leg is a pushforward

A leg is a deterministic function of the random state; its distribution is the
**image (pushforward) of the source measure through that function**. The source is a
finite set of weighted atoms `(state_k, p_k)`. To get a leg `f`'s distribution:

1. **Evaluate** `z_k = f(state_k)` at every atom.
2. **Bin by output**: value `z_k` carries mass `p_k`. `f` is generally
   **many-to-one** (a saturating XOL maps a range of `X` to one ceded value), so
   atoms landing on the same `z` have masses **added** — *accumulate*, do not sort.
   (This is why the old comonotone "sort-and-relabel" only worked for monotone 1-1
   maps like a quota share; an XOL needs binning.)
3. **Re-bucket** `z` onto a regular grid → a `GridDistribution`, so `q / var / tvar /
   cdf` all route through the one canonical object.

In parallel, **exact moments** come straight off the source —
`E[f^m] = Σ_k p_k f(state_k)^m` — with *no* re-bucketing (the "EX" ground truth;
means add exactly). Each leg yields **(exact moments, `GridDistribution`)**.
`pushforward_1d`, `BivariateDistribution.pushforward`, and `transformed_moments`
are this one operation over different atom sets.

### Dimensionality lives in the source, not the leg

Every leg has the universal signature `f(X, Y)`. The atoms decide the dimension:

- **Degenerate source (`Y = κ(X)`):** atoms `(x_i, κ(x_i))` on the comonotone curve,
  parameterized by the univariate grid; `f(x, κ(x)) = ψ(x)` pushes over the
  univariate density. **1-D.**
- **Full bivariate:** atoms `(x_i, y_j, p_ij)`; `f(x_i, y_j)` pushes over the joint.
  **2-D.**

> A leg is **1-D iff its map factors through `X` alone** (`f(X, Y) = ψ(X)`); 2-D iff
> it genuinely needs `Y`. A degenerate source forces `Y = κ(X)`, so every leg factors
> through `X` → all 1-D. A full bivariate makes `Y` independent information → any leg
> touching it is 2-D.

| Leg | map `f(X, Y)` | degenerate (`Y=κ(X)`) | full bivariate |
|---|---|---|---|
| ceded loss | `Y` | `κ(X)` → ψ(X), **1-D** | recovery → **2-D** |
| retro gross premium | `φ_retro(X − Y)` (net account loss) | `φ(X − κ(X))` → ψ(X), **1-D** | needs `X − Y` → **2-D** |
| swing ceded premium | `φ_swing(Y)` | `φ(κ(X))` → ψ(X), **1-D** | `φ(Y)` → **2-D** |
| net loss | `X − Y` | `X − κ(X)` → ψ(X), **1-D** | `X − Y` → **2-D** |
| reinstatement premium | `D + h(R)` (axis `Y = R` unlimited; ceded loss `A(R)`) | — (intrinsically occurrence) | `h(R)` → **always 2-D** |
| index cover (basis risk) | `X − payout(Y)` (Y = index, copula-coupled) | — | **2-D** |

Reinstatement is *always* 2-D (an occurrence construct; its `Y`-axis is the
**unlimited** recovery `R`, actual ceded loss the capped `A(R) = min(R, cap)`). The
index-cover row shows the generic payoff: `Y` is not a function of `X`, so it is
unreachable from gross/ceded language.

### Building the leg set, and the feature override

The **base** leg set (a View-assembled set of maps): `gross_loss = X`,
`ceded_loss = Y`, `net_loss = X − Y`, `gross_premium = P_G`, `ceded_premium = P_C`,
`gross_expense = E_G`, `commission = C`, plus derived underwriting combinations.

A **feature fills one leg** = swaps one base map for its φ-map — the *only*
feature-specific step, keyed once by the terms' `(target_leg, loss_basis, φ)`:

- swing → `ceded_premium.map = φ_swing(Y)`
- retro → `gross_premium.map = φ_retro(X − Y)`
- corridor → `ceded_loss.map = φ_corr(Y / P)·P` (and `net_loss` recomputes)
- reinstatement → **two** swaps, `ceded_premium.map = D + h(R)` and
  `ceded_loss.map = A(R)`, plus the decision-3 aggregate-cover legs.

The kernel then pushes the *whole* set over the source, uniformly.

### Net is a fresh pushforward, not arithmetic

`net` is **not** subtraction of the gross and ceded `GridDistribution`s — they are
*dependent* (same `(X, Y)`). Net is a **fresh pushforward of `f_net(X, Y) = X − Y`**.

- **Means add** (`E[X − Y] = E[X] − E[Y]`).
- **SDs / percentiles do not** — `Var(X − Y)` carries the covariance, which only the
  joint knows. Degenerate source → `X − κ(X) = ψ(X)`, 1-D; full bivariate → 2-D.

This is why the GCN's Mean rows add across columns while SD / percentile rows are
per-column marginals — separate pushforwards of the same source. ("Means add, SDs
don't.")

## §4 The View (insurance)

A View is composition over a `LegSet`. It:

- **accepts** a source it understands (a cession-style bivariate: degenerate for
  aggregate reins, full for occurrence reins);
- **assembles** the standard leg set (with any one feature override);
- **owns the labels** — a `name → (perspective, category)` dictionary
  (`category` ⇒ `kind`: premium→consideration, loss/expense→obligation); and
- **groups** for reporting: `gcn_df` (by `category × perspective`, the Gross/Ceded/Net
  waterfall via `gcn_assemble_column`); `summary_df` (by `kind` — the **fixed** small
  `Consideration / Obligation / Margin` table; **never morphs to `gcn_df`**, guideline
  1 of `dev/reporting-guidelines.md`).

Reinsurance, variable rating, and reinstatement are View configurations differing
only in which leg(s) the feature overrides. A non-insurance consumer can define its
own View (or use a generic marginal report) over the same kernel.

---

## Target structure / modules

- **`legs.py` (kernel, domain-free):** `Leg`, `LegSet` (ordered named legs +
  `evaluate(source)` → per-leg moments + `GridDistribution`s + leg algebra). No
  insurance terms.
- **Source:** reuse `BivariateDistribution` / `BivariateAggregate`; add the source
  protocol (`pushforward` / `moments` / degenerate-graph advertisement) so a
  univariate+`κ` and a full joint look the same to the kernel.
- **Insurance View:** the GCN / summary labeling + assembly (refactored out of
  `_pnl` / the two analysis classes). `ReinstatementAnalysis` and
  `VariableRatingAnalysis` become **View builders** over the kernel; their bespoke
  extras (`tail_df`, `plot`, `bs_*`, `_repr_html_`) stay for now (pended for full
  collapse).

---

## Workstreams

- **`[summary-fix]`** (today) — delete the `if self._gcn is not None: return
  self.gcn_df` morph in `PnL.summary_df` (`_pnl.py:469`); `summary_df` is always the
  fixed small table. Update tests asserting the morph.
- **`[legs-core]`** — `legs.py`: `Leg`, `LegSet`, the source-driven evaluator, leg
  algebra, generic per-leg stats. Domain-free; unit-test on a synthetic source.
- **`[source]`** — the bivariate source protocol incl. the degenerate (univariate +
  `κ`) fast path; reuse existing bivariate constructors.
- **`[insurance-view]`** — the View: label map, leg-set assembly (+ one feature
  override), `gcn_df` (via `gcn_assemble_column`) and the fixed `summary_df`.
- **`[variable-on-core]`** — rebuild `VariableRatingAnalysis` as a View builder;
  `tests/test_variable_rating_analysis.py` stays green.
- **`[reinstatement-on-core]`** — rebuild `ReinstatementAnalysis`'s leg set + GCN /
  summary on the kernel; keep its bespoke extras; the full 62-test suite stays green
  (only `summary_df` no-morph snapshots shift).

## Implementation order

1. `[summary-fix]` — small, immediate.
2. `[legs-core]` + `[source]` — get the kernel + protocol right with synthetic tests.
3. `[insurance-view]` + `[variable-on-core]` — simpler View; prove the kernel.
4. `[reinstatement-on-core]` — the harder leg set (two-map + agg cover); tests green.

## Pended (explicitly not in this pass)

Perfect exhibits (`validation_df` = old summary, `reins_*` flavors, column-unit
purity / presentation, `tail_df` / `plot` / `_repr_html_` unification — per
`dev/reporting-guidelines.md`); full class collapse into one `ContractAnalysis`;
**additional source couplings** (independent / copula / clash / shuffle-of-Min
construction beyond what already exists) — adopt the abstraction now, **build a new
coupling only when an application lands** (YAGNI); occurrence-basis variable-rating
*surface*; retro + reinsurance; generic non-insurance reporting beyond what is needed;
`FEATURES.csv` rows.

## Tests / regression bar

`tests/test_legs.py` (new, synthetic source): degenerate-vs-full agreement, many-to-one
binning, leg algebra, `net` as a fresh pushforward (means add, SDs differ). All
existing variable-rating + reinstatement tests green; `summary_df` no-morph snapshot
updates are the only intended diffs. `uv run pytest` green; `test_suite.agg` parses +
snapshot-matches.

## Housekeeping

Plan-based change → bump `1.0.0a*`; `CHANGELOG.md` section; `dev/TODO.md` updated;
move this plan to `dev/done/` at close. **Docs:** §3 graduates to a docs page when
`legs.py` lands (generic `X` / `Y` / `κ` notation). Do not build the doc tree in the
loop.

---

## Relation to existing plans

- **Replaces** `dev/plan-pnl-legs-unification.md` (intermediate draft, deleted) and
  **supersedes** the `[analysis]` / `[exhibit]` workstreams of
  `dev/done/plan-variable-rating.md` and `dev/done/plan-reinstatements.md` — both now
  sit on the kernel. Legs-model design from
  `dev/done/plan-variable-rating-appendix.md` §1/§2/§6 is absorbed into §1–§3.
- **Builds on** the shipped Phase-1 `PnL` + `gcn_assemble_column`
  (`dev/plan-pnl-first-class.md` — kept in `dev/` as the reporting basis — and
  `dev/done/plan-pnl-gcn-reins.md`).
- **Unblocks** `dev/plan-pnl-portfolio.md` — once a `pnl` is a clean leg set over a
  bivariate, book-level P&L composes leg sets across units on the same kernel.
- **Untouched:** `dev/plan-display-mode.md`,
  `dev/plan-signed-bounded-window-overflow.md`, `dev/plan-plotting-punchups.md`.
- Moves toward `dev/reporting-guidelines.md` (fixed `summary_df` now; rest pended).
