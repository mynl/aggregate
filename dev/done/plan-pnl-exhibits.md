# Plan — [PnL-Exhibits] fixed-shape, correct exhibits on the P&L value object

Status: **DONE** (`1.0.0a123`, 2026-07-01). Follows `dev/done/plan-pnl-api.md`
(the a122 full-Monty). Author walked each exhibit type; items below.

Landed: Phase 1 (PnL-class fixes) + Phase 2 (always-PnL routing + margin_df +
analysis attach). Full suite **2194 passed**; `FEATURES.csv` audit clean.
Version a123; CHANGELOG / TODO updated. Docs unchanged (no stale refs; the DecL
grammar and doc symbols are untouched).

Guiding rule (author): **exhibits are fixed.** An exhibit may grow *columns*
(e.g. `stats_df` on an Aggregate) or *rows* (e.g. `validation_df`, or `summary_df`
on a `Portfolio` — one row per unit), but the shape is otherwise fixed: you know
what it is on sight. This must hold for every `PnL` exhibit.

Answer to author's framing question — *"creating all these exhibits is entirely
generic; it relies only on args passed onto the constructor, right?"*: **Yes, for
the `PnL` value object.** `summary_df`, `stats_df`, `density_df` (and the proposed
`margin_df`) are pure functions of the per-leg value arrays + the shared `probs`
captured at construction (`self._cons`, `self._obl`, `self._result`, `self._probs`).
Nothing reads back to a stochastic engine. The only *domain-specific* exhibit is
the tower/analysis `gcn_df`.

---

## Reproductions (current behavior, a122)

- `ob1` — plain `pnl 10000 premium less 85% lr sev lognorm 50 cv 3 poisson`
  → `PnL`. summary_df has cols `[EX, % Consid, SD, CV, Skew, P01, Median, P99]`.
  **Bug:** `consideration` row shows `EX=9999.99998`, `SD=0.446`, `Skew=8.9e-14`
  though it is a constant point mass at 10000.
- `ob2` — `retro basic ... premium ...` → **`VariableRatingAnalysis`** (its
  `summary_df` is the section/item Gross/Ceded/Net/Impact table — the "morph").
- `ob3` — `... aggregate net of 1200 xs 8000 deposit 400 cede 30% ...`
  → **`PnLTower`**: `has gcn_df=True, has stats_df=False`.
- `ob4` — `... 20% loss expenses` → `PnL`. **Bug:** `density_df['expense']`
  collapses to a point mass at `1599.06` (= `0.20 × E[gross loss]`).

Root causes found:
- **SD/EX noise on a constant leg** = the source aggregate's `p_total` sums to
  `0.99999998` (clipped tail mass ~1.5e-8). For a constant leg `values=c`:
  `mean = c·Σp`, `var = c²·Σp·(1−Σp)` → `EX=9999.99998`, `SD≈0.45`. **Not** dust
  (0.45 ≫ VALIDATION_NOISE), so `_snap_noise` cannot fix it; the fix is to
  **normalize `probs` to sum to 1** at `PnL` construction.
- **Skew dust** (8.9e-14) *is* below `validation.noise` → `_snap_noise`
  (`aggregate.moments._snap_noise`) is the right tool.
- **Expense collapse** = `resolve_expense` deliberately resolves a `loss`-basis
  term to a *scalar* `rate × E[gross loss]` (its docstring: "a fraction of the
  expected gross loss (deterministic)"). Author wants `loss`-basis expense to be
  **stochastic** — a per-atom `rate × x` (this is LAE, proportional to *actual*
  loss).

---

## Items and disposition

### [Exhibit-Summary-Fixed] — `summary_df` fixed shape
- Template = the `ob1` `PnL.summary_df` (the correct one).
- Columns fixed. Rows vary **only** with the number of consideration/obligation
  legs (each leaf + a Total when >1) — same discipline as `Portfolio.summary_df`
  varying by unit.
- Rename column `% Consid` → **`Scaled`**.
- (Renaming `consideration` → `premium` is the **caller's** job at construction,
  not this exhibit's.)
- The "morph" is really a *routing* artifact: `ob2` is a different *class*. See
  [Exhibit-Routing].

### [Exhibit-Summary-Noise] — constant legs must be exact
- **Normalize `probs`** so `Σp = 1` at `PnL.__init__` (principled: a P&L *is* a
  probability distribution; the clipped source-tail is a discretization artifact).
  Makes a constant leg exact: `EX = c`, `SD = 0`.
- Apply `_snap_noise` to `SD`, `Skew` (and CV) in the rendered exhibits to clear
  residual float dust.

### [Exhibit-Stats-Property-Fixed] — `stats_df` → property, fixed layout
- `stats_df` becomes a **property** (not a method). Commit to a single **scale**
  at construction: default = **`E[consideration]`**; the RP case uses
  `gross − deposit` as the natural scale (caller-set at construction).
- `% of Total` → one fixed **`Scaled`** column per leg, dividing by the committed
  scalar scale (not "pretending" `SD_scaled = SD(loss ratio)` — just a unitless,
  comparable number).
- `CV` and `Skew` **scaled** cells → `nan` (not meaningful under scaling).
- Guard CV blow-up: `mean ≈ 0` → CV `nan` (reuse the `_moments_of` zero-mean
  guard; extend to a relative floor).
- One row index level (no section/item).
- Percentile row labels: `f'P{q*100:.3g}'` → `P1, P5, P10, P25, P50, P75, P90,
  P95, P99` (and `P99.5` etc. if the ladder grows).

### [Exhibit-Margin-DF] — `gcn_df` off `PnL`; new `margin_df`  ⟵ **OPEN**
- No `PnL` carries `gcn_df` (it is domain-specific → stays on the analysis /
  tower objects).
- `ob3` (a ceded-premium `pnl`) should expose the **standard** `stats_df` and a
  **new `margin_df`**, *not* `gcn_df`.
- **OPEN Q — what is `margin_df`?** Its exact contents/shape are undefined.
- **OPEN Q — routing:** does a ceded-premium `pnl` (`ob3`) now return a `PnL`
  (with `margin_df` + `stats_df`) instead of a `PnLTower`? And does retro-premium
  (`ob2`) return a `PnL` instead of `VariableRatingAnalysis`? i.e. should
  `build('pnl …')` **always** yield a `PnL`, with the waterfall/analysis machinery
  hanging off it or on a separate accessor?

### [Exhibit-Expense-Stochastic] — `loss`-basis expense is per-atom
- In the plain `PnL` path, a `loss`-basis expense term becomes a callable
  `lambda x: rate * x` (stochastic, per atom) rather than a scalar
  `rate × E[loss]`.
- `fixed`- and `premium`-basis terms stay scalar (deterministic) for a fixed
  premium.
- **OPEN Q — GCN path:** `resolve_expense` currently feeds *scalar* magnitudes
  into `_gcn_magnitudes` (commission credits, per-side split). Should `loss`-basis
  expense also be stochastic there? (Tied to the routing decision.)

### [Exhibit-Stochastic-Engine] — `PnL` keeps its generator
- `create_pnl` currently **discards** the `source`. Author wants it **retained**
  as `self._stochastic_engine`, exposed via a `stochastic_engine` property
  ("a P&L *has a* stochastic generator; we're agnostic about what it is").
- Update the module docstring's "consumes and throws away its engine" philosophy
  to "holds an opaque reference; never *depends* on it for the exhibits."

---

## Decisions (author, 2026-07-01)
1. **Routing = Always return `PnL`.** Every `build('pnl …')` returns a `PnL` value
   object with the fixed exhibits. The Gross/Ceded/Net tower and the
   reinstatement / variable-rating analyses become **attached accessors**
   (`pnl.tower` / `pnl.analysis`), **never** the top-level return.
2. **`margin_df` = stat-rows × perspective-cols** — today's `gcn_df` layout kept
   as `PnL.margin_df`: rows = `EX / SD / CV / Skew / P1…P99`, columns = `Gross /
   Ceded / Net / benefit`. Present only on a cession-bearing P&L (forwarded from
   the attached tower/analysis); a plain P&L has none.

## Concrete design (always-`PnL`)

**`create_pnl` gains** (all committed at construction):
- retain `source` as `self._stochastic_engine` (property `stochastic_engine`);
- **normalize `probs`** to sum to 1 (kills the constant-leg SD/EX noise);
- commit a **scale**: default `E[Total consideration]`; a caller may pass an
  explicit scalar (the RP case: `gross − deposit`). Store `_scale_value` +
  `_scale_label`.

**PnL exhibits (all fixed):**
- `summary_df` — cols `[EX, Scaled, SD, CV, Skew, P1, Median, P99]`; rows = each
  consideration leg (+ Total if >1), each obligation leg (+ Total if >1), result.
  Snap SD/CV/Skew dust with `moments._snap_noise`.
- `stats_df` — **property**; index = `EX/SD/CV/Skew + P{q*100:.3g}` ladder;
  columns = per-leg `(value, Scaled)`; `Scaled` divides by the committed scale;
  `CV`/`Skew` `Scaled` cells = `nan`; CV guarded near zero mean.
- `density_df` — unchanged (ordered `{leg: GD}`).
- `margin_df` — the waterfall, forwarded from `self._tower` / `self._analysis`
  (reshaped from `gcn_df`); raises a clear error on a plain P&L.

**Attachment per recipe kind** (the **net** position is the returned face):
- `plain` → PnL, nothing attached.
- `gcn` → **net** PnL (net premium / net-loss marginal / net expense); attach
  `_tower` = the GCN `PnLTower`.
- `reins` → **net** PnL (stochastic net premium `gross − D − h(R)`); attach
  `_analysis` = `ReinstatementAnalysis`.
- `var` / `retro` → **net** PnL; attach `_analysis` = `VariableRatingAnalysis`.

**Minor defaults (chosen, not blocking):**
- `summary_df` percentiles `P01/P99` → **`P1/P99`** (align with `stats_df`).
- `loss`-basis expense: **stochastic** (`rate·x` per atom) in the plain path; in
  the GCN path the per-side commission/magnitude split still needs the scalar
  `E[loss]` form — keep scalar there for now (noted).

## Execution order
1. **[PnL-class fixes]** — normalize+snap, `stochastic_engine`, `Scaled` rename,
   `stats_df` property + committed scale, `loss`-expense stochastic (plain).
   (Touches the `PnL` surface; ripples into the plain-PnL tests.)
2. **[Always-PnL routing]** — net-PnL face + `.tower`/`.analysis` attach +
   `margin_df`; rewrite the gcn / reins / var routing + their tests. **Reverses
   the a121/a122 return types** — the larger, staged piece.
