# Plan — first-class `PnL`: a distribution object like `Aggregate`

> **Status: DRAFT — not executed.** Builds on the landed `PnL` veneer (Stages
> A–E, `dev/done/plan-pnl.md`). Goal: make `PnL` a *first-class* object whose
> reporting/distribution surface mirrors `Aggregate` — `density_df`, `stats_df`,
> `info`, `_repr_html_`, `qd`, and a richer `summary_df` — so a `PnL` reads and
> prints like an aggregate, just for the **net** position.
>
> **Terminology.** The **risky leg** is `pnl.agg` (the obligation). Every member
> sorts into one of three buckets:
> - **pass-through** — identical meaning to the leg (frequency, severity, claim
>   count, `bs`/`log2`/grid, reins description): delegate to `.agg`.
> - **transform** — the *net* (`C ∓ X`): `density_df`, `stats_df`, moments,
>   `q`/`cdf`/`sf`, `var`/`tvar`, tail.
> - **PnL-specific** — `consideration`, `summary_df`, `gcn_df`, `evaluate`,
>   `prob_loss`.

---

## Confirmed decisions (author)

- **D1 — `pnl_df` becomes `density_df`.** The PnL's own distribution is the
  **net**, so `PnL.density_df` is the net in the **full `Aggregate` schema**
  (`loss`/`p_total`/`p`/`F`/`S`/`lev`/`exa`/`exlea`/`exgta`/`exeqa`/…). `pnl_df`
  retires (one canonical name). The **obligation** is `pnl.agg.density_df`.
- **D2 — index by the net outcome.** The index is the net value `C ∓ X`. The
  transform can be **non-injective** (e.g. an aggregate stop-loss caps the loss →
  a *mass point* in the net), so the frame is built with a **groupby-sum on the
  net value** (the existing `_frame_from` already groups; extend it to the full
  schema). The direct-sum columns (`lev`/`exlea`/`exgta`) must carry the grid
  origin (numerics-2 convention) since the net grid straddles 0.
- **D3 — explicit delegation, no `__getattr__`.** Every first-class member is a
  deliberately written property/method (pass-through, transform, or new). No
  blanket attribute fallback to `.agg` (house "less magic" rule).
- **Boundary rule (state once, apply everywhere):** `pnl.<x>` is the **net**
  (the position); `pnl.agg.<x>` is the **obligation** (the leg). `info` /
  `description` / `explanation` make the split explicit.

---

## Phase A — the data surface (`density_df`, `stats_df`, risk measures)

1. **`density_df` (net, full schema).** Generalize `_frame_from` to emit every
   `Aggregate.density_df` column on the net grid, groupby-summed on the net
   value (D2). Reuse `Aggregate`'s column builders where possible (factor the
   shared `lev`/`exlea`/`exgta`/`exeqa` logic out of `_aggregate.py` rather than
   duplicating). Retire `pnl_df` (update internal callers: moments, `q`/`cdf`/
   `sf`, plot).
2. **`stats_df` (net).** Freq / Sev rows pass through from `agg.stats_df`; the
   **Agg row is the net transform** (`mean → C ∓ E[X]`, `sd` unchanged,
   `skew → −skew` for a loss leg). This is exactly the affine transform removed
   from `Aggregate._describe_signed` in Stage B — it now lives, cleanly and in
   one direction, on the `PnL`.
3. **Risk measures.** `var`/`tvar` (and `q`/`cdf`/`sf`, already present) read the
   net `density_df`. Confirm sign conventions for a payoff (more-is-better) net.

**Open sub-decision (A):** the outcome column/index *name*. Full-schema
consumers (pricing, Portfolio) read `density_df['loss']`; the value here is the
*net*. Options: keep the column `loss` for drop-in compatibility (meaning set by
the boundary rule) with the **index named `net`**; or name both `net` and accept
special-casing. Lean: index `net`, keep a `loss` column for schema parity —
revisit if it reads dishonestly.

## Phase B — text reporting (`info`, `qd`)

4. **`PnL.info`** (item 6). New title (e.g. `Profit & Loss object NAME`), the
   **same rows as `Aggregate.info`** plus **consideration rows near the top**
   (consideration, net mean, `P(loss)`, breakeven `gini_p`?). So an aggregate's
   info content is a *subset* of a PnL's, in a different order. Grid/freq/sev
   rows pass through (they describe the leg). Follow the fixed-layout `info`
   convention from `test_hygiene4.py` (every row present, `n/a` placeholders).
5. **`qd` handles `PnL`** (item 5). `qd(pnl)` prints the PnL the way it prints an
   aggregate — name heading + `summary_df` (the signed additive table / GCN).
   Dispatch in `utilities.qd` on the `PnL` type.

## Phase C — rich reporting (`_repr_html_`, `summary_df` quantiles)

6. **`PnL._repr_html_`** (item 3). Mirror `Aggregate`'s: name as a heading, then
   the `summary_df` rendered. So a `PnL` displays richly in Jupyter like an agg.
7. **`summary_df` quantile columns** (item 2). Add `q(0.01)`, `q(0.05)`,
   `q(0.95)`, `q(0.99)` columns to the signed `summary_df` (the Consideration /
   Obligation / Margin rows). Quantiles read the net `density_df`. (Decide how
   they apply to the Consideration/Obligation rows vs only the Margin row — the
   Margin row is the net; the leg rows may show the leg's own quantiles or be
   blank.)

---

## Tests

Extend `tests/test_pnl.py`: `density_df` full-schema + mass-point under an
aggregate stop (D2); `stats_df` net transform (Freq/Sev pass-through, Agg row
net); `var`/`tvar`; `info` row set (superset of agg, consideration rows present);
`qd(pnl)` runs; `_repr_html_` contains the name + summary; `summary_df` quantile
columns. Mirror any DecL in `decl-testers.agg`.

## Out of scope / later

GCN multi-stage (`dev/plan-pnl-gcn-reins.md`), PortPnL
(`dev/plan-pnl-portfolio.md`). Plot return-period axis
(`dev/plan-plot-return-period.md`) is independent.
