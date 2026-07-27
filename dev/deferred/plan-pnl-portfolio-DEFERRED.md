# Plan — `PortPnL`: portfolios of P&L positions — DEFERRED

> **DEFERRED past `1.0.0b1` (author, 2026-07-27).** Not in scope for the beta.
> The **constant-consideration** case (option 1 below) is already delivered
> another way: `[PnL-Engine-Source]` (`1.0.0a125`) subsumed `[Portfolio-of-PnL]`
> — `pnl NAME <prem> less port.NAME` builds over the portfolio net-net total,
> with `port.exp_premium` accumulated for `inherit premium`. No bespoke
> `PortPnL` class was needed. What this file still holds open is **option 2**,
> the loss-sensitive case (net-then-combine ≠ combine-then-net once
> considerations depend on the loss) plus `pnl` *units inside* a `port`, which
> keeps its existing `NotImplementedError` gate. Text below is the original
> design exploration, unedited.

> **Status: DRAFT — design exploration.** Deferred from the v1.0 PnL plan
> (`dev/done/plan-pnl.md` §3.2). Today a `pnl` unit in a `port`/`bivariate`
> raises `NotImplementedError`. This plan works out the design before any build.

---

## The core tension (why it was deferred)

A loss-sensitive consideration `f_i(X_i)` must be netted **per unit before
combining** — `margin_i = f_i(X_i) − X_i`, then `book = Σ margin_i`. But the
existing Portfolio combine convolves the **losses** into `Σ X_i`, which throws
away the per-unit `X_i` that `f_i` needs and any way to attribute premium back to
a unit. So **net-then-combine ≠ combine-then-net** once considerations are
loss-sensitive.

The two coincide only for **constant** considerations (the net is affine in the
one loss): `Σ (C_i − X_i) = Σ C_i − Σ X_i`.

---

## Design options

1. **Constant-consideration `PortPnL` (cheap, correct — likely v1).**
   For constant considerations the book net is
   `make_pnl(portfolio_total, consideration = Σ C_i)` plus a **stacked per-unit
   signed summary** (each unit's Consideration/Obligation/Margin + a total row,
   the analogue of `Portfolio.summary_df`). The obligation combine `Σ X_i` *is*
   the existing Portfolio FFT. This recovers the original §3.2 idea and is exact.
   Restrict to constant considerations; reject loss-sensitive ones with a clear
   message.

2. **Loss-sensitive `PortPnL` (hard — later).** Needs per-unit *net* densities
   carried through the combine, or a joint model. Candidates:
   - carry per-unit margin densities via the bivariate / multivariate machinery
     (works for small unit counts; the dependence model must be stated);
   - a sample/simulation path for larger books;
   - a copula over the units' losses, netting per unit then accumulating.

3. **Evaluation.** Book-level `evaluate` (the book's breakeven acceptability
   index) plus each unit's standalone `evaluate` — the book index *and* a
   per-unit acceptability profile from the same objects.

4. **Book-level vs per-unit consideration.** Per-unit considerations are the
   cheap path. A single **book-level** consideration split back to units is an
   allocation problem (reuse the portfolio allocation machinery) — defer.

---

## Likely shape

- Class `PortPnL` (mirrors the `Aggregate`/`Portfolio` pair; `pnl` stays the only
  `pnl`-family keyword). Constructed from a set of `PnL` units or
  `port.make_pnl(considerations={unit: …})`.
- v1: **constant considerations only** → reuse the Portfolio FFT + stacked signed
  summary + total. Loss-sensitive and book-level allocation deferred.
- Reporting mirrors `Portfolio.summary_df` (per-unit blocks + total), in the
  signed P&L convention.

## Open questions

Construction surface (build a `port` of `pnl`s vs `port.make_pnl(...)`); whether
to lift the current `pnl`-in-`port` parse-time rejection to a constant-only
acceptance; how GCN units compose in a book; the dependence model for the
loss-sensitive case.
