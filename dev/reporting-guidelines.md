# Reporting guidelines (DecL objects: PnL, Aggregate, analyses)

> **Status: DRAFT direction (author, 2026-06-29).** The target we move toward as
> the PnL workflow settles. Captured here so plans can reference it; we are **not**
> retrofitting every report at once — see each plan's "pended" scope.

1. **A report's ROWS are fixed.** A report does not morph as the object picks up
   other properties. `pnl.summary_df` is one thing whether or not the pnl carries
   reinsurance (today it wrongly becomes `gcn_df` when reins is added — that is the
   first fix). **Columns** may change — e.g. `stats_df` / `reins_*` gain columns by
   reinsurance flavor — rows do not.
2. **Columns are pure: one unit per column.** Never mix currency rows (premium,
   loss) with ratio rows (LR, CR) under a single column — that breaks formatting and
   is just untidy. Ratios live in their own columns / tables (tidy data).
3. **Presentation-ready.** Capitalized row / column headings and index names.
4. **`summary`** — short, user-facing "what *is* this object?".
5. **`validation`** — short, user-facing "is this object calculating correctly?"
   (this is the *old* `summary`). Typical fields: `EX Est, EX, Diff, CV Est, CV,
   Diff, Sk Est, Sk`.
6. **`reins_<flavor>`** — reinsurance views, present **only** when reinsurance is
   present; split into occurrence / aggregate / total; shows the impact of the
   cession.
