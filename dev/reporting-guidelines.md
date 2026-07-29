# Reporting guidelines (DecL objects: PnL, Aggregate, analyses)

> **Status: DRAFT direction (author, 2026-06-29).** The target we move toward as
> the PnL workflow settles. Captured here so plans can reference it; we are **not**
> retrofitting every report at once — see each plan's "pended" scope.

## 0. What a first-class citizen is

*(Author, 2026-07-29. Declared in code as `FIRST_CLASS_CLASSES` / `FCC_REQUIRED`
in `src/aggregate/constants.py`, audited by `dev/regen_features.py`, asserted by
`tests/test_fcc_surface.py`. Change the tuple, not this list, and the check
follows.)*

The **first-class citizens (FCC)** are `Aggregate`, `Portfolio`,
`BivariateAggregate`, `PnL` and `Distortion`. `Severity` is near-first-class.

Two criteria put a class on the list, and both must hold:

1. **It can be created in DecL.** `Bounds`, `Frequency` and `GridDistribution`
   are reached *from* an object and never declared, so they are out however
   useful they are.
2. **It flows through to the `aggregate_api` (aLL) SPA**, which calls exactly the
   members below on whatever object it is handed.

An FCC **must** carry:

| Group | Members |
|---|---|
| Discovery and identity | `info`, `help` |
| DecL trailer | `note`, `hints`, `tags`, `doc` |
| Declaration round-trip | `program`, `pprogram` |
| DataFrame quartet | `summary_df`, `validation_df`, `stats_df`, `density_df` |
| Graphics | `plot` |

`Severity` carries all of it except the DataFrame quartet: it is a look-through
onto a frozen scipy random variable, not a compute result, so the frames have
nothing to report.

Everything else is **optional**, and a caller reaches it defensively with
`getattr`. The `*_description` (short) / `*_explanation` (long) narrative strings
are the main such family. Optional does not mean unconstrained: wherever one half
of a pair is present the other must be too. `FCC_UNPAIRED_NARRATIVES` names the
stems still short a half, and it empties with `[FCC-Contract-Gaps]`.

## The reports themselves

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
