# Reporting guidelines (DecL objects: PnL, Aggregate, analyses)

> **Status: DRAFT direction (author, 2026-06-29).** The target we move toward as
> the PnL workflow settles. Captured here so plans can reference it; we are **not**
> retrofitting every report at once — see each plan's "pended" scope.
>
> **Half of this has landed.** §0, what a first-class citizen is, shipped at
> `1.0.0a170` and is now declared in code; §0 below is a pointer to where. What
> is still open is §"The reports themselves", what those reports *contain*,
> tracked as `[Reporting-Guidelines]` in `dev/TODO.md` and gating
> `[Accounting-Summary-DF]`. This file moves to `dev/done/` when that lands.

## 0. What a first-class citizen is

**Settled, and declared in code. This section is a pointer, not a
specification.** The membership rule and the required surface landed at
`1.0.0a170` (`[FCC-Contract]`), and each part of the contract lives in exactly
one place:

| What | Where |
|---|---|
| Who is first-class, and who is near-first-class | `FIRST_CLASS_CLASSES` / `NEAR_FIRST_CLASS` in `src/aggregate/constants.py` |
| The two membership criteria, the `Severity` exemption, the optional-member and narrative-pairs rules | the comment block immediately above those tuples, same file |
| Every member an FCC must carry | `FCC_REQUIRED`, same file, grouped in its own comment |
| Declared temporary holes | `FCC_CONTRACT_EXCEPTIONS` / `FCC_UNPAIRED_NARRATIVES`, same file: both empty since `1.0.0a172`, and they must stay empty at `1.0.0b1` |
| The audit | `dev/regen_features.py`, its FCC CONTRACT and NARRATIVE PAIRS sections |
| The assertion | `tests/test_fcc_surface.py` |
| The user-facing writeup | "The first-class-citizen contract" in `docs/2_aggregate_overview/features.rst` |

Read the tuples and that comment block, never a prose copy of them. This section
held such a copy until `1.0.0a357`, and it had already drifted: it still listed
`doc` as a required trailer value more than fifty releases after `1.0.0a301`
dropped `doc` from `FCC_REQUIRED`. Nothing it said was unique, which is why it is
gone rather than corrected.

What remains live in this file is the section below.

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
