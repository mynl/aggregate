# `[PnL-Ratio-Frame]` — retire `Scaled`, add `ratio_df` and `legs_df`

Landed `1.0.0a185`, 2026-07-30. Follows `[Tier-Subtotal-Rows]` (`a184`), whose
subtotal blocks become rows of `ratio_df`.

## Context

The author wanted four things off a P&L: per-block loss and expense ratios,
premium as a share of gross premium, and margin as a share of gross margin. They
suspected one column could not carry all of it, and pointed at the `pricing_df`
pattern from `analyze_distortions`: a frame of raw materials from which
presentation tables are built, usually by selecting one metric and unstacking,
with `summary_df` as the presentation-ready layer.

Measurement showed worse than "trying to do too much". `Scaled` divided every
cell by one committed number, `E[grand total consideration]`, which on a walk is
gross premium **minus every cession**. On a two-group tower (premium 100, loss
40, ceded premium 30, recovery 8) the gross premium cell read **1.43** and the
gross "loss ratio" read **-0.571** against a true 0.40. None of the four
requested readings appeared anywhere.

The combined-ratio reading was an accident of the single-group case, where the
divisor happens to coincide with that block's own premium. The original design
(`dev/done/plan-pnl-exhibits.md:77-83`) called the column *"just a unitless,
comparable number"*, explicitly not an accounting ratio.

`[Reporting-Guidelines]` settled the direction: rule 1 permits column changes
(*"**Columns** may change ... rows do not"*), and rule 2 both forbade the
multi-purpose `Scaled` the author was imagining and licensed the replacement:
*"Columns are pure: one unit per column ... Ratios live in their own columns /
tables (tidy data)."*

## Decisions taken

`Scaled` dropped entirely, `E_consideration` kept. Both ratio readings reported
as scalar columns. `Leg` gains a `kind` with `stats_df` unchanged. `ratio_df` a
`PnL` property.

## What landed

### `Scaled` and its machinery, removed

`_CARD_COLS` loses `'Scaled'`; `_SCALE_INVARIANT_STATS`, `scaled_stats_df`, the
`scale` property, `_resolve_scale`, the `_scale_*` attributes and the `scale=`
constructor argument all go, along with the reinstatement builders' two explicit
`scale=float(P_G - D - pc)` arguments. The `info` block's `scale` row becomes
`E[consideration]`.

**`E_consideration` deliberately kept**, against the letter of the option the
author picked. It is `E[grand total consideration]`, exactly the `P` denominator
`ratio_df` needs, and `evaluate` calibrates its distortion set to it as the
premium target. It is an accessor, not scaling machinery.

### `Leg.kind`

Validated against the new module constant `LEG_KINDS` (`'premium'`, `'loss'`,
`'expense'`, `'recovery'`, `'commission'`), default `None`. Set at all 63 `Leg`
call sites across the builders, each determined by the leg's label. This retains
what `_resolve_expense_split` already knew and then threw away, and it replaces
string-matching on leg labels, which was the only alternative: nothing in
`stats_df` distinguishes `'LAE'` from `'Loss'` but the text.

`stats_df` gains no index level, so no existing exhibit changes shape. An
unclassified leg folds into `L` as the residual, documented, so a hand-built
ledger that declares no expense legs reports `E = 0` rather than guessing.

### `ratio_df`

Indexed by `Step`, one row per block: each group, each tier subtotal, `All`.
Columns `P L E C M LR ER CR E_LR E_ER E_CR P_share M_share`.

**The sign convention is the design's keystone.** The amounts are signed *in the
gross direction*: consideration as booked, obligations negated. So a cession's
ceded premium and recovery are both negative. That single choice buys three
properties at once:

1. **Amounts add across blocks**, so layers sum into their tier and tiers into
   `All`.
2. **`M == P - L - E - C` holds identically**, because it *is* the signed row
   sum. `1 - CR == M / P` is the same identity read the other way.
3. **Every ratio keeps its conventional sign**, because numerator and
   denominator flip together. A cover that paid back three times its premium
   reads `LR = 3.10`, not `-3.10`.

Ratios are re-derived from each row's own amounts, never averaged from the blocks
below, following the `pricing_df` rule. `L`, `M`, `P` and `LR` keep the
`PENTAGON_STATS` spelling so the frame concatenates and diffs against a pricing
frame; `E` / `C` / `ER` / `CR` extend it, and `Q` / `a` / `PQ` / `ROE` have no
meaning on a ledger.

### The two loss ratios

The author's caution drove this and it was right: when premium is random and
correlated with loss, `E[L/P] != E[L]/E[P]`, so neither can be labelled "the"
loss ratio. Both are reported, `LR` for the ratio of means (the `pricing_df` and
rate-filing convention) and `E_LR` for the mean of the ratio.

Availability turns on whether the **denominator** is random, not on whether a
joint exists. A constant premium factors out of `E[X / P]`, so `E_*` is exact on
every route and repeats the plain ratio; a random premium needs atoms to average
over, so it is `nan` on a route with none, and `nan` where some atom carrying
probability has a vanishing premium. Never a silent fallback to the ratio of the
means.

> **Corrected in `a186`.** As shipped at `a185` this gated on the joint rather
> than the denominator, and the columns were named `EX_*`. That blanked the
> whole frame on a stitched peel, where premium is in fact always constant
> (peeling is guaranteed-cost by construction), so the numbers were always
> exactly computable. See the `a186` CHANGELOG entry.

### `legs_df`

One row per **declared** leg: `Step` / `View` / `Line` / `kind` / `EX` / `SD`.
Derived rows absent by design, being sums of these. The only place `Leg.kind`
surfaces, and the frame to pivot when the wanted ratio is not one `ratio_df`
carries.

### Raw materials, not exhibits

Per the author (2026-07-30), `pricing_df` is raw *materials*, from which
presentation tables are made; `summary_df` is the presentation layer. So both new
frames are unformatted and deliberately absent from `qd` and `_repr_html_`, which
keep rendering `summary_df`. `ratio_df.T` gives the stat-down-the-side
orientation, matching the documented `pricing_df.T` convention.

## Verified

- `M == P - L - E - C` on every block of a two-tier peeled walk, and the `All`
  row's amounts equal the column-wise sums of the blocks.
- `All` `LR` equals `sum(L) / sum(P)` and is **not** the average of the block
  loss ratios.
- Retro-rated premium: `LR = 0.2426` against `E_LR = 0.2256`, a 7% relative
  gap, `E_LR < LR` as the retro implies (premium rises with loss, damping the
  per-atom ratio). Deterministic premium: the two agree to `rel=1e-12`.
- Stitched peel: every `LR` / `CR` cell live. (At `a185` the `E_*` cells were
  `nan` here; `a186` makes them live and exactly equal to the plain ratios, the
  peel premium being constant.)
- A cession block reads `P < 0`, `L < 0`, `LR > 0`.
- `P_share == M_share == 1` on the gross block.
- Every library-built leg is classified; `LAE` reads `'expense'` and the gross
  block's `CR == LR + ER`.
- `Scaled` / `scaled_stats_df` / `_resolve_scale` absent from `src`, `tests` and
  `dev`; `E_consideration` present.

Test churn: `tests/test_create_pnl.py` (the whole `scale` section replaced),
`tests/test_pnl.py`, `tests/test_pnl_expenses.py`, `tests/test_fcc_surface.py`
(via the `info` row), plus new cases in `tests/test_pnl_peel.py` (53 to 61) and
`tests/test_variable_rating_decl.py`. Four `dev/FEATURES.csv` rows rewritten.
