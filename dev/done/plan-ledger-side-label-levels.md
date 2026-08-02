# [Ledger-Side-Label-Levels]

**Executed at `1.0.0a189`.** One thing differed from the plan as written; see
*What differed* at the foot.

## Context

The `PnL` sheets named their row levels `View` and `Line`. Both were wrong in
small ways that compound when you read a tower.

`Line` assumes a line of business. The ledger makes no such assumption anywhere
else, and the level does not hold lines: it holds presentation labels, which
are declared leg names on a leg row and `Total` on a derived one.

`View` was vaguer still, and worse, `Total` underneath it was doing three
different jobs at once: a within-step subtotal of several legs, the direct
result before any purchase, and the grand result after all of them. On a tower
you read `Total` eight times and had to work out from the step which sense was
meant each time.

## Decisions

**`Side`, not `Leg`.** `Leg` is the public class for an individual declared
cash flow (`GWP`, `LAE`, `Occ1 recovery`), and under this change those all move
to the `Label` level. A level named `Leg` would therefore be the one level in
the frame containing no leg names. `side` was already the internal word
(`_ledger_plan` and `_side_index` pass `side='cons'|'obl'`), so this promotes an
existing term and widens it to cover `Margin` rather than inventing one.

**`summary_df` renames to match.** Its level carries `Net` and `Impact` as well
as the three sides, so it is really a merge of the two `stats_df` levels. Naming
it the same way anyway beats inventing a second word for a second frame.

**`Direct` and `Net` are gated on the ledger containing a `buy` group.** One
condition governs the whole rename. A plain single-group `pnl` is one `sell`
group whose legs are already net (`GWP (net)`, `Subject (net)`), and a ledger
merging two sold books has no net to take. Both keep `Total` throughout, since
there is nothing for either to be direct *of*.

## What landed

Level names, in `src/aggregate/_pnl.py`:

| frame | before | after |
|---|---|---|
| `stats_df`, single group | `(View, Line)` | `(Side, Label)` |
| `stats_df`, tower | `(Step, View, Line)` | `(Step, Side, Label)` |
| `summary_df` | `View` / `(Step, View)` | `Side` / `(Step, Side)` |
| `legs_df` columns | `Step, View, Line, ...` | `Step, Side, Label, ...` |

Row labels, when the ledger contains a `buy` group:

| row kind | before | after |
|---|---|---|
| `group_result`, `sell` group | `Total` | `Direct` |
| `group_result`, `buy` group | `Total` | `Total` |
| `grand_total` (both sides) | `Total` | `Net` |
| `grand_result` | `Total` | `Net` |
| `running_net`, `total_impact`, `group_total`, `tier_total`, `tier_result` | unchanged | unchanged |

`_view_index` became `_side_index` and `_VIEW_DEFAULTS` became
`_SIDE_DEFAULTS`, both private, both with the gate documented in place.

## Reading the target

The two-tier peel program now renders exactly as the author specified:

```
Step            Side           Label
Subject         Consideration  GWP
                Obligation     Subject / LAE / Fixed Exp / Acq Exp / Total
                Margin         Direct
Occ2            Consideration  Occ2 premium
                Obligation     Occ2 recovery / Occ2 commission / Total
                Margin         Total
                               Net
All occurrence  Consideration  Total
                Obligation     Total
                Margin         Total
All             Consideration  Net
                Obligation     Net
                Margin         Net
                               Impact
```

## Verification

Roughly 170 sites across twelve test files, plus `docs/cookbook/_06_00_pnl.qmd`,
`docs/cookbook/_06_02_walk.qmd` and `dev/FEATURES.csv`. Tier 2 and tier 3 clean
apart from the three pre-existing `library.agg` and cookbook failures carried in
from `d7c96d9 Cookbook work`, none of which touch a file this change edits.

New case: `test_direct_and_net_need_something_bought` pins the gate on a
hand-built two-`sell`-group ledger and on the single-group `pnl`.

## What differed

The plan said a step's own result could be looked up at
`(step, 'Margin', 'Total')`. After the change that key varies by ledger
position, which broke `test_pnl_peel.py::_assert_column_foots`, a helper that
foots every step. The fix is a small helper, `_step_result`, taking the **first
`Margin` row of the step's block in plan order**. That is uniformly the step's
own result: `Direct` on the direct step, `Total` on a cession, `Net` on the
grand block, always ahead of the running net or impact that may follow it.
Worth knowing for any exhibit doing the same thing.

Two leg filters also needed `'Direct'` adding to their exclusion set
(`test_composition_matrix.py`, `test_pnl_consolidated_walk.py`); without it a
`Direct` row was summed as though it were a declared leg.

## Not in scope

`'All occurrence'` and `'All aggregate'` remain domain-specific step labels,
accepted for v1.0 per the author.
