# [First-Step-Label]

**Executed at `1.0.0a191`.** One author correction plus one design question
resolved in flight; see *What differed* at the foot.

## Context

Given

```
xpnl Deal as "ABC" 33333.33 premium as "XYZ" less
  agg Deal_e as "LLL" ...
```

three labels are declared and only two of them reached the sheet. `XYZ` named
the consideration leg and `LLL` named the loss leg, both correct, but the
**first `Step`** read `LLL` as well, taking the engine's label. That put the
subject business's name in the step column and left the deal's own name, `ABC`,
nowhere at all. The step column is the one place a walk names *what you are
looking at*, so it should carry the P&L's label.

Alongside it, `a189` had given the direct block's margin row the fixed word
`Direct`. That reads fine on an unlabelled sheet and badly on a labelled one:
the block is already named for the business on its loss leg, and the margin is
that business's margin.

## What landed

* **The first `Step` is the P&L's own `as` label**, defaulting to `'Gross'`.
  Six builders construct a first `sell` group and all six now agree:
  `build_plain_pnl` (walk face), `build_xpnl_walk`, `_peel_per_atom`,
  `_peel_stitched`, `_build_variable_walk`, `_build_variable_walk_occ` and
  `build_reinstatement_pnl`.
* **The direct block's margin row takes the engine's `as` label**, default
  `'Gross'`, carried by the new `Group.margin_label` and threaded through the
  builders as `margin_key`, beside the existing `prem_key` / `loss_key`.
* **The default consideration leg label is `'Premium'`**, matching `'Loss'`.

`_side_index` reads `Group.margin_label` only on the direct block of a ledger
that contains a `buy` group, which is the `a189` gate unchanged. A plain
single-group `pnl` and a ledger merging two sold books keep `Total` throughout.

## The raw label, not the resolved one

`underwriter.py` passed `label=inner.label` to the builders. That is the
**resolved** label, which falls back to the object's name, so an unlabelled
`xpnl bare ...` would have produced a first step called `bare` rather than
`Gross`. The call sites now pass `label=inner._label`, the raw `as` clause,
`None` when absent. Nothing else changes: `PnL` resolves `None` straight back to
its own name, so the display label is what it always was.

Worth remembering as a general trap. `LabeledMixin` deliberately erases the
difference between "unlabelled" and "labelled with the name" at the `.label`
property; any caller that needs to *branch* on whether a label was given has to
read `_label`.

## What differed

**The literal analogy would not have worked.** The author's instinct was that
the margin row should read `LLL` "by analogy with the loss side". Taken
literally that means the margin label is the loss *key*, whose default is
`'Loss'`, so an unlabelled sheet would have read `(Gross, Margin, Loss)`: the
margin row announcing itself as a loss. Raised before implementing; the author
chose the loss *label* with `'Gross'` as its own default, which keeps the
analogy where it is meaningful and drops it where it is not.

**Two test filters were exposed as fragile, for the second time.** Filters of
the form `Label not in ('Total', 'Direct', 'Net', 'Impact')` worked only while
the derived labels were a fixed vocabulary. Now that the direct margin can be
*any* declared label, and can therefore equal a leg's label, an exclusion list
cannot separate legs from derived rows. Both sites are now structural:
`Side != 'Margin' and Label not in ('Total', 'Net')`. Any exhibit picking out
declared legs should use `legs_df` or that shape, never a label list.

**Two over-broad renames, caught by the suite.** `'premium'` is three different
things: a leg label (moved to `'Premium'`), a `Leg.kind` enum value, and an
expense-basis string. The latter two are lowercase and unchanged. A blanket
replace hit all three, which `test_pnl_expenses` and `test_pnl_peel` caught
immediately.

## Verification

Tier 2 and tier 3 clean apart from the three pre-existing `library.agg` and
cookbook failures carried in from `d7c96d9 Cookbook work`.

`test_acceptance_walk_steps` now pins both halves: an unlabelled `xpnl` steps
from `'Gross'` while the engine's label names the direct loss leg *and* its
margin, and a labelled one takes its own label as the first step.
`test_margin_label_names_the_direct_block` pins `Group.margin_label` directly,
and `test_direct_block_and_net_need_something_bought` pins that it is ignored
when the ledger buys nothing.
