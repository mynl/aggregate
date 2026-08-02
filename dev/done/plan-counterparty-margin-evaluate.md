# [Counterparty-Margin-Evaluate]

**Executed at `1.0.0a188`.** One thing differed from the plan as written; see
*What differed* at the foot.

## Context

`PnL.evaluate` (a187) solves `rho_g(margin) = 0` on every margin row of the
ledger. On a tower with reinsurance that promise was half empty: every
reinsurance row came back `NaN`.

Measured on a two-tier program (two occurrence layers, two aggregate layers,
peeled top-down), six of the thirteen margin rows reported `E[M] <= 0`:
`Occ1 result`, `Occ2 result`, `Agg1 result`, `Agg2 result`,
`All occurrence result` and `All aggregate result`. That is arithmetically
correct and analytically useless. A cession's margin to the buyer is negative
by construction, because you pay for cover, so the `E[M] > 0` guard fires on
every purchased layer there will ever be.

The question worth asking about a purchased layer is what stress the
**seller's** position survives, and that is the buyer's margin negated.

## The rule

The flip is keyed off `Group.role`, not off a row label:

| row kind | role | why |
|---|---|---|
| `group_result` | the group's own `role` | a cession group is a `buy` |
| `tier_result` `(lo, hi)` | `buy` iff every group in the span is `buy` | a mixed span stays as booked |
| `total_impact` | `buy` iff every group after the first is `buy` | the impact is every cession combined |
| `running_net`, `grand_result` | always `sell` | the holder's own net position |

On any builder-produced ledger a role rule and a label rule agree exactly, since
the direct result is the `sell` group and a margin row labeled `Total` is a
cession. They diverge on a hand-built ledger with two `sell` groups, where a
label rule would wrongly flip the second book. `role` is right by construction
and is already the canonical word.

## What landed

**`src/aggregate/_pricing.py`.** `EVAL_SIGN = {'sell': 1.0, 'buy': -1.0}` and
`_eval_sign(role)`, which raises rather than defaulting so a typo cannot
silently evaluate the wrong side of the trade. `role` leads `EVAL_COLS`,
qualifying the whole row by naming whose position the parameters describe.
`evaluate_margin`, `evaluate_constant_premium` and `no_distribution_panel` all
take `role='sell'`; the negation happens once, at the top of
`_evaluate_margin_arrays`, as `(-x)[::-1], p[::-1]`. Negating reverses the sort
order, so the reversal is required, not cosmetic.

**`src/aggregate/_pnl.py`.** `PnL._row_role(kind, payload)` implements the table
above; `evaluate` passes it through to both the solve and the
no-distribution branch, so a stitched impact row still reports whose position it
was even where there is nothing to solve.

**The two faces.** `Aggregate.evaluate` and `Portfolio.evaluate` needed no code
change: an aggregate is an obligation written, so the default `sell` is right.
Docstrings gained the `role` column.

## What this buys

The panel becomes a buy decision. A layer whose `gini_p` sits above the running
net immediately over it is priced above the holder's own acceptability, so
buying it lowers the net. On the two-tier program (`ph`):

| Step | `gini_p` | role |
|---|---|---|
| `Subject result` | 0.329 | sell |
| `Occ2 result` | 0.586 | buy |
| `net through Occ2` | 0.255 | sell |
| `Occ1 result` | 0.535 | buy |
| `net through Occ1` | 0.172 | sell |
| `Agg2 result` | 0.240 | buy |
| `Agg1 result` | 0.179 | buy |
| `margin` | 0.161 | sell |

`DegenerateEvaluationWarning` becomes rare and meaningful: it fires on a layer
priced below its own expected recovery, which is a finding, rather than on the
routine fact that cover costs money.

## Verification

Tier 2 and tier 3 both clean apart from the three pre-existing `library.agg` and
cookbook failures carried in from `d7c96d9 Cookbook work`
(`test_library_is_written_in_the_canonical_layout`,
`test_committed_pages_are_up_to_date`, `test_recipe_runs[agg:LimitProfile]`),
none of which touch any file this change edits.

New cases in `tests/test_pnl.py`: every ceded layer prices and carries
`role == 'buy'`; the flip equals the negated-array solve cell for cell; a layer
compares against the net above it; the impact row flips on an unpeeled walk;
an unknown `role` raises. `tests/test_distortion_calibrate.py` pins
`role == 'sell'` on both faces. The two-tier program is
`EV.CedeTower` in `src/aggregate/agg/decl-testers.agg`, round-tripped by
`test_grammar_sync`.

## What differed

`tests/test_pnl_peel.py::test_stitched_peel_evaluates_every_layer_but_not_the_impact_row`
was expected to keep passing and did not, for a reason worth recording. Its
`OCC2` program buys `100 xs 100` for a deposit of 60 against an expected
recovery of 123.76, and `300 xs 200` for 40 against 123.89. Those layers are
deliberate bargains, so read from the seller's side they carry `E[M] <= 0`: a
position priced below its own expected loss survives no stress at all. The test
was asserting `status == 'ok'` on the layers, which had been true only because
the *cedent* was the one making money on them.

Renamed to `test_stitched_peel_evaluates_its_nets_and_flags_underpriced_layers`
and rewritten to pin the new truth, which is the better test: it covers the
degenerate-after-flip case that the warning now exists for. The program was left
alone, since several other tests in the file assert on its booked amounts.

## Not in scope

`[Named-Cherny-Madan-Families]` (#23) and `[PnL-Density-DF-Running-Nets]` both
stay pending in `dev/TODO.md`.
