# [Evaluate-Positions-Only]

**Executed at `1.0.0a190`.** Two author corrections to the `a188` panel, both
about what a ledger row *is*.

## `total impact` is not a position

The impact row is the grand result less the first group's, so it measures what
the ledger's purchases did to the bottom line. It is therefore a **difference
between two positions, not a position**. Nobody holds it, so "what stress does
it survive" is not a question with an answer, and an acceptability index for it
reads as though it were one.

`a188` gave it a role and evaluated it where it had a law. That was wrong on
the merits, and the stitched-peel case (where it has no law, being a delta of
two statistics whose sides ride different marginals) was a symptom rather than
the whole story: even the unpeeled walk, where the row does carry a
`GridDistribution`, was answering a question nobody asked.

It is now excluded outright. `_MARGIN_KINDS` drops `'total_impact'`, and the
constant's comment carries the reason so the next reader does not add it back.

The ceded program **as a position** is unaffected. It is already in the sheet,
under the tier subtotal rows (`All occurrence result`, `All aggregate result`),
which are genuine positions someone holds.

`total impact` remains an ordinary ledger row everywhere else: `stats_df`,
`summary_df`, `density_df`, the footing checks. Only `evaluate` skips it.

### The dead branch

`total_impact` was the only ledger row any builder ever declares as
`('delta', ...)` (`_pnl_builders.py:1184`), so excluding it made the panel's
`_DeltaRow` branch unreachable and `_pricing.no_distribution_panel` unused.
Both deleted rather than left as speculative machinery; git history has them if
a future row kind ever needs the shape. `_DeltaRow` itself stays, since it is
still how the stitched builder materializes the impact row for the stats sheet.

## A netted row is neither side

A running net and the grand result net buying against selling. That is not
`sell`, which is what `a188` called them: a book written and a book written
minus the cover bought against it are different kinds of thing, and the panel
should say so.

`EVAL_SIGN` gains `'net': 1.0`. The arithmetic is unchanged, since a netted
margin is already in payoff orientation and reads as booked, exactly as `sell`
does. What changes is that the column now names three kinds of position rather
than folding two of them together.

| row kind | role |
|---|---|
| `group_result`, `sell` group | `sell` |
| `group_result`, `buy` group | `buy` |
| `tier_result` | its span's shared role, or `net` when the span mixes |
| `running_net`, `grand_result` | `net` |

The mixed-span case fell out of the rewrite: `a188` returned `sell` for a span
covering both roles, which has the same defect. A span that mixes is a netting,
so it reads `net`. It cannot arise from a builder, but the rule is now stated
rather than defaulted.

`Aggregate.evaluate` and `Portfolio.evaluate` still report `sell`: an aggregate
is an obligation written, not a netted ledger.

## Reading the panel now

```
Step                    role      ph
Subject result          sell  0.3285      the book written
Occ2 result              buy  0.5857      a layer, priced from the seller's side
net through Occ2         net  0.2547      what buying it left you holding
...
margin                   net  0.1609      the bottom line
```

Three roles, three kinds of position, and every row is one.

## Verification

Tier 2 and tier 3 clean apart from the three pre-existing `library.agg` and
cookbook failures carried in from `d7c96d9 Cookbook work`.

`test_evaluate_flips_the_impact_row_on_an_unpeeled_walk` became
`test_evaluate_reports_positions_only_never_the_impact`, which asserts the row
is absent on **both** routes (peeled and not) while remaining a ledger row in
`p._rows`, so the exclusion is about what the row means and not about whether
it happens to have a law. The peel test's warning match moved from `'no joint'`
to `E\[M\]`, that message having been the impact row's.
