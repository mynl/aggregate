# `[Tier-Subtotal-Rows]` — the whole occurrence and aggregate program on a peeled walk

Landed `1.0.0a184`, 2026-07-30. Companion to `[Layer-Peeling-Shorthand]`
(`a183`, `dev/done/plan-layer-peeling.md`) and to the pending
`[Peel-Aggregate-Tier-Only]`.

## Context

`xpnl ... peel top-down` books one group per reinsurance layer. That gained the
per-layer detail and lost the tier lines the plain walk carried: a five-layer
occurrence tower had no row for the program as a whole. The author asked for
those back, positioned after the last layer of each tier.

## Design

A tier that peels into **two or more** steps gains a three-row block after the
last step it spans; a tier that peels into one step already is its own subtotal
and gets nothing. That rule mirrors `group_total`'s "only if more than one leg"
and is what keeps the plain tier walk and a one-layer-per-tier peel byte
identical to `a183`.

`_ledger_plan(groups, result_name, tier_spans=())` takes `(label, lo, hi)`
triples, `hi` exclusive, and emits:

| row label | kind | payload |
|---|---|---|
| `<label> total consideration` | `'tier_total'` | `(lo, hi, 'cons')` |
| `<label> total obligation` | `'tier_total'` | `(lo, hi, 'obl')` |
| `<label> result` | `'tier_result'` | `(lo, hi)` |

Payloads are tuples because they key `PnL._by_kind`.

**Step labels are `'All occurrence'` / `'All aggregate'`**, echoing the grand
`'All'` step. Deliberately *not* `_tier_label`, which prefers the first
*declared* layer label and so would have collided with that layer's own step.
`Line` reads `'Total'`, like the group and grand totals the block sits between,
so exhibits filtering `Line` on `'Total'` already treat it correctly and the
existing footing-test filters needed no change.

Spans come from `build_xpnl_peel`, where the tier split already existed as
`occ_steps` / `agg_steps`. Groups run `[gross] + occ covers + agg covers`, so the
spans are the cover ranges offset by the one gross group.

## The three routes

- **per-atom**: a subtotal is the `grand_total` branch restricted to `egs[lo:hi]`
  (and `tier_result` the span-restricted sum of `result_values`). A partial sum
  over the same atoms, so the κ ladder, footing, GD and exact moments all follow.
  Zero new math.
- **massive**: a span-restricted comprehension over `leg_entries`, the same shape
  as the existing `running_net` branch. One extra sweep key, not a second sweep.
- **stitched**: nothing in the kernel, which is keyed by row label. The builder
  needs the tier's **whole** cession as its own marginal, because summing the
  per-layer recovery vectors would add densities rather than variables, and the
  tier recovery is not an affine of any cumulative net already held. One further
  FFT for the occurrence tier (`_sev_transform_marginal` on the cumulative ceder
  captured from its last step) and no FFT for the aggregate tier
  (`_agg_transform_marginal`, a pushforward of the net-of-occurrence aggregate).

`_peel_stitched` re-derives the plan independently of `PnL.__init__`, so the same
`tier_spans` goes to both; a mismatch surfaces as the "no entry supplied for
ledger row" error rather than silently drifting.

## Three latent kernel problems fixed while in there

1. `_assemble_rows`, `_init_massive` and `_view_index` each closed their row-kind
   dispatch with a bare `else` that *meant* `'total_impact'`, so an unhandled
   kind was silently booked as the impact row. All three now name
   `'total_impact'` explicitly and raise on anything unknown.
2. `PnL.__add__` reconstructed from `_group_specs` / `_scale_arg` /
   `result_name` only, so any new constructor argument silently dropped on
   composition. It now carries `tier_spans`, shifting the right-hand operand's
   spans by the left-hand group count, since `+` renumbers groups.
3. `density_df` carried group results but would have skipped the tier results;
   it now includes them, so they are plottable.

## Verified

- **Cross-path anchor:** on a two-layer occurrence tower the peel's
  `All occurrence` block (summed from its own per-layer rows) and the plain
  walk's `ceded occ` step (one lumped group off the occurrence joint) agree to
  **1.6e-7** in consideration, obligation and margin. The walk is the looser of
  the two, riding the coarser 2-D joint.
- Per-atom route: all nine κ columns plus `EX` foot at every step, subtotals
  included.
- Two layers in each tier: both blocks appear, each after its own last layer,
  and `Gross + All occurrence + All aggregate == All` in margin (relative
  tolerance, since those four rows ride four independently computed marginals
  and linearity holds to the rebucketing's first-moment accuracy, measured
  ~1.7e-10 relative).
- One layer per tier: `stats_df` **and** `summary_df` byte identical to the
  plain tier walk, both directions.
- `tests/test_pnl_peel.py` 40 → 53 cases; `decl-testers.agg` section AI gains
  `AI.Both`.
