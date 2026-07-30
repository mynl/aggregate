# `[Layer-Peeling-Shorthand]` — layer-by-layer `xpnl`

Landed `1.0.0a183`, 2026-07-30. Carries `[Walk-Step-Default-Labels]` with it.
The precursor bug fix `[Ceder-Gap-Knot]` landed separately as `1.0.0a182`.

## Context

`xpnl` exploded a guaranteed-cost program into **tiers**, not layers: at most
three groups, gross to one lumped occurrence cover to one lumped aggregate
cover. A multi-layer tower

```
occurrence net of 100 xs 100 and 300 xs 200
```

collapsed both layers into a single `ceded occ` group, so the ceded premium,
commission and marginal impact of each layer were invisible. The ask: introduce
layers one at a time, in a specified order, one group per layer.

This was the project's own deferred item, recorded in `dev/done/plan-yapnl.md`:
*"a DecL shorthand adding reins layers one at a time (top-down / bottom-up), one
group per layer, each group's result the layer's marginal impact."*

## Design decisions

**A nullable DecL clause on the existing statement, not a `build` kwarg and not
new object kinds.** `build`'s `**kwargs` are contractually `update()` kwargs, so
`pnl_order=` would have needed new plumbing and, worse, would have been
invisible in the program: `recipe.decl` would print something that does not
reproduce the object, and the round-trip gate fails on spec key-set difference.
New `xtdpnl` / `xbupnl` kinds would put a presentation option into the object
kind (`pnl` vs `xpnl` is a real kind distinction), need a third keyword for the
deferred explicit-order form, and cost six hand-maintained mirrors each. Peeling
is a recipe variant, not a new type (`[Decision-XPnL-Is-A-Recipe]`), so it is a
modifier on `xpnl`.

**Keyword `peel`**, matching the pre-existing `[Layer-Peeling-Shorthand]` label.
`onion` was the author's coinage but is overloaded in project history (the
retired `[Xpnl-Onion-2D]` label named a different feature) and would have been a
synonym for a concept that already had a name.

**Both tiers, occurrence on the marginal route.** See below.

**The explicit-tuple order is deferred**, tracked as
`[Peel-Aggregate-Tier-Only]`. Direction-only peeling walks prefixes and suffixes
of the stored layer list, which never manufacture a gap; an arbitrary
permutation does, which is what made `[Ceder-Gap-Knot]` a prerequisite for it.

## The asymmetry that shaped the implementation

**Aggregate layers decompose per-atom, exactly.** They are disjoint intervals on
the *same* aggregate subject axis, so
`ceder([L1..Lk])(x) == sum_i ceder([Li])(x)` identically. Each layer's
incremental recovery is therefore just `ceder([Li])` as an ordinary `Leg` on the
one shared source. The scenario (κ) ladder survives and every column foots.

**Occurrence layers do not.** The ceded-occurrence aggregate is not a function
of the gross aggregate, because the random claim count decouples them, which is
why the tier walk reaches for the 2-D `occ_bivariate`. Peeling `m` occurrence
layers per-atom needs an `(m+1)`-axis joint: `[One-2D-Source]` forbids it and
three axes at `log2=16` is 2^48 cells. Nor can the total ceded axis be split
after the fact, because `sum_i ceder(X_i)` does not determine the per-layer
allocation.

But the caveat is much softer than "not exact". Every ledger row of an
occurrence peel is the compound of a **deterministic per-claim severity
transform**, so each row's marginal is exactly one FFT of a transformed
severity:

| row | transform | shift |
|---|---|---|
| `Loss` | `x` | 0, sign −1 |
| `<layer i> recovery` | `ceder([Li])(x)` | 0, sign +1 |
| `net through <layer k>` | `x - ceder(cumulative set k)(x)` | premium less expenses plus running commissions, sign −1 |

The aggregate tier then rides the net-of-occurrence aggregate as a plain
pushforward (no FFT). So the marginals are exact, `E[ceded through k]` foots
exactly by linearity, and only the dispersion columns are marginal. That is
precisely the kernel's `stitched_rows` seam, which `a141` kept as "the designated
no-joint assembly seam"; this is its second consumer.

Cost: `2m` one-dimensional FFTs on the existing grid, against the 2 that
`reins_density_df` already pays and the `m` that `reins_stats_df` already pays.

## Route selection

Stitched iff **two or more** occurrence layers are peeled. With at most one, the
joint's ceded axis already *is* that layer, so nothing has to be split and the
per-atom route applies. That makes a one-layer-per-tier program reproduce the
tier walk byte for byte in either direction, which is the suite's primary
regression anchor.

Consequence, documented in `construction_explanation`: the route is per-object,
so peeling occurrence layers on a program that also carries aggregate layers
pulls the aggregate steps onto the marginal ladder too. That is
`[Peel-Aggregate-Tier-Only]`.

## What landed

| File | Change |
|---|---|
| `src/aggregate/decl.lark` | `peel_clause` (nullable, `approx_clause` shape) on both `pnl_out` and `xpnl_out`, so `pnl ... peel` is a semantic error not a parse error; `PEEL.2` terminal; `peel` in the `ID` reserved list; comment block rewritten |
| `src/aggregate/parser.py` | `_PEEL_DIRECTIONS`, `peel_set` (validated), `peel_none` returning `{}` so unpeeled programs are untouched, `_pnl_spec` slot |
| `src/aggregate/parser_errors.py`, `decl_pygments.py`, `agg.sublime-syntax` | keyword mirrors |
| `src/aggregate/decl_writer.py` | `_render_peel` in the `_render_approx` shape, wired into `_render_pnl` |
| `src/aggregate/underwriter.py` | `_resolve_reins_economics` keeps `pc_*_by_layer` / `c_*_by_layer` beside the untouched totals; `_factory` pops `peel` and rejects `pnl`, a port engine, and every non-`gcn` recipe with a message naming the reason; recipe key; `_snapshot_pnl` dispatch |
| `src/aggregate/_pnl_builders.py` | `build_xpnl_peel` plus `_peel_per_atom` / `_peel_stitched` / `_peel_explanation`; helpers `_peel_steps`, `_peel_side_economics`, `_affine_row`, `_const_row`, `_sev_transform_marginal`, `_agg_transform_marginal`, `_layer_descriptor`, `_reins_layer_label`, `_tier_label` |
| `tests/test_pnl_peel.py` | 40 cases: the degenerate anchor, both routes, footing, engine agreement, labels, gap fillers, every rejection, round-trip |
| `src/aggregate/agg/decl-testers.agg` | section AI, seven programs |

`src/aggregate/_pnl.py` needed no structural change: `_ledger_plan` was already
m-group generic.

## Why the two items are coupled

`_ledger_plan` raises on duplicate row labels. The undeclared cover-step fallback
was the single string `'ceded occ'`, so `m` peeled layers would have produced `m`
identical labels and a hard `ValueError`. Peeling therefore could not ship until
undeclared steps got distinct per-layer labels, which picked the answer the TODO
had already proposed: the declared layer label if present (already pooled into
`label_map` as `{layer_index: label}` since `a132`), else the DecL layer
descriptor, reusing the writer's own `_render_reins_clause`. Descriptors cannot
collide because the validator rejects overlapping cessions.

Test churn from the rename landed in the same sweep: `test_decl_labels`,
`test_pnl`, `test_pnl_ceded_premium`, `test_pnl_engine_source`,
`test_reinstatement_decl`, and the new `test_pnl_peel`.

## Verified

- `occ 300 xs 200` / `occ 100 xs 100` peel: every `EX` column foots to 1e-9, and
  the peeled recoveries sum to the engine's own `E[ceded occ]` to 1.25e-10.
- The closing margin matches the consolidated `pnl` to the digit
  (`447.646754`), and is order-invariant between the two directions. The lumped
  tier walk reads `447.646817`, slightly off because it rides the coarser 2-D
  joint; the peel is the more accurate figure.
- Aggregate-only peel: all nine κ columns plus `EX` foot at every step.
- Mixed tiers: the peeled aggregate recovery matches the engine's `p_agg_ceded`
  to 6.45e-12 and the implied final net matches `p_agg_net` exactly.
- One layer per tier: `stats_df` identical to the tier walk, both directions.
