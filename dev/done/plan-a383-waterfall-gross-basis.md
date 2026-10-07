# Plan a383: the waterfall's second capital basis ([Waterfall-Gross-Basis])

Status: draft for review. Target version 1.0.0a383 (current: 1.0.0a382).

## Goal

The margin walk (`PnL.walk_df` / `PnL.evaluation_df`, served by the
`economic_waterfall` exhibit) evaluates every step's capital in one state: the
whole book's net result landing at its 1-in-100. Add a second, gross-anchored
basis alongside it: each step's conditional mean given the **gross** result
landing at its own 1-in-100, plus the cost-of-capital quotient against that
basis. The reader then sees both answers side by side: the return on the
capital the firm actually holds (net basis), and the program's performance in
the gross stress state (gross basis).

## Why (standalone summary of the analysis)

Both bases are exact conditional-mean ladders and both foot down the sheet,
because conditional means are additive under any common conditioning event.
What the choice of basis determines is which row anchors to its own quantile:

* **Net conditioning** (current): the grand-result row's cell is its own 1%
  quantile, so the column decomposes the capital the firm actually holds.
  Right for attribution and performance measurement of the in-force program.
  Blind spot: a cover is evaluated in the states remaining *after* the
  program worked, so a highly effective hedge removes its own states from
  the net tail and looks weak against the residual tail.
* **Gross conditioning** (new): the gross row anchors instead, and the column
  reads as a stress test: in the gross 1-in-100 environment, who pays?
  Right for judging program design against the underlying risk. Blind spot:
  two covers responding to the same gross state both look excellent even
  when jointly redundant.

The motivating example (`BuildRe`, a 4-layer occurrence tower at 100%
placement under `mixed gamma 0.175`) shows the bases can disagree violently:
on the net basis every occurrence layer prices out at an identical
Div CoC of 215.5%, because the net-of-occurrence subject (retained claim
capped at the 250 retention) carries claim-count information only, so every
layer's conditional recovery is the same multiple (1.48808) of its mean. On
a gross basis the conditioning sees the large claims the net gives away, and
the layers differentiate. Showing both is the point.

## The mathematics, and the one blocked cell

The stitched route's scenario ladder ([Palm-Ledger],
`_pnl_builders._palm_scenario_ladder`) rests on the Palm identity
(`Aggregate.palm_kappa`): conditional means of **per-claim sums** given
another per-claim sum, E[sum f(X_i) | sum n(X_i) = s], one FFT per row. Two
escape hatches extend it past per-claim sums:

1. **Coarsening on the conditioning side.** The final net T(S_r) is a
   deterministic function of the retained compound S_r, so conditioning on
   it is a coarsening of conditioning on S_r. The tower property turns
   E[row | T(S_r)] into a density-weighted level-set average of the
   already-computed kappa, implemented as the rebucket pushforward. Any
   nonlinearity in T is laundered this way.
2. **Measurability on the target side.** An aggregate-cover row g(S_r) is
   sigma(S_r)-measurable, so against the S_r subject it is the
   deterministic vector g(xs), and it transports exactly through hatch 1.

Under gross conditioning the subject S_g is itself a per-claim sum, so every
per-claim row (gross loss, every occurrence premium, recovery, result,
running net through the occurrence tier, net-of-occurrence) is exact 1-D
Palm with the identity conditioning (`palm_kappa(fn, None)`). But an
aggregate-cover row g(S_r) now has **neither hatch**: it is not a per-claim
sum, and S_r is not a function of S_g (the random claim count decouples
them), so E[g(S_r) | S_g = s] needs the joint law, a genuinely 2-D object.
The unique blocked pairing is (gross conditioning, aggregate-tier row).

**Affine carve-out.** Linearity of conditional expectation reopens the
target side when g is globally affine on the subject's support:
g(S_r) = a S_r + b gives E[g(S_r) | S_g] = a kappa_r(S_g) + b with
kappa_r = `palm_kappa(retained_fn, None)`. An aggregate quota share
(`<y> po <limit> xs 0` with the limit beyond the support) qualifies; any
kinked transform (an xs layer, a cap, a corridor, an aggregate deductible,
or any sum of layers that is not secretly a quota share) does not.

**Per-atom route.** On the shared-atoms route (`PnL._scenario_ladder`) none
of this arises: the ladder is an exact atom-slice average anchored on the
grand result, and a gross-anchored twin is the same slice arithmetic
anchored on the first group's result row. It is exact for **every** row,
nonlinear aggregate covers included, because the atoms carry the joint.

## Decision taken: option 1 (exact or absent)

The gross basis is served only where it is exact. No plug-in approximations
(evaluating g at the conditional mean carries a Jensen gap and breaks
footing), and no 2-D machinery on the fast stitched route (deferred, see
Out of scope).

Eligibility per route:

* **Per-atom route**: always eligible (exact for all rows).
* **Stitched Palm route**: eligible under the existing ladder conditions
  (guaranteed-cost, frequency with `freq_pgf_prime`, unsigned unwindowed
  grid) **and** the aggregate tier is absent or numerically affine on the
  subject's effective support. Test affinity on `t_vals = agg_transform(xs)`
  restricted to buckets where the subject density exceeds the noise floor;
  when affine, recover (a, b) from that region and build the aggregate-tier
  gross cells as a kappa_r + b.
* Everything else (massive one-sweep, ineligible frequency): no gross
  column, same as the net ladder today.

### Open fork for review: strict absence vs truncation

When the stitched route is Palm-eligible but the aggregate tier is
**nonlinear** (the `BuildRe` case: `12200 xs 82000`), two choices:

* **Fork A, strict all-or-nothing.** The gross column is entirely absent,
  matching the existing "no half-filled ladders" principle. Cost: the
  motivating example itself loses the column.
* **Fork B, truncation (recommended).** Serve exact gross cells for every
  row whose Palm spec is `const` or `claim` (the occurrence tier down
  through net-of-occurrence) and NaN for the aggregate tier and everything
  downstream of it (the aggregate cover's rows, the running net after it,
  the grand result, total impact). Framed as a truncation, not a hole: the
  column is exact on the sub-ledger it claims and visibly ends at a
  declared boundary; the caption says the gross basis cannot see through a
  nonlinear aggregate transform. Footing assertions apply only to complete
  columns.

The author decides at plan review. The implementation cost difference is
small (fork B is fork A plus serving the partial dict instead of `None`).

## Names (vetted against the existing surface)

New column labels (presentation strings, no attribute collisions):

* `walk_df`: `M01 diversified` **renamed** `M01 div net`; new column
  `M01 div gross`. Order: `Margin`, `M01 standalone`, `M01 div net`,
  `M01 div gross` (gross appended last; flag at review if the author
  prefers gross before net to match the walk's top-down reading).
* `evaluation_df`: `Div CoC` **renamed** `Div CoC net`; new column
  `Div CoC gross`. `SA CoC`, `MSD`, `Premium spent`, `Margin spent`, `CR`
  unchanged.

The renames are breaking for anything keying on the old labels; the
CHANGELOG entry says so. `rg` confirms the only in-repo consumers are the
format sheets and the exhibit snapshot fixture (regenerated), with no doc
pages referencing either label.

New internals (checked with `rg`, no collisions with existing attributes or
methods on `PnL`):

* `PnL._palm_gross_ladder`: the stitched gross-anchored ladder,
  `{row label: [value per PERCENTILE_LADDER point]}`, `None` when
  ineligible (under fork B, aggregate-tier labels map to NaN lists instead
  of being dropped, so the dict stays total over the plan rows).
* `PnL._gross_scenario_ladder()`: the per-atom gross-anchored twin of
  `_scenario_ladder`, anchored on the first group's result row.
* `_pnl_builders._palm_scenario_ladder` grows the gross pass; whether as a
  second return value or a small shared helper is an implementation choice,
  but the two ladders should share the per-row spec walk so they cannot
  drift.

## The change, step by step

### [Stitched-Gross-Ladder] build the gross-anchored Palm ladder

In `_pnl_builders._palm_scenario_ladder` (or a sibling it shares its spec
walk with):

1. Stage 1 bis: for each `claim` spec, `palm_kappa(fn, None)` (identity
   conditioning, subject the gross aggregate). `const` specs copy; `delta`
   specs subtract as today. The conditioning density returned is the gross
   aggregate's.
2. Aggregate tier: if absent, nothing to do. If `t_vals` is affine on the
   effective support, serve `a * kappa_r + b` per the carve-out. Else fork
   A or B per the review decision.
3. Anchors: the gross result row is affine decreasing in S_g
   (`result = p_gross - expense_total - S_g`), so the bucket at ladder
   point q is `round((shift - gross_gd.q(q)) / bs)`, mirroring the net
   anchor off the grand result. The gross result row's GridDistribution is
   already in `entries`.
4. No stage-2 transport: the gross subject is the conditioning variable
   itself.

Store the result on the `PnL` as `_palm_gross_ladder` beside
`_palm_ladder`. Both computed at build time, all nine ladder points (the
marginal cost over one point is nine index lookups per row).

### [Per-Atom-Gross-Anchor] the shared-atoms twin

`PnL._gross_scenario_ladder()`: identical slice arithmetic to
`_scenario_ladder` with the anchor row swapped from the grand result to the
first group's result (reachable through `self._plan`; the first
`group_result` row). Lazy, like `_scenario_ladder`. Level-set semantics on
a non-monotone gross row are inherited and documented the same way.

### [Waterfall-Dual-Columns] serve both bases

In `PnL._waterfall_frames`:

* The net cells keep their current source (the ledger's kappa column).
* The gross cells come from `_palm_gross_ladder[label][0]` (stitched) or
  `_gross_scenario_ladder()[label][0]` (per-atom), NaN when neither exists
  (and per fork B, NaN on aggregate-tier rows).
* `evaluation_df` gains `Div CoC gross = _capital_ratio(margin, gross cell)`
  beside the renamed `Div CoC net`. `_capital_ratio` already returns NaN on
  a missing input, so truncation propagates for free.
* Docstrings for `walk_df` and `evaluation_df` updated: the two bases, the
  anchor symmetry (on the Gross row the gross cell coincides with its
  standalone M01; on the closing row the net cell does), the eligibility
  and truncation story.

### [Captions-And-Format-Sheets] presentation

* `exhibits/_pnl.py` `_economic_waterfall_frames`: rewrite the two captions
  to name the question each basis answers (net: return on held capital;
  gross: performance in the gross stress), state the sign convention once,
  and cover the availability cases (both absent, net only, both present,
  gross truncated at the aggregate tier under fork B).
* `formats/formats-raw.yaml`: rename the `M01 diversified` key to
  `M01 div net`, add `M01 div gross` (money); rename `Div CoC` to
  `Div CoC net`, add `Div CoC gross` (ratio).

### [Tests] assertions and fixtures

In `tests/test_exhibits.py` (extending the existing waterfall block) and
`tests/test_pnl.py`:

* Anchor symmetry: on a per-atom tower, the Gross row's `M01 div gross`
  equals its `M01 standalone` exactly; on a stitched build, within one
  bucket. The closing row's `M01 div net` keeps its existing anchor check.
* Footing: `M01 div gross` foots down the walk when complete (per-atom
  always; stitched occurrence-only program); the net column's footing test
  stays.
* Differentiation: on the all-placed occurrence-only tower where the net
  basis collapses (equal `Div CoC net` across layers), assert the gross
  basis separates them (pairwise distinct `Div CoC gross`). This encodes
  the motivating finding.
* Eligibility: nonlinear aggregate cover produces fork A's absent column or
  fork B's NaN aggregate tier (per the review decision); an affine
  aggregate transform (quota-share-shaped `po` cover) serves the full
  column and its cells equal `a * kappa_r + b`.
* `_capital_ratio` pass-through of NaN cells (existing behavior, one new
  case).
* Regenerate `tests/data/exhibit_snapshots.json` via
  `tests/capture_exhibit_snapshots.py` and read the diff deliberately: only
  waterfall blocks should move.

### [Release-Hygiene]

Version bump to 1.0.0a383, one commit carrying code, `pyproject.toml`,
`CHANGELOG.md` (one paragraph; the renames called out as the breaking
fact), this plan moved to `dev/done/` when the author says done,
`dev/TODO.md` touched only if an entry tracks this. No doc `.rst` pages
reference the renamed labels (`rg` clean), so no doc edits; note in the
commit that docs need no rebuild for this change.

## Files

* `src/aggregate/_pnl_builders.py`: gross ladder construction, affinity
  test, anchor derivation.
* `src/aggregate/_pnl.py`: `_palm_gross_ladder` attribute,
  `_gross_scenario_ladder()`, `_waterfall_frames`, `walk_df` /
  `evaluation_df` docstrings.
* `src/aggregate/exhibits/_pnl.py`: waterfall captions.
* `src/aggregate/formats/formats-raw.yaml`: label keys.
* `tests/test_exhibits.py`, `tests/test_pnl.py`,
  `tests/data/exhibit_snapshots.json` (regenerated).

## Acceptance

1. `uv run pytest` green (tier 2), including the new assertions above.
2. On the `BuildRe` program (nonlinear aggregate cover): fork A, the gross
   column absent with the caption explaining why; fork B, exact gross cells
   through `All occurrence` and NaN below, and the occurrence layers'
   `Div CoC gross` pairwise distinct while `Div CoC net` remains the
   collapsed 215.5% family.
3. On `BuildRe` with the aggregate cover removed: full gross column,
   footing exact, Gross-row anchor property holds.
4. On a per-atom tower: full gross column, exact anchor and footing.
5. Snapshot diff confined to waterfall blocks.

## Out of scope (recorded, not planned)

* [Gross-Ladder-RAW-View]: serving the full nine-point gross ladder as a
  RAW perspective beside `economic_df`'s net kappa columns. The ladder is
  computed and stored; only the surface is deferred.
* [Agg-Tier-Joint-At-Anchors]: filling the nonlinear aggregate tier under
  gross conditioning exactly via a small dedicated joint of (S_r, S_g) at
  the nine anchor states (coarse 2-D FFT of the image pair). Revisit if
  aggregate-cover programs turn out to be where the gross reading matters
  most.
* Marginal (with/without) capital per cover, the purchase-decision
  companion: a difference of net quantiles readable off the running-net
  rows' own distributions in peel order. A separate exhibit if wanted.

## Execution log (2026-10-01)

Rulings taken at plan review, 2026-10-01: **fork B** (truncation) for the
nonlinear aggregate tier, and **net then gross** column order (the plan
default). Executed in one bump and one commit per [Release-Hygiene].

Divergences from the plan as written:

* **Version retarget: 1.0.0a384, not a383.** The plan was written against
  a382, but `[Panel-No-Title] a383` landed before execution, so the target
  moved to a384. The plan filename keeps its a383 stem.
* **Per-row affinity test, not a single test on `agg_transform`.** The plan
  says to test `t_vals = agg_transform(xs)`; with two or more aggregate
  covers the cumulative net can be affine while an individual layer row is
  kinked (slopes trading off), so the implemented test
  (`_affine_coefficients`) runs per 'agg' spec row on that row's own
  vector. For a single aggregate cover the two tests coincide; the per-row
  form is strictly more exact and serves fork B's row-wise truncation
  directly. The support mask reuses `palm_conditional_mean`'s
  machine-epsilon-relative floor.
* **The differentiation test asserts the collapse with premiums made
  proportional to the layer means.** The net-basis CoC collapse requires
  ceded premiums proportional to expected recoveries, which arbitrary
  deposit premiums do not satisfy, so the test
  (`test_waterfall_gross_separates_layers_the_net_basis_collapses`) builds
  a probe first, reads the layer means, and rebuilds with deposits at a
  common 1.25 loading. The motivating mechanism verified exactly: the
  conditional recovery multiple agrees across layers to ~1e-12 on the net
  basis and separates by ~35% on the gross basis.
* **The plan stays in `dev/` at the commit.** Both the house rule and this
  plan's own [Release-Hygiene] move it to `dev/done/` only when the author
  says done; the commit therefore does not carry the plan file.
* No `dev/TODO.md` entry tracks this work, so that file is untouched.
* **Two consumers the plan's blast-radius claim missed.**
  `tests/test_pnl_peel.py` indexes `walk_df['M01 diversified']` in the
  Palm-vs-joint agreement test (updated to the new label), and the
  hand-curated `dev/FEATURES.csv` descriptions of `walk_df` /
  `evaluation_df` name the old columns in prose (updated, audit green).
  The plan's "only the format sheets and the snapshot fixture" was wrong.

Acceptance: all five criteria met. The BuildRe-shaped fixtures live in
`tests/test_exhibits.py` (`peel_agg_affine`, `peel_agg_nonlinear`, and the
differentiation program); the snapshot diff was confined to the six
`economic_waterfall` blocks, read and accepted.
