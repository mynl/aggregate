# Plan [Pricing-Natural-Allocation]: five subtabs, the natural allocation, and the kappa plot

> **Status: DONE, moved to `dev/done/` 2026-08-20.** Every phase landed:
> LIB N1 to N5 at `aggregate` 1.0.0a281 to a285, on top of the companion
> notes' four phases at a277 to a280, then API B1 (the allocate route,
> a103), B2 (the five subtab pane, a104) and the sync that took the
> scaffolding down (a105). Section 8's acceptance criteria were all checked
> in the browser on both reference programs at a105. The `Massive joint`
> control shipped as the greyed placeholder section 7 specified, and stays
> greyed until LIB `[Massive-Kappa-Second-Sweep]` lands, which LIB
> `dev/TODO.md` tracks. The history below is left as drafted.
>
> **Status: DRAFT v3, 2026-08-14, awaiting author review.** Written from the
> author's rulings of the same day: four subtabs became five with the Plot
> leaf; the Portfolio question of v1's section 3 is ruled (the Portfolio's
> current Allocate content is correct and stays; a new stand-alone frame is
> built for it); v3 records two further rulings, the portfolio stand-alone
> anchor (on the total, decision 2) and the memoized joint (decision 6).
> Canonical copy lives in `aggregate_api/dev/`;
> `aggregate_REFACTOR/dev/done/plan-pricing-natural-allocation.md` is a symlink to
> it, the `plan-3d-plot.md` arrangement. Nothing is implemented. Line anchors
> are LIB `1.0.0a275` and API `1.0.0a99`.
>
> **Companion document**: LIB `dev/notes-net-natural-allocation.md`
> (`[NetCeded-Kappa-Band]`, notes, 2026-08-14), a measured working session
> that specifies the kappa band chart, the band iterator that lifts the
> massive refusal, and three sizing defects in the joint's public entry
> point. Its proposed phases are prerequisites for the second half of this
> plan and are tracked there, not duplicated here.
>
> **Sequencing**: the API phases assume `dev/plan-pricing-form.md` phase 1
> has landed (the shared `createPricingForm` factory, planned `1.0.0a100`);
> the new Allocate leaf is that component's fourth mount point. The LIB
> allocation frame (phase N3 here) additionally requires the notes'
> `[Sizing-And-Passthrough]` phase, without which a default-built joint can
> carry a deficit of 0.535 and answer questions anyway (notes, section 7).

## The design in one paragraph

The Pricing pane becomes five subtabs, `Calibrate  Stand-alone  Allocate
Plot  |  Evaluate`, the first four grouped and Evaluate staying on the far
side of the divider, unchanged. **Stand-alone** prices the parts alone: for a
reinsured `Aggregate` that is today's Allocate content renamed (each view
priced as its own distribution, the difference rows per the standing
perspective ruling); for a `Portfolio` it is new, each unit priced alone with
each calibrated family and the sum of the parts set against the portfolio
total. **Allocate** splits the whole across the parts: for a `Portfolio` that
is today's Allocate content unchanged (`analyze_distortions`, the per-unit
premium allocation, which was correctly named all along); for an
occurrence-reinsured `Aggregate` it is new, the a274
`BivariateAggregate.natural_allocation` splitting the calibrated gross
premium to ceded and net off the joint. **Plot** illustrates the allocation
with kappa curves: for a `Portfolio` the kappa panel the overview plot
already carries on its right hand side, served alone; for an
occurrence-reinsured `Aggregate` the new kappa band chart of the notes,
conditional band and deterministic ceiling included. The symmetry is the
design: stand-alone compares the sum of the parts to the whole, allocate
decomposes the whole into the parts, and the plot shows the conditional
machinery that makes the decomposition what it is.

## 1. The two-by-two, ruled

`[Standalone-Prices-The-Parts, Allocate-Splits-The-Whole]` (author,
2026-08-14, superseding the v1 section 3 question). The Portfolio's
`pricing_df` is a genuine allocation (`analyze_distortions` documents its
`allocation=` parameter as the tail-share choice for the per-unit premium
allocation, `_portfolio.py:4010-4013`) and stays under Allocate. The
stand-alone path for a book is built fresh, by the same criterion that
renamed the reinsured tab: same fitted families, each part priced as its own
distribution, then compared.

| Source | Stand-alone | Allocate |
|---|---|---|
| `Aggregate`, occurrence cession | `reins_price_df`: every view its own price; INSURER stars the basis, drops `ceded`, appends `less` rows (today's content, renamed) | **new**: `natural_allocation` off the joint, the gross premium split to ceded and net, footing exactly |
| `Portfolio` | **new**: each unit priced alone per family, sum of parts against the portfolio total | `calibration_df` plus `pricing_df` from `analyze_distortions` (today's content, unchanged, stat slices INSURER and all) |
| `Aggregate`, no cession | the single calibration row (degenerate: one part, which is the whole) | not offered, greyed |

The two comparisons the pane exists to put on screen: sum of stand-alone
parts against the whole (the diversification story, Portfolio Stand-alone),
and net allocated against net priced alone or calibrated directly (the
consistency story, the two middle tabs read together on a cession).

## 2. Rulings this plan rests on

`[Five-Pricing-Subtabs]` (author, 2026-08-14). Calibrate, Stand-alone,
Allocate, Plot, then the divider, then Evaluate.

`[Allocate-Is-The-Natural-Allocation]` (author, 2026-08-14). The Allocate
name belongs to the additive decomposition of one premium, whatever the
parts are: units of a book, or the halves of an occurrence program.

`[Difference-Is-A-Perspective]` (author, 2026-08-11, standing). Gross less
net is the buyer's allowance for reinsurance, not the price of the cession.
The stand-alone exhibit keeps that structure verbatim through the rename.

`[NetCeded-Natural-Allocation]` (LIB a274) and `[Bivariate-Exeqa]` (a273)
supply the pricing machinery; `[NetCeded-Kappa-Band]` (the notes) supplies
the plot's specification and the sizing honesty the joint build needs. The
a274 out-of-scope note reserved this surface as "an additional view of the
`pricing.allocate` exhibit"; this plan is that slot being filled, as its own
subtabs rather than a block.

## 3. The exhibits after this plan

### `pricing.stand_alone`, on `CalibrationResult` (renamed key, one new branch)

Registry key `pricing.allocate` renames to `pricing.stand_alone` for the two
`Aggregate` branches (reinsured: `reins_price_df` with the INSURER
restructure; plain: the calibration row). Content, formats, captions and the
star-drop-difference structure are untouched, captions reread for the word
"allocate" and reworded where they would now claim the wrong tab.

The new Portfolio branch serves `calibration_df` then the new
`stand_alone_df` (section 4): rows `(distortion, unit)` with, per family, a
`sum of parts` row and a `total` row, columns the pentagon octet. RAW
carries all of it; the units, the sum and the total are all measurements or
arithmetic on measurements, and the core of the page is the sum sitting
against the total. INSURER appends a `sum of parts less total` row per
family with the ratios recomputed on the differenced levels
(`complete_pentagon`, the `_difference_rows` pattern): the diversification
benefit, the buyer's reading of pooling. Caption seed: "Each unit priced as
its own distribution with the same fitted families, at the book's
calibrated asset level; the sum of those prices against the book priced
whole at the same level. The gap is what pooling is worth under that
family."

### `pricing.allocate`, on `CalibrationResult` (kept key, one moved branch, one new)

The Portfolio branch moves here **unchanged**: `calibration_df` plus
`pricing_df`, stat slices under INSURER, captions as they are. A Portfolio
reader sees no change at all on this tab, which is the ruling.

The new occurrence branch serves `natural_allocation_df` (section 5): rows
`(distortion, view)` over `gross`, `ceded`, `net`, columns the pentagon
octet, ceded plus net footing to gross exactly and the gross row constant
across families at the calibrated premium. RAW and INSURER are identical:
the allocation is already the cedent's one-basis reading, the ceded row here
is the cedent's allocated cost of the program rather than a reinsurer's
price, so nothing drops, nothing stars, and there are no difference rows
because the whole table is a decomposition already. Caption seed: "The
calibrated gross premium split across the occurrence program on one
consistent basis. Each family's distorted view of the gross distribution
sets the weights; the kappa curve off the joint says what ceded and net each
earn under them; ceded plus net foot to gross exactly. This is neither view
priced on its own, which is the Stand-alone tab, nor the difference of two
such prices: it is one price decomposed. Fractions are computed on the
joint's grid and applied to the calibrated premium; the joint's residual
against the fine density (rho_gap) is reported, not absorbed."

**Availability** moves from `_perspectives_always` to a predicate,
`_perspectives_allocation`: full tuple for a `Portfolio` source; full tuple
for an `Aggregate` source with an occurrence program
(`_source.occ_reins is not None`, a public attribute) calibrated on gross
(`result.reins_view == 'gross'`); empty otherwise. A fit struck on net has
no gross premium to split, so the basis gate is structural, not a
preference. `pricing.stand_alone` stays `_perspectives_always`.

### The chart: `kappa`, one name across three sources

A new registered chart, name `kappa`, drawn by the Plot leaf through the
ordinary chart route (`available_charts`, `build_chart_doc`, `doc_hash`
ETags). Calibration-independent by design: kappa curves condition on an
outcome, not on a distortion, which is why this is a chart on the built
object and not a fourth pricing POST.

* On `Portfolio`: the kappa panel the `port` overview chart already draws on
  its right hand side, extracted into a single-panel document (the emitter
  code reuses, not duplicates, the panel builder).
* On `BivariateAggregate`: the two-panel kappa band chart of the notes'
  section 6 (`[Chart-Kappa-Band]`): curves, conditional band, identity
  reflection, share panel with the single-layer ceiling. That registration
  and its emitter belong to the notes' phase and land there.
* On `Aggregate` with an occurrence program: a thin delegate that builds the
  joint (honest sizing, per the notes' `[Sizing-And-Passthrough]`) and
  emits the same band chart. Where the built joint is held so the exhibit
  frame and the chart do not each build one is decision 6.

## 4. LIB: `CalibrationResult.stand_alone_df` (Portfolio sources)

Lazy, cached in `_frames` beside `pricing_df` and `reins_price_df`;
`AttributeError` on a non-Portfolio source, matching the siblings.

**The anchor is the total's, once** (author, 2026-08-14). A portfolio
calibration resolves its anchor on the portfolio total: `p` becomes
`a = total.q(p)`, or the caller's `a` is taken directly, and that one asset
level is the level of the whole exercise. Nothing re-resolves per unit.
Every row of the frame prices at that common `a`: each unit priced with the
family's distortion capped at the book's asset level, the `total` row the
book priced whole at the same level, which is the family's fitted premium
(target plus its `error`). The `a` column is constant down the frame and
reads the calibration's own asset level. This also means a `p` calibration
and an `a` calibration produce the same frame whenever they resolve to the
same level; there is no per-unit anchoring rule to state and no open
corner. (The per-unit `q_i(p)` alternative was considered and rejected: the
`p` belongs to the calibration on the total, not to the units. The
unlimited alternative was rejected too: a mass-at-zero family such as
`ccoc` priced unlimited charges the top grid bucket and tracks `log2`, the
a274-documented pathology, and an unlimited sum has no tie to the
calibrated premium.)

The `sum of parts` row sums the amount columns (`L`, `M`, `P`, `Q`) with
ratios recomputed and carries the common `a`; the comparison against the
`total` row is well posed because every row prices at one asset level under
one distortion: sub-additivity gives sum at or above total for every
concave family.

Tests, fast tier: sub-additivity on the reference book (sum of parts at or
above total in `P` for every concave family); the `total` row ties to
`distortion_df`'s fitted premium per family; unit rows tie to
`Distortion.price` called directly on a unit's density at the calibration
`a`; the `a` column constant and equal to the calibration's; caching
pinned (one sweep, second access free).

## 5. LIB: `CalibrationResult.natural_allocation_df` (occurrence sources)

As v1, now with the sizing dependency explicit. Guards first: occurrence
program present, `reins_view == 'gross'`, refusals naming the gross basis
and the Calibrate tab. Then:

1. Build the joint once through `occ_bivariate(views=('gross', 'ceded'))`
   under the measured default sizing of the notes' `[Sizing-And-Passthrough]`
   phase (exact lattice when affordable, per-axis windows, reported when not
   exact). **Blocked on that phase**: today's defaults can hand back a
   deficit-heavy joint that warns once and answers anyway, and a priced
   exhibit must not be built on one silently. A large `rho_gap` or deficit
   surfaces through the envelope warning channel, not a failure.
2. For each family, `joint.natural_allocation(dist, P=P_cal)`, `P_cal` the
   shared calibrated premium off `calibration_df` (decision 1).
3. Stack to `(distortion, view)` rows, `PENTAGON_STATS` columns; the
   `a = inf` unlimited-reading convention rides through (`Q`, `PQ`, `ROE`
   blank, as on an unlimited `reins_price_df` quote); `.attrs` carries
   `rho_gap` per family and the joint's realized sizing.

Tests: footing at 1e-12 per family through the stack; gross row equal to
`P_cal` everywhere; both guards' sentences; the cache; `rho_gap` finite,
small on the reference program, and invariant to `P_cal`.

## 6. LIB phases

One bump and one commit each, numbers assigned at execution (a276 up as
free). N1 and N4 move exhibit snapshots; read those diffs deliberately.
The notes' phases (`[Sizing-And-Passthrough]`, `[Joint-Row-Bands]`,
`[Kappa-Band-Columns]`, `[Chart-Kappa-Band]`) interleave per the notes'
own proposed order; only the dependencies below bind.

* **N1 `[Standalone-Rename]`**: the key rename and branch split of
  section 3. After it, a Portfolio's `pricing.allocate` is byte-identical
  to today's; the aggregate branches answer under `pricing.stand_alone`;
  the allocation predicate exists (Portfolio-only until N4). Wire-visible
  with exactly one consumer, the sibling API, moving in lockstep (B1); the
  old key's aggregate meaning is gone, not aliased.
* **N2 `[Portfolio-Standalone-Frame]`**: `stand_alone_df` (section 4) and
  its exhibit branch on `pricing.stand_alone`.
* **N3 `[Calibration-Natural-Allocation-Frame]`**: `natural_allocation_df`
  (section 5). Requires the notes' `[Sizing-And-Passthrough]`.
* **N4 `[Natural-Allocation-Exhibit]`**: the occurrence branch of
  `pricing.allocate`; the predicate widens to gross-calibrated occurrence
  results; `available_exhibits` answers accordingly (tested: a
  net-calibrated occurrence result lists `pricing.stand_alone` only).
* **N5 `[Kappa-Chart-Surfaces]`**: chart `kappa` on `Portfolio` (the
  extracted overview panel) and on occurrence `Aggregate` (the joint-building
  delegate). Requires the notes' `[Chart-Kappa-Band]` for the band emitter
  it delegates to. Name vetted at execution against the chart registry and
  `Tweedie.kappa` (a frame name conflict the a273 plan already dodged;
  a chart registry name is a different namespace, but vet it anyway).

Every phase: envelope contract (blocks reconstruct hash for hash through
`gt.TableDoc.model_validate`) over the new fixtures, and the RAW invariant
sweep (every RAW block names a real public frame on the dispatched object)
extends over `stand_alone_df` and `natural_allocation_df`.

## 7. API phases

After LIB lands and a sync records the versions, and after
`plan-pricing-form.md` phase 1.

### Phase B1 `[Allocate-Route]`

* `POST /v1/objects/{oid}/pricing/allocate`. Body is the calibrate shape.
  For an `Aggregate`: `basis` must be `'gross'` or absent, anything else
  HTTP 400 with "the natural allocation splits a gross premium; calibrate
  on gross". For a `Portfolio`: `basis` refused as on today's calibrate
  path for books. The runner calibrates at the stated anchor and target,
  then builds the `pricing.allocate` envelopes, both perspectives,
  warnings captured through `library_warnings()`. Stateless like its
  siblings (the decision 8 precedent): a press recomputes.
* The calibrate route's bundle becomes `pricing.calibrate` plus
  `pricing.stand_alone`. Two cost consequences, both improvements: a
  Portfolio Calibrate press stops paying for the `analyze_distortions`
  sweep (it moves behind the Allocate press), and starts paying for the
  stand-alone sweep, which is per-unit `Distortion.price` calls and cheap.
  The occurrence joint is only ever built behind the Allocate press or the
  Plot leaf, never on Calibrate.
* Capability: new flag `can_natural_allocation(obj)`, true for a
  `Portfolio` and for an `Aggregate` whose `occ_reins` is not `None`.
  Vetted: `can_allocate` is taken (Bounds, Allocation Bounds) and must not
  be touched. The `_CALIBRATION_BASES` comment naming "`pricing.allocate`'s
  RAW reading" follows the rename to `pricing.stand_alone`.
* Watch item, not a blocker: the occurrence route's press pays for the 2D
  joint. Measure on `agg.mynl.com` hardware at execution; the notes'
  measured sizings say an in-core bounded-severity joint is sub-second and
  a fine one is real work, and the cap, if one proves needed, takes the
  `AGGAPI_MAX_CHART_DETAIL` shape.

### Phase B2 `[Five-Subtabs-Pane]`

* `nav.js` `NAV_GROUPS.pricing` becomes `calibrate`, `standalone` (label
  `Stand-alone`, flag `canPrice`), `allocate` (label `Allocate`, flag
  `canNaturalAllocation`), `plot` (label `Plot`, gated on chart `kappa` in
  the capability's chart list, no new flag), `evaluate` (unchanged,
  `dividerBefore` stays). Hints: standalone, "the calibrated families
  applied to each part as prices in their own right, and the sum against
  the whole"; allocate, "one premium split across the parts on one basis:
  units of a book, or the halves of an occurrence program"; plot, "the
  kappa curves behind the allocation: what each part expects, given the
  whole". `why` for a greyed Allocate: "needs a book of units or an
  occurrence cession". `check-nav.mjs` expected leaves update in the same
  commit: `standalone` for `['agg', 'agg_reins', 'port']`, `allocate` and
  `plot` for `['agg_reins', 'port']`.
* Panes: `pane-allocate` renames `pane-standalone` (still filled by the
  Calibrate press; the "One press, two panes" comment stays true and is
  reworded). New `#leaf-allocate` wrapper: its own `createPricingForm`
  mount (verb `Allocate`; basis row locked, `Gross` live alone for an
  `Aggregate` with the `why` "net and ceded are this tab's outputs, not its
  inputs", the whole row greyed for a `Portfolio` as on Calibrate), the
  massive placeholder, `pane-allocate`. `PRICING_LEAF` grows: standalone
  reads `pricing.stand_alone` off `_calibration`; allocate reads
  `pricing.allocate` off a new `_allocation` state, cleared in
  `forgetPricing()`, empty-state line "Press Allocate." The press writes
  the held `_pricing` pentagon like the other verbs and does not overwrite
  `_calibration` (decision 4).
* The Plot leaf is a chart leaf: it loads on activation through the chart
  route like every other chart leaf, ETag-cached by `doc_hash`. It is the
  first pricing leaf that fetches on activation, and that is right: it
  asks a question of the object, not of a form. If the occurrence joint
  build proves slow on the VPS, the fallback is a draw button, recorded
  here so it is a decision rather than a surprise.
* **The massive placeholder.** A disabled control under the Allocate form,
  a small labeled toggle `Massive joint`, greyed, `why`: "the disk-backed
  joint cannot serve the kappa curve yet; a post 1.0 item". No wire field,
  no handler. The notes' `[Kappa-Band-Columns]` phase is what will lift the
  refusal (the band iterator makes `exeqa_df` and therefore
  `natural_allocation` massive-clean), at which point the toggle becomes
  real and the joint sizing decision moves onto the form.

Tests: route round trips on both reference programs (rows foot; the
Portfolio allocate envelope byte-identical to today's for the same press);
basis refusals; `check-nav.mjs` green; a grep for `pricing.allocate`
app-side finds only the new consumers; the Plot leaf draws for `port` and
`agg_reins` and is absent for a plain `agg`.

## 8. Acceptance

1. On `BasicBookRe` (`occurrence net of 276 xs 55`), calibrated gross at
   `coc=0.15, p=0.99`: Allocate shows ceded plus net footing to gross per
   family at display precision, gross row constant; Stand-alone is
   pixel-identical to today's Allocate tab for the same press, title aside;
   Plot draws the band chart, and the band brackets the kappa curve.
2. On `port Basic` (the two-unit reference book): Allocate is byte-identical
   to today's; Stand-alone shows `(distortion, unit)` rows with sum of
   parts at or above total for every concave family; Plot draws the kappa
   panel matching the overview plot's right hand side.
3. The three-way net reading is on screen in one session on `BasicBookRe`:
   net priced alone (Stand-alone), net as the gross premium's share
   (Allocate), and net calibrated directly (Calibrate, basis Net), three
   different numbers, each labeled by its tab.
4. A net-basis calibration leaves Allocate refusing cleanly at every layer:
   exhibit unavailable in LIB, HTTP 400 at the route, the greyed-basis form
   never sends it.
5. `rho_gap` and any sizing compromise are visible in the served attrs and
   warnings, not absorbed.
6. The deletion grep is clean: no consumer of the old aggregate-side
   `pricing.allocate` meaning survives on either side of the wire.

## 9. Decisions (ruled where marked; the rest adopted unless the author objects)

1. **`P` per family is the shared calibrated premium** off `calibration_df`,
   not each family's fitted premium (target plus `error`). One market gross
   premium splits; families differ in their fractions alone, and the gross
   row reading constant down the table is the visible statement of that.
2. **Stand-alone unit anchoring: on the total, once. Ruled** (author,
   2026-08-14). The calibration anchor resolves on the portfolio total and
   every row prices at that common asset level; no per-unit re-anchoring,
   per section 4. Note this is deliberately not the `reins_price_df`
   re-anchoring rule; views of one program are alternative wholes, while
   units are parts of one whole whose anchor is the book's.
3. **Sum and total ride in RAW; the benefit difference row is INSURER's**,
   per `[Difference-Is-A-Perspective]`. The sum is arithmetic on
   measurements and the total is a measurement, so both are RAW facts; the
   `sum of parts less total` reading is the buyer's restructure.
4. **The allocate route does not overwrite `_calibration`.** Writing
   another leaf's held state from a side effect is how panes drift out of
   sync with their own button. The shared `_pricing` pentagon does update.
5. **Spelling**: registry key `pricing.stand_alone` (underscore, matching
   `bs_window`, `tail_behavior`); leaf id `standalone`; display label
   `Stand-alone` (a hyphenated compound, which the house rule permits);
   frame `stand_alone_df`; chart name `kappa`.
6. **The joint is built once and shared. Agreed** (author, 2026-08-14).
   The exhibit frame (N3) and the chart delegate (N5) both need the sized
   joint; LIB memoizes it on the `Aggregate`, keyed by the resolved sizing
   tuple, so a Plot visit after an Allocate press costs a lookup. An
   app-side cache would be plumbing the library can do better, and the
   memo key makes a resize an honest rebuild.
7. **`rho_gap` rides in attrs and the caption mechanism, with no invented
   warning threshold**; the sizing phase's own reporting covers the
   dishonest-grid case.
8. **RAW equals INSURER on the occurrence allocation**, per section 3.
   Revisit only if a REINSURER perspective ever arrives, which would read
   the ceded row differently.

## 10. Deliberately not in this plan

* **A direct-net-calibration comparison block.** The three-way net reading
  (acceptance 3) lives across tabs; folding a second, net-struck
  calibration into a gross-keyed exhibit is a different question inside
  this one's answer. If flipping tabs proves too weak a reading aid, it
  returns as its own ask with this paragraph as its history.
* **Any live massive path.** The placeholder is the whole massive surface
  this round; the notes own the road to lifting it.
* **Portfolio gross and ceded allocation.** `analyze_distortions` accepts
  only the net view of a book by design; nothing here reopens that.
* **Capital allocation off the joint.** `natural_allocation` returns the
  unlimited reading (`a = inf`); allocating the anchor capital across the
  program halves is its own future question.
* **The band on the Portfolio kappa plot.** Unit kappas come off the FFT
  independence trick, not a stored joint, so a conditional band there is
  new machinery with no session behind it. The Portfolio Plot is the mean
  curves, as the overview already draws them.

## 11. Cross-document ledger (to execute alongside this plan)

* LIB `dev/TODO.md`: `[Massive-Kappa-Second-Sweep]` and the
  `[Bivariate-Total-Exeqa]` entries gain pointers here and to the notes;
  the massive toggle placeholder is the app surface waiting on them.
* LIB `dev/done/plan-natural-allocation-to-occurrence-net-ceded.md` foresaw
  its surface as "an additional view of the `pricing.allocate` exhibit";
  the done file stays as written and this plan is the correction of
  record: subtabs, not a block.
* API `dev/TODO.md` and `dev/api-punchlist.md`: the Pricing leaf matrix
  rows update to five subtabs when B2 lands.
* `dev-files.md` "Getting to 1.0" (the author's master list): add this
  plan and the notes' phases.

## Cadence

LIB first, interleaved with the notes' phases per their dependencies
(minimum before the API moves: N1; minimum for the full pane: all of N1 to
N5 plus the notes' sizing and chart phases); the API syncs so `/v1/meta`
reports both versions, then B1 and B2 after the pricing form plan's phases.
On completion this plan moves to `aggregate_api/dev/done/` and the LIB
symlink follows it.
