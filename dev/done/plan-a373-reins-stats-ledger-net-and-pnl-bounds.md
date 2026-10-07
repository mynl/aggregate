# Plan a373+: reinsurance stats on the grid basis, a net-of-tier ledger block, Bounds on a P&L

Written 2026-09-30 against `1.0.0a372`. Four punchups from a review of the SPA panes
Re > Summary, Re > Stats, PnL > Ledger and Bounds, all traced to root cause in this
session. Three are LIB changes in this repository; the fourth has a LIB half here and
an API half in `V:\dev\aggregate-api`, documented in this one plan so a single agent
can execute everything.

This plan is self-contained: it states current behavior, the diagnosis, the change,
the files, and the acceptance checks. No conversation context is needed.

## Executive summary

| Step | Label | Repo | Bump |
|---|---|---|---|
| 1 | [Reins-Stats-Grid-Basis] | LIB | yes |
| 2 | [Ledger-Net-Of-Tier] | LIB | yes |
| 3 | [Bounds-PnL-Engine] | LIB | yes |
| 4 | [Bounds-PnL-Engine-API] | aggregate-api | API repo's own versioning |

Three LIB version bumps, one commit each, in this order. Nominally `a373`, `a374`,
`a375`; re-read `pyproject.toml` at execution time, the author commits independently.
Each bump carries its code, the `pyproject.toml` bump, its `CHANGELOG.md` paragraph,
and any hygiene in one commit. Tier 2 (`uv run pytest`) before each commit; tier 3
(`uv run pytest -m 'slow or not slow'`) plus the numerics gate
(`-W error::RuntimeWarning`) once, at the final LIB bump (step 1 touches numerics).

---

## Step 1. [Reins-Stats-Grid-Basis]

### Current behavior and diagnosis

All in `reins_stats_df` (`src/aggregate/_reinsurance.py:969` and the delegating
property docstring `src/aggregate/_aggregate.py:2412`). The frame mixes three
probability bases, and two of them are blind to the picks adjustment
(`Adjusting for picks`, `src/aggregate/_aggregate.py:4502`), which reshapes the
discretized severity after the frozen `scipy` objects are set:

1. **The numerators are right.** Layer ceded moments, aggregate densities, and the
   `EX` reference in `reins_view_stats` all read the discretized, picks-adjusted
   gross severity `p_sev_gross` from `reins_density_df`.
2. **The conditioning probability is wrong.** The per-layer conditional severity
   divides the grid raw moments by `pr = agg.sev.sf(attach)`
   (`_reinsurance.py:1024`, used at `1144`), the continuous mixture survival, which
   never sees the picks adjustment. The same `pr` feeds the layer frequency
   thinning (`thin_moments(pr, ...)` at `1155`).
3. **The displayed `pr_attach` / `pr_detach` are also wrong on a picks book.** They
   come from the ground-up frozen components `sev.fz.sf` (`_gsf` at `1050`,
   `_gross_detach` at `1065`), equally picks-blind.

Measured on `agg.Capstone.FullProgram` (library entry, `src/aggregate/agg/library.agg:228`,
a picks-adjusted five-component mixture under a four-layer occurrence tower):

| layer | `sev.sf(a)` used | grid `P(X > a)` actual | sev mean shown | sev mean, grid basis |
|---|---|---|---|---|
| 50% po 500 xs 500 | 0.106539 | 0.089443 | 136.5 | 162.6 |
| 90% po 1000 xs 1000 | 0.020524 | 0.035451 | 708.6 | 410.2 |
| 95% po 3000 xs 2000 | 0.003764 | 0.006637 | 1,631.5 | 925.1 |
| 5000 xs 5000 | 3.117e-05 | 4.350e-04 | **20,734.3** | 1,485.8 |

The 5000 xs 5000 conditional severity mean of 20,734 exceeds the 5,000 limit
because the divisor is 14x too small. Dividing all three raw moments by the same
wrong constant also drives the implied variance negative
(`MomentAggregator.static_moments_to_mcvsk | weird var < 0` fires twice on this
build), which is why cv and skew show as `NaN` on layers 2 and 4. On the grid basis
the conditional mean is bounded by `share * limit` and the variance is nonnegative
**by construction**, because numerator and denominator come from the same measure.

Separately, the gross cover terms row shows the claim-count-weighted average of the
component policy terms (`_reinsurance.py:1098` to `1107`): limit 4,057.69 and attach
16.81 for the capstone book, which reads as noise.

### Decisions (made by the author 2026-09-30, do not relitigate)

* **Gross cover terms**: `limit = max` over components of policy limit (`inf` if any
  component is unlimited), `attach = min` over components of policy attachment. The
  weighted average goes.
* **Ceded totals row stays as is**: `limit` = placed capacity `sum(share * limit)`,
  `attach` = min layer attachment. This is deliberate; add one sentence to the
  docstring stating the convention so it reads as chosen, not accidental.
* **Every probability in the frame reads the picks-adjusted, bucketed severity**
  (the grid). The continuous `sf` and ground-up `fz` bases go entirely.
* **The per-layer severity is conditional on a loss to the layer**, computed
  self-consistently on the grid.

### The change

In `reins_stats_df` (`src/aggregate/_reinsurance.py`):

* Replace `_sf`, `_gsf`, `_gross_detach` and the `en` / `ws` weighting block with two
  grid helpers over `p_sev_gross`:

  ```python
  def _sev_gt(t):   # P(subject > t), exclusive: attach basis
      return float(np.sum(p_sev_gross[xs > t]))
  def _sev_ge(t):   # P(subject >= t), inclusive: detach basis
      return float(np.sum(p_sev_gross[xs >= t]))
  ```

  The semantics change with the basis: these are probabilities **per modeled claim**
  (the conditional count `n`), read from the modeled subject severity, no longer
  ground-up exposure probabilities. That is the only basis that is consistent with
  every other number in the frame, and the only one the picks adjustment reaches.

* **Gross column**: `limit = float(np.max(lim))`, `attach = float(np.min(att))`;
  `pr_attach = _sev_gt(0.0)` (a modeled claim produces a positive subject loss);
  `pr_detach = _sev_ge(max_limit)` when the max limit is finite, else `NaN` (a claim
  exhausts the largest policy). Note `pr_detach` stays `NaN` for an unlimited book,
  preserving `test_stats_df_pricing_meta_rows`.
* **Occurrence layer k**: `pr_attach = _sev_gt(a)`; `pr_detach = _sev_ge(a + y)`
  when finite else `NaN`; and the **same** `pr_attach` value is the conditioning
  divisor for the conditional severity and the thinning probability for the layer
  frequency. One number, three uses, one basis. The identities
  `freq mean = n * pr` and `sev_cond = sev_uncond / pr` become exact by
  construction (`test_stats_df_occ_layer_conditional_severity` then passes with
  equality rather than luck).
* **Ceded total**: `pr_attach = _sev_gt(min_attach)`. **Net total**:
  `pr_attach = _sev_gt(0.0)`. Terms unchanged per the decision above.
* **Aggregate stage**: already grid-based (`_pr_agg`, `_ge`); leave, but confirm the
  exclusive-attach / inclusive-detach convention matches the occurrence stage.
* Delete the now-dead helpers and the comment block at `1041` to `1078` explaining
  the old fz basis.

### Documentation lockstep (no doc build; note pending rebuild in the commit)

* `src/aggregate/_aggregate.py:2412` (`reins_stats_df` property): rewrite the
  `meta` bullet. Remove "ground-up exposure probabilities" and the
  `self.sevs[i].fz` explanation; state the grid basis (per modeled claim, from the
  picks-adjusted bucketed severity), the max/min gross terms, and the deliberate
  placed-capacity ceded convention. Remove the parenthetical about the layer freq
  using "a separate basis": there is now one basis.
* Grep and fix stale wording: `rg 'claim-count-weighted|ground-up exposure' src docs`.
  Known hits: `_aggregate.py:2448`, comments in `_reinsurance.py`, and check
  `docs/2_aggregate_overview/pipeline-reinsurance.rst`.
* The exhibit captions in `src/aggregate/exhibits/_aggregate.py` (`terms_caption`,
  `moments_caption`) still read correctly ("the chance a loss reaches it"); confirm,
  do not rewrite.

### Tests

In `tests/test_reins_reporting.py`:

* `test_stats_df_meta_rows`: update the gross-terms comment and assertion to
  max/min (the single-component test values are unchanged; add a two-component
  mixture case where max/min differs from the weighted average).
* `test_stats_df_pricing_meta_rows`: values survive (uniform severity, grid and
  continuous agree to grid tolerance); keep tolerances.
* Add `test_stats_df_picks_book_conditional_severity` (mark `slow`, the capstone
  chain costs seconds): build `agg.Capstone.FullProgram`, assert for every
  occurrence layer that the conditional sev mean is `<= share * limit`, that cv and
  skew are finite wherever `pr_attach > 0`, and that
  `freq mean * sev mean == agg mean` per layer to `rel=1e-9`.
* Sweep the other files touching these rows for pinned values:
  `test_derived_programs.py`, `test_exhibits.py`, `test_reins_view_pricing.py`,
  `test_chart_structure.py`, the bivariate suites. Use tier 1
  (`pytest -n0 --dist no --testmon-forceselect`) to find breakage cheaply.

The spec snapshot (`tests/data/expected_specs.json`) is untouched: no grammar or
transformer change.

### Acceptance

`agg.Capstone.FullProgram` layering table shows gross limit 10,000 / attach 0; the
5000 xs 5000 layer sev mean ~1,486 with finite cv / skew; no
`weird var < 0` output during the build; full fast suite green.

---

## Step 2. [Ledger-Net-Of-Tier]

### Current behavior

The ledger row template is `_ledger_plan` (`src/aggregate/_pnl.py:620`), the single
source of truth materialized by all three evaluation routes (`_assemble_rows`
~`1007`, `_init_massive` ~`1139`, the stitched route ~`1223`; each raises if a kind
is unhandled). A layer-peeled walk passes `tier_spans` (built in
`src/aggregate/_pnl_builders.py:965` to `978`) and gets one subtotal block per tier
that peels into two or more steps (`[Tier-Subtotal-Rows]`): tier total
consideration, tier total obligation, tier result. That block is **the tier's own
program** (what the occurrence purchases cost and returned), not the running
position. The running net through the tier exists only as the last layer's
`net through <layer>` row, and the `economics()` docstring says so explicitly
(`_pnl.py:1610`: "There is no Net row on a tier block").

On `Capstone.PnL` the reader who wants "net of the occurrence program, before the
QS" must know that the `5x5 / Margin / Net` row (887.50) is that number, and there
is no consideration or obligation split of the position at all.

### The change (author decision: a whole Consideration / Obligation / Margin block)

After each tier span's rows, **when at least one group follows the span**
(`hi < len(groups)`; when nothing follows, the grand `All` block already is the
net-of-tier position), emit a cumulative **net block** for the position through the
tier, groups `[0, hi)`:

* `('net_total', (hi, 'cons'))`, label `net of <tier> consideration`: the sum of
  every consideration leg in groups `0..hi-1`. Same machinery as `grand_total`
  restricted to the span.
* `('net_total', (hi, 'obl'))`, label `net of <tier> obligation`: ditto for
  obligation legs (gross losses and expenses plus the tier's recoveries and
  commissions).
* `('net_result', hi)`, label `net of <tier>`: the running net through group
  `hi - 1`. This is **the same atom** as `('running_net', hi - 1)`; alias it in
  `_by_kind` rather than recomputing, on all three routes.

Emit the block for **every** tier span followed by a later group, regardless of span
width. The subtotal block keeps its `>= 2` gate (a one-group tier's own rows already
are its subtotal), but the net block is a different thing and is equally missing on
the plain tier walk, so a one-group span earns it too. Confirm at execution that the
plain walk route passes `tier_spans` (both call sites around
`_pnl_builders.py:978` and `:1053` appear to); if it does not, thread them through.

Step label: derive from the span label by replacing a leading `All ` with
`Net of ` (`All occurrence` becomes `Net of occurrence`), else prefix `Net of `.

### Presentation and consumers

* `economic_df` row index (`_side_index`): the three rows land at
  `(<net step label>, 'Consideration'|'Obligation'|'Margin', 'Net')`, after the tier
  subtotal block, before the next tier's first group.
* `economics()` card: add the `(<net step label>, Consideration|Obligation|Margin)`
  block in the same position; **update the docstring at `_pnl.py:1603` to `1611`**,
  whose "no Net row on a tier block" sentence this change supersedes.
* `evaluate()`: leave `_MARGIN_KINDS` (`_pnl.py:2503`) unchanged. The net block's
  margin is the same position as the last running net, already evaluated; adding
  `net_result` would duplicate a panel row. Say so in a comment beside
  `_MARGIN_KINDS`.
* `_RESULT_KINDS` (`_pnl.py:788`): unchanged; `net_result` is a running position,
  not a step's own result, exactly like `running_net`, which is also absent.
* `economic_ratios_df` iterates group spans (`_ratio_spans`, `_pnl.py:2005`) and
  never reads the new kinds; confirm, do not change.
* Charts: check `src/aggregate/charts/_emit_pnl.py` (ledger and waterfall emitters)
  key on row kinds and are undisturbed by the new rows; the waterfall in particular
  must not pick up the net block as a step. Same check for
  `src/aggregate/exhibits/_pnl.py`.
* The SPA renders the ledger frame generically through the exhibit route: **no API
  or SPA work**.

### Name vetting

New kind strings `net_total` / `net_result` and the `net of <tier>` labels: `rg`
them across `src/aggregate` before coding; they must not collide with existing
kinds, `_by_kind` keys, or ledger labels (`_ledger_plan` raises on duplicate labels,
which is a second guard).

### Tests

* Extend the ledger tests (find them via `rg 'tier_result|Tier-Subtotal' tests`):
  on a peeled two-tier P&L (`Capstone.PnL` shape, or the smaller fixture the
  existing tier tests use), assert the net block exists after the occurrence tier
  and not after the final tier; that
  `net_total cons + net_total obl == net_result` (EX column and every kappa
  column, the footing property); and that `net_result` equals the last occurrence
  layer's running net exactly.
* Plain tier walk: the block appears after the occurrence group when an aggregate
  tier follows.
* `evaluate()` panel row count is unchanged by this step.

### Acceptance

`Capstone.PnL` ledger shows a `Net of occurrence` block (Consideration 24,062.50,
Obligation -23,175.00, Margin 887.50) between `All occurrence` and `QS`; columns
foot; full fast suite green.

---

## Step 3. [Bounds-PnL-Engine] (LIB half)

### Current behavior

`aggregate.bounds` accepts `Portfolio`, `Aggregate`, `pd.Series`, `pd.DataFrame`
(`_resolve_obj`, `src/aggregate/bounds.py:105`, raising at `:144`; `_extract_pmf`,
`:656`, raising at `:694`). A `PnL` is none of these, so `Bounds(pnl)` raises and
the API capability gate greys the whole Bounds group for any P&L object.

A `PnL` wraps exactly one engine, an `Aggregate` or `Portfolio`, on `self.engine`
(`_adopt_engine`, `src/aggregate/_pnl.py:2810`; `None` only for a hand-built kernel
P&L). The library precedent is `[PnL-Reins-Passthrough]`: the reins frames delegate
to the engine, and one exhibit function serves both hosts.

### The change

See through to the engine, in the library, so every caller (the API routes pass the
object straight to `Bounds` / `PricingBounds`) works unchanged:

* `_resolve_obj`: before the existing isinstance ladder, unwrap a P&L:

  ```python
  from ._pnl import PnL          # local import, matching the module's pattern
  if isinstance(obj, PnL):
      if obj.engine is None:
          raise TypeError(
              'Bounds: this P&L wraps no engine (hand-built kernel P&L). '
              'Bounds needs the loss distribution of an Aggregate or '
              'Portfolio engine.')
      obj = obj.engine
  ```

  then fall through. The display name is the engine's own name: the bounds answer
  on the engine's output distribution (net for a net-of program), and the name
  should say whose distribution answered.
* `_extract_pmf`: the same unwrap, so `PricingBounds` and every pmf path accept a
  P&L.
* `AllocationBounds` stays `Portfolio` only. A P&L wrapping a `Portfolio` reaches
  it through the unwrap automatically; book-level P&L is deferred work anyway
  (`PnLBook` raises `NotImplementedError` by design).
* Update the accepted-types docstrings: `bounds.py:42`, `:115` to `:120`, `:146`,
  `:162`, `:664`, `:696`, `:1527`.

Semantics note for the docstrings: Bounds on a P&L answers on the **engine's loss
distribution** (the P&L's stochastic obligation leg), not on the P&L margin, which
is signed and is not a loss variable.

### Tests

In the bounds test file (`rg 'class Bounds|def test.*bounds' tests` to locate):

* `Bounds(pnl)` and `Bounds(pnl.engine)` agree on `p_star` and the head of
  `tvar_df` for a small reinsured P&L fixture.
* A kernel P&L (engine `None`) raises `TypeError` with the message above.
* `PricingBounds` accepts the P&L as `x_source`.

### Acceptance

`Bounds(build_of_a_pnl, premium, ...)` computes; full fast suite green; tier 3 and
the numerics gate run here, at the final LIB bump.

---

## Step 4. [Bounds-PnL-Engine-API] (API half, repo `V:\dev\aggregate-api`)

Executed after step 3 lands, under that repo's own versioning and CHANGELOG rules.
No LIB bump.

* **`src/aggregate_api/capability.py:482` `can_bounds`**: light for a P&L whose
  engine qualifies:

  ```python
  return isinstance(obj, (Aggregate, Portfolio)) or \
      isinstance(getattr(obj, 'engine', None), (Aggregate, Portfolio))
  ```

  Update its docstring: the isinstance rationale paragraph gains a sentence saying
  the engine see-through mirrors the library's own unwrap, so the flag and the
  constructor cannot disagree. `can_allocate` unchanged (`Portfolio` only).
* **Routes**: no change. `run_envelope` / `run_pricing_bounds`
  (`src/aggregate_api/bounds.py:228`, `:310`) pass `entry.obj` straight to the
  library classes, which now unwrap.
* **SPA, premium seed**: the Bounds leaves seed the premium field above the
  object's headline mean (`web/src/main.js` around `:71` and `:337`). For a P&L the
  headline mean is the **margin** (837 for `Capstone.PnL`), below the engine's
  expected loss (5,178.5 net), so the seeded premium would be rejected by the
  library's floor check. Requirement: for a P&L, seed from the engine's loss mean.
  Investigate at execution where the SPA gets the mean (object summary payload) and
  either surface the engine mean there or fetch it for the seed; smallest honest
  change wins, YELL if it implies more than a small payload addition.
* **SPA, leaf gating**: `showBoundsLeaf` (`main.js:3517`) and the comment block at
  `:3529` describe a client-side check tied to the accepted classes; confirm it
  keys off `canBounds` and needs no edit.

### Acceptance

In the app, opening a P&L object shows the Bounds group lit; compute returns the
same table as the underlying engine object; the premium field seeds above the
engine's expected loss.

---

## Hygiene checklist (every LIB bump)

* One commit per bump: code + `pyproject.toml` + one-paragraph `CHANGELOG.md`
  entry opening `**[Label] ...**` + test updates. Subjects:
  * `[Reins-Stats-Grid-Basis] a373: reins stats probabilities and cover terms read the grid`
  * `[Ledger-Net-Of-Tier] a374: ledger gains a net-of-tier consideration/obligation/margin block`
  * `[Bounds-PnL-Engine] a375: Bounds and PricingBounds accept a PnL via its engine`
  (renumber from `pyproject.toml` at execution time)
* Docs `.rst` edits in lockstep where wording went stale; no doc build (author
  rebuilds outside the loop); note "docs pending rebuild" in the commit that edits
  them.
* Never push. Move this plan to `dev/done/` only when the author says done.

---

## Execution log (2026-09-30)

Executed as planned at `a373` / `a374` / `a375`; step 4 in `aggregate-api`
after. Divergences and findings, none plan-breaking:

* **[Reins-Stats-Grid-Basis] a373.** `test_stats_df_meta_rows` pinned the
  continuous `pr_attach` 0.75; on the grid basis it reads 0.74988, so the
  assertion took the same `abs=1e-3` half-bucket tolerance the pricing meta
  test already used. The 8 reins exhibit canonical snapshots were recaptured
  (`tests/capture_exhibit_snapshots.py`); the diff touches exactly the 8
  `reins/*` keys, probability and conditional severity cells only, including
  gross `pr_attach` 1.0 to 0.9928 (the zero bucket's mass now excluded, as the
  per-modeled-claim basis implies). Acceptance hit exactly: capstone 5000 xs
  5000 sev mean 1485.78, `pr_attach` 4.350e-4 matching the plan's table, no
  `weird var < 0`.
* **[Ledger-Net-Of-Tier] a374.** The net gate needed `lo < hi` besides
  `hi < len(groups)`: the peel builders pass an *empty* span for an absent
  tier (an aggregate-only peel carries `('All occurrence', 1, 1)`), which
  otherwise emits a net block aliasing a nonexistent running net. `hi >= 2`
  added for the same alias reason (theoretical `(0, 1)` span). Presentation
  labels live in a new `_net_span_labels` (keyed by `hi`) rather than widening
  `_span_labels`, which gates the subtotal block and feeds `_blocks`
  (`economic_ratios_df`) and `density_df`; widening it would have changed
  frames the plan ruled untouched. The plain walk (`build_xpnl_walk`) indeed
  did not pass `tier_spans`; threaded through as the plan anticipated.
  `_peel_stitched` supplies the new rows as its own entries: net consideration
  is a constant, net obligation an affine of the cumulative-net marginal
  already in hand (shift `commissions - expenses`), net result reuses the
  running-net entry; Palm specs in lockstep. `LEDGER_ROW_FLAGS` gains
  `net_result: muted` (it is the running net under a tier-level name);
  `net_total` rows stay unflagged like `tier_total`. The plan's "`economics()`
  card" is the `summary_df` property (`economics` is the resolved-economics
  dict attribute); the card and docstring edits landed there. Test ripples:
  two pinned step lists gained `Net of occurrence`
  (`test_pnl_peel.py::test_occurrence_tier_precedes_the_aggregate_tier`,
  `test_both_tiers_get_their_own_subtotal`,
  `test_pnl_consolidated_walk.py::test_acceptance_walk_steps`), and
  `test_ratio_df_has_a_row_per_block_including_the_tier_subtotals` now
  excludes net steps, since `economic_ratios_df` is deliberately unchanged.
* **[Bounds-PnL-Engine] a375.** As planned. One test adjustment:
  `PricingBounds` names its x source `'X'` explicitly, so the P&L test anchors
  on the vertex table (`_T`) agreeing with the engine-built instance rather
  than on the display name.
* **[Bounds-PnL-Engine-API] aggregate-api a178.** Two of the plan's premises
  were stale, both resolved API-side with the smallest honest change:
  * **"Routes: no change" did not hold.** `_require_risk` type-checks at the
    door, and the envelope's second panel calibrates onto and reads off the
    object handed to `Bounds` (`bounds._obj.distortions`), which a P&L cannot
    answer (no `calibrate_distortions`). `_require_risk` now resolves a P&L
    to its engine and returns the resolved risk; both bounds runners use it.
  * **The premium-seed requirement is moot, but the submit path was not.**
    Since api a100 the Bounds form seeds from the held pricing, not from
    `mean * 1.25`, so the P&L margin never reaches the seed and no payload
    change is needed. The real gap was the form's submit path, which resolves
    its premium and asset pair through `POST /pricing/preview`
    (`price_pentagon`), refused for a P&L. `run_pricing_preview` now resolves
    a P&L to its engine, preview only; `run_calibration` and the other
    runners still refuse a P&L.
  * The two Bounds leaves' `why` strings updated ("needs a loss distribution:
    an aggregate, a portfolio, or a P&L wrapping one"); flag table and three
    new route tests in `tests/test_bounds.py`; full API suite 404 passed;
    re-synced (records aggregate `1.0.0a375`); SPA rebuilt. A leftover
    `aggregate-api.exe` server process was holding the executable during the
    sync and was stopped (the author had stopped the server for this
    rebuild; one instance survived).
