# Plan: [Exhibit-Official-Channels]

> **CLOSED and moved to done, 2026-08-13.** Phases 1 to 6 landed at a251 to
> a256; phase 7 was struck (a250 fixed the cause); phases 8 to 10 were
> superseded by `dev/done/plan-pricing-exhibits.md` and executed through it
> (LIB a259 to a263, API a83 to a85). The Sharpen leaf cutover the phase 5
> exhibit enabled landed app side at `aggregate_api` a94, emptying
> `tables.FORMATS`. The one item this plan still owed, the **Bounds
> registration** carried in phase 10 ("the bounds leaves stay with the
> official channels plan"), is NOT done: it moves to the author's master list
> in `dev-files.md` ("Getting to 1.0"), and the app's two Bounds leaves
> remain the last pandas in its table pipeline until it lands.

> **Scope boundary, author, 2026-08-11.** This plan is about the **framework**:
> that the exhibit machinery is complete, that it is used, and that it works.
> Whether a particular exhibit's reading of the business is the right one is the
> author's question, not the plan's. So a phase may say "this block is built
> outside the official channel and must come through it" and may not say "this
> caption is wrong". Only `RAW` and `INSURER` exist; `INSURED` and `REINSURER`
> stay declared vocabulary and nothing here implements them.

Closes `dev/note-from-aggregate-api-round-6.md`, all six items, plus the Pricing pane restructure the author specified on 2026-08-11. (Item 4, the pricing leaves, now closes through `dev/plan-pricing-exhibits.md`, which supersedes phases 8 to 10 below; this plan's remaining open work is the Bounds registration in phase 10.) The goal in one sentence: **every table the app draws arrives through `aggregate.exhibits`, so no client builds a table document from a pandas frame it fetched.** When this lands the api's `tables.FORMATS` is empty, its `pricing.py` and `bounds.py` document assembly is gone, and the library owns every caption, format and row flag a reader sees.

## The invariant this plan installs

**RAW is exactly the public frame.** One block per frame, in the frame's own orientation, with no split, no dropped rows and no rearrangement. A RAW block carries a caption saying what the frame *is*, and the column formats for units the frame cannot carry itself. Nothing else.

**INSURER is the only perspective that may restructure.** It may re-orient, split one frame into several blocks, merge, drop rows, and re-caption to say what the frame *means*. Ruling `[Perspective-May-Restructure]`: **the block list itself may differ between perspectives**, not merely the content of each block. So `meta['blocks']` is a property of the `(exhibit, perspective)` pair and a client must not assume parity across perspectives.

The invariant is also a forcing function, which is its real value: **every RAW block must correspond to a real public frame.** That tells us exactly what new frames and result objects are owed, and stops the exhibit layer from becoming a second place where frames are invented.

## Rulings

`[Perspective-May-Restructure]` Block count may change between perspectives. Pricing is the big example; reinsurance is the first.

`[BS-Window-Widen]` Widen the published `bs_window_df` rather than registering the exhibit against the private `_bs_window_df`. If the curated subset dropped the two columns that answer the leaf's question, the curation was wrong and gets corrected.

`[Sharpen-Grid-Is-A-Reading]` `score` is a **column on `sharpen_df`** used for decision making, not a second fact. So the score grid is a *reading* of a published frame, which is what an INSURER block is for. **No new public frame.**

`[Waterfall-Frames-Are-Owed]` Author ruling, 2026-08-11. The invariant's forcing function bites immediately on shipped code: `economic_waterfall` computes its two blocks inside the exhibit layer and no public frame stands behind them. **Promote them.** `PnL.walk_df` and `PnL.evaluation_df` become public frames and the exhibit serves them, rather than the invariant being written with an exemption in it on the day it lands.

`[Pricing-Keyed-On-Result]` Register pricing exhibits on the **result objects**, not through an `inputs=` channel on `build_exhibit`. An exhibit stays keyed on an object, and a pricing result is an object. The registry does not change at all.

`[Calibrate-Allocate-Evaluate]` The Pricing group is three leaves, not two. **Calibrate** is all about the distortions: what was fitted and what it was fitted to. **Allocate** is the pentagon detail spread across the views. **Evaluate** takes a premium you already hold and reports what it survives, with no distortion supplied.

## Where the three leaves live today

I read `aggregate_api/src/aggregate_api/pricing.py` (653 lines) against the library. The answer to "who is handling Allocate" is **both, split down the middle**, and the half in the api is the half that is wrong.

### Calibrate: the library has it, the api shows something else

`calibrate_distortions` already computes both halves. `distortion_df` is the per family receipt (`param_name`, `param`, `error`, `gini_p`, `area`) and `calibration_df` is the one row target (`coc`, `p`, `F(a)` then the pentagon octet, where `ROE` equals `coc` as a free self check). Both carry `attrs['reins_view']`. Both exist on `Aggregate` and `Portfolio`.

But `run_price_pentagon` surfaces `distortion_df` and **never surfaces `calibration_df`**. Instead it calls `obj.price_pentagon(**anchor, **target)` separately and shows that. So the pentagon is computed twice by two routes and the reader sees the route that does not record what the distortions were actually fitted to. Calibrate becomes correct by serving the two frames the library already builds.

### Allocate: two axes, and only one of them is in the library

**Across units**, the library owns it. `analyze_distortions()` returns `pricing_df` with rows `(distortion, stat)` and units across; the api slices it four ways into `stat_LR` / `stat_P` / `stat_PQ` / `stat_ROE`. That slicing is presentation and becomes an INSURER perspective.

**Across views**, the api owns it entirely, in `run_reins_price`. It calibrates the distortion set on one basis, then applies that same set unchanged to every other basis and differences them, which is the allowance for reinsurance in the rate. That cross view step is **not in the library** and is the one genuinely missing capability in this whole plan.

Note that `Portfolio.analyze_distortions` explicitly *refuses* the view axis: `reins_view='gross'` or `'ceded'` raises, because allocating either needs a twin portfolio of gross or ceded units the library does not build. So the view axis is an `Aggregate` capability, and the refusal on `Portfolio` is a decision already taken rather than a gap.

### Evaluate: the library has it on all three classes

`evaluate` is on `Aggregate`, `Portfolio` and `PnL` and returns a DataFrame indexed `(Step, distortion)` with columns `role`, `param_name`, `param`, `gini_p`, `error`, `status`. It solves per family for the shape at which the risk adjusted margin reaches zero. `run_evaluate` is genuinely thin, about twenty five lines of real work, and needs only a result object to hang an exhibit on.

One precision on the author's framing: a distortion is not an *input* to Evaluate, which is right, but distortions are its *output*, since the panel reports the breakeven shape per family plus the family agnostic `gini_p`. And it is not PnL only: an `Aggregate` or `Portfolio` evaluates its own position in one block, while a `PnL` evaluates every margin row of its ledger, so a tower reads as the gross deal, each layer as a position, and the running net.

## The `_BasisView` finding

`aggregate_api.pricing._BasisView` is a sixty line shim class that fakes an `Aggregate` so `calibrate_distortions` can be pointed at a chosen `reins_density_df` column. Its docstring says:

> ``Aggregate.calibrate_distortions`` resolves its survival, expected loss and premium target from ``self.density_df`` ... **There is no public keyword for pointing it at another one**, so this presents a chosen ``reins_density_df`` column with the small surface that method reads.

**That is stale.** `calibrate_distortions(reins_view=...)` exists on both `Aggregate` and `Portfolio`, and `Aggregate.reins_views` is the capability query, whose own docstring says it "Mirrors `available_charts` and `available_exhibits`, which answer the same shape of question about the same object". The library's view set is also **richer** than the shim's: `['gross', 'ceded', 'net']` plus `'ceded occ'` and `'net occ'` on a two stage program, against the api's three (`gross`, `net occ`, `net`, with no `ceded` at all).

The keyword was also **broken on exactly the views the api needs**, which is presumably why the shim survived a223. That is fixed: `a250` `[Reins-Density-Fuzz]` found the cause (`reins_density_df` never called `remove_fuzz`, so a survival built on it ticked back up and `_calibration_survival` stopped on its exactness assertion) and closed it in one line. Four of five program shapes failed on at least one view before that commit and none fail after it.

**So `_BasisView` is deletable today**, with no further library work: the keyword it works around now works on every member of `reins_views`. The api is at `a71` against `a250` and has not synced, which is the only reason the shim is still in the tree.

## Verified starting state

Facts checked against `1.0.0a250`, because several contradict what the note assumes.

`reins_stats_df` is `(component, measure)` down and `(view, layer)` across: 26 rows raw, 17 under insurer once `_stats_insurer_moment_store` drops `ex1/ex2/ex3`. Both blocks the api wants are a transpose and a split of that one frame.

`bs_window_df` is missing exactly two columns against the private frame, `W` and `coverage`, and nothing else. `coverage` is a **string** (`'1-1e-12'`, `'E[N]-adj 1-1e-12'`) carrying precision a float cannot. The stray `level_0` is `bs_window_df.index.name is None`; the values are `moment`, `sbj`, `used`.

**`sharpen_score` does not exist in this library.** It is the api's name for `sharpen_df.score` unstacked by `d_log2`. `sharpen_df` is already indexed `(d_bs, d_log2)`, so no `set_index` is needed. `sharpen` is on `Aggregate` and `Portfolio` only.

**The pricing frames already exist** (`analyze_distortions().pricing_df`, `analyze_distortion()`'s `pricing_df` plus `audit_df`, `PricingResult.df`), all dataclasses in `results.py`. What is missing is a key, not a frame.

**Calibration and evaluation have no result object.** `calibrate_distortions` returns a bare `distortion_df` and stashes `calibration_df` and `distortions` on `self`; `evaluate` returns a bare DataFrame. A bare DataFrame cannot be dispatched on, since every frame would match.

`AnalyzeDistortionResult` and `AnalyzeDistortionsResult` already relabel at construction; `PricingResult` does not. None of the three carries a back reference to its source, so `_title_name` has nothing to read.

## Phases

One version bump and one commit each.

**Phases 1 to 6 landed 2026-08-11, `a251` through `a256`**, which closes note items 1, 2, 3, 5 and 6. Only item 4, the computed pricing exhibits, is open. Phase 7 is struck. **Phases 8 to 10 are superseded, 2026-08-12**: the design discussion they waited on happened and its outcome is `dev/plan-pricing-exhibits.md` (canonical copy in the API repo, symlinked here), which carries phase 8 forward as its LIB phase L1, phase 9 as L4, phase 10 as L5, and adds two phases the discussion surfaced, an unbounded anchor guard (`p=1` on an unbounded risk, phase L2) and an asset anchor on `evaluate` (phase L3). The sections below stay as written for the record; the new plan is the working copy.

| Phase | Version | Landed |
|---|---|---|
| 1 `[PnL-Consideration-Rounding]` | `a251` | Rounds in `_pnl_consideration`. Three mirrored programs in `decl-testers.agg` re-render; `DP.OddPremium` added as the inherited case that must not round. |
| 2 `[Chart-Doc-Reader]` | `a252` | `load_chart_doc` beside `canonical_dict`, hash for hash over every emitted document through real JSON. |
| 3 `[Exhibit-Perspective-Contract]` | `a253` | Invariant written and swept; `PnL.walk_df` and `PnL.evaluation_df` promoted, blocks renamed to match. |
| 4 `[BS-Window-Diagnostics]` | `a254` | `W` and `coverage` published; four served frames given honest index names, with a standing test. |
| 5 `[Sharpen-Exhibit]` | `a255` | Twelfth exhibit, RAW one block and INSURER two, `SHARPEN_FORMATS` carried upstream. Not snapshotted, the audit timing a cell. |
| 6 `[Reins-Insurer-Orientation]` | `a256` | Aggregate INSURER splits into `reins_layer_terms` and `reins_layer_moments`, layers down the rows. Portfolio unchanged, having no layer axis. |

### Phase 1: `[PnL-Consideration-Rounding]`

Note item 5. Round the premium in `_program._pnl_consideration`, where the number is produced, instead of leaving `pnl_program` to write `1428.5840984231345 premium` into a program a reader is meant to keep and edit. No decimals above 100, two at or below. The api's `_round_pnl_premium` is idempotent and deletes the day this lands.

### Phase 2: `[Chart-Doc-Reader]`

Note item 6. `load_chart_doc(d)` in `charts/ir.py` beside `canonical_dict`, exported from `charts/__init__.py`. Contract: `doc_hash(load_chart_doc(canonical_dict(doc))) == doc.hash`. Tested over `agg`, `reins` and `joint_surface`, each through a real `json.loads(json.dumps(...))`.

### Phase 3: `[Exhibit-Perspective-Contract]`

Write the invariant into the `aggregate.exhibits` docstring and `Perspective`, with a test asserting that every RAW block names a public frame that returns an equal frame, and that a perspective may return a different block list. Goes first so later phases cite it.

**Promote the waterfall's two frames first**, per `[Waterfall-Frames-Are-Owed]`. `economic_waterfall` on `PnL` is the one shipped exhibit whose RAW blocks (`walk`, `evaluation`) are computed in the exhibit layer with no frame behind them, so the test cannot be written until they are. `PnL.walk_df` and `PnL.evaluation_df` become public properties over the existing builder, the exhibit serves them under their own names, and the captions stay where they are. The block names change from `walk` / `evaluation` to `walk_df` / `evaluation_df` so a block still says which frame it is.

### Phase 4: `[BS-Window-Diagnostics]`

Note item 2. Published columns become

    ['applies', 'selected', 'x_min', 'x_max', 'W', 'bs', 'log2', 'log2_need', 'coverage', 'clipped', 'note']

with `index.name = 'method'`, which kills the stray `level_0` in the served stub. `W` numeric, `coverage` the string it is. Sweep the other published frames for the same unnamed index bug.

### Phase 5: `[Sharpen-Exhibit]`

Note item 3. Predicate on `sharpen_df is not None` (the library side of the api's `has_sharpen`). RAW is one block, `sharpen_df`. INSURER is two: the score grid `sharpen_df.score.unstack('d_log2')`, and the full walk. The api's two ledes delete. First exercise of `[Perspective-May-Restructure]`, deliberately the small one.

### Phase 6: `[Reins-Insurer-Orientation]`

Note item 1. INSURER splits the layering analysis into two blocks with layers down the rows: the contract (`share`, `limit`, `attach`, `pr_attach`, `pr_detach`, `pr_loss`, `lol`, `output`) and the consequence (`freq`, `sev`, `agg` each with `mean`, `cv`, `skew`). RAW keeps the current orientation. So 2 blocks raw, 3 insurer. Pure pandas over `reins_stats_df`.

**Aggregate only, discovered on execution.** A `Portfolio`'s `reins_stats_df` is a different frame: indexed `(view, measure)` with units across, no `component` level and no layer axis at all. There is no layering to turn over, so its insurer view is untouched. The block names in the open questions below were adopted as written.

### Phase 7: `[Calibration-Survival-Noise]`, struck

**Removed, 2026-08-11.** `a250` `[Reins-Density-Fuzz]` landed the fix at the cause rather than at the symptom: `reins_density_df` now calls `remove_fuzz` like every other density frame, so the view path feeds `_calibration_survival` the same shape of input the default path always did. The rest of what this phase proposed, replacing the bare assertion with a toleranced exception, was ruled against in the same release: the assertion states a property that follows from "this is a pmf", it was right to stop, and relaxing the test that noticed a real bug is the wrong direction. Nothing is owed here.

### Phase 8: `[Pricing-Result-Objects]`

> Superseded 2026-08-12 by `dev/plan-pricing-exhibits.md` phase L1; kept for the record.

The API change `[Calibrate-Allocate-Evaluate]` needs, and the precondition for phases 9 and 10.

`calibrate_distortions` on both classes returns a `CalibrationResult` carrying `distortion_df`, `calibration_df`, `distortions` and the inputs it was called with (`coc`, `p` or `a`, `kind`, `reins_view`). **Breaking**: it currently returns a bare `distortion_df`. The two frames are split across a return value and instance state today, which is precisely why calibration cannot be exhibited and why its output reads as part of pricing.

`evaluate` on all three classes returns an `EvaluationResult` carrying the panel plus the premium it was measured against. **Breaking**, same reason: a bare DataFrame cannot be dispatched on.

All result objects gain a `_source` back reference (the `PnL._source` precedent) so `_title_name` resolves and a title reads `Calibration: Book` rather than `Calibration: CalibrationResult`. `PricingResult` gains the `_relabel` its two siblings already apply at construction.

### Phase 9: `[Reins-View-Pricing]`

> Superseded 2026-08-12 by `dev/plan-pricing-exhibits.md` phase L4; kept for the record. Both decisions it left open are taken there: the frame grows the octet (the author's target exhibit shows it), and the difference rows ride under INSURER per the perspective ruling below.

**Mostly already built, which I did not know when this plan was first drafted.** `Aggregate.reins_price_df` and `Portfolio.reins_price_df` price every view of a cession with every calibrated distortion, returning `(distortion, view)` rows with columns `a` / `el` / `bid` / `ask` / `margin`. Each view resolves its own `a = q(p)`, so the comparison holds the threshold fixed rather than the capital, exactly as the api intends. It covers all five views where the api's `_REINS_BASES` covers three. `reins_view=` landed at `a223` and `reins_price_df` at `a224`; both were unusable on most program shapes until `a250` swept the fuzz out of `reins_density_df`, and verified working across twenty five combinations after it.

So this phase is not a new method. What is left is two decisions and a registration.

**The frame is a quote, not a pentagon.** `reins_price_df` gives `el` / `bid` / `ask` / `margin`; the api's table gives the full octet `L, M, P, Q, a, LR, PQ, ROE` via `Pentagon.solve` per row. The octet is strictly more, and completing it is what `_pentagon_row` does in the api today. Decide whether `reins_price_df` grows the octet, or the Allocate exhibit completes it in the exhibit layer, or the api's octet was more than the leaf needed.

**The library and the api disagree about differencing, and the library is explicit.** `run_reins_price` emits `"{basis} less {name}"` rows and calls the gross minus net difference the allowance for reinsurance in the rate. The `reins_price_df` docstring says the opposite in as many words: "Views are separate distributions, not a decomposition, so a gross price less a net price is a comparison of two programs rather than the price of the cession. **The `ceded` row is the price of the cession.**" `Portfolio._reins_view_density` repeats it. This is a modeling disagreement rather than a formatting one and it has to be settled before the exhibit is designed, because it decides whether the Allocate table carries difference rows at all.

### Phase 10: `[Pricing-Exhibits]`

> Superseded 2026-08-12 by `dev/plan-pricing-exhibits.md` phase L5, whose block lists differ from the sketch below in two ways the author specified: `pricing.calibrate` carries `distortion_df` only (`calibration_df` moves to Allocate, and the pentagon becomes the live preview line on the form, not a table), and direct registration on `AnalyzeDistortionsResult` / `AnalyzeDistortionResult` / `PricingResult` is out of that plan's scope. **Bounds stays here** and is now this plan's only open work.

Register the three leaves on the result objects. No registry machinery changes, no `inputs=`, no schema, no signature change to the eleven existing builders.

**Calibrate**, on `CalibrationResult`. RAW is `calibration_df` and `distortion_df`. INSURER captions the target for what it means and the per family receipt for what was fitted.

**Allocate**, on `AnalyzeDistortionsResult` (unit axis) and on the phase 9 result (view axis). RAW is the frame whole. INSURER splits the unit frame per stat, which is exactly the api's four `stat_*` tables, and those `FORMATS` entries delete. `AnalyzeDistortionResult` and `PricingResult` register here too; the `PricingResult` scalars `price`, `a_reg` and `reg_p` ride in `meta` rather than as a one row table.

**Evaluate**, on `EvaluationResult`. RAW is the panel. INSURER carries the reading of `gini_p` and the `status` column, and the `DegenerateEvaluationWarning` distinction between a position unacceptable at any stress and one that cannot lose.

**Bounds**, on `Bounds`, which is registered for nothing today: the cloud, weight and tvar frames, deleting the api's `bounds.py` assembly.

The app's flow becomes validate the form, call the library method, ask for the exhibit. The inputs are the arguments of the method the caller already makes, and the library validates them and raises properly, so nothing is re-validated in a schema that could drift.

## What deletes on the api side

`tables.FORMATS` entirely, the two sharpen ledes, `pricing.py`'s document assembly and its `_pentagon_row` / `_pentagon_diff` / `_sub` pentagon arithmetic, the `_BasisView` shim and `_REINS_BASES` (**both deletable now**, on a sync to `a250`, ahead of anything else here), `bounds.py`'s assembly, `_round_pnl_premium`, and any hand written chart doc rehydrator. The frame route survives only for `density_df` and `reins_density_df`, which is agreed and not outstanding.

## Open questions

**Settled: the Allocate table carries the difference row, because the difference is a perspective.** Author ruling, 2026-08-11. The `ceded` row is how the **seller** priced the cession: a distortion applied to the ceded distribution, which is a position in its own right. `gross less net` is what the **buyer** feels they paid, the drop in their own cost of risk from buying it. Both are real and they are not equal, and which one is "the price" depends on which side of the treaty you stand.

That is not a formatting choice, it is exactly what :class:`Perspective` is for, and it is the first genuine use for the two members declared at 1.0 and left unimplemented. The buyer of reinsurance is the cedent, so the felt price is the **INSURER** reading; the seller's price is the **REINSURER** reading. So the Allocate exhibit is one frame under two perspectives rather than one table with an argument, and `[Perspective-May-Restructure]` carries it.

**Deferred to the pricing discussion, author 2026-08-11: only `RAW` and `INSURER` exist.** `_IMPLEMENTED_PERSPECTIVES` names those two and `exhibit_frames` refuses the rest, so serving the seller's reading as `REINSURER` is a framework change of its own: the enum gate, a ruling on whether an unregistered exhibit answers `REINSURER` at all (the INSURER default rule would silently make it equal `RAW` on all eleven existing exhibits, which is not what anyone means), and the perspective list on the wire. None of that belongs in phases 1 to 6, and it is not settled that the pricing leaves need it. Take it up when the redesigned Pricing pane is designed.

The two library docstrings that currently say differencing is "a comparison of two programs rather than the price of the cession" (`reins_price_df` and `Portfolio._reins_view_density`) are right about the *seller's* price and wrong to state it without the qualifier. They want a sentence naming the buyer's reading rather than a reversal.

**Settled 2026-08-12: `reins_price_df` grows the octet.** The author's target Allocate exhibit shows `a, L, M, P, Q, LR, PQ, ROE` per `(distortion, basis)` row, recorded as `[Allocate-Carries-The-Octet]` in `dev/plan-pricing-exhibits.md`. Since `el` is `L`, `ask` is `P` and `margin` is `M`, the frame adopts the pentagon vocabulary and completes `Q` and the ratios; `bid` is dropped entirely (author, same day: too confusing), so the columns are the canonical octet alone.

**Naming is already settled**, and I was wrong to raise it: `reins_price_df` exists and is the fourth `reins_*` frame, so there is no new name to choose and no collision with unit `allocation`.

**Settled: block names for phase 6** are `reins_layer_terms` and `reins_layer_moments`, adopted at `a256`. They are not the first block names with no frame attribute behind them after all: `economic_ratios` has served `amounts`, `ratios` and `legs` under INSURER since `a206`, and phase 5's `score_grid` joined them. The convention is narrower than it looked, and now stated on `Perspective`: a **RAW** block names a frame, an INSURER block names a reading.

**Does `CalibrationResult` supersede the instance attributes?** Phase 8 keeps `self.distortion_df` and `self.calibration_df` so nothing breaks twice. If they are meant to go, that is a second deprecation and should be decided now. *Carried forward as `dev/plan-pricing-exhibits.md` question 5, recommendation keep.*

**Settled 2026-08-12: Calibrate keeps neither.** The pentagon completed from the form inputs becomes the **live preview line** on the Calibrate form, debounced like Quick Re, recorded as `[Pentagon-Is-The-Preview]` in `dev/plan-pricing-exhibits.md`. `pricing.calibrate` carries `distortion_df` alone, and `calibration_df` moves to the Allocate exhibit as its degenerate single-view case. The caller who wants the pentagon without calibrating has it directly: that is what `price_pentagon` is, and the preview route serves exactly that call.
