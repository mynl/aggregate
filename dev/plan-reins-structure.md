# Plan: reinsurance structure diagram (`structure` chart, `tower` panel kind)

Status: revised after review against the code at `1.0.0a348`, 2026-09-25. The
scope rulings below are decisions, not proposals. Author agrees and signs off
plan. Companion plan: `aggregate-api/dev/plan-structure-chart.md` (SPA
translator support; do not start it until this plan lands, per the sequencing
ruling below).

## Goal

A broker-slide style reinsurance program diagram emitted from a DecL program:
a gross block on the left, a per-occurrence tower and an aggregate tower of
layer rectangles on loss axes, each layer labeled with its terms and, when the
object is built, its economics and risk statistics. Optionally each tower is
joined to the matching quantile (Lee) curve on a shared loss axis, so every
attachment reads off as a return period. The prior art is the author's
tranching diagram (`PMIR_StudyNote/python/archive/tranching-problem.ipynb`,
`structuring_diagram`): a narrow block panel beside a quantile curve with
hairline rules carrying each boundary across to the curve, boundary ticks in
currency instead of a continuous axis.

## Current behavior

There is no structure diagram. The reinsurance chart surface today is
`charts/_emit_reins.py` (`'reins'`): gross, ceded and net densities per claim
and a Lee diagram for the year. The program's *shape*, its layers, shares,
attachments, retention, gaps and premiums, is visible only as text
(`reins_description`) and tables (`reins_summary_df`, `reins_stats_df`).

`dev/TODO.md` carries this as `[Reinsurance-Structure-Diagrams]` (#45), whose
row still reads "under-specified; confirm source / scope (PMIR code?)". This
plan answers the scope question; rewrite the row when the plan is ruled.

## Scope rulings (settled with the author, do not reopen)

- **Aggregate and PnL only.** The emitter registers on `Aggregate` and on
  `PnL`, where it unwraps `.engine` and requires a single `Aggregate`.
  Portfolio is out of scope for this plan; the registry mechanism leaves it
  open for a later plan with no code debt.
- **The towers always draw; the Lee curves are optional sugar** behind a
  `lee=True` emitter argument. The un-updated object supports geometry only.
- **Label selection is explicit, not a numeric LOD.** An `annotate` argument
  takes a tuple of field names from a documented vocabulary. The emitter
  renders the selected fields in one fixed canonical order regardless of the
  order passed. Default tuple: `('geometry', 'premium', 'el', 'lr')`.
- **The argument is `annotate`, not `labels`** (ruled 2026-09-25 after the
  review found the collision). `labels` is already a property on both host
  classes, the `LabeledMixin` view over `label_map` whose own documented
  example is `a.labels.occ_reins[0]`, the per-layer `as` names. A
  `labels=('geometry', 'premium')` argument on a reinsurance chart would read
  as a request for layer names, which is the one thing it does not mean.
  `annotate` is free: it occurs nowhere in `aggregate` as an attribute,
  method, keyword argument or DecL word, only as matplotlib's `ax.annotate`
  inside renderer code. The module constant is `DEFAULT_ANNOTATE`.
- **Quota share is an aggregate-stage word.** `share po inf xs 0` on the
  aggregate stage is labeled "`<share>` quota share". The identical clause on
  the occurrence stage produces the same cession but keeps its literal
  `po inf xs 0` wording. One name means one thing.
- **Share is width.** A layer's rectangle spans `[0, share]` of the tower
  width; the unplaced remainder `[share, 1]` draws as a co-participation
  block (grey or hatched). This is the market-slide convention.
- **Output is IR only.** The emitter produces a `ChartDoc`; the matplotlib
  renderer (`plots/_chartdoc.py`) and the SPA translator each learn the new
  panel kind. No direct-SVG path: matplotlib already saves SVG, and a second
  emission format would be a third rendering path to maintain.
- **The library lands first, and the SPA dark window is accepted** (ruled
  2026-09-25). `web/src/charts/chartdoc-to-echarts.js` pins
  `CHART_IR_VERSION = 2` and returns `null` for anything newer, so between
  this plan landing and the companion landing the SPA draws **no** charts at
  all, not merely no structure chart. That is the pin-and-check contract
  working as documented in `docs/3_reference/3_x_API_Stability.rst`, and the
  author accepts the window rather than pre-bumping the SPA's ceiling. State
  the window in the CHANGELOG entry so a reader of the history knows why the
  SPA went quiet for one version.

## Design

### Data sources (nothing new is computed)

- Geometry: `spec['occ_reins']` / `spec['agg_reins']` lists of
  `(share, limit, attach)` plus `occ_kind` / `agg_kind` (`'net of'` /
  `'ceded to'`). Available un-updated. `_validate_reins_layers` guarantees
  ordering and non-overlap; gaps are legal and drawn as gap blocks.
- Per-layer economics from the spec when the DecL priced its layers:
  `occ_reins_premium` / `agg_reins_premium` entries (deposit / rol / rate
  resolved by `_program._pnl_layer_premium`), ceding commission, reinstatement
  schedules. The stored premium is **placed** (`rol` resolves to
  `val * share * limit`); the diagram shows **100% terms**, so the emitter
  divides by share once, in one place.
- Per-layer display names come from the label surface, `agg.labels.occ_reins`
  and `agg.labels.agg_reins`, not from the raw `occ_reins_label` spec key the
  parser emits. The `_LabelView` already returns `None` for a missing site,
  which is exactly the fallback the geometry line needs, and it is the one
  place the `as` clause is resolved.
- Risk statistics from `reins_stats_df` (per-layer meta and moment rows at
  both stages) and densities from `reins_density_df`. Note that
  `reins_stats_df` carries `mean` and `cv`, and **no `sd` row**: the `sd`
  annotation field is `mean * cv`, computed in the emitter.
- Fallback pricing: an optional `distortion` argument routes through
  `reins_price_df` for layers the DecL did not price. Spec premium wins when
  both exist.
- **Reference distributions.** The occurrence tower reads against the gross
  severity. The aggregate tower reads against **`p_agg_subject`**, the
  aggregate of the *requested occurrence output*, which is what
  `reins_stats_df` itself uses for every aggregate-stage `pr_attach`,
  `pr_detach` and `lol` (`_reinsurance.py:1155`). It is `p_agg_net_occ` under
  an `occurrence net of` program and `p_agg_ceded_occ` under an
  `occurrence ceded to` one, so naming either column directly would be right
  for half the programs and silently wrong for the other half. Getting this
  wrong misstates every attachment probability on the aggregate tower. The
  Lee panels use the same pairing.

### IR change: panel kind `'tower'`, `CHART_IR_VERSION` 2 -> 3

- Add `'tower'` to `PANEL_KINDS`. The word is deliberate: DecL already spells
  a layer-list shorthand `tower <doutcomes>` (`decl.lark:562`), so the panel
  kind and the language agree on what a tower is.
- A tower panel's `x_axis` is a placement axis with `unit='ratio'`, range
  `[0, 1]`, drawn without ticks. `'ratio'` is the existing `AXIS_UNITS` entry,
  documented as "a unitless share"; do **not** add a `'share'` unit. The
  panel's `y_axis` is a loss axis in currency.
- New frozen dataclass `TowerBlock`: `panel_id`, `x0`, `x1`, `y0`, `y1`,
  `role`, `label` (the block's headline), `label_lines` (tuple of the
  selected annotation fields, already formatted), `open_top` (bool, for an
  unlimited layer). Carried on a new `ChartDoc.blocks` tuple, empty for every
  existing chart.
- New module constant `BLOCK_ROLES = ('layer', 'retention',
  'co_participation', 'gap', 'gross')`, declared and commented beside
  `SERIES_ROLES` and `MARK_ROLES` and under the same "open vocabulary,
  documented additions only" discipline, and validated in
  `TowerBlock.__post_init__` the way `Mark` validates `orient`. Note that
  `'gross'` will exist in two vocabularies, as a series role and as a block
  role; they are different vocabularies and the overlap is intended.
- **`_ALWAYS` needs a `TowerBlock` entry.** `_canonical` looks the type up
  with a bare subscript (`ir.py:1198`), so a dataclass absent from the table
  raises `KeyError` rather than defaulting. The always-emitted fields are
  `('panel_id', 'x0', 'x1', 'y0', 'y1', 'role')`: a zero coordinate is a real
  reading and must not be omitted as a default.
- **`human_strings` must walk the blocks.** `ChartDoc.tex` is required to be
  total over `human_strings(doc)`, and `complete_tex` *raises* on a `typeset`
  key the document does not expose (`ir.py:1390`, `ir.py:1422`). Each block's
  `label` and every entry of its `label_lines` is a human-facing string, so
  `human_strings` grows a blocks pass and the emitter builds its `tex` through
  `complete_tex` as every other emitter does.
- Boundary ticks and the tower-to-Lee hairlines are **`Mark`s**, which exist
  already: horizontal marks at each layer boundary labeled with the currency
  amount, `faint=True` for the joining rules. No new mark machinery. This is
  the first shipped emitter to set `faint`, so drop the "No shipped emitter
  sets it at present" sentence from the `Mark` docstring (`ir.py:952`). The
  matplotlib renderer already honors it (`plots/_chartdoc.py:712`).
- Bump `CHART_IR_VERSION` to 3. Consumers pinning 2 refuse the new documents
  rather than mis-drawing them, which is the versioning contract working. See
  the sequencing ruling for what that costs the SPA for one version.

### Emitter: `charts/_emit_structure.py`, chart id `'structure'`

- `chart_structure = _emitter_base('structure')`, registered for `Aggregate`
  and `PnL`. Availability: either reins slot is populated (reuse the
  `_emit_reins._has_occurrence` pattern, widened to both slots).
- Signature: `_structure(obj, annotate=DEFAULT_ANNOTATE, lee=False,
  distortion=None)`. Emitter arguments are chart *content* options in the
  sense the `charts/__init__` docstring allows; nothing renderer-facing.
- **An option that cannot be honored raises**, following the house pattern at
  `_emit_bivariate.py:353`: `lee=True` or a `distortion` on an un-updated
  object raises `ValueError` naming `update()` as the fix. Silently dropping a
  requested panel is the failure mode the plan is guarding against, and the
  geometry-only chart stays available under the `lee=False` default, so the
  raise costs a caller nothing they did not ask for.
- Panels, left to right: `gross` (a one-block tower: the gross slab labeled
  with name, exposure, and when built mean / SD / CV / percentiles), `occ`
  tower, optional `occ_lee` xy panel, `agg` tower, optional `agg_lee` xy
  panel. A stage with no program contributes no panels. Tower and its Lee
  panel share the loss axis (the y axis); the Lee panel's x axis is the
  return-period-paired probability axis exactly as in `_emit_reins`.
- The vertical window: with a built object, `loss_window` on the reference
  distribution, as `_emit_reins` does. Un-updated with an unlimited top
  layer: the finite tower top (max `attach + limit` over finite layers) plus
  fixed headroom, and the unlimited block draws to the window top with
  `open_top=True`.
- Annotation vocabulary (fixed canonical order): `geometry` (the
  `share po limit xs attach` line, or the quota-share wording, or the
  layer's own resolved label), `premium` (100% terms), `el`, `lr`, `rol`,
  `lol`, `sd` (`mean * cv`), `pr_attach`, `pr_exhaust`, `reinstatements`,
  `cede`. Fields whose source is absent (not built, no premium) are silently
  omitted, so one `annotate` tuple works at every enrichment tier.

### Matplotlib renderer: `plots/_chartdoc.py`

- Add `'tower'` to `_NATIVE` and a `_render_tower_panel`: filled rectangles
  from `TowerBlock`s (`fill_betweenx` as in the tranching notebook), boundary
  yticks from the panel's marks, centered label stacks, open-top blocks drawn
  without a top edge, x axis hidden. Width ratios: tower panels narrow, Lee
  panels wide (the notebook's 1:2 reads well).
- Honest degradation per the module's own rule: a renderer without the kind
  says so; no silent fallback to xy.

## Files

- `src/aggregate/charts/ir.py`: `PANEL_KINDS`, `TowerBlock`, `BLOCK_ROLES`,
  `ChartDoc.blocks`, `_ALWAYS`, `human_strings`, the `Mark.faint` docstring,
  `CHART_IR_VERSION = 3`.
- `src/aggregate/charts/_emit_structure.py`: new.
- `src/aggregate/charts/__init__.py`: export and register.
- `src/aggregate/plots/_chartdoc.py`: tower rendering.
- `tests/`: see acceptance.
- Docs, which carry the IR version as a literal and must move with it:
  `docs/2_aggregate_overview/features.rst` (two places, and it derives from
  `dev/FEATURES.csv` via `dev/regen_features.py`),
  `docs/2_aggregate_overview/pipeline-exhibits-and-charts.rst`, and the
  surrounding prose in `docs/3_reference/3_x_API_Stability.rst` and
  `docs/3_reference/3_x_Charts.rst`. Per `CLAUDE.md` the `.rst` edits land in
  lockstep and the author rebuilds the doc tree outside the loop.
- `dev/TODO.md` (`[Reinsurance-Structure-Diagrams]`), `CHANGELOG.md`, version
  bump per house rules.

## Acceptance

- `available_charts` answers `'structure'` exactly when a reins slot is
  populated, on `Aggregate` and on `PnL`-wrapping-one-`Aggregate`; a
  `Portfolio` never answers it.
- Un-updated object: geometry-only document, no stats fields. `lee=True` or a
  `distortion` on that object raises `ValueError` naming `update()`.
- Built object with DecL-priced layers: premium lines show 100% terms and
  reconcile to the spec's placed premium divided by share; `distortion`
  fills only unpriced layers.
- Aggregate-stage `x po inf xs 0` labels as quota share; the same clause on
  the occurrence stage does not.
- Aggregate tower `pr_attach` values reproduce the `('agg', 'layer.k')`
  `pr_attach` row of `reins_stats_df` exactly, on **both** an
  `occurrence net of` fixture and an `occurrence ceded to` one, which is what
  pins the `p_agg_subject` basis rather than one column that happens to
  coincide. `decl-testers.agg:653` (`EV.CedeTower`) is the `ceded to` case.
- `canonical_json` round-trips a document with blocks, and `doc_hash` of the
  reloaded document equals the original's.
- A document with no blocks canonicalizes to its pre-change bytes **but for
  `ir_version`**: `blocks=()` equals its default and so is omitted by
  `_canonical`, while `ir_version` is in `_ALWAYS` and therefore always
  emitted, which means every existing document's hash moves with the bump.
  Assert the field-level claim (no `blocks` key present), not hash equality,
  which cannot hold and must not be written as though it could.
- `complete_tex` succeeds on a document whose only human-facing strings live
  in its blocks, proving `human_strings` walks them.
- Image-gated mpl render of one occ + agg + Lee example alongside the
  existing chart image tests.

## Execution log

Phases, one bump each: `[Tower-Block-IR]` (the IR schema), `[Structure-Emitter]`
(`_emit_structure.py`), `[Tower-Renderer]` (matplotlib). The SPA dark window
spans all three, until the companion plan lands.

### Rulings taken at execution, 2026-09-25

Four of the plan's factual premises did not survive review against
`1.0.0a348`. The author ruled on each before any file was touched.

- **Economics come from `PnL.economics`, and only a `PnL` has them.** The plan
  read per-layer premium from `spec['occ_reins_premium']` /
  `spec['agg_reins_premium']`. Those keys never reach an `Aggregate`: a plain
  `agg` strips the ceded-premium and cede clauses with
  `IgnoredDecLClauseWarning` ("a plain 'agg' has no premium context"), and on
  the P&L path `Underwriter._resolve_reins_economics` pops them from the pnl
  spec into `PnL.economics`, whose `pc_occ_by_layer` / `c_occ_by_layer` /
  `pc_agg_by_layer` / `c_agg_by_layer` lists are the per-layer figures at the
  placement share. So `premium`, `lr`, `rol`, `cede` and `reinstatements` are
  **P&L-only** annotation fields. A built `Aggregate` still gets geometry, the
  risk statistics, and `lee=True`, because `reins_stats_df` and
  `reins_density_df` are its own (author's ruling: "aggregate should get lee
  too, that is determinable from an agg"). The `_by_layer` lists can be absent
  even on a P&L, since a scalar-API or zero-fallback economics dict carries
  only the side totals, so the emitter reads them defensively.
- **No `distortion` argument, anywhere.** The plan routed fallback pricing
  through `reins_price_df`, which is indexed `(distortion, view)` over the five
  `reins_views` and carries no per-layer row; per-layer distortion pricing does
  not exist in the library, so the argument would have been new computation
  against the plan's own "nothing new is computed". Author's ruling: "this is
  about input pricing, not derived pricing."
- **The `ceded to` acceptance fixture is `J.Re11`, not `EV.CedeTower`.**
  `decl-testers.agg:653` is `occurrence net of ... aggregate net of`
  (`occ_kind == 'net of'`), so it cannot pin the `p_agg_subject` basis.
  `_test_suite.agg:201` `J.Re11` (`occurrence ceded to 15 xs 5 poisson
  aggregate net of 20 xs 0`) is the minimal `ceded to` case with an aggregate
  stage; `J.Re07` is the multi-layer one.
- **The annotation field is `pr_detach`, not `pr_exhaust`.** `pr_detach` is the
  `reins_stats_df` meta row the field reports, and one concept takes one name.

### Divergences

- `load_chart_doc` grows a `blocks` pass. The plan's `ir.py` file list omitted
  the function, but its own acceptance criterion requires the round trip, and
  the loader pops only `panels` / `axes` / `series` / `marks`, so blocks would
  have returned as plain dicts with `TowerBlock.__post_init__` never run.
- `TowerBlock` joins `ir.__all__` and the `charts/__init__` re-export list, as
  every other IR dataclass does.
- `TowerBlock.__post_init__` validates `x0 <= x1` and `y0 <= y1` as well as
  `role`. A reversed rectangle is an emitter bug that otherwise draws nothing
  and reports nothing.
- `dev/FEATURES.csv` carries "``CHART_IR_VERSION`` is 2" as a literal in the
  charts-module comment row, which is the source `features.rst` regenerates
  from, so it moves with the bump.
- The per-layer label is read with `.get(k)`. The plan expected `_LabelView` to
  return `None` for a missing site; `labels.occ_reins` is a plain dict and a
  missing index raises `KeyError`.
- A gap has two spellings, an implicit hole between consecutive layers and an
  explicit zero-share layer (`_validate_reins_layers` allows both), and the
  emitter draws a `'gap'` block for each.
