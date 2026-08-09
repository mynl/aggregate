# Plan [Chart-IR]: a chart intermediate representation, semantics vs realization

> **Release status: additive side project, does NOT gate `1.0.0b1`.** `aggregate.charts` ships marked **provisional in the sense of PEP 411**: public and encouraged, but not part of the 1.0 API contract, and free to change in a minor release with no deprecation period. That covers the IR schema, the canonical form and therefore the document hash, the registry, and the set of charts that exist; `CHART_IR_VERSION` is how a consumer detects a change. Dependencies point inward, so **1.0 ships whether or not this plan is finished.** The one edge into pre-existing code is the per-chart conversion of a bespoke plot, each gated by a before-and-after image diff (`tests/data/chartdoc_baselines/`) and **deferrable chart by chart past 1.0**, so an unconverted chart is a valid shipping state rather than a loose end. Explicitly post-1.0: conversion of the charts the app does not use, and full matplotlib-renders-the-IR convergence. Live items sit in the *Provisional modules* section of `dev/TODO.md`, outside the beta gate. Status recorded in the module docstrings, `docs/3_reference/3_x_API_Stability.rst`, and the `CHANGELOG.md` preamble.

> **Status: DRAFT, approved 2026-08-04 for phased execution.** Three passes, each lands independently; matplotlib rendering the IR is the committed post 1.0 convergence. Companion plan: `dev/plan-exhibits.md`.

## Principle

The boundary is semantics vs realization, not data vs display. IR (library side): series (name, role, values), axes (label, scale, suggested range), chart kind, subplot and shared axis structure, annotations that carry meaning. Log or not is statistical meaning, not styling. Renderer side: colors, fonts, theming, hover, sizing. The app gets ONE generic IR to ECharts adapter plus small per chart override dicts: a translator, not a matplotlib recreation.

Honest statement of where this starts: the app draws 8 charts, all client side, and the semantics of every one live in JavaScript. `twoPanelData()` (aggregate_api `web/src/charts/exhibits.js:687`) is already a de facto IR, its docstring says so, and the app TODO records the intent to lift it into a real IR with adapters. This plan lifts that seam into the library, where chart meaning joins table meaning ([Exhibits-Module]).

## Placement and names

The IR must not import matplotlib (`plots/__init__.py` imports plt at line 26, and `tests/test_plots_boundary.py` enforces the boundary), so it cannot live under `plots/`. New package `src/aggregate/charts/`:

- `charts/ir.py`: frozen dataclasses (no pydantic, aggregate's dependency set is unchanged), `CHART_IR_VERSION = 1`, hand rolled `canonical_dict` / `canonical_json` / `doc_hash` mirroring greater_tables' determinism and naming (sorted key UTF-8 JSON, sha256 first 12 hex).
- `charts/_emit_*.py`: per class emitters, pure pandas and numpy.
- `charts/__init__.py`: the public surface plus `available_charts(obj)` mirroring the exhibits capability mechanics.
- `plots/_chartdoc.py`: the one generic mpl renderer, inside the mpl boundary. plots may import charts, never the reverse; test_plots_boundary.py gains the assertion (importing aggregate.charts does not import matplotlib).

Names vetted: `ChartDoc` (mirrors TableDoc), `ChartSeries` (not Series, pandas clash), `ChartAxis` (not Axis, matplotlib clash), `Panel` (subplot slot), `Mark` (meaningful annotation with a role), `SurfaceData` (the z grid for the pilot). Roles and kinds are documented string vocabularies (like GT's RowFlag literals), not enums, so version 1 grows without schema churn. Nothing star exported; users write `from aggregate import charts`.

## Pass one [Chart-Inventory] (lands alone, no code)

Deliverable `dev/chart-inventory.md`: one row per drawing; human review splits each cell into semantic (goes in the IR) or incidental (the renderer's business). The table is the requirements doc for the schema pass.

Rows: the 8 app charts (agg, port, sev, pnl two panel; reins triple with gross, ceded, net and the survival accumulated client side today; distortion g(s) unit square; bvagg heatmap; bvagg 3-D surface with its block summed display grid), and every plots/ layer 2 compositor plus the pedagogy and ft entries. The out of scope charts are inventoried anyway because the schema is designed against the full inventory, not the easy subset.

`plot_twelve` (`pedagogy.py:1611`) is inventoried **per panel**, twelve rows, not one row for the composed figure. Several of its panels are expected to end up in aLL, the kappa panels and the two unit independence views in particular (author, 2026-08-05), but the aLL designs are not settled and are deliberately not developed before this IR exists: they will be authored directly as ChartDoc emitters once designed. The composed twelve panel figure itself stays bespoke. Panel map from the docstring: (1,1) density, (1,2) log density, (1,3) bivariate density; (2,1) kappa, (2,2) alpha, (2,3) beta; (3,1)/(4,1) per unit S, gS, alphaS, beta gS; (3,2) margin density, (4,2) cumulative margin; (3,3) stand alone M, (4,3) natural M. Most are take-columns-and-plot x-y panels; the inventory's data input column must name the supported accessor per the note below.

Data source note (correct as of numerics-2): Portfolio `density_df` **no longer carries per unit `p_<unit>` columns** (removed at numerics-2; see the `unit_density` docstring, `_portfolio.py:2660`). Supported sources: `unit_density(unit, view)` (native grid, read from the owning Aggregate), `unit_density_df(view)` (long form, `(unit, loss)` MultiIndex), and `aligned_unit_density_df(grid='total')` as an explicitly labeled display adapter (presentation only, never compute input). The `exeqa_*` kappa columns remain on `density_df`; the alpha, beta, S, gS, margin and M layer curves come from `allocation_diagnostics(distortion, ...)`; `independent_density_df` holds the independent combine. Emitters read these accessors, never legacy total grid `p_` columns; in particular the port two panel emitter reads `unit_density_df`, as the app route already does.

Columns: data inputs (which frame or GridDistribution, which slice), panels and shared axes, scales, reference marks, window logic, series naming, and the incidental column (colors, hover, legend toggling, twin axis cosmetics). Judgment calls surfaced for the author inside the table: the twin return period axis (semantic pairing or renderer trick), reference line toggles (view state or IR), display grid binning (semantic, since mass preservation is meaning).

Commit `[Chart-IR] aNNN: chart inventory` after author review of the semantic vs incidental split.

## Pass two [Chart-Schema] (lands alone, sign off gate)

`charts/ir.py` plus tests, no emitters. Sketch, finalized against the inventory:

- `ChartDoc`: ir_version, name, title, panels, axes, series, marks, meta.
- `Panel`: id, kind ('xy', 'surface', 'heatmap'), axis ids, series ids, read_axis ('x' or 'y', capturing that the tail panel is read probability to loss).
- `ChartAxis`: id, label, scale ('linear' or 'log'), suggested_range, kind, unit ('currency', 'probability', 'return_period', 'density').
- `ChartSeries`: name, role ('density', 'survival', 'identity', 'gross', 'ceded', 'net', ...), panel_id, x, y, or surface: SurfaceData for grid kinds.
- `Mark`: panel_id, orient, at, label, role ('mean', 'break_even', 'capital_anchor'), faint.

Guardrail: no `extra_mpl_kwargs` and no renderer passthrough of any kind. A need the schema cannot express changes the schema visibly, or the chart stays bespoke and is listed as bespoke.

Representability check: the field list is proven on paper against the twelve plot panels (multi series per unit x-y families off `density_df` exeqa columns and `allocation_diagnostics`, the paired S/gS curve families, the bivariate density panel) without scheduling their conversion. If a panel the author expects in aLL cannot be expressed, the schema changes before sign off, not after.

Author signs off on the field list before any emitter is written. Commit `[Chart-IR] aNNN: chart IR schema v1`.

## Pass three [Chart-Conversions] (one chart per commit)

Each conversion is one emitter `chart_<name>(obj, **semantic_options) -> ChartDoc`, the shared mpl renderer growing exactly what that chart needs, and the acceptance gate. Order after the pilot, easiest semantics first: distortion g(s); reins triple; sev; agg; pnl; port two panel; bvagg heatmap.

Acceptance gates, two harnesses because the before side differs:

- mpl side (where a plots/ compositor exists): `tests/test_chartdoc_render.py` using `matplotlib.testing.compare.compare_images` against committed baselines. Pin the baseline mpl version and RMS tolerance at gate setup (author confirms both). Pixel near identical or the conversion does not land.
- app side (where the current truth is twoPanelData): `capture_fixtures.py` captures ChartDoc payloads; `smoke-exhibits.mjs` runs the legacy build and the ChartDoc adapter over the same fixtures, diffing the resulting ECharts option JSON with incidental keys stripped.

## Pilot [Chart-Surface-Pilot]: the 3-D surface

Chosen because ECharts renders it well and mpl does not, which forces the IR to be designed from semantics rather than matplotlib's vocabulary, and forces renderer capability declaration.

- Emitter `charts/_emit_bivariate.py: chart_joint_surface(bv, display_log2=None) -> ChartDoc`. Pulls the joint matrix, block sums to the display grid inside the emitter (migrating surfaceGrid's mass preserving reduction from the app's surface.js: reduction is meaning), labels the axes from the component names (migrating axisNames), z axis linear with `meta['z_log_ok'] = True` declaring the log toggle meaningful; one surface series.
- App: `GET /v1/objects/{oid}/chart/joint_surface` returning canonical_json with the doc_hash ETag (the frame route pattern). One generic `chartdoc-to-echarts.js` adapter walks panels, axes, series, marks; a small per chart override dict carries the echarts-gl specifics (visualMap ramp, camera). surfaceOption becomes the override dict plus the adapter; surfaceGrid deletes.
- mpl renderer declares capability per panel kind: for 'surface' it renders the 2-D pcolormesh projection with a contour overlay (the honest 2-D reading of a z grid) and stamps "(projection)" in the title; `strict=True` raises `ChartCapabilityError` instead. This establishes the capability declaration pattern.

The 2-D charts from the inventory then become confirmations rather than the template the schema overfits to.

## Scope and graduation

1.0 scope: the 8 app charts only, plus schema representability (not conversion) of the twelve plot panels. The pnl emitter reads GridDistributions directly (absorbing the app's `serializers.pnl_density_frame`); the sev emitter absorbs `severity_density_frame`'s log spaced sf inversion; the port emitter reads `unit_density_df`. Listed as bespoke and untouched until after 1.0: pedagogy.py (including the composed twelve panel figure), the ft.py illustrations, the bounds plots, scatter and sample_compare, the massive bivariate pyramid, plot_distortion_affine. Known, temporary duplication of plotting decisions between plots/ compositors and emitters is the price of shipping.

Charts that do not exist yet, the kappa chart and the two unit independence view for aLL in particular, are deliberately not developed app side first. They wait for the IR and are then born as ChartDoc emitters; that ordering is the point of doing the IR now.

Post 1.0 committed direction, not 1.0 work: converge the mpl compositors themselves onto the renderer, chart by chart, under the image diff gate.

Graduation path for experimental app side charts: born app side is fine; graduation means the semantic decisions move into an upstream emitter, the app keeps only the adapter and its override dict, and the app side builder deletes in the same app commit; the app TODO tracks each one.

## Rollout and cadence

Each library pass bumps `1.0.0aNNN` with a CHANGELOG section and a one line commit `[Chart-IR] aNNN: summary`; app commits follow the app's conventions.

1. **[Chart-Inventory]**: the inventory document, author review gate.
2. **[Chart-Schema]**: `charts/ir.py` plus boundary tests, author sign off gate.
3. **[Chart-Surface-Pilot]**: pilot end to end (library emitter, then the app route, adapter, and surface.js reduction removal in one app commit).
4. **[Chart-Conversions]**: one chart per library commit plus its paired app commit; a check-charts.py sibling of check-frames.py when the second chart lands.

Passes one to three are done (`a197`, `a198`, `a199`), the sign off gate closed 2026-08-05, and four charts are converted (`joint_surface` `a199`, `distortion` `a202`, `reins` `a210`, `severity` `a212`). **Part two below is the rest of pass four**, restated as one job per first-class class, and it changes the plan in two ways the original did not anticipate: `Object.plot()` itself moves onto the IR now rather than post 1.0, and the schema reopens a third time to carry the view toggles. Both follow from the app going purist. Read part two with `dev/note-all-chart-asks.md`, which is the app's written ask, and `aggregate_api/dev/plan-plot-ir-api.md`, which is its half of the work.

---

# Part two: every first-class class draws through the IR

> **Status: APPROVED, 2026-08-09.** Drafted with six open gates and settled the same day; every one of them is answered below and the answers are folded into the jobs. One gate was withdrawn as badly posed. Two answers changed the plan's shape: the aggregate keeps its Lee panel and loses its log panel to a toggle, and panel arrangement is the renderer's business, not the document's.

## What changed, and why the plan grew a second part

The original plan held one thing back deliberately: converting a `plots/` compositor onto the renderer was the committed **post 1.0** convergence, and an emitter shipped alongside its compositor, agreeing with it under an image diff, but never replacing it. That is still the state today. `Distortion.plot` draws through `plots/_distortion.py`, `chart_distortion` emits a document that renders pixel for pixel identically through `plots/_chartdoc.py`, and nothing in the library calls the second path. Four charts sit in that holding pattern.

Two facts closed it. The app is going **purist**: it deletes every app-side chart builder in one commit, so an object with no emitter has no picture at all, which makes `chart_agg` the critical path for the whole app rather than one item in a queue. And keeping a compositor and an emitter in step by hand, chart by chart, for a picture they already agree on to the pixel, is exactly the duplication the plan called "the price of shipping" when it was three charts and cheap. Across the whole first-class surface it is neither.

So the shape of a conversion changes. It was: write an emitter, prove it agrees with the compositor, leave both. It is now:

```python
def plot(self, ...):
    return plot_chartdoc(build_chart_doc(self, '<name>', ...))
```

with the compositor deleted in the same commit. One set of semantic decisions, two renderers, no third path. The image gate does not go away; it becomes the thing that licenses the deletion.

## The shape of every job

Each job below is the same seven steps. Only the semantics differ.

1. **Emitter.** `charts/_emit_<x>.py` defines `chart_<name>` as a `singledispatch` generic on the owning class and calls `register_chart`. Pure pandas and numpy, reads public accessors, never imports matplotlib.
2. **Renderer growth.** `plots/_chartdoc.py` grows only what this chart needs, and only in terms the schema already has. A need it cannot express is an author decision, not a keyword argument.
3. **Image gate.** Generate the baseline from the *current compositor*, then diff the IR rendering against it (`tests/test_chartdoc_render.py`, `tests/data/chartdoc_baselines/`, mpl 3.10.9, tolerance 2.0). The residual is measured and recorded in the CHANGELOG, as `a202` and `a209` did. A deliberate change to what is drawn is declared up front and the baseline is regenerated from the *new* picture after the author has looked at it, never quietly.
4. **Rewire.** The `plot` method becomes the two-liner above.
5. **Delete.** The compositor and its private helpers go in the same commit. `plots/__init__.py` loses the export; `dev/FEATURES.csv` is regenerated; any `.rst` naming the function is fixed in lockstep.
6. **Publish.** `available_charts(obj)` picks the new entry up with no extra work, since it is derived from the dispatch registry. The one addition is `primary_chart`, which says which chart is the object's *own* picture.
7. **Verify.** Tier 1 in the edit loop, tier 2 before declaring the job done, tier 3 at the bump. The image gate is not a substitute for either.

Each job is one bump, one CHANGELOG section, one one-line commit.

## The names (settled)

Vetted against the existing surface, per the standing rule, before anything is written.

The pseudo-code in the request was `ir = object.create_ir(); generic_plotter.plot(ir)`. The mechanism is right and already exists; the spelling follows the exhibits sibling rather than inventing a parallel one. Exhibits gave the first-class classes **no new methods at all**: `exhibits.build_exhibit(obj, name, perspective=...)` and `exhibits.available_exhibits(obj)` are module functions, and the object is an argument. Charts already has `available_charts(obj)` matching that. So the missing half is:

```python
charts.build_chart_doc(obj, name, **options) -> ChartDoc  # sibling of build_exhibit
charts.primary_chart(obj) -> str | None                   # which chart is the object's own picture
```

`build_chart` was the first draft and the author's objection retires it: the name does not say whether it renders a picture or produces a document, and in a module whose whole point is that those are different things, that is the one ambiguity worth spending four characters on. `build_chart_doc` says it, and it says it by naming the type it returns, which is the reason it wins over the `build_chart_ir` spelling considered alongside it. "IR" names the idea the module is built around, but `ChartDoc` is the thing in your hands, it matches `TableDoc` on the exhibits side, and a reader who has met one has met both.

`build_chart_doc` resolves the registry entry, checks the predicate, dispatches, and stamps the hash, so `stamp` stops being every emitter's job. Module function rather than a method on each class, for three reasons: it matches the sibling module, it keeps a `Bounds`-family chart reachable without deciding which of three classes owns it, and it adds nothing to any instance namespace, so the `self.approximate` shadowing hazard cannot recur.

`primary_chart` answers a question the app has to ask and cannot currently answer: `available_charts(agg)` returns `reins`, `severity` and (after job 2) the aggregate's own chart, and the Overview leaf needs to know which one is Overview. Registration gains `register_chart(name, emitter, predicate=None, primary=None)` where `primary` is the class, or tuple of classes, for which this chart is the primary picture. `reins` is primary for nothing; `severity` is primary for `Severity` and not for `Aggregate`.

Checked: no `build_chart_doc`, `primary_chart` or `create_ir` anywhere in `src/aggregate`, and no `chart` attribute or method on any first-class class. `Aggregate.figure` (the compositor's stashed figure) is the one nearby name, and it survives the rewire unchanged.

## `[Chart-Declared-Readings]`, the optional-feature switches (settled)

The third deliberate reopening of the signed-off v1 schema, after `support` at `a214`. It lands **before** the aggregate job, which needs it: under the ruling below the log panel becomes a declared reading, so without the declarations the aggregate chart cannot be written at all. The app also rebuilds its control strip from the document in the commit that adopts `chart_agg`, and building it twice is the waste this exercise exists to avoid.

The principle is unchanged and it settles the question of whether a toggle belongs here: **which readings a quantity admits is a fact about the quantity, not about the drawing.** A log reading of a heavy tail is meaningful; a log reading of a distortion's unit square is not. That is the same statement `scale` already makes, widened from one reading to the set of them.

Four fields, three of them promotions of hooks that already exist.

**1. Scales, per axis.** `ChartAxis.scales: tuple`, the scales this axis may be read on, with `scale` staying the default reading. `__post_init__` fills it with `(scale,)` when omitted, so it is always concrete and no consumer handles `None`. It goes in `_ALWAYS`: by the polarity rule the `support` bug taught, the value a consumer must act on is the *non-singleton* one, but a bare axis must still read as "fixed", and making that explicit costs nothing. This generalizes `meta['z_log_ok']`, which retires: the surface pilot's z axis declares `scales=('linear', 'log')` instead, `plot_chartdoc(log_z=...)` honors the declaration rather than the meta key, and `chartdoc-to-echarts.js:95` stops special casing it.

**2. Full range, per axis.** `ChartAxis.full_range: tuple`, the whole data extent, offered as the alternative to `suggested_range`. Presence is the declaration: an axis that carries both offers the toggle, one that carries only `suggested_range` has no other honest reading. Carrying the numbers rather than a boolean is deliberate. The extent of a log axis with an exact zero, or of a survival curve floored at `LOG_FLOOR`, is not the naive min and max of the series, and working it out is emitter knowledge. This is the field the distortion chart does not set.

**3. Reciprocal reading.** `ChartAxis.reciprocal_of` already exists, drafted at schema time as J1 and never yet emitted. This job gives it its first use and states the contract: the paired axis sits in `doc.axes`, is not named by any panel, points at the drawn axis, and its presence means "offer this reading". Confirmed against the app: `rightPairs` (`exhibits.js:388`) keeps loss on x in both modes and swaps only y between `S` and `1/S`, so this is one axis read two ways and `reciprocal_of` is the right shape. The app's own button title calls it a transpose, which its code is not; worth correcting app side.

**4. Panel realizations.** `Panel.kinds: tuple`, the realizations this panel supports, with `kind` the default, filled with `(kind,)` when omitted, in `_ALWAYS`, exactly parallel to `scales`. This answers the app's question 2 the way it hoped: `heatmap | surface` is **one document declaring two realizations**, not two registry entries that must be kept in step by hand forever. It also improves the capability pattern. `plot_chartdoc` currently renders a `surface` as a labeled 2-D projection and stamps "(projection)" as a confession; with `kinds` it picks a realization the document says is available, and only degrades when there is none.

**Surfacing rule, which the emitters must respect so they do not over-declare:** the app shows one control if **any** axis declares a reading, and applies it to **every** axis that declares it. Declaring `log` on an axis therefore asserts two things: that a log reading of it is meaningful, and that it is reasonable for it to move when its siblings do. A two-panel chart gets one "log y" button, not two.

**Canonical control order**, fixed app side as a house rule and recorded here so the emitters and the matplotlib renderer agree: log/linear, then full/zoom, then probability/return period, then heatmap/surface.

**Hash and version.** Adding two fields to `_ALWAYS` changes every document hash. That is a content address changing when content changes, which is what it is for; the app re-captures fixtures. `CHART_IR_VERSION` **stays 1**, on the rule that the version marks the point where a reader that ignores what it does not know would draw something *wrong*. A reader ignoring `scales`, `full_range` and `kinds` draws the default reading, which is correct and complete, so nothing is broken by ignorance. That rule goes into `ir.py` next to the vocabularies, because it is the question that will be asked at every future reopening, and `a214` set the precedent of not bumping.

## Panel arrangement: the renderer's, not the document's (settled)

The document says **which panels, in what order**, and stops there. Arrangement is realization: two panels side by side, four as a two by two, four in a row on a wide canvas, four stacked on a phone. That is a renderer looking at its own canvas, and no field here can know the canvas.

Order is a **hint**, and the only one. Panels come in the order they should be read, and a renderer is free to realize `ABCD` as one row, as `AB` over `CD`, or as whatever its space allows. So there is no `ChartDoc.layout`, no mosaic string, no rows and columns. The one arrangement fact that is genuinely semantic already has a home: `Panel.aspect = 'equal'` says a unit square must stay square, which is a statement about the reading and not about the canvas.

Consequences, both renderer-side work rather than schema work. `plot_chartdoc` currently puts every panel in one row (`_chartdoc.py:337`), which is fine for two and poor for four, so it grows a wrap rule of its own choosing. The app has the same job from the other side, replacing `twoPanelBox` and `squareBox` with a layout that reads panel count and aspect off the document. Neither needs permission from the other, which is the point of answering it this way.

## `ChartDoc.tex` is total (settled)

Every human-facing string in a document carries **both** forms: plain text, and TeX. The analogy is alt text in HTML. You write both because they serve different consumers, and you do not make one consumer guess. matplotlib uses the TeX form, ECharts uses the plain form, and the emitter that writes the string writes both.

This is stronger than what `ir.py` says today. The current docstring describes a **partial** lookup with a fallback (`ir.py:441`, implemented at `_chartdoc.py:107`), which was the right reading of the a209 design but is not the contract the author wants. Under totality a missing entry is an **emitter bug**, not a document saying "this string has no typeset form". So this job carries:

* the docstring rewritten to state totality, since the app asked in writing and the answer must not drift;
* the fallback kept, as a safety net that keeps a renderer honest rather than as a licensed state;
* a test asserting totality over every emitted document, which is what makes it a contract instead of a promise. A plain word maps to itself and the entry is still written, so the check is a simple set difference over the strings a document exposes.

The related point stands and costs nothing: the plain form does not have to be ASCII. `ǧ(s)` in `constants.py` is the instinct to keep, and Unicode carries most actuarial labels honestly.

## The nine jobs

Ordered as executed, per the author. The schema job goes first because the aggregate needs it. The aggregate goes next because it is the app's critical path and the landing demo. Distortion and P&L follow because they are nearly free once the aggregate exists. Portfolio and reinsurance go **last**, because both need design work before they can be written, and doing them early would mean writing them twice. Severity sits in the middle on the same condition: its drawing is expected to change too, so it holds its slot only while its design is ready, and otherwise joins the back.

### 1. `[Chart-Declared-Readings]`

The schema section above, plus tests. `charts/ir.py` only, no emitter changes beyond retiring `meta['z_log_ok']` from `_emit_bivariate.py` and teaching `plot_chartdoc` to read the declarations. Baselines are unaffected: nothing about the default reading changes.

### 2. `[Chart-Aggregate]`, the critical path

`chart_agg` for `Aggregate`, the app's landing demo, and the job that sets the pattern for everything after it.

`plot_aggregate` (`plots/_aggregate.py:24`) draws three panels as `'ABC'`: the density, the log density, and the Lee diagram, with the severity drawn alongside the aggregate on each, and separate discrete and continuous branches.

**Settled: two panels, the first and the third. The middle one becomes a toggle.** Panel A is the density and declares `scales=('linear', 'log')` on **both** its axes, because it is toggling log x *and* log y that turns panel A into panel B. So the old middle panel is not lost, it is one click away and the reader chooses it, which is what the declared readings are for. Panel C, the Lee diagram, stays a panel: loss against probability is a different reading, not a rescaling, and `read_axis` records that it is interrogated probability to loss. The probability axis carries `reciprocal_of`, which is the return-period reading, and which is today's `quantile_x='return'` keyword promoted from an argument to a declaration.

Two consequences, both accepted deliberately. **`Aggregate.plot()` changes what it draws**, so the image gate cannot pass against the old baseline and the new baseline is generated only after the author has looked at the new figure. And the app's current right-hand panel (survival against loss) is **not** what the library publishes; the Lee orientation is. The author does not like the app's present right-hand panel, the ruling is that the library decides, and the app adopts.

Also settled in this job: the discrete and continuous branches become `support='atomic'` plus the renderer's ladder, which is what `a214` built the ladder for; the severity companion series keeps role `density` and its own name; `xmax` becomes the x axis `suggested_range`, with `full_range` the whole grid.

### 3. `[Chart-Distortion]`

One panel, so the easy one, and the rewire is nearly all of it. `chart_distortion` shipped at `a202` and its image residual has been **0, pixel for pixel**, since `a209`, which means the deletion of `plots/_distortion.py`'s compositor is already licensed by a measurement rather than by an argument. `Distortion.plot` (`spectral.py:1675`) becomes the two-liner.

The one thing not to wave through: `Distortion.plot` takes `xs`, `n`, `both` and `plot_points`. Each argument is either a semantic option that moves to the emitter or a realization detail that goes, and that is decided argument by argument in the job.

`plot_distortion_affine` (`spectral.py:1005`, `:3135`) is **not** in this job. It stays bespoke and is listed as bespoke.

### 4. `[Chart-PnL]`

**Settled: a P&L delegates to the aggregate emitter over its own result grid**, so once job 2 exists this is a thin registration rather than a new chart. `plot_pnl` (`plots/_aggregate.py:164`) reads `PnL.result`, and that is a `GridDistribution` like any other.

What is genuinely its own, and is the whole of the work: the outcome axis is **signed**, so the window is not anchored at zero; the bad tail is the **low** end, so the cumulative panel reads `F` where a loss distribution reads `S`, which the app already knows (`exhibits.js:384`) and the emitter now owns; and the `break_even` mark at 0 is the reading, not decoration. No severity companion, since a P&L is an accounting result.

### 5. `[Chart-Severity]`

`chart_severity` shipped at `a212`, so the emitter exists, but **this is not a rewire and should not be planned as one: the severity plot is expected to change** (author, 2026-08-09). What it changes to is not settled here.

The starting position, for whoever writes it. `Severity.plot` (`_severity.py:1808`) draws **four** panels as `'AB\nCD'`; `chart_severity` emits **two**, the two-panel family that `charts/_two_panel.py` serves. Under job 2's ruling the scale twins among the four are declared readings rather than panels, which is the same collapse that took the aggregate from three to two, so the arithmetic says two panels and their readings. That is a starting position and not a conclusion, because the aggregate's answer turned on which readings the author actually wants in front of a reader, and a severity is a different object with a different set.

So the job opens by asking what the severity chart should be, exactly as `[Chart-Portfolio]` and `[Chart-Reins]` do, and it keeps this slot in the order only if that design is ready when its turn comes. If it is not, it moves to the back with the other two rather than being written twice.

### 6. `[Chart-Bounds]`

**Settled: two panels.** Left is the sampled cloud, shaded by weight, with the min and max envelope band. Right is the band again with the calibrated distortions on it, all five of them: CCoC, TVaR(p*), PH, Wang and Dual.

That is a deliberate consolidation. `plot_bounds_envelope` (`plots/_bounds.py`) draws **three** panels today and splits the five distortions across the last two, `['ccoc', 'tvar']` on one and `['ph', 'wang', 'dual']` on the other, which is an accident of how they were added rather than a reading anyone wants. One panel with five curves is the picture. Both panels are equal-aspect unit squares, so `Panel.aspect = 'equal'`, and the envelope band is a `y2` band series, which is the first real use of a field the schema has carried since v1. The average-extreme overlay rides on the right panel as it does today.

The other two drawings are their own emitters and their own commits if they need them: `plot_weights` (`bounds.py:513`), a level-set grid over `(p_lo, p_hi)`, so a `heatmap` panel; and `Bounds.plot` (`bounds.py:1000`), the `(items, P, max_t)` hull view.

Why it matters beyond the picture: `aggregate_api/bounds.py` renders `plot_envelope` to SVG server side, so once the app deletes its server-rendered figure route the Bounds tab is **the only reason matplotlib remains a dependency of the api at all**. This job removes an entire rendering pipeline, a dependency, and a class of "why does the server need a display" questions, in one commit on each side.

### 7. `[Chart-Bivariate]`

Two pieces. The `joint_surface` document declares `kinds=('surface', 'heatmap')`, which promotes the app's heatmap from a hidden WebGL fallback to a declared reading and gives matplotlib a native realization instead of a confessed projection. Then `BivariateAggregate.plot` (`bivariate.py:2770`), the two-panel contour of the joint severity and the joint aggregate, becomes an emitter over two grid panels, and `plot_slice` (`bivariate.py:3567`) is assessed: it is an x/y reading of a grid and may simply be an overlay on the same document rather than a chart of its own.

> **REMINDER, and the author asked to be prompted for it: punchups are wanted on the 3-D plot, and they have not been written down yet.** Ask for them when this job starts, before any emitter is written, because a punchup that lands after the document is designed is a schema question and one that lands before it is just a chart. The app's surface workstream also holds open items here (`aggregate_api/dev/api-punchlist.md`, Punchups Round 4 item 6).

The massive pyramid (`_bivariate_massive.py`) and the explorer (`_bivariate_explore.py`) stay bespoke and are listed as bespoke.

### 8. `[Chart-Portfolio]`

**Last, and blocked on design work the author will do first.** `plot_portfolio` (`plots/_portfolio.py:15`) draws the total plus each unit as linear and log density in two panels, which under job 2's ruling is one panel and a declared log reading. But the portfolio chart is not simply the aggregate chart with more curves on it: **something with the kappas is being added**, and what that panel is has not been designed. Writing the emitter before that design exists means writing it twice.

What is already known and does not need re-deciding: series are role `total` for the total, drawn last so it sits on top, and role `unit` for each unit, named by its resolved label so the app links a unit's legend toggle across panels by name; the data comes from `unit_density_df` and `unit_density(unit)` on the native grid, never a legacy `p_<unit>` column, which no longer exists; the κ columns are `exeqa_*` on `density_df`, and the allocation layer curves come from `allocation_diagnostics`.

The `plot_twelve` inventory (`dev/chart-inventory.md`) is the input here, not a separate exercise: its κ panel and its two-unit independence view are the panels the author expects to want, they were deliberately not built app side first, and this is the job where the first of them is born as an emitter.

### 9. `[Chart-Reins]`

**Last with the portfolio, and blocked the same way.** `chart_reins` shipped at `a210` and is registered for both `Aggregate` and `Portfolio`, so unlike the other converted charts the emitter is not the question. The reinsurance drawings are getting the same fine-tuning pass the portfolio chart is getting, and the emitter is rewritten to whatever that pass decides before anything is rewired or deleted.

Until then `chart_reins` stays as it is and keeps working. The app can adopt the current document whenever it likes; that adoption is not blocked by this job.

## What the app asked for that is already done

Recorded so it is not built twice. **`chart_reins` for `Portfolio` already exists** (`_emit_reins.py:238`), so the ask in `dev/note-all-chart-asks.md` and the blocked item in the app's TODO are both stale; the Reinsurance Plot leaf will light on a reinsured portfolio as soon as the app adopts the emitter. The note was written against `a210`, when the registration was `Aggregate` only.

## Rollout

Nine bumps, nine commits, house format, one line each, CHANGELOG carrying the detail.

```
[Chart-Declared-Readings] aNNN: an axis declares the scales and ranges it may be read on, and a panel the forms it may take
[Chart-Aggregate] aNNN: an aggregate emits density and Lee, and the log panel becomes a reading of the first
[Chart-Distortion] aNNN: a distortion draws through its own document and the compositor goes
[Chart-PnL] aNNN: a P&L delegates to the aggregate emitter over its signed result, read from the low tail
[Chart-Severity] aNNN: a severity emits its redesigned panels and the compositor goes
[Chart-Bounds] aNNN: the envelope is two panels, the cloud and all five calibrated distortions on one band
[Chart-Bivariate] aNNN: the joint density declares both its realizations, and the contour pair emits
[Chart-Portfolio] aNNN: a portfolio emits total, units and the kappa reading
[Chart-Reins] aNNN: the reinsurance document is redrawn to the round-3 design and the compositor goes
```

Version numbers are not pinned here: another workstream is bumping in this worktree, so each job takes the next free number at the moment it commits, per the parallel-commit protocol.

Paired app commits follow the app's own plan (`aggregate_api/dev/plan-plot-ir-api.md`) and its conventions. The one hard synchronization point is `[Chart-Declared-Readings]` landing before the app's item 6, and `[Chart-Aggregate]` before its item 4. Note the app's item 5 assumes the two-panel exhibit it draws today; job 2 publishes density and Lee instead, so the app adopts the library's orientation and that is worth telling them before they build the adapter's xy path around the other one.
