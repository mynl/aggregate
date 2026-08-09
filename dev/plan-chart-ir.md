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
