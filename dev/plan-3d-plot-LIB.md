# plan-3d-plot-LIB: the two library edits the joint surface is waiting on

Status: **READY TO EXECUTE, written 2026-08-21**, anchors read against
`1.0.0a306`. Both items were ruled by the author on 2026-08-12 and neither is
built. Nothing in `aggregate_api` or its SPA changes when they land, verified
the same day this was written, so this is library work end to end and no round
note is owed.

The three-party plan is `aggregate_api/dev/done/plan-3d-plot.md`, symlinked
here as `dev/done/plan-3d-plot.md`. It moved to `done/` when the app half
finished; its sections **5.1.2** and **5.8** are the specifications, carry the
measurements and are the reasoning this document does not repeat. Read them
first. What follows is the execution sheet: the edits, the tests that move, the
acceptance checks, and one consequence for the library's own renderer that the
canonical plan does not name because it was written from the app's side.

This is also the file `dev/TODO.md` has pointed at since a258. That pointer was
dangling: the review notes it promised were folded into the canonical plan's
section 5.0 instead, and no such file was ever written.

## 1 What already landed

Two bumps, both in `charts/_emit_bivariate.py` unless noted.

* **a257 `[Joint-Density-Clip]`**: both 2-D FFT de-fuzz sites route through one
  helper that warns on a large negative rather than preserving it, on a floor
  relative to the mass the grid carries. This was section 5.3.
* **a258 `[Joint-Surface-Contract]`**: the display coordinate became the
  block's **first** fine cell rather than its last (5.1); the window is
  measured on the fine lattice and cropped **before** the block factor is
  chosen, worth a factor of fourteen in resolution (5.2); and the surface block
  carries both lattices as origin, step and count, plus `bs`, `k`, the realized
  window, the exact marginals, the fine-lattice means, the deficit and `z` as a
  declared-dtype base64 block (5.4 to 5.6).

Section 5.0 of the canonical plan records the seven items and the five points
where the shipped code departs from what the plan drafted. Additive throughout,
so `CHART_IR_VERSION` stayed at 2.

## 2 The ruling that scopes what is left

The author answered both open questions on 2026-08-12. The app recorded them at
`aggregate_api` a78 and they are in the canonical plan's header.

1. **Section 4.2.1: do not trade the full grid rule for the payload.** The
   derived quantities stay the app's, and the library serves a complete grid to
   compute them on, with `window` as the drawing range inside it. Specified in
   full as **5.8**.
2. **Section 5.1.1: take the representative point**, so the coordinate becomes
   the point a block's mass actually sits at and `edge = "mid"` becomes
   literally true. Specified as **5.1.2**.

**Section 4.2.1.1 is closed as declined by the same ruling.** The library does
not grow a third lattice carrying the total's density and kappa. Do not build
the `total: {t0, dt, nt, k, density, kappa}` block that section costs out, and
do not add `cond_mean`. The reasoning for the ask, its measured cost and the
recommendation to take it are all still in that section; the author read them
and chose the complete grid instead. Should it ever be reopened, the note there
about a reduced conditional mean being a ratio of two block sums rather than a
block sum of fine conditional means is the part worth keeping.

## 3 `[Joint-Surface-Representative-Point]`: the coordinate becomes the point the mass sits at

### The defect

A display cell covers fine atoms at `a, a + bs, ..., a + (k - 1) * bs`. The
emitter labels it `a` and declares `edge='left'`, which invites a consumer to
add `dx / 2` to reach a midpoint when the truth wants `(k - 1) * bs / 2`. The
gap is `bs / 2` on every axis. Measured on `Indep`'s y axis, where `k = 2`, a
mean read off the declared coordinate is off by -0.2649 display buckets and a
mean read off a cell midpoint by +0.2350, against **-0.0150** for the
representative point. Seventeen times better, and the residual is window
truncation rather than convention.

### The edit

`charts/_emit_bivariate.py`, in `_joint_surface`:

* line 311, `display_x = xs[lo_x:hi_x:kx]` becomes
  `display_x = xs[lo_x:hi_x:kx] + (kx - 1) * bs_x / 2`, and line 312 the same
  for y. `x0` and `y0` follow, being the first display coordinate.
* line 341, `edge='left'` becomes `edge='mid'`. `SURFACE_EDGES` in
  `charts/ir.py:194` already carries `'mid'`, so nothing in the IR moves.
* lines 344 to 349, the `window` box. The outer edges of the outer cells become
  `(x0 - dx / 2, x0 + (nx - 1) * dx + dx / 2)` rather than
  `(x0, x0 + nx * dx)`, and the same for y. Section 5 of this document moves
  these two lines again, which is expected: the convention moves them here and
  the decoupling moves them there.
* the docstring, lines 269 to 284. The paragraph beginning "The coordinate is
  the block's first fine coordinate" is now wrong in its first sentence and
  right in its history, so rewrite the claim and keep the account of what a258
  replaced. The paragraph after it, on the residual bias being intrinsic and
  one-sided, needs the sharper statement: under the representative point the
  residual is the deviation of the within-block mass from uniform, which is
  second order and **either sign**, rather than up to a whole bucket low.

Nothing else moves. `dx`, `dy`, `nx`, `ny`, `bs`, `k`, `z`, `kept`,
`marginals`, `moments` and `deficit` are all untouched, which is what makes
this a clean diff to read.

### One consequence to state rather than discover

The first display cell of a positive-support law then extends to `-bs / 2`,
which looks like support below zero and is not. The fine lattice already does
this and `pcolormesh` already draws it for any centered grid in this library,
so it is consistent rather than new.

### Tests that move

All in `tests/test_chart_surface_pilot.py`.

* `test_display_grid_starts_where_the_fine_lattice_does` (line 112): `s.edge`
  becomes `'mid'`, and the two origin assertions become
  `s.x0 - (s.k[0] - 1) * s.bs[0] / 2 == float(indep.axis_xs[0][0])` and the y
  twin, which is 8.1's phrasing. The step assertions are unaffected.
* `test_display_mean_bias_is_bounded_by_one_bucket` (line 131): its bound
  `-step < grid_mean - fine_mean <= 0` is the old convention's guarantee and
  the sign half of it no longer holds. Make it two-sided and tighter, and
  rewrite the docstring, which explains the one-sidedness. The measured
  residual on `Indep`'s y axis is -0.0150 display buckets, so a two-sided bound
  of half a bucket is honest and still catches a regression.
* `test_window_reports_the_mass_it_kept` (line 173): the two box equalities at
  lines 181 and 182 take the mid form above, and the comment naming `'left'`
  goes with them.
* `test_low_edge_snaps_to_zero_on_positive_support` (line 201): `s.y0 == 0.0`
  becomes `s.y0 == (s.k[1] - 1) * s.bs[1] / 2`, since the snap places the first
  fine cell at the origin and the coordinate now names that block's center.
* the module docstring, line 4, says "the left-edge coordinate convention".

There is **no committed chartdoc baseline for this chart** to regenerate:
`tests/data/chartdoc_baselines/` holds `agg.png` and `distortion.png` only.
The document hash changes, so any fixture captured downstream is stale, which
app side means one re-capture of `aggregate_api/dev/fixtures/charts.json`.

### Acceptance

* `x0 - (k - 1) * bs / 2 == xs[lo]` exactly, both axes, at every `detail` the
  pilot parametrizes.
* the display mean of `Indep`'s y axis lands within half a display bucket of
  `moments['mean'][1]`, either side, where today it is 0.2649 low.
* `nx`, `ny`, `k`, `dx`, `dy` and `z` are byte-identical to the previous
  release for the same call, which is the check that this edit touched the
  coordinate and nothing else.

## 4 `[Joint-Surface-Whole-Grid]`: the window stops being the grid

### The defect

The emitter crops before it emits, so the served grid **is** the window,
`window.x` and `window.y` come back equal to the lattice bounds, and there is
nothing outside them for a consumer to compute on. Every conditional the app
forms is then normalized by the visible mass, which is a different and less
interesting object whose mean moves whenever the window does. Measured by the
prototype's `check-kappa-window.js`: at a useful depth a third of a cut at
constant total is on screen, and kappa comes out 4.9 to 7.8 percent wrong with
the error flipping sign along the total, so it bends the shape of the curve the
chart exists to show.

### The edit

Same function. `_axis_plan` is **not** touched: the block factor is still
chosen from the cropped extent, which is 5.2 and is what makes `detail` mean
something. What changes is what the reduction is applied to.

1. **Reduce the whole fine axis, not the crop.** Line 310 becomes
   `z = _reduce(_reduce(density, kx, axis=0), ky, axis=1)`, with
   `display_x = xs[::kx]` and `display_y = ys[::ky]` at lines 311 and 312,
   carrying section 3's `(k - 1) * bs / 2` offset. The blocking anchors at
   index 0, so a law supported from the origin is drawn from the origin.
2. **Pad a ragged tail rather than dropping it.** `_align` currently keeps the
   crop inside `top = n - n % k`, so a fine axis whose length is not a multiple
   of `k` loses its last partial block. Pad the fine axis with zeros to a whole
   multiple of `k` before reducing: the spacing stays uniform, which every
   interpolation downstream depends on, and mass is preserved exactly, the pad
   being outside the support. On today's objects both axes are FFT grids and so
   powers of two, so this is belt and braces rather than a live bug, and it is
   cheaper to write now than to diagnose later on the first object that is not.
3. **`window` becomes the drawing range.** Lines 344 to 349 keep `p` and
   `kept` exactly as they are and stop being computed from `x0` and `nx`: the
   box is the data bounds of the quantile window, now a sub-rectangle of the
   lattice, taken from `lo_x`, `hi_x`, `lo_y`, `hi_y` in the convention section
   3 leaves in force. `kept` keeps its meaning to the letter, the fraction of
   the mass inside those bounds, and is now a sum over that sub-rectangle
   rather than over the whole array.
4. **The marginals go on the whole lattice**, lines 322 and 323, so
   `_reduce(marg_x, kx)` rather than `_reduce(marg_x[lo_x:hi_x], kx)`. This is
   not optional: `SurfaceData.__post_init__` in `charts/ir.py` refuses a
   marginal whose length differs from its axis, so a whole-grid axis with a
   cropped marginal raises at construction.
5. **The `window` field's docstring**, `charts/ir.py` around line 458, says
   "the resulting outer edges in data coordinates". That was true when the box
   was the grid. It becomes the drawing range inside a larger lattice, and the
   sentence should say so, since this field is now the only thing telling a
   renderer which part of the mesh is the subject.

Everything else is untouched: `bs`, `k`, `deficit`, `moments`, the encodings,
and section 3's representative point, which applies to the whole lattice the
same way it applies to the crop.

### What it costs

The whole reduced grid, at the `k` the window chose, from the canonical plan's
table: Clayton 256 x 256 against 75 x 78 today, `Indep` 64 x 8192 against
31 x 116, IndepFreq 128 x 512 against 82 x 108, IndepSigned 128 x 256 against
66 x 126. `Indep` is the honest worst case, a Lomax on a lattice wide enough to
hold its tail: 524k cells, 2.1 MB as float32 before transport compression. The
author ruled 2026-08-12 that size is not the constraint here and `detail` is
the lever if it becomes one. If it does, the knob to reach for is **not** a
crop: it is a `context` parameter saying how far beyond the window to carry,
with the whole grid as its default, so the choice is stated in the document
rather than made silently by the emitter.

### Tests that move

* `test_window_is_taken_before_the_reduction` (line 153): `s.ny == 116` becomes
  false by design, and the invariant it defends survives. Re-express it: `s.dy`
  is still 8.0, two fine buckets rather than 128 of them, which is the whole
  point of choosing `k` from the crop; the window box now spans about 116
  display cells, `(window['y'][1] - window['y'][0]) / s.dy`; and `s.ny` is the
  whole reduced axis. The comparison against the whole axis reduced to 128
  cells first stays, and is now a comparison of two `dy` values rather than of
  two axis lengths, which reads better anyway.
* `test_window_reports_the_mass_it_kept` (line 173): `kept` is now a sum over
  the sub-rectangle, not over `z` entire. `z.sum() / indep.density.sum()`
  becomes `1 - deficit` to 1e-12, and that is worth asserting in its own right.
* `test_window_zero_is_the_whole_grid` (line 192): still true, and no longer
  the distinguishing fact, since every depth emits the whole lattice. What
  separates `window=0` from `window=4` is now the box and the step: `k` comes
  from the crop, so a deeper window gives a finer `dy`. Say that here, because
  it is the one place the two knobs are visibly independent.
* `test_low_edge_snaps_to_zero_on_positive_support` (line 201) and
  `test_low_edge_does_not_snap_on_signed_support` (line 210): the snap is a
  property of the window, not of the emitted axis, so the assertions move from
  `s.y0` and `s.x0` onto `s.window['y'][0]` and `s.window['x'][0]`.

### Acceptance, from 5.8's own list

* `build_chart_doc(indep, 'joint_surface', window=4)` returns a y axis of about
  8192 cells with `window.y` naming a sub-range of roughly 116 of them, rather
  than a 116 cell axis whose window is the whole of it.
* `sum(z)` is the joint's whole placed mass, `1 - deficit`, to 1e-12, rather
  than `kept * (1 - deficit)`.
* `window.kept` still matches an independently computed sum over the crop to
  1e-9, which is now a statement about a sub-rectangle rather than about the
  whole array.
* both marginals are as long as their axes, and the x marginal is unchanged by
  a change in the y depth.

## 5 The library's own renderer, which the canonical plan does not cover

**A finding, 2026-08-21, and the one thing here that is not in the plan.** The
plan was written from the app's side, and the app clips: `windowRange` in
`web/src/charts/surface-geometry.js` reads `window` as a sub-rectangle inside
the grid and falls back to the whole grid when a document declares none. The
library's own renderer does not.

`plots/_chartdoc.py` `_z_grid_panel` draws `surf.x`, `surf.y` and `surf.z`
entire with `pcolormesh`, then pins the limits to the mesh extent at line 277
so an overlay cannot widen them. It reads neither `surf.window` nor
`surf.edge`. So on the day section 4 lands, `plot_chartdoc(build_chart_doc(bv,
'joint_surface'))` starts drawing `Indep` as 8192 cells of mostly empty tail
with the interesting box a sliver at the origin, which is the picture the
window exists to prevent. `tests/test_chart_surface_pilot.py` renders these
documents at lines 436 and 449, so the suite exercises the path.

Two ways to close it, and they are a real choice.

* **Renderer only.** `_z_grid_panel` sets `xlim` and `ylim` from `surf.window`
  when the document declares one, falling back to the mesh extent when it does
  not. Zero wire change, zero effect on any other consumer, and it fits the
  limits discipline already in that function. This is the conservative choice
  and the one to take absent a ruling.
* **Document level.** The emitter populates `ChartAxis.suggested_range` on the
  two plane axes from the window box, and the z-grid panel honors it through
  the `_axis_window` and `_apply_axis` route every other panel kind already
  uses, whose docstring says exactly this: the window is the emitter's answer
  to which slice of the grid is worth looking at. It adds no field, and it puts
  the meaning in the document where every renderer sees it, which is the house
  principle. It is **not** app-neutral: `chartdoc-to-echarts.js` `axisWindow`
  reads `suggested_range`, so the flat heatmap reading of this document would
  start honoring it too. That is probably an improvement and it is still a
  behavior change the app did not ask for, so it wants the author's word and a
  line in the round note that would then be owed.

Either way, note that `pcolormesh` is called with `shading='nearest'`, which
treats the coordinates as **cell centers**. Against today's `edge='left'` that
is a half display bucket of displacement in the library's own picture. Section
3 makes the renderer's existing assumption true rather than merely close, which
is a second reason to land it first.

## 6 Order, bumps and cadence

**Two bumps, section 3 then section 4**, one commit each, each with its own
CHANGELOG section, per the house rules. Not one bump carrying both.

The reason is the diff. Section 3 moves every coordinate by `(k - 1) * bs / 2`
and moves no shape: `nx`, `ny`, `k`, `z` and the marginals come out identical.
Section 4 moves every shape and, of itself, no convention. Landed separately,
each produces a diff with one cause, and the oversight condition on this work,
that the coordinate shift be read deliberately rather than skimmed, is
satisfiable. Landed together, a reader cannot tell which change moved what.

The renderer fix of section 5 rides with section 4, since that is the bump that
makes it necessary, unless the author takes the document-level route, in which
case it is its own bump and its own round note.

When both have landed, move this document and the symlinked canonical plan into
`dev/done/` together, tick `dev/TODO.md`, and tell the app: its only chore is
re-capturing `dev/fixtures/charts.json` and re-running `smoke-charts.mjs`.

## 7 What does not change, said out loud

* **`CHART_IR_VERSION` stays 2.** No field is added, removed or retyped.
  `edge` takes a value the vocabulary already carries, and `window` keeps its
  shape and its meaning while its relationship to the lattice changes. Phase
  two, dropping the `x`, `y` and `z` arrays in favor of the lattice fields and
  the encoded block, is the breaking change that bumps it to 3, it waits on the
  SPA having moved, and it is not this work.
* **`_axis_plan`, `_block_factor`, `_window_indices` and `_snap_to_zero` are
  untouched.** The window is still measured on the fine lattice, `k` is still
  chosen from the crop, and the snap still applies to the window.
* **The kappa emitter is untouched.** `chart_kappa` shares the module and
  nothing else.
* **The app and the API are untouched.** Verified 2026-08-21: the decode reads
  `mid` already, `windowRange` already treats the window as a sub-rectangle,
  and the two API round-trip tests for the `detail` and `window` knobs run
  unskipped and pass, the emitter having honored both since a258.
* **Nothing here needs a round note**, on the current reading. The one path
  that would is the document-level option in section 5.

## 8 Verified facts, 2026-08-21

Read against `1.0.0a306` and `aggregate_api` `1.0.0a114`.

* `charts/_emit_bivariate.py`: `_axis_plan` at 168, `_reduce` at 213,
  `_joint_surface` at 226, the two `_axis_plan` calls at 307 and 308, the
  reduction at 310, `display_x` and `display_y` at 311 and 312, `kept` at 317,
  the two display marginals at 322 and 323, `SurfaceData` from 334, `edge` at
  341, the `window` dict at 344, the marginals at 350, `moments` at 354.
* `charts/ir.py`: `SURFACE_EDGES = ('left', 'mid')` at 194, the `window` field
  documented from 458, the marginal length check in `__post_init__`.
* `plots/_chartdoc.py`: `_z_grid_panel` from 231, `pcolormesh` with
  `shading='nearest'` at 259, the limits pinned at 277, `_axis_window` at 101
  and `_apply_axis` at 287, neither reached from the z-grid path.
* `tests/test_chart_surface_pilot.py`: the eight cases named in sections 3 and
  4, and the renderer cases at 436 and 449.
* `tests/data/chartdoc_baselines/` holds two files, neither for this chart.
* App side: `web/src/charts/surface-geometry.js` `windowRange`,
  `web/src/charts/surface-grid.js` decoding `surface.window` as
  `quantileWindow`, and `web/src/charts/chartdoc-to-echarts.js` `axisWindow`
  reading `suggested_range`.
