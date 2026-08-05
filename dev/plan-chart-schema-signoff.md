# Plan [Chart-Schema-Signoff]

> **Status: DRAFT for approval, written 2026-08-05. Not executed.** Closes the pass two sign off gate in `dev/plan-chart-ir.md`. No chart conversions land here; the remaining seven resume once this clears.

## Settled by the author

1. The twelve plot bivariate panel must be expressible: **grid panel overlays** plus an **`iso_total`** series role.
2. The matplotlib renderer **draws the axis labels the IR carries**.
3. Labels get a **plain text form plus an optional TeX companion**, direction my call, decided below.
4. Image gate pins confirmed: matplotlib **3.10.9**, RMS tolerance **2.0**.

## The six judgment calls, all closed 2026-08-05

`J1` **agreed**: the twin return period axis is IR. `ChartAxis.reciprocal_of` stays.

`J2` **agreed, with the reading sharpened**: the IR carries the mark's description, not just its position, so `1-in-250` is the library's word and not the renderer's formatting of a number. Satisfied by the existing `Mark` fields (`at`, `label`, `role`, `faint`); no new field. Visibility remains renderer view state.

`J3` **agreed**: right edge labels for the reduced surface grid. Fidelity and level of detail matter, so the center versus right edge divergence against 1-D `bin_density` is recorded as a post pilot reconciliation item rather than papered over.

`J4` **agreed**: camera, texture and lighting stay renderer side in the per chart override dict, with the down the diagonal angle preserved. Real artists ship.

`J5` **agreed**: one float dust floor at 1e-15.

`J6` **agreed**: band series stay. `ChartSeries.y2` stays.

**Schema freeze.** With these closed and the three changes below made, version 1 of the field list is **signed off and closed to additions**. The guardrail in `dev/plan-chart-ir.md` is now the operative rule: a chart the schema cannot express either changes the schema visibly, through a fresh author decision, or stays bespoke and is listed as bespoke. No further additions ride in on a conversion.

## Decision on item 3: plain text is primary, TeX is a lookup

`ChartDoc.tex`, one optional dict mapping a plain string to the renderer ready TeX string. A renderer that can typeset does `doc.tex.get(s, s)`; a renderer that cannot ignores the field entirely and is still correct.

Why this way round: the **required** field must be the one every renderer can consume, or an ECharts only consumer inherits markup it cannot parse. Plain text is also the legend identity the app links series toggles by, so it must stay stable and hashable. One dict covers series names, axis labels, mark labels, panel titles and the document title at once, so no future string field needs a twin. It defaults empty, is omitted from the canonical form at default, and therefore **moves no existing hash**.

Rejected alternative: per field twins (`name_tex`, `label_tex`, `title_tex`, ...). More discoverable locally, but five fields instead of one and it churns the schema every time a string field is added.

Stored values are exactly what matplotlib consumes, delimiters included (`'$\\check g(s)$'`), so a renderer never has to guess where the math starts and an emitter can mix text and math in one string.

**No validation rejecting `$` in plain fields.** It reads as an obvious guard and it is wrong: a currency axis is legitimately labeled `Loss ($)`. The rule is documented and reviewed, not enforced.

## Phase one [Chart-Grid-Overlays]

No visual change to anything that exists, so it lands on its own.

- `ir.py`: `ChartDoc.__post_init__` currently refuses an x/y series on a grid panel (`ir.py:412`). Relax to: a grid panel carries **at most one** surface series **plus any number** of x/y series drawn over it, in document order; an `xy` panel still refuses a surface.
- `ir.py`: `SERIES_ROLES` gains `'iso_total'`, the level set of x + y, labeled through the series name (`Sum = 10,000`).
- `plots/_chartdoc.py`: `_render_grid_panel` draws overlay series after the mesh, so the mpl side cannot silently drop them.
- `tests/test_charts_ir.py`: a representability test that hand builds the twelve plot bivariate panel (a density grid, eight iso total lines, equal aspect) and asserts it validates. Representability only, per the 1.0 scope. No emitter.

Diagonals ride as two point line series rather than a new `Mark` orientation: no new geometry concept, the label is already the series name, and the renderer needs no capability it will not need anyway for the surface contour overlay.

`J5` lands here too, since it is a constant and not a look change: `LOG_FLOOR = 1e-15` joins `constants.py` beside the other display constants, and the renderer's private `_LOG_FLOOR` (`plots/_chartdoc.py:29`) collapses onto it. `constants.py` rather than the charts package because the `plots/` compositors need the same value and are not chart IR code, so importing from constants leaves the charts and plots dependency direction untouched. Name checked free across `src/aggregate`. Emitters use it as the gap floor as each conversion lands; the compositors converge chart by chart under the image gate, exactly as the inventory recommended, so nothing outside the charts lane changes appearance in this plan.

## Phase two [Chart-Plain-Text-Names] with [Chart-Axis-Labels]

These two land **together in one commit** because both move the same pinned baseline image, and one regeneration is honest where two would be noise.

Names:

- `ir.py`: `ChartDoc.tex` added; the docstring states the plain text rule as a schema rule, not a style note.
- `charts/_emit_distortion.py`: the dual's name becomes `ǧ(s)` with `tex` carrying `'$\check g(s)$'`. U+01E7 is present in DejaVu Sans (verified) and ECharts renders Unicode natively.
- `plots/_distortion.py`: the compositor has **two** names for the same curve today, `$g\check$` at line 86 and `Dual {label}` at line 91. Settle on one, matching the emitter.
- `plots/_chartdoc.py`: typeset through `doc.tex.get(s, s)`.
- The app adapter needs no change: it never reads `tex`, which is the point.

Axis labels:

- `plots/_chartdoc.py`: draw `ChartAxis.label` on both axes, through the tex map.
- `plots/_distortion.py`: the compositor gains the same two labels, so **one** baseline still pins both sides of the seam. This is a visible change to `Distortion.plot` and gets a CHANGELOG line of its own.
- Regenerate `tests/data/chartdoc_baselines/distortion.png` with `python tests/test_chartdoc_render.py --regen`; both render tests keep passing against the single regenerated image.

Alternative, if you would rather not touch the compositor's look: two baselines, the compositor frozen on the old one and the renderer pinned to its own. It weakens the seam check to a historical assertion, so I recommend against it, but it is a one line change to this plan.

## Gate pins, recorded

`tests/test_chartdoc_render.py` keeps matplotlib 3.10.9 (other versions skip rather than fail on font metrics) and RMS tolerance 2.0 against a measured conversion residual of 0.05. Confirmed, no longer provisional; the CHANGELOG says so when this lands.

## Cadence

Two version bumps, two commits, in order: `[Chart-Grid-Overlays]` then `[Chart-Plain-Text-Names]`. Each carries its CHANGELOG section and its `dev/TODO.md` update. Versions are taken at the commit moment, since the exhibits lane is bumping in parallel on the same branch.

After this, the conversion queue resumes: the distortion's paired app commit (which needs the adapter's `xy` realization), then reins triple, sev, agg, pnl, port, bvagg heatmap.
