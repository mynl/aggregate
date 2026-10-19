# Plan: library figures fit a document's page and take its type

`[Library-Figures]`, in two steps: `[Library-Figures-Fit-Page]` and `[Library-Figures-House-Type]`.
Drafted 2026-10-19 at 1.5.3 (`d85f0c2`) at Steve's request, from the house-style review. Ruled
2026-10-19, executed as 1.6.0a2 and 1.6.0a3; see Rulings and Execution log at the foot.

This is the library half of a pair. The other half is
`D:\Projects\house-style\dev\plan-library-figures.md`, in which the house style's setup calls the
function this plan adds. This half lands first and is released to PyPI before that one executes.

## Goal

The writing projects under `D:\Projects` share one house style, a Quarto extension named `house`
(`D:\Projects\house-style`, kit 1.11.0). Its figures are drawn at the size they print: half, three
quarters or the full 6.3in text width, with labels and titles at 9pt and ticks and legend at 8pt,
in STIXGeneral to sit beside STIX Two text. A figure the document draws itself follows that. A
figure aggregate draws for the document, `a.plot()` or `port.plot()`, does not, and in a paper it
sits beside the house figures looking like a different publication.

After this plan, a document can tell aggregate how wide its page is, and every library figure
then fits that width and is drawn at its final size, so its type prints at the size the style
names. aggregate's own style moves to the house's sizes and face (Steve, 2026-10-19: "aggs own
figure face = STIXGeneral to match house").

## Current behavior

- **Nothing is applied at import.** `import aggregate` does not load matplotlib
  (`tests/test_plots_boundary.py`). `aggregate.style.use()` applies `data/aggregate.mplstyle`;
  `context()` and `rc_params()` are its companions (`plots/_style.py:74`, `:93`, `:126`).
- **Every library figure is given an explicit size,** so `rcParams['figure.figsize']` is never
  used. The sizes are multiples of `FIG_W = 3.5` and `FIG_H = 2.45` (`constants.py:53`), or
  literals:
  - `plots/_chartdoc.py` `plot_chartdoc` draws for `Aggregate.plot`, `Aggregate.reins_occ_plot`,
    `Severity.plot`, `Portfolio.plot`, the PnL `plot`, `Bounds.plot_envelope` and
    `Distortion.plot`. A single panel is `(FIG_W, FIG_H)`, a square one `(FIG_H, FIG_H)`, a heatmap
    or tower `(1.25*FIG_W, FIG_W)` (`:1634`). Several panels are `(n*FIG_W, FIG_H)` through
    `make_grid`, all-square `(n*FIG_H, FIG_H)`, or with towers
    `(0.5*FIG_W*sum(ratios), TOWER_HEIGHT*FIG_H)` (`:1673`). Only `Distortion.plot` accepts `ax`,
    and `ax` is refused for a multi-panel document (`:1616`).
  - `make_grid` and `make_mosaic` (`plots/_style.py:183`, `:146`) default to
    `(ncols*FIG_W, nrows*FIG_H)`.
  - Literal sizes elsewhere: the bivariate plots (`plots/_bivariate.py:65`, `:100`;
    `plots/_bivariate_massive.py:292`, `:366`), the bounds plots (`plots/_bounds.py:45`, `:84`),
    FourierTools (`plots/_fourier.py:65`, `:68`, `:206`, `:285`; `ft.py:53`, `:86`, `:205`, `:224`),
    `tweedie.py:930`, `:933`, `:947`, `plots/_portfolio.py:41`, and about sixteen functions in
    `pedagogy.py` (`:138` to `:1349`), from `(FIG_W, FIG_H)` up to `(18, 6)`.
- **So a two-panel `Aggregate.plot()` is 7.0 × 2.45in.** In a 6.3in text block Quarto shrinks
  the finished picture by 10%, type included.
- **Type sizes are fixed in code in about fifteen places,** and these override any style,
  including a document's:
  - `plots/_chartdoc.py`: `LABEL_POINTS = 0.65 * FONT_SIZE`, 5.85pt (`:305`), for block labels
    and tower tick labels and in the fit arithmetic (`:874`, `:939`, `:954`, `:1272`);
    `FONT_SIZE + 1` and `FONT_SIZE - 1.7` (`:1169`, `:1174`); `legend(fontsize='xx-small')`
    (`:1504`).
  - `plots/_bivariate_massive.py`: `fontsize=6`, `7`, `9` and `7` (`:211`, `:322`, `:327`,
    `:338`).
  - `legend(fontsize='x-small')` in `plots/_bounds.py:108`, `plots/_fourier.py:92`, `:262`,
    `:264`, `pedagogy.py:1037`, `:1062`; `legend(prop={'size': 7})` at `pedagogy.py:1753`.
- **`aggregate.mplstyle`:** `font.size: 8.5`; labels and ticks `small` (7.1pt), legend `x-small`
  (5.9pt), title `medium` and `semibold`; `font.sans-serif: Segoe UI, Helvetica Neue, Arial,
  DejaVu Sans`, so the face depends on the machine (Segoe UI exists only on Windows);
  `figure.figsize: 3.5, 2.45`. Its header says it was aligned to the aLL site in sans, replacing an
  older serif look.
- **What aLL takes from here.** aLL renders charts in the browser from the chart IR
  (`charts/ir.py`, which never imports matplotlib), so matplotlib styling does not reach it, with
  one exception. aggregate-api's `GET /v1/meta/style` (`routes/meta.py:95` to `153`) serves the
  style's colors and `font.size`, and `FIG_W` and `FIG_H` as the panel aspect. aLL's chart theme
  sets its base text at `max(10, font_size + 2)` pixels (`web/src/charts/theme.js:126`).
- **Tests.** `tests/test_chartdoc_render.py` compares pixels with baselines in
  `tests/data/chartdoc_baselines/` after `use()`. `tests/test_chartdoc_ladder.py` assumes today's
  panel width when it picks atom counts, because `plot_chartdoc` chooses stems, steps or lines
  from the axes' width in pixels (`_axes_width_px`, `:326`). No test checks a figure or type size.

### Against the 2026 decision to expose no plotting settings

`dev/done/plans-considered-and-rejected.md` rejected a `[plotting]` config section, and the
comment at `constants.py:48` repeats it: "users restyle via matplotlib's native mplstyle /
rcParams". The reasons were that every proposed key duplicated an rcParam, and that moving
default-argument constants is a broad refactor.

The page width is the one thing matplotlib has no rcParam for. `figure.figsize` is the size of a
single figure. In a Quarto document it is set from the format's `fig-width`, which the house sets
to half the text width. A library figure of two or three panels needs to know the whole width it
may take. The setting proposed here is a run-time call, not a config file key. It is off by
default, so nothing changes for anyone who does not call it. It lives in one helper. Steve rules
on it (question 1).

## The change

### [Library-Figures-Fit-Page] the next free minor

- **`aggregate.plots.fit_page(width=None, max_height=None)`.** It records a page width and an
  optional height cap, in inches, and `fit_page()` clears them. It lives in `plots/_style.py` and is
  re-exported from `plots/__init__.py`. The name was checked with `rg -w` and collides with
  nothing.
- **One private helper, `_fit_to_page(figsize)`, applied to every library figure's size** at every
  site listed above, so the rule lives in one place. With no page set it returns the size
  unchanged. With a page set it only ever shrinks a figure:

  ```
  s = min(1, width / w)
  w' = w * s
  h' = min(max(h * s, min(h, 2.0)), max_height)
  ```

  A figure narrower than the page keeps its size. A wider one is fitted to the page. Its height
  falls with it, but never below 2in, because a row of three 3.5in panels scaled to 6.3in would
  otherwise be 1.5in high. The height never exceeds the cap. On the house page (6.3in, cap 8in):

  | Figure | Today | Fitted |
  |---|---|---|
  | one panel | 3.5 × 2.45 | 3.5 × 2.45 |
  | `Aggregate.plot()`, two panels | 7.0 × 2.45 | 6.3 × 2.2 |
  | three panels | 10.5 × 2.45 | 6.3 × 2.0 |

  On the house slide (5.3in, cap 2.4in), one panel becomes 3.5 × 2.4 and two panels 5.3 × 2.0.
- **The figure is drawn at the fitted size,** not drawn at full size and shrunk, so type prints at
  the size the style names. `plot_chartdoc`'s pixel-driven choices then see the real width, which
  is correct.
- `Portfolio.scatter(figsize=...)` is fixed in passing: `plots/_portfolio.py:41` ignores its
  arguments and hard-codes `(10, 10)`. The fitting step touches that line anyway.

### [Library-Figures-House-Type] the next free minor after it

- **No library code names a type size.** Every site listed above takes its size from the active
  style: a legend uses `legend.fontsize`, and text drawn on the axes uses the tick or label size.
  `LABEL_POINTS` becomes a function that resolves the active tick-label size to points when the
  figure is drawn. The fit arithmetic at `:874` and `:939` then measures what is actually drawn.
  `FONT_SIZE` and `LEGEND_FONT` stay exported, so no import breaks, with a comment that the library
  no longer reads them.
- **`aggregate.mplstyle` takes the house type and face:** `font.size: 9`, `axes.titlesize: 9`,
  `axes.labelsize: 9`, `xtick.labelsize: 8`, `ytick.labelsize: 8`, `legend.fontsize: 8`;
  `legend.labelspacing: 0.25`, `legend.borderpad: 0.3`, `legend.handlelength: 1.6`,
  `legend.handletextpad: 0.5`; `font.family: serif`, `font.serif: STIXGeneral`,
  `mathtext.fontset: stix`, `pdf.fonttype: 42`. The title weight becomes `bold`, because
  STIXGeneral has no semibold, and matplotlib prints `findfont: Failed to find font weight
  semibold` into a notebook cell when asked for one. Colors, grid, line widths, `figure.figsize`
  and dpi are unchanged. The header comment records the ruling and the reason.
- **Values from house-style's `house.mplstyle` (kit 1.11.0).** After this step the two files
  agree on type and face. The house keeps its own copy, so its look does not move when this one
  does.
- **aLL:** `FIG_W` and `FIG_H` are unchanged, so the panel aspect aLL reads is unchanged.
  `font.size` moving from 8.5 to 9 raises aLL's chart base text from 10.5 to 11 pixels. That is
  the only visible effect there, and it is said in the CHANGELOG.
- **The Sphinx docs** call `aggregate.style.use()` (`docs/conf.py:16`, `:23`), so every docs
  figure changes face at the next docs build.

### Not in scope

- `axs=` on `Aggregate.plot` and `Portfolio.plot`, so a document could place them in its own grid.
  `plot_chartdoc` refuses `ax` for multi-panel documents because of width ratios and towers.
  `fit_page` makes it unnecessary for the house's purpose.
- `Distortion.plot_affine(ax=)` accepts `ax` and ignores it (`plots/_distortion.py:32`). That is a
  finding for `dev/TODO.md`, not part of this plan.

## Files

- `[Library-Figures-Fit-Page]`:
  - `plots/_style.py`, `plots/__init__.py`, and every figure-making site listed above
  - `tests/test_fit_page.py` (new)
- `[Library-Figures-House-Type]`:
  - `data/aggregate.mplstyle`, `plots/_chartdoc.py`, `plots/_bivariate_massive.py`,
    `plots/_bounds.py`, `plots/_fourier.py`, `pedagogy.py`, `constants.py` (comment only)
  - the baselines in `tests/data/chartdoc_baselines/`
- Each step also changes `src/aggregate/__init__.py` (`__version__`), `CHANGELOG.md` and
  `dev/TODO.md`.

## Acceptance

- **[Library-Figures-Fit-Page]**
  - With no page set, every figure's size is unchanged, and the pixel baselines in
    `tests/test_chartdoc_render.py` pass **without regeneration**. This is the proof that the step
    changes nothing for anyone who does not call `fit_page`.
  - With `fit_page(6.3, 8.0)`, `Aggregate.plot()` is 6.3 × 2.2in, a three-panel grid 6.3 × 2.0in,
    and a single panel unchanged, by `get_size_inches()`. With `fit_page(5.3, 2.4)`, no figure is
    wider than 5.3in or taller than 2.4in. `fit_page()` restores today's sizes.
  - `Portfolio.scatter(figsize=(4, 4))` makes a 4 × 4in figure.
  - `uv run pytest -m 'slow or not slow'` at the bump. `test_chartdoc_ladder.py` passes unchanged,
    since no page is set there.
- **[Library-Figures-House-Type]**
  - `rg "fontsize=|prop=\{'size'" src/aggregate` finds no number and no named size outside the
    style file.
  - Under `aggregate.style.use()`, `Aggregate.plot()` saved to PDF shows axis labels at 9pt and
    ticks at 8pt, read from its spans, with `STIXGeneral` the only face in `pdffonts`.
  - The chart baselines are regenerated with `--regen`, and each new image is looked at beside
    the old one before the commit. `test_chartdoc_ladder.py` is checked: larger tick labels narrow
    the axes slightly, so a case near a band edge may move, and that is recorded rather than
    tuned away.
  - `uv run pytest -m 'slow or not slow'` at the bump.

## Consumers

- **house-style** calls `fit_page(width=..., max_height=...)` from its `figures()` and
  `configure()`, with an import guard so that an older aggregate is left alone. That is its own
  plan, named above.
- **aggregate-api / aLL**: only the 0.5px base text change described above. No plan is needed
  there.
- **The writing projects** pick both steps up through the house kit and a floor of
  `aggregate>=` this release in their `pyproject.toml`, at their migration to the kit.

## Open questions for Steve

1. **A setting, against the 2026 rejection.** `fit_page` is a run-time call, off by default, for
   a quantity matplotlib has no rcParam for. The alternative that keeps aggregate free of any
   plotting setting is `figsize=` on every public plot method, about forty signatures, with every
   document call naming its size. The recommendation is `fit_page`.
2. **Never enlarge.** The rule only shrinks, so a single-panel library figure stays 3.5in wide on
   a page whose own single figures are 3.0in (half width). Matching half exactly would put the
   house's size policy inside aggregate. The recommendation is to leave it.

## Rulings

2026-10-19, Steve: question 1 "yes". `fit_page` is accepted as a run-time call despite the 2026 rejection of a plotting config section. Question 2 was not asked again, so its recommendation stands: the rule only shrinks. Versioning: `[Library-Figures-Fit-Page]` lands as 1.6.0a2 and `[Library-Figures-House-Type]` as 1.6.0a3, rolled up with 1.6.0a1 (`[DecL-Word-List]`) into the 1.6.0 release. The choices above are rulings, not proposals.

## Execution log

Divergences from the plan:

**[Library-Figures-Fit-Page], 1.6.0a2, 2026-10-19.**

1. **Where the fit applies.** Inside `make_grid` and `make_mosaic`, to the size given or defaulted, which covers `plot_chartdoc` and every caller of the two helpers, and at each direct `plt.subplots`, `plt.figure` and pandas `figsize=` site: `ft.py` (4), `tweedie.py` (3), `plots/_bounds.py` (2), `plots/_fourier.py` (4), `plots/_portfolio.py` (1) and `pedagogy.py` (15). A size passed explicitly to `make_grid` or `make_mosaic` is fitted too: the page is the hard constraint.
2. **A site the plan missed.** `_reinsurance.py:316`, `make_mosaic('AB\nCD', figsize=(12, 9))`, is fitted through `make_mosaic` with no edit.
3. **`Portfolio.scatter` forwards all its arguments**, not just `figsize`: `plots/_portfolio.py` hard coded `marker`, `s`, `alpha` and `diagonal` as well.
4. **`pedagogy.similar_risks_example` is broken, and stays so.** It calls `Portfolio.plot(figsize=(6, 4))`, which `Portfolio.plot` does not accept, and later `Bounds.plot_envelope(axs=...)`, which it does not accept either, so it raises `TypeError` before and after this step. Its `figsize` call is left as it was and the function is recorded in `dev/TODO.md`.
5. **`fit_page` validates.** A width or cap that is not a positive number raises `ValueError`. `fit_page(max_height=...)` alone caps heights and leaves widths.
6. **Acceptance.** The four chartdoc pixel baselines pass without regeneration, and `test_chartdoc_ladder.py` passes unchanged. `tests/test_fit_page.py` checks the plan's table, the house page and slide bounds, clearing, and `Portfolio.scatter(figsize=(4, 4))`. `Aggregate.plot()` on the house page measures 6.3 x 2.205in, the plan's 6.3 x 2.2.

**[Library-Figures-House-Type], 1.6.0a3, 2026-10-19.**

1. **Which style size each site takes.** Tower block labels and tower tick labels: `ytick.labelsize`, 8pt, where they were `LABEL_POINTS`, 5.85pt. Matrix cell values: `axes.labelsize`, 9pt (was 10pt). Matrix cell annotations: `xtick.labelsize`, 8pt (was 7.3pt). Every legend: `legend.fontsize`. The massive bivariate strip total, contour labels and origin badge: `xtick.labelsize`; its marginal title: `axes.titlesize`. A matrix value stays one step above its annotation, as before.
2. **A private helper, `plots._style._points(key)`**, resolves a font size rcParam, named or numeric, to points as matplotlib would, for the fit arithmetic. `_chartdoc._label_points()` replaces `LABEL_POINTS` and calls it at draw time; drawing sites that need no arithmetic pass `mpl.rcParams[key]` straight through. `FONT_SIZE` leaves `_chartdoc`'s imports; it and `LEGEND_FONT` stay exported from `constants` and `plots`, with the comment the plan asked for.
3. **A tower label is dropped at the larger size.** In the `structure` baseline the thin 0 to 5 retention band of the per occurrence tower loses its "Retained" label, which no longer fits a row at 8pt. That is the drop rather than spill rule doing its job, and it is recorded, not tuned.
4. **Baselines.** All five regenerated (`agg`, `distortion`, `matrix`, `structure`, `structure_log`) and each looked at beside the old one. Layout is unchanged apart from item 3.
5. **Ladder.** `test_chartdoc_ladder.py` passes unchanged, and also passes with `aggregate.style.use()` applied first (it does not apply the style itself): no case sat near a band edge.
6. **The PDF check is a committed test**, `tests/test_style.py::test_house_type_and_face_in_a_pdf`: under the style, `Aggregate.plot()` saved to PDF prints only 8pt (ticks, legend), 9pt (axis labels, titles) and 10.8pt spans; the last is the suptitle at matplotlib's `figure.titlesize: large`, which neither this style nor house.mplstyle sets. The only faces are STIXGeneral Regular and Bold (`pdffonts` agrees), and nothing is logged about a missing font weight.
7. **`font.sans-serif` stays** in `aggregate.mplstyle`. It is unused while the family is serif, but code that asks for sans explicitly still gets the old list. The note about `aggregate.mplstyle.original` stays too.
8. **Legends now follow whatever style is active.** A user who never calls `aggregate.style.use()` gets matplotlib's default legend size (10pt) on library figures, where the code forced `x-small` or `xx-small` before. Stated in the CHANGELOG.
