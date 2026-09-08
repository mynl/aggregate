# plan-pk-tab: the Pricing > Pr Ruin pill (probability of eventual default)

Status: DRAFT v2, 2026-09-05, incorporating the author's rulings of the same
day (recorded in the Rulings section). Canonical in the API repo, the
plan-pricing-exhibits arrangement; a plain copy sits in LIB `dev\`
(no symlink, by ruling); the API copy is canonical.
Written for a reader with no conversation context.

## Goal

A new pill on the **Pricing pane, last in the row, after Evaluate, labeled
Pr Ruin**: the probability of eventual default (eventual ruin, psi) for an
aggregate whose frequency supports it. Poisson is served by the
Pollaczeck-Khinchine formula, a renewal (wait clause) frequency by cepstral
Wiener-Hopf factorization; the two paths match as closely as possible: one
pane, one form, one figure layout, the solver chosen by the object's
frequency.

The pane is a small form above two plots.

- **Form.** The premium input group **exactly mirrors the base premium
  group of the Calibrate subtab**, same fields, same conversions, same
  layout, so a user who has calibrated already knows how to price here; the
  engine derives the margin-to-loss ratio `rho` from it (premium rate
  `c = (1 + rho) E[X] / E[W]`; `c` itself never appears in the UI). The
  **debounced preview row stays**, the pentagon preview line the Calibrate
  form carries. One additional input: *initial capital*, entered as a
  **probability of eventual default** p and converted to surplus `u`
  through the ruin function's capital lookup `find_u(p)`; the resolved `u`
  and the achieved psi at the grid point display beside it.
- **Left plot.** Sample surplus paths for the resolved `(c, u)`:
  `U(t) = u + c t - S(t)`, the expected trend line, the
  law-of-the-iterated-logarithm funnel, ruin times marked (the rug of ticks
  under the zero line), the title reporting simulated ruins against exact
  psi. About 50 paths drawn.
- **Right plot.** `psi(u)` over a reasonable range of initial surplus,
  linear scale with a log reading offered, marker at the resolved
  `(u, psi(u))`.

This is a teaching aid, not a production pricing tool; the sizing rulings
below reflect that.

## What already exists (all LIB, verified 2026-09-05 at 1.0.0a338)

The mathematics is done; this plan is a serving exercise.

- `Aggregate.pollaczeck_khinchine(rho, ...)` and
  `Aggregate.wiener_hopf(rho, ...)`, both returning the shared
  `RuinFunction(ruin, find_u, mean, density)` named tuple: `ruin` is the
  `psi(u)` Series on the u grid, `find_u` the capital lookup `p -> u`
  (index or interpolate), `density` the equilibrium (PK) or all-time-maximum
  (WHopf) density.
- `pedagogy.ruin_example(agg, rho, u0, *, n_sims, n_plot, seed, ...)`: the
  exact figure this pane wants, drawn in matplotlib. Left panel sample paths
  with trend, LIL funnel and a ruin-time rug; right panel `psi(u)` with the
  `(u0, psi(u0))` marker; a one-column summary DataFrame (exact and
  simulated psi, safety loading, horizons, grid diagnostics). It dispatches
  on frequency exactly as this pane must, and its simulation samples the
  discretized severity and wait laws, so the simulated model is the model
  the psi computation prices.
- `_renewal.ruin_cepstral` and the defective-wait machinery underneath
  `wiener_hopf`.

What does not exist: a chart-IR emitter, an exhibit, and any app surface.
`ruin_example` lives in `pedagogy`, which is figure generation for the docs,
not core API; its simulation core must be lifted somewhere servable.

## LIB half

**[Ruin-Engine]** Lift the simulation core of `pedagogy.ruin_example` into
a private helper beside the solvers (suggested: `_ruin_paths(...)` in
`_renewal.py` or `_aggregate.py`), returning arrays rather than drawing:
per-path claim times and cumulative losses, ruin time per path, plus the
summary scalars. `pedagogy.ruin_example` becomes a thin matplotlib consumer
of the same helper, so the docs figure and the served chart cannot drift.
Deterministic seed by default (ruling 2).

**[Ruin-Exhibit]** New exhibit `ruin` on `Aggregate` only (Portfolio is
out of scope permanently, ruling 5), predicate: updated and frequency in
{poisson, renewal}. One block, the pane's stats strip: exact psi at u,
simulated psi with its standard error, rho and the premium restated the way
the Calibrate form states it, safety loading, mean severity, E[W],
horizons, and for the Poisson path the Lundberg exponent and bound if
cheap. Served through a method taking the form inputs (a property cannot,
and the exhibit route already carries options).

**[Ruin-Chart]** New chart `ruin` (Aggregate only), two panels with
independent x axes (`Panel` / `ChartAxis` support this):

- panel `paths`: about 50 path series (role `sample`, ruling 4), the trend
  line (role `mean`), the LIL funnel, ruin-time marks.
- panel `psi`: the `psi(u)` curve (log-capable axis, the survival-axis
  declaration pattern), the `(u, psi(u))` marker.

Options: the Calibrate-form premium inputs, `p` (or `u`), `seed`,
`n_plot`, `detail`. `meta` carries the scalars the exhibit also serves
(rho, u, psi_exact, psi_sim, seed), so the pane labels itself from the
chart doc alone. Register in the chart registry; `available_charts` gates
on the frequency predicate.

**[Ruin-Downsampling]** Payload control, mandatory (see plan-payload, in
preparation; this chart must not repeat the approximation chart's megabyte
documents).

- Paths: decimate each path's time axis to about 256 to 1024 points,
  **preserving each interval's running minimum** so a dip below zero is
  never thinned away; a path that ruins keeps its exact ruin point.
- psi curve: about 256 u points, log-spaced toward the tail where the
  curve is flat in linear u.
- The lattice payload form applies to any regular grid kept.

## Dynamic behavior (the IR question)

Can the payload carry enough numbers to make the pane dynamic as the form
changes? Two tiers:

1. **Scalars in `meta`**: free, and in scope here. The pane re-labels
   itself (resolved u, achieved psi, loading) with no extra request.
2. **The affine observation**: paths are `U(t) = u + c t - S(t)` and the
   simulated `S(t)` skeletons do not depend on `(c, u)` at all, so a
   served set of skeletons could in principle be re-drawn client side for
   any form value, instantly. But re-deriving ruin markers and the title
   count client side is business logic in the app, and `psi(u)` depends on
   `c` regardless, so the right panel needs the server anyway.
   **Recommended: not in v1.** Follow the pentagon-preview precedent: every
   form change issues a debounced chart-doc request and the server
   recomputes. With downsampling the doc is small and the FFT psi is fast;
   the affine redraw becomes an upstream ask later only if the round trip
   proves too slow. If ever adopted, the transform must be *declared in the
   doc* (the `reciprocal_of` / `return_period_map` pattern), so the library
   still owns the meaning and the app implements a declared reading
   generically.

## API half

**[Ruin-Route]** Nothing new: the pill uses `available_charts` /
`build_chart_doc` with options, and `available_exhibits` / `build_exhibit`
for the stats block, through the existing wire contracts. A capability
flag `can_ruin` (frequency in {poisson, renewal}) joins the caps payload so
the pill greys correctly with the standard tooltip why.

**[Ruin-Pane]** Pr Ruin leaf in `nav.js`, **last in the Pricing group,
after Evaluate**. The form reuses the Calibrate premium group component
verbatim (extract it if it is not already shared), keeps the debounced
preview row, and adds the probability-of-default input; both feed a
debounced chart-doc re-request. Two-panel render through the generic
`chartdoc-to-echarts.js` adapter (a per-chart override only if the rug
marks need one). Hint: "sample surplus paths and the probability of
eventual default; Poisson or renewal frequency only".

**[Ruin-Sample]** A Sample action re-rolling the seed: the same route with
`seed` omitted; the server picks one and reports it in `meta` (ruling 2).

## Order of work

LIB first, then API, the plan-3d-plot rule: emitter and any reader change
land together. Phases [Ruin-Engine] then [Ruin-Exhibit] and [Ruin-Chart]
together (one LIB bump each or combined, author's call),
[Ruin-Downsampling] inside the chart bump, then API [Ruin-Route] and
[Ruin-Pane] in one API version.

## Acceptance

- A Poisson book and a renewal book (`5 years sev 10 * expon wait gamma 2`
  is in `decl-testers.agg` section AD) both serve the pill; a negbin book
  greys it with the why.
- Exact psi in the exhibit matches `pollaczeck_khinchine` / `wiener_hopf`
  directly; simulated psi within 3 standard errors at `n_sims = 1000`.
- The chart doc for default settings is under 200 kB canonical JSON.
- Form changes re-render within the debounce without layout shift; the
  resolved `u` and achieved psi display beside the inputs.
- The Calibrate subtab is pixel-unchanged by the form-group extraction.
- LIB docs figure (`5_x_pk.rst` route) unchanged in appearance after the
  [Ruin-Engine] lift.

## Rulings (author, 2026-09-05)

1. **Name**: `ruin`, not `default`, for the LIB exhibit and chart registry
   names and the capability flag; the pill label is Pr Ruin.
2. **Seed**: fixed default seed (hash-stable, cacheable docs); explicit
   `seed=None` for Sample.
3. **Premium input**: follows the Calibrate form's base premium group
   exactly, conversions included; the preview row stays.
4. **Sizing**: `n_sims` capped at 1000 for the probability estimate; about
   50 paths drawn. This is a teaching aid, not production.
5. **Portfolio**: never.
6. **Placement**: Pricing pane, last pill after Evaluate, labeled Pr Ruin
   (supersedes the draft's More > Default).
