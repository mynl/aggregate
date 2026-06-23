# Plan P2 — library-wide plotting subsystem

> **Status: DRAFT — not executed.** Cross-cutting and mechanical; see
> `plan-README.md`. Independent of the other plans but cleanest *after* P1 and
> *before* the P3/P4 splits (it lifts plotting out of the god files so they are
> smaller and matplotlib-free when reorganised).
>
> **Two passes, not one (the plots are overdue a refresh).** Plotting has not been
> touched in a long time and is high on the pre-beta list. So P2 is explicitly
> *port-then-revise*: **(A) Restructure** — move the existing plot bodies into the
> three-layer canvas×content shape with **output identical** (the safety net: a
> behaviour-preserving move you can verify); then **(B) Refresh** — a deliberate
> revision pass on the now-centralised plots (styling, panels, defaults). Keep them
> as distinct commits — the same "don't improve while moving" rule that governs
> P3/P4: restructure green first, *then* revise on the clean foundation.
>
> **Release mechanics (CLAUDE.md).** Pass A is *pure moves* → tidying, no bump
> required; the matplotlib-defer step is behaviour-adjacent (import timing only,
> output identical) — note in `CHANGELOG.md`, baseline unaffected. Pass B
> *intentionally changes output* → bump `1.0.0a*` + `CHANGELOG.md`; verified by
> visual review, not a frozen-figure guarantee. `uv run pytest` green before every
> commit (the non-plot suite stays green throughout; plots have no numeric
> baseline).

---

## 0. The idea

Plotting is **not** a per-class concern — it spans the whole library:

| Class | Entry points |
|---|---|
| `Aggregate` | `plot`, `reins_occ_plot` |
| `Severity` | `plot` |
| `Distortion` (spectral) | `plot` + helpers |
| `Portfolio` | `plot`, `scatter`, `sample_compare`, `sample_density_compare` |
| `Bounds` | `plot` + envelope/weights |
| `Bivariate` | `plot` |
| `FourierTools` | `plot`, `_plot_fourier1d` |

~12 entry points across 7 classes, ~200 matplotlib references, much of it
duplicated axis/tick/figure boilerplate. Treat it as **one subsystem, organised by
class inside** — combined as a subsystem (one matplotlib boundary, one styling
point), split by class within (locality). The combine-vs-split-within-class
dichotomy is false: do both.

---

## 1. Target layout — three layers

The subsystem is organised on **two axes at once**: *what is drawn* (content) is
the reuse axis; *who composes it into a figure* (class) is a thin compositor axis.
Reading the real code confirms this is the natural grain — every `.plot()` method
is a **compositor**: `Aggregate.plot` lays out an `'ABC'` mosaic (density / F /
Lee), `Portfolio.plot` an `'AB'` mosaic (density / log-density), and both draw the
**same content panels** (density, F/S, Lee) on different layouts and data slices.
Splitting on content writes that drawing **once**; splitting on class keeps the
canonical `.plot()` discoverable. Three layers:

```
aggregate/plots/__init__.py      # thin / lazy boundary; the single matplotlib entry

# Layer 0 — "make the space": generic canvas + style + settings (no domain knowledge)
aggregate/plots/_style.py        # absorbs style.py + FIG_W/FIG_H, tick formatters,
                                 #   axis/limit helpers, reads shared plot settings,
                                 #   AND generic canvas creators
                                 #   (mosaic/gridspec/figsize -> (fig, axd))

# Layer 1 — "populate the space": content panel workers, by WHAT is drawn.
#   Each renders ONE content type into a PROVIDED Axes from a public _df slice.
aggregate/plots/_density.py      # plot_density(ax, ser, *, discrete, label, ...)
aggregate/plots/_distribution.py # F and S (CDF / survival), linear + log
aggregate/plots/_quantile.py     # the Lee / quantile panel
aggregate/plots/_kappa.py        # exeqa_* allocation curves (Portfolio today)
aggregate/plots/_distortion.py   # g / g_dual curves
aggregate/plots/_bounds.py       # envelope cloud + weights (single-consumer: panel
aggregate/plots/_bivariate.py    #   and compositor collapse into one content module
aggregate/plots/_fourier.py      #   where there is no cross-class reuse to factor)

# Layer 2 — "combine": per-class compositors, by WHO. Thin: ask Layer 0 for the
#   canvas, then populate each Axes via Layer 1. Hold the class's named variants.
#   These compose the Agg / Sev / Port / … exhibits.
aggregate/plots/_aggregate.py    # plot_aggregate, plot_reins_occ
aggregate/plots/_portfolio.py    # plot_portfolio, scatter, sample_compare, ...
aggregate/plots/_severity.py     # plot_severity
```

The class keeps only a **one-line delegating stub** (the public API `agg.plot()`
is unchanged); the compositor body moves to Layer 2 — which is also what makes the
god files smaller and matplotlib-free for the P3/P4 split, not just the
import-defer:

```python
# on the class (distributions.py / portfolio.py):
def plot(self, axd=None, xmax=0, **kw):
    from aggregate.plots import plot_aggregate   # function-local = the lazy-load
    return plot_aggregate(self, axd=axd, xmax=xmax, **kw)
```
```python
# Layer 2 — plots/_aggregate.py: the thin combiner ("subcontracts" the work)
def plot_aggregate(agg, axd=None, xmax=0, **kw):
    if axd is None:
        fig, axd = make_mosaic('ABC', figsize=(3 * FIG_W, FIG_H))   # Layer 0
    plot_density(axd['A'], ...)          # Layer 1
    plot_distribution(axd['B'], ...)     # Layer 1
    plot_quantile(axd['C'], ...)         # Layer 1
    return axd
```

**Single-consumer content** (Bounds envelope cloud, Bivariate 2-D, FourierTools)
has no cross-class reuse to factor, so its content module *is* its compositor — one
file, still content-named, future-proof if a second caller ever appears.

**Every plotting class gets the canvas×content combiner — no exceptions.**
`Portfolio`, `Aggregate`, `Bounds`, `Distortion`, `Severity`, `Bivariate`, and
`FourierTools` each lose their hand-rolled figure setup and become a Layer-2
compositor (ask Layer 0 for the canvas, populate via Layer 1) plus a one-line
class stub. Uniformity here *is* the consistency win the refresh (Pass B) is for.

---

## 2. Why combined wins for this library specifically

- **The decisive lazy-load lever.** matplotlib is referenced in 6+ files today;
  centralising gives **one** import boundary for the whole library, so a lazy
  `plots` subpackage means `import aggregate` never touches matplotlib until
  someone plots *anything*. Bigger import-time win than any other refactor step.
- **`style.py` already exists** — one subsystem applies the look uniformly and
  deduplicates the repeated tick/figure boilerplate into `plots/_style.py`.
- **Forces plotting onto the public `_df` surface.** A function outside the class
  can only reach `agg.density_df`, `port.density_df`, `bounds.weight_df`, … — the
  `<noun>_df` convention being frozen at 1.0. Centralising *dogfoods* that data API
  and stops plots depending on private arrays. Healthy coupling.
- **Consistency serves the intuition-tool mission** — plots are how users "see what
  is going on"; they should look like one library.
- **Content split kills the real duplication.** `Aggregate.plot` and
  `Portfolio.plot` both draw density and F/S panels today, written twice; a
  by-content Layer 1 (`_density`, `_distribution`, `_quantile`, `_kappa`) writes
  each panel **once** and lets every compositor reuse it. The per-class split alone
  would leave that duplication in place — the content axis is what removes it.

Split-within-class stays acceptable only for a plot bound to *ephemeral private
internals* or a throwaway debug plot — rare here, since plots render computed
`_df` state.

---

## 3. Phased execution

**Pass A (3.1–3.4) is the structural port — output identical, tests green after
each.** Pass B (3.5) is the deliberate refresh.

- **3.1 — Panel inventory (do this first; it defines the module boundaries).**
  For each `.plot()` and named variant, list the content panels it draws and the
  public `_df` columns each reads. That table *is* the Layer-1 split: it shows
  which content (density, F/S, Lee, kappa, …) is drawn by ≥2 compositors (→ a
  shared `_<content>.py` worker) versus genuinely single-use (→ collapsed into its
  class/content module). This is the by-content analog of the two-file rule:
  **promote a worker to a shared content module only when the inventory shows ≥2
  callers or a clearly-named content type — never pre-abstract a one-caller panel.**
- **3.2 — Layer 0.** Create `plots/_style.py`: absorb `style.py` and the shared
  `FIG_W`/`FIG_H`/tick-formatter/axis-limit helpers, **plus** the generic canvas
  creators (`make_mosaic`/figsize → `(fig, axd)`) that the compositors will call.
  No domain knowledge here — just "make the space."
- **3.3 — Layers 1 & 2, one class per commit.** For each class
  (`Aggregate` → `Severity` → `Distortion` → `Portfolio` → `Bounds` → `Bivariate`
  → `FourierTools`): move its `.plot()` body into `plots/_<class>.py` as the thin
  compositor (asks Layer 0 for the canvas, populates each Axes via Layer 1),
  factoring each content panel into its `plots/_<content>.py` worker as the
  inventory dictates (the first class introduces `_density`/`_distribution`/
  `_quantile`; a later class that draws the same content *reuses* the worker rather
  than re-writing it — that reuse is the payoff). Replace the class method with the
  one-line delegating stub (function-local import).
- **3.4 — Single matplotlib boundary.** Make `plots/__init__.py` the only entry;
  confirm `python -X importtime -c "import aggregate"` no longer imports matplotlib
  at package import, and add the hygiene test (§5) asserting no module outside
  `plots/` imports matplotlib at top level.

- **3.5 — Refresh (Pass B; bumps).** With every plot now a thin compositor over
  shared content workers, revise *on that clean foundation*: modernise styling in
  Layer 0 once (it propagates everywhere), tidy panel defaults/labels/limits in the
  Layer-1 workers, and adjust compositor layouts as desired. Output changes by
  design — verify by visual review. Because the drawing is centralised, a single
  edit to `_density`/`_style` now updates every figure that uses it, which is the
  whole point of doing the restructure first.

---

## 4. Lazy loading — scope note

Two flavours, opposite answers:

- **Your own pure-Python logic** (reinsurance, pricing): lazy-loading buys nothing
  and isn't a real Python operation — methods bind at class-definition time. The
  module organisation is for humans, not import speed.
- **Heavy third-party deps** (`matplotlib`): the one real import-time win, and this
  plan delivers it *globally* — once every matplotlib use lives in `plots`,
  deferring the import is one change per plot module via the function-local import
  in §1.

A further, *optional, measure-first* step is package-level `__getattr__` (PEP 562)
to defer the whole `plots` subpackage (and `tweedie`, `bivariate`) until first
attribute access — the numpy/scipy trick. Only after
`python -X importtime -c "import aggregate"` confirms matplotlib (and what else) is
actually the cost. `numba` is already optional; this extends the same "heavy thing,
loaded on demand" philosophy to plotting.

> Net: **lazy-load heavy deps, not your own logic.** The matplotlib defer falls out
> of the extraction nearly for free.

---

## 5. Guardrails

- **Output unchanged — in Pass A only.** Moving a plot body (3.1–3.4) must not
  change the figure; visual spot checks where feasible, and the non-plot test suite
  stays green. The guarantee is **deliberately lifted in Pass B (3.5)**, where
  output changes by design and visual review replaces it — do not let a stale
  "figure must not change" reflex block the refresh.
- **`_df`-only access.** Plot functions read the public `<noun>_df` surfaces, not
  private arrays — if a plot needs a private array, expose it as a `_df` column or
  leave that plot on the class (the rare exception in §2).
- **One matplotlib boundary.** After 3.4, no module outside `plots/` imports
  `matplotlib` at top level. Add a hygiene test asserting this.
- **Layer discipline: panels take an Axes, compositors own the figure.** A Layer-1
  content worker (`plot_density`, …) renders into a **provided** `Axes` and never
  creates a figure or calls `plt.subplot_mosaic` — that belongs to the Layer-0
  canvas helper, invoked by the Layer-2 compositor. This is what makes a panel
  reusable across compositors and embeddable in a caller's own grid. A panel that
  reaches for `plt.figure()`/`plt.show()` is mis-layered.

---

## 6. TODO / sequencing

This is **P2**. Independent of P1/P3/P4 but best landed after **P1** and before the
**P3/P4** splits, so the god files are smaller and matplotlib-free when
reorganised. Takes `Aggregate` *and* `Portfolio` (and the rest) plotting in one
pass — do not duplicate this work inside the split plans.
