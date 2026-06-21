# Plan P2 — library-wide plotting subsystem

> **Status: DRAFT — not executed.** Cross-cutting and mechanical; see
> `plan-README.md`. Independent of the other plans but cleanest *after* P1 and
> *before* the P3/P4 splits (it lifts plotting out of the god files so they are
> smaller and matplotlib-free when reorganised).
>
> **Release mechanics (CLAUDE.md).** Mostly *pure moves* (no behaviour change) →
> tidying, no bump required; the matplotlib-defer step is a behaviour-adjacent
> change (import timing only, output identical) — note in `CHANGELOG.md`, baseline
> unaffected. `uv run pytest` green before every commit.

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

## 1. Target layout

```
aggregate/plots/__init__.py     # thin / lazy boundary; the single matplotlib entry
aggregate/plots/_aggregate.py   # plot_aggregate, plot_reins_occ
aggregate/plots/_severity.py
aggregate/plots/_distortion.py
aggregate/plots/_portfolio.py
aggregate/plots/_bounds.py
aggregate/plots/_bivariate.py
aggregate/plots/_fourier.py
aggregate/plots/_style.py        # absorbs style.py + shared FIG_W/FIG_H, tick formatters, axis helpers
```

Each class keeps a **thin, discoverable** `.plot()` (and named variants) that
delegates — the public API (`agg.plot()`) is unchanged, only the implementation
moves out:

```python
def plot(self, **kw):
    from aggregate.plots import plot_aggregate   # function-local = the lazy-load
    return plot_aggregate(self, **kw)
```

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

Split-within-class stays acceptable only for a plot bound to *ephemeral private
internals* or a throwaway debug plot — rare here, since plots render computed
`_df` state.

---

## 3. Phased execution (pure move; tests green after each)

- **3.1** Create `plots/_style.py`, absorbing `style.py` and the shared
  `FIG_W`/`FIG_H`/tick-formatter/axis helpers currently duplicated across files.
- **3.2** Move each class's plot bodies into `plots/_<class>.py` as functions;
  replace each method body with a thin delegating call (function-local import).
  One class per commit (`Aggregate` → `Severity` → `Distortion` → `Portfolio` →
  `Bounds` → `Bivariate` → `FourierTools`).
- **3.3** Make `plots/__init__.py` the single matplotlib boundary; confirm
  `python -X importtime -c "import aggregate"` no longer imports matplotlib at
  package import.

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

- **Output unchanged.** Moving a plot body must not change the figure; visual spot
  checks where feasible, and the non-plot test suite stays green.
- **`_df`-only access.** Plot functions read the public `<noun>_df` surfaces, not
  private arrays — if a plot needs a private array, expose it as a `_df` column or
  leave that plot on the class (the rare exception in §2).
- **One matplotlib boundary.** After 3.3, no module outside `plots/` imports
  `matplotlib` at top level. Add a hygiene test asserting this.

---

## 6. TODO / sequencing

This is **P2**. Independent of P1/P3/P4 but best landed after **P1** and before the
**P3/P4** splits, so the god files are smaller and matplotlib-free when
reorganised. Takes `Aggregate` *and* `Portfolio` (and the rest) plotting in one
pass — do not duplicate this work inside the split plans.
