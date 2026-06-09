# Plan — hygiene 2: consistent `info`, approximate-note tidy, window-aware plots

> **Status: ready to execute.** Three small, mostly independent items. Bumps
> `1.0.0a*` (code change). Author (prod-claude) executes and commits. Items 1 and
> 2 touch the same `approximate` surface and are best done together; item 3 is
> standalone.

---

## Item 1 — consistent `info` layout (no scaffold branching)

**Goal.** Each of `Aggregate.info`, `Portfolio.info`, `Distortion.info` always
emits the **same scaffold** for its type — the fixed lines appear in the same
order every time, regardless of object state. Only genuinely *feature-gated*
blocks (a `pnl` readout, a signed-window block, reinsurance descriptions) may come
and go; the core header/params/footer never reorder or vanish.

**The concrete offender — the `approximate` marker** (distributions.py:3973–3976).
Today it is emitted **only when active** and **before** the severity line:

```python
# claim count / frequency / [approximate if != exact] / severity ...
```

Fix: make it a **permanent header line**, positioned **after** severity, so the
top block is always

```
aggregate object name    ...
claim count              ...
frequency distribution   ...
severity distribution    ...
approximate              exact | sgamma | slognorm
```

i.e. *freq → sev → approx*, with `approximate exact` shown for an ordinary
aggregate. Drop the `if ... != 'exact'` guard. (The richer approximate readout —
original program + fitted params — is item 2; this line is just the always-present
marker.)

**Audit the other two for scaffold branching** while here:
- `Portfolio.info` (portfolio.py:957) and `Distortion._compute_info`
  (spectral.py:1126) — confirm every conditional is *content/feature* driven
  (e.g. `if self.bs > 0` = "not yet updated", legitimate; per-kind distortion
  params are intrinsic), **not** a layout variant. Leave intrinsic per-kind
  content alone; only fix lines that appear/disappear without a feature reason.

**No number movement** — `info` is a display string, not a frozen baseline.
Update any doctest/guide snippet that pins the old approximate-line position.

---

## Item 2 — `approximate` note carries the original program

**Goal.** For an approximated aggregate the note should read as *(existing user
note) + (original DecL program) approximated by (sgamma|slognorm) with fitted
params (...)* — so the human-readable record shows **what was approximated** and
**how**.

**Root cause of the current loss.** The note is assembled inside
`Aggregate.__init__` (distributions.py:3417–3422):

```python
# "the full original program is retained on self.program" -- NOT TRUE here:
note = f"{note}; {_approx_note}"   # _approx_note = fit moments only
```

but `self.program = ''` is set a few lines later (3437) and the **real** program
text is assigned *externally* by `build_many` only **after** `__init__` returns
(underwriter.py:831–885). So at note-assembly time the program is empty — the note
can never include it, and the reassuring comment is stale. That is the "old code
jutting in."

**Fix (recommended shape).** Stop baking an incomplete string into `self.note` at
construction; compose the approximate description **where `self.program` is
available** (i.e. lazily, at display / round-trip time):

1. At construction, keep `self.note` as the user's **pure** note, and store a
   small structured record of the fit on the object — e.g.
   `self._approx_fit = {'kind': approximate, 'sev_name': sev_name,
   'loc': sev_loc, 'scale': sev_scale, 'shape': sev_a,
   'm': _m, 'cv': _cv, 'skew': _sk}`. (`self.approximate` already holds the kind.)
2. Add a helper, e.g. `_approx_description()`, that renders
   `f"{self.program} ≈ {kind}: {sev_name}(loc={..}, scale={..}, …), "
   f"m={..:.6g} cv={..:.6g} skew={..:.6g}"` using `self.program` (now populated).
3. Surface it in two places:
   - **`info`** — under the item-1 `approximate` line, indent the description
     when not `exact` (this is the natural home and dovetails with item 1).
   - **the round-trip note** — wherever the note is written for `to_agg` /
     `repr`, append `_approx_description()` to the user note. If a single
     materialised `self.note` string is required for round-trip, finalise it in
     `build_many` right after `obj.program = program` (the one point where both
     the user note and the program coexist), rather than in `__init__`.

Either way the **original program is preserved** (it already lives on
`self.program` / `self._spec`); the bug is purely that the note was assembled too
early to see it. Pick the lazy-render path unless a frozen `self.note` is needed.

**Decision to confirm with the author:** whether the program text belongs *inside*
`self.note` (round-trips into `note{...}`, slightly recursive but harmless) or is
shown only in `info` as its own block. The plan recommends: structured fit on the
object + rendered in `info`; **also** append to the round-trip note so `to_agg`
output is self-describing.

**No number movement** — note/info are display only. Add a small test: an
approximated agg's `info` contains both the original program text and the fitted
family/params, and its `note` is non-empty and round-trips.

---

## Item 3 — window-aware `plot` (aggregate correct; severity deferred)

**Goal.** `Aggregate.plot` (distributions.py:5623) and `Portfolio.plot`
(portfolio.py:2490) should set their x-axis from the **actual output window** —
the live support of the aggregate — so they are correct whether the window is the
ordinary `[0, …]`, a thin-tailed window with `x_min > 0`, or a signed P&L window
straddling 0. Today the limits lean on 0-based assumptions:

- `Aggregate.plot` discrete branch forces `mn = 0` unless `xs[0] < 0`
  (5648), and continuous uses `f(hi) = [-0.02·hi, 1.02·hi]` (via `_limits`),
  which assumes a 0-anchored axis and wastes space (or misplaces the left edge)
  when the window does not start at 0.
- `Portfolio.plot` similarly takes `self._limits()` without reference to the live
  window.

**Fix.** Drive the linear x-limits from the **live support** — the `density`
property (a39: `density_df.query('p_total > 0')`) gives `[loss.min(), loss.max()]`
— padded, falling back to the moment estimate when `agg_density is None`. This
makes `_limits(stat='range')` window-correct for all three window types in one
place (it already special-cases signed via `xs[0] < 0`; generalise that to "use
the realized support" rather than "assume 0 unless signed"). Apply the same in
`Portfolio._limits`.

**Scope / explicit deferral.** This item makes the **aggregate** curve correct.
The **severity overlay** (drawn on its own `sev_density_df` grid, which need not
match the aggregate window) is a separate question — *whether* and *how* to show
the severity alongside a windowed aggregate. **Defer it**: leave the current
severity overlay as-is for now and **add a `dev/TODO.md` item** — "plot: revisit
the severity overlay vs the aggregate output window (own-grid severity may fall
outside / be misscaled against the windowed aggregate)."

**Possible movement.** Plots are not part of the frozen numeric baseline, but
several docs embed rendered figures; note in the CHANGELOG that plot axes for
non-zero-origin / signed aggregates change (a visual improvement). No `pytest`
numbers move.

---

## Housekeeping

- Bump `1.0.0a*` in `pyproject.toml` (next free alpha; after the a48 splice fix if
  that lands first).
- `CHANGELOG.md` section: *"Hygiene: `info` always emits the `approximate` line
  (freq → sev → approx, `exact` shown); approximate note now records the original
  program + fitted params (was lost — assembled before `program` was set);
  `Aggregate`/`Portfolio` plots are window-aware (correct x-limits for
  non-zero-origin and signed windows). Severity-overlay-vs-window deferred (TODO)."*
- Add the **plot severity-overlay** TODO to `dev/TODO.md` (item 3 deferral).
- Move this plan to `dev/done/` when complete.
