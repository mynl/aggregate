# plan-chart-reflect.md

`[Chart-Reflected-Reading]`, a sixth declared reading for the chart IR.

Status: drafted 2026-08-13, rulings taken, **not executed**. Target version `1.0.0a267` (LIB is at `1.0.0a266`).

## Context

The chart IR offers five renderer switches today: `log`, `full_range`, `return_period`, `invert`, `kind`. Each follows one rule, `[Chart-Declared-Readings]` in `dev/done/plan-chart-ir.md:150-170`: the document declares which readings a quantity honestly admits, the renderer's switch acts wherever the declaration exists and nowhere else, and the app derives its control strip from the declarations rather than from a chart name. There is deliberately no per-chart allow list of options anywhere in the code.

One reading is missing. A Lee panel draws the quantile function against `p`, the non-exceeding probability, and inverted it draws `F(x)`. What nobody can ask for is `S(x) = 1 - F(x)`, the survival function, which is the reading an actuary reaches for most often and the one a log axis was invented for. The same gap sits on a distortion: `g(s)` on the unit square has probability on both axes and no reading but the one it is drawn in.

So: `reflect`, the map `v` to `1 - v` on any axis that declares it. On the Lee panel of `agg`, `pnl`, `severity` and `reins` it turns the probability axis into the exceedance probability, and the drawn curve into the survival function. On `distortion` and `envelope` it reflects both axes of the unit square. It is a change of coordinates and not of data, exactly as `return_period` and `invert` already are, so it lives in the renderer's argument list and never in `build_chart_doc`.

## The four rulings behind this plan (author, 2026-08-13)

1. **Declaration is a paired axis.** A new `ChartAxis.complement_of` pointer, mirroring `reciprocal_of` in every respect: an undrawn axis sitting in `doc.axes`, carrying its own label, its own `scales`, its own window. The alternative, a boolean plus a label field on the drawn axis, was refused because it has no place to say that `S(x)` is log readable while `F(x)` is not, which is the main reason to want the reading at all.
2. **One boolean.** `reflect=True` acts on every axis that declares the reading and on no other, exactly as `log` does. No `reflect_x` / `reflect_y` pair, which `invert` would make ambiguous, and no list of axis ids.
3. **The distortion declares it, and the legend goes stale.** Reflecting both axes of `{(s, g(s))}` gives `{(u, ǧ(u))}`, the dual. When `dual=True` the document already draws `ǧ` as its own named series, so reflecting swaps which curve each legend entry traces. The picture is right, the names are stale, and the docs say so. A reader who wants it clean passes `dual=False`.
4. **Reflect flips the return-period map.** Where both readings are asked for on one axis, reflect maps the values and exchanges `complement` with `reciprocal`, so the two compose rather than special casing. On a loss chart that is exactly the return-period curve already drawn. On a signed `pnl` chart, whose map is `reciprocal` because the adverse tail is the low one, reflect plus return period reads the *upside* tail's return period, which is a real picture unreachable any other way.

## Phase 1: the schema (`src/aggregate/charts/ir.py`)

**`ChartAxis.complement_of: str = None`**, added after `reciprocal_of` at `ir.py:607`, with a NumPy docstring paragraph modeled on `reciprocal_of`'s at `ir.py:580-592`: the id of the drawn probability axis this axis is the reflected reading of, presence is the declaration, the map is `v` to `1 - v`, the paired axis sits in `ChartDoc.axes` and is named by no panel.

**Validation** in `ChartDoc.__post_init__`. The `reciprocal_of` block at `ir.py:1011-1032` becomes a loop over both pointer fields, so the three existing checks (names an existing axis, is not itself drawn, points at something drawn) serve both and the error messages name which pointer failed. Two new checks fall out of doing it as a loop:

- one axis carries at most one pointer. `complement_of` and `reciprocal_of` on the same axis would name a chained reading, and chaining is not how the composition works: the flip in ruling 4 is what composes them.
- at most one paired axis per (target, pointer) pair, since `_paired_reading` returns the first match and a second would be silently unreachable. This tightens `reciprocal_of` too, which today has the same latent looseness and no emitter that exercises it.

**`REFLECTED_RETURN_PERIOD_MAP`**, a new module constant beside `RETURN_PERIOD_MAPS` at `ir.py:127`, mapping `'reciprocal'` to `'complement'` and back, with the comment carrying ruling 4's argument. It lives in `ir.py` rather than in a renderer because both renderers need the same fact and it is a statement about the vocabulary.

**Do not add `complement_of` to `_ALWAYS`** (`ir.py:1098`). Its default `None` means "this axis offers no reflected reading", which is the neutral absence a reader may ignore, so the omit-at-default rule at `ir.py:1076-1097` holds and no existing document's hash moves because of the field itself.

**`CHART_IR_VERSION` stays 2.** By the rule at `ir.py:87-104` and the a240 precedent, a reader that ignores `complement_of` sees an extra axis in `doc.axes` that no panel names, which is exactly what `return_period` already looks like to it, and draws the default reading correctly and completely.

## Phase 2: the renderer (`src/aggregate/plots/_chartdoc.py`)

`plot_chartdoc` gains `reflect=False`, placed between `full_range` and `return_period` (`_chartdoc.py:582`), and passes it into `_render_xy_panel` (`:440`). The insertion changes positional order for a caller who passes positionally; every call site in both repos uses keywords, and the module is provisional under PEP 411, so this is noted rather than worked around.

**`_paired_reading` generalizes** (`:354`), taking the pointer attribute as a defaulted argument so the existing name and the JS comment that cites it both survive:

```python
def _paired_reading(doc, axis_id, attr='reciprocal_of'):
    for a in doc.axes:
        if getattr(a, attr) == axis_id:
            return a
    return None
```

**The per-axis map becomes a composed callable.** Today `x_map` / `y_map` hold a return-period map name or `None`, and `coords` (`:474`) branches on it. They become a small composed function built once per axis, so the exchange under `invert` at `:472` keeps working unchanged and `coords` loses its branch:

```python
def _reading_map(reflected, period):
    """The coordinate map for one axis: the reflection, then a return period."""
    if not reflected and period is None:
        return None

    def apply(values):
        v = np.asarray(values, dtype=float)
        if reflected:
            v = 1.0 - v
        return v if period is None else _return_periods(v, period)
    return apply
```

The axis selection at `:458-472` becomes, per drawn axis: if `reflect` and a `complement_of` pairing exists, substitute that axis and set `reflected`; then if `return_period` and a `reciprocal_of` pairing exists, substitute that axis instead and take the document's map, flipped through `REFLECTED_RETURN_PERIOD_MAP` when `reflected`. Both readings on, the axis shown is the return-period one, whose label is right in either case.

**Two behaviors stay keyed on the return period specifically, not on "a map is present".** This is the trap in the refactor and both sites need a comment saying so:

- the `MAX_RETURN_PERIOD` cap at `:491-495`, which exists because the quantile function saturates and `T` diverges. A reflected probability axis is bounded in `[0, 1]` and needs no cap.
- the companion window release at `:553-554`, which exists because a return-period reading re-slices the panel into the deep tail. Reflection is a bijection of `[0, 1]` onto itself and the companion axis is unaffected, so its window must stand.

**Windows need no arithmetic.** The reflected axis carries its own `suggested_range` and `full_range` from the emitter, which is the payoff of ruling 1, so `_axis_window` (`:101`) is untouched.

**The atomic ladder needs no change, and the docstring should say why.** `_draw_atomic` (`:183-191`) picks a step direction off the axis units, and a reflected probability axis is still `unit='probability'`. Reflection reverses the monotone direction of the curve, so a right-continuous step ought to become left-continuous, and it does, for free: matplotlib's step drawstyles are defined on the *order of the points given*, not on the axis direction, so drawing `steps-post` over the reflected point sequence produces exactly the mirror of the picture it produced before. Points `(x0, y0), (x1, y1)` with `x0 < x1` draw the corner at `(x1, y0)`; reflected they draw it at `(1 - x1, y0)`, which is the mirror of `(x1, y0)`. Same reason `invert` needed no explicit switch at a240 ("two things fell out rather than being written"). Add a Notes paragraph to `_render_xy_panel` recording the argument, because it looks like a bug until someone works it out.

**A log reading of a reflected axis** is handled by `_decade_floor` (`:70-88`) already: a survival axis declaring `suggested_range=(0.0, 1.0)` on a log scale is the same shape as the `mass` axis, which declares `(0.0, ordinate_top)` with `scales=('linear', 'log')` and works today.

## Phase 3: the emitters

Six documents gain paired axes. Portfolio gains nothing: `_emit_portfolio.py` has density and kappa panels and no probability axis at all. (Note in passing that its `meta['return_period_map'] = 'complement'` at `_emit_portfolio.py:206` pairs with nothing and does nothing. Out of scope, worth a TODO line.)

**`_emit_aggregate.py` `outcome_doc`** (serving `agg` and, through it, `pnl`), after the `p` axis at `:135-136`:

```python
# Named by no panel: the reflected reading of 'p'. The survival
# function is the reading a log axis exists for, which the
# non-exceeding probability is not, so the scales differ.
ChartAxis(id='survival', label='Exceeding probability',
          unit='probability', scales=('linear', 'log'),
          complement_of='p', suggested_range=(0.0, 1.0)),
```

**`_emit_severity.py`** (after `:149`) and **`_emit_reins.py`** (after `:146`): the same three lines, the same id and label.

**`_emit_distortion.py`** (after `:73`) and **`_emit_bounds.py`** (after `:163`), the two unit-square documents, which both name their axes `s` and `g`:

```python
ChartAxis(id='s_complement', label='1 - s', unit='probability',
          complement_of='s', suggested_range=(0.0, 1.0)),
ChartAxis(id='g_complement', label='1 - g(s)', unit='probability',
          complement_of='g', suggested_range=(0.0, 1.0)),
```

Labels are the literal coordinates rather than `s` and `ǧ(s)`: the reflected point is `(1 - s, 1 - g(s))` and that is true of every series on the panel, whereas naming it `ǧ` asserts an identity that holds only for the `g` curve. The unit square survives the reflection because both windows are `(0, 1)`, so `aspect='equal'` and `_square_window` (`:399`) are untouched. `envelope` is included because its axes are the same two probabilities and the reflected envelope is the dual envelope, a real object; call it out at review since ruling 3 named the distortion specifically.

## Phase 4: the entry points

`reflect=False` added to the signature and forwarded to `plot_chartdoc`, docstring paragraph each:

| method | file:line | note |
|---|---|---|
| `Aggregate.plot` | `_aggregate.py:4608` | |
| `Aggregate.reins_occ_plot` | `_aggregate.py:1433` | |
| `PnL.plot` | `_pnl.py:2447` | the signed chart, where reflect plus return period is a new picture |
| `Severity.plot` | `_severity.py:1808` | |
| `Distortion.plot` | `spectral.py:1675` | today takes no reading switches at all; it gains this one only, because it is the only reading the unit square declares |
| `Bounds.plot_envelope` | `bounds.py:531` | same, and same reason |
| `Portfolio.plot` | `_portfolio.py:3012` | **unchanged**, no probability axis |

## Phase 5: tests

`tests/test_chartdoc_readings.py` is the home suite, extending its synthetic-document style (`drawn(doc, **kwargs)` at `:40`). New cases, following the naming of the inversion block at `:194-283`:

- the reflection maps the values, `1 - v`, and only on the declaring axis
- the declared axis is substituted: label, `scales`, window all come from it
- a panel whose document declares no pairing is untouched
- marks reflect with their axis (mirroring `:236`)
- reflect rides inversion: `invert=True, reflect=True` on a Lee panel
- reflect plus return period on a `complement` document is identical to return period alone
- reflect plus return period on a `reciprocal` document reads `T = 1 / (1 - p)`, the flip
- the step direction survives the mirror (the Phase 2 argument, asserted on the drawn path)
- `MAX_RETURN_PERIOD` and the companion window release do not fire on reflect alone

`tests/test_charts_ir.py`, beside the `reciprocal_of` validation cases at `:418-441`: names an existing axis, is not itself drawn, points at something drawn, one pointer per axis, one paired axis per target, and `complement_of` omitted at default leaves an existing hash unchanged.

Per-chart declaration assertions in `test_chart_agg.py` (beside `:77-113`), `test_chart_severity.py`, `test_chart_pnl.py`, `test_chart_reins.py`, `test_chart_distortion.py`, `test_chart_bounds.py`.

**Hash churn to expect.** Six documents gain an axis, so their canonical bytes and hashes move. Nothing in `tests/` pins a hash literal (checked: every `doc_hash` assertion in `test_charts_ir.py` compares computed values), and `tests/data/chartdoc_baselines/` holds image baselines of default readings, which do not move. The API's chart ETags invalidate once on sync, which is correct and harmless.

## Phase 6: docs and release hygiene

- `docs/2_aggregate_overview/pipeline-exhibits-and-charts.rst`: the declared-readings section at `:200-336`, the chart-by-chart entry points at `:214-233`, the "Readings offered" column of the panel table at `:243-333`, and the sentence at `:334` that today says "Its five switches".
- `dev/summary-exhibits-and-charts.md`, the source that rst table is transcribed from.
- `docs/2_aggregate_overview/features.rst`: a worked `reflect=True` example beside the `return_period=True` one at `:948-960`.
- `dev/TODO.md`, `dev/FEATURES.csv` (`uv run python dev/regen_features.py`, public surface changed), plan moved to `dev/done/` at close.
- One version bump to `1.0.0a267` in `pyproject.toml`, one `CHANGELOG.md` section under `[Chart-Reflected-Reading]`, one commit, subject only:
  `[Chart-Reflected-Reading] a267: a probability axis declares its reflection, and a Lee panel reads the survival function`

The CHANGELOG entry must record the version-skew consequence, since the API is the other consumer: an older `aggregate` build calling `load_chart_doc` on a new document raises `ChartAxis carries unknown field(s) ['complement_of']`. That is the standing consequence of every additive field, and the fix is the standing one, sync LIB before serving.

## The API half (owed after the bump, not executed here)

LIB owns the meaning, the app draws what it is served, and the two renderers are kept in step by hand. Nothing on the wire changes: `reflect` is a renderer switch, so the route signature, `_chart_options`, the `(oid, name, window, detail, encoding)` cache key and the ETag at `routes/objects.py:2177-2304` are all untouched. What is owed, for a round note or the API's `dev/TODO.md`:

- `web/src/charts/chartdoc-to-echarts.js`: `pairedReading` (`:258`) takes the pointer field; `readings` (`:304-338`) gains `reflect: axes.some((a) => a.complement_of)`; `panelAxes` (`:726-744`) gains the substitution and the flip; the composed map reaches `realizeXy` (`:756-770`) and the marks (`:948-952`); `view` defaults at `:1310`.
- `web/src/charts/mount.js`: `VIEW_DEFAULTS` (`:60-93`) gains `reflect: false` and `VIEW_KEY` moves `v4` to `v5`; `CONTROLS` (`:145-184`) gains a `reflect` button placed between `full range` and `return period`, before it rather than after because the return-period reading composes on top of it. That order is the canonical control order and should be stated in both places.
- `dev/scripts/smoke-charts.mjs` (`:142-202`): a reflect assertion in the same shape as the existing per-reading ones.
- `uv sync --all-extras` on the API side before anything is believed, per the standing version-skew trap.

## Verification

1. Edit loop, tier 1: `.venv/Scripts/python.exe -m pytest -n0 --dist no --testmon-forceselect`.
2. Before declaring done, tier 2: `UV_PROJECT_ENVIRONMENT=.venv uv run pytest` (`UV_LINK_MODE=copy`).
3. At the bump, tier 3: `UV_PROJECT_ENVIRONMENT=.venv uv run pytest -m 'slow or not slow'`.
4. Interactive smoke, all six documents both ways:
   ```python
   from aggregate import build
   a = build('agg Eg 100 claims sev lognorm 100 cv 2 poisson')
   a.plot(invert=True)                            # F(x)
   a.plot(invert=True, reflect=True)              # S(x)
   a.plot(invert=True, reflect=True, log=True)    # the log survival plot
   a.plot(reflect=True, return_period=True)       # same curve as return_period alone
   ```
   and on a signed `PnL`, confirm `reflect=True, return_period=True` reads the upside tail rather than the shortfall.
5. **Before and after Artifact.** This adds new figure readings, so render each affected chart with the switch off and on, publish the comparison page, and wait for the author's read before the commit.

## Open points for review

- **Name collision.** `reflect` already means `c - X` on a severity in this library (`sev_reflect`, `_apply_reflect`, `ReflectedSeverityClampWarning`). Different namespaces, a DecL severity clause against a renderer switch, and the same geometric idea in both. Kept because the author named it, flagged because the house rule is one canonical name per concept. The renderer's private helper is `_reading_map`, not `_reflect`, so there is no symbol collision.
- **Distortion labels.** `1 - s` and `1 - g(s)` are literal and always true; `s` and `ǧ(s)` would read better and are true only of the `g` curve. Also decide whether `1 - g(s)` wants a `tex` entry through `complete_tex`.
- **`envelope` inclusion**, per Phase 3: ruling 3 named the distortion, and the envelope is the same unit square with the same two probability axes.
