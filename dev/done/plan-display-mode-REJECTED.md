# Plan — `dev` / `user` display mode via a `ReprMixin` — REJECTED

> **REJECTED 2026-07-27 (author): "not worth effort."** Never executed — there
> is no `ReprMixin` or `display_mode` anywhere in `src/`. The `[Display-Mode]`
> item is struck from `dev/TODO.md`; the reasoning is summarized in
> `dev/done/plans-considered-and-rejected.md`. The one durable idea inside it —
> a *required* display surface every first-class object must expose — is already
> served by `dev/FEATURES.csv` + `tests/test_fcc_surface.py`, and the
> report-shape question it was gated behind lives on as `[Reporting-Guidelines]`
> (`dev/reporting-guidelines.md`). Text below is the original draft, unedited.

> **Status: DRAFT — not executed.** A **presentation-only** mode toggle that
> chooses *which* view a first-class object renders, without changing any
> computed value. `user` (default) keeps the new tidy headline; `dev` restores
> the old QA-style view. Round 1: `Aggregate`, `Portfolio`, `PnL`.

---

## Goal & the one hard invariant

The author is used to the old `_repr_html_` (the moment-vs-estimate QA table) and
wants to switch back and forth. Implement a **display mode** with two values:

- **`user`** (default) — the current headline: `summary_df` (moments +
  percentiles) + the `tail_df` return-period table. Unchanged from today.
- **`dev`** — the *old* view: the `validation_df` QA frame (moment-vs-estimate)
  plus the old text/info blob.

**Invariant (non-negotiable): mode is presentation-only.** It selects *which
frame is rendered*; it never changes a computed number, a stored attribute, or
the result of any method. A round-trip `dev`↔`user` on the same object yields
byte-identical numerics. Enforced by a test (see Tests).

**No renaming (author point 4).** `validation_df` and `summary_df` keep their
names and meanings. `dev` mode simply renders `validation_df` where `user` mode
renders `summary_df`. We are choosing a frame to display, not moving names around.
Grounding (verified):
- `_aggregate.py:2274` `_repr_html_` today builds `_html_info_blob()` +
  `summary_df` + `tail_df`.
- `_aggregate.py:3753` `validation_df` (`= _describe()`) is the old QA frame.
- `_aggregate.py:3804` `summary_df` is the new moments+percentiles headline.
- The "old" `_repr_html_` body (info blob + QA table) is recoverable from git.

---

## Design

### 1. `ReprMixin` — the capability packet

A small mixin class (new file `src/aggregate/_repr_mixin.py`, or fold into an
existing display home) added to each first-class object's bases. It owns the
**dispatch**, not the per-class content:

```python
class ReprMixin:
    _display_mode = None                    # None => inherit the global

    @property
    def display_mode(self):                 # resolve: instance -> runtime -> config
        return resolve_display_mode(self._display_mode)

    @display_mode.setter
    def display_mode(self, value):
        self._display_mode = _validate_mode(value)   # 'dev' | 'user' | None

    def _repr_html_(self):
        if self.display_mode == 'dev':
            return self._repr_html_dev()
        return self._repr_html_user()

    def __str__(self):
        if self.display_mode == 'dev':
            return self._str_dev()          # info + validation_df
        return self._str_user()             # info + summary_df
```

Why a mixin, not a base class: `Portfolio` is not an `Aggregate`, and `PnL` is
*composition* (has-a `Aggregate`). A shared base class would impose a false
is-a; a mixin grants the display capability to all three regardless of hierarchy.

Each host class supplies the renderers for **both** the HTML and the text view:
- `_repr_html_user` / `_repr_html_dev` — Jupyter HTML. `user` = the current body;
  `dev` = the old body (info blob + `validation_df`), restored from git.
- `__str__` is **also routed** (author decision): `dev` = `info` + `validation_df`;
  `user` = `info` + `summary_df`. Today's `__str__` (`_aggregate.py:2073`) is
  `info` + `summary_df`; its output silently flipped to the new percentile frame
  when the old `summary_df` was renamed `validation_df` (it references the frame
  *by name*), so routing it restores the old text view honestly. **`info`
  (value_type / `bs` / `log2` / x-range / premium) renders in both modes** — it
  is always wanted.

`__repr__` (`_aggregate.py:2066`, `f'{name}, <super>'`) stays short and stable —
it is the eval-style identity line, not a report, and was untouched by the
display change. Only `_repr_html_` and `__str__` are mode-routed.

### 2. Mode resolution — three layers, resolved at *display* time

`_repr_html_` fires when the object is displayed (often long after `build`), so
mode must be read then, not frozen at construction. Resolution order (highest
first):

1. **instance override** — `obj.display_mode = 'dev'`, or `build(..., mode='dev')`
   (sugar that sets the override on the constructed object). `None` = inherit.
2. **runtime global** — `aggregate.set_display_mode('dev')`, a lightweight
   mutable module-level value for interactive flips ("show me dev now"), seeded
   from the config default at import.
3. **config default** — a new `DisplaySettings.mode` field in `config.py`
   (default `'user'`), overridable by `~/.aggregate/config.toml` and an
   `AGGREGATE_DISPLAY_MODE` entry added to the env allow-list.

Layer 3 reuses the existing frozen `Settings` singleton (`get_settings()`); layer
2 is needed *on top* because that singleton is read-once-per-session (frozen), so
it cannot itself provide an interactive runtime flip. `build(mode=)` is the
highest-priority call-site layer, matching the existing config cascade
(`config.py:11`).

### 3. The config field

Add a `DisplaySettings` section dataclass (mirrors `BuildSettings` etc.) with
`mode: str = 'user'`, wire it into `Settings`, add `AGGREGATE_DISPLAY_MODE` to the
env allow-list, and document it in `data/config.default.toml`.

---

## Round-1 scope: `Aggregate`, `Portfolio`, `PnL`

Each gains `ReprMixin` and the two renderers. This is part of making these three
**first-class citizens** with a *uniform, enforced* display contract — the same
consistency goal `dev/FEATURES.csv` tracks by hand.

- **`Aggregate`** — `user` = current (`summary_df`+`tail_df`); `dev` =
  `validation_df` + info blob.
- **`Portfolio`** — analogous; `_portfolio.py:596` `_repr_html_` splits into
  `_user`/`_dev` (its validation/summary analogues).
- **`PnL`** — `user` = the new signed `summary_df`/GCN view; `dev` = its
  validation-style frame (or, where a PnL has no separate QA frame, the agg's
  `validation_df` + the signed summary — decide in execution).

### FEATURES.csv contract (the consistency win)

`ReprMixin` is the natural home for the **required display surface** every
first-class object must expose (`_repr_html_`, `summary_df`, `validation_df`,
`info`, a `qd` hook). Declaring these on the mixin (even as `NotImplementedError`
stubs) turns the FEATURES.csv "consistency" audit from hand-maintained into
type-nudged. Update `dev/FEATURES.csv` + the introspection cross-check when the
surface lands ([[project_features_csv]]).

---

## Tests (`tests/test_display_mode.py`)

- **Toggle (HTML + text):** with mode `user`, `a._repr_html_()` and `str(a)`
  contain the `summary_df` headline (percentile columns); with `dev`, they contain
  the `validation_df` QA columns (e.g. `Est EX` / `Err EX`) and not the
  percentiles. `info` (bs / log2 / value_type) appears in **both** modes. Both
  views, all three of Agg/Port/PnL.
- **Numeric invariance (the invariant):** for a fixed object, every public
  numeric attribute / frame (`actual_m`, `est_m`, `density_df`, `summary_df`,
  `validation_df`, …) is identical under `dev` and `user`. Mode changes display
  only.
- **Resolution order:** instance override beats runtime global beats config
  default; `build(mode='dev')` sets the instance override; unset instance inherits
  the global; `set_display_mode` flips already-built objects at display time.
- **Config/env:** `AGGREGATE_DISPLAY_MODE=dev` seeds the default;
  `DisplaySettings.mode` round-trips through TOML.
- **Default is `user`** with no config/env present.

---

## Out of scope / future

- `Severity`, `Distortion`, `Bounds`, … gain the mixin in a later round.
- Other *named* display options (float precision, which frames render) — add as
  distinct options, **not** by overloading the `dev`/`user` axis (one axis = one
  concept). [[feedback_one_canonical_name]]
- Routing `qd` through mode (round 1 routes `_repr_html_` + `__str__`; `qd` is
  round 2 — see Open questions). `__repr__` stays minimal by design.

---

## Open questions (author)

1. **`qd` routing — round 1 or 2?** `_repr_html_` and `__str__` are both routed in
   round 1 (author confirmed text should switch too). Open: does `qd`
   (`utilities.qd`) also switch its lead frame by mode now, or in round 2? *Lean:
   round 2, once the mixin shape is proven on `_repr_html_`/`__str__`.*
2. **Mode value names** — `dev` / `user` (author's words) vs `full` / `concise`.
   *Lean: keep `dev`/`user`.*
3. **`PnL` dev frame** — does a `PnL` get its own validation frame, or does `dev`
   show `pnl.agg.validation_df` + the signed summary? Decide in execution.
4. **Mixin home** — new `_repr_mixin.py` vs folding into an existing module.

---

## Housekeeping

Plan-based code change → bump `1.0.0a*` in `pyproject.toml`; add a `CHANGELOG.md`
section (display mode: `dev`/`user` toggle, default `user`, env
`AGGREGATE_DISPLAY_MODE`); update `dev/FEATURES.csv` + cross-check; full suite
green before any commit; move this plan to `dev/done/` and tick `dev/TODO.md` at
close.
