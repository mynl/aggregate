# Plan: `[Label-Canonical]` — one human-facing `label`, no `display_` twins

## Motivation

The `LabeledMixin` surface currently exposes **two** near-synonym public names —
`display_label` (stored explicit, the DecL `as` clause, may be `None`) and
`display_name` (resolved property: `display_label → derived default → name`).
They are two legitimate *roles* (stored slot vs. resolved output), but the
*words* collide: `name`, `display_label`, `display_name` are three
near-synonyms for what a reader parses as one idea, and the `display_` prefix is
noise. This violates the house **one-canonical-name** rule and reads like a
backwards-compat crutch even though it isn't.

**Target public surface — two words, each pulling its weight:**

| Name        | Role                                                                 |
|-------------|----------------------------------------------------------------------|
| `ob.name`   | the handle / knowledge id (the DecL bareword). Unchanged — too embedded to rename to `id`. |
| `ob.label`  | the one human-facing string. A **property**, resolved: `_label → _label_default() → name`. Never blank. |
| `ob._label` | private stored slot: the explicit label or `None` (the DecL `as` value). |

`display_name` (the resolved property) → **`label`**. `display_label` (the
stored attribute) → **`_label`** (private). Publicly there is now only
`name` + `label`; the confusing middle twin is gone.

## Scope decisions (ruled by author)

- **Convert roll-your-own label classes to the mixin — no holdouts.** The only
  such class is `copula.py`'s `Copula` / `CopulaShuffle`, which carry a
  hand-rolled `display_name` (empty-string sentinel) playing the *stored-label*
  role. They already have a `name` handle, so they fit the mixin cleanly.
- **`_pnl.py` `Leg` / `Group` are NOT mixin candidates and stay as-is.** They
  hold a single `self.label` string that is both handle and human label — there
  is no name/label split to model, and forcing the mixin would impose a
  distinction they don't need. Their `.label` attribute is unrelated to this
  rename and must be left untouched. *(This carve-out is intentional — recorded
  here so a future reader doesn't "finish the job".)*

## Blast radius (measured)

- `display_label`: ~155 refs across 12 files (attribute, **spec key**,
  constructor kwarg, docstrings).
- `display_name`: ~63 refs across 10 files (property reads, copula's stored
  attr, docstrings).
- Derived spec keys: `engine_display_label`, and the parser-internal
  `_premium_label` (which lands as the final spec key `consideration_label` —
  already clean, no `display_`).
- Tests: only `tests/test_decl_labels.py` (16 refs).
- Docs `.rst`: **no** references — no doc `:attr:` sweep needed.
- Frozen snapshot `tests/data/expected_specs.json`: **no** `display_label` key
  (labels post-date the SLY snapshot) → **no snapshot regen**.

This is why it's a plan, not a sed: `display_name` maps to the **public
property** on mixin hosts but to the **private `_label`** on copula; and the
grammar already uses `label` as a rule name (below). A blind rename breaks both.

---

## Phase `[Inventory-And-Grammar-Disambiguation]`

The grammar (`decl.lark`) already defines two rules:

```
display_label: AS label   -> display_label_some
label: ID                 -> label_id
```

i.e. `label` already names the rule that captures the raw text, and
`display_label` wraps `AS label`. Renaming the **spec key** to `label` collides
conceptually with the existing `label` grammar rule.

- **Grammar rules:** rename the wrapper rule `display_label` → **`as_label`**
  (and its alias `display_label_some` → `as_label_some`) so the grammar reads
  `as_label: AS label`. The inner `label: ID` rule (raw text capture) keeps its
  name — it is the token, not the spec key.
- Update the matching transformer methods in `parser.py`
  (`display_label_some`, and the `_render_label`/`display_label` handling).
- Re-run the grammar → `ref_include.rst` regeneration afterward per the standing
  note (write to real `docs/`, not `src/docs/`).

## Phase `[Mixin-Surface-Rename]` — `_labeled.py`

- `display_label` attribute → `self._label` (private).
- `display_name` property → **`label`** property; body becomes
  `self._label or self._label_default() or self.name`.
- `_display_default()` → **`_label_default()`** (purge `display`).
- `_init_labels(*, display_label=None, label_map=None)` →
  `_init_labels(*, label=None, label_map=None)`; assigns `self._label`.
- `_title_name` keys off `self._label is not None` and leads with the resolved
  label (`f'{self.label} ({self.name})'` when `_label` set).
- Update the module docstring (the `display_label` / `display_name` prose) and
  the `LabeledMixin` class docstring surface list.

## Phase `[Spec-Key-And-Parser-Rename]` — `parser.py`

- Spec key `'display_label'` → **`'label'`** everywhere it is produced.
- `'engine_display_label'` → **`'engine_label'`**.
- Parser-internal `head['_premium_label']` → `head['_label']` (transient; the
  final emitted key `consideration_label` is unchanged).
- All `_init_labels(display_label=...)` call sites (17 across `_aggregate.py`,
  `_portfolio.py`, `_severity.py`, `_pnl.py`, and 13 in `spectral.py`) →
  `_init_labels(label=...)`.
- The build path that pulls the label out of the spec and into `_init_labels`
  reads the new `'label'` key.

## Phase `[Unparser-And-Consumers]`

- `decl_writer.py`: `_render_label(spec.get('display_label'))` → `'label'`;
  `engine_display_label` → `engine_label`.
- `underwriter.py`: `spec.pop('engine_display_label')` → `'engine_label'`.
- `spectral.py`, `bounds.py`, `plots/_distortion.py`, `plots/_portfolio.py`,
  `_portfolio_common.py`, `_aggregate.py`: `.display_name` reads → `.label`;
  `display_label=` constructor kwargs → `label=` (e.g.
  `Distortion('tvar', p=p0, label='TVaR(...)')`).
- Constructors of every host (`Aggregate`, `Portfolio`, `Severity`, `PnL`, each
  `Distortion*`) take `label=` instead of `display_label=`.

## Phase `[Copula-To-Mixin]` — `copula.py`

- `class Copula:` → `class Copula(LabeledMixin):`.
- `__init__(..., display_name='')` → `label=None`; call
  `self._init_labels(label=label)` once `self.name` is set.
- Same for `CopulaShuffle.__init__`.
- Replace the two `if self.display_name: return self.display_name` repr
  branches with the resolved `self.label`.
- **Wrinkle — `name=None` default:** `Copula.__init__(self, name=None, ...)`
  lets the resolved chain bottom out at `None`. Give copula a
  `_label_default()` (kind-based pretty, e.g. `'Gumbel(0.4)'`) so `label` is
  never blank even when both `name` and `_label` are unset. Confirm the
  `''`-vs-`None` sentinel change doesn't break existing copula call sites.

## Phase `[Tests-And-Round-Trip]`

- Update `tests/test_decl_labels.py` (16 refs: `display_label`/`display_name`
  → `label`; any `_init_labels`/kwarg usage).
- Add/confirm `decl-testers.agg` lines exercising `as` on object, engine
  (`engine_label`), and premium (`consideration_label`) still round-trip
  through the renamed unparser.
- Full gate: `uv run pytest -m 'slow or not slow'`.

## Phase `[Docs-And-Release]`

- Regenerate `ref_include.rst` from the grammar (real `docs/` path).
- Bump `pyproject.toml` `1.0.0a132` → `1.0.0a133`.
- `CHANGELOG.md`: new `## 1.0.0a133` section — public rename
  `display_name`/`display_label` → `label` (+ `label=` kwarg), copula folded
  onto `LabeledMixin`, note the `_pnl.Leg/Group.label` carve-out; flag as a
  breaking rename (pre-1.0, acceptable).
- `dev/TODO.md`: mark item done; move this plan to `dev/done/`.
- `dev/FEATURES.csv`: re-run `uv run python dev/regen_features.py` — the label
  surface row(s) changed, so audit/curate.

## Open question for author

The parser-internal `_premium_label` → `_label` rename is cosmetic (transient
key inside `head`, never emitted). Fold it into this pass, or leave it? Default:
fold, for zero lingering `display`/`premium_label` naming. Steve: FOLD. 
