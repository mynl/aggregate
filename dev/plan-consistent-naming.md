# plan-consistent-naming

## ⚠️ Coordination note — do NOT run this in parallel

This plan renames public properties/methods in the two hottest files in the
repo (`distributions.py`, `portfolio.py`) and touches shared docs/house files
(`CLAUDE.md`, `CHANGELOG.md`). It is a guaranteed merge-collision magnet. **Run
it on its own, when no other branch is editing `distributions.py` /
`portfolio.py`.** It is explicitly *not* a tight-fix / parallel-safe change.

Because it renames public API it is a **plan-based code change → version bump
+ CHANGELOG entry** (with a breaking-change note), per the release workflow.

## Context

The narrative/reporting surface on `Aggregate` and `Portfolio` should follow one
rule: **`<item>_<aspect>`, aspect is a noun, and every short/long narrative is a
property returning a string.** Three families already obey it; one does not, and
two have smaller warts. This plan makes them uniform so `ob.help('.*description')`
/ `.*explanation` introspection returns a clean, predictable set.

### Target convention (the `tail` family is the template)

| Aspect | Form | Kind | Returns |
|---|---|---|---|
| structured frame | `<item>_df` (or `<item>_describe` where it mirrors `.describe()`) | property | DataFrame |
| short narrative | `<item>_description` | property | str |
| long narrative | `<item>_explanation` | property | str |
| predicate | `valid` | property | bool |

Reference family, already correct: `tail_class`, `tail_description`,
`tail_explanation`, `tail_df` — leave untouched.

## Current state — what's consistent vs not

| Family | Status | Notes |
|---|---|---|
| **tail** | ✅ template | nothing to do |
| **bs / window** | ⚠️ minor | props `bs_window_df`, `bs_description`, `bs_explanation` are correct properties. Workers `bs_describe()` / `bs_explain()` are module funcs in `distributions.py` (739, 775) providing the `color=` variant — near-homonyms of the props but in a different namespace. |
| **reins** | ⚠️ one fix | `reins_describe` is a `@property` df (mirrors `.describe()`) — **keep**. `reins_density_df`, `reins_stats_df` correct. **`reins_description(kind, width)` is a method, not a property** — must become a property. |
| **validation** | ❌ the offender | `valid` (bool property) is fine. **`explain_validation()` is `<verb>_<item>` — backwards from every other family, and a verb not a noun.** |

## Changes

### 1. validation — the main fix

Rename the long-narrative method to match `tail_explanation` / `bs_explanation`:

- **`explain_validation()` → `validation_explanation` (property, str).**
- Keep **`explain_validation()` as a thin back-compat alias** that returns
  `self.validation_explanation` (it is the documented mechanism in `CLAUDE.md`
  and is called widely in `dev/` notes). Mark it deprecated in its docstring;
  do not remove this cycle.
- `valid` (bool predicate property) — **unchanged**; predicates read fine as
  adjectives.
- Optional completeness (only if a short form is wanted): add a
  `validation_description` short property. The current text is long-form, so
  this is genuinely optional — note it but default to *not* adding it unless the
  author asks, to keep blast radius down.
- Leave the `validate_*` module functions (`validate_discrete_distribution`,
  `_validate_moments`, `_validate_reins_layers`) alone — they are verbs that
  *perform* validation, so verb form is correct.

Call sites to update (both classes define `explain_validation`; internal callers
read it): `distributions.py` (~5 refs incl. the `_describe` info-row at
line 4535), `portfolio.py` (~4), `underwriter.py` (2), `utilities.py` (3),
`constants.py` (1, a comment/centralized-wording ref). Point internal callers at
the new property; the alias covers anything missed.

### 2. reins — make the narrative a property

`reins_description(self, kind='both', width=0)` (`distributions.py:5986`) is the
short narrative but takes args, so it can't be a bare property as-is.

- Add property **`reins_description` (str)** returning the `kind='both', width=0`
  text — the consistent narrative surface.
- Move the parameterized body to a helper **`_reins_description(self, kind='both',
  width=0)`** (private) or a public **`reins_description_for(kind, width)`** if an
  external parameterized form should stay supported.
- Update the two internal callers `self.reins_description('occ')` /
  `('agg')` at `distributions.py:4533-4534` to the helper.
- **Breaking change:** any external `a.reins_description('occ')` call breaks
  (property takes no args). Note in CHANGELOG. (Grep shows no test/doc callers —
  only the two internal sites — so external exposure is low.)

### 3. bs / window — lowest priority, optional

The `bs_*` *properties* are already correct. Only wart is the verb workers
`bs_describe` / `bs_explain` shadowing the noun props by one letter. **Default:
leave them** — they're module-level plumbing for the `color=` variant, ~24
internal call sites in `distributions.py`, and renaming buys little. If desired
later, rename workers to non-homonym names (e.g. `_format_bs_grid` /
`_format_bs_grid_long`) in a separate pass. Out of scope here unless asked.

## Changes — narrative content & `tail_df` data (added 2026-06-19)

These are content/wording fixes on the same reporting surface, batched into this
plan because they touch the identical hot lines. They go beyond pure renaming,
but the blast radius and the files are the same, so they ride along.

### 4. `bs_description` / `bs_explanation` — say `x_max`, not `top`

The one-liners and prose print the realised upper grid edge as `(top=…)` /
`top=…`. The grid frame (`bs_window_df`) already exposes that edge as the
**`x_max`** column, so the narrative should use the same word — no reason for the
prose to invent a synonym for a column the reader can see.

- **Aggregate:** `bs_describe` (`distributions.py` ~762-763) and `bs_explain`
  (~808-809, and the clip message ~820-822).
- **Portfolio:** `bs_description` (`portfolio.py` ~995-996) and `bs_explanation`
  (~1041-1042, clip ~1048-1050).
- Keep the internal helper `_bs_grid_top` and the local `top` variable (they
  *compute* `x_min + 2**log2 * bs`); **only the user-facing label changes** from
  `top=` to `x_max=`.
- Target one-liner: `portfolio grid: bs=1, log2=9, x_min=0, x_max=512`.

### 5. `Portfolio.tail_df` — complete the `total` row

`Portfolio.tail_df` (`portfolio.py:1053-1089`) builds the `total` row as
`INFO_NA` and fills only the tail classes / concentration. Two fixes:

a. **Complete `min` / `max` on the `total` row — reuse the existing derivation,
   don't reinvent it.** The total's structural support is already worked out
   carefully per unit: each unit's `tail_df` aggregate `min` / `max` comes from
   `tail._agg_support`, which handles `inf`, signed (P&L) books, and limit/splice
   caps. The total support is the additive combine of those ends, and that
   additive combine (with the `0 * inf` / signed handling) is exactly what
   `tail._agg_support` already implements. **Decision (author): read `min` / `max`
   straight from the existing Portfolio call that already returns the combined
   structural support** — do not hand-roll a fresh sum. *Source call: TBD — author
   to name the specific method/attribute; the implementer wires the `total` row to
   it.* (The realized-grid extent `density_df.index[0]` / `[-1]` and the `q()`
   quantiles are **not** it — they are grid-bounded and report a finite max for an
   unbounded total.)

b. **Tail class stays worst-of; the per-*side* label is not a blanket worst-of.**
   The decay class of the total **is** the worst-of (`self.tail_class`, the
   thickest summand under independence) — keep that. The bug is that today *both*
   `left_tail` and `right_tail` get that single worst-of `decay` label, so a book
   of non-negative units reports a `super-exponential` **left** tail when its
   floor is finite. Apply the worst-of decay only where the corresponding support
   end is infinite, and `bounded` where it is finite — the `_side_class` rule
   (`left_tail = bounded if total min finite else worst-of decay`; same on the
   right). With 5a giving a finite total `min`, the left side correctly reads
   `bounded`. This is the "port reports left tail super-exp ⇒ should be bounded"
   fix; it does **not** change `tail_class`.

### 6. Replace `concentration_p` with `cv` (Aggregate **and** Portfolio)

`concentration_p = Phi(mean / sd)` is opaque (and saturates at ~1 for any real
book). Replace the *reported* diagnostic with the **coefficient of variation
`cv = sd / mean`** — directly interpretable, and `inf` (when `mean == 0`) is a
perfectly good value.

- `tail.concentration(m, sd)` returns `(concentrated_flag, cv)` instead of
  `(flag, p)`. The **`concentrated` flag stays** — it gates the windowed
  left-lift, and `concentrated ⇔ cv < CONCENTRATION_CV` already, so only the
  second returned value changes from `p` to `cv = sd / m` (`inf` when `m == 0`).
- Rename `TailRow.concentration_p → cv` (the dataclass field + docstring,
  `tail.py:996`, ~981-983) and the `tail_df` column `concentration_p → cv`
  (`tail.py:1341,1345`).
- Narratives: `describe_rows` (`tail.py:722-724`) and `explain_rows`
  (`tail.py:793-795`) — `"concentrated / not concentrated (cv=…)"`, and
  `"It is concentrated: cv ~ …."`.
- `Portfolio.tail_df` (`portfolio.py:1084-1085`) writes `cv` on the `total` row
  (the total-moment `cv`, from the same `concentration(agg_m, agg_sd)` call).
- **Breaking:** public `tail_df` column rename → CHANGELOG note.

### 7. Rewrite `bs_explanation` to the new template (`window width`, not `span`)

Replace the current portfolio `bs_explanation` prose. **"window width" replaces
"span"** throughout (the `mm`/`rms`/`sum`/`sbj` candidates are all window widths
`W`, and `bs_window_df` already calls the realised value a window). The author's
target template (`<< >>` = a formatted number; `[ … ]` clauses appear only when
true; note the nested `[ ]` in the log2 discussion):

> The portfolio tail is `<<bounded | bounded-or-super-exponential | power>>`
> left/right. Log2 is `<< >>`. The unit tails are: `D1` bounded/bounded and
> `D2` bounded/subexponential. The recommended window width `<< >>` is based on
> portfolio method of moments `<< >>`, RMS(units) `<< >>`, sum(units) `<< >>`,
> and single big jump of `<< >>`. The window produces a raw bs `<< >>` which
> dyadically rounds to `<< >>` producing a final `<< >>` window width. [The
> distribution is discrete. [The window width supports lowering log2 to
> `<< >>`.]] [It has a natural lower support bound of `<< >>`.] [It has a natural
> upper support bound of `<< >>`.] [The distribution is concentrated with a CV
> of `<< >>`.] The recommended x_min is `<< >>` resulting in x_max of `<< >>`.
> [The analysis suggests increasing log2 to `<< >>`.]

Data sources for the placeholders (all already on `_bs_window_df` / `tail_df`):

| Placeholder | Source |
|---|---|
| portfolio tail left/right | `self.tail_df.loc['total', ['left_tail','right_tail']]` (post item 5b) |
| log2 / final window width | `used` row `log2`, `W` (= `x_max − x_min`) |
| MM / RMS / sum / sbj widths | `df.loc[['mm','rms','sum','sbj'], 'W']` |
| unit tails | per-unit `tail_df` rows |
| raw bs → dyadic bs | pre-round `bs` vs realised `used['bs']` |
| discrete / lowering log2 | discrete flag; `log2_need < log2` ⇒ headroom (see item below) |
| natural lower/upper bound | `total['min']` / `total['max']` finite (item 5a) |
| concentrated + CV | `concentration()` flag + `cv` (item 6) |
| x_min / x_max | `used` row |
| increasing log2 | `_bs_clip['need_log2']` when a clip occurred |

Open wording points to settle during implementation (not blockers): the exact
phrasing of the dyadic-rounding sentence, and whether the closing "increasing
log2" suggestion is always derivable cleanly (it is, from `_bs_clip`; when there
is no clip the clause is simply omitted).

**Aggregate mirrors this but simpler** — no unit rows, no `rms`/`sum` candidate
ordering. The `Aggregate.bs_explanation` template collapses to: the aggregate
tail one-liner; the winning method and its window width vs the candidates that
applied; raw→dyadic bs; the discrete / lower-log2, natural-bound, and
concentrated-CV optional clauses; the realised `x_min`/`x_max`; and the
clip "increase log2" suggestion when truncated.

## Blast radius summary

| Change | Files touched | Risk |
|---|---|---|
| validation rename (+alias) | `distributions.py`, `portfolio.py`, `underwriter.py`, `utilities.py`, `constants.py`, + `CLAUDE.md` doc mention | **high** — 5 source files incl. both hot files; public API |
| reins_description → property | `distributions.py` only (def + 2 call sites) | low (single file) but breaking for external arg callers |
| bs workers (optional) | `distributions.py` only | low; deferred |
| `top` → `x_max` label (item 4) | `distributions.py`, `portfolio.py` | low; pure label |
| `Portfolio.tail_df` total row (item 5) | `portfolio.py` (reuse `tail._agg_support` / `_side_class`) | low; reuses existing support derivation, adds per-side label |
| `concentration_p` → `cv` (item 6) | `tail.py`, `portfolio.py` | low-med; public column rename (breaking) |
| `bs_explanation` rewrite (item 7) | `distributions.py`, `portfolio.py` | med; prose-heavy, both classes |

Published `docs/` has **zero** `explain_validation` symbol references (the
"validation" hits there are the English word), so docs need no symbol edits —
only confirm prose still reads correctly. `dev/` planning notes reference the old
name historically; leave them or sweep in a final pass.

## Sequencing

1. reins_description → property (single file, do first — smallest, self-contained).
2. validation rename + back-compat alias (the main, multi-file change).
3. (optional) bs worker rename — separate pass, only if requested.
4. `top` → `x_max` label (item 4) — trivial, do alongside the bs work.
5. `concentration_p` → `cv` (item 6) — touches `tail.py` `concentration()`,
   `TailRow`, `tail_frame`, both narratives, and `Portfolio.tail_df`; land before
   the `bs_explanation` rewrite (the template's CV clause depends on it).
6. `Portfolio.tail_df` total-row completion + left-tail fix (item 5).
7. `bs_explanation` rewrite to the new template (item 7), Portfolio then
   Aggregate mirror.
8. Version bump in `pyproject.toml` + `CHANGELOG.md` entry noting the
   `explain_validation` → `validation_explanation` rename (alias retained), the
   `reins_description` property/breaking-arg change, **and the
   `concentration_p` → `cv` column rename**. Mark the item done in `dev/TODO.md`.

## Verification

- `uv run pytest` — full suite (no test references `explain_validation` or the
  parameterized `reins_description`, so green = no internal callers missed).
- Smoke: `from aggregate import build, qd; a = build('agg Dice dfreq [3] dsev [1:6]'); a.update(...)` then check
  `a.validation_explanation`, `a.explain_validation()` (alias, same text),
  `a.reins_description` (property), and a reinsured agg's `reins_describe`.
- `qd(a)` still renders the info block (it calls `explain_validation` /
  `reins_description` internally at `distributions.py:4533-4535`).
- `rg "explain_validation|reins_description\(" src` afterwards to confirm only the
  alias + helper remain.
- `rg "concentration_p|top=" src` afterwards — should be **zero** hits (all
  renamed to `cv` / `x_max=`).
- Build a 2-unit portfolio (`port USE.Port agg A 100 claims sev lognorm 100 cv 2
  poisson agg B 50 claims sev gamma 50 cv 1 poisson`), then check: `p.tail_df`
  `total` row has finite `min`/`max` and `bounded` left tail (item 5), a `cv`
  column (item 6), and that `p.bs_description` / `p.bs_explanation` read with
  `x_max` and the new window-width template (items 4, 7).
