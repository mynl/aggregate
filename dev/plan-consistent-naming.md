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

## Blast radius summary

| Change | Files touched | Risk |
|---|---|---|
| validation rename (+alias) | `distributions.py`, `portfolio.py`, `underwriter.py`, `utilities.py`, `constants.py`, + `CLAUDE.md` doc mention | **high** — 5 source files incl. both hot files; public API |
| reins_description → property | `distributions.py` only (def + 2 call sites) | low (single file) but breaking for external arg callers |
| bs workers (optional) | `distributions.py` only | low; deferred |

Published `docs/` has **zero** `explain_validation` symbol references (the
"validation" hits there are the English word), so docs need no symbol edits —
only confirm prose still reads correctly. `dev/` planning notes reference the old
name historically; leave them or sweep in a final pass.

## Sequencing

1. reins_description → property (single file, do first — smallest, self-contained).
2. validation rename + back-compat alias (the main, multi-file change).
3. (optional) bs worker rename — separate pass, only if requested.
4. Version bump in `pyproject.toml` + `CHANGELOG.md` entry noting the
   `explain_validation` → `validation_explanation` rename (alias retained) and the
   `reins_description` property/breaking-arg change. Mark the item done in
   `dev/TODO.md`.

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
