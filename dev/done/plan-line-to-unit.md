# plan-line-to-unit

## ⚠️ Coordination note — big blast radius, run solo

This renames the oldest naming wart in the library — the portfolio sub-component
"line" (line-of-business) → "unit" — across `portfolio.py` (the hottest file),
several output-frame labels, public keyword arguments, ~6 test files, and the
docs. It is a guaranteed merge-collision magnet and is **not** parallel-safe.
Run it on its own branch with nothing else touching `portfolio.py`.

It removes/renames public API (`line_names`, the `'line'` index label, `line=`
kwargs) → **plan-based code change ⇒ version bump + CHANGELOG with a
breaking-change section**, per the release workflow.

## Context / motivation

When `aggregate` began, a `Portfolio`'s sub-components were called *lines* (line
of business). *Pricing Insurance Risk* (Mildenhall & Major, 2022) settled on
**unit** as the better generic term — a line, geography, business unit, single
account, segment, reinsurance layer all read naturally as a "unit". Somewhere in
the 0.20.x series `unit_*` proxies were bolted on over the original `line_*`
storage, leaving two half-names for one concept. This plan finishes the job:
**`unit` becomes canonical, `line` is removed**, the thin `unit_* → line_*`
pass-throughs are deleted (the canonical names absorb them), and the docs follow.

## The careful part — what "line" means, and keep vs. rename

`line` is badly overloaded in this codebase. A blind replace would corrupt
matplotlib calls, parser error reporting, and an actuarial term of art. The
rename touches **only the line-of-business sense**. Classify every hit first.

| Sense | Examples | Action |
|---|---|---|
| **LOB sub-component (the target)** | `line_names`, `line_names_ex`, `line_name_pipe`, `line_renamer`, `for line in self.line_names`, the `'line'` index/column label, `line=`/`lines=` kwargs | **rename → unit** |
| source-text line | `source_line`, `program_line`, "line N", parser/underwriter/`parser_errors` line-number machinery, "parse each line" | **keep** |
| matplotlib | `linewidth`, `axhline`, `axvline`, `linestyle`, `linefmt`, `Line2D`, `set_line` | **keep** |
| substring coincidence | `linear`, `newline`, `inline`, `baseline`, `multiline`, `headline`, `underline`, `online`, `lifeline`, `splitline` | **keep** |
| actuarial term of art | **"loss on line"** / `lol` (= expected layer loss ÷ limit; `distributions.py:3267,3405`) — *not* line-of-business | **keep** |
| prose | "one-line summary", "five-line core", "three aligned lines" | **keep** (wording) |
| local accumulator var | `line = (f'... grid ...')` in `bs_describe` (`distributions.py:762`) | keep or rename to `text`; cosmetic |

The Definition of Done (below) is calibrated to this table: after the change,
`rg line -g *.py` should return only rows from the **keep** classes.

## Canonical target API

| Today | After | Notes |
|---|---|---|
| `Portfolio.line_names` (real storage, set in `__init__`) | `Portfolio.unit_names` (real storage) | |
| `Portfolio.line_names_ex` | `Portfolio.unit_names_ex` | `+ ['total']` |
| `Portfolio.line_name_pipe` | `Portfolio.unit_name_pipe` | regex alternation |
| `Portfolio.unit_names` / `unit_names_ex` / `n_units` (pass-throughs, lines 4695–4706, *"what these should have been called!"*) | **deleted** — names now belong to the real properties; `n_units` stays as `len(unit_names)` | |
| `Portfolio.line_renamer` / `self._line_renamer` | `Portfolio.unit_renamer` / `self._unit_renamer` | |
| output frame index label `name='line'` | `name='unit'` | **breaking** (see B) |
| `*.pentagon_at(line='total')`, `Bounds(line='total')`, `Pentagon.as_frame/from_row(line=)` | `unit='total'` | **breaking** (see C) |
| `multivariate` ctor `lines=None` | `units=None` | **breaking** (see C) |
| `Portfolio.unit_density` / `unit_density_df` / `aligned_unit_density_df` / `add_exa(unit_state)` | **unchanged names** (already correct); only their `line_names` internals change | |

**Decision point — back-compat alias?** The author has wanted `unit` since 2019
and asked to "get rid of line completely." Recommendation: **clean break, no
`line_names` alias** (this is a `1.0.0a*` pre-release; adding a reverse
pass-through would re-introduce exactly the wart we are deleting). If external
notebooks are known to lean on `line_names`, the fallback is a single
deprecation-alias property `line_names → unit_names` (warns once) kept for one
cycle — but that costs one "explainable" `rg line` hit and is **not** the default
here. Confirm before implementing.

## Changes (staged — tests are expected RED until Stage F; green only at the end)

### A. Core storage & accessors (`portfolio.py`)

1. `__init__` storage: `self.line_names` (210) → `self.unit_names`; populate at
   283; `self.line_names_ex` (290) → `self.unit_names_ex`; `self.line_name_pipe`
   (291) → `self.unit_name_pipe`; `self._line_renamer` → `self._unit_renamer`.
   *Caveat:* `unit_names`/`unit_names_ex`/`n_units` currently exist as
   **properties** (4695–4706). To make `unit_names` a plain attribute, **delete
   those property defs** so the instance attribute is not shadowed (recall the
   CLAUDE.md attribute-shadows-method hazard). Keep `n_units` as a property
   returning `len(self.unit_names)`.
2. Every internal `self.line_names` / `self.line_names_ex` / `self.line_name_pipe`
   reference → unit form (~30 sites: 283, 290–292, 443, 512, 518, 877, 1166,
   1896, 2586, 2606, 2892, 2904, 2956–2958, 3117, 3298, 3729, 3737, 3898, 3909,
   3933, 4002, 4047, 4104, 4133, 4261, 4266, 4676, 4775–4784, `sample`,
   `bodoff`, …). Rename loop vars `for line in self.line_names` → `for unit in
   self.unit_names` as you go (they feed `f'p_{unit}'`-style column interpolation
   — the *value* is the unit name, so emitted column names are unaffected).

### B. Output-frame label `'line'` → `'unit'` (breaking — handle as a coupled set)

The literal string `'line'` is used both to **name** an index/column and to
**reference** it later, so rename every producer and consumer together or frames
break. Sites: `portfolio.py` 1892 `set_index(['line','log'])`, 1919
`set_index('line')`, 3590, 3735 `index.name='line'`, 4261, 4369, 4488; plus
`pentagon.py` 272/280 and `distributions.py:8210` (single-`Aggregate` price
frame). Grep `"'line'"` / `'"line"'` to confirm none are missed. **Breaking:**
any downstream `groupby('line')` / `.loc['line']` / `df.index.name == 'line'` in
user code (and in `tests/test_price_pentagon.py:90`) must move to `'unit'`.

### C. Keyword arguments `line=` / `lines=` → `unit=` / `units=` (breaking)

- `Portfolio.pentagon_at(..., line='total')` (`portfolio.py:3751`) → `unit='total'`;
  internal `_line_capital_at(..., lines=None)` (3954) → `units=None`.
- `Bounds.__init__(..., line='total')` (`bounds.py:203`) → `unit='total'`
  (and its ~33 internal `line` refs; classify against the keep-table first).
- `Pentagon.as_frame(line='total')` (`pentagon.py:269`) and
  `from_row(..., line='total')` (311) → `unit=`.
- `BivariateAggregate` (`bivariate.py`, formerly `multivariate.py` — renamed at
  MV-7/a80) carries a **full LOB-sense surface** that must rename as a coupled set
  (none of it is keep-table — no matplotlib / source-line / `linear` coincidences):
  - ctor `BivariateAggregate.__init__(name, lines=None, …)` (`bivariate.py:564`)
    → `units=None`; same for the `_init_netceded(self, name, lines, …)` param
    (630, 638, 665, 667) → `units`.
  - real storage `self.line_names` (set at 602 and 658) → `self.unit_names`;
    `self._line_specs` (601, 607, 737) → `self._unit_specs`; `self.lines` (list
    of `Aggregate`, set 615; read 626, 739, 963, 986) → `self.units`.
  - ~20 internal `self.line_names` reads (1146, 1189–1190, 1214, 1276, 1310,
    1339, 1347, 1392, 1436, 1452, 1468, 1484, 1544, 1562, 1564) → `unit_names`;
    the `lines=…` text in the two `__repr__`/log strings (1562–1564) is the LOB
    sense → `units=…`.
  - docstring attrs `lines` / `line_names` (524, 546, 549) and the prose
    "for that line" (518, 705) / "three or more correlated lines" (4) → unit form
    (Stage G prose, but in this file so do it here).
  - **spec key `'lines'` → `'units'` (coupled producer/consumer set):** produced
    in `parser.py` at 551, 564, 586, 642 (`bv_out_copula`, `_nofreq`,
    `_bv_out_viewpair`, `_clash_spec`); consumed by `decl_writer._render_bvagg`
    at 563, 579, 582 and by the ctor via `BivariateAggregate(**spec)`
    (`underwriter.py:858`). Rename all together or the bivariate round-trip
    breaks. `Aggregate.occ_bivariate` (`distributions.py:3043`) builds via
    `nc_agg=`/`nc_views=` and passes **no** `lines=` kwarg — unaffected.

`allocation_bounds(..., units=None)` (`portfolio.py:652`) is **already** `units`
— leave it; it confirms the target spelling.

### D. `line_renamer` → `unit_renamer`

Rename the property (`portfolio.py:4603`), the cache attr `_line_renamer`, the
local `rename(ln)` comment "numbered lines", and the ~6 call sites. Behaviour
unchanged.

### E. Other modules (internal LOB refs only)

Sweep `spectral.py`, `pedagogy.py`, `decl_writer.py`, `iman_conover.py`,
`results.py`, `pentagon.py`, `ft.py` for the **LOB** sense (cross-check each
against the keep-table — `pedagogy.py`'s 112 hits are mostly plot `linewidth` /
`axline`, so the real count is small). `unit_density`/`add_exa` internals already
covered in A.

### F. Tests (must end green)

Update LOB references so the suite passes:
- `.line_names` → `.unit_names` (~30 refs): `test_unit_density.py` (8),
  `test_numerics3_distortion.py` (6), `test_numerics2_objective.py` (5),
  `test_bivariate.py` (`.line_names`/`kind=='bvagg'` LOB refs),
  `test_portfolio_peg_regression.py:127,130`.
- `'line'` index assertions → `'unit'`: `test_price_pentagon.py:90`.
- **Leave** the text-line parametrize machinery (`LINES`, `_line_id`,
  `parametrize("line", …)` in `test_decl_parser.py`, `test_splice_suite.py`) and
  the `parser_errors` "line"/"column"/"source_line" assertions
  (`test_parser_errors.py:224,262`) — those are source-text lines (keep). Optional
  tidy: rename `LINES`→`PROGRAMS` to cut DoD noise, but it's not required.
- **Regression baselines:** `tests/baseline/`, `test_portfolio_peg_regression.py`
  and any snapshot that captures the `'line'` index name or `line_names` ordering
  must be **regenerated** after the rename (note in the PR which snapshots moved).

### G. Docs sweep

32 `.rst` files, ~105 `line`-word hits — but filter through the keep-table
(matplotlib, "one-line", source lines survive). Rename the LOB prose ("line of
business" → "unit", "by line" → "by unit") and any `:attr:`line_names``/`line=`
cross-references to the new names. Per CLAUDE.md: keep `.rst` edits in lockstep
with the code, grep stale `:attr:` / `:meth:` targets, **do not** run the full
doc build in the loop — note "docs pending rebuild" for the author. Check
`.qmd`/notebook examples too if any reference `line_names`.

## KEEP list — the explainable residue (the DoD allowlist)

After the change, the only `rg line -g *.py` hits should be:
`parser_errors.py` + parser/underwriter source-line machinery; matplotlib
(`linewidth`, `axhline`, `axvline`, `linestyle`, `linefmt`, `Line2D`);
substrings (`linear`, `newline`, `inline`, `baseline`, `multiline`, `headline`,
`underline`); the actuarial **"loss on line" / `lol`**; prose "one/five-line";
and the test text-line parametrize helpers. Each is explainable in one phrase.

## Blast radius summary

| Area | Files | Risk |
|---|---|---|
| Core storage + accessors + delete pass-throughs (A) | `portfolio.py` | **high** — hottest file, ~30 sites, attribute-shadow hazard |
| `'line'` index label (B) | `portfolio.py`, `pentagon.py`, `distributions.py` | **med-high** — breaking output; producer/consumer must move together |
| `line=`/`lines=` kwargs + bivariate surface (C) | `portfolio.py`, `bounds.py`, `pentagon.py`, `bivariate.py`, `parser.py`, `decl_writer.py` | med — breaking for keyword callers; bivariate `'lines'` spec key is a coupled producer/consumer set |
| `line_renamer` (D) | `portfolio.py` | low |
| other modules (E) | `spectral.py`, `pedagogy.py`, `decl_writer.py`, `iman_conover.py`, … | low (after keep-table filter) |
| tests (F) | ~6 LOB test files + baselines | med — baselines may need regen |
| docs (G) | ~a dozen `.rst` after filter | med — volume, pending rebuild |

DecL grammar (`decl.lark`) needs **no** change — it already uses "unit" in
comments and never had a `line` token; no `build()`-language surface change. The
**one** spec-key rename is internal: the `BivariateAggregate` spec carries a
`'lines'` key (set by the parser, read by the ctor and `decl_writer`) → `'units'`
(see C). It is not a DecL keyword and persisted bivariate specs are pre-1.0, so
this is a safe internal rename, but producer and consumer must move together.

## Sequencing

1. A — core storage rename + delete the three pass-through properties.
2. B — `'line'` label set (producers + consumers together).
3. C — kwargs.
4. D — `line_renamer`.
5. E — other modules.
6. F — tests + regenerate baselines.
7. `uv run pytest` → green.
8. G — docs sweep (lockstep; no full build).
9. Version bump (`pyproject.toml`) + `CHANGELOG.md` breaking section
   (removed `line_names`/`line_names_ex`/`line_name_pipe`/`line_renamer`;
   `'line'` index → `'unit'`; `line=`/`lines=` → `unit=`/`units=`); mark the item
   done in `dev/TODO.md`.

## Verification & Definition of Done

- `uv run pytest` — full suite green (the real gate; tests are red until F).
- Smoke: build a 2-unit portfolio, check `p.unit_names`, `p.unit_names_ex`,
  `p.n_units`, `p.unit_renamer`, that a priced frame's `index.name == 'unit'`,
  and that `p.unit_density(p.unit_names[0])` still works. Confirm `p.line_names`
  now raises `AttributeError` (clean break).
- `rg "line_names|line_name_pipe|line_renamer|name='line'|name=\"line\"|\bline\s*=\s*'total'" src tests` → **zero** hits.
- **DoD:** `rg line -g '*.py'` returns only **keep-table** rows — few, and each
  explainable in a phrase (source lines, matplotlib, `linear`/`newline`, loss on
  line, prose). Capture that residual list in the PR summary as evidence.

## Resolved decisions (author, 2026-06-19)

1. **Back-compat alias for `line_names` — NO. Clean break.** `line_names` and the
   other `line_*` names are removed outright; accessing them raises
   `AttributeError`. No reverse pass-through.
2. **`'line'` index-label rename → `'unit'` — approved**, output-breaking and
   intended. Any saved analysis keyed on the `'line'` index moves to `'unit'`.
3. **Cosmetic tidy — `bs_describe` local `line`→`text`: include** it in Stage E.
   `LINES`→`PROGRAMS` in the test parametrizers: **leave** (optional, not done —
   keeps the diff focused; the `LINES` symbol is an explainable keep-table hit).

> Separately flagged: the author wants a standalone **"investigate `bs_describe`"**
> item (Track B in `dev/TODO.md`) — independent of this rename; the `line`→`text`
> tidy here does not pre-empt that review.
