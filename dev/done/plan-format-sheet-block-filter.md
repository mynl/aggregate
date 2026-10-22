# Plan: format sheets send greater_tables only the columns a block has

`[Format-Sheet-Block-Filter]`, one step. Drafted 2026-10-22 at 1.7.1 (`3f342d3`) during the
greater_tables 8.1.0 rollout. **Status: executed 2026-10-22 as 1.10.0.** See Rulings and the
Execution log at the foot.

This is the library half of a pair. The other half is
`V:\dev\aggregate-api\dev\plan-greater-tables-8.md`, which moves aggregate-api and the
aggregate-relativity plugin to greater-tables 8.1.0. This half lands first and is released to PyPI
before that one executes.

## Goal

Every exhibit block builds without a greater_tables `formatters key ... matches no column` warning
under greater-tables 8.1.0, while a genuine mistake in a block's own `formatters` still warns.
aggregate requires greater_tables 8.1 or later, so every user moves to it at their next upgrade;
the development environment moves first, the suite is green there, and the release ships both.

## Background: what changed in greater_tables

- greater_tables 7.4.0 (2026-10-21) added `_check_formatter_keys` in `engine/build.py`. Every key
  of `TableSpec.formatters` that is not a label of the table (a stub name, a data column, or the
  leaf of a tuple column) raises a `UserWarning` with the text
  `formatters key 'CV' matches no column and is ignored.` The warning is attributed to the first
  frame outside greater_tables, here `exhibits/_core.py` in `build_exhibit`.
- PyPI went from greater-tables 7.0.0 straight to 8.1.0 on 2026-10-22, so 8.1.0 is the first
  release that brings the check to aggregate's users.
- The check stays as it is. Limiting it to the key `'all'` was considered and rejected on
  2026-10-22, because every other misspelled key would then do nothing and report nothing. The rule
  adopted instead: **a shared declaration applies where it matches and stays quiet; an entry
  written for one table must match or warn.** The filtering belongs where a shared declaration
  meets one table, which in aggregate is the format sheet.

## Current behavior

- **The sheets.** `src/aggregate/formats/formats-raw.yaml` and `formats-insurer.yaml`, overridable
  from `~/.aggregate` and the working directory (`sheet_paths`). Each has exact `columns:` entries
  (about 80 in raw and 15 in insurer: `CV`, `Skew`, `limit`, `attach`, `loss`, `pr_attach`, `L`,
  `M`, `P`, `Q`, `a`, the capital columns and the rest), regex `patterns:`, and scoped `exhibits:`
  sections.
- **`FormatSheet.block(exhibit=None, rename=None, labels=())`**, `exhibits/_formats.py` line 180.
  Patterns are expanded against `labels`, so they yield only labels the block carries. The exact
  entries (global, with the exhibit's scoped ones laid over them, then renamed through `rename`)
  are returned whole, whatever the block carries. The Returns section (lines 204 to 206) contains
  the assumption: "Labels the frame does not carry are harmless in both: greater_tables looks
  formats up by label and resolves a selector list against the columns it actually has." For
  `formatters` that held through greater_tables 7.3 and has been false since 7.4.0. For tag
  selectors it still holds: greater_tables' `resolve_cols` intersects a list with the table's
  labels and does not warn.
- **`build_exhibit`**, `exhibits/_core.py` line 577, the merge at lines 649 to 653. Per block it
  calls `read_sheet(df)`, which is `sheet.block(name, rename, _block_labels(df))` (`_sheet_reader`,
  line 529), lays the block's own `formatters` over the sheet's, and calls `gt.build` once.
  `_block_labels` (line 516) returns the data column labels (a tuple column contributes its leaf)
  and then the index names.
- **The result.** Every block of every exhibit warns once for each exact sheet entry it lacks.
  Measured 2026-10-22 through aggregate-api, with this worktree's aggregate 1.7.1 and greater-tables
  8.1.0: one `pricing.allocate` request
  (`tests/test_pricing_exhibits.py::test_ccoc_allocates_on_an_unbounded_book` there) carried 538
  such warnings in its response. On greater-tables 7.0.0 it carries none. Released aggregate 1.6.0
  has the same code, so its users see the same once they install greater-tables 8.1.0.
- **The development lock** (`uv.lock`) holds greater-tables 6.0.0, so the suite has never run
  against the check. The main checkout `V:\dev\aggregate` holds the same lock; it follows when
  `dev` merges and is not edited separately.

## The change: [Format-Sheet-Block-Filter]

1. **Filter exact entries to the block.** In `FormatSheet.block`, when labels are given, keep an
   exact entry, and its exact tag, only if its label after `rename` is among `labels`. Patterns
   already behave this way. The filter runs after the rename, because `labels` arrive relabeled.
2. **No labels still means the whole sheet** (ruling 2). Change the default to `labels=None` and
   filter only when `labels is not None`; patterns expand against no labels when none are given,
   as now.
   `sheet.block()` with no labels keeps returning every exact entry, which
   `tests/test_exhibit_formats.py` relies on (`sheet.block()[0]['P']`, the `ratio_cols` selector
   test). An explicit empty list filters to nothing, which is right for a block with no columns.
   `FormatSheet` sits in the provisional exhibits API (PEP 411), so the default change needs no
   deprecation; it is named here for review.
3. **Rewrite the Returns sentence** to state the new contract: an exact entry the block does not
   carry is left out, because greater_tables 7.4 and later warns on a `formatters` key that matches
   no column; tags are filtered the same way for consistency, though greater_tables tolerates a
   selector label it lacks.
4. **Leave a block's own `formatters` unfiltered.** In `build_exhibit` the block's kwargs are laid
   over the sheet. They come from a frame builder or a plugin (aggregate-relativity's
   `FORMATTERS`) and are written for that one frame, so an unmatched key there is a real mistake
   and keeps warning.
5. **Move the development lock.** `uv lock --upgrade-package greater-tables` (6.0.0 to 8.1.0),
   then `uv sync --all-extras --inexact`. **Never a plain `uv sync --all-extras`:** it deletes the
   optional `aggregate-ntt` engine and has also taken `gmpy2`, `rich` and `csv-grid` (`CLAUDE.md`,
   the `uv sync` note). Check with
   `uv run python -c "import greater_tables; print(greater_tables.__version__)"`, which must print
   8.1.0.
6. **Raise the floor** to `greater_tables>=8.1.0` in `pyproject.toml` (ruling 1). greater_tables
   8.1 has the floors 6 had (Python 3.12, numpy 2.0, pandas 2.2, plus pydantic and pyyaml), so no
   other floor moves. Update the two comments that describe the dependency: the one above
   `dependencies`, which is written about greater_tables 6, and the co-development note after the
   extras, whose path `C:/S/AI/greatest-tables` is from the old machine and is now
   `V:/dev/greatest-tables`.

No new public names. The one signature change is the `labels` default in step 2.

## Tests

- **Unit, `tests/test_exhibit_formats.py`:**
  - an exact global entry absent from `labels` is not returned, and one present is;
  - a scoped exact entry (the `P` collision on `tail`) is filtered the same way;
  - with a `rename`, the filter applies to the displayed label: a key renamed from `CV` is kept
    when `labels` holds the new name and dropped when it holds only `CV`;
  - the existing tests that call `sheet.block()` without labels pass unchanged.
- **Integration, `tests/test_exhibits.py`:** build every exhibit registered for the fixture
  objects the module already uses, under `warnings.catch_warnings(record=True)` with
  `warnings.simplefilter('always')`, and assert no recorded warning whose message contains
  `matches no column`. No version guard is needed, since the floor is 8.1 and the check arrived in
  7.4. A second case builds one block whose own `formatters` names a column it lacks and asserts
  the warning does fire, so the check is shown to be live.
- **The suite:** `uv run pytest` (it runs `-n auto`) on greater-tables 8.1.0, green. Then read the
  warnings summary for any greater_tables `UserWarning` left over; each is fixed or explained in
  the execution log.
- **Output changes.** greater_tables 7.1.0 to 8.1.0 change some rendered output. Check any test
  diff against the greater_tables CHANGELOG entries marked **Output changes** before updating an
  expectation. The likely ones: 7.4.0, where a float column that printed a nonzero value as zero
  now takes the decimals its smallest value needs, or SI past five; 7.0.0, where `$` math
  detection is strict, so `$10 to $20` is two prices; 8.1.0, where a markdown string input sets
  `*em*`, `**strong**` and `` `code` ``; and 7.1.0 to 7.6.0 for TikZ geometry, if a test compares
  TeX.

## Acceptance

- Under greater-tables 8.1.0, building every exhibit on the test fixtures records zero
  `matches no column` warnings, and a misspelled key in a block's own `formatters` still warns.
- `uv run pytest` is green, and `uv.lock` holds greater-tables 8.1.0.
- The existing format sheet tests pass unchanged.

## Release

- A minor version, assigned at execution: the fix alone would be a patch, but the new floor moves
  every user to greater_tables 8, whose tables differ in places. CHANGELOG:
  - Changed: requires greater_tables 8.1 or later. Tables rendered through it can differ from
    greater_tables 6 and 7; its CHANGELOG marks each such release **Output changes**.
  - Fixed: exhibits no longer raise one greater_tables warning per format sheet column a block
    lacks, because the sheet sends only the columns the block carries. Seen with greater-tables
    7.4 and later, where one aggregate-api pricing request carried 538.
- Release to PyPI per `V:\worktrees\RELEASE.md` (`release-aggregate.ps1`). That is Steve's step.
- The consumer half, `V:\dev\aggregate-api\dev\plan-greater-tables-8.md`, then executes.

## Rulings (Steve, 2026-10-22)

1. **The greater_tables floor is `>=8.1.0`**, here and in every consumer: "move everyone onto
   >=8.1. there was a lot of good work went into 8. I think it is much better." The draft had
   proposed keeping `>=6`, since the fix alone does not need 8.
2. **The `labels` default is `None`**, meaning the whole sheet, with any list filtering. The
   alternative, keeping `()` and filtering only a non-empty list, would leave a block with no
   columns receiving the whole sheet.

## Execution log

**1.10.0, 2026-10-22**, on `dev` after 1.9.0 (`298f095`). Steps 1 to 6 landed as written. Review found every cited path, line number and snippet matching the code at 1.9.0, and no name to vet beyond the `labels` default.

Divergences and findings:

1. **A third copy of the floor.** `_import_greater_tables` in `exhibits/_core.py` gave the floor as `greater_tables>=6` in its `ImportError` text. Changed to `>=8.1`. The plan named only the two `pyproject.toml` comments.
2. **The lock diff has one line beyond greater-tables.** pexpect's dependency on ptyprocess loses its `sys_platform != 'emscripten' and sys_platform != 'win32'` marker, a re-resolution by uv 0.12.7. No effect: pexpect itself is still required under that marker (`uv.lock`, the ipython entry), so ptyprocess installs nowhere new.
3. **`labels` is listed once at the top of `block`.** Patterns and the new filter both walk it, so a generator would have reached the filter empty. Covered by `test_labels_may_be_a_generator`, and an explicit empty block by `test_an_empty_block_gets_nothing`; both are additions to the plan's test list.
4. **The live warning case injects its key by monkeypatching `_core.exhibit_frames`.** No registered block sets its own `formatters` with an absent column, so the misspelled key is laid onto the `summary` block's kwargs for the one test.
5. **Exhibit snapshots re-captured: 8 of 166 cases, 938 cells.** Every moved cell is an inferred format on a column with no sheet reading, and every one is greater_tables 7.4.0's Fixed entry (a float column takes the decimals its smallest value needs, or goes to SI past five). The `approximation` exhibit on `Aggregate`, `ReinsAggregate` and `ReinsPortfolio` (both perspectives) gains a decimal on several columns. `reins_summary_df` on `PeelMarginal` (both perspectives) moves `Change CV` from `dec 5`, which set `1.55e-06` as `0.00000`, to SI 3, which sets `0.2558` as `255.757m`. `Change EX` was already SI. The filter alone moved no IR: nothing else in the snapshot file changed.
6. **Finding for the author: `Change CV` in SI.** It reads like `Change EX` now, but `255.757m` for a change of 0.256 in a CV is an odd reading for an actuarial table. Both labels are on `PENDING_VOCABULARY` in `tests/test_exhibits.py`, so the fix is a sheet entry (the `residual` style, say) when that punch list is ruled. Not done here.
7. **`.doc-venv` is not touched.** It is absent from this worktree; `doc-test-uv.ps1` syncs it from the lock when it next runs.

Tests: tier 1 on `tests/test_exhibit_formats.py` and `tests/test_exhibits.py`, 328 passed. Tier 3 gate `UV_PROJECT_ENVIRONMENT=.venv uv run pytest -m 'slow or not slow'` on greater-tables 8.1.0, 6483 passed; its warnings summary holds no greater_tables warning and no `matches no column`.
