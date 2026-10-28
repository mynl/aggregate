# Plan: the yelt keyword, and ELT reporting as two narrative strings and one table

`[YELT-Keyword]` and `[ELT-Reporting-Punchup]`. Drafted 2026-10-28 against
aggregate 1.18.2. **Status: ruled 2026-10-28 (rulings 28 to 34), ready
for review and execution.** Two minor bumps. `[YELT-Keyword]` can land at
any time. `[ELT-Reporting-Punchup]` lands after `[ELT-Shared-Event-Rollup]`
(`dev/plan-elt-shared-rollup.md`), whose rolled columns `elt_df` exposes.

## Goal

1. **One word for annual tables.** The DecL clause `ylt` becomes `yelt`, in
   the grammar and everywhere else: code, messages, form names, tests,
   documentation and corpora. No alias remains.
2. **Reporting.** An aggregate declared with `elt` or `yelt` reports through
   three public members instead of today's three frames:
   - `elt_description`: a few short aligned lines (what the table is, the
     grid, a one-line verdict of the checks);
   - `elt_explanation`: the long prose (the model in words, each source's
     provenance, every check with its value, tolerance and status);
   - `elt_df`: the combined table the model actually uses, one row per event
     (or year), with rolled-up parameters for shared events.

   `loss_table_summary_df`, `loss_table_stats_df` and
   `loss_table_validation_df` are removed. No other public name is added.

## Rulings

Author, 2026-10-28 (numbering continues from `dev/plan-elt-shared-rollup.md`):

28. **[Loss-Table-Frames-Retired]** A `_df` member of `Aggregate` is added
    only with care. `loss_table_stats_df` is dropped: its `estimate` against
    `table` duplicates `validation_df`. `loss_table_summary_df` and
    `loss_table_validation_df` stop being frames. This supersedes ruling 11
    (`[Loss-Table-Report-Names]`) of `dev/done/plan-elt.md`.
29. **[Loss-Table-Narrative]** The summary becomes the short narrative and
    the validation the long one, a `*_description` / `*_explanation` pair, so
    both reach aLL's Narrative pane with no edit there.
30. **[Loss-Table-Df]** One new frame: the combined table in use. For an ELT:
    event, rate, and the rolled-up mean, maximum and SDs.
31. **[ELT-Report-Names]** The members are `elt_description`,
    `elt_explanation` and `elt_df`, and they serve `yelt` aggregates too:
    `elt` is the family's name. The stem `elt` is the section name in aLL's
    Narrative pane.
32. **[ELT-Reporting-Version]** The three frames arrived in 1.14.0 and were
    never published (the last PyPI release is 1.11.0), so their removal is a
    regular minor bump with a one-sentence CHANGELOG note, not a
    deprecation. The same holds for the keyword rename.
33. **[ELT-Burke-Total]** Pended (ruling 25 of the roll-up plan). This plan
    leaves the Burke branch and `SHARED_EVENT_RULE` in place; its `moment
    integration` check moves into the explanation like every other check
    and appears only under that rule.
34. **[YELT-Keyword]** There is one way to declare an annual table, and it
    is `yelt`. A row is one simulated year, taken as one outcome, in three
    forms: `YELT1` losses only (the inline vector `yelt [0 0 100 300]`, or a
    file with only a loss column, years numbered implicitly), `YELT2`
    `[year loss]` and `YELT3` `[year loss prob]`, with equal probabilities
    when `prob` is omitted. This renames ruling 4's `[YLT-Annual-Outcomes]` surface;
    the semantics (annual outcomes, aggregate terms only, complete year sets)
    are unchanged. Bracketed labels of record in `dev/done/` plans and the
    source comments that cite them keep their spelling (`[YLT-Annual-Terms]`
    and the like), so a grep from a comment still finds its plan.

## Current behavior, 1.18.2

- **The `ylt` clause.** Terminal `YLT.2` (`src/aggregate/decl.lark:983`),
  rules `ylt_sources`, `ylt_source`, `ylt_data` (:104 to 117), body
  alternative `agg_body_ylt` (:88), and `ylt` in the ID terminal's reserved
  list (:1047). Forms `YLT1` (losses only: the inline vector
  `ylt [0 0 100 300]` or a file with only a loss column), `YLT2`
  (`[year loss]`) and `YLT3` (`[year loss prob]`) (`_loss_table.py:396`,
  :564, :592, :1106).
- **Where `ylt` appears** (`rg -i "\bylt\b|YLT[0-9]|ylt_"`, 2026-10-28):
  `tests/test_ylt.py` 56, `src/aggregate/_loss_table.py` 41,
  `src/aggregate/parser.py` 13, `tests/test_loss_table_files.py` 11,
  `src/aggregate/decl.lark` 9, `docs/4_agg_language_reference/ref_include.rst`
  9 (the generated grammar reference), `src/aggregate/agg/decl-testers.agg`
  7, `docs/4_dec_Language_Reference.rst` 7,
  `tests/test_loss_table_reports.py` 5, `src/aggregate/decl_writer.py` 5,
  `src/aggregate/underwriter.py` 2, `src/aggregate/_aggregate.py` 2, one
  each in `tests/test_elt_uncertainty.py`, `tests/test_elt.py`,
  `src/aggregate/parser_errors.py`, `src/aggregate/decl_words/__init__.py`
  and `docs/2_aggregate_overview/info-strings.rst`. The spec snapshot
  `tests/data/expected_specs.json` has none.
- **Downstream, aLL** (`V:/dev/aggregate-api`): `web/src/decl-words.json`
  (2), `src/aggregate_api/loss_table_files.py` (4), `tests/test_loss_table_files.py`
  (2).
- **The frames.** `Aggregate.loss_table_summary_df`, `loss_table_stats_df`
  and `loss_table_validation_df` (`src/aggregate/_aggregate.py:3969` to
  4004) delegate to `summary_frame` (`src/aggregate/_loss_table.py:1291`),
  `stats_frame` (:1384) and `validation_frame` (:1444).
  - Summary, indexed `(part, item)`: part `table` (kind, form, units, unit
    labels, sources, rows, events, zero-rate events, event rate, mean annual
    loss, severity atoms, largest event exposure, secondary uncertainty row
    laws, largest event support, model-moment method, eltq version; for an
    annual table years, zero-probability years, zero-loss years, probability
    of zero, year probabilities, alignment, outcomes, mean); part `grid`
    (`bs`, `log2`, severity edge, tail rate above edge, `n/a` before
    `update`); one part per source and unit (origin, form, effective form,
    `no2u`, rows, file facts, ignored columns).
  - Stats, indexed `(component, measure)`: `table`, `model`, `estimate`,
    `error` for severity and annual mean, CV and skewness. For ELT6, `table`
    and `model` are both eltq's model moments, and `estimate` against them
    is what `validation_df` already reports.
  - Validation, one row per check (`check`, `value`, `tolerance`, `status`,
    `detail`): form projection, one row per event and unit, event rate
    agreement, zero-rate events, row classification, moment integration
    (Burke rule only), year completeness, and the grid checks (quantized
    mass, secondary mean error, tail rate and cumulant shares above the
    edge), the grid rows `not computed` before `update`.
- **`info`** has one `loss table` row (`_aggregate.py:3937`, from
  `_loss_table.describe`). It stays.
- **The narrative convention.** A short `*_description` and a long
  `*_explanation`; where one half exists the other must too (the pairs rule,
  `src/aggregate/constants.py:138`, checked by `dev/regen_features.py` and
  `tests/test_fcc_surface.py`). Short lines are formatted with
  `constants.info_row` (`tail_description`, `_aggregate.py:1274`, is the
  model). The pair is optional on first-class classes and reached with
  `getattr`.
- **aLL's Narrative pane.** `narrative_for`
  (`V:/dev/aggregate-api/src/aggregate_api/capability.py:735`) collects every
  public attribute ending `_description` or `_explanation`, groups by stem,
  drops empty strings, and appends stems not in `NARRATIVE_ORDER`
  alphabetically. A new pair appears with no edit there, provided it is the
  empty string on an aggregate that is not a loss table. aLL does not use
  the three frames.
- **Users of the frames.** `tests/test_loss_table_reports.py` (15 uses),
  `tests/test_elt_uncertainty.py` (12), `docs/4_dec_Language_Reference.rst`
  (2), `dev/FEATURES.csv`, `CHANGELOG.md` history.
- **Per-event data at lowering.** ELT6 keeps its rows on `EltModel`
  (`_loss_table.py:114`; after the roll-up, `rolled_columns` holds one row
  per event). ELT3, ELT4 and `no2u` keep only the coalesced severity atoms;
  the per-event `event_rate` and `event_loss` are locals of `_lower_elt`
  (:874 to 885). An annual table has its year probabilities `p` and annual
  `totals` at lowering (:1130). `_loss_table_meta` lives in the aggregate's
  spec (`_aggregate.py:2826`).

## Design

### [YELT-Keyword] the rename

Mechanical, with no behavior change:

- **Grammar.** `YLT.2` becomes `YELT.2: /yelt(?![a-zA-Z0-9._:~\-])/`; the
  rules become `yelt_sources`, `yelt_source`, `yelt_data` and
  `agg_body_yelt`; the reserved list swaps `ylt` for `yelt`, which frees
  `ylt` as an ordinary identifier.
- **Code.** Transformer and writer methods, `_loss_table.py` functions and
  constants (`_lower_ylt` to `_lower_yelt` and the like), `meta['table']`
  value `'yelt'`, form tags `YELT1`, `YELT2`, `YELT3`, every message and the
  `info` text.
- **Tests.** `tests/test_ylt.py` becomes `tests/test_yelt.py`; the other
  four test files change their clauses and expected strings.
- **Documentation and corpus.** `docs/4_dec_Language_Reference.rst`,
  `docs/2_aggregate_overview/info-strings.rst`, the regenerated grammar
  reference, `src/aggregate/decl_words`, and the `ylt` lines of
  `src/aggregate/agg/decl-testers.agg` (lines 1468 to 1489).
- **File readers.** Column aliases (`year_id`, `loss`, `prob` and their
  variants) are unchanged.
- **aLL.** After the bump, the author updates `aggregate-api`:
  `web/src/decl-words.json`, `src/aggregate_api/loss_table_files.py` and its
  test. The CHANGELOG entry states this downstream action.

Check before the rename that no shipped corpus, library or test uses `yelt`
as an identifier (it becomes reserved).

### [Loss-Table-Description] the short narrative

Lines formatted with `info_row`, the empty string for an aggregate that is
not a loss table. For an ELT, in order:

- `loss table`: form, events and rows, units (with labels where given);
- `event rate`: $\Lambda$ and the mean annual loss;
- `secondary uncertainty` (ELT6 only): the row laws (beta, two-point, atom
  counts) and, when shared, `{k:,d} shared events rolled up`;
- `grid`: `bs`, `log2` and the severity edge, `n/a` before `update`;
- `checks`: counts of pass, warning and not computed, naming each warning.

For a YELT: `loss table` (form, years, units, the probability convention),
`annual outcomes` (outcomes, probability of zero, mean annual loss), `grid`
and `checks`. The text of the existing summary rows is the source; nothing
is computed that the summary does not compute today.

### [Loss-Table-Explanation] the long narrative

Prose in three paragraphs, the empty string when not a loss table:

1. **The model**: compound Poisson with rate $\Lambda$ and severity the
   rate-weighted mixture of event totals; for ELT6 the row laws and the
   roll-up of shared events (with counts and the number of rolled rows with
   inadmissible variance); for ELT3 and ELT4 the coalescing of equal event
   losses; for a YELT the annual outcomes and their probabilities.
2. **Sources**: per source and unit, the origin (inline, or the file path,
   size and SHA-256 prefix), form, effective form, `no2u`, rows and ignored
   columns.
3. **Checks**: one sentence per check with its value, tolerance, status and
   detail, warnings first; grid checks before `update` show
   `not computed: needs update`.

### [Loss-Table-Df] the table in use

`Aggregate.elt_df`, built on access, an empty `DataFrame` when not a loss
table. Rows with zero rate (or zero probability) are excluded, since the
model does not use them; the description counts them.

| Form | Index | Columns |
|---|---|---|
| ELT6 | `event` (the ID as text) | `rate`, `mean`, `max`, `sdi`, `sdc`, `units` |
| ELT4, or ELT6 with `no2u` | `event` | `rate`, `mean`, `max`, `units` |
| ELT3 | `event` | `rate`, `mean`, `units` |
| YELT | `year` | `prob`, `loss` |

`units` is the number of input rows behind the row: 1 for a one-row event,
more for a rolled event. For ELT6 the five numeric columns are exactly what
eltq receives (`EltModel.columns()`, rolled under the roll-up); for the
deterministic forms `mean` and `max` are event totals (sums over units); for
a YELT `loss` is the annual total over units. Rows are in event order of
first appearance (the model's order).

No `law` column: eltq exposes row classification only as counts, and the
parent plan delegates classification to eltq. A per-row law is a later
upstream request if wanted.

Data: ELT6 reads `EltModel`. For ELT3, ELT4 and `no2u`, `_lower_elt` keeps
its per-event arrays (ids, rate, event total, summed maximum, row count) as
read-only arrays in the meta under `event_table`; a YELT keeps its year ids,
probabilities and totals under the same key. The memory is the order of what
`EltModel` already holds for ELT6. The executor checks that no path
deep-copies the meta per call (it rides in the spec).

### [Loss-Table-Private-Helpers] one source for strings and tests

`summary_frame` and `validation_frame` stay in `_loss_table.py` as private
`_summary_rows` and `_check_rows`, the single source of both strings; tests
that inspect a check's value or status call `_check_rows`. `stats_frame`,
its constants and the three `Aggregate` properties are deleted.

## Steps

### [YELT-Keyword], minor bump

1. `[YELT-Keyword]` as above.
2. Grammar reference regeneration and `dev/FEATURES.csv` via
   `dev/regen_features.py`.
3. Release: `__version__`, `CHANGELOG.md` (the rename, and the aLL update as
   the downstream action), `dev/TODO.md` (the `[DecL-Loss-Tables]` entry's
   wording); move this plan to `dev/done/`.

### [ELT-Reporting-Punchup], minor bump

1. `[Loss-Table-Private-Helpers]`, then `[Loss-Table-Description]`,
   `[Loss-Table-Explanation]` and `[Loss-Table-Df]`.
2. Tests (below); `tests/test_loss_table_reports.py` and
   `tests/test_elt_uncertainty.py` move from the frames to `_check_rows`,
   the strings and `elt_df`.
3. Documentation: the two mentions in `docs/4_dec_Language_Reference.rst`
   become the narrative pair and `elt_df`.
4. Release: `__version__`, `CHANGELOG.md` (the removal stated as the second
   short paragraph), `dev/TODO.md`, `dev/FEATURES.csv` via
   `dev/regen_features.py`.

## Files

| Area | Files |
|---|---|
| Grammar and writer | `src/aggregate/decl.lark`, `src/aggregate/parser.py`, `src/aggregate/parser_errors.py`, `src/aggregate/decl_writer.py`, `src/aggregate/decl_words/__init__.py` |
| Loss tables | `src/aggregate/_loss_table.py` (the rename; `_summary_rows`, `_check_rows`, the two renderers, `event_table` in `_lower_elt` and the YELT lowering; `stats_frame` deleted) |
| Aggregate and underwriter | `src/aggregate/_aggregate.py` (the rename; three properties out, `elt_description`, `elt_explanation`, `elt_df` in), `src/aggregate/underwriter.py` |
| Tests | `tests/test_ylt.py` renamed `tests/test_yelt.py`, `tests/test_loss_table_files.py`, `tests/test_loss_table_reports.py`, `tests/test_elt_uncertainty.py`, `tests/test_elt.py` |
| Documentation and corpus | `docs/4_dec_Language_Reference.rst`, `docs/2_aggregate_overview/info-strings.rst`, `docs/4_agg_language_reference/ref_include.rst` (regenerated), `src/aggregate/agg/decl-testers.agg` |
| Release | `src/aggregate/__init__.py`, `CHANGELOG.md`, `dev/TODO.md`, `dev/FEATURES.csv` |
| Downstream | `V:/dev/aggregate-api` (three files, after the first bump, by the author) |

## Names checked

On 2026-10-28, `rg` over `src/aggregate` found no use of `elt_description`,
`elt_explanation`, `elt_df` or `event_table`. To check at execution: `yelt`
as an identifier anywhere in `src/aggregate/agg`, `tests` and `docs` (it
becomes reserved), and `Aggregate` and its parents for an attribute the new
members could shadow.

## Acceptance and tests

`[YELT-Keyword]`:

- Every `yelt` form (`YELT1` vector and file, `YELT2`, `YELT3`, units,
  aggregate terms) builds with the moments the `ylt` tests asserted.
- `ylt` no longer parses as a clause.
- The writer round trips `yelt`, and `info` shows `YELT2, 4 years, ...`.
- The grammar ambiguity sweep and the full fast suite pass.

`[ELT-Reporting-Punchup]`:

- **Pairs rule.** `tests/test_fcc_surface.py` and `dev/regen_features.py`
  pass with the new pair.
- **Empty when not a loss table.** For `build('agg A 1 claim dsev [1:6]
  poisson')` both strings are `''` and `elt_df` is empty, so aLL drops the
  section.
- **Description.** For `LT.Beta`, `LT.Disjoint`, a shared ELT6, an ELT3, an
  ELT4 with `no2u` and a YELT: the expected lines, before and after
  `update` (grid `n/a` before; check counts after).
- **Explanation.** Each check of `_check_rows` appears once with its
  status; a warning (a pinned `bs` that fails the tail check) is listed
  first and named in the description's `checks` line.
- **`elt_df`.** Columns and index per form as tabled; a shared ELT6 shows
  one row per event with the rolled values and `units > 1`, equal to
  `EltModel.columns()`; zero-rate events absent; ELT3 and ELT4 event totals
  equal the hand sums; a YELT's `prob` sums to one.
- **Retired names.** The three frames are gone from `Aggregate` and
  `dev/FEATURES.csv`.

## Release gates

Edit loop: `tests/test_yelt.py`, `tests/test_loss_table_reports.py`,
`tests/test_elt_uncertainty.py`, `tests/test_fcc_surface.py`,
`tests/test_grammar_ambiguity.py`. At each bump: the full fast suite with
`UV_PROJECT_ENVIRONMENT=.venv`. No numerics change, so the `RuntimeWarning`
gate is not required. Docs are not built.

## Deferred

- Whether a loss-table check warning should also move the aggregate's own
  `validation_description` verdict.
- A per-row `law` column, after an eltq request for per-row classification.
- `Portfolio` narration of its loss-table units.

## Execution log

**Review, 2026-10-28, against 1.19.0.** The plan was drafted against 1.18.2; `[ELT-Shared-Event-Rollup]` landed as 1.19.0 before execution, as the plan requires. Every named path and symbol matched. Line numbers in `_loss_table.py` moved by about 120 with the roll-up (`EltModel` :125, `describe` :375, `_lower_elt` :810, `_lower_ylt` :1130, `summary_frame` :1414, `stats_frame` :1519, `validation_frame` :1579); those in `decl.lark` and `_aggregate.py` did not. The `ylt` counts matched the plan file by file. `yelt` was used nowhere as an identifier (`src/aggregate/agg`, `tests`, `docs`, and `V:/dev/aggregate-api`), and `elt_description`, `elt_explanation`, `elt_df` and `event_table` collide with nothing on `Aggregate`, its parent `Frequency` or its mixins.

**1.20.0, 2026-10-28, `[YELT-Keyword]`.** The rename, the grammar reference regenerated with `aggregate.parser.grammar(add_to_doc=True)` (eleven lines, all the rename), the spec snapshot recaptured with no diff, and `dev/regen_features.py` clean (no member moved). New tests in `tests/test_yelt.py`: `test_info_shows_the_yelt_form` and `test_ylt_is_no_longer_a_clause` (`ylt` refused as a clause and accepted as a name, `yelt` refused as a name).

Divergences:

1. `[YELT-Sublime-Syntax]` `agg.sublime-syntax` carried `ylt` twice (the discrete keywords and the ID mirror) and is not in the plan's list; renamed, since `tests/test_grammar_sync.py` holds it to the grammar.
2. `[YELT-Fixture-Directory]` The synthetic fixtures `tests/data/ylt` moved to `tests/data/yelt` with the test module; the plan named the module only.
3. `[YELT-Article]` Two parser messages built `an {table}` (a ragged row width, `approximate` on a loss table body) and would have shown `an yelt`; they choose the article from the kind.
4. `[YELT-Alignment]` The renamed rules in `decl.lark` and the `yelt` lines of `decl-testers.agg` were realigned by hand to their neighbors' columns. The terminal label in `parser_errors` stays `(year loss table)`.
