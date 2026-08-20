# plan-formats: the format sheets `[Format-Sheets]`

Status: **DONE, moved to `dev/done/` 2026-08-20.** Drafted 2026-08-14 with
the author's rulings of the same day in section 8. All three code phases
landed at a286 to a288, with the `[Format-Sheet-Patterns]` follow up at
a295; the execution record, the rulings taken during execution and the
divergences from the drafted tables are in section 9. The shipped sheets
under `src/aggregate/formats/` are the source of truth for the values now,
not the section 4 tables. What is left is not plan work: the
`PENDING_VOCABULARY` punch list (31 labels after a295) and the naming drift
the registry caught, both tracked in `dev/TODO.md`.

## 1 What this is

Today the library's column formats live in seven module level dicts
(`MEASURE_FORMATS`, `BS_WINDOW_FORMATS`, `SHARPEN_FORMATS`,
`PENTAGON_FORMATS`, `CALIBRATION_FORMATS`, `DISTORTION_FORMATS`,
`EVALUATION_FORMATS`), applied block by block in the frames builders. That is
correct in ownership terms (the library owns the reading of every number it
serves) but scattered in mechanism: the same fact, "a CV reads as `.1%`", is
declared wherever someone remembered to declare it, and a column with no
entry anywhere falls to greater_tables inference silently.

This plan centralizes the vocabulary into two YAML **format sheets** shipped
with the package:

- `formats-raw.yaml`: the default reading of every named column, keyed by
  column label.
- `formats-insurer.yaml`: an **overlay**, holding only the entries where the
  INSURER reading differs from RAW. Absent means same. Two full sheets would
  drift; a delta cannot.

The sheets become the default; a house override file can sit on top; an
explicit `formatters` entry in a frames builder still wins. Nothing changes
on the wire or in the app: formats travel inside the served `TableDoc` per
column, exactly as now, so this is invisible to every consumer and is purely
a change in where the library keeps its own mind.

A second payoff is deliberate: the sheet doubles as a **registry of the
column vocabulary**. It asserts that a column name means one thing across the
package, and the enforcement test in section 6 makes naming drift (two frames
using one name for different units, or a new frame inventing a synonym) a
test failure instead of a quiet inconsistency.

Provisional tier (`aggregate.exhibits`), so this can land in a minor release
with no ceremony. LIB only; no round note, no app work, no API phase.

## 2 Design decisions

1. **YAML.** Comments are the house currency (every entry gets its why) and
   JSON has none. `pyyaml>=6.0` is a declared dependency of
   `greater_tables>=6`, which is a plain dependency of this package, so YAML
   parsing is already guaranteed in every supported install. Open question 5
   asks whether to declare it directly anyway.
2. **Wire safe by construction.** A YAML entry can be any declarative form
   `parse_sugar` accepts: a sugar string (`',.2f'`, `'.3g'`, `'si'`), an int,
   or a mapping of `FormatSpec` fields (the only way to reach `scale`,
   `prefix`, `suffix`, `negative: paren`, `null`, `zero`). It cannot be a
   callable, and that is a feature: a callable never reaches the IR, so a
   sheet loaded from YAML can only say things a client can re-render.
3. **Precedence, low to high:**
   1. greater_tables dtype and tag inference (the guess);
   2. `formats-raw.yaml` (the shipped default reading);
   3. `formats-insurer.yaml`, applied only when the perspective is INSURER;
   4. the house override file (section 5), same two layer shape;
   5. an explicit `formatters` entry in a frames builder or insurer override.
   The sheet is the default, explicit overrides win.
4. **Applied post relabel.** `exhibit_frames` relabels through the host's
   `_relabel` after the builder returns, and greater_tables formatters key on
   the displayed column label. The sheets are therefore applied in
   `build_exhibit`, after relabeling, with keys translated through the same
   renamer, so a relabeled column keeps its format. This also fixes the
   latent bug the current dicts have: a `renamer` that touches `CV` silently
   detaches `MEASURE_FORMATS` today.
5. **Column key, global scope, case sensitive.** The default entry is one
   column label, applied wherever that label appears on a served block. The
   survey (section 4) found the existing vocabulary already consistent
   (`x_min`, `bs`, `error`, `LR` each carry one format everywhere they
   appear), so global scope matches reality. `p` (a probability) and `P`
   (premium) coexist by case. A per exhibit scoped section ships in the
   schema from day one (author ruling, 2026-08-14): it is the escape hatch
   for a future collision, one label meaning two things in two exhibits,
   and having it in the schema means a collision is an entry rather than a
   redesign. No current entry needs it.
6. **Named styles, and tags.** (Author ruling, 2026-08-14.) The sheet
   carries a `styles:` section, a small named vocabulary (`money`,
   `ratio`, `probability`, `residual`) mapping each name to one format,
   and a column entry may reference a style instead of spelling a format.
   Define the reading once, `ratio: '.1%'`, and every ratio column points
   at it; changing the house ratio precision is then one line. Styles
   resolve at load, so greater_tables and the IR see only concrete
   formats. Four style names are special because greater_tables owns them
   as semantic column tags (`ratio`, `year`, `date`, `raw`): a column
   referencing one of those is also stamped with the matching column tag,
   so consumers learn the column's kind as well as its reading. That
   retires the two `ratio_cols` call sites.
7. **Block local formatting stays in code.** Three current uses are driven
   by a block's shape, not by a column's meaning, and stay where they are:
   the sharpen `score_grid` (dynamic step columns, all `'.5f'`), the
   allocation stat slices (columns are unit names, formatted uniformly per
   frame via `float_format`), and any future block whose columns are data.
   The sheet holds vocabulary, not mechanics.
8. **Locations follow the `.agg` rules.** (Author ruling, 2026-08-14.)
   The shipped sheets are package data in a new `src/aggregate/formats/`
   directory, the sibling of `aggregate/agg/`. Overrides are found the way
   `Underwriter` resolves databases, the same three stops with nearest
   winning: the shipped sheets are the base, `~/.aggregate/` (the
   `Underwriter.user_dir`) overlays them, the working directory overlays
   that. Same file names at every stop; a stop with no file contributes
   nothing. This makes a separate house override mechanism unnecessary:
   the override IS the search path, exactly as a user `.agg` database is.
9. **pyyaml declared directly, `safe_load` only.** (Author ruling,
   2026-08-14, after the weighing in section 8f.) `pyproject.toml` gains
   `pyyaml>=6.0` in phase 2.

## 3 Where formatting stands today (survey, 2026-08-14)

Formats only; captions and row flags are out of scope. "none" means
greater_tables inference. "same" in the INSURER column means no difference
from RAW.

| Exhibit | Classes | Block(s) | RAW formats today | INSURER today |
|---|---|---|---|---|
| summary | Aggregate, Portfolio, PnL | summary_df | `CV .1%`, `Skew .3f` (MEASURE_FORMATS) | same |
| summary | Distortion, BivariateAggregate | summary_df | none | same |
| tail | Aggregate, Portfolio | tail_df | none | same |
| stats | all five | stats_df | none | same |
| validation | all five | validation_df | none | same |
| reins | Aggregate, Portfolio | reins_stats_df, reins_summary_df | none | same (blocks restructure, no formats) |
| economic | PnL | economic_df | none | **MEASURE_FORMATS** (`CV .1%`, `Skew .3f`) |
| economic_ratios | PnL | economic_ratios_df, legs_df | none | **ratios block tagged `ratio_cols`**; no format strings |
| economic_waterfall | PnL | walk_df, evaluation_df | evaluation_df tagged `ratio_cols`; else none | same |
| dependency | BivariateAggregate | dependency_df, axis_support_df | none | same |
| bs_window | Aggregate, Portfolio, BivariateAggregate | bs_window_df | BS_WINDOW_FORMATS | same |
| sharpen | Aggregate, Portfolio | sharpen_df | SHARPEN_FORMATS | walk same; **adds score_grid, every column `.5f`** |
| tail_behavior | Aggregate, Portfolio | tail_behavior_df | none | same |
| pricing.calibrate | CalibrationResult | distortion_df | DISTORTION_FORMATS | same |
| pricing.stand_alone | CalibrationResult | calibration_df; stand_alone_df or reins_price_df | CALIBRATION_FORMATS; PENTAGON_FORMATS | same formats (blocks restructure) |
| pricing.allocate (book) | CalibrationResult | calibration_df, pricing_df | CALIBRATION_FORMATS; none | **stat slices via `float_format`** (`LR .1%`, `P ,.2f`, `PQ .3f`, `ROE .1%`) |
| pricing.allocate (occurrence) | CalibrationResult | natural_allocation_df | PENTAGON_FORMATS | same |
| pricing.evaluate | EvaluationResult | evaluation_df | EVALUATION_FORMATS | same |

Reading the RAW against INSURER columns: of the four differences, one is an
accident the sheet erases (economic's RAW serving `CV` unformatted, when RAW
has carried unit formats since a226), one is a tag worth lifting into the
sheet (`economic_ratios`), and two are block local mechanics that stay in
code per decision 7 (score_grid, stat slices).

## 4 The sheets, as drafted from the survey (AUTHOR EDITS HERE)

### 4.0 styles (draft content)

The named vocabulary of decision 6. Column rows below spell literal
formats for surveyability; at transcription, every row whose format
matches a style collapses to the style reference, so these four values
are the ones that matter most.

| Style | Format | Note |
|---|---|---|
| money | `,.2f` | the a259 ruling: money is money at every scale |
| ratio | `.1%` | also stamps the greater_tables ratio column tag |
| probability | `.5f` | the fifth place is where asked-for and grid-deliverable part company |
| residual | `.2e` | a good fit and an exact fit must read apart |

### 4.1 formats-raw.yaml (draft content)

One row per column of the current vocabulary, then the accepted
candidates (author ruling e, 2026-08-14) below the rule. Edit formats in
place. Two rows carry inline FLAGs wanting a ruling.

| Column | Format | Currently from | Note |
|---|---|---|---|
| CV | style `ratio` | MEASURE_FORMATS | |
| Skew | `.3g` | MEASURE_FORMATS (`.3f`) | upgraded: installed greater-tables 6.0.0 parses `g`, significant figures. The `_core.py` comment saying `g` does not exist comes out |
| L | `,.7g` | PENTAGON_FORMATS | money is money at every scale (the a259 ruling) |
| M | `,.7g` | PENTAGON_FORMATS | |
| P | `,.7g` | PENTAGON_FORMATS | |
| Q | `,.7g` | PENTAGON_FORMATS | |
| a | `,.7g` | PENTAGON_FORMATS | |
| LR | style `ratio` | PENTAGON_FORMATS | style `ratio`; also covers economic_ratios, where it is tagged today |
| PQ | `.3f` | PENTAGON_FORMATS | a ratio, reads as one (author ruling in plan-pricing-exhibits) |
| ROE | style `ratio` | PENTAGON_FORMATS | |
| coc | style `ratio` | CALIBRATION_FORMATS | |
| p | `.5f` | CALIBRATION_FORMATS | probability; distinct from premium `P` by case |
| F(a) | `.5g` | CALIBRATION_FORMATS | |
| param | `.5g` | DISTORTION_FORMATS, EVALUATION_FORMATS | agrees in both |
| error | `.5g` | DISTORTION_FORMATS, EVALUATION_FORMATS | residual; scientific so a good fit and an exact fit read apart |
| gini_p | `.5g` | DISTORTION_FORMATS, EVALUATION_FORMATS | agrees in both |
| area | `.5g` | DISTORTION_FORMATS | |
| x_min | `,.1f` | BS_WINDOW_FORMATS, SHARPEN_FORMATS | agrees in both |
| x_max | `,.1f` | BS_WINDOW_FORMATS | |
| W | `,.1f` | BS_WINDOW_FORMATS | |
| bs | `,.6g` | BS_WINDOW_FORMATS, SHARPEN_FORMATS | dyadic fractional, significant digits |
| clipped | `.2e` | BS_WINDOW_FORMATS | |
| score | `.5g` | SHARPEN_FORMATS | |
| extent | `,.5g` | SHARPEN_FORMATS | |
| u_sev_mean | `.2e` | SHARPEN_FORMATS | |
| u_sev_cv | style `ratio` | SHARPEN_FORMATS | |
| u_sev_skew | `.5g` | SHARPEN_FORMATS | |
| u_agg_mean | `.2e` | SHARPEN_FORMATS | |
| u_agg_cv | style `ratio` | SHARPEN_FORMATS | |
| u_agg_skew | `.5g` | SHARPEN_FORMATS | |
| aliasing | `.5g` | SHARPEN_FORMATS | |
| deficit | `.5g` | SHARPEN_FORMATS | |
| seconds | `.3f` | SHARPEN_FORMATS | |
| T | `,.1f` | new, tail_df | return period; ladder values are round in practice |
| VaR | `,.0f` | new, tail_df | FLAG: `,.0f` reads a loss grid at whole currency, but the pentagon amounts ruling says money is `,.2f`. Pick one; the winner may want a second style (`loss_amount`) beside `money` |
| TVaR | `,.0f` | new, tail_df | as VaR, same FLAG |
| xsVaR | `,.0f` | new, tail_df | as VaR, same FLAG |
| VaR/Mean | `.2f` | new, tail_df | leverage, a multiple; FLAG: or `.3f` to match `PQ` |
| ER | style `ratio` | new, economic_ratios_df | lifted from the `ratio_cols` call |
| CR | style `ratio` | new, economic_ratios_df, evaluation_df | both call sites serve it |
| E_LR | style `ratio` | new, economic_ratios_df | mean of ratios |
| E_ER | style `ratio` | new, economic_ratios_df | |
| E_CR | style `ratio` | new, economic_ratios_df | |
| P_share | style `ratio` | new, economic_ratios_df | |
| M_share | style `ratio` | new, economic_ratios_df | |
| Premium spent | style `ratio` | new, evaluation_df | lifted from the `ratio_cols` call |
| Margin spent | style `ratio` | new, evaluation_df | |

### 4.2 formats-insurer.yaml (draft content)

Partly these diffs are to demonstrate capability, but they are also real. The RAW user should
be more at home with scientific notation, hence swapping `g` for `f`.

| Column | Format | Currently from | Note |
|---|---|---|---|
| Skew | `.3f` | MEASURE_FORMATS (`.3f`) | upgraded: installed greater-tables 6.0.0 parses `g`, significant figures. The `_core.py` comment saying `g` does not exist comes out |
| L | `,.2f` | PENTAGON_FORMATS | money is money at every scale (the a259 ruling) |
| M | `,.2f` | PENTAGON_FORMATS | |
| P | `,.2f` | PENTAGON_FORMATS | |
| Q | `,.2f` | PENTAGON_FORMATS | |
| a | `,.2f` | PENTAGON_FORMATS | |
| LR | style `ratio` | PENTAGON_FORMATS | style `ratio`; also covers economic_ratios, where it is tagged today |
| PQ | `.3f` | PENTAGON_FORMATS | a ratio, reads as one (author ruling in plan-pricing-exhibits) |
| ROE | style `ratio` | PENTAGON_FORMATS | |
| coc | style `ratio` | CALIBRATION_FORMATS | |
| p | `.5f` | CALIBRATION_FORMATS | probability; distinct from premium `P` by case |
| F(a) | `.5f` | CALIBRATION_FORMATS | |
| param | `.5f` | DISTORTION_FORMATS, EVALUATION_FORMATS | agrees in both |
| error | `.5f` | DISTORTION_FORMATS, EVALUATION_FORMATS | residual; scientific so a good fit and an exact fit read apart |
| gini_p | `.5f` | DISTORTION_FORMATS, EVALUATION_FORMATS | agrees in both |
| area | `.5f` | DISTORTION_FORMATS | |
| bs | `,.6f` | BS_WINDOW_FORMATS, SHARPEN_FORMATS | dyadic fractional, significant digits |
| score | `.5f` | SHARPEN_FORMATS | |
| extent | `,.5f` | SHARPEN_FORMATS | |
| u_sev_skew | `.4f` | SHARPEN_FORMATS | |
| u_agg_skew | `.4f` | SHARPEN_FORMATS | |
| aliasing | `.4f` | SHARPEN_FORMATS | |
| deficit | `.4f` | SHARPEN_FORMATS | |

## 5 Files, loading, application

- Shipped sheets are package data in `src/aggregate/formats/`
  (`formats-raw.yaml`, `formats-insurer.yaml`), read via
  `importlib.resources`. The loader then looks for the same file names in
  `~/.aggregate/` and the working directory (decision 8, the `.agg`
  rules), merges nearest wins, parses once, and caches the merged result.
  Every entry is validated through `parse_sugar` at load, so a bad string
  raises once, naming file and key, not per cell.
- YAML values are always quoted, by convention stated in the file header:
  several sugar strings begin with characters YAML reserves (`,` in
  `',.2f'` cannot start a plain scalar), so bare scalars are a foot gun
  the convention removes.
- Schema sketch:

  ```yaml
  # formats-raw.yaml: the library's default reading, by column label.
  # Values: format sugar, an int, a FormatSpec field mapping, or the name
  # of a style. Always quote sugar strings.
  styles:
    money: ',.2f'      # money is money at every scale
    ratio: '.1%'       # also stamps the greater_tables ratio column tag
    probability: '.5f'
    residual: '.2e'
  columns:
    CV: '.1%'          # a CV reads as a percentage
    Skew: '.3g'        # three significant figures, wide dynamic range
    L: money
    LR: ratio
    error: residual
  exhibits: {}         # scoped exceptions, exhibit then column: the
                       # collision escape hatch (decision 5), empty today
  ```

- Application in `build_exhibit`, post relabel (decision 4): for each block,
  start from the merged sheet (raw, then insurer overlay when the
  perspective is INSURER, styles already resolved), translate keys through
  the object's renamer, then lay the builder's own `formatters` on top.
  GT-native style references (`ratio`, `year`, `date`, `raw`) also merge
  into the block's tag selectors (`ratio_cols` and friends).
- The seven module dicts are deleted in phase 3 and their exports removed
  from `exhibits.__all__`. `STAT_SLICE_FORMATS`, `STAT_SLICES`,
  `STAT_SLICE_TITLES` and the score_grid literal stay (decision 7).

## 6 Enforcement

A sweep test alongside the existing perspective sweeps in
`tests/test_exhibits.py`: build every exhibit under both perspectives on the
reference objects, collect every served **data** column label, and assert
each is either in the sheet or matched by an exemption. Exemptions are
regex patterns declared next to the test with a reason each, expected to
cover: unit named columns (portfolio slices, `pricing_df`), kappa scenario
columns (`^κ`), percentile ladder columns, sharpen's dynamic step columns,
and string columns. A failure reads as "new column `foo` served with no
declared reading": the author either adds a sheet entry, adds an exemption
with a reason, or renames the column to an existing word, which is the
vocabulary discipline working as intended.

## 7 Phases

Each executed phase bumps the version with a CHANGELOG section, per house
rules.

1. `[Format-Survey]` This document, tables 3 and 4. No code. Author edits
   section 4 (the two FLAG rows and the style values) and confirms the
   pyyaml recommendation below.
2. `[Format-Sheet-Files]` The `src/aggregate/formats/` package directory;
   transcribe the edited tables into the two YAML files, collapsing
   repeated formats onto style references; the loader
   (`exhibits/_formats.py`) with the three stop search path, style
   resolution, load time validation; tests for parsing, overlay and
   nearest wins merging, and caching. `pyproject.toml` gains `pyyaml>=6.0`
   and ships the new package data.
3. `[Format-Sheet-Application]` Wire into `build_exhibit` post relabel with
   the precedence of decision 3; delete the seven dicts and their
   `exhibits.__all__` exports; fold the two `ratio_cols` call sites into
   sheet styles; update the `_core.py` comments (the `g` kind note, the
   MEASURE_FORMATS asymmetry note). Regenerate any exhibit fixtures that
   hash formatted output.
4. `[Format-Sheet-Enforcement]` The sweep test of section 6 with the
   exemption list.

## 8 Author rulings (2026-08-14) and what remains

Rulings, as given against the drafted open questions:

a. **The seven dicts all die.** Confirmed: `MEASURE_FORMATS`,
   `BS_WINDOW_FORMATS`, `SHARPEN_FORMATS`, `PENTAGON_FORMATS`,
   `CALIBRATION_FORMATS`, `DISTORTION_FORMATS`, `EVALUATION_FORMATS` are
   deleted in phase 3. `STAT_SLICE_FORMATS` and the score_grid literal
   survive as block mechanics per decision 7, which is the one deliberate
   exception.
b. **Locations follow the `.agg` rules**: shipped sheets in
   `src/aggregate/formats/`, overridden from `~/.aggregate/` and the
   working directory, nearest wins (decision 8).
c. **Scoped sections ship from day one**, as the name collision escape
   hatch (decision 5).
d. **Styles**: a named style vocabulary with columns referencing styles,
   the reading defined once (decision 6).
e. **All candidates accepted**: the tail ladder and ratio tag rows are in
   table 4.1, two with FLAGs wanting a precision ruling.
f. **pyyaml, weighed.** For declaring it directly: the package imports it,
   so the dependency documents a fact; it is one line; it is immune to
   greater_tables ever swapping its YAML library, which would otherwise
   break this loader with a confusing transitive error; resolution cost is
   zero because pyyaml is already in every install. Against: a second
   constraint on one package to keep in sync, and YAML's quoting rule for
   sugar strings (section 5). The genuine alternative is TOML via stdlib
   `tomllib`: zero dependency, comments supported, always quoted strings,
   and the in-house precedent of `~/.aggregate/config.toml`. YAML stands
   per the author's opening ruling (GT's sheet language is YAML, and the
   mapping form matches GT's own config vocabulary). **Ruled (author,
   2026-08-14): declare `pyyaml>=6.0` directly, parse with `safe_load`
   only.**

Remaining before execution: the section 4 edit (styles table, the two
FLAG rows, any format changes in place). Phases 2 to 4 then run in order,
each with its version bump and CHANGELOG section.

## 9 Execution record (2026-08-14, a286 to a288)

All three code phases landed. The shipped sheets are the source of truth
for the values now, not the tables in section 4: read
`src/aggregate/formats/*.yaml`.

**Rulings taken during execution**, on top of section 8:

1. **RAW money stays `,.7g`** as drafted, so an amount at or above ten
   million reads in exponent form (`2.5e+07`) under RAW and to the cent
   under INSURER.
2. **The tail ladder follows the money entry.** `VaR`, `TVaR` and `xsVaR`
   point at the `money` style rather than taking a `,.0f` of their own, so
   there is one currency word in the sheet and no second style.
3. **`VaR/Mean` reads `.3f`**, matching `PQ`: both are dimensionless
   multiples, so one reading covers both.
4. **`u_sev_cv` and `u_agg_cv` read `.5g`, not the `ratio` style.** They are
   relative errors running from about 1e-7, and `.1%` prints 1e-7 and
   3.4e-5 both as `0.0%`, which is the comparison the probe audit exists to
   support.
5. **The INSURER overlay keeps its fixed decimals on the residuals**
   (`error` at `.5f`, `u_*_skew` / `deficit` / `aliasing` at `.4f`), so a
   very close fit reads as zero in that view deliberately and the raw view
   is where the size of the miss is read.

**Divergences from the drafted tables**, each mechanical:

a. The insurer overlay lost the rows that agreed with RAW (`LR`, `PQ`,
   `ROE`, `coc`, `p`) and its five money rows, which collapsed into one
   redefinition of the `money` style. Section 1's own rule: a delta cannot
   drift, and an overlay row that agrees with its base is a copy waiting to
   go stale. The insurer sheet is one style and thirteen columns.
b. **Styles merge across layers before columns resolve against them.** That
   is what lets the overlay move `money` in one line; resolving each sheet
   independently would have needed the five rows back.
c. `E` and `C` were added to the money entry. They sit in the `amounts`
   block beside `P`, `L` and `M` and are the same unit, and the sweep would
   have asked about them on the next commit.
d. **The scoped section is used on day one.** `P` is premium as a data
   column in twenty served blocks and the probability ladder as `tail_df`'s
   index in ten, so `exhibits: {tail: {P: probability}}` ships. Without it
   the insurer money format prints the 1 in 1000 and the 1 in 10000 rungs
   both as `1.00`. Section 5's "empty today" is wrong as shipped.
e. The sweep asks only about **float** columns, and skips a block whose
   column axis is named, plus `stats_df` and `reins_stats_df` by name (same
   case, unnamed axis). The exemption list in section 6 anticipated regex
   patterns for unit and kappa columns; the structural rule covers most of
   them, and two regexes remain for the kappa and percentile ladders.
f. The 43 labels with no declared reading are not exemptions. They ride in
   `PENDING_VOCABULARY` in `tests/test_exhibits.py`, a two way ratchet, as
   the author's punch list.

**What the registry caught first**, which is the section 1 argument
arriving: four spellings of a mean (`Mean`, `mean`, `EX`, and `Est EX`),
three of a standard deviation (`SD`, `sd`, and the composed `Est CV`
family's partner), and three of a skewness (`Skew`, `Sk`, `skew`). None is
wrong where it sits and no two are the same word. That is a naming
question, not a formatting one.

## Verified facts (2026-08-14)

- Installed `greater-tables` 6.0.0 (LIB `.venv`) parses `'.3g'` to
  `FormatSpec(kind='gen', digits=3)`: the Skew upgrade is available now.
- `pyyaml>=6.0` is a declared runtime dependency of `greater_tables`
  (its `pyproject.toml`), so YAML parsing is present in every supported
  install of this package.
- The relabel ordering (builder, then `_relabel`, then IR build) is
  `_core.py` `exhibit_frames` / `build_exhibit`; formatters applied by
  builders today key on pre relabel names and would miss under a renamer
  that touches a formatted column.
