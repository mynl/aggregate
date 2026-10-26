---
bibliography: D:/Projects/Biblio/uber-library.bib
csl: D:/Projects/Biblio/journal-of-risk-and-uncertainty.csl
---

# Plan: event and year loss tables in DecL

`[DecL-Loss-Tables]`. Drafted 2026-10-26 against aggregate 1.11.1 and the
local eltq 2.10.0 source. **Status: `[Loss-Table-Deterministic-Import]`
executing from 2026-10-26 in four bumps (rulings 13 to 16); the two
uncertainty steps wait on the eltq half, drafted as its own plan in
`V:/dev/eltq/dev/`.** The author's rulings below are settled; the remaining
recommendations are proposals.

## Goal

Build an ordinary `Aggregate` directly from catastrophe model event loss
tables (ELTs) or annual scenarios with specified probabilities in year loss
tables (YLTs).
DecL states the data sources and their units. ELTs become a compound Poisson
model, with `eltq` constructing the event severity when secondary uncertainty
is present. YLTs become an empirical annual distribution, equivalent to
`dfreq [1] dsev [annual outcomes] [probabilities]`, with equal probabilities
when omitted. Both use the existing Aggregate machinery.

The result is one univariate aggregate for the total of all the stated units.
It retains the loss-table declaration, source identities, and a compact audit of the
conversion. Large severity vectors pass as arrays, never through generated
DecL text. A later extension can reuse the normalized event and unit data for
a joint distribution of two units. YLT support lands with deterministic ELT
support in the first implementation step.

## Rulings

Author, 2026-10-26:

1. **[ELT-Secondary-Uncertainty]** Compute the result with secondary uncertainty
   when the source supplies it. Add the optional modifier `elt no2u ...` to
   request the result without it. A declaration returns one `Aggregate`; it
   does not return a pair of with/without results.
2. **[ELT-Event-Identity]** There is no `peril` field or keyword in this design.
   Event IDs are globally unique identifiers of physical events. The same
   physical event may have one row for each affected unit, all carrying that
   same event ID. Distinct physical events, including those from different
   catalogs, must have different IDs.
3. **[ELT-Optional-Maximum]** Support ELT3 and ELT4 as deterministic event tables:
   `[id rate mean]` and `[id rate mean max]`. The fourth column is useful for
   reporting even without secondary uncertainty. ELT6 is
   `[id rate mean max sdi sdc]`.
4. **[YLT-Annual-Outcomes]** Include YLT support in the same project: YLT1 is
   losses only, YLT2 is `[year_id loss]`, and YLT3 is `[year_id loss prob]`.
   YLT1 and YLT2 give years equal probability; YLT3 supplies their probabilities.
   Each row is a complete annual scenario. The model is `dfreq [1] dsev`
   with the corresponding probabilities and no additional Poisson compounding.
5. **[ELT-Moments-Before-Grid]** Read the tables and compute model moments
   before selecting the grid. Use the standard aggregate bucket/window
   routines, informed by the model and its tail, and defer eltq's severity
   quantization until `update`. `hints{}` remains an optional override;
   a separately required `elt_bs` is not the design.
6. **[Loss-Table-Large-Grids]** Plan for grids substantially larger than the
   usual `log2=16`. Add the global setting `elt_max_log2`, expressed as an
   exponent, to govern the ELT grid budget. Do not introduce `elt_max_points`
   or a smaller hidden source-grid cap.
7. **[ELT-No2u-Scope]** `no2u` modifies its own source table and converts
   ELT6 to ELT4. Apply this conversion before checking compatibility.
   Effective ELT6 cannot mix with ELT3/4; converted ELT6 can. Applying
   `no2u` to only some ELT6 sources while others retain 2U is an error.
8. **[ELT-Maximum-Compatibility]** ELT3 and ELT4 may be combined, in which
   case the ELT4 sources drop their maximum columns and the combined table
   is ELT3. An entirely ELT4 result retains maximums for reporting.
9. **[YLT-Year-Completeness]** YLTs include zero-loss years explicitly.
   All unit YLTs have the same number of rows and the same complete year
   set. Missing years are errors, never implicit zeros.
10. **[YLT-Annual-Terms]** YLTs accept aggregate reinsurance and reject
    occurrence terms, including through references and P&L composition.
    Annual totals do not identify their constituent occurrences.
11. **[Loss-Table-Report-Names]** Follow the existing public naming pattern:
    `loss_table_summary_df`, `loss_table_stats_df`, and
    `loss_table_validation_df`. Extend the existing `info` string. Introduce
    another suffix only when none of the existing suffixes is appropriate.
12. **[ELT-Windowed-Support]** A very large exposed value $M$ with small
    expected loss must not force a grid reaching $M$. Retain $M$ in the
    probability model and distinguish it from the numerical window. Check
    eltq's existing windowing behavior before designing the adapter.

Author, 2026-10-26, at the execution review (these supersede the text below
where they conflict):

13. **[Loss-Table-Step-Bumps]** `[Loss-Table-Deterministic-Import]` lands as
    four bumps, one commit each: `[Quote-Safe-Statements]` (both splitters
    lift quoted strings), `[Loss-Table-Inline]` (grammar, exact IDs, ELT3/4
    and `no2u` lowering, YLT1/2/3, writer, composition, `info`),
    `[Loss-Table-Files]` (readers, aliases, path base, digests) and
    `[Loss-Table-Reports]` (the three `loss_table_*_df` frames).
14. **[ELT-Grid-Budget-Deferred]** `elt_max_log2` moves to
    `[ELT-Single-Event-Uncertainty]`, where it is designed with the window
    contract it serves. In `_bucket_window.bs_window`, `log2` is a cap that
    the `moment` and `bounded_small` methods fill, so a default ELT budget
    of 24 would allocate $2^{24}$ buckets for every ELT; a true budget needs
    a rule for choosing `log2` below the cap. Deterministic ELTs and YLTs
    size like any aggregate: the build default, `hints{log2=...}` or
    `build(..., log2=...)`.
15. **[DecL-No-Data-Word]** The optional `data` filler is dropped. The new
    reserved words are `elt`, `ylt`, `no2u` and `unit`; the source follows
    the unit clause directly.
16. **[Eltq-Reflected-Plan]** The eltq prerequisites (`[Eltq-Model-Moments-And-Windows]`,
    `[Eltq-Windowed-Shared-Events]`) are drafted as a reflected plan in
    `V:/dev/eltq/dev/`, to be reviewed and executed there. eltq 2.10.0 is
    local only (PyPI stops at 1.12.0), so the uncertainty steps also need an
    eltq release before aggregate can declare the dependency.

Units may represent business units, perils, regions, or their combinations;
their names do not determine dependence. ELT event identity determines shared
occurrences. For YLTs, matching year IDs identify the same annual scenario.

## Background and verified integration points

The mathematical background is the author's
[CMELT.qmd](D:/Projects/aggregate/secondary-uncertainty/CMELT.qmd) and
[MVMELT.qmd](D:/Projects/aggregate/secondary-uncertainty/MVMELT.qmd). The former
describes the row distributions, including boundary and inadmissible-variance
cases. The latter explains why combining unit rows into one fitted beta
generally changes the model. The published algorithms are @Burke2016 and
@Homer2017a, with the industry conventions collected in @MitchellWallace2017.


Relevant past plans are `[Agg-As-Severity]` in
`AGG-LINK/dev/done/plan-agg-port-as-sev.md`, the discrete bivariate work in
`plan-bv-discrete.md`, `[Dfreq-One-Claim-Shortcut]` in
`plan-dfreq-one-claim-shortcut.md`, and `[DecL-Word-List]` in
`plan-decl-word-list.md`.
The archived TODO already scopes this work under
`[Data-To-Dsev-Preprocessors]`. This plan implements its ELT and YLT limbs;
observed-loss resimulation remains separate. It does not complete the live
`[Portfolio-Shared-Mixing-Dependence]` item.

| Current implementation | Consequence for this plan |
|---|---|
| `decl.lark:agg_body` is shared by ordinary aggregates and inline P&L engines; `agg_out` is also a portfolio item. | Add separate ELT and YLT body alternatives and preserve those composition paths. |
| `parser.py:UnderwritingLexer.preprocess` and `raw_statements` split statements; nested brackets already work for `dbvsev`. | Reuse nested matrix handling, but make quoted text safe before accepting file paths. |
| `Underwriter._factory` resolves a deferred `sev_ref` before calling `Aggregate(**spec)`. | An ELT can use the same construction boundary without doing file I/O in the parser. |
| `Aggregate.__init__` creates severity objects and calls `moms()` before any density is built. | Construct a moment-bearing ELT severity at this point; its discretization is deferred. |
| `Aggregate._bs_window` delegates to `_bucket_window.bs_window`; `recommend_bucket` has been retired. | Preserve the current annual grid selector. Do not introduce a second annual sizing algorithm in the underwriter. |
| `SeverityDHistogram` and `_DiscreteRV` provide exact moments and layer calculations for supplied atoms. | Deterministic ELTs and YLTs use this path directly. ELT6 keeps its model moments independently of its eventual grid. |
| `Aggregate.update` selects `_bs_window`, then `update_work` establishes `xs_sev` and calls `discretize`. | Quantize ELT6 inside this existing discretization stage, before occurrence reinsurance and the frequency transform. |
| `_bounded_severity_window` reads severity support and frequency moments; `_bucket_window` also has CDF-based diagnostics. | Supply true support and model moments without forcing full exposure coverage; add measured tail checks and handle unavailable general CDF diagnostics. |
| `parser.py:dsev_main` gives omitted probabilities equal weights; `dfreq` produces empirical frequency, with `exp_en=-1` as its constructor sentinel. | YLT lowering can reproduce the existing `dfreq [1] dsev [...]` spec directly. |
| `Aggregate.one_claim` recognizes empirical frequency concentrated at one; the convolution helper copies the discretized severity on that path. | YLT needs no new distribution class or compounding algorithm. |
| `Recipe.source` records an originating `.agg` path or `session`; recipes retain symbolic specs. | Resolve relative ELT paths using recipe provenance and preserve that provenance through nested construction. |
| `decl_writer._render_agg`, P&L rendering, and `Underwriter.to_agg` reconstruct declarations. | ELT syntax needs a writer as well as a parser. Export must not accidentally expand a million-point severity. |
| `eltq.quantize_arrays` returns `raw, secondary, meta`; `multiunit_views` returns requested view arrays and metadata. | The multi-unit result has no separate raw array. Mean-only totals must be formed by grouping event means. |
| Both eltq APIs require `bucket_size`; they can derive `grid_length`. | eltq does not choose `bs` or annual FFT `log2`. Pass length explicitly to avoid automatic coverage of the largest maximum. |
| The univariate route supports windows with overflow diagnostics; the current shared-event route requires full support. | Extend shared-event windowing upstream; do not force large schedules to their full maximum. |
| eltq's multi-unit input requires contiguous event groups, one row per unit, and one rate per event. | Normalize, validate, group, and sort before calling it. |
| The inspected eltq API has no public grid-independent first-three-moment calculation for the shared-event total. | Add a moment-only API before integrating that route; do not derive its theoretical moments from its final output grid. |

The paths and symbol names above were checked in the local sources. The old
architecture summary in `CLAUDE.md` predates the split into `_aggregate.py`,
`_severity.py`, and `_frequency.py`; implementation should follow these files.

## Scope and boundaries

The first implementation includes inline tables, local tabular files, labels,
multiple units, deterministic ELTs, YLT1/2/3, secondary uncertainty
with Burke's coupling, `no2u`, applicable reinsurance, recipe round trips, and
diagnostics. ELT4 maximums appear in summaries when the combined effective
form remains ELT4; combining with ELT3 drops them.

An event's unit losses are added before applying an occurrence treaty on the
declared aggregate. The resulting occurrence loss is compounded annually;
aggregate reinsurance then acts on that annual result. This is the same
ordering as an ordinary aggregate after its severity has been constructed.
For a YLT, sum units within each annual scenario and apply aggregate
reinsurance to those annual totals. The YLT surface excludes
occurrence terms because an annual total contains no event breakdown.

Unit-specific deductibles, unit-specific reinsurance, vendor financial terms,
new models of annual event dependence, a joint output, and allocation by unit are out
of scope. eltq already implements some of these operations, but exposing them
requires separate DecL and reporting decisions. Ordinary `Portfolio` units
continue to combine independently. Users must put units sharing physical
events inside the same ELT aggregate to retain that dependence. Likewise,
dependent YLT units must enter one YLT aggregate with aligned years. Their
supplied annual dependence is preserved; no new dependence model is fitted.
Year-event-loss transaction tables remain a separate extension. Do not mix
ELT and YLT sources inside one aggregate body.

## DecL surface

### [ELT-Syntax] declaration and modifiers

Proposed surface, with braces denoting optional material in this sketch only:

```text
agg NAME {as "Human name"}
    elt {no2u} {unit UNIT {as "Human unit name"}} SOURCE
    {elt {no2u} {unit UNIT {as "Human unit name"}} SOURCE}
    {occurrence reinsurance}
    {aggregate reinsurance}
    {existing orientation and trailer clauses}
```

`SOURCE` is an inline matrix or a double-quoted local file path. Modifier
order is fixed: `no2u`, then `unit`, then the source (ruling 15 dropped an
optional `data` filler). Every clause has its own explicit metadata;
there is no state inherited from the preceding clause.

An omitted unit is `Total`. `unit` uses existing DecL identifier rules and
`as` uses the existing label rules. Several sources may contribute disjoint
events to one unit. Several units may share events. Repeating a unit name
with conflicting explicit labels is an error; omission of a label does not
erase one already supplied.

**`no2u` applies to its own ELT clause and converts ELT6 to ELT4.** Remove
the SD fields from its effective table, retain its maximum, and model each
row as an atom at its mean. Record the original form and conversion in
provenance. Check compatibility after this projection: every source must
be effective ELT6, or every source must be effective ELT3/4. A remaining
ELT6 alongside a converted ELT4 is an error. Computing the mean-only
counterpart therefore means adding `no2u` to every ELT6 clause. The modifier
is redundant and harmless on ELT3/4. Naturally deterministic zero-SD rows
within an unmodified ELT6 are valid and do not change its table form.

One ELT body determines its own count and severity. It cannot also contain
`claims`, `dfreq`, `sev`, `dsev`, a user-specified frequency, or `wait`.
It produces Poisson frequency without a trailing `poisson` token. Occurrence
and aggregate clauses follow the complete list of ELTs, never an individual
source. Initial syntax does not add per-source insurance terms or a public
`dsev2u` keyword. Method-of-moments `approximate` is deferred; an explicit use
with an ELT body should receive a clear diagnostic.

Single newlines remain whitespace. A blank line or a line-final semicolon
ends a statement, as in the current language. Commas between values or rows
remain optional under the existing grammar. This feature does not broaden
the statement separator rules.

### [ELT-Examples] accepted declarations

The corrected basic example:

```decl
agg Basic as "My easy example"
    elt [
        [1 .1 100]
        [2 .1 40]
        [3 .1 80]
    ]
```

Its ELT4 counterpart carries additional reporting information:

```decl
agg BasicWithMaximums
    elt [
        [1 .1 100 1000]
        [2 .1 40 600]
        [3 .1 80 2000]
    ]
```

The two declarations have identical loss distributions. A maximum is neither
an additional loss nor an instruction to introduce uncertainty.

Independent catalogs represented as units, with globally distinct event IDs:

```decl
agg MultipleCatalogs as "Wind and earthquake"
    elt unit Wind "d:/data/wind.csv"
    elt unit Quake "d:/data/quake.csv"
```

The mean-only version of a six-column source:

```decl
agg MeanOnly
    elt no2u unit Property "d:/data/property.csv"
```

Two units exposed to the same events, followed by a treaty on their total:

```decl
agg SharedEvents as "Personal and commercial property"
    elt unit Personal as "Personal lines" [
        [1 .1 300 3000 400 300]
        [2 .1 100 1000 371 267]
    ]
    elt unit Commercial as "Commercial lines" [
        [1 .1 200 1000 300 200]
        [2 .1 200 4000 150 533]
    ]
    occurrence net of 1000 xs 500
    aggregate net of 2000 xs 1000
    hints{log2=22}
```

The last example gives an optional large-grid budget. Omitting the trailer
uses the configured budget; in both cases the existing sizing routines
choose `bs` from model moments, with tail checks on the numerical window.

### [ELT-Grammar] parser responsibilities

Add an `agg_body_elt` alternative to the shared body. Its transformer emits
an ordered `elt_sources` list plus the ordinary label, reinsurance, and
trailer keys. It does not read files, import eltq, construct DataFrames, or
quantize. Each source records its kind, inline rows or path, unit identity
and label, and `no2u` flag. There is no new top-level recipe kind.

Inline rows have exactly three, four, or six fields. The event ID needs its
own token handling: preserve integer text or a quoted string without routing
it through the global floating-point `NUMBER` transformer. Losses and rates
use the existing numeric expression grammar and subsequent numeric checks.
Reject ragged matrices and mixed row widths within one inline source.

Quote handling is part of this phase. The current comment regex strips `#`
and `//` inside quoted strings, and bracket scanning sees brackets inside
strings. Protect strings consistently in both `preprocess` and
`raw_statements`, alongside the existing trailer protection. Preserve path
contents, repeated spaces, backslashes, and punctuation; forward slashes
are the canonical writer's Windows path spelling. Literal double quotes in
paths can remain unsupported under the existing `STRING` contract. Reuse
one scanner or masking helper rather than allowing the two splitters to drift.

New reserved words are `elt`, `ylt`, `no2u`, and `unit`. Adding a word to the
reserved set can invalidate an existing user identifier of that exact spelling.
Check the corpora, document this compatibility change, and give a rename
diagnostic where possible. Human labels remain free to contain these words.

### [YLT-Syntax] annual observations

Use the same source, unit, and label conventions:

```text
agg NAME {as "Human name"}
    ylt {unit UNIT {as "Human unit name"}} SOURCE
    {ylt {unit UNIT {as "Human unit name"}} SOURCE}
    {aggregate reinsurance}
    {existing orientation and trailer clauses}
```

An inline source is a flat vector of annual losses (YLT1), rows
`[year_id loss]` (YLT2), or rows `[year_id loss prob]` (YLT3). A file supplies
the corresponding columns. The flat
form supplies implicit year IDs `1, ..., n` in source order. Multiple unit
vectors of equal length therefore explicitly pair their positions as the
same simulated years. Explicit year IDs are preferred for files and any
workflow that reorders the data. YLT3 probabilities attach to
annual scenarios. There is no event rate, maximum, or secondary-uncertainty
parameter in a YLT source.

```decl
agg AnnualSample as "Four equally likely years"
    ylt [0 0 100 300]
```

This is equivalent to `agg AnnualSample dfreq [1] dsev [0 0 100 300]`.
Repeated losses retain their multiplicities. The probability of zero is
one half, not one third after deduplicating the distinct loss values.

```decl
agg WeightedAnnualSample
    ylt [[1 0 .5] [2 100 .3] [3 300 .2]]
```

This is equivalent to `dfreq [1] dsev [0 100 300] [.5 .3 .2]`. The annual
mean is 90 and the variance is 12,900. Repeated losses coalesce by summing
their probabilities after year alignment.

```decl
agg AnnualUnits as "Aligned annual outcomes"
    ylt unit Personal [[1 0] [2 100]]
    ylt unit Commercial [[1 100] [2 0]]
    aggregate net of 50 xs 25
```

Both years have gross total loss 100. The annual recovery is 50 and the net
loss is 50 in both years. Combining the two marginal unit distributions as
independent would give a different answer and must not happen here.

File sources use `ylt unit Personal "d:/data/personal-years.csv"`, or a single
long-format file with a unit column. Omitted unit means `Total`. Labels,
source-relative paths, and canonical export use the same rules as ELTs.

Add `agg_body_ylt` and ordered `ylt_sources` alongside their ELT counterparts.
Flat values use the existing numeric-expression grammar; IDs in paired rows
use the same exact identifier handling as ELT event IDs. The YLT body has
no `occ_reins` slot. It also rejects `no2u`, `claims`, `dfreq`, `sev`, `dsev`,
frequency clauses, and ELT numerical hints. `no2u` cannot remove uncertainty
already realized in annual outcomes. `approximate` remains deferred as for
the ELT body. Parsing does no file I/O and creates an ordinary `agg` recipe.

On an ELT, `occurrence` transforms each physical event's loss. On a YLT,
the single synthetic severity draw is a whole year's loss, so applying the
same spelling would transform the annual total while appearing to price an
event treaty. Reject it with guidance to use `aggregate` for annual terms
or supply event-level information for occurrence terms. Apply this check
also when a named YLT aggregate is reused with added occurrence clauses or
wrapped in a P&L occurrence program; retain its source kind through resolution.
The low-level, explicitly authored `dfreq [1] dsev [...]` API is unchanged.

## Data contract

### [ELT-Columns] normalized input

The normalized long table has columns in this order:

```text
unit, event_id, rate, mean, max, sdi, sdc
```

Preserve source number, source row, source width, and `no2u` in associated
provenance. Unit labels belong in a separate mapping keyed by identity.
Do not store a display label in place of the unit key.

| Input form | Required columns | Model without `no2u` | Meaning of `max` |
|---|---|---|---|
| ELT3 | `id rate mean` | Atom at `mean` | Unknown, retained as missing in input reporting. |
| ELT4 | `id rate mean max` | Atom at `mean` | Supplied exposure or maximum, retained for reporting. |
| ELT6 | `id rate mean max sdi sdc` | Fitted row law, combined across units within each event. | Upper bound used in the row law. |

Validate each original form, apply `no2u`, then choose the combined form:

| Effective source forms | Combined form |
|---|---|
| All ELT6 | ELT6, including naturally zero-SD rows. |
| All ELT4, including converted ELT6 | ELT4, retaining maximums. |
| Any ELT3 with only ELT3/4 companions | ELT3; drop every maximum column. |
| Any ELT6 with ELT3/4 companions | Error; explain which ELT6 sources still need `no2u` for a deterministic build. |

A five-column source or a partial pair of SD columns is an error. Form is
determined by columns, not by whether all SD values happen to be zero.
An unmodified zero-SD ELT6 still cannot mix with ELT4. Validate supplied
fields before projection; `no2u` does not excuse malformed input.

For deterministic computation, effective event support is its mean even
when an ELT4 maximum is much larger. The combined ELT3 reports maximums
as unavailable, not partial totals or imputed means. Provenance may record
that a source supplied maximums which were dropped; effective summaries
must not quietly reintroduce them.

### [Loss-Table-Files] readers, column aliases, and paths

ELT and YLT imports share a small explicit pandas reader registry: CSV, TSV,
Parquet, Feather, and Excel. Start with CSV and Parquet in the acceptance
suite; the others use the same normalization path. CSV and TSV have a header row; Excel uses the first
sheet initially. Unsupported extensions and missing reader engines produce
messages naming the path and the missing capability. Arbitrary pandas
function names, reader expressions, and pickles are not part of the syntax.

Normalize header whitespace, case, spaces, and underscores before consulting
one alias table. At minimum:

| Canonical | Recognized aliases |
|---|---|
| `event_id` | `id`, `event_id`, `event`, `eventid` |
| `rate` | `rate`, `frequency`, `freq`, `annual_rate` |
| `mean` | `mean`, `mean_loss`, `meanloss`, `loss` |
| `max` | `max`, `max_loss`, `maximum_loss`, `exposure`, `limit` |
| `sdi` | `sdi`, `sd_indep`, `sdind`, `stddevi` |
| `sdc` | `sdc`, `sd_corr`, `sdcor`, `stddevc` |
| `unit` | `unit`, `unit_id` |
| `year_id` (YLT) | `year_id`, `year`, `trial_id`, `trial`, `simulation_id`, `sample_id`, `id` |
| `loss` (YLT) | `loss`, `annual_loss`, `annual_loss_amount` |
| `prob` (YLT3) | `prob`, `probability`, `p` |

Apply the ELT or YLT alias map according to the declaration keyword, so
`id` means an event in one and a year in the other. Two columns mapping to
one canonical name are an error even if their values agree. Extra columns
may remain in source metadata but are not silently used as inputs.
Do not treat an annual expected loss column as a conditional mean:
`mean` is the mean given the event, not rate times that mean. An arbitrary
vendor mapping can initially be handled by renaming columns before import;
a general mapping language is a separate extension.

A file may already contain a `unit` column. A clause unit supplies the unit
when that column is absent and must agree with it when present. Without either,
use `Total`. A conflicting clause and column is an error, not an overwrite.
Thus a single long-format file can describe several units without splitting
it into files. Model-family detection counts loss fields, not the optional
unit or provenance columns.

Resolve ELT and YLT relative paths against the originating `.agg` file's
directory. For a session declaration, capture the session working directory
at registration.
Carry that base through references, portfolio items, and inline P&L engines.
Do not let a later working-directory change alter an already registered
recipe's interpretation. Preserve the authored path in the symbolic recipe
and record the resolved path in build metadata. When exporting into another
directory, rebase the path or write an absolute path; never silently change
which file the declaration names.

Event and year IDs are exact identifiers. Preserve string IDs, including
leading zeroes,
and integer IDs without float conversion. Normalize an integer ID to its
decimal string so integer `1` matches textual `"1"`, while `"001"` remains a
distinct ID. Reject floating-point file ID columns with instructions to use
integer or text IDs; a float may already have lost precision. Read CSV ID
columns as text after identifying the header. Empty IDs and null IDs are
errors. File and inline representations must obey the same identity rule.

### [ELT-Validation] fail before computing

Validate the complete combined input, including rows marked `no2u`:

1. Each `(event_id, unit)` occurs once. Duplicate rows, including identical
   duplicates from two files, are errors. Summing them would silently double
   exposure or erase a missing subunit distinction.
2. All rows of an event carry the same annual rate. Proposal: compare with
   `rtol=1e-12`, `atol=0`, then use the first rate and record any accepted
   nonexact agreement. Write that canonical rate into every effective row
   before calling eltq, whose multi-unit check requires exact equality.
   Never sum or average the rates across units.
3. Rates and all supplied loss fields are finite and nonnegative. Rates may
   exceed one; they are intensities, not probabilities. Require `mean <= max`
   whenever a maximum is supplied.
4. Positive-rate zero-loss events remain valid events with an atom at zero.
   Missing unit rows mean no loss for that unit in that event; do not expand
   the sparse table into every event by every unit combination.
5. Zero-rate events contribute no probability or annual moments and are
   reported as excluded. An empty table or a declaration with no positive
   event rate raises a useful error. A positive-rate all-zero-loss table is
   the valid degenerate annual loss at zero.
6. Boundary or inadmissible ELT6 moments follow the explicit row policy below.
   They are distinct from malformed data and must appear in diagnostics.

eltq can drop malformed rows under its own input policy. The aggregate adapter
must prevalidate so a user does not get an apparently valid smaller portfolio.
Translate any unexpected row-drop diagnostic into an error with source and
event context. A rate or mean mismatch after conversion is also an error,
subject to separately stated floating-point and discretization tolerances.

An event ID accidentally shared by unrelated catalogs with different unit
names can be indistinguishable from a genuine shared event. The contract must
state that users namespace distinct catalogs before loading when necessary.
There is no automatic file-based namespace, because it would destroy genuine
shared events split across files.

### [YLT-Data-Contract] years, units, and zero losses

The YLT normalized table is `unit, year_id, loss, prob`, with source and row
provenance alongside it. Losses are finite and nonnegative. YLT1 has a loss
column; YLT2 adds a year ID; YLT3 adds `prob`. A probability column requires
an explicit year ID. Optional unit metadata does not change the form.
Sources have uniform row width and all probabilities in a YLT3 source must
be present. YLT does not reuse the ELT `mean` or `rate` field.

Probabilities are finite and nonnegative. Each unit's complete table has
year probabilities summing to one within `rtol=1e-12`,
`atol=1e-15` (proposed tolerance). Normalize only this accepted roundoff
residual and record it. A material shortfall or excess is an error: do not
silently reinterpret percentages, counts, event rates, or arbitrary weights
as probabilities. All-zero probabilities are an error. Zero-probability
years are retained for alignment and provenance but do not affect support,
moments, or the output law. A separate `weight` or `rate` column is unsupported.

For YLT1/2, assign `1/n` using the complete year count. For multiple units,
each year's
probability must agree across units with the same tolerance; propagate one
canonical per-year value before constructing the total. Count that value
once, rather than adding or multiplying it across units. YLT1/2 sources may
coexist with YLT3 only when the latter specifies the same uniform year law;
otherwise reject the conflicting scenario probabilities. No omitted
probability field silently inherits a nonuniform law from another unit.

Explicit year IDs identify simulated annual scenarios, not physical events
or necessarily calendar years. Every unit in a declaration refers to the
same simulation and year namespace. Independence must not be inferred from
different unit names. Equal IDs from unrelated simulation runs are not an
automatic pairing contract; align those runs deliberately before importing,
or model their marginal annual distributions as independent portfolio units.

**Confirmed completeness rule:** every unit supplies the same complete
year set, exactly once, and hence the same number of rows. Each source
table is a complete YLT; partial-year file shards are not part of this
release. A long-format source has the same row count for each unit.
There is one row per `(year_id, unit)`. Duplicate keys, even with identical
losses, are errors. Equal losses in different years are valid and remain
separate observations until after year alignment. Missing years are errors,
not silently zero and not an instruction to use an inner join. Report
which units and representative years are missing.

Zero-loss years with positive probability must be explicit. Their count
determines the zero mass for YLT1/2; their probabilities determine it for
YLT3. A file listing only loss
years is not the complete YLT described by this syntax. Entirely omitted
zero years cannot be detected from the remaining rows, so completeness is
part of the caller's data contract. Sparse inputs with a declared total
trial count or explicit trial universe can be added later if needed.
An empty YLT is an error; a nonempty all-zero YLT is valid, as is one year.

For a flat vector or one-column file, generate year IDs from row positions
before sorting or coalescing losses. A long-format file containing several
units must include explicit year IDs. Multiple single-unit positional sources must have
equal lengths and assert alignment by position. Permit only one implicit-ID
source per unit. Explicit and implicit IDs can
coexist only when their complete normalized year sets match. Never treat
missing rows in a shorter vector as zeros.

Normalize and validate each unit's year set and probability law, then sum
losses within years before coalescing equal totals and their probabilities.
Keep year counts, zero-year counts, and zero-year probability in metadata.
These are different quantities for YLT3. Joint output and unit allocation remain deferred, but
the source recipes retain the alignment needed to reconstruct them later.

## Probability model

### [YLT-Empirical-Annual-Law] probability per annual scenario

For $n$ supplied annual scenarios, let $y_{ju}$ be unit $u$'s loss in year
$j$ and $a_j=\sum_u y_{ju}$. Let $p_j=1/n$ for YLT1/2 and the supplied
probability for YLT3. The annual law is

$$
\Pr(A\in B)=\sum_{j=1}^{n}p_j\mathbf 1\{a_j\in B\},
\qquad
\mathsf E[A^r]=\sum_{j=1}^{n}p_j a_j^r,
\qquad \sum_jp_j=1.
$$

Thus each distinct loss value has the sum of its scenarios' probabilities.
For YLT1/2 this is its multiplicity divided by $n$. Use distribution moments,
with no unbiased-sample correction. In the uniform case this means a
denominator of $n$, not $n-1$. The variance is
$\mathsf E[A^2]-(\mathsf E[A])^2$; the ELT compound-Poisson cumulant
formula below does not apply. The count of synthetic annual draws is
identically one, and $\Pr(A=0)=\sum_{j:a_j=0}p_j$.

This is a finite empirical distribution. Existing annual grid selection
and discrete rebucketing still introduce their ordinary numerical limits,
but there is no separate ELT quantization, source bucket, or tail
extrapolation. No `eltq` import is needed. The model carries any dependence
already present in the aligned unit outcomes, including annual event
dependence and realized secondary uncertainty, without attempting to infer
or reverse how the simulation generated them.

### [ELT-Poisson-Mixture] one count per physical event

Let $e$ index distinct physical events, with rate $\lambda_e$, and let
$X_e=\sum_u X_{eu}$ be the total loss in one occurrence of that event.
Occurrences of different events are independent Poisson processes; unit
losses inside an occurrence need not be independent. Define

$$
\Lambda=\sum_e\lambda_e,
\qquad
F_X(x)=\frac{1}{\Lambda}\sum_e\lambda_e\Pr(X_e\le x).
$$

Then annual loss has the ordinary compound Poisson representation

$$
A=\sum_{k=1}^{N}X_k,
\qquad N\sim\operatorname{Poisson}(\Lambda),
\qquad X_k\stackrel{\mathrm{iid}}{\sim}F_X.
$$

For deterministic rows, form $m_e=\sum_u\mu_{eu}$ first, then use atoms at
$m_e$ with weights $\lambda_e/\Lambda$. Coalesce equal loss values by adding
their weights. A source's number of units does not affect its event rate.
With disjoint event IDs the same construction is just the rate-weighted
mixture of the independent catalogs.

Do not remove a zero severity atom and leave $\Lambda$ unchanged. If the
severity has probability $p_0$ at zero, annual zero-loss probability is
$\exp[-\Lambda(1-p_0)]$, not generally $\exp(-\Lambda)$.

### [ELT-Row-Laws] secondary uncertainty and effective moments

For a nondegenerate row with $0<\mu<M$, set

$$
\sigma=\sigma_I+\sigma_C,
\qquad p=\frac{\mu}{M},
\qquad k=\frac{\mu(M-\mu)}{\sigma^2}-1,
\qquad \alpha=pk,\quad\beta=(1-p)k.
$$

When $0<\sigma^2<\mu(M-\mu)$, the row loss is $M$ times a
$\operatorname{Beta}(\alpha,\beta)$ variable. Its raw moments are

$$
\mathsf E[X^r]=M^r\frac{(\alpha)_r}{(\alpha+\beta)_r},
\qquad r=1,2,3.
$$

Here $(a)_r$ is the rising factorial. Delegate row classification and numerical
evaluation to eltq; do not copy its beta kernel into aggregate.

Use eltq's `variance_policy="bernoulli"`: at or above the variance bound,
the effective row law is supported on $0$ and $M$, with probability $\mu/M$
at $M$. Its moments are $\mathsf E[X^r]=\mu M^{r-1}$. Zero SD, zero exposure,
boundary means, and rows below eltq's resolution threshold become atoms as
specified by eltq. In particular, a boundary mean with positive stated SD
does not produce that stated variance; record the effective treatment.
`no2u` explicitly requests the ELT4 projection and atom at $\mu$. Preserve
original-form provenance without restoring discarded fields to effective
model summaries.

For a single-row event, these formulas give model moments before quantization.
For every compound Poisson model, the annual cumulants satisfy

$$
\kappa_r(A)=\sum_e\lambda_e\mathsf E[X_e^r],\qquad r=1,2,3.
$$

Thus $\mathsf E[A]=\sum_{e,u}\lambda_e\mu_{eu}$ for both uncertainty
settings, before financial terms. A variance adjusted to a Bernoulli boundary
must use the effective variance in this audit, not the infeasible input SD.

### [ELT-Shared-Events] Burke coupling

For each occurrence of a shared event, Burke uses one common normal score
$Z_c$ and independent unit scores $Z_u$:

$$
w_u=\frac{\sigma_{C,u}}{\sqrt{\sigma_{I,u}^2+\sigma_{C,u}^2}},
\qquad
V_u=w_uZ_c+\sqrt{1-w_u^2}Z_u,
\qquad
X_u=F_u^{-1}(\Phi(V_u)).
$$

Here $F_u$ is the CDF of the monetary row loss, so there is no additional
factor of $M_u$ outside its inverse. Atom rows bypass the loading calculation.
The common score is fresh for each event occurrence; it is not an annual
factor shared by every catastrophe.

Use eltq's `coupling="burke"`, `z_rule="stratified"`, and `z_nodes=32`
defaults initially. Retain the effective values in the audit. The adapter's
internal options record has a named coupling field so future alternatives
do not require another architecture. The initial public DecL does not add a
coupling selector. Exposing eltq's `comonotone`, `independent`, `homer`, and
`matched` modes is a later, small surface extension with explicit model
documentation; those modes already exist upstream.

Shared-event second and third moments require cross moments. For example,

$$
\mathsf E[(\textstyle\sum_uX_u)^2]
=\sum_u\mathsf E[X_u^2]+2\sum_{u<v}\mathsf E[X_uX_v].
$$

The nonlinear beta inverse means the Gaussian-score correlation is not
generally the loss correlation. Summing row variances, applying the standard
SD roll-up formula, or fitting one beta to rolled moments is not Burke's
model. Compute analytic AAL for all routes and the first three moments of
the effective total before update, using analytic formulas where available
and a separate numerical moment pass for shared events. Mark numerical
model moments as such; they must not be read back from the grid being checked.

## Construction and grid design

### [Loss-Table-Preparation] parse, read, and construct the model

Use `src/aggregate/_loss_table.py` for shared readers, exact identifiers,
provenance, and audit records, with separate ELT and YLT preparation functions.
`Underwriter._resolve_elt` and `_resolve_ylt` run beside `_resolve_sev_ref`
before aggregate construction. Parsing remains free of I/O. Preparation
reads each source once, validates the combined input, and retains an immutable
snapshot of its numeric arrays and identifiers. A later update uses that
snapshot; only a fresh recipe build rereads files.

Deterministic ELTs and YLTs can already become exact discrete atoms during
preparation. For an ELT6 with active secondary uncertainty, prepare a model
with its distinct-event rate, first three severity raw moments, effective
support, zero-mass information, numerical settings, and diagnostics. Do not
allocate a bucketed severity or choose a bucket during construction.

For ELTs, stable event/unit codes and whole-event ordering are computed once.
`no2u` projects ELT6 to ELT4 before form compatibility is checked, retaining
original-form provenance. A subsequent ELT3 combination drops maximums.
For an entirely deterministic declaration, including
all-`no2u` ELT6, sum means within events and coalesce equal losses with their
rate weights. This path requires no eltq import.

### [ELT-Moment-Pass] model information before discretization

Use the row-law formulas above for single-unit events, with the same atom
and Bernoulli classification used by eltq. Compute rate-weighted raw moments
$C_r=\sum_e\lambda_e\mathsf E[X_e^r]$ and pass $C_r/\Lambda$ as severity
moments. Poisson compounding then gives annual cumulants $C_1,C_2,C_3$ through
the existing moment machinery. These are moments of the effective model,
including `no2u` and inadmissible-variance treatment, not sums of unchecked
input SDs or moments from the final density.

For shared events, compute the total's moments under the declared coupling
without a loss grid. Conditional on a factor value $z$, let
$a_{ur}(z)=\mathsf E[X_u^r\mid Z_c=z]$. Starting with $t_1=t_2=t_3=0$, add
one conditionally independent unit using simultaneous updates:

$$
t_1'=t_1+a_{u1},\qquad
t_2'=t_2+2t_1a_{u1}+a_{u2},\qquad
t_3'=t_3+3t_2a_{u1}+3t_1a_{u2}+a_{u3}.
$$

Integrating each $t_r(z)$ against the common-factor law gives the event's
raw moments. Conditional unit moments are one-dimensional integrations of
the fitted row laws, with direct formulas for atom and Bernoulli cases.
The recursion is linear in the number of units at each factor evaluation;
there is no enumeration of all pairs and triples and no unit-by-loss grid.
A stratum-based implementation may integrate conditional moments over each
stratum, but must identify and control its approximation to Burke's law.

The shared-event second and third moments are numerical model moments.
Record their method, tolerance, refinement comparison, and convergence
status. Compare them with independent high-accuracy small cases. Keep exact
AAL and single-row marginal moments as checks. Refine the moment calculation
independently of the eventual loss bucket; a change in `bs` or `log2` must
not change the theoretical moments. Inadequate numerical moment convergence
must be reported, not replaced by a rolled beta or invented analytic skew.

**Upstream prerequisite:** the inspected eltq 2.10.0 public surface does not
provide this moment-only pass. Propose `eltq.moments_arrays` and
`eltq.multiunit_moments`, names checked as unused, returning rate, raw
moments, effective support, zero-mass data, and row/integration diagnostics.
These are new APIs to implement and release, not existing calls to depend
on. They reuse eltq's row classification and numerical kernels. Do not
copy its beta or copula integration into aggregate, import private reference
helpers as a public dependency, or run a coarse hidden quantization as the
moment pass. The returned model summaries should permit aggregate's
construction to stay linear in table size apart from the stated integration.
Benchmark the shared-event pass: being independent of grid size does not
make nonlinear cross-moment integration free.

### [ELT-Severity-Protocol] a model-bearing severity

Introduce `SeverityElt`, following the repo's `Severity<Kind>` convention,
with an internal `sev_kind='elt'` and a private `_elt_model` constructor
payload. The public entry remains `agg ... elt ...`; there is no additional
`sev elt` or `dsev2u` syntax in this plan. The result remains an ordinary
`Aggregate`, with one severity component and ordinary Poisson frequency.

`Aggregate.__init__` passes the prepared model into the severity factory and
records `SeverityElt.moms()` through its existing component/moment code.
Keep `sev1`, `sev2`, and `sev3` as model moments throughout the object's life.
Supply the genuine finite effective severity support to the tail and bounded
window interfaces. No integer lattice is claimed for an uncertain beta
mixture, even when all input currency amounts happen to be integers.

Implement explicit initialization hooks for this severity. The ordinary
`Severity` constructor currently calls `_build`, splice/reflection checks,
and attachment/layer setup; some of those paths query a CDF. An ELT body
has no severity-layer, splice, reflection, or conditioning syntax, so its
initialization should install that identity contract directly, with valid
support, moments, and zero-mass metadata. It must not fabricate a temporary
histogram or moment-matched scipy law to get past those calls.

The first-release protocol is deliberately explicit:

| Operation | Before `update` | After `update` |
|---|---|---|
| `moms`, mean/CV/skew, support, boundedness, input audit | Available from the prepared model, with numerical-moment status where applicable. | Unchanged by bucketing. |
| Aggregate severity/annual density, grid CDF and quantiles | Unavailable, as for an Aggregate without a density. | Supplied by the existing Aggregate grid interfaces. |
| ELT window-tail queries and bounds | Specialized upstream queries become available at sizing time, with method and error metadata; no hidden loss grid at construction. | Available to validate the selected window independently of its bucketed density. |
| General direct ELT-severity CDF/SF/PPF or PDF integration | A full scipy-style interface is not required; unavailable operations give a targeted message. | A supported grid-backed reading is identified as numerical and uses the current grid. |
| Exact discrete support atoms | Available for the existing deterministic `dsev` path. | A discretized uncertain density does not turn the underlying model into an exact discrete severity. |

Audit consumers of `self.sevs`, `fz`, and `support_atoms`, including tail
reporting, `_bounded_severity_window`, `_single_big_jump_window`,
`feasible_bucket`, plotting, severity error analysis, and `sharpen`.
Moment/support candidates and specialized tail checks precede final quantization.
A diagnostic needing
a continuous CDF that this provider cannot supply reports unavailable
rather than evaluating against a made-up law. Where existing Aggregate
methods can use its computed severity density, route them there after update.
This capability handling is necessary implementation work, not an assumption
that changing `moms()` alone satisfies the full scipy severity interface.

### [ELT-Source-Grid] the standard update chooses the grid

There is one selected bucket spacing. Use the existing sequence:

1. Constructor: normalized table snapshot, model moments, support, and rate.
2. `update`: resolve the ELT grid budget and ordinary `bs`/window settings,
   then call the existing `_bs_window` / `_bucket_window.bs_window` machinery.
3. Check the candidate's event-tail probability and moment loss. Feed any
   required extension back into the existing sizing calculation; do not
   extend automatically to the largest exposure.
4. `update_work`: establish the actual physical severity grid `xs_sev`.
5. `discretize`: call eltq at that spacing and explicit length, check its
   numerical error and overflow, and divide the rate-weighted severity once
   by the distinct-event rate. An automatic-grid retry discards the previous
   attempt and records why the grid changed.
6. Continue with existing occurrence reinsurance, frequency transform,
   aggregate reinsurance, and validation.

`update=False` computes model statistics without quantizing or building an
annual density. With automatic `bs`, ordinary aggregate moments drive the
existing moment-window candidate. True support remains available to the
bounded candidate, which competes with the moment candidate as it does now.
The tail checks below supplement this selection. Explicit
`build(..., bs=..., log2=...)` and `hints{bs=...; log2=...}` retain their
current precedence, subject to the documented ELT resource ceiling.
There is no separate `elt_bs`, source-lattice snapping, or mandatory hint.

Extend `Aggregate.discretize` with a small `SeverityElt` dispatch to its
model quantizer; ordinary severities keep `discretize_severities` unchanged
unless a shared hook is cleaner. Return the expected list of component
PMFs so existing frequency weighting and downstream calculation remain in
place. Do not replace `self.sevs` or theoretical `stats_df` with the computed
histogram after the first update.

For one unit row per event, use `eltq.quantize_arrays` with
`method='survival'`, `normalize=False`, explicit `bucket_size` and
`grid_length`, and the controlled `top_bucket` policy described below.
`overflow='error'` is a full-support check and is unsuitable as the default
for a deliberate window. Shared events use `multiunit_views`, `terms=None`,
`views=('ground_up',)`, `normalize=False`, and Burke's coupling, after the
required upstream windowing extension. Naturally zero-SD ELT6 rows remain
atoms in this route; a mixture of converted `no2u` ELT4 and active ELT6 is
rejected before reaching it.

The severity and annual grids share `bs`. A shorter accepted severity window
at this spacing may be padded once into the annual array. This is a storage
optimization, not a second resolution. Its overflow remains measured at its
own edge. Returned rate mass must agree with $\Lambda$ before division.
Release the univariate API's unused raw array promptly. A secondary-only
output is an optional memory optimization, separate from the required APIs.

### [ELT-Support-In-Sizing] exposure, numerical extent, and tail error

**Verified behavior, 2026-10-26.** Read
[ELTQ-implementation.qmd](D:/Projects/aggregate/secondary-uncertainty/ELTQ-implementation.qmd),
particularly `sec-large-exposure`, before inspecting the current local
eltq 2.10.0 sources. The article explicitly describes large schedules with
small means and the introduction of windowed grids in 2.0.0. The current
sources establish these distinct facts:

| Surface | Current behavior | Adapter consequence |
|---|---|---|
| `quantize_arrays` and `ExceedanceCurve.grid` | The caller supplies spacing. Omitted length covers the largest contributing support; an explicit shorter length uses `top_bucket`, `drop`, or `error`. | Always supply the selected length; eltq is not an automatic bucket-size selector. |
| Univariate overflow metadata | `rate_above_grid` is rate beyond the final bucket edge; `mean_above_grid` is its first moment in excess of the top node. | Read both with their actual meanings; exact retained rate does not imply an accurate mean. |
| `curve.rs:NARROW_STEPS=40`, `narrow_bulk` | A narrow-event integration search moves outward in SD steps, looking for the rate-floor criterion or a support endpoint. Failure within 40 steps sends the event to the regular route. | This is an internal method-selection limit, not a universal $\mu+40\sigma$ cutoff or a rule for `bs`. |
| `curve.rs` tail walk | Stops at the selected grid or a small rate floor, currently $10^{-20}$ times the beta-event rate. | A local work-saving floor is not the aggregate window's probability or moment-error contract. |
| `views.rs` and `reference_multiunit.py` | Shared-event grids currently require the full sum of rounded unit support spans and allocate full-support unit PMFs. A short explicit grid fails; there is no window overflow argument. | Add upstream shared-event windowing before using this route on large schedules. Resizing only its final output is insufficient. |

The article's implementation account predates some local changes; the code
checks above establish the current API contract. Relevant source locations
are `python/eltq/__init__.py`, `python/eltq/_meta.py`, and
`crates/eltq-core/src/{quantize,curve,deposit,views,multiunit}.rs` in
`V:/dev/eltq`.

For each positive-rate event retain its true support

$$
B_e=\sum_u b_{eu},\qquad B=\max_e B_e,\qquad
b_{eu}=\begin{cases}
\mu_{eu},&\text{an effective atom},\\
M_{eu},&\text{an uncertain row reaching }M_{eu}.
\end{cases}
$$

An ELT4 maximum is reporting metadata, not the deterministic atom's support.
A `no2u` conversion likewise has loss at its mean. Zero-rate rows do not
enlarge computational support. For active ELT6, retain the original $M$ in
the beta/Bernoulli law; substituting a numerical cutoff for $M$ would change
the distribution and its moments.

Expose $B$ to structural tail reporting and the existing bounded-window
candidate, but **do not make coverage of $B$ a mandatory grid constraint**.
A bounded candidate helps when genuinely tighter than the moment window.
Compound Poisson annual loss remains unbounded even when event loss is
bounded. A fitted moment window alone is also not evidence that the event
tail has been resolved. Distinguish true support, selected numerical
extent, and measured error in both bucket and loss-table reporting.

**Proposed numerical contract.** Use moments to select the initial grid,
then accept a shorter window only with quantitative tail checks. Let $c$
be its top node and $t=c+h/2$ its upper bucket edge. Define

$$
R(t)=\sum_e\lambda_e\mathsf P(X_e>t),\qquad
D_r(t,c)=\sum_e\lambda_e\mathsf E[(X_e^r-c^r)\mathbf1_{\{X_e>t\}}],
\quad r=1,2,3.
$$

Under top-bucket overflow, $D_1$ is the AAL lost by moving the overflow to
$c$; $D_2,D_3$ are the corresponding losses in compound-Poisson cumulants.
The probability of at least one overflowing event in a year is
$1-\exp(-R(t))$. These describe overflow separately from the ordinary
within-window rounding and interpolation error. Atoms in $(c,t]$ are also
placed at $c$; include that boundary rounding in discretization checks.

Reuse existing accuracy settings where appropriate: initially use
`discretization.window_nines` for the annual overflow-probability target,
and `validation.eps` for the relative moment-error budget, with a proposed
allocation of at most one tenth of that budget to overflow in each $D_r/C_r$.
The inspected defaults are 12 nines and `eps=1e-4`. These are implementation
proposals to calibrate with fixtures, not claimed accuracy of existing eltq.
Use stable rate-space calculations rather than subtracting a CDF from one;
handle zero $C_r$ explicitly. Report achieved values and bounds, including
integration uncertainty. Do not reuse an annual materiality threshold as
an unqualified event-tail tolerance.

Single-unit beta tail moments can be computed without a loss grid:

$$
\mathsf E[X^r\mathbf1_{\{X>t\}}]
=M^r\frac{(\alpha)_r}{(\alpha+\beta)_r}
 \overline I_{t/M}(\alpha+r,\beta),
$$

where $(a)_r$ is the rising factorial and $\overline I$ the beta survival
function. Atoms and Bernoulli rows have direct expressions. Reuse stable
upstream kernels and preserve explicit endpoint atoms. The existing
exceedance curve and first-moment overflow diagnostics provide much of the
univariate machinery; add public higher-moment/tail diagnostics as needed,
rather than copying eltq's numerical engine into aggregate.

For shared events, use the sum's law under the selected coupling or a
conservative bound with stated error. Do not replace it by a beta fitted
to the sum. Marginal union bounds can safely bracket a candidate, but a
failed loose bound is not evidence of material tail loss. Refine a loose
bound or evaluate the coupled tail before forcing a vast window. Supply
this diagnostic through the upstream extension below. Higher-moment
checks are essential: a very small tail rate can still carry material AAL,
variance, or skew. Checking only the retained mass or AAL is insufficient.

The ordinary selector owns the final grid. If a candidate fails tail checks,
extend its physical extent, grow the bucket count within the effective
`log2` budget where possible, or coarsen `bs` and recheck body resolution.
Retain all candidate diagnostics, but not all large arrays. At the resource
ceiling, report the unmet accuracy and required change instead of silently
accepting it. A pinned grid that fails the specified tail contract raises
an actionable error. Accepted small overflow is explicitly reported and
put in the top bucket to preserve the Poisson rate; do not silently drop
it, change the frequency, or renormalize away evidence. A deliberately
looser target can use existing numerical settings and must remain visible.
Check the physical `xs_sev` grid, not a shifted annual display coordinate.
Annual FFT truncation and aliasing retain their separate existing checks.

**Regression examples.** The article's $\mu=20{,}000$, $\sigma=60{,}000$,
$M=10^9$ row should need a window far below $M$ at suitable accuracy.
A second valid beta with $\mu=100$, $\sigma=10{,}000$, $M=10^9$ has
$\alpha\approx9.99\times10^{-5}$ and $\beta\approx999$. Above
$\mu+40\sigma=400{,}100$ it still has probability about
$7.024\times10^{-5}$ and **67.05% of its mean**. These values were checked
using beta survival and partial-moment formulas. Neither a blanket
40-SD rule nor an assumption that small damage ratio makes the tail
immaterial is acceptable. The retained model and measured error decide.

### [Eltq-Windowed-Shared-Events] upstream prerequisite

Extend `multiunit_views` and both supported backends to accept an explicit
window with a named overflow policy and diagnostics matching the univariate
semantics. Keep existing full-support behavior when no window is requested.
This is a coordinated prerequisite, alongside the grid-independent moment
APIs, not a capability of eltq 2.10.0.

Bound conditional unit PMF storage by the selected window and compute the
retained prefix of the event-total convolution with sufficient FFT padding
or another alias-free algorithm. Track the event-total overflow and its
first three moment effects separately. Apply the overflow policy to the
event total: capping each unit at the final event cap and then summing
would produce the wrong upper tail. Nonnegative losses permit a truncated
prefix calculation, but omitted mass and its moments must still be
accounted for. No intermediate array may secretly extend to $M/h$.

Expose a grid-independent tail query, rigorous bound, or adaptive tail
calculation with a documented error estimate for window selection and
validation. Factor quadrature error remains distinct from window error;
32 factor nodes do not certify a 12-nines tail target. Refine or bound the
factor integration as needed and report failure to achieve the target.
Validate windowed output against full-support small cases and independent
joint references, including atom endpoints, mixed atom/beta ELT6 rows,
large exposures, and off-grid sums. Full-support comparisons still use
one plus the sum of per-unit rounded support spans, not the ceiling of
their summed monetary maximum.

### [Loss-Table-Large-Grids] budget, memory, and refresh

Add `DiscretizationSettings.elt_max_log2` and its documented default in
`data/config.default.toml`, following the configuration pattern already
used by `_bucket_window`. A module-level binding, if needed, follows the
existing uppercase convention `ELT_MAX_LOG2`; the public setting is
`elt_max_log2`. Proposed initial value: **24**, subject to the synthetic
memory/performance study. It is a maximum exponent, not a demand to
allocate $2^{24}$ buckets for every table.

Use this setting as the default automatic ELT budget. Preserve explicit
caller, recipe-hint, and underwriter `log2` precedence; an explicitly smaller
budget is respected. No ELT route may exceed the configured ceiling,
including automatic window growth or `sharpen`; an explicitly larger
request explains how to raise `elt_max_log2` instead of being silently
clipped. Distinguish an omitted `log2` from an explicit value of 16.
`Underwriter` already uses `_UNSET` at construction, but `Aggregate.update`
currently defaults directly to 16; carry explicitness through these paths
without changing ordinary aggregates' default. Ordinary YLT sizing keeps
its existing settings and can also request large grids.

There is no second smaller source-grid cap or mandatory hint. Production
examples use 22, with checks at 20, 22, and 24 where memory permits. Support
larger requests when the global ceiling is raised and memory permits.
One float64 array uses 8 MiB at 20, 32 MiB at 22, and 128 MiB at 24.
Estimate peak working memory using all live severity, padded FFT, aggregate,
reinsurance, and worker buffers; one array's size is not the process budget.
Preflight before allocation and report the grid, estimated bytes, and the
operation needing them. Avoid event-by-grid and unit-by-grid dense matrices,
pandas output frames merely to compute moments, and unnecessary copies.
Bound backend worker/chunk memory, including the unused raw output.

Retain at most the last reusable event-severity result per object, keyed by
the immutable input/model identity, physical grid, overflow policy, and
integration settings. Changing `bs`, grid length/origin, or model settings
invalidates it. Each new grid in `update` or `sharpen` re-quantizes from the
original model, so refinement can recover detail and previously windowed
tails. Clear dependent gross/ceded severity and joint caches as well; an
old `sev_density_gross` must not overwrite the new grid. Avoid accumulating
an array cache entry for every sizing or convergence probe.

Keep `elt_nodes_per_decade`, `elt_nodes_per_sigma`, and `elt_z_nodes` as
optional hints separated from ordinary update kwargs. Add `elt_moment_rtol`
for grid-independent moment integration, initially targeting `1e-6`
relative agreement in nondegenerate second and third raw moments. Set its
final documented default from acceptance evidence, with achieved-error
reporting and a zero-scale absolute criterion. Source-construction settings
change on rebuild; changing only the annual grid does not change model
moments. Unknown `elt_` hints fail with a suggestion. YLTs reject ELT
integration controls. Prefer the existing numerical settings to a growing
collection of overlapping ELT-only accuracy knobs.

### [YLT-Materialization] reuse the discrete annual path

`Underwriter._resolve_ylt` aligns annual scenarios and their probabilities,
sums unit losses within each year, then coalesces equal totals by adding
scenario probabilities. It constructs the existing `dfreq [1] dsev` spec:

```python
{
    'freq_name': 'empirical',
    'freq_a': [1],
    'freq_b': [1.0],
    'exp_en': -1,
    'sev_name': 'dhistogram',
    'sev_xs': distinct_annual_totals,
    'sev_ps': probability_by_distinct_total,
}
```

`exp_en=-1` is the existing empirical-frequency sentinel; the expected
count derived from it is one. Preserve zero-loss probability and leave
`exp_attachment` unset. Add labels, aggregate reinsurance, and ordinary
trailers. The exact weighted empirical moments are already available
before any grid is built. Zero-probability scenarios do not enlarge the
computed support, though their identifiers remain in the alignment audit.

`Aggregate.one_claim` should be true. Its existing severity-copy path gives
the annual law on the selected grid before aggregate financial terms.
Use ordinary `bs` and `log2`, with no eltq call, secondary-uncertainty
integration controls, or new YLT frequency/distribution class. Unlike the
ELT6 model, the empirical YLT is fully described by its atoms, so repeated
updates simply re-discretize those same exact atoms.

### [Loss-Table-Moment-Reporting] theory and estimates

For deterministic ELTs and YLTs, theoretical moments are exact weighted
sums over input atoms. For uncertain ELTs, they are moments of the prepared
row/coupling model, analytic where possible and explicitly numerical for
shared-event integration. They remain fixed as grid resolution changes.

Normal Aggregate estimated severity and annual moments come from the
computed arrays and are compared with those theoretical moments. The
loss-table audit adds model-moment convergence, effective-row replacements,
and eltq quantization diagnostics. Do not describe moments computed from
the same final histogram as an independent theoretical check. Continue to
separate moment-integration error, event-window overflow, severity
discretization error, and annual FFT error. Passing one layer does not
establish accuracy of the others.

## Audit, persistence, and dependencies

### [Loss-Table-Audit] public reporting

Follow existing Aggregate naming and reuse established suffixes:

- `loss_table_summary_df`: input kind, source paths/digests, source and
  effective forms, conversions, units and labels, row counts, and combined
  totals. ELTs add distinct-event counts/rates, effective 2U status,
  appropriate exposure summaries, backend/version/settings, model support,
  and the selected grid after update. YLTs add complete year count,
  alignment mode, probability convention, zero-year count/probability,
  and distinct annual outcome count. A source row and an event are different
  counting units; name summary totals accordingly.
- `loss_table_stats_df`: the additional comparisons needed to understand
  the loss-table conversion. Compare source/effective-model moments,
  discretized severity moments, and annual results, reusing existing
  `stats_df` values instead of duplicating their calculation. Identify
  analytic model values, numerical model moments, and grid estimates.
  Include overflow's effects on the first three compound-Poisson cumulants
  and distinguish these from in-window discretization and annual FFT error.
- `loss_table_validation_df`: categorized checks, measured values or bounds,
  tolerances, pass/warning/failure/not-yet-computed status, and a bounded
  number of source/event/unit or year/unit examples. Include form projection,
  rate/probability agreement, year completeness, row-law replacements,
  moment integration convergence, window overflow, and grid moment errors.
  Invalid input still raises before building a partial portfolio; attach
  concise source context rather than returning a misleading valid object.
- Existing `Aggregate.info`: add a short string summary of kind, effective
  form, event/year and unit counts, 2U status where applicable, and whether
  only the model or also its density is available. Reuse the current `info`
  property and formatting conventions; do not add another info API.

Keep normalized internal metadata in `_loss_table_meta` and derive these
views from it. No public `loss_table_diagnostics` record or new audit suffix
is needed. For a non-loss-table aggregate the new frames are empty with
documented columns and ordinary `info` remains applicable. Return defensive
copies. Do not create event-by-grid frames, duplicate huge source tables in
reports, or expose megabytes of metadata through repr, errors, or tooltips.

For YLTs, source moments are exact weighted moments of the annual law,
including population variance. Event rates and eltq fields are inapplicable.
Zero-year count and probability are different for YLT3. Annual grid error
is separate from the empirical model; sample uncertainty is not a numerical
defect. Before `update`, show the model statistics and mark grid-based checks
as not yet computed, never as passed or zero error.

For effective ELT4/6, an event exposure summary sums unit maximums within
that event. It is not total insured value across a catalog, because the
same property can be exposed to many events. Label a maximum across event
exposures separately from their sum. ELT4 maximums and model support remain
distinct. `no2u` retains the maximum in its ELT4 projection, but if any ELT3
source is combined with it the entire effective maximum column is dropped.
Report that projection; do not publish a partial maximum total. Source-form
provenance need not be discarded when fields leave the effective model.

Report normalization and moment checks without normalizing the evidence
away. Use the existing `validation.eps` (currently `1e-4`) for source moment
comparisons, with the separately budgeted window contribution and moment
integration tolerance described above. Calibrate against fixtures before
release. For zero AAL verify concentration at zero with a scale-aware
absolute check. Warnings identify the appropriate remedy: a larger grid
budget, a wider window, finer buckets, or tighter integration. Increasing
`log2` does not itself extend a pinned monetary window. Do not claim a
return-period or tail guarantee merely because the AAL check passes.

### [Loss-Table-Reproducibility] recipes, model snapshots, and densities

`Recipe.spec` and canonical DecL retain the sources, modifiers, labels, and
numerical hints. A fresh build rereads files and prepares the model; an existing
Aggregate retains its immutable input/model snapshot and does not reread on
`update`. ELT6 updates quantize that model on the newly selected grid, while
deterministic ELT/YLT updates discretize their original weighted atoms.
Record a digest of the input bytes or canonical table content, the resolved
path, and file metadata at build time. A path alone does not freeze data.
Do not introduce a process-wide file or distribution cache in this release.

Ordinary object copying and `Aggregate(**obj.spec)` must preserve the
prepared model, theoretical moments, and compact audit. Use `_elt_model`
for the immutable ELT6 model payload and `_loss_table_meta` for reporting
metadata. Reconstructing a prepared object must not open files or require
that they still exist. Copying can share read-only input arrays; mutable
grid caches must not be shared. Avoid dangling references to a mutable
DataFrame or to an underwriter. Source recipes and prepared constructor
specs are deliberately different representations.
The writer must choose explicitly which it renders: normal recipe export
emits `elt` or `ylt` as appropriate. A deliberate export of a computed
severity as `dsev` freezes that approximation and is a separate operation;
ordinary reconstruction must retain the ELT model and its ability to refine.
Use the model's content digest in spec hashes and compact summaries, and
define serialization of the prepared payload without turning its repr into
a hidden serialization format. Keep routine export compact.

An ELT- or YLT-backed aggregate referenced as `agg.NAME` keeps the recipe's
source base and modifiers. `sev agg.NAME` still means the computed annual output
of that aggregate, with the existing requirement for its annual `bs` and
`log2` hints; it does not mean the raw event severity. Portfolio and P&L
construction must carry loss-table provenance through the shared aggregate body.
The same object can be used downstream like an ordinary aggregate, but a
total-only severity does not recover the original unit decomposition.

### [ELT-Dependency] packaging

Recommend an optional `elt` extra declaring a compatible tested eltq version
and the selected columnar-file engine. Pin the lower bound to the release
that provides the required public moment/tail diagnostics, and for shared
events the windowed multi-unit route. The inspected 2.10.0 is the starting
point, not yet a sufficient dependency for this design.
Choose the upper compatibility bound at that release. Include the extra in the user-facing
`all` extra. Import eltq lazily at the first secondary-uncertainty build;
ordinary imports, parsing, ELT3/4, an entirely `no2u` build, and YLT builds
must work without it. YLT requires only existing core dependencies apart
from any optional file-reader engine; there is no YLT extra. A columnar
reader can be installed independently of the `elt` extra. Reader-specific
requirements remain separate errors.

Both packages currently target Python 3.12 and later. eltq's Rust extension
and SciPy reference fallback are upstream responsibilities. Record its
reported backend; do not build another fallback in aggregate. Run numerical
integration fixtures against the supported backend and a small reference
sample where available. Do not silently substitute a different model if an
optional dependency is unavailable.

Use `uv` for dependency changes. In this checkout, sync all extras with
`--inexact` when preserving the discovered `aggregate-ntt` installation;
never sync a single extra and prune the rest of the development environment.
Implement the moment/tail APIs and shared-event windowing as coordinated,
separately tested eltq changes before the dependent aggregate stages. Their
releases are prerequisites recorded below. This planning task changes neither
repository's code.

### [Loss-Table-Names] collision audit

Planning-time `rg` checks found no existing definitions or spec keys for
`elt_sources`, `_resolve_elt`, `ylt_sources`, `_resolve_ylt`, `_loss_table_meta`,
`loss_table_summary_df`, `loss_table_stats_df`, `loss_table_validation_df`,
`elt_nodes_per_decade`, `elt_nodes_per_sigma`, `elt_z_nodes`, `elt_moment_rtol`,
`elt_max_log2`, `ELT_MAX_LOG2`, `_elt_model`, or `SeverityElt` in
`src/aggregate`. Recheck against the current surface before landing.
`info` already exists and is extended in place. The `_loss_table.py` path
was also checked. Proposed upstream `moments_arrays` and
`multiunit_moments` were checked in eltq and are unused; choose any additional
public tail-diagnostic name only after checking its existing API.
No `elt_bs` is introduced.

`decl_writer._elt_div` already exists and means elementwise division; it is
unrelated and keeps its name. `unit`, `data`, and `peril` occur in prose
and other APIs; this does not justify ambiguous generic Aggregate attributes.
Only `unit` becomes a keyword here, alongside `elt`, `ylt`, and `no2u`.

## Implementation sequence

Three aggregate implementation steps plus the eltq numerical prerequisites,
each with the appropriate repository's version and release hygiene.
Choose the next available versions when work starts;
do not reserve version numbers from the current checkout. Move this plan
into `dev/done/plan-elt.md` in the commit that lands its first step and keep
the execution log there thereafter.

### [Loss-Table-Deterministic-Import]

Implement the source specs, ELT and YLT body grammars, exact event/year IDs,
quote-safe preprocessing, file normalization, ELT3/4 grouping and Poisson
lowering, YLT1/2/3 year/probability alignment and empirical-frequency lowering,
and source provenance. Add the writer and composition paths together, including the
annual-only reinsurance check for YLTs. Introduce the three loss-table
report frames and the existing `info` extension, including ELT form
projections and YLT year/probability summaries. Introduce `elt_max_log2`
and explicit/implicit grid-budget handling. YLT is complete in this step
and does not wait for secondary uncertainty or an eltq install.
Accept `no2u` ELT6 through its validated ELT4 projection in this step; this
requires no uncertainty engine. Active ELT6 construction gives a specific
unsupported message until the next step lands.

Acceptance: basic mean and variance, shared versus distinct event IDs,
ELT4 distributional equivalence, YLT equivalence to `dfreq [1] dsev`, repeated
loss multiplicities, unequal scenario probabilities, zero years,
aligned-unit dependence, completeness errors, annual reinsurance,
file/inline parity, duplicate and rate errors,
exact ID preservation, quoted paths, round trips, portfolio and P&L embedding,
and compatibility of existing corpora.

### [Eltq-Model-Moments-And-Windows] upstream prerequisites

Add and validate the moment-only APIs described in `[ELT-Moment-Pass]`,
effective support/classification diagnostics, and single-unit tail-moment
queries needed for window checks. The single-row path must agree with
closed-form atom, beta, and Bernoulli calculations;
the shared path must agree with independently integrated small couplings
and report its convergence. Prove that it creates no loss grid and that
its outputs are independent of future `bs` and `log2`. Benchmark large-table
cost and bounded working memory. Release the APIs before the aggregate
adapter declares the dependency. This work can proceed alongside the
deterministic ELT/YLT implementation.

Before the shared-event aggregate step, complete
`[Eltq-Windowed-Shared-Events]`: bounded intermediate grids, event-total
overflow and moment diagnostics, coupled-tail error control, Rust/reference
parity, and an upstream release. Do not ship a purportedly windowed adapter
which still allocates the full unit maxima internally.

### [ELT-Single-Event-Uncertainty]

Add the optional dependency, model preparation, `SeverityElt`, standard
bucket sizing with measured tail checks, `elt_max_log2`, update-time
quantization, moment/overflow reporting, and model-snapshot copying/persistence.
Support ELT6 where every event has one unit row, even if the declaration
contains many units. A shared event gives
a clear message until the final step lands.

Acceptance: known beta and Bernoulli moments, degeneracies, zero atoms,
`no2u` parity with deterministic atoms, mass/rate agreement, fixed theoretical
moments across grid changes, no quantization before update, automatic sizing
without hints, huge-$M$/small-mean windows, material thin-tail rejection,
missing dependencies, and standard reinsurance on the computed severity.
Choose and document final source-warning tolerances from fixture evidence.

### [ELT-Shared-Event-Uncertainty]

Add the multi-unit moment and windowed quantization routes, Burke coupling,
stable whole-event grouping, shared-event metadata, and joint-reference
acceptance cases. Enforce effective-form compatibility before dispatch.
Complete documentation and larger synthetic timing/memory checks.
Automatic sizing through the existing routines is part of both
uncertainty stages, not a deferred alternative.

Acceptance: one event counted once, naturally zero-SD rows within ELT6,
rejection of active ELT6 mixed with `no2u` projections, sparse missing-unit
rows, single-unit route parity at the stated accuracy, and the sum of a
two-unit occurrence joint compared with the
one-dimensional event-total route. Check moment integration, chosen bucket
spacing, factor strata, and annual-grid coverage separately. Run the large
grid tests and verify bounded cache memory. Keep bivariate output deferred.

## Files and coordinated updates

| Area | Files or surfaces |
|---|---|
| Parsing and rendering | `src/aggregate/decl.lark`, `parser.py`, `parser_errors.py`, `decl_writer.py`, `agg.sublime-syntax` |
| Construction and source hints | `underwriter.py`, `recipe.py` if source-base retention needs it, new `_loss_table.py` with ELT and YLT adapters |
| Severity, moments, discretization, and sizing | `_aggregate.py`, `_severity.py`, `_bucket_window.py`, `tail.py`, and `_aggregate_compute.py` only if the discretization hook belongs there |
| Grid configuration | `config.py`, `data/config.default.toml`, configuration help and explicit/implicit `log2` handling |
| Dependencies | `pyproject.toml`, `uv.lock` |
| Tests | New `tests/test_elt.py` and `tests/test_ylt.py`, focused cases in parser, ambiguity, unparser, recipe, underwriter, bucket, and composition suites |
| Fixtures | Small synthetic input files under `tests/data/elt/` and `tests/data/ylt/`; inline examples in the shipped DecL corpus |
| Documentation | Parser reference, aggregate input guide, DecL help, worked ELT and YLT examples using the master bibliography where cited |
| Release | `src/aggregate/__init__.py`, `CHANGELOG.md`, live `dev/TODO.md`, `dev/FEATURES.csv` via `dev/regen_features.py` |
| Upstream prerequisites | eltq public moment/tail diagnostics, shared-event windowing with bounded intermediate arrays, Rust/reference implementations, tests, and release metadata in its own checkout |

The derived DecL word list should pick up reserved words from the grammar.
Update any manually categorized declaration words and the checked mirrors,
especially friendly terminal labels and Sublime syntax. Regenerate the
grammar reference and inspect the spec snapshot diff after grammar changes.
Consumers outside aggregate refresh their generated keyword resources after
release; no consumer-app changes are included. The explicitly scoped eltq
moment/tail and windowing work is the numerical dependency, with its own releases.

## Acceptance and verification

### [YLT-Probability-Oracles]

Check exact empirical moments and equivalent hand-authored `dfreq [1] dsev`
on the same annual grid. The internal `one_claim` predicate must be true.

| Case | Required result |
|---|---|
| `ylt [0 0 100 300]` | Atoms $0,100,300$ with probabilities $1/2,1/4,1/4$; mean 100; variance 15,000. |
| Same input as YLT2, uniform YLT3, or a one-column file | Same distribution and moments; all four years counted. |
| Same input with `aggregate net of 150 xs 50` | Annual net outcomes $0,0,50,150$; mean 50; variance 3,750. |
| `ylt [[1 0 .5] [2 100 .3] [3 300 .2]]` | Mean 90; variance 12,900; zero probability 0.5. |
| Same YLT3 with `aggregate net of 150 xs 50` | Net outcomes $0,50,150$, probabilities $.5,.3,.2$; mean 45; variance 3,225. |
| YLT3 losses $0,100,100$ with probabilities $.5,.3,.2$ | Coalesced probabilities $.5,.5$ at $0,100$; mean 50; variance 2,500. |
| Units $[0,100]$ and $[100,0]$ in the same two years | Total always 100, variance zero. Independent convolution would give variance 5,000 and must not occur. |
| Two units in years with nonuniform probabilities | Sum each year's losses, keep its probability once, then coalesce. Do not add or multiply probabilities across units. |
| A zero-probability year with an extremely large loss | Retained in the complete year set, excluded from moments and effective support. |
| A single year of loss 75 | Atom at 75; mean 75; variance zero, with normal degenerate-statistic conventions. |
| Three explicit zero years | Atom at zero; three years in summary; no Poisson zero-count calculation. |
| Year IDs reordered independently in explicit-ID unit tables | Identical aligned total law. Reordering a positional source instead changes its declared pairing. |

Boundary tests cover missing years, equal row counts with different year
sets, null/negative losses, duplicate `(year_id, unit)`, mismatched vector
lengths, duplicate unit sources, empty tables, mixed ELT/YLT clauses,
`no2u`, and ELT integration hints. All-absent zero years are documented as
unknowable from the data, not claimed to be detected. Reject incomplete
source shards rather than silently defining a new year universe.

For YLT3, reject nonfinite/negative probabilities, all-zero probabilities,
materially nonunit sums, arbitrary weights, and probability disagreements
between units. Record accepted roundoff corrections. Uniform YLT3 can mix
with YLT1/2; nonuniform YLT3 cannot silently assign probabilities to them.
Check file/inline parity and probability aliases. Normalizing a sum near
one must not hide a material missing-year or missing-probability error.

Check annual reinsurance on plain and named YLT sources, P&L use with annual
terms, and rejection of occurrence clauses even through a named reference.
Without reinsurance, the discretized severity and annual output follow the
existing one-claim identity. Rebuilding after changing the YLT rereads it;
an existing object's update retains its weighted annual law. These
operations work without eltq. Round trips render `ylt` compactly, retaining
YLT3 probabilities and never synthesizing event rates or Poisson frequency.

### [ELT-Probability-Oracles]

Use independent small calculations as correctness oracles, with pinned
grids and tolerances that separate quantization error from FFT error:

| Case | Required result |
|---|---|
| Basic ELT3 in the examples | $\Lambda=0.3$, AAL $=22$, annual variance $=1800$. |
| Same table with supplied ELT4 maximums | Identical severity and annual law; maximums appear only in appropriate input reports. |
| Event 1 at rate $0.1$, unit means 100 and 40 | $\Lambda=0.1$, event loss 140, AAL 14, annual variance 1960. |
| Distinct events at rate $0.1$, losses 100 and 40 | $\Lambda=0.2$, AAL 14, annual variance 1160. |
| One ELT6 row: rate $0.1$, mean 50, max 100, SDs 10 and 10 | Beta parameters $2.625,2.625$; AAL 5; annual variance 290 before discretization. |
| Same ELT6 row with `no2u` | Atom at 50; AAL 5; annual variance 250. |
| Same mean/max with SD sum 60 | Bernoulli replacement on 0 and 100; AAL 5; annual variance 500; a replacement diagnostic. |
| Positive-rate event with mean zero | Its rate remains counted and its zero atom survives; the annual zero-mass formula agrees. |
| Entire positive-rate table at zero | Degenerate annual loss at zero with a finite, truthful audit. |
| Two shared units with ordinary atom or Bernoulli laws | Compare event-total moments and the annual law with a separately calculated finite joint. |

The published Homer example from MVMELT is a valuable integration fixture,
including its inadmissible rows, but it must use its stated event exposures.
Do not mix Table 3's event-1 exposure of 10,000 with Table 11's unit sum of
4,000. Keep that distinction in the fixture explanation. A high-resolution
joint or reference integration can check Burke totals; the main release
suite need only carry small deterministic fixtures, not rebuild the articles.

### [ELT-Contract-Tests]

Test each material boundary rather than every helper in isolation:

- Parser/writer round trips for ELT3/4/6 and YLT1/2/3, `no2u` where applicable,
  unit and aggregate labels, repeated sources, multiline matrices, and standard
  reinsurance. Confirm parsing never opens an input path.
- Missing columns, duplicate aliases, partial SD columns, ragged matrices,
  nonfinite or negative inputs, mean above maximum, empty data, conflicting
  rates, duplicate event/unit rows, and conflicting unit metadata.
- Integer IDs beyond float's exact range, leading zeroes, quoted string IDs,
  CSV/Parquet equivalence, and path punctuation including spaces, `#`, `//`,
  and brackets. A changed working directory must not redirect a recipe.
- ELT row permutations and disjoint-event file partitioning preserve the law
  within summation tolerance. Duplicate files fail. A unit absent from an
  event contributes zero without dropping the other units' event rate.
- ELT6 `no2u` plus ELT4 gives ELT4; adding ELT3 gives ELT3 and drops all
  maximums. All-ELT4 retains maximums. Active ELT6 plus ELT3, ELT4, or a
  converted ELT4 fails. Zero-SD values do not change an ELT6's form.
  Validate malformed original fields before projection. Round trips retain
  the authored modifier and sources while reports show the effective form.
- Without eltq, ordinary builds, parsing, ELT3/4, YLTs, and all-`no2u` ELT6 still
  work. An uncertainty build fails at the correct boundary with installation
  guidance. Bad data is not silently dropped by a backend.
- `update=False` reads and validates the tables and computes model moments,
  with no eltq quantization or annual density. Later updates quantize the
  preserved model. Rebuilding after a file change reads it again and records
  a new digest. Changing only the grid leaves theoretical moments unchanged.
- A copied object and `Aggregate(**obj.spec)` preserve the model and reports
  without rereading a file; mutable caches are isolated. Recipe export stays
  compact. `agg.NAME`, `sev agg.NAME`, portfolio use, and P&L use preserve the
  meanings and restrictions above.
- The three loss-table frames use the agreed suffixes and distinguish source,
  effective model, estimates, and not-yet-computed results. `info` remains a
  concise string. A mixed ELT3/4 summary cannot report partial maximums.
- Ordinary occurrence and aggregate treaties agree with the same manually
  supplied `dsev`. An occurrence treaty sees the sum across units; it must
  not be applied separately to each unit before summation.
- All existing special annual occurrence terms retain their current plain
  `agg` versus P&L behavior. This feature does not silently strengthen the
  capabilities of a plain `agg`.
- Source mass agrees with distinct-event rate to a scale-aware floating-point
  tolerance before normalization. AAL error is recorded and tested against
  the chosen source tolerance. An annual FFT mass deficit remains a separate
  existing validation result.
- Resource-limit failures occur before oversized allocation. Normal repr,
  help, and export remain bounded in size for a large file input.

### [ELT-Grid-And-Window-Tests]

- Automatic ELT sizing works without hints using moments available before
  quantization. A large $M$ with negligible relevant tail does not force
  coverage of $M$ or an allocation proportional to $M/h$.
- For single-unit rows, compare survival and raw tail moments with independent
  beta partial moments and exact atom/Bernoulli calculations. Include both
  large-exposure examples in `[ELT-Support-In-Sizing]`; verify the 40-SD
  counterexample is rejected by the error checks.
- A rare remote Bernoulli atom can have tiny probability and material first,
  second, or third moment. The window must retain it or fail the target;
  exact retained rate alone cannot pass. Do not lose an endpoint atom through
  a strict/weak inequality error.
- `top_bucket` retains distinct-event rate and reports overflow's probability
  and moment effects. Independent formulas distinguish first moment above
  the edge from first moment in excess of the top node. Test the final half
  bucket separately. No tail loss is hidden by normalization.
- Shared-event windows match full-support small references within separate
  discretization, factor integration, and overflow budgets. Their retained
  prefix has no circular wraparound. Sum-before-cap behavior differs from
  cap-before-sum as expected. Intermediate unit allocations respect the
  selected extent, even when each $M$ is huge.
- Extent changes and resolution changes have separate probes: widen at fixed
  `bs` to test overflow; refine `bs` at fixed monetary extent to test rounding;
  refine integration settings separately. Reusing a clipped histogram for a
  wider update must not prevent recovery of the original tail.
- Check `elt_max_log2`, explicit caller/hint/underwriter precedence, omitted
  versus explicitly supplied 16, small exact cases choosing fewer buckets,
  and an oversized request failing before allocation. Raising the setting
  permits a larger request. Ordinary non-ELT defaults are unchanged.
- Run representative synthetic grids at `log2=20,22,24` in the appropriate
  performance tier, gated by available memory rather than silently shrinking
  the requested grid. Verify reported peak memory includes backend and FFT
  temporaries. The fast suite uses smaller cases for the same contracts.
- Successive updates, automatic sizing retries, and `sharpen` retain bounded
  cache memory, recompute densities when the grid changes, and preserve model
  moments. Source window error and annual FFT aliasing remain separate checks.

### [ELT-Release-Gates]

Use focused test files in the edit loop. Finish each implementation step with
the full fast suite, and run the complete suite at its version-bump boundary.
For the numerical steps use the RuntimeWarning gate as required by the repo:

```powershell
$env:UV_PROJECT_ENVIRONMENT = '.venv'
uv run pytest
uv run pytest -m 'slow or not slow' -W error::RuntimeWarning
```

Preserve and restore any existing `UV_PROJECT_ENVIRONMENT` value when issuing
these commands in a shared shell. Do not repeatedly run both gates during
the edit loop. Performance evidence is a synthetic large-ELT wall-time and
peak-memory measurement against a direct eltq call with the same options;
set a regression budget from that baseline, not an invented speed target.
There is no quantizer call during model preparation and none for YLTs or
deterministic ELTs. A successful uncertainty update normally quantizes once;
measured sizing retries must have a recorded reason, bounded memory, and a
termination criterion. Avoid array-to-DecL parsing and
event-by-unit dense tables. A large YLT check measures read, alignment,
coalescing, and ordinary update cost; it has no eltq timing baseline.

Keep code and docs references in step. Do not build the documentation tree
during verification; record that the author will rebuild it. Regenerate the
parser snapshot and read its diff rather than treating it as a correctness
oracle. At each implementation bump, update the version, one-paragraph
CHANGELOG entry, applicable TODO and generated references, and commit and
push the coherent unit under the repo's standing release rules. This draft
alone is a documentation change, with no version bump or commit.

## Remaining implementation choices

The five clarification questions are resolved in the rulings above:
automatic moments-first sizing; table-local ELT6-to-ELT4 `no2u` with effective
form compatibility; ELT3/4 dropping maximums; complete equally sized YLTs
including zero years; and annual-only YLT reinsurance. YLT3 probabilities,
`elt_max_log2`, and reporting names are also settled. Do not reopen these
as implementation approval questions.

Concrete proposals still requiring numerical evidence are the initial
`elt_max_log2=24`, moment integration defaults, allocation of existing error
budgets to event-window overflow, and final upstream tail-query interfaces.
Measure these in the stated fixtures and performance study. Optional
packaging and file-reader choices remain implementation recommendations.
The upstream shared-event windowing work is now an explicit dependency;
full exposure coverage is not an acceptable silent fallback.

## Deferred extensions

- **[YLT-Sparse-Inputs]** A separate future syntax could declare a year
  universe for incomplete source files. The current contract requires full
  YLTs including zero years. Unequal probabilities are already in scope
  through YLT3; arbitrary weights are not a substitute for probabilities.
- **[YLT-Joint-Units]** Form an empirical joint by pairing unit losses in
  each supplied year. This needs no common-event Poisson model; it preserves
  the same year alignment used for the total.
- **[ELT-Coupling-Selection]** Expose the existing eltq coupling alternatives
  with clear distinctions between marginal-preserving couplings, Homer
  roll-up, and the variance-matched construction.
- **[ELT-Joint-Units]** Use `eltq.occurrence_pair_joint` to obtain a rate-weighted
  occurrence joint and compound it with the same shared Poisson count. A
  pair's contributing event rate can differ from the full total's rate;
  use the pair's metadata deliberately.
- **[ELT-Unit-Allocation]** Integrate `occurrence_unit_first_moment` and annual
  conditional means without inferring allocation from the total-only law.
- **[ELT-Unit-Terms]** Apply unit-level terms before summation, using eltq's
  structured terms rather than forwarding an arbitrary fragment of DecL.
- **[ELT-Frozen-Inputs]** Provide an explicit portable data snapshot/export
  workflow for users who need a declaration independent of mutable files.
- **[ELT-General-Continuous-Queries]** Extend `SeverityElt` beyond its model
  moments, specialized tail checks, and grid-backed results to a complete
  continuous CDF/PDF/PPF interface if a later use case requires it.

## Execution log

Record final rulings, actual versions, acceptance results, and material
deviations here as the plan is executed.

**Execution review, 2026-10-26.** The integration table checked out against
1.11.1 except for the grid budget (ruling 14). Two clarifications from the
check: `multiunit_views` already accepts `grid_length` but has no window or
overflow policy (a short grid raises `GridOverflowError`), and its `views`
default is all four views, so the adapter passes `('ground_up',)`
explicitly. `Recipe` stores no base directory, and `_add` keeps a relative
`source` as given, so `[Loss-Table-Files]` adds one. Rulings 13 to 16 were
recorded.

**1.11.2 `[Quote-Safe-Statements]`.** `parser._lift_literals` and
`_restore_literals` replace the trailer-only lift in both `preprocess` and
`raw_statements`: one alternation lifts quoted strings and single-line
`note`/`tags`/`hints` bodies, whichever starts first. Fixes existing labels:
`as "Cat #1"` was truncated to `as "Cat` and `as "a[1]"` became
`"a [1] "`. The spec snapshot regenerated with no diff. Tests:
`tests/test_quoted_literals.py`.

Planning checks on 2026-10-26: read the implementation article before tracing
the current eltq sizing and overflow code; checked the beta-tail examples
with SciPy and the weighted YLT moments with exact rational arithmetic.
Checked document math delimiters, code fences, prose conventions, citation
keys, and proposed-name collisions. eltq is not installed in aggregate's
development environment, so its API behavior was verified from source;
no eltq runtime integration or implementation test suite was run.

## References

::: {#refs}
:::
