# Plan: punchups, August 24

Status: drafted 2026-08-24. A running punch list: each numbered item is
self-contained, executes independently, and bumps the version on its own.
More items will be added; keep each section tight.

## 1 [Reins-Economics-On-Bvagg-Ignore-Warn] economics clauses crash every bvagg build

### Symptom

The app's GCN control on a priced occurrence program fails with
`Aggregate.__init__() got an unexpected keyword argument 'occ_reins_premium'`.
Minimal reproduction (verified 2026-08-24):

```
grossceded agg DiceThreeEvenDice
  dfreq [3] dsev [2 4 6 8 10 12]
  occurrence net of 6 xs 6 deposit 5
```

The same program without `deposit 5` builds fine, and a plain `agg` with the
clause builds with the `IgnoredDecLClauseWarning`.

### Diagnosis

The reinsurance-economics clauses (ceded premium `deposit` / `rol` / `rate`,
`cede`, reinstatements, the variable-rating features, retro) parse anywhere the
shared agg body parses and land in the spec as the `_AGG_IGNORED_ECONOMICS_KEYS`
(`underwriter.py:61`). They are not `Aggregate.__init__` kwargs. The plain
`agg` factory branch knows this: it warns and filters a copy before
construction (`underwriter.py:1394` to 1407). The `bvagg` branch does not: it
passes the parsed spec straight through (`obj = BivariateAggregate(**spec)`,
`underwriter.py:1665`), and both unit-construction sites splat the raw unit
spec into `Aggregate`:

- netceded / view-pair mode: `self._nc_agg = Aggregate(**units[0][2])`,
  `bivariate.py:1400`. Hit by `netceded` / `grossceded` / `grossnet`, so by
  the GCN button on any program carrying a premium clause.
- copula mode: `self.units = [Aggregate(**s) for s in self._unit_specs]`,
  `bivariate.py:1342`. Verified failing with the same TypeError when a unit
  carries `deposit`.

Unaffected: `Aggregate.occ_bivariate` (passes a pre-built `nc_agg`), and the
discrete `dbvsev` mode (no unit specs).

### Fix, at the factory layer

One shared helper, two call sites, zero duplication. Extract the plain-agg
branch's warn-and-filter dance (`underwriter.py:1402` to 1406) into a module
function beside the key list, shape roughly
`_strip_ignored_economics(name, spec, context) -> spec`: detects the clauses,
emits the `IgnoredDecLClauseWarning` with the context's remedy sentence, and
returns a filtered copy (the spec unchanged when nothing is present). The
`agg` branch becomes a one-line call to it; the `bvagg` branch calls it once
per `('agg', unit_name, unit_spec)` in `spec['units']`, named by unit. The
key list and the filter live once.

Copy discipline matters: `parsed.spec` is the stored recipe's spec, so the
helper returns a copy and the bvagg branch rebuilds the outer spec, the units
list, and each tuple around it, never mutating in place. The stored program
keeps the full declaration, exactly as the agg branch arranges. The
loss-structure keys (`occ_reins`, `occ_kind`, the agg twins) stay: the
netceded joint requires the cession layers. Only the economics keys go.

Wording: `ignored_clauses_message`'s remedy sentence ("fold it into a 'pnl' /
'xpnl' by reference") is agg-specific and wrong for a joint, which is loss /
loss by design. The `context` parameter picks the remedy; the clause-list
sentence is common. The bvagg remedy is one sentence saying the joint carries
loss only and the clause is ignored.

Why the factory and not `BivariateAggregate.__init__`: the key list and the
warning wording live in `underwriter.py` beside the agg twin, and every
raw-spec construction route runs through `_build_object`; direct
`BivariateAggregate` construction is not a DecL surface.

### Tests and hygiene

Two new cases asserting the build succeeds and the warning fires: the
view-pair prefix with `deposit` (the GCN shape) and a copula unit with
`deposit`. No grammar or transformer change, so `expected_specs.json` does not
move. Version bump, one-paragraph CHANGELOG entry under this label. App side
needs nothing: the GCN control already gates on cession presence and the
served build starts succeeding, with the warning traveling on whatever
channel build warnings already use.

## 2 [Reins-Insurer-Moments-Block] layer frequency and cover columns

Target: the `reins_layer_moments` block of the reins exhibit under the
insurer perspective (`exhibits/_aggregate.py`, `_reins_insurer_aggregate`),
the app's Reins, Summary subtab. All three changes are view level; the
backing frame `reins_stats_df` (`_reinsurance.py:937`) does not move and RAW
keeps the full store, per `[Perspective-May-Restructure]`.

**Frequency to the layer: already done at the frame level, pin it.** The
layer columns thin the gross count by the attach probability
(`MomentAggregator.thin_moments(pr, f1, f2, f3)`, `_reinsurance.py:1123`),
so layer freq mean is ground-up frequency times P(subject reaches the
layer). Verified 2026-08-24 on the dice example: layer 6 xs 6 shows freq
mean 1.5 = 3 x 0.5, and freq mean times sev mean equals agg mean on every
row (layer rows conditional, gross / ceded / net unconditional). No code
change; add a test asserting the identity so it cannot regress silently.

**Drop the freq cv and skew columns** from this block. The frame keeps them;
the raw perspective still serves them.

**Add cover columns before freq**: `(cover, share)`, `(cover, limit)`,
`(cover, attach)`, read off the same stats frame's meta rows and served
under a `cover` component group, so the header reads cover | freq | sev |
agg over measures share limit attach | mean | mean cv skew | mean cv skew,
exactly the target column list. Update the block caption (it currently
promises "the same rows as the contract above"; with the cover columns
repeated the block stands alone). No app work: the pipeline serves blocks
whole.

## 3 [ZT-ZM-Recalibrate-Default] swap the meaning of `!` on zt / zm

**The surprise.** `10 claims ... zt` does not produce 10 expected claims:
the bare form is the textbook (a, b, 1) base parameterization (the exposure
clause sets the base mean and the reweighting shifts it), and the trailing
`!` pins the realized mean to the clause. When you say n claims you expect
n claims, so the recalibrated reading should be the default and `!` should
mean do not recalibrate, the base-parameterization opt-out that earns the
exclamation mark.

**The swap, mechanically.** Grammar unchanged (both alternatives exist);
only the transformer flips: `freq_zm` / `freq_zt` (`parser.py:1297` to
1321) set `freq_pin_mean=True` on the bare forms and the `_pin` variants
leave it unset. Ripples, all named now:

- `decl_writer.py:378` emits `!` on the un-pinned spec instead; round-trip
  tests confirm.
- Grammar comments `decl.lark:311` to 317 and 488 to 491 (the frequency-side
  `!` gloss) rewritten; the severity / dsev / dwait `!` (unconditional) is
  unrelated and out of scope.
- `_frequency.py:600` (the infeasible-solve message says "append ! to pin")
  and the `Aggregate` docstrings at `_aggregate.py:2253` and 3161 reworded.
  With pinning the default, an infeasible solve now hits programs that used
  to build; the message must point at `!` as the escape.
- Corpus: roughly 20 zt / zm lines across `library.agg` and
  `decl-testers.agg` (grep again at execution, including `_test_suite.agg`).
  Toggle the `!` on each so every entry keeps its intended realized
  behavior; the `expected_specs.json` regen then shows exactly those lines
  and nothing else, the diff read deliberately. `tests/test_frequency_zm.py`
  expectations flip alongside.
- Doc pages naming the old semantics: lockstep `.rst` edits, build deferred
  to the author as usual.

**Decision for the author.** `Aggregate` is stable tier and takes
`freq_pin_mean=False` as a constructor kwarg (`_aggregate.py:2193`).
Recommended: leave the kwarg name and default alone and have the parser set
it explicitly both ways, so the flip is confined to the DecL surface where
the surprise lives. The alternative, flipping the kwarg default too, keeps
DecL and constructor readings aligned at the cost of a stable-tier behavior
change for direct constructor callers.

App ripple: none. `zt` / `zm` are already in `decl-keywords.json` and `!`
is not a keyword.

## 4 [Reins-Insurer-Terms-Block] loss replaces pr_loss, integer output

Target: the `reins_layer_terms` block, same function and same view-level
scope as item 2; `reins_stats_df` keeps `pr_loss` in its meta rows and RAW
still serves it.

**`loss` replaces `pr_loss`**, in the same column position (between
pr_detach and lol): the row's expected aggregate loss, read off the frame's
`('agg', 'mean')` in the insurer view, no new frame row. The pairing is the
point: loss is the placed figure (the ceder applies share, so a 50% placed
layer shows half the dollars) while lol is mean over share times limit and
so is share independent. Verified on the current frame: layer lol already
divides by the placed limit, so lol does not move. Caption gains a sentence
saying which of the pair reflects share.

**`output` serves as integer 0 or 1**, not 0.0 / 1.0: cast the column in
the view. Safe: the meta row is always set (default 0.0, never NaN), and
the served TableDoc carries dtypes, so the grid renders it as an integer
without app work.

## 5 [Format-Program-Picks-Line] picks on its own line, and colorized

**Layout.** `format_program`'s spread layout currently lands the whole
severity clause on one line because `_render_dist` appends the picks
fragment inline (`decl_writer.py:222`, `_render_picks` at 278). Change the
severity clause fragment to a `_Block`: head is the clause without picks,
the picks fragment its child, so `_render_spread` gives
`picks [attachments] [losses]` its own line one level deeper. `_render_terse`
space-joins a block back to one line, so `spec_to_decl`, the `to_agg`
export, and the round-trip corpus stay byte identical; the trailing `!` of
an unconditional severity rides at the end of the last fragment so the
terse byte order (picks then bang) does not move. Tests: a spread-layout
case for a picks program plus the existing round-trip suites.

**Colorizing: the gap is app side.** LIB already colors `picks` (verified
2026-08-24: `AggLexer` emits Token.Keyword, `decl_pygments.py:221`). The
app editor does not: `web/src/decl-keywords.json`, the hand-curated mirror
behind CodeMirror highlighting and offline completion, has no `picks`
entry. One-line app edit adding `picks` to the appropriate group, recorded
here as the paired ask per the grammar-ripple agreement (the a249 model
case) so it is not lost; it rides with the app's next bump, not this repo's.

---

# Execution log

Written as each item lands, per the house convention: what the plan specified, what the code does instead, and why. A recorded divergence is a good outcome; an unrecorded one is the failure.

## Review, 2026-08-24, against `1.0.0a320`

Every landmark the plan names was verified against the tree before any edit. All of them still read as written: `underwriter.py:61` `_AGG_IGNORED_ECONOMICS_KEYS`, the warn and filter dance at 1402 to 1406, `BivariateAggregate(**spec)` at 1665, `bivariate.py:1342` and `1400`; `exhibits/_aggregate.py` `_reins_insurer_aggregate` with its two blocks, `_reinsurance.py:937` `reins_stats_df` and `1123` `thin_moments`; `parser.py:1297` to 1321, `decl_writer.py:222` and `278` and `378`, `decl.lark:311` to 317 and 483 to 491, `_frequency.py:600`, `_aggregate.py:2193`, `2253` and `3161`; `decl_pygments.py:221`. Three claims were re-measured rather than taken on trust, and all three hold: the item 1 symptom reproduces exactly as quoted; the item 2 thinning identity holds on the dice example (layer freq mean 1.5 = 3 x 0.5, and freq mean times sev mean equals agg mean on all four columns); and the item 4 share reading holds (`0.5 po 6 xs 6` shows loss 0.5 against 6 for the full placement while lol stays 1.0 in both).

Names vetted: `_strip_ignored_economics` is free across `src/aggregate`, and `_IGNORED_CLAUSE_REMEDIES` is new. Nothing else on the list introduces a public name.

App ripple, stated even where nothing is owed. Item 1: none, the GCN control already gates on cession presence. Items 2 and 4: none, the pipeline serves blocks whole. Item 5: one line, `picks` into `web/src/decl-keywords.json`, confirmed absent there on 2026-08-24 and confirmed not already carried by the API's own `plan-punchups-aug-24-API.md` (which states it shares nothing with this list). Item 3: none, `zt` / `zm` are already in the mirror and `!` is not a keyword.

**One blocker, item 3.** Its "Decision for the author" is undispositioned, so item 3 is not executed. The other four items are executed, on the plan's own statement that each is self contained.

## Item 1 [Reins-Economics-On-Bvagg-Ignore-Warn], landed `1.0.0a321`

Executed as specified: one shared helper, both call sites, no duplication.

Divergences.

1. **The remedy wording lives in a module dict, not a branch.** The plan said the `context` parameter picks the remedy. It is implemented as `_IGNORED_CLAUSE_REMEDIES`, a two entry mapping beside `_IGNORED_CLAUSE_NAMES`, so the key list, the clause names and the remedy sentences all sit together as the plan intended for the first two. `ignored_clauses_message` gained the `context` parameter (defaulted to `'agg'`, so the signature is compatible) and `_strip_ignored_economics` passes it through.

2. **The bvagg branch guards on `units` being present and on the unit kind.** The plan said call it once per `('agg', unit_name, unit_spec)` in `spec['units']`. The discrete `dbvsev` mode carries no `units` at all, so the loop is guarded; a non `agg` unit kind passes through untouched, since `pnl` components are refused a few lines later inside `BivariateAggregate` and filtering them first would only change which error the reader sees.

3. **`stacklevel` moved from 2 to 3.** The warn is one function deeper than it was, so the reported location is unchanged.

4. **Four tests, not two.** The plan asked for the view pair case and the copula case. Landed with two more, matching the shape of the `agg` block above them in the same file: the stored recipe keeps the economics (the copy discipline the plan is explicit about, asserted directly rather than inferred), and an undecorated joint emits nothing. A `uwb` fixture with `update=False` serves all four: every assertion is `__init__` state, and a joint update costs about 23 seconds against a fraction of a second for the build.

5. **`bvagg.NAME` is not a reference form.** The copy discipline test was first written to rebuild by dotted reference, mirroring `test_rebuild_by_name_still_warns` on the agg side. There is no `bvagg.NAME` builtin prefix in the grammar, so it rebuilds by bare name instead, which exercises the same stored spec.

### The gate at `a321`, and why it is not green

**Three failures are pre-existing at `HEAD` (`92a05a4`), not caused by anything here.** Proven rather than assumed: a detached worktree at `HEAD` with `PYTHONPATH` pointed at its own `src` fails exactly these three and no others. They are `test_agg_libraries.py::test_library_retired_the_letter_prefixes`, `test_agg_libraries.py::test_library_is_written_in_the_canonical_layout` (27 entries not canonical; the test names its own remedy, `python dev/done/reflow_library.py`) and `test_library_entries.py::test_split_limit_policy_prices_the_per_accident_limit` (relative gap 1.7e-05 against a 1e-9 tolerance, which reads as fallout from `a318` no longer rebuilding hinted entries on the auto sized grid). All three are the author's, in the `library.agg` reorganization that is still in flight, and all three are untouched by this list. Note for item 5, corrected after measuring: that canonical layout test compares the **flattened** statement (`UnderwritingLexer.preprocess(canonical)[0]` against the file's statement), not the layout, so a layout change cannot move it. Verified at `a324`: the offender list is the same 27 names, in the same order, before and after. What item 5 does change is what a future `dev/done/reflow_library.py` run would write into `library.agg`, since that script writes the spread layout: `LayerPicks.Uniform` and `MED.WithPicks` would gain a picks line. That run is not made here, the library reorganization being the author's and in flight.

**Everything else that failed was a memory allocation failure, and they move between runs.** Four full runs of `-m 'slow or not slow'` produced three different extra failures: `test_bivariate.py::test_mv_explain_flags_clipped_book` and `test_reins_bivariate.py::test_netceded_refuses_a_pin_it_cannot_honor` in the first, `test_composition_matrix.py::test_gc_feat_walk_has_occ_step_and_true_gross` in the second (`numpy._core._exceptions._ArrayMemoryError: Unable to allocate 8.00 MiB`), none in the second for the first two, and a worker dying outright in the third (`[gw0] node down: Not properly terminated`). `dev/TODO.md` `[Bivariate-Gate-Flake]` names the first two of those tests by node id and describes exactly this profile: load dependent, unreproducible in isolation, two different tests in two different bivariate modules pointing at a shared cause rather than at either test.

**What was run clean instead, and its result.** Tier 2, the fast suite, `-m 'not slow'`: **4803 passed, only the three pre-existing library failures**, no memory failures at all. Then `-m slow -n 2` for the quarantined modules: 171 passed with one flake, and `tests/test_reins_bivariate.py` alone at `-n 2` passes whole (40 passed). Together those two commands cover everything the tier 3 gate covers.

**The one flake was then shown to be impossible to attribute to this change.** `test_netceded_refuses_a_pin_it_cannot_honor` builds `OCC`, which carries no economics clause at all, so `_strip_ignored_economics` returns its argument unchanged; and it is an `agg`, so it never reaches the `bvagg` branch this item edits. It then calls `occ_bivariate` directly on the built object, which does not go through the factory. The edit is a provable no-op on that test.

## Item 2 [Reins-Insurer-Moments-Block], landed `1.0.0a322`

All three changes landed at view level, exactly as scoped: `reins_stats_df` does not move and RAW keeps the full store. The header now reads cover | freq | sev | agg over share limit attach | mean | mean cv skew | mean cv skew, the target column list.

Divergences.

1. **`_LAYER_MOMENTS` became a mapping, and `cover` joined `_LAYER_COMPONENTS`.** The plan described three separate edits (pin the frequency, drop freq cv and skew, add a cover group). They collapse into one data structure: `_LAYER_COMPONENTS` gains `'cover'` at the front and `_LAYER_MOMENTS` turns from a flat tuple of three measures into a component keyed mapping. That is what lets the block's column order be read off one declaration rather than assembled in the body, and it is why the freq narrowing costs no code at all. A new `_moment_rows` does the slice and the relabel; `_layer_rows` is untouched.

2. **The cover rows are relabeled, not recomputed.** `('meta', 'share')` in the store becomes `('cover', 'share')` in the view. That keeps the block a pure reading of `reins_stats_df` and keeps `test_reins_aggregate_insurer_is_a_reading_of_the_raw_frame` meaningful; that test gained a two line mapping from the `cover` group back to `meta` rather than an exemption.

3. **The plan's "no code change, add a test" for the thinning identity became two assertions, not one.** `test_reins_layer_frequency_is_the_thinned_count` asserts both halves: that freq mean times sev mean is agg mean on every row where frequency and severity are present, and that each layer's frequency really is the gross frequency times `pr_attach`. The second is the one that would catch a regression in `thin_moments` itself; the first is the one that would catch a view that reordered or mislabeled a column.

4. **`pandas` is now imported in `exhibits/_aggregate.py`.** It was not, the module having previously done all its work through `_core` helpers and frame methods. The relabel needs `pd.MultiIndex.from_tuples`.

5. **The exhibit IR snapshot moved, and the diff was read by key rather than by line.** `tests/data/exhibit_snapshots.json` regenerates through `tests/capture_exhibit_snapshots.py`. Exactly one of its 124 keys changed, `reins/insurer/ReinsAggregate`, with none added and none removed, which is the whole check that the change is confined to the block it targets.

## Item 4 [Reins-Insurer-Terms-Block], landed `1.0.0a323`

Both changes landed as specified, at view level, with `reins_stats_df` keeping `pr_loss` and RAW still serving it. `loss` sits in the old `pr_loss` position between `pr_detach` and `lol`, and `output` serves as `int64`.

Divergences.

1. **Executed out of the plan's numeric order, immediately after item 2.** Items 2 and 4 edit the same function and the same two exhibit tests, and each regenerates the same one key of `exhibit_snapshots.json`. Running them adjacently meant one read of that code rather than two with item 3's corpus sweep in between. Nothing in either depends on the other, so this is ordering only.

2. **The source of a contract row is declared, not branched.** `_LAYER_TERM_SOURCES` maps the one row that is not a `meta` row to where it is read from, and `_term_rows` consumes it. `_LAYER_TERM_INTEGERS` does the same job for the cast. Both are one entry mappings today, which is deliberate: the alternative is a special case for `loss` and another for `output` buried in the block body, where the next reader has to find them.

3. **The integer cast is guarded on completeness, not asserted.** The plan reasons that the cast is safe because `col` defaults `output` to 0.0 and never leaves it `NaN`, which is true of every column the store builds today. The code still checks `notna().all()` before casting, because an all `NaN` column would otherwise turn into a `TypeError` at exhibit build time on some future stage that does not set it, and a float flag is a cosmetic problem where a raised exhibit is not.

4. **A second test came with it.** The plan asks only that the column change. `test_reins_layer_loss_reflects_share_and_lol_does_not` pins the pairing the plan calls the point of the change: a `0.5 po 6 xs 6` layer is a one twelfth share, its `loss` is one twelfth of the fully placed layer's, and its `lol` is identical. That is the assertion that would catch `loss` being read off an unplaced quantity later.

5. **No documentation edit was owed.** `pr_loss` is documented in `_aggregate.py` and `docs/2_aggregate_overview/pipeline-reinsurance.rst`, and both describe `reins_stats_df`, which is unchanged. Checked rather than assumed.

## Item 5 [Format-Program-Picks-Line], landed `1.0.0a324`

The layout half landed as specified. `picks [attachments] [losses]` gets its own line one level deeper in the spread layout, terse is byte for byte what it was, and the trailing `!` of an unconditional severity rides at the end of the last fragment so the terse byte order does not move. The colorizing half is an app ask, carried below and not touched here.

Divergences.

1. **The split is a `split=` flag on `_render_dist`, not a second function.** The plan said to change the severity clause fragment to a `_Block`, which is what `_render_sev_clause` now returns; the mechanism underneath is `_render_dist(spec, split=True)` handing back `(head, picks_fragment)`. `_render_dist` has two other callers, `_render_wait` and `_render_sev_out`, and neither wants a block, so the joined form stays the default and those two are untouched.

2. **The `clash` renderer had to be flattened explicitly, which the plan does not mention.** `_render_bvagg` builds a clash component with `_join([_render_layers(sa), _render_sev_clause(sa)])`, and `_join` is a plain `' '.join` over strings: a `_Block` reaching it is a `TypeError`. A clash component is one line by construction, so both sites now wrap the clause in `_render_terse`. This is the one place the change would have broken something, and no existing test covered a clash with a picks severity, so it would have gone out silently.

3. **The interior severity label rides on the picks line.** The plan says only that the bang does. The label closes the whole clause in the terse form, so it has to follow the picks for the bytes to be unchanged, which puts it at the end of the picks child in spread: `picks [50 75 100] [30 12 5] as Picked`. It re-parses, the preprocessor folding the lines back into one statement, but it reads as though the label belongs to the picks rather than to the severity. Worth a look if the author dislikes it; the alternative costs the byte identity.

4. **Three tests and three corpus lines, against the plan's one case.** The plan asks for a spread-layout case for a picks program. Landed with that plus one pinning the bang and the label on the last fragment, and one pinning that a picks-free severity keeps its bang on the severity line (the case that proves the block is conditional). The three matching programs are in `decl-testers.agg` under DW, and round-trip.
