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
