# plan-pricing-exhibits, the LIB half: review and execution notes

Written by the LIB agent, 2026-08-12, against
`aggregate_api/dev/plan-pricing-exhibits.md` (the canonical copy; LIB's is a
symlink). Companion to whatever the API agent writes for its half. Read
section 4 first if you only want to know what the app has to code against.

> **Moved to done 2026-08-13.** The three API phases this file left "untouched
> and unblocked" have since landed at `aggregate_api` a83 (A1, routes), a84
> (A2, the pane) and a85 (A3, the deletions), so the plan is executed end to
> end and the canonical copy sits in the API repo's `dev/done/`.

**Verdict on the plan: in order, and executed.** All five LIB phases are done,
in five version bumps, `1.0.0a259` to `1.0.0a263`. The acceptance criteria the
plan owns on this side all pass, including the one that was the point:
**the round trip closes.** Everything below is either a statement of what
shipped or a place where the plan and the code disagree and the code is
deliberate. There are nine of those and they are collected in section 3, none
of them large, three of them things the plan does not mention at all and that
execution turned up.

The three API phases (A1 routes, A2 pane, A3 deletions) are untouched and
unblocked. Nothing here needs LIB to move again.

---

## 1. What landed

| plan phase | shipped in | where |
|---|---|---|
| L1 `[Pricing-Result-Objects]` | `1.0.0a259` | `results.py`, `_pricing.py`, `_aggregate.py`, `_portfolio.py`, `_pnl.py`, `utilities.py`; `tests/test_pricing_results.py` |
| L2 `[Unbounded-Anchor-Guard]` | `1.0.0a260` | `_pricing.guard_unbounded_anchor`, four call sites; `tests/test_unbounded_anchor.py` |
| L3 `[Evaluate-Asset-Anchor]` | `1.0.0a261` | `_pricing.py`, `_aggregate.py`, `_portfolio.py`; `tests/test_evaluate_anchor.py` |
| L4 `[Reins-View-Pricing]` | `1.0.0a262` | `_reinsurance.py`, `_pricing.py`, `_aggregate.py`, `_portfolio.py`; `tests/test_reins_view_pricing.py` |
| L5 `[Pricing-Exhibits]` | `1.0.0a263` | `exhibits/_pricing.py` (new), `exhibits/_core.py`, `exhibits/__init__.py`; `tests/test_exhibits.py` |

One bump per phase, in plan order, as the plan's cadence section asks. The
exhibit count goes from 12 to **15**.

Full fast suite green at every bump (4,047 cases at a263). Two failures in the
`slow` suites (`test_massive_pnl_one_sweep_ledger`,
`test_mv_explain_flags_clipped_book`) are **pre-existing**: the first
reproduces unchanged at `f1dce86` (a258, before any of this work), the second
passes in isolation there and fails only under the grouped parallel run, which
is the known bivariate flake. Neither is touched by anything here and neither
was introduced by it.

### The acceptance criteria, measured

**1. The round trip closes.** Calibrate at `coc = 0.15, p = 0.99`, take the
implied premium at the resolved asset level, evaluate that premium at the same
anchor, recover each family's calibrated parameter. On the author's three
reference programs, worst relative miss over all five families:

| program | assets | premium | worst relative recovery |
|---|---|---|---|
| `BasicBook` | 30,862 | 25,350.64 | 5.8e-09 |
| `BasicBookRe` | 16,387 | 13,386.50 | 5.9e-09 |
| `port Basic` | 521 | 197.49 | 4.7e-15 |

`ccoc` recovers 0.15 exactly on all three, which is the sharper test, since its
closed form reads `assets` and `el` directly rather than solving.

**3. `p = 1` on `BasicBook` fails cleanly**, with a message written for the
wire. **4. Hash for hash**: now asserted in LIB as well, see 2.4.

Criterion 2, the target screenshots, is an eyeball check the author owns; the
INSURER block for a reinsured Aggregate reproduces the described shape
(`gross*`, `net`, `gross less net` per distortion, octet columns), asserted
structurally in `tests/test_exhibits.py`.

### Things the plan worried about that turned out to be free

- **`[Pricing-Keyed-On-Result]` needed no registry change,** exactly as the
  plan predicted. `register_simple_exhibit` and `fn.register` took the result
  classes without noticing they were not first class citizens.
- **The RAW invariant sweep extended over result fixtures with no change of
  wording,** also as predicted. The four result fixtures joined the shared
  `objects` set in `tests/test_exhibits.py`, so they inherit the invariant
  sweep, the caption sweep, the named index sweep and the snapshots without a
  single new assertion being needed for any of them.
- **`available_exhibits(built_object)` is untouched.** Asserted. The app's
  capability payload does not move because these exist.

---

## 2. The contract as built

Code against this section where it disagrees with the plan.

### 2.1 The two result objects

```python
CalibrationResult                      # calibrate_distortions returns this
    distortions        {name: Distortion}
    distortion_df      the per family receipt
    calibration_df     the shared one row target
    coc                the cost of capital fitted to (derived when lr= given)
    lr                 the loss ratio as stated, or None
    p, a               the resolved pair
    anchor             'p' or 'a': which one the caller fixed
    kind, names, reins_view
    pricing_df         lazy: allocation across units      (Portfolio source)
    reins_price_df     lazy: allocation across views      (reinsured Aggregate)

EvaluationResult                       # evaluate returns this
    evaluation_df      the panel
    premium            what it was measured against, or None on a PnL
    reins_view, p, a, names
```

Both borrow `name`, `label`, `_title_name` and `_relabel` from `_source`
through `SourcedMixin`, so an exhibit title reads `Calibrated distortions:
BasicBook`. The three older results (`AnalyzeDistortionResult`,
`AnalyzeDistortionsResult`, `PricingResult`) gained `_source` and the same
delegation.

**The breaking part is the return value only.** `obj.distortions`,
`obj.distortion_df` and `obj.calibration_df` are set exactly as before, so
`obj.calibrate_distortions(...)` followed by `obj.distortion_df` reads
unchanged. Forty three call sites moved across six existing test files, every
one of them the same mechanical shape: `.calibrate_distortions(...)` grows a
`.distortion_df`, `.evaluate(...)` grows an `.evaluation_df`. That the whole
break is one mechanical rewrite is the best evidence that the stored
attributes carried the weight they were meant to.

Which lazy frame exists depends on the source, and asking for the wrong one
raises `AttributeError` naming the reason rather than returning an empty
frame. An `Aggregate` with no cession has neither: one distribution, nothing
to spread, and its allocation story is the single `calibration_df` row.

### 2.2 The anchor guard

`guard_unbounded_anchor(obj, p, where=...)` refuses `p == 1` when the object
reports `bounded` False. Live at `calibrate_distortions`, `price_pentagon`,
`price_pentagon_ex`, `reins_price_df` and `evaluate`.

Only the exact spelling `p = 1` is refused. Three ways past it, each saying
something different: `a=` names the level (`a=obj.q(1)` reproduces the old
number with the caller having asked for it), a `p` below 1 asks a question the
distribution can answer, `obj.bounded = True` certifies a support the
heuristic could not prove. On a `Portfolio` the test is the worst-of over
units, which is the right test there.

### 2.3 `reins_price_df`, the octet

Columns are `PENTAGON_STATS` and nothing else: `L, M, P, Q, a, LR, PQ, ROE`.
`el` is `L`, `ask` is `P`, `margin` is `M`. `bid` is gone and is no longer
computed. **On the unlimited quote (no `p`, no `a`) `a` is `inf` and `Q`, `PQ`
and `ROE` are `NaN`**; see 3.3.

### 2.4 The three exhibits

Registry names carry a dot, as the plan asks: `pricing.calibrate`,
`pricing.allocate`, `pricing.evaluate`. The module level bindings are
`pricing_calibrate` and friends, since a dot is not an identifier.

Block lists as built, which is the table to code the pane against:

| source | exhibit | RAW blocks | INSURER blocks |
|---|---|---|---|
| any | `pricing.calibrate` | `distortion_df` | same |
| `Portfolio` | `pricing.allocate` | `calibration_df`, `pricing_df` | `calibration_df`, `stat_LR`, `stat_P`, `stat_PQ`, `stat_ROE` |
| reinsured `Aggregate` | `pricing.allocate` | `reins_price_df` | `reins_price_df` (narrower rows) |
| plain `Aggregate` | `pricing.allocate` | `calibration_df` | same |
| any | `pricing.evaluate` | `evaluation_df` | same frame, INSURER caption |

The reinsured INSURER block, per distortion, in this row order: the calibrated
basis starred (`gross` reads `gross*`), the other whole program views, then one
`basis less view` row per other view with `a` and the three ratios recomputed
from the differenced amounts. `ceded` and `ceded occ` are dropped.

**The envelope round trip is now asserted in LIB.** Every served block, over
every object, exhibit and perspective, reconstructs through
`gt.TableDoc.model_validate(gt.canonical_dict(doc))` to the same `hash`. That
contract used to live only on the API side; it is cheap here and it is where
the documents are made.

The snapshot file grew 14 keys and **no existing snapshot moved**, which is the
blast radius to expect from a purely additive phase.

---

## 3. Where the code and the plan disagree

Nine. Three are things the plan does not mention and execution turned up
(3.2, 3.4, 3.5); the rest are choices the plan left implicit. All are
deliberate.

### 3.1 `PnL.evaluate` does change its return type, and the plan says both

Phase L1 says `EvaluationResult` is breaking because "a bare DataFrame cannot
be dispatched on", and the leaf matrix gives `PnL` the Evaluate leaf. Phase L3
says "`PnL.evaluate` is deliberately unchanged this round (decision 4)".

Both are right about different things and the code reads them that way. The
**type** changes at L1, on all three classes, because it has to: a `PnL` cannot
serve `pricing.evaluate` unless its `evaluate` returns something dispatchable.
The **anchor** is what L3 leaves alone, which is what decision 4 is actually
about: each ledger row is its own position and the right anchor semantics there
want their own discussion. So `PnL.evaluate` takes no `p=` or `a=`, and its
result carries `premium=None`, `p=None`, `a=None`, because every ledger row has
its own consideration and no single number stands for them.

### 3.2 Two existing `p = 1` callers, both legitimate, both moved

The plan does not mention that anything calls `p=1` today. Two things do, both
in the Bounds suites (`tests/test_bounds.py`, `tests/test_chart_bounds.py`),
both on Poisson books, and both **deliberate**: the IME 2022 bounds
methodology takes the top of the realized grid as the asset level on purpose,
and the same fixtures compute `port.q(1)` two lines away to get the capital.

They now write `a=port.q(1)`. That is the same number with the choice made
visible, which is precisely what the guard exists to force. Nothing in the
library itself called `p=1`.

Worth knowing for the app: the guard is a **behavior change for any caller who
was passing `p=1` and getting an answer**. There is no deprecation window,
which is right for a number that was wrong, and the message names the escape
hatch.

### 3.3 The unlimited quote has no capital, and blanks rather than reading zero

Plan L4 says the grown `reins_price_df` carries "the canonical octet in
`PENTAGON_STATS` order and nothing else" and says nothing about the **default**
call, where neither `p` nor `a` is given and the price is unlimited. That
default is documented and deliberate, so it had to be answered.

The arithmetic answer is available: `Q = inf - P = inf`, `PQ = P / inf = 0`,
`ROE = M / inf = 0`. It is also a false reading. A 0% return on infinite
capital is not what an unlimited quote means; what it means is that the
question does not apply. So `a` stays `inf`, which is what the caller asked
for, and `Q`, `PQ` and `ROE` are blank.

This is the one place the frame does not simply pass through
`complete_pentagon`: the completion runs and the three cells are then cleared
for the rows whose asset level is not finite.

### 3.4 A loss ratio target needs a refusal the plan does not mention

`lr=` can ask for something no distribution can deliver, and a `coc=` target
cannot. Given `coc`, the premium is `nu * L + delta * a`, which sits between
the expected loss and the assets by construction. Given `lr`, the premium is
`L / lr`, which is free to land **above** the assets.

It does so on ordinary inputs. On `BasicBookRe` at `p = 0.99`, `lr = 0.7`
implies a premium of 18,481 against assets of 16,387: negative capital, a
negative implied cost of capital, and a calibration that then chases a premium
target above the essential supremum and reports "questionable convergence" for
wang, dual and tvar with a divide by zero on the way. Every number in that
receipt is garbage and nothing says so.

So the conversion refuses, with a message naming the implied premium, the
assets and the expected loss. The app will hit this: the Calibrate form offers
`CoC | LR` and a reader typing 0.7 on a thinly capitalized book is a normal
thing to do. **It is an HTTP 400 with a readable `detail`, exactly like the
L2 guard**, and the preview line is where it should land.

### 3.5 `ccoc` joining the panel exposed two spellings of one column

The plan's decision 3 (`ccoc` joins the evaluate families when anchored) has a
consequence it does not mention: it is the first time the calibration receipt
and the acceptability panel contain the same family, and they spelled its
`param_name` differently. A family with no declared `param_name` read `r` in
`distortion_df` and `param` in `evaluation_df`, so the Allocate and Evaluate
tables would have disagreed about the name of the same number on the same
screen.

Both now go through `_param_name` and read `r`. The four families that declare
a name (`a`, `lam`, `b`, `p`) are unaffected, so nothing else moved.

### 3.6 The result class names follow the existing family, not the house rule

`CalibrationResult` and `EvaluationResult` are suffix form. The house rule
(`CLAUDE.md`, naming conventions) is the `Base<Kind>` **prefix** form, which
would make them `ResultCalibration` and `ResultEvaluation`.

The plan names them the suffix way and I kept that. They are siblings of
`AnalyzeDistortionResult`, `AnalyzeDistortionsResult` and `PricingResult`,
three names that shipped long before the rule was written down and that are
public. One family spelled two ways reads worse than one family spelled against
the rule. Recorded in the module docstring as the deliberate exception it is,
rather than left to look like an oversight. If the author wants the five
renamed, that is a separate one line change and a CHANGELOG note.

The mixin is `SourcedMixin`, which does follow the `<Role>Mixin` rule.

### 3.7 There was no app caption to move for `pricing.calibrate`

Plan L5 says the Calibrate caption is "the current app text about calibrated
distortions, moved upstream and owned by the library". There is no such text:
`main.js` carries only the section **title** `Calibrated distortions`. So that
caption is new library prose rather than a move, and the author should read it
as new writing. The other two captions are genuine moves, from `main.js`
2406-2412 and 2613-2618, edited for the wider audience a library docstring has.

### 3.8 A portfolio allocation is always on the book's own total

`CalibrationResult.pricing_df` calls `analyze_distortions` with no
`reins_view`, because `analyze_distortions` refuses `gross` and `ceded` by
design (it would need a twin portfolio of gross units that the library does not
build). So a calibration made on `reins_view='gross'` for a book reads as **the
gross-calibrated set applied to the net book**, which is a real and deliberate
reading and not a mixed basis by accident. It is documented on the property.
The plan does not discuss the case; recording it so the app does not present
that combination as something it is not.

### 3.9 The relabel step drops the ordered categorical from a served index

Pre-existing behavior of `LabeledMixin._relabel`, not introduced here, but the
pricing exhibits are where a consumer is most likely to trip over it: the
`distortion` axis is an ordered `CategoricalIndex` on the frame and arrives as
plain strings in the served block, because `DataFrame.rename` cannot keep the
dtype. **Row order survives**, which is what the categorical was carrying.

For the app: read the canonical distortion order off the row order, never off
the dtype. Two tests here pass `check_categorical=False` for exactly this
reason and say so.

---

## 4. For the API and the SPA half

Nothing here needs LIB to move again; this is the list to code against.

1. **`calibrate_distortions` and `evaluate` return objects now.** `.distortion_df`,
   `.calibration_df`, `.evaluation_df`. The routes hold the result; the
   exhibits dispatch on it. `build_exhibit(result, 'pricing.allocate', ...)`
   is the whole call.
2. **The Calibrate route body maps straight onto the signature:** exactly one
   of `p` / `a`, exactly one of `coc` / `lr`, optional `basis` as `reins_view`.
   The library now owns the `lr` conversion, so `Pentagon.solve` in
   `pricing.py` deletes at A3 as the plan says.
3. **Three `ValueError`s become HTTP 400 with a readable `detail`,** not two:
   the L2 `p = 1` guard, the `lr` no-capital refusal (3.4), and the existing
   "exactly one of" validations. All are written to be shown to a reader as a
   sentence, and the preview line is the right place for the first two.
4. **`price_pentagon(reins_view=...)`** is what the preview line wants. Both
   legs of the anchor come off the named view, so the preview answers on the
   basis the reader chose. `price_pentagon_ex` gained the same keyword.
5. **The evaluate result carries its anchor.** `premium`, `reins_view`, `p`,
   `a` are all on it, so the Evaluate pane does not have to remember what it
   asked for. An anchored panel reports five families, an unanchored one four:
   `ccoc` is in `EVAL_FAMILIES_ANCHORED` only.
6. **Read the block list off `meta['blocks']`,** which is now a property of the
   source shape as well as the perspective. Section 2.4 is the table, but the
   envelope says it and the envelope is authoritative.
7. **Formats are resolved into the document.** `tables.FORMATS`'s six pricing
   keys have library equivalents (`PENTAGON_FORMATS`, `CALIBRATION_FORMATS`,
   `DISTORTION_FORMATS`, `EVALUATION_FORMATS`, `STAT_SLICE_FORMATS`) and the
   app does not need to name any of them. The stat slices use greater_tables'
   `float_format` rather than `formatters`, since their columns are units
   rather than statistics and the format is uniform over the frame.
8. **`reins_price_df` columns are the octet.** Anything in the app that reads
   `el`, `ask`, `bid` or `margin` from a served pricing frame is reading a
   column that no longer exists. `bid` has no replacement by ruling.
9. **The exhibit names carry a dot.** `pricing.calibrate` as a URL segment is
   fine; just do not assume a name is a Python identifier.

---

## 5. What LIB has not done, and does not think it should

- **No `PnL.evaluate` anchor.** Decision 4, deferred by the plan, and the
  per row position question is genuinely open: a ledger row's asset level is
  not obviously the book's, and a purchased layer's is not obviously anything.
  Worth its own conversation, not worth guessing.
- **No Bounds exhibit registration.** That stays with
  `dev/plan-exhibit-official-channels.md`, which this plan supersedes only for
  phases 8 to 10. It is that plan's last open item.
- **No natural allocation block.** `pricing.allocate` is designed with room for
  the gross to ceded and net allocation when
  `dev/plan-natural-allocation-to-occurrence-net-ceded.md` lands, and the
  builder is one `if` away from taking it. Out of scope here, as the plan says.
- **No `dev/FEATURES.csv` regen.** The file carries another agent's
  uncommitted refresh in the working tree, so regenerating it would have
  staged their work with mine. The pricing surface moved (five new or changed
  public signatures and two new public classes), so it **is** owed: run
  `uv run python dev/regen_features.py` once the in-flight refresh lands.
- **No `dev/TODO.md` edit staged.** Same reason: the entry it belongs in
  already carries an uncommitted edit from the plan's own cross-document
  ledger. The progress line is written into the working copy and left for the
  author to commit with theirs.
- **Three unused imports and five em dashes left uncommitted.** Ruff caught
  three `F401`s in the a259 and a263 files (`EvaluationResult` in core
  `_pricing.py`, `PENTAGON_STATS` and `pricing_calibrate` in
  `exhibits/_pricing.py`) and `results.py` carried five em dashes inherited
  from the file it replaced. Both are pure tidying, which by the house rule
  does not bump a version and so is not Claude's to commit; they sit in the
  working tree. The repo has 90 pre-existing ruff findings, so ruff is not a
  gate here, but these three are new and are mine.
- **No docs rebuild.** Per the house rule, `.rst` edits stay in lockstep and
  the build is the author's, outside the loop. Nothing here renamed a symbol
  the docs reference; the return type changes are described in the docstrings
  and the CHANGELOG.
