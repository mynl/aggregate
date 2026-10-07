# Plan a400: the v1.0 loose ends ([V1-Loose-Ends])

Status: draft for review, written 2026-10-07. Target version 1.0.0a400 onward
(current: 1.0.0a399). One bump and one commit per phase, per the house
[Release-Hygiene] rule.

Retargeted from a397 to a400 on 2026-10-07: the author landed a397
(`[PnL-Positions-And-Margin-Ratio]`), a398 (`[Reins-Annual-Triple]`) and a399
(`[Capstone-Capped]`) between this plan being written and being read. The file
stem moved with the target; nothing else in the plan depends on the number.

> The old 1,706-line `dev/TODO.md` was archived to
> `dev/done/TODO-2026-10-07.md` on 2026-10-07, in the same pass that wrote this
> plan, and a short live `dev/TODO.md` was started in its place, led by
> `[Portfolio-Shared-Mixing-Dependence]`. References to "the TODO" below are
> historical and resolve against the archive.

## Goal

Close the correctness items standing between `1.0.0a399` and the `1.0.0`
re-version commit. The release mechanics themselves (the version bump, the
classifier move, the tag, the merge, PyPI, the api's dependency floor) are
**not** here: they live in `V:\worktrees\BETA-MERGE.md`, which was rewritten on
2026-10-07 for a direct 1.0 cut with no beta.

These items were the live correctness entries in the `dev/TODO.md` "Beta gate"
section, which that file declared a release blocker. The author confirmed all
four on 2026-10-07, and ruled on phase 1 the same day in a way that reshaped it
from a fix into a contract (see there). Phase 4 is the one where this plan
recommends **against** the TODO's instruction, on evidence gathered below.

## Why these four and not the other twenty

`dev/TODO.md` carried 24 open items under a heading reading "Beta gate, blocks
`1.0.0b1`". Triaged against a direct 1.0 cut, the gate splits cleanly:

* **Correctness, and therefore here**: phases 1 to 3 below. Phase 2 puts
  silently wrong output in front of a user, which is the worst class of defect
  to freeze into a 1.0 API promise. Phase 1 turned out **not** to be a defect
  in its answer, only in the fact that the right answer is undocumented and
  unpinned; see the ruling there.
* **Cut mechanics**: phase 4, and it should be deferred (see there).
* **Everything else is documentation or polish** and genuinely post-1.0: the
  six `Docs & packaging` items (none has a code dependency, as that section
  says itself), `[Reporting-Guidelines]`, `[Bivariate-DecL-Label]`,
  `[Display-Surface-Punchups]`, `[Plotting-Punchups]`,
  `[Validation-Calc-Review]`, `[Library-Round-Two]`, `[Recipe-Library]`,
  `[Rationalize-Tests]`, `[Showcase-Examples-Tune]`,
  `[Session-Build-Clobbers-The-Trailer]`, `[Unparser-Reference-Gaps]`.

The 1.0 API promise in `CHANGELOG.md`'s preamble is about **names**, not
numbers: it says explicitly that "a correctness fix changes results within any
release". So phases 1 to 3 may move numbers, and that is in policy.

---

## Phase 1 [Outside-Grid-Is-Nan]: make the honest `nan` deliberate and documented

**LANDED `1.0.0a400`.** The author ruled 2026-10-07: report `0` where zero is
known, `nan` everywhere else, never a fabricated number. Implemented as
`aggregate.utilities.below_grid_fill`, wired into all four interpolators, and
pinned by `tests/test_cdf_outside_grid.py` (24 cases). The open sub-case below
was answered by that ruling: knowledge-based, not uniform.

### The ruling that shapes this phase

`dev/TODO.md` filed this as `[Windowed-Cdf-Nan]`, a bug, with a proposed
one-line fix of `fill_value=(0.0, 1.0)` or `(0.0, cum[-1])`.

**The author ruled on 2026-10-07 that the fix is wrong and the `nan` is right.**
Filling `0.0` below a window asserts that the law has no mass there, and
filling `1.0` above asserts the distribution is exhausted at the top of the
grid. Neither is known. A window is a *computational* restriction, not a
statement about the law, so outside it the mass is **uncomputed, not zero**. The
TODO entry's premise, "no mass below `x_min` by construction", holds for the one
`[Windowed-Grid-Breaks-Calibration]` repro it was diagnosed on and does not
generalize. Returning a number there would be making it up.

So this phase does not change the answer. It makes the answer **deliberate,
documented and pinned**, which it currently is not.

### Current behavior, measured 2026-10-07

Three regions, on `agg T 10 claims sev lognorm 100 cv 2 poisson` (ordinary
zero-based grid, grid top 262,140):

| region | `cdf` today | `sf` today |
|---|---|---|
| `x < 0` (below an ordinary grid) | `nan` | `nan` |
| below a window, `x < x_min` | `nan` | `nan` |
| inside the grid | correct | correct |
| `x` above the grid top | `0.9999999999991542`, the cumsum top | `8.46e-13` |

Two things worth noting against the TODO entry. First, the above-grid value is
**already** the flat-extrapolated cumsum top, not `1.0`, so the feared
`sf` collapse from `5e-12` to exactly `0` was only ever a consequence of the
proposed fix, not of today's code. Second, the `nan` is **not** deliberate: it
is an emergent artifact of `kind='previous'` with
`fill_value='extrapolate'`, because scipy's `previous` kind has no previous
knot below the first. Confirmed in isolation on scipy 1.17.1.

That is the actual defect. The behavior is right by accident, so a future
change of interpolator or scipy version could silently turn it into a
fabricated number, and nothing in the suite would notice.

### Sites

| site | line |
|---|---|
| `src/aggregate/_aggregate.py` | 7711 |
| `src/aggregate/_aggregate.py` | 7733 |
| `src/aggregate/_portfolio.py` | 2404 |
| `src/aggregate/_portfolio.py` | 2424 |

(The TODO cited `_aggregate.py:6357` and `_portfolio.py:2073`; the code has
moved since a289.)

### The change

1. **State the contract in the docstrings** of `Aggregate.cdf`, `Aggregate.sf`,
   `Portfolio.cdf` and `Portfolio.sf`: outside the computed grid the answer is
   `nan`, because the mass there was not computed, and `nan` is reported rather
   than a fabricated `0` or `1`. Say in `Notes` why, in one or two sentences,
   so the next reader does not file it as a bug a third time. `q` is unaffected
   and can say so.
2. **Pin it with tests**, per class, asserting `nan` below the grid and below a
   window. This is the part that has value: it converts an accident into a
   contract and makes any future silent change to a fabricated number fail the
   suite.
3. **Make the `nan` explicit rather than emergent** if it can be done without
   changing any in-grid value: an explicit `fill_value=np.nan` on the low side
   says what is meant, where `'extrapolate'` merely happens to produce it.
   Verify byte-identical in-grid results before and after; if that cannot be
   had cleanly, keep `'extrapolate'` and rely on the tests from step 2, and
   record the reason in the execution log.

### The one open sub-case, for the author

`cdf(-1)` and `cdf(-100)` on an **ordinary, nonnegative, non-windowed** grid
also return `nan` today. That region is different in kind from the others: an
aggregate of nonnegative claim amounts has exactly zero mass below zero, and we
**do** know it. Returning `nan` there is a missing answer rather than an honest
refusal.

Two defensible rules:

* **Uniform (recommended).** Outside the computed grid you get `nan`, always,
  with no special cases. It never claims anything, needs no branch on grid
  kind, and is one sentence to document. A user asking for `cdf(-1)` of a loss
  distribution is probably confused, and `nan` is a fair answer to a confused
  question. It also stays correct for signed aggregates, where negative values
  **are** in support.
* **Knowledge-based.** Return `0.0` below zero on a nonnegative grid, where it
  is genuinely known, and `nan` only where the mass is uncomputed. More
  informative, but it needs a branch on whether the grid is signed or windowed,
  and it reintroduces exactly the kind of case analysis that produced the
  original bug.

Recommend uniform. Confirm or override before implementing step 1, since it is
the sentence the docstring has to say.

### Acceptance

* The four docstrings state the outside-grid contract and say why.
* A test per class pins `nan` below the grid and below a window, and pins the
  above-grid value as the cumsum top, so all three regions are contractual.
* Every in-grid value is numerically identical to the pre-change build. Assert it.
* The numerics gate is green:
  `uv run pytest -m 'slow or not slow' -W error::RuntimeWarning`.

---

## Phase 2 [Unparse-Dense-Spec-Guard]: `spec_to_decl` silently emits wrong DecL

**LANDED `1.0.0a401`.** Two divergences from the plan as written. (1) The
discriminator is **structural**, not a sentinel-key list: a dense spec is the
constructor-argument dict and so contains every public `__init__` parameter,
which needs no maintenance and cannot false-positive (verified over all 780
corpus programs, widest parser spec 28 keys against 62 parameters). (2) Private
constructor parameters must be excluded: `Aggregate.spec` carries every public
argument but omits `_tweedie`, so the first cut of the subset test missed by
exactly one key and the guard silently did not fire. That near-miss is now its
own test.

### Current behavior

`decl_writer.spec_to_decl` documents that it takes the **sparse** parser spec.
Hand it the **dense** `Aggregate.spec` and it emits wrong DecL with no error
anywhere. The dense dict spells "unset" as `0` or `None` where the parser
simply omits the key, and `0` is legitimate for `exp_premium` and `sev_scale`,
so:

* `13.7376 claims` renders as `0 premium at 0 lr`,
* the severity picks up a `0 *` scale,
* a spurious `poisson 0 0 loss` appears.

The result **re-parses and builds**, to `est_m = 0` and `est_cv = nan`, with no
error raised. The only reason this is not already biting is that it first
raises `AttributeError` on `label_map`, because a present-but-`None` value
defeats the `.get('label_map', {})` default.

Found at a218 while checking whether a constructor-built object could be
decompiled.

### The change

**A guard, not a decompiler.** Detect the dense shape and raise, pointing the
caller at the sparse parser spec. A real object-to-DecL decompiler needs a
per-key inverse of the constructor's defaulting and is a separate, larger
question. Do not conflate the two.

`dev/plan-approximate-punchup.md` deferred the same gap as
`[Spec-Decompile-Robustness]` at a331; it folds in here rather than becoming a
second entry. Since a331 the `approximate` object mode is `build`-born and
carries a `program`, so no library path hands a user a program-less object. The
guard is for a user's own `Aggregate(**kwargs).spec`.

### Acceptance

* `spec_to_decl(Aggregate(**kwargs).spec)` raises a clear error naming the
  sparse spec as the expected input, rather than returning wrong DecL.
* The `label_map` `None` handling is fixed too, so the guard is what fires
  rather than an incidental `AttributeError`.
* Every existing sparse-spec call site is unaffected. The unparser corpora
  (`test_decl_unparser`, 444 cases) must stay green.

---

## Phase 3 [Far-Tail-Raw-Moment-Inflation]: DIAGNOSED 2026-10-07, fix needs a ruling

**The hunt is finished. The cause is not what the TODO recorded, and the fix
moves reported numbers, so it is the author's call whether it lands before
1.0.**

### What the TODO claimed, and what is actually true

The entry said higher **raw** moments degrade as the grid grows, blamed "a
survival function reaching the grid as ``1 - cdf``", and reported third raw
moment errors of 1.5%, 12.8%, 103%, 828% at ``log2`` 18 to 21.

Two corrections. First, the **raw** third moment is fine: measured against the
exact compound Poisson moments it is accurate to 2.3e-06, 1.4e-05, 1.6e-05,
6.5e-06 at those four grids. The mean error reproduces the recorded 6.4e-6
exactly, so this is the right program and grid. Second, there is no ``1 - cdf``
anywhere in it.

What *does* blow up is the third **central** moment, which is why it surfaces
as skewness: at ``log2 = 21`` the library reports ``est_skew = 7.537`` against
an exact ``3.5355``, a 113% error, and ``valid`` carries ``AGG_SKEW``. The raw
third moment hides this because it is dominated by the mean term, so the error
is invisible there. The recorded "103%" was the central moment all along.

### The mechanism, confirmed by arithmetic

A two-bug interaction, neither of which is wrong on its own.

1. **``utilities.remove_fuzz`` eats genuine density on a tall grid.** It zeroes
   ``|x| < eps`` with ``eps = np.finfo(float).eps``, an **absolute** 2.22e-16.
   Its docstring asserts "the exact aggregate has no genuine density below
   machine epsilon". On ``lognorm 200 cv 2`` at ``bs = 6.427``, ``log2 = 21``
   the genuine far-tail density is legitimately around **1e-26**, so that
   premise is false: the call zeroes **1,977,176** of 2,097,152 buckets and
   removes a net **+4.632e-12 of real mass**. The comment at
   ``_aggregate.py`` line ~4555 repeats the claim, and adds that the aggregate
   "has no negative density even under aliasing", which is also false here:
   795,744 buckets are negative at that grid.
2. **``moments.xsden_to_mwrangler`` relocates the shortfall to the worst
   possible place.** It computes ``pg = 1 - den.sum()`` and adds
   ``pg * (xs[-1] + bs) ** k`` to the k-th moment, placing any deficit at the
   *implied maximum loss*. That is right for genuine right-truncation. Here the
   "deficit" is fuzz-removal spread across the whole tail, and the top of the
   grid is 13,478,396, whose cube is 2.449e21.

The product is the whole error, exactly:

```
pg * xsm**3  =  4.632e-12 * 2.449e21  =  +1.134e10      on an ex3 of 3.0e10
```

Measured, same object: ``sum(d * x**3)`` is 2.99962e10 from the cleaned
density, the phantom term adds 1.13420e10, and the reported ex3 is 4.13382e10.
Feed the **raw** density to the same wrangler and ``pg = 0``, ex3 is 3.00002e10
and the skew is 3.5354, correct. The ``if pg > VALIDATION_NOISE`` gate does not
help: it guards only the log message, and ``4.6e-12`` exceeds the ``1e-12``
floor anyway.

### Why this is not fixed in this pass

Any of the three candidate fixes **moves `est_skew` and can move validation
flags across the corpus**, hours before a 1.0. That is the author's decision,
not a plan's.

* **(A) Make ``remove_fuzz``'s threshold relative** to the density's own scale
  rather than an absolute epsilon of 1.0. This is the root fix and the
  recommendation: the function's premise is simply wrong for a tall grid. It
  touches every ``remove_fuzz`` caller, so it needs its own blast-radius pass.
* **(B) Take the aggregate moments from the raw density**, dropping the
  ``remove_fuzz`` call at ``_aggregate.py`` line ~4568. Measured more accurate
  at every grid tested (``log2 = 18``: raw -0.004% against cleaned +0.176%;
  ``log2 = 21``: raw -0.002% against cleaned +113%). But ``remove_fuzz`` was
  introduced *because* far-tail fuzz corrupted the skew, so there is a case
  this would regress that was not found in this session. Do not take it
  without finding that case.
* **(C) Renormalize after fuzz removal**, or suppress the ``pg`` adjustment
  when the shortfall is attributable to fuzz removal rather than truncation.
  Narrowest change, but it treats the symptom: the removed mass was spread
  through the tail, and the bug is that it gets relocated to the top.

Recommendation: **(A)**, as its own plan after 1.0, with (C) as the cheap
interim if the author wants ``est_skew`` honest for 1.0. Either way the two
false comments should be corrected now, since they assert things this session
measured to be untrue.

### Acceptance, revised

Phase 3 delivers the **diagnosis**, which is complete and recorded above. The
fix is deferred to `dev/TODO.md` under the corrected label. No version bump:
nothing in the library changed.

## Phase 4 [Scaffold-Retirement]: recommend DEFER past 1.0

`dev/TODO.md` filed this under "At the cut" with the instruction "do at the
`1.0.0b1` cut". **This plan recommends not doing it before 1.0**, on the
following evidence gathered 2026-10-07.

### What the TODO entry assumed

It names four test dependents: `tests/data/expected_specs.json` plus
`capture_spec_snapshot.py`, `test_decl_parser.py`, `test_splice_suite.py`, and
the `conftest` fixtures.

### What is actually there

**Eleven test files** read `_test_suite`, not four:

```
test_decl_parser       _test_suite                                   (only this)
test_decl_unparser     _test_suite _test_suite2 decl-testers.agg
test_grammar_sync      _test_suite _test_suite2 decl-testers.agg library.agg
test_splice_suite      _test_suite _test_suite2
test_agg_libraries     _test_suite _test_suite2 decl-testers.agg library.agg
test_underwriter       _test_suite decl-testers.agg library.agg
test_feasibility       _test_suite library.agg
test_agg_as_severity   _test_suite decl-testers.agg
test_cantor_severity   _test_suite decl-testers.agg
test_grammar_ambiguity _test_suite decl-testers.agg
test_renewal_decl      _test_suite decl-testers.agg
```

Plus five non-test consumers: `src/aggregate/config.py`
(`TEST_SUITE_FILENAME = '_test_suite.agg'`), `src/aggregate/underwriter.py`
(`test_suite_file`, and `interpret_file`'s default), the shipped
`src/aggregate/data/config.default.toml`, `docs/4_dec_Language_Reference.rst`,
and both `scripts/bucket_baseline.py` and `scripts/freeze_knowledge.py`
(`DEFAULT_DATABASES = ("_test_suite",)`).

Four of those test files collect **1,614 test cases** between them, and
`test_decl_parser` reads the scaffold and nothing else, so deleting
`_test_suite.agg` deletes that file's entire 326-case parser corpus.

### Why the stated benefit does not justify that

The concrete shipping consequence is that `agg/*.agg` is declared package data,
so the scaffold does go into the wheel. That is **27,075 bytes**
(`_test_suite.agg` 26,079 plus `_test_suite2.agg` 996), in a wheel that already
ships `decl-testers.agg` at 131,625 bytes and `library.agg` at 68,337 bytes
deliberately. Shipping 27 KB of extra DecL text alongside 200 KB of it is not a
defect a user can observe.

Set against that: rewiring eleven test files and five other consumers, putting
1,614 test cases at risk, in the hours before a release. `dev/TODO.md` also
records that `[Rationalize-Tests]` must be done **first** ("Overlaps
`[Scaffold-Retirement]`, do them in that order"), and that item is itself open.

### What to do instead, and it is cheap

The one part of the entry that is a live defect today is its last sentence,
which predicted that `CLAUDE.md` would go stale. It has:

* `CLAUDE.md`'s **Testing** section says "Every line of
  `aggregate/agg/test_suite.agg` is exercised as its own parametrized test
  case". **There is no file of that name.** The scaffold is `_test_suite.agg`,
  the shipped libraries are `library.agg` and `decl-testers.agg`.
* The same section calls the SLY snapshot the primary test mechanism, which the
  file's own "Running the suite efficiently" section contradicts.

Fix that section to describe the suite as it is. It is documentation only, so
it carries no version bump, and it removes the misdirection that would
otherwise mislead the next reader of this repo.

### The real leak, found while checking this, and fixed

The author's question on 2026-10-07 was whether SLY references leak into
user-visible places if the scaffold stays. They did, and not in the `.agg`
files, which are clean DecL with category headers and no mention of SLY. The
only SLY in shipped package data is a provenance comment in `decl.lark`
explaining the Earley migration, which is legitimate and worth keeping.

The leak was in **published documentation**.
`docs/4_dec_Language_Reference.rst`'s "Test Suite Programs" section told readers
the mechanism was a "parse check + **SLY-snapshot** shape check". SLY is gone;
the snapshot is captured from the current Lark parser. The same section also
told readers to call `build.interpreter_file(...)`, which does not exist: the
method is `interpret_file`. Both fixed 2026-10-07, and the section now describes
the corpus and the snapshot's actual status as a change detector rather than a
correctness oracle.

That is the whole user-visible cost of keeping the scaffold, and it was fifteen
lines of docs rather than a sixteen-file rewiring.

Then move `[Scaffold-Retirement]` to `dev/TODO.md`, sequenced behind
`[Rationalize-Tests]`, for 1.1. **Done 2026-10-07**: both are entered there
under "Tests and the parity scaffold" with the evidence summarized.

---

## Proposed additions, for the author's ruling

Not in the confirmed four. Raised here because they were found while triaging
and each is cheap.

### [Zero-Share-Layer-Moments]: a `nan` in a reported frame

Currently filed under `dev/TODO.md` "After the cut", which this plan argues is
the wrong section: it is the **same defect class as phase 1**, a `nan` reaching
a user-visible answer.

A `0% po` layer in an occurrence program
(`occurrence net of 20 xs 20 and 0% po 20 xs 40 and inf xs 60`) cedes nothing,
so its ceded second moment lands as float noise just below zero.
`moments.mcvsk` takes the square root of that, emits
`invalid value encountered in sqrt`, and `reins_stats_df` reports the layer's
`cv` and `skew` as `nan` beside a mean of `-1.8e-11`.

`CLAUDE.md`'s own numerics-gate rule says a `nan` that reaches an answer wants
a fix, not an `np.errstate` guard. The fix is to clamp a variance that is noise
below zero to zero in the reinsurance moment path, plus a decision on whether a
zero-share layer reports zeros or blanks across the column. Found at a352;
currently masked because the gapped program is exercised on the declared tier
only (see the `_EVERY_TIER` comment in `tests/test_chart_structure.py`).

### Two front-page items, for launch day

Both from the never-triaged `dev/TODO.md` section "From the beta-gate review,
folded in 2026-07-27", which is the condensed record of the objective library
review taken at a89.

1. **`[README-Scope-Statement]`**: say what the library is **not**, on the
   front page. Loss development and IBNR, triangles, stochastic reserving,
   credibility, GLM and experience rating, multi-year dynamics, inflation and
   trend, and cat-model internals are all correctly out of scope. The review's
   argument is that stating so is the best defense against unfair
   missing-feature critiques, and 1.0 is when those arrive. Name the seam too:
   cat-model output enters cleanly as empirical `dsev` or histogram severities.

   This is also the honest home for the review's **number one** functional
   finding, `[Portfolio-Shared-Mixing-Dependence]`: portfolio units combine by
   independent convolution, and the dependence that exists is scattered across
   three mechanisms that do not compose (the two-peril copula in
   `bivariate.py`, sample-only Iman Conover, allocation-only comonotonic). The
   review called it "the gap users will find first" in a capital-allocation
   tool. It cannot be built before 1.0. It can be stated, which turns a
   discovered limitation into a declared scope boundary.

2. ~~**`[PIR-Reproduction-Documented]`**~~ **DONE 2026-10-07, by the author.**
   The README now carries a "Pricing Insurance Risk Examples" section saying
   those exhibits re-create under 0.30.1, linking the Baseline release tag and
   the blog post on building PIR exhibits for a custom portfolio, and saying
   why the functionality was dropped. Nothing further owed here.

~~The `[README-Stable-Body]` copy fix~~ **DONE 2026-10-07, by the author.** The
README's Purpose paragraph now reads "builds **essentially exact** compound
(aggregate) probability distributions", replacing "builds approximations to",
which had fought the exact-not-approximate claim the library rests on.

---

## Sequencing

Phases 1 and 2 are independent and both small. Phase 1 needs only the
uniform-versus-knowledge-based sub-ruling, which affects one documented
sentence, not the design. Phase 3 is a hunt and should start early so there is
time to decide whether it makes the cut. Phase 4's cheap half (the `CLAUDE.md`
fix) is independent of everything.

Recommended order: phase 1 and phase 2 in either order, phase 3 running
alongside, phase 4's documentation fix whenever, and the proposed additions on
the author's word.

## Release hygiene

Per `CLAUDE.md`: each phase bumps the version in `pyproject.toml`, adds its
one-paragraph `CHANGELOG.md` section, and is committed by Claude as a single
coherent commit with a one-line subject in the form
`[V1-Loose-Ends] a400: <terse summary>`. No body, no trailers. The phase 4
documentation fix carries no bump and is left uncommitted for the author.

A short live `dev/TODO.md` was started on 2026-10-07 and already carries the
items this plan defers (`[Scaffold-Retirement]` behind `[Rationalize-Tests]`,
and `[Far-Tail-Raw-Moment-Inflation]` / `[Zero-Share-Layer-Moments]` as
conditional entries). Tick those as phases land, per `CLAUDE.md`.

This plan moves to `dev/done/` only when the author says done.
