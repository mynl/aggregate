# Plan: refuse severity numbers a family cannot take, and repair the zero-shape mean cv form

`[Severity-Parameter-Count]`. Drafted 2026-10-29 against aggregate 1.34.0; rechecked
2026-10-31 against 1.43.1, line numbers and corpus scan updated, and the three
corrections of the review of 2026-10-30 applied (marked **2026-10-31** below).
**Status: executed at 1.51.0, 2026-11-01**; see the execution log at the end. The
mixed-vector question was ruled the other way from the recommendation
(**2026-11-01** below): a placeholder `_`, not a `gamma` rewrite. It is
the `agg` half of a rule ruled the same day (author, 2026-10-29): passing a
parameter a family does not take is an error, a shape for a zero-shape family,
two shapes for a one-shape family and so on, in a severity and in a `chfagg`
alike. The `chfagg` half is `[CHF-Arithmetic-Form]` in
`dev/done/plan-characteristic-function-aggregate-parity.md`.

**Size.** A small plan: about 250 lines including docstrings and tests, one
step, one minor bump. Nothing gets slower. It refuses input that parses today,
so the CHANGELOG entry has the second paragraph a breaking change earns.

## Goal

1. A `sev` clause whose positional numbers do not match the family's shape
   count is refused at parse time, with a message giving the spelling that
   does what was probably meant (`25 * gamma 4`, `30 * norm + 100`).
2. A `mean cv` form the family cannot honor is refused rather than returning
   `nan`.
3. The `mean cv` form of a zero-shape family (`norm`, `expon`, `uniform` and
   the rest) gives the declared mean and coefficient of variation, as it
   already does for a one-shape family.
4. The Python API refuses the same mistakes when the values make them
   visible.

## Current behavior, 1.34.0 (measured 2026-10-29)

Each clause was built in `agg T 1 claim sev ... fixed` with `update=False`,
and the frozen law's mean and standard deviation read off:

```
sev norm 100 30           mean   0     sd 1         both numbers dropped
sev expon 5               mean   1     sd 1         dropped
sev uniform 0 10          mean   0.5   sd 0.289     dropped
sev gamma 4 25            mean   4     sd 2         the 25 dropped
sev norm 100 cv 0.3       mean 100     sd 1         warning: achieved cv 0.01
sev 2 * norm 100 cv 0.3   mean 200     sd 2
sev expon 10 cv 1         mean  11     sd 1
sev uniform 10 cv 0.2     mean  10.5   sd 0.289
sev beta 2                mean nan     sd nan
sev beta                  mean nan     sd nan
sev burr 2                mean nan     sd nan
sev burr 100 cv 0.5       mean nan     sd nan
sev cauchy 10 cv 1        mean nan     sd nan
sev gamma                 ValueError: sev_a not set and sev_cv=0 is invalid
```

For comparison, the one-shape `mean cv` path is right and composes with the
arithmetic: `sev 2 * gamma 100 cv 0.5 + 10` has mean 210 and standard
deviation 100.

**Causes.**

- The parser (`src/aggregate/parser.py:2793` to `2809` at 1.43.1) records what was written:
  `sev0_zero_params` no numbers, `sev0_one_param` one in `sev_a`,
  `sev0_two_params` two in `sev_a` and `sev_b`, `sev0_mean_cv` the mean and
  CV, each with `sev_scale` 1.0. It has the count but not the family's.
- `SeverityScipy._build` (`src/aggregate/_severity.py:2854`) has the
  family's shape count from `scipy.stats` and reads only what that count
  needs: the zero-shape branch uses `sev_loc` and `sev_scale` and ignores
  `sev_a` and `sev_b`; the one-shape branch ignores `sev_b`; the two-shape
  branch takes whatever `sev_a` and `sev_b` hold, including the defaults
  `nan` and 0.
- The zero-shape `mean cv` path is `loc = mean`, then `scale = cv * loc` only
  if `sev_scale == 0`. `sev0_mean_cv` sets `sev_scale` to 1.0, so the scale is
  never derived. The rule `loc = mean`, `scale = cv * mean` would also be
  wrong for any family whose standard law has a mean other than 0 or a
  standard deviation other than 1: `expon 10 cv 1` would come out with mean
  20.

**Blast radius.** A scan of every statement in `src/aggregate/agg/*.agg`
(906 severity clauses; the four `sev0_*` callbacks wrapped to record each
family and the count written, then every statement parsed) finds nine that a
strict count refuses, every one writing `expon 1`, where the dropped number
happens to be the default:

| file | line | clause |
|---|---|---|
| `_test_suite.agg` | 92 | `sev 100 * expon 1 + 10` |
| `_test_suite.agg` | 168 | `sev 100 * [lognorm expon] [.5 1] wts [0.6 .4]` |
| `_test_suite.agg` | 169 | `sev [50 100] * [lognorm expon] [2 1] + 10 wts=2` |
| `_test_suite.agg` | 170 | `sev [50 100] * [lognorm expon] [2 1] + 10` |
| `_test_suite.agg` | 287 | `[2.764e3 ...] * expon 1 wts [...]` |
| `library.agg` | 375 | `sev 100 * [lognorm expon] [0.5 1] wts [0.6 0.4]` |
| `recitations.agg` | 299 | `sev [200 200] * [expon pareto] [1 3] + [0 -200] wts [...]` |
| `recitations.agg` | 306 | `sev 1000 * expon 1 !` |
| `reproductions.agg` | 990 | `sev expon 1` |

No corpus clause uses a zero-shape `mean cv` form, a two-shape family with
too few numbers, or a two-shape `mean cv` other than `beta`. `rg` over `docs`,
`tests` and the writing tree `D:/projects` (`.qmd`, `.md`, `.py`, `.agg`,
`.ipynb`) finds none of the refused spellings either.

The mixed vectors are the one case a strict count makes awkward: a vector of
families shares one list of numbers, so `[lognorm expon] [0.5 1]` has to give
`expon` a number it cannot take. For the exponential there is an exact
rewrite, `[lognorm gamma] [0.5 1]`, since the exponential is the gamma with
shape 1; the two build identical severities. See the question at the end.

## Design

**The rule.** A family has a shape count from `scipy.stats` (`gen.shapes`),
or from the library for its own severities (`cantor` takes one, optional, with
the default 1/3). A `sev` clause writes either that many positional numbers,
or `mean cv v`; scale and location are the arithmetic `c *` and `+ b`, as
today. Anything else is refused:

| written | family | message, in substance |
|---|---|---|
| numbers on a zero-shape family | `norm 100 30` | norm has no shape parameter; for location 100 and scale 30 write `30 * norm + 100`, or `norm 100 cv 0.3` |
| one number on a zero-shape family | `expon 5` | expon has no shape parameter; for scale 5 write `5 * expon`, for a shift `expon + 5` |
| too many on a one-shape family | `gamma 4 25` | gamma takes one shape parameter; for shape 4 and scale 25 write `25 * gamma 4` |
| too few | `beta 2`, `burr` | beta takes two shape parameters, got one |
| `mean cv` on a two-shape family other than `beta` | `burr 100 cv 0.5` | burr has no mean cv form; give its two shapes |
| `mean cv` on a zero-shape family with no finite mean or variance | `cauchy 10 cv 1` | cauchy has no finite mean, so no mean cv form |
| a number for a family that takes none, in a vector (**2026-11-01**, replaces the `gamma 1` rewrite) | `[lognorm expon] [0.5 1]` | expon has no shape parameter, and in a vector of families `_` holds its place: write `[lognorm expon] [0.5 _]` |

For a two-number zero-shape clause the suggestion follows scipy's positional
order, location then scale, which is how `ss.norm(100, 30)` reads. A refusal
is a `ValueError` raised from the transformer, so it arrives with the DecL
position as every other parse error does. Nothing is reinterpreted.

**Where.**

- `src/aggregate/_severity.py` gains `severity_shape_count(name)`, returning
  `(required, allowed)` or `None` for a name it cannot resolve (left to
  `SeverityScipy` to refuse as today): `(k, k)` for a scipy family with `k`
  shapes, `(0, 1)` for `cantor`. The family word list in `_classify_sev`
  stays the one source of what is not scipy.
- `parser.py`: the four `sev0_*` callbacks call one private check,
  `_check_severity_count(ids, count, form)`, per family in `ids` (a list for
  a vector), which raises with the table's messages. The arithmetic is applied
  later by `sev1_scaled` and `sev2_add`, so the check gets exactly the numbers
  written after the family word.
- `SeverityScipy._build`, for the Python API: a zero-shape family with a
  finite `sev_a`, a family of at most one shape with a nonzero `sev_b`, and a
  two-shape family with `sev_a` still `nan` are refused, in short messages
  without DecL spellings. An explicit `sev_b=0` cannot be told from the
  default and stays accepted, which is harmless because it is dropped
  unchanged. Library code never passes either value to a family that cannot
  take it (`rg` 2026-10-31: `_aggregate.py:2617`, the `approximate` fit,
  passes `sev_a=nan` for `norm` and `sev_b=0.0`).

**The zero-shape `mean cv` repair.** With $m_0$ and $s_0$ the mean and
standard deviation of the family's standard law (`gen.stats(moments='mv')`),
a declared mean $\mu$ and coefficient of variation $v$ give

$$\text{scale} = v\mu/s_0, \qquad \text{loc} = \mu - \text{scale}\cdot m_0,$$

and the clause's own shift is added to `loc` after, as the one-shape path adds
it. `norm 100 cv 0.3` is then N(100, 30); `expon 10 cv 1` is the exponential
with mean 10; `expon 10 cv 0.5` the exponential with scale 5 shifted by 5;
`uniform 10 cv 0.2` the uniform on $[10 - 2\sqrt3, 10 + 2\sqrt3]$. The
placeholder `sev_scale` of 1.0 that `sev0_mean_cv` writes is not read: a
multiplier is already folded into the mean by `sev1_scaled`, as it is for a
one-shape family. A reflected severity (`sev_reflect`) builds the positive
base at location 0 and reflects it, as today. The Notes of `_build` state the
formula and why the earlier rule was wrong for `expon`.

**Corpora.** The nine clauses above are rewritten: the four scalar ones drop
the `1` (`100 * expon + 10`, `1000 * expon !`, `expon`, and the vector of
scales over `expon`), the five vectors write `gamma` for `expon`. The rewrite
is exact: `agg A 8 claim sev 100 * [lognorm expon] [.5 1] wts [0.6 .4] fixed`
and its `gamma` twin pick the same grid and agree to $7\times10^{-20}$ against
a largest bucket of $2.4\times10^{-4}$. Then
`tests/capture_spec_snapshot.py`, which covers `_test_suite.agg`, is rerun and
the diff read: `D.Sev02` and `N.Comm.Auto.MED` lose `sev_a`, `G.Mixed09`,
`G.Mixed10` and `G.Mixed11` change `sev_name`, and no other statement moves.
**2026-10-31:** the corpus scan rerun at 1.43.1 finds the same nine clauses
(`_test_suite.agg` 93, 169, 170, 171, 288; `library.agg` 375;
`recitations.agg` 299, 306; `reproductions.agg` 990) and no new one.
**2026-11-01:** the five vectors keep `expon` and write `_`, as `[lognorm expon] [.5 _]`;
the snapshot's `sev_a` entry for `expon` goes to `nan` and no `sev_name` changes.

**Docs.** A paragraph in `docs/4_dec_Language_Reference.rst` beside the
arithmetic paragraph (`:98`): positional numbers are the family's shapes in
scipy's order, scale and location are written with `*` and `+`, and the
`mean cv` form, with the zero-shape formula. The generated grammar reference
is regenerated if its severity text describes the counts. The docs build is
left to the author.

## Files

`src/aggregate/parser.py`, `src/aggregate/_severity.py`, the four corpus
files above, `tests/data/expected_specs.json` (regenerated),
`docs/4_dec_Language_Reference.rst`, a new `tests/test_severity_parameter_count.py`,
`CHANGELOG.md`, `dev/TODO.md`, and the version in
`src/aggregate/__init__.py`.

## Acceptance

- `norm 100 30`, `expon 5`, `uniform 0 10`, `gamma 4 25`, `beta 2`, `beta`,
  `burr 2`, `burr 100 cv 0.5`, `cauchy 10 cv 1`, `gamma` and
  `[lognorm expon] [0.5 1]` raise `ValueError` at parse time, and each message
  contains the suggested spelling where the table gives one.
- Each suggested spelling builds the law the message names: `30 * norm + 100`
  has mean 100 and standard deviation 30, `25 * gamma 4` mean 100 and
  standard deviation 50, to $10^{-12}$.
- `norm 100 cv 0.3`, `2 * norm 100 cv 0.3`, `expon 10 cv 1`,
  `expon 10 cv 0.5 + 3` and `uniform 10 cv 0.2` have the declared mean (plus
  shift) and coefficient of variation to $10^{-12}$ **read from the frozen law**
  (`fz.stats(moments='mv')`), with no warning. **2026-10-31:** the severity's own
  `sev_m` and `sev_cv` differ for `norm 100 cv 0.3`, because 0.04% of the law sits
  below zero and the severity clamps it (`30 * norm + 100`, the same law, has
  `sev_m` 100.0034 and `sev_cv` 0.29987); the check reads the frozen law, as the
  measurements above do, and the run confirms the achieved-CV warning stays quiet.
- **2026-10-31, known false failure, not this plan's:** `100 * norm + 1000`, the
  same law as `norm 1000 cv 0.1`, reports `fails sev skew, agg skew` because
  validation skips the skewness test only when the exact value is below
  $10^{-12}$ and a skewness that is zero to rounding can sit above it. Blessing
  the `mean cv` form makes it more visible; `[Validation-Verdict]` removes it.
- Each rewritten corpus program gives the same aggregate density as its
  1.34.0 spelling, bucket for bucket, to $10^{-15}$.
- `Severity('gamma', sev_a=4, sev_b=25)` and `Severity('norm', sev_a=100)`
  raise; `Severity('gamma', sev_a=4, sev_scale=25)` builds.
- The spec snapshot diff is the five `_test_suite.agg` lines above and
  nothing else.
- Tier 3, `uv run pytest -m 'slow or not slow'`, passes.

## Question for the author

**2026-10-31:** the author's instruction to run the plan as written is read as
taking the recommendation below (refuse, with the `gamma 1` rewrite in the
message); confirm at the review before execution.

**The mixed-vector case.** A shaped vector with a zero-shape member, such as
`[lognorm expon] [0.5 1]`, has to give that member a number it cannot take.
**Recommend refusing it** as the rule requires, with the `gamma 1` rewrite in the
message. That covers every vector in the corpora and the writing tree. The
cost is that a shaped vector mixing in a zero-shape family with no one-shape
twin, `[lognorm norm] [0.5 ?]`, can no longer be written. The alternative
accepts a number for a zero-shape member of a vector only, which keeps that
spelling but is a silent drop again, in one place.

**Ruling, 2026-11-01.** The author declined both. Refusing with a `gamma` rewrite has no substitute for the normal, and accepting a number is the silent drop the rule exists to stop. The author asked for "a symbol that means DUMMY_PARAMETER", with which the user says "I know what is going on, and there is no cheating involved", and chose `_`, subject to its parsing.

## Execution log, 1.51.0, 2026-11-01

Executed against 1.50.0 in one step and one bump. The rule, the zero-shape `mean cv` repair and the acceptance list landed as specified, with the divergences below.

**The placeholder `[Shape-Placeholder]`, from the ruling.** The grammar gains `shapes`, `shapes_held` and the terminal `SHAPE_HOLE` (`_`), reached only from `sev0`, so it cannot enter an exposure, weight, limit or `mean cv` list, and a test confirms each of those is a parse error. A list reaches `shapes_held` only if it holds a `_`, so a plain list stays on `numbers` and the parse is unambiguous. `_` parses to `nan`, the value of an unset `sev_a`; the engine needed no change, and a test run through the Python API before any edit gave a largest bucket difference of 0.0 against the old spelling. `_` works in the second list as well, `[lognorm beta] [0.5 2] [_ 3]`. Also added: `decl_writer._fmt_shapes` so programs round-trip, a label in `parser_errors`, and a rule in each colorer (`decl_pygments`, `agg.sublime-syntax`), both required by `tests/test_grammar_sync.py`.

**Size.** About 410 lines of source and 320 of tests, against the plan's estimate of 250 in all and the estimate of 50 more for the placeholder given when it was ruled. The excess is mostly the messages, which build a corrected form for each refusal, and the docstrings. Nothing gets slower.

**Check signature.** The plan's `_check_severity_count(ids, count, form)` became `_check_severity_count(ids, shapes)`, because a vector is checked by column and needs the values, with `_check_severity_mean_cv(ids)` for the moment form. Too few is checked before too many across a vector, so a suggested spelling is not itself refused. New helpers in `_severity.py`: `severity_shape_count` as planned, `standard_law_moments`, `_scipy_shape_count`. `rg` found no collision for any of them. A list of the wrong length for a vector of families now raises at parse; it used to fail later, at the broadcast in the build.

**Goal 2 widened.** The plan's table lists the families with no `mean cv` form. Measuring neighbors at 1.50.0 found the same fault by value, so these also raise, at build since they depend on the values: `norm -100 cv 0.3` gave N(0, 1), `norm 100 cv 0` gave N(100, 1), `beta 2 cv 0.2` and `beta 0.5 cv 5` gave `nan`, and a failed shape solve gave `nan` (`t 100 cv 0.5`, `pareto 100 cv 0.5`, `lomax 100 cv 0.5`, `skewnorm`, `loggamma`) or a degenerate law (`invweibull`). `_cv_to_shape` runs its solve under `np.errstate` and logs at info, since the refusal now reports the failure. `pareto 100 cv 0.5` is attainable at $a = 1 + \sqrt{5}$ and fails only because the solve starts at shape 1, where the Pareto has no mean: `[Mean-CV-Solver-Start]` in `dev/TODO.md`.

**Python API widened.** As a result `Severity('norm', sev_mean=100, sev_scale=30)` and `Severity('norm', sev_cv=0.3, sev_loc=100)` raise. Both gave the intended law with a spurious printed warning. `sev_mean` and `sev_scale` alone on a zero-shape family cannot be told from `norm 100 cv 0`, which arrives with the placeholder `sev_scale` of 1. `rg` over `src`, `tests` and `V:/dev` finds no caller.

**`self.sev_loc` stays the shift.** The old zero-shape branch wrote the derived location to `self.sev_loc`, which `_validate_moments` then adds to the target, double counting. The location now goes only into the frozen law, as on a one-shape family, which is what keeps the achieved-CV warning quiet. `Severity._unshifted_law` reads the frozen law at 1.50.0, so `fz0` and `_shift` needed no change.

**A glued minus is not refused.** `gamma 4 -10` and `expon -5` resolve to the shift in a `sev` clause, the same at 1.50.0 and after, compared against an export of HEAD. The law is the intended one, so nothing is refused. `chfagg` resolves the same text to two numbers and refuses it. `monograph/.../_040_severity.qmd` and the published paper say a negative shift must have a space, which is out of date for `sev`.

**Blast radius, correcting the plan.** The plan's claim that `docs`, `tests` and the writing tree hold none of the refused spellings no longer held at 1.50.0. Rewritten here: `tests/test_tail.py` and `tests/test_renewal_agg.py`, where `test_tail.py:159` wrote `expon 100` and so had been testing a standard exponential, and the commented copy at `_test_suite.agg:287`. The live sources elsewhere (author, 2026-11-01: all of them, program text only) are listed below. `reproductions.agg` and `recitations.agg` are copied into `src/aggregate/agg` at release from `D:/Projects/aggregate` (`V:/worktrees/release-agg-sources.txt`), so a rewrite here alone would not survive a release.

**Control lists.** `V:/worktrees/dev-files.md` no longer exists, so there is no row to update. `decl-keywords.json` is gone from `aggregate-api` (`tests/test_grammar_sync.py` gives `decl_words.decl_words` as its replacement, and `_` is not a word), so the app owes nothing.

**Acceptance as measured.** The nine rewritten programs against their old spellings at 1.50.0, on fixed grids: largest bucket difference 0.0 on all nine, which is stronger than the $10^{-15}$ asked, and against 1.50.0 rather than 1.34.0, since the old spelling no longer parses. In the suite the oracle is the `gamma 1` twin, to $10^{-15}$. The snapshot diff is the five `_test_suite.agg` lines and nothing else. Frozen-law moments match to $10^{-12}$ with no printed warning.

**Gates.** Tier 3, `.venv/Scripts/python.exe -m pytest -m 'slow or not slow'`: 10382 passed, 1 skipped (`plotly` not installed). Numerics, the same with `-W error::RuntimeWarning`: 10382 passed, 1 skipped.

**Live sources rewritten outside this repository** (program text only, each committed in its own repository): `aggregate/reproductions` 1.28.1 (`Embrechts1993.qmd`, whose `sev expon {1/BETA:g}` only worked because `BETA` is 1, `reproductions.agg`, `_programs/Embrechts1993.json`); `aggregate/recitations` 1.13.2 (`Frees2018a.qmd`, `recitations.agg`, `_programs/Frees2018a.json`); `aggregate/education` (`_slides/decl-reference.qmd`, `_slides/m09-probability-of-eventual-ruin.qmd`); `aggregate/monograph` (`_060_mixed_severity.qmd`, `_070_vectorization.qmd`); `Using/aggregate` (`pk-wh.qmd`, `ruin.qmd`), with the new page `severity-parameters.qmd`.

**Left for the author.** Prose was out of scope, so these stay as they are: the callout at `monograph/posts/3_user_guides/DecL/_060_mixed_severity.qmd:134` ("it is necessary to enter a dummy shape parameter 1 for the exponential ... a known bug in the parser"), which is now false; `_040_severity.qmd:173` on a negative shift needing a space; and the listing `sev lognorm 50 1.75` at `_080_reinsurance.qmd:43`, whose intent (mean and cv, or shape and scale) is the author's to say, and which is refused if run. Frozen and untouched: the published paper's `[expon pareto] [1 2.5]`, the older blog posts, rendered output, and the blog's copy of `reproductions.agg`, which `update-reproductions` refreshes.
