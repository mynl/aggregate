# Plan: the validation verdict covers the stable mean, the reinsured views and undefined moments, and a failing build warns

`[Validation-Verdict]`. Drafted 2026-10-31 against aggregate 1.43.1 (`03aaec6`). Four minor
bumps: `[Stable-Completed-Mean]`, `[Verdict-Reinsured-Views]`, `[Verdict-Moment-Floors]`,
`[Validation-Warning]`. Agreed by the author on 2026-10-30
(`D:\Projects\Using\bugs\aggregate-review-answers-1.43.qmd`, items 6 and 1). It runs after
`[Lattice-Grid-Sizing]`, `[Shifted-Severity-Moments]` and `[Using-Notes-Wrong-Answers]`, which
remove most of the false alarms a warning would otherwise raise.

## Goal

The verdict (`validation_description`) is the one line a reader checks. After this plan:

- a tilted stable law with a mean reads `not unreasonable` (today `fails agg mean` by the tail
  mass beyond its window, with the closed-form gap only in the explanation);
- a reinsured Aggregate whose cession is rebucketed off its lattice reads `reinsurance; net fails
  sev cv, agg cv` (today `reinsurance; subject not unreasonable`, mean right, law wrong), and
  `valid.passes` is false for it;
- a skewness that is zero to rounding is not failed, and an analytic moment the library could
  not compute is reported as such rather than passing silently;
- a build whose verdict fails emits one `ValidationWarning` with the verdict, where today
  nothing is said at any level.

## Current behavior

All anchors are `src/aggregate/_validation.py` unless stated. Measured 2026-10-31.

1. **Stable mean.** `chfagg X stable 1.5 1` reads `fails agg mean`: window mean -0.02652
   against 0. `_completed_mean` (`characteristic_function.py` 3066 to 3108) adds the closed-form
   tail and returns `(0.0265164, 4.94e-07)`, but its only consumer is `validation_explanation`
   (3179 to 3186). The verdict's test is `valid_characteristic_function` 531 to 535,
   `if float(err.get(('agg', 'mean'), 0.0)) > eps`, on the `error` column written from the raw
   window mean by `_write_empirical_moments` (`characteristic_function.py` 2582 to 2603). The
   same window error feeds the info row (2964 to 2969), `validation_df` (3028 to 3033) and
   `validation_score_terms` (630 to 642). `_sharpen_terms` (3444 to 3445) already drops the
   mean term for power tails. Five programs measured: window errors 2.65e-2, 5.3e-3, 2.87,
   2.3e-2, 1.77e-2; completed-mean errors 4.9e-7, 1.0e-7, 4.4e-6, 7.7e-7, 4.9e-7. Untilted
   (`hints{tilt=False}`) and `alpha < 1` return `None`. Tests pinning today's reading:
   `tests/test_characteristic_function.py` 1299 to 1314 and 1336 to 1340; the ruling they
   record is `dev/done/plan-characteristic-function-aggregate-parity.md` 52 to 53, and
   `dev/TODO.md` 160 to 164 (`[CHF-Tail-Completion]`) records that the mean row flags until then.
2. **Reinsured views.** `valid_aggregate` compares `gross_empirical` with `mixed` (242 to 248,
   298) and at 337 to 343 only marks `Validation.REINSURANCE`; the text is `reinsurance; subject
   ...` (167 to 169) and the long form at 799 to 804 gives a cession no independent analytic
   moments. `Validation.passes` (`constants.py` 237 to 254) returns true whenever REINSURANCE is
   set, so `3 claims 1500 xs 250 dsev [1000 2000 3000] occurrence net of 500 xs 500 fixed`
   reads `reinsurance; subject fails sev mean, agg mean` with `passes` true. `valid_portfolio`
   skips the total's checks when any unit is reinsured (409 to 418). What exists for the views:
   `_reinsurance.reins_view_stats` (878 to 1021, `Aggregate._reins_view_stats`) computes two
   bases per view, `EX`, the exact image of the on-grid subject through `occ_ceder`,
   `occ_netter`, `agg_ceder`, `agg_netter` (841 to 869, 967 to 999), and `Est`, the rebucketed
   density (979 to 988, 1000 to 1004); `stats_df['empirical']` is the realized view and
   `after_occ` the post-occurrence moments (`_aggregate.py` 4809 to 4884). The net and ceded
   densities are `reins_density_df` (`_reinsurance.py` 433 to 518). `EX` is exact whenever the
   subject pmf is exact on the grid, which is the case the subject verdict certifies; the gap
   `Est` against `EX` is the rebucketing error, which is what goes wrong.
3. **Near-zero skewness.** The skew test (306 to 312, CV twin 299 to 305) skips when
   `abs(theo) <= VALIDATION_NOISE` (1e-12 absolute, `config.py` 274). `100 * norm + 1000` has
   analytic skewness 9.775e-12 from quadrature residue (`moms()` gives 1030000000.139 for an
   exact 1030000000) and empirical -2.4e-13, and reads `fails sev skew, agg skew`.
   `MomentAggregator.static_moments_to_mcvsk` (`moments.py` 385 to 413) floors the variance
   against cancellation (392 to 394, `abs(var) <= VALIDATION_NOISE * scale`) but not the
   skewness numerator (411 to 412). The same absolute floor is in `valid_portfolio` 463 to 476,
   `valid_characteristic_function` 537 to 541 and `validation_score_terms` 636 to 639.
4. **`nan` analytic moment.** `100 * gamma 2 + 1000000` reads `not unreasonable` with analytic
   cv and skew `nan`: the numerical second moment is off by 9.2e-8 relative, the variance is
   -72,596.8, `moments.py` 395 to 399 logs `weird var < 0` six times and takes `sqrt`, and the
   `np.isfinite(theo)` gate (302, 309) then skips the test as it would an infinite moment. The
   only signal is `logger.error`, which the package does not route anywhere.
   `[Shifted-Severity-Moments]` cures this program; the gate stays wrong for any other
   `nan`.
5. **Silence.** `agg L 3 claims 1500 xs 250 dsev [1000 2000 3000] fixed` reads `fails sev mean,
   agg mean` and the build emits nothing at any logging level or warning filter. Validation is
   lazy (first access of `.valid`, `_aggregate.py` 4996 to 5046, reset by `update_work` at
   4620) and its findings are `logger.info` (253, 257, 282, 304, 311, 320, 332). The template
   for a construction-time warning is the deficit `warn_once` in `update_work` 4792 to 4799
   (`DefectiveDistributionWarning`, key `'defective-construction'`); `Portfolio.update` has its
   own at `_portfolio.py` 2944 to 2964, and updates non-signed units through `update_work`
   (2905), signed or windowed ones through `update` (2852); `AggregateCHF.update` ends at
   `characteristic_function.py` 2375 to 2377. `warn_once` is `constants.py` 521 to 566 with
   `warn_once_isolated` 581 to 606; sharpen's probes run isolated (`_bucket_window.py` 2483 to
   2486) and its final update does not. The name `ValidationWarning` is free in `src/`.

## Steps

### `[Stable-Completed-Mean]`

**Change.** `valid_characteristic_function` tests the mean on the completed mean when
`obj._completed_mean()` is not `None`: `err = _noise_aware_rel_error(completed, obj.actual_m)`.
`_write_empirical_moments` writes that same completed error into the `error` column of the
`('agg', 'mean')` row when available, so the info row, `validation_df` and the score agree with
the verdict; `validation_explanation` keeps its sentence. The docstring at 506 to 508 and the
TODO entry `[CHF-Tail-Completion]` are reworded (the `tvar` and `lev` completion stays open).

**Files.** `src/aggregate/_validation.py`, `src/aggregate/characteristic_function.py`,
`tests/test_characteristic_function.py` (1299 to 1314 and 1336 to 1340 change expectation),
`dev/TODO.md`.

**Acceptance.** The five programs above read `not unreasonable` with mean error below 1e-5;
`stable 1.5 1 hints{tilt=False}` still flags ALIASING and `stable 0.7 1` still reads
`grid infeasible ...; fails pmf deficit`; `test_the_completed_stable_mean_is_the_theoretical`
passes unchanged; `validation_score` of `2 * stable 1.5 1 + 10` is below 1.

### `[Verdict-Reinsured-Views]`

**Change.** For a reinsured Aggregate, `valid_aggregate` also compares, for the realized view
(`occ_kind` or the aggregate cover's kind), the `Est` basis against the `EX` basis of
`_reins_view_stats` for the severity (occurrence stage) and the aggregate, mean, CV and
skewness at the same `eps` and floors as the subject. A failure sets the existing flags
(`SEV_CV` and the rest) together with a new `Validation.VIEW` bit, and `explain_validation`
renders `reinsurance; subject not unreasonable; net fails sev cv, agg cv` (the view's word
from `occ_kind`, `ceded` or `net`). `Validation.passes` stops treating REINSURANCE as a pass:
it is a pass when nothing in the `fails` list is set, subject or view. `valid_portfolio` applies
the same to the total through `Portfolio.reins_density_df` when a unit is reinsured, instead of
skipping (409 to 418). The long form at 799 to 804 is reworded: the subject is checked against
its analytic moments and the cession against the exact image of the subject on its grid.

**Files.** `src/aggregate/_validation.py`, `src/aggregate/constants.py` (the flag, `passes`),
`src/aggregate/_reinsurance.py` (if `reins_view_stats` needs a cheap accessor for the two
bases), `tests/test_validation.py`, `tests/test_reins_reporting.py`,
`tests/test_library_entries.py` (entries whose `passes` changes are read and accepted).

**Acceptance.** `agg X 1 claims dsev [1000 2000 3000] occurrence net of 1500 xs 250 fixed` at
1.43.1's grid (`bs` 1000) reads `reinsurance; subject not unreasonable; net fails sev cv, agg
cv` and `passes` is false; after `[Lattice-Grid-Sizing]` the same program at `bs` 50 reads
`reinsurance; subject not unreasonable; net not unreasonable` and passes. The 68 silent wrong
answers of the lattice study at 1.43.1 (`dev/lattice-study/res_base.pkl`, class `WRONG-silent`)
all read a view failure under this verdict; the study's `classify` gains the view words. A
layer on bucket edges (`tests/test_validation.py` 243 to 251) stays clean.

### `[Verdict-Moment-Floors]`

**Change.** Two floors, one per mechanism.

- `static_moments_to_mcvsk` floors the skewness numerator as it floors the variance: with
  $s_3 = |e_3| + 3|e_1 e_2| + 2|e_1|^3$, a numerator with `abs(num) <= VALIDATION_NOISE * s_3`
  is 0. A symmetric law computed by quadrature then has skewness exactly 0 and the existing
  gate skips it. The same floor in `Severity._stats` (`_severity.py` 2601 to 2606).
- A theoretical CV or skewness that is `nan` while the empirical is finite is a defect of the
  analytic computation, not a pass: it sets a new `Validation.ANALYTIC_UNDEFINED` bit that
  renders outside the `fails` list, as INFEASIBLE does (`analytic cv, skew undefined; ...`),
  and `passes` treats it as INFEASIBLE is treated. `inf` stays a legitimate value (an infinite
  moment) and keeps today's skip. The `weird var < 0` path in `moments.py` 395 to 399 sets the
  variance to `nan` as it does and the verdict shows it.

**Files.** `src/aggregate/moments.py`, `src/aggregate/_severity.py`,
`src/aggregate/_validation.py`, `src/aggregate/constants.py`, `tests/test_moments.py` (190 to
197 keeps its `nan` assertion), `tests/test_validation.py`.

**Acceptance.** `100 * norm + 1000` reads `not unreasonable`; `norm 1000 cv 0.1` (the same law
through `[Severity-Parameter-Count]`) too. A program forced to a negative variance (the
`tests/test_moments.py` 190 fixture built as an Aggregate) reads `analytic cv, skew undefined;
not unreasonable` and `passes` is false. The symmetric-die and symmetric-portfolio tests
(`tests/test_validation.py` 21 to 36, 80 to 89) pass unchanged.

### `[Validation-Warning]`

**Change.** `class ValidationWarning(UserWarning)` in `constants.py`, in `__all__`, with the
docstring pattern of its siblings. At the end of `Aggregate.update_work` (after the `error`
column, 4883 to 4887), `Portfolio.update` (after `_write_empirical_stats`, 2985) and
`AggregateCHF.update` (2377), when `not self.valid.passes`:

```
warn_once(f'{self.name}: {self.validation_description}. See validation_explanation.',
          ValidationWarning, key=('validation', self.name, str(self.valid)), stacklevel=3)
```

Computing `valid` there is a read of `stats_df`, not a new computation. Internal probes that
must not warn (sharpen's per-cell builds, bivariate marginals at `bivariate.py` 1649, 1658,
1959, 2052, agg-as-severity inner builds at `underwriter.py` 2216 to 2228) run inside
`warn_once_isolated` already or are wrapped in it in this step; the list in the report of
2026-10-31 is the checklist. INFEASIBLE and the lattice warning have their own classes and are
not repeated here: the key excludes those bits.

**Files.** `src/aggregate/constants.py`, `src/aggregate/_aggregate.py`,
`src/aggregate/_portfolio.py`, `src/aggregate/characteristic_function.py`,
`src/aggregate/bivariate.py`, `src/aggregate/underwriter.py`, `tests/test_warning_policy.py`,
`tests/test_library_entries.py`, `docs/2_aggregate_overview/features.rst` (the warnings list).

**Acceptance.** `agg L 3 claims 1500 xs 250 dsev [1000 2000 3000] fixed` at a pinned `bs` 1000
emits exactly one `ValidationWarning` whose text contains `fails sev mean, agg mean`, and a
second build of the same program in the session emits none. Every entry in `library.agg` and
`reproductions.agg` builds with the warnings it earns listed in the execution log; the number
is expected to be small after the three earlier plans, and each one is a program to look at,
not a reason to lower the bar. The tests that assert no warning during a build (list in the
report: `test_approximate.py` 329 and 338, `test_agg_as_severity.py` 170 to 186,
`test_lattice_bucket.py` 166, `test_ntt.py` 869 and 876, `test_prob_loss_assets.py` 144 to
205, `test_stable.py` 84, `test_unit_density.py` 160 to 199, and the `recwarn` checks) pass
unchanged. Tier 3 with the numerics gate.

## Not in this plan

- `[CHF-Tail-Completion]` for `tvar` and `lev` beyond the window.
- A verdict for a continuous severity's cession against analytic layer moments: the `EX`
  basis on the grid is the reference here, and it is exact only for a lattice subject.
- Routing `logger.error` to the user; the warning is the channel.

## Execution log

Executed from 1.54.0 (`29bb5fd`), 2026-11-01. Line numbers in "Current behavior" are 1.43.1's and
several have moved; every symbol still exists under its name.

### Review at 1.54.0

Measured before any edit, against the plan's claims:

- **Item 3 is out of date.** `100 * norm + 1000` reads `not unreasonable` at 1.54.0, with analytic
  skewness exactly 0: `[Shifted-Severity-Moments]` cured it. The floor is still wanted, on a different
  premise: the 858-program sweep has 86 failing verdicts and the floor clears 43, the skew-only false
  alarms `[Correctness-Oracle]` listed (`1 claims dsev [115 120 125] fixed`).
- **Item 4 is cured as predicted.** `100 * gamma 2 + 1000000` has skewness 1.4143.
- **Items 1, 2 and 5 hold,** at the pinned grids the acceptance names.
- **`warn_once_isolated` does not silence a warning.** Inside it every call warns, which is what
  sharpen's counting probe needs. `[Validation-Warning]` needs another mechanism.
- **"Computing `valid` there is a read of `stats_df`" stops being true** once `[Verdict-Reinsured-Views]`
  lands if the views go through `reins_density_df`, which runs up to three FFTs per unit.
- **`AggregateNTT.update` rebases `mixed` after `update_work` returns,** so a verdict computed at the
  end of `update_work` would be the float one.
- **The app grades the verdict by its text.** `validationState` in the API's `web/src/main.js` strips
  `reinsurance; subject ` and compares to `not unreasonable`. See the ripple note below.
- `V:/worktrees/dev-files.md`, the control list the skill mentions, no longer exists, so no row is kept.

### `[Stable-Completed-Mean]`, 1.55.0

`stable 1.5 1`, `2 * stable 1.5 1 + 10`, `stable 1.1 1`, `2 * stable 1.5 0.5` and `stable 1.5 -1` read
`not unreasonable` with mean errors 4.9e-7, 1.0e-7, 4.4e-6, 7.7e-7 and -4.9e-7. `stable 1.5 1
hints{tilt=False}` still flags ALIASING, `stable 0.7 1` still reads `grid infeasible for this law; fails
pmf deficit 5.792e-03`, `test_the_completed_stable_mean_is_the_theoretical` passes unchanged, and
`validation_score` of `2 * stable 1.5 1 + 10` is 1.0e-3.

Divergences:

- **The verdict's code does not change.** The plan has `valid_characteristic_function` compute the
  completed error and `_write_empirical_moments` write it too. Writing it once in the `error` column
  suffices, since the verdict, the `info` row and the score already read that column. Only its
  docstring moves. `validation_df` computes its own `Err EX`, so it takes the same number explicitly.
- **`test_the_store_feeds_the_accessors` also changes,** which the plan does not list: it pinned `Err
  EX` at the window's -5.3e-3. It now checks the completed error and the window shortfall separately.
- **`_sharpen_terms` is left alone.** `u_agg_mean` stays dropped for a power tail, as the plan's
  current-behavior note describes, so no sharpen score moves.
