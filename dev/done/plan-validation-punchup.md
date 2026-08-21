# Plan: validation punchup, an aliasing test that measures aliasing and a feasibility reading for the grid

> **Status: EXECUTED at `1.0.0a311` to `1.0.0a313`, 2026-08-21.** See the execution notes at the end for what landed and the five places it diverged from the plan's letter, two of them on author rulings taken during execution. Original statement follows.
>
> Two changes to what validation says, both driven by measurement rather than by tuning a threshold. Part one replaces the aliasing test, which currently has no true positives anywhere in the shipped corpus. Part two adds a feasibility reading that tells a user when the grid they asked for cannot reproduce their severity's mean at any bucket size, and why.
>
> Drafted 2026-08-21 out of the diagnostic work behind `dev/done/plan-signed-bounded-window-overflow.md`. The author ruled on the four open design questions the same day; the rulings are recorded inline and in the "Ruled out" section, which is as load bearing as the rest of the plan.
>
> Supersedes the remaining open half of `[Validation-Calc-Review]` (#49) in `dev/TODO.md`, and closes `[Aliasing-Test-Misfires-On-A-Reference-Severity]` outright.

## The measurements this plan rests on

Every `agg` in `src/aggregate/agg/_test_suite.agg` and `src/aggregate/agg/library.agg` was built and its validation state recorded, 257 programs in all, at `1.0.0a308`. Anything below that says "the corpus" means that run. The probe scripts are throwaway and are not checked in; the numbers are reproducible from the recipe base in a few lines.

## Part one: the aliasing test does not measure aliasing

### What fires today

`Validation.ALIASING` fires when the aggregate mean relative error exceeds `ALIASING_RATIO` (10) times the severity mean relative error, silenced only when the aggregate error is itself below `VALIDATION_NOISE` (1e-12). `_validation.py:205` for `Aggregate`, `:352` for `Portfolio`, identical code.

It fires six times in the corpus. With `eps = 1e-4` for scale:

| program | sev mean error | agg mean error | ratio |
| :--- | ---: | ---: | ---: |
| `G.Mixed07` | 1.1e-13 | 9.2e-11 | 841 |
| `Y.Renewal.LayerMix` | 1.4e-12 | 6.9e-10 | 477 |
| `ReinsuranceOccurrenceTower` | 2.2e-16 | 1.0e-12 | 4634 |
| `ReinsuranceOccurrenceWithAggLimit` | 2.2e-16 | 1.0e-12 | 4634 |
| `ReinsuranceAggregateLayer` | 2.2e-16 | 1.0e-12 | 4634 |
| `SignedPremiumMinusLoss` | 1.2e-11 | 3.5e-07 | 28170 |

Every one is dust. The largest aggregate mean error among them is 3.5e-7, three hundred times **below** the tolerance at which we call anything a failure, and three of the six have machine epsilon in the denominator. The false positive rate is six out of six.

The cause is structural, not a badly chosen ratio. The test is a pure ratio with a floor on the numerator only, and that floor is `VALIDATION_NOISE`, an arithmetic dust floor eight orders of magnitude below `eps`. Any severity that discretizes essentially exactly, which includes a discrete law, an integer lattice, a reinsurance layer landing on bucket edges, and the `sev agg.NAME` reference that produced `[Aliasing-Test-Misfires-On-A-Reference-Severity]`, puts near zero in the denominator and the ratio explodes on nothing. Raising `aliasing_ratio` does not fix a ratio whose denominator is `2.2e-16`.

### What the test claims to measure

`validation_explanation` says `ALIASING` means "mass is coming off the top of the grid and landing back at the bottom". There is a direct measurement of exactly that, built from numbers `stats_df` already carries:

```
predicted = E[N] * E_empirical[X]        # the discretized severity's own aggregate mean
residual  = |E_empirical[A] - predicted| / |predicted|
```

`E[N]` is the realized frequency mean (`stats_df['mixed'][('freq', 'mean')]`), `E_empirical[X]` and `E_empirical[A]` are the `gross_empirical` severity and aggregate means, so the comparison is apples to apples under reinsurance for the same reason the existing checks are. Because the predictor uses the **discretized** severity mean, severity discretization error cancels exactly and there is no denominator to blow up. What is left is the error introduced by the convolution step alone, which is the thing the message is about.

The corpus bears this out. The residual is at or below 1e-6 for every program that is not already flagged `DEFECTIVE`, including all fourteen whose severity mean error exceeds `eps` (the worst, `GrossCatXOL` and `USXOLTower`, miss the severity mean by 41.6% and have a residual of 1.0e-6). In this library, on real programs, the FFT step is exact and every mean error is severity discretization.

### It also separates the two mechanisms, which the ratio cannot

Forcing a genuine failure on `100 claims sev gamma 100 cv 1 poisson` at `log2=14, bs=1`:

| padding | residual | pmf deficit | mechanism |
| ---: | ---: | ---: | :--- |
| 0 | 5.4e-5 | 0 | wrap, that is, true aliasing |
| 1 | 1.4e-6 | 3.3e-5 | truncation, mass dropped |

Residual material with the deficit immaterial is wrap. Residual material with the deficit material is truncation, which `DEFECTIVE` already owns and already explains. Note that the default `padding=1` runs the FFT on a doubled grid and discards the top half, so true wrap needs the aggregate to reach past **twice** the grid. Real aliasing is rare by construction, and a flag that almost never fires is the honest outcome rather than a sign the change went too far.

### Design

**`[Aliasing-Direct-Measure]`.** Replace the ratio in `valid_aggregate` (`_validation.py:203` to `:207`) with the residual above. Guard `predicted == 0` (a mean zero aggregate) by skipping the test, and take absolute values so a signed aggregate works. No new data is read: `stats_df` already carries all three inputs.

**`[Aliasing-Threshold-Setting]`.** Add `validation.aliasing_eps`, default `1e-5`, and retire `validation.aliasing_ratio`. `eps` itself is too loose: the forced `padding=0` case above sits at 5.4e-5 and is genuine, so a threshold at `eps` would miss it. `VALIDATION_NOISE` is far too tight, which is the present bug. One order of magnitude inside `eps` is the band where a convolution error is real but has not yet failed the mean outright. Retiring `aliasing_ratio` removes a `config.toml` key; the `CHANGELOG` entry says so, and the two `ALIASING_RATIO` module constants (`_validation.py:34`, `_portfolio.py:28`) and their importers in `distributions.py`, `portfolio.py` and `_aggregate.py` go with it.

**`[Aliasing-Under-Defective]`.** Author ruling 2026-08-21: when the deficit is material, fire `DEFECTIVE` only. A deficit above `deficit_materiality` is already an unambiguous statement that something is wrong, its explanation already ends "Raise log2, or widen the grid", and adding a second flag saying the same thing in different words is noise. So `ALIASING` sets only when the residual exceeds `aliasing_eps` **and** the deficit is below `deficit_materiality`, which makes the flag mean wrap and nothing else, exactly as its explanation claims. The existing suppression idiom in `explain_validation` (CV suppressed under mean) is the precedent.

**`[Portfolio-Aliasing-Parity]`.** `valid_portfolio` (`_validation.py:350` to `:354`) takes the same change with the same threshold. The predictor at portfolio level is the sum of the units' own empirical aggregate means, since each unit's FFT is separately validated above and the portfolio step is the convolution of the units. Author ruling 2026-08-21: yes, keep the two in step.

**Wording.** With `ALIASING` restricted to wrap, the existing long explanation is already correct and needs no change. The short form in `explain_validation` becomes something that names the measurement rather than the ratio, along the lines of "aggregate mean lost in the convolution, possible FFT wrap; raise log2".

### Tests

Extend `tests/test_validation.py`:

* The six corpus programs above build clean, asserted by name, with `ALIASING` clear.
* `100 claims sev gamma 100 cv 1 poisson` at `log2=14, bs=1, padding=0` sets `ALIASING` and not `DEFECTIVE`; the same at `padding=1` sets neither.
* The same at `log2=12, bs=1, padding=1` sets `DEFECTIVE` and not `ALIASING`, which is the `[Aliasing-Under-Defective]` ruling.
* A reference severity (`sev agg.NAME` on a matched grid, the `SplitLimitPolicy` shape) validates clean, which is the regression test for `[Aliasing-Test-Misfires-On-A-Reference-Severity]`.
* Portfolio parity: one portfolio case per branch.

## Part two: telling the user when the grid cannot work

### The problem, stated exactly

`library.agg` ships `GrossCatXOL`, a US hurricane ILW model, `1.74 claims sev lognorm 8.501 cv 14.624 poisson` in billions. It is a real model, not a constructed pathology. Its severity mean is wrong by 41.6% and validation says "fails sev mean, agg mean", which tells the user nothing they can act on.

Here is what is actually happening. The lognormal has sigma 2.317 and median 0.580, so the median claim is a fifteenth of the mean. The sizer reaches to 8.3e6 to cover the aggregate tail, which at `log2=16` forces `bs=200`, so the median claim is one three hundred and forty fifth of a single bucket. Under the `round` scheme bucket zero collects everything below `bs/2` and places it at exactly zero, and the fraction of the **mean** supplied below 100 is 46.2%. That single effect is the whole error.

The general statement is a ratio of two quantiles of the size biased law. Writing `P_1` for the size biased distribution of the severity, the discretized mean loses `P_1(X <= bs/2)` at the bottom and `P_1(X > n*bs)` at the top, and those are the only first order losses, so a mean accurate to relative `delta` needs `bs/2` below the `delta/2` quantile of `P_1` and `n*bs` above its `1 - delta/2` quantile. For a lognormal `P_1` is `LN(mu + sigma^2, sigma)`, the quantile ratio is `exp(2 z sigma)` with `z = Phi^-1(1 - delta/2)`, and the requirement collapses to a closed form in which the mean cancels and only sigma survives:

```
log2 >= 2 z sigma / log 2 - 1  ~=  11.23 * sigma - 1     (delta = 1e-4)
```

Built on exactly the grid the formula prescribes, the realized error lands on the target:

| severity | sigma | log2 required | realized mean error |
| :--- | ---: | ---: | ---: |
| `lognorm 200 cv 2` | 1.269 | 13.2 | 2.0e-6 |
| `lognorm 200 cv 5` | 1.805 | 19.3 | 2.6e-5 |
| `lognorm 200 cv 10` | 2.148 | 23.1 | 1.9e-5 |
| `lognorm 8.501 cv 14.624` | 2.317 | 25.0 | 1.8e-4 |

Against a working `log2=16` the practical boundary sits near sigma 1.51, that is a CV around 3. Below it an unlimited lognormal is routine, above it no bucket size works. `sharpen` agrees empirically: on `GrossCatXOL` it improves the score from 2440 to 446 against a target of 0.5, reports that it ran out of `bs_limit` while still improving, and cannot reach the target at `log2_cap=22`.

The full derivation, with the connection to Mandelbrot's moment localization argument that explains why this is a property of the lognormal rather than an accident of our grid, is written up at `C:/s/AI/aggregate-monograph/posts/2_technical_guides/2_x_against_lognormal.qmd`, section "Localization and the choice of bucket size".

### Design

**`[Severity-Feasibility-Reading]`.** Compute two numbers alongside the sizing rows in `_bucket_window.bs_window` and store them on the aggregate as `_bs_feasibility`, next to the existing `_bs_clip` and `_bs_snap`:

* `bs_max`, twice the `delta/2` quantile of the size biased severity, that is, the largest bucket that does not zero the mean out of bucket zero.
* `log2_required`, `log2(x_hi_grid / bs_max)` where `x_hi_grid` is the realized grid top, so it accounts for the reach the sizer actually bought rather than an idealized one.

Take `delta` from `validation_eps`. For a mixture, take the binding component, that is, the smallest `bs_max`.

**Name check.** `log2_need` is already a `_bs_window_df` column meaning "the log2 a row's window needs at the chosen bs", which is a different quantity. Do not reuse it. `log2_required` is the new name and the docstring says how the two differ. `bs_max`, `_bs_feasibility` and `InfeasibleGridWarning` are all new across `src/aggregate`.

The size biased cdf is `F_1(x) = (LEV(x) - x * S(x)) / E[X]`, so the quantiles come from a scalar root find in log space against `_moms_analytic` for the closed form kinds (`lognorm`, `pareto`, `gamma`, `expon`) and `_numerical_moms` otherwise. Two root finds per severity component at sizing time. If the numerical path proves too slow on some kind, fall back to reporting nothing for that kind rather than guessing, and say so in the docstring.

**`[Validation-Infeasible-Flag]`.** Add `Validation.INFEASIBLE`, set when `log2_required` exceeds the realized `log2` by more than a slack of 0.5. Author ruling 2026-08-21: this is a new validation type, not a failure. It fires whether or not the moments happen to fail, because it is a statement about the grid rather than about the outcome, and it leads the explanation the way `DEFECTIVE` does, because it explains the moment failures underneath it rather than adding to them.

Consequently `Validation.passes` must treat it as transparent. The current expression is `rv == NOT_UNREASONABLE or bool(rv & REINSURANCE)`; the minimal change that preserves the reinsurance behavior exactly is to mask `INFEASIBLE` out of the first term. An infeasible object whose moments fail still fails, and one whose moments happen to hold still passes, carrying the reading. Note the current `REINSURANCE` arm makes a reinsured object with failing moments pass, which is separately odd and is **not** in scope here; `INFEASIBLE` deliberately does not join that arm, because doing so would make the flag hide the very problem it names.

`exhibits/_core.py:110` sorts the failure flags into a `_SEV_FAILURES` and an `_AGG_FAILURES` mask for row emphasis. `INFEASIBLE` is a property of the severity against the grid, so it joins `_SEV_FAILURES`.

**`[Infeasible-Grid-Warning]`.** Add `InfeasibleGridWarning` to `constants.py` and raise it once through `warn_once`, keyed on the severity parameters plus `log2`, in the pattern `ReflectedSeverityClampWarning` established. The author's framing is that bucket selection is one of the largest hurdles FFT methods face and that the more educational we can be here the better, so the message earns its length. It should name the three numbers that make the situation legible and the one action that resolves it:

```
Severity 'lognorm 8.501 cv 14.624' cannot be reproduced on this grid.
The body needs bs <= 0.030 (46.2% of the mean is supplied below the
first half bucket, and is placed at 0); the grid reaches 1.3e7, so
matching the mean to 1e-4 needs log2 = 25, and log2 = 16 was used.
An unlimited lognormal with sigma above about 1.5 has this problem at
any bucket size, because its mean is furnished far above its median.
Add an occurrence limit, or accept the reported moment errors.
```

The long `validation_explanation` for `INFEASIBLE` carries the reasoning: what the size biased distribution is, why the mean and the tail want different resolutions, and why an occurrence limit is the fix rather than a finer grid. The short `explain_validation` string is one clause, "grid infeasible for this severity".

**`bs_description` and `bs_explanation`.** Both gain the reading, the short one as a clause and the long one as the full account. These are the existing narrative surface for grid choice and are where a user already looks, so no new public member is needed.

### Ruled out, deliberately

**`[Window-Resolution-Floor]`, rejected.** The sizer currently buys reach unconditionally and pays for it with `bs`. Capping `bs` at the resolution requirement and letting the top clip was on the table. Author ruling 2026-08-21: buying reach is the decision, keep it. It also would not have rescued the case that motivated it, moving `GrossCatXOL` from 41.6% to about 8.6%. Better warning does the work instead, which is what part two is.

**`[Mean-Preserving-Discretization]`, rejected for a gross severity.** One moment local moment matching, spreading each bucket's mass across its two bounding lattice points, makes the discretized mean exact at any `bs` and would take `GrossCatXOL` from 41.6% to zero. The machinery half exists: `_rebucket_to_grid(scheme='linear')` already does mean preserving linear scatter for discrete atoms, and `sev_calc='moment'` is an accepted option that raises `NotImplementedError`. Author ruling 2026-08-21: specifically do not do this. It hides a real problem in an insidious manner. It would repair the reported number while leaving the model's mean furnished by a region no one has an opinion about, which is worse than an obvious 41.6% error because it is silent. The fundamental fix is an occurrence limit on the severity. The existing use for reinsurance bucketing stands, where the grid is forced and the means have to work, but a gross severity does not get the same treatment. Leave the `NotImplementedError` in place and let its message stay as it is.

**Fixing `GrossCatXOL`, rejected.** Author ruling 2026-08-21: shipping it is fine, it is an education point. Once `[Severity-Feasibility-Reading]` exists the library will say out loud what is wrong with it, which is the teaching.

### Tests

New `tests/test_feasibility.py`, or an extension of `tests/test_bucket_sizing.py`:

* The four severities in the table above report a `log2_required` within 0.5 of the tabulated value.
* `GrossCatXOL` sets `INFEASIBLE`, warns `InfeasibleGridWarning` once, and its `bs_description` names `log2_required`.
* `lognorm 200 cv 2` at the default grid does not set `INFEASIBLE` and does not warn.
* A bounded severity of the same sigma (`1e12 xs 0 sev exp(19.595) * lognorm 2.581`, that is `N.US.Hurricane`, sigma 2.581 and a mean error of 0.089%) does not set `INFEASIBLE`. This is the test that proves the reading measures feasibility rather than thickness.
* `passes` is `True` for an object carrying `INFEASIBLE` alone and `False` when a moment flag rides with it.
* The whole corpus: no program gains `INFEASIBLE` other than the ones whose severity mean already fails, asserted as a count so a future regression shows up as a number rather than a scroll.

## Order of work

1. Part one entire, `[Aliasing-Direct-Measure]` through `[Portfolio-Aliasing-Parity]`. Self contained, changes no numbers, only verdicts. One version bump.
2. `[Severity-Feasibility-Reading]`, the computation and its two numbers, reported through `bs_description` and `bs_explanation` only. No flag yet, so the blast radius is narrative. One version bump.
3. `[Validation-Infeasible-Flag]` and `[Infeasible-Grid-Warning]`, the flag, the `passes` mask, the explanation and the warning. One version bump.

Splitting at those three points keeps each bump bisectable and puts the flag, which is the only piece that changes what `valid` returns, in its own commit.

## An observation found on the way, not in scope

Higher **raw** moments degrade as the grid gets taller at fixed `bs`. On `lognorm 200 cv 2` at `bs=6.427`, the relative error on the third raw moment runs 1.5%, 12.8%, 103%, 828% as `log2` goes 18, 19, 20, 21, while the mean error stays flat at 6.4e-6. The far tail carries mass at the float resolution floor (a constant 2.22e-16 per bucket, which is one ulp of 1.0 and points at a survival function computed as `1 - cdf` somewhere in the chain), and `x^3` weighting turns that dust into a large number. It is a real finding and probably a real bug, but it is a separate one: it does not touch the mean, it does not touch the aliasing test, and diagnosing it properly means tracing the `sf` evaluation rather than the validation code. Recorded here so it is not lost; it wants its own `dev/TODO.md` entry.

## Housekeeping

Three plan based bumps, so three `pyproject.toml` bumps, three `CHANGELOG.md` sections and three one line commits. In `dev/TODO.md`: close `[Aliasing-Test-Misfires-On-A-Reference-Severity]` at step 1, and narrow `[Validation-Calc-Review]` (#49) to its remaining half, auditing the algorithm against the published paper and making the docs match, since the aliasing false positive it names is what step 1 fixes. Add the far tail raw moment observation as a new entry. Move this plan to `dev/done/` when step 3 lands.

## Execution notes, `1.0.0a311` to `1.0.0a313`, 2026-08-21

Executed in the plan's three steps, one version bump each, as specified.

* **a311** `[Aliasing-Direct-Measure]`, `[Aliasing-Threshold-Setting]`, `[Aliasing-Under-Defective]`, `[Portfolio-Aliasing-Parity]`.
* **a312** `[Severity-Feasibility-Reading]`.
* **a313** `[Validation-Infeasible-Flag]`, `[Infeasible-Grid-Warning]`.

### Five places the execution diverged from the plan's letter

1. **The wrap gate is `VALIDATION_NOISE`, not `deficit_materiality`** (author ruling 2026-08-21, taken on re-measurement). The plan's claim that "the residual is at or below 1e-6 for every program that is not already flagged DEFECTIVE" does not hold: re-measuring all 257 programs at a310 found eight non-defective programs above 1e-6, topping out at 1.9e-4, and five of them clear the proposed `aliasing_eps`. Every one is truncation, mass dropped, with a deficit between 1.5e-5 and 9.6e-5, all under materiality, and three of the five pass validation today. Gating at materiality would have called them wrap. The discriminator that actually works is the one the plan's own forced-failure table shows: wrap **conserves** mass, so a genuine wrap carries a deficit of exactly zero, while truncation always carries a measurable one. At `VALIDATION_NOISE` the corpus goes to zero firings and the forced `padding=0` case still fires.

2. **`log2_required` is measured against the severity's own `reach`, not the realized `grid_top`.** The plan's design bullet says `log2(x_hi_grid / bs_max)`; its reference table, its message text ("needs log2 = 25") and its tests ("within 0.5 of the tabulated value") all say `log2(reach / bs_max)`, and the grid based form reads 15.6 for `lognorm 200 cv 2` against a tabulated 13.2, so it would fail the plan's own test. The `N.US.Hurricane` control settles it on meaning rather than on arithmetic: that severity is **thicker** than the cat model (sigma 2.581 against 2.317) and perfectly feasible because it is limited, while its chosen `bs` is still ten times coarser than its body wants. A feasibility flag has to say "no grid works", not "this grid is coarse", and only the reach based form does. `grid_top` is carried in the reading so the narrative can still quote what the grid reaches.

3. **The quadrature fallback is declined.** The plan allowed a `_numerical_moms` path for kinds outside the four `_partial_e` handles in closed form, with permission to "fall back to reporting nothing for that kind rather than guessing" if it proved slow. It is declined outright: quadrature inside a bisection at sizing time is both slow and fragile on exactly the heavy tails the reading is for. Coverage is `lognorm`, `gamma`, `pareto` and `expon`, which is 132 of the 255 building corpus programs, and everything else reports nothing. `grid_is_infeasible` is `False` on a missing reading, so no flag is ever raised on ignorance.

4. **`INFEASIBLE` sits outside the `fails ...` list rather than inside it.** Placed as the plan describes, the short form reads "fails grid infeasible for this severity", which contradicts the ruling that it is a reading and not a failure. It leads instead, on the existing `reinsurance` idiom: `grid infeasible for this severity; fails sev mean, agg mean`, or `grid infeasible for this severity; not unreasonable` when the moments hold.

5. **`docs/2_aggregate_overview/bucket-selection.rst` is rewritten** for both halves, at the author's request mid execution. Its "Two failure modes" section becomes three, the third being a severity no grid in the budget can resolve; a new "Feasibility: when no bucket size works" section derives the size biased requirement, tabulates the four reference severities against their realized errors, works the `N.US.Hurricane` control, records the two deliberate rejections, and carries the flag and the warning. The same section corrects the aliasing paragraph for a311. `pipeline-aggregate.rst` carries the matching `valid` list entries.

### Measured outcome

| | before | after |
| :--- | ---: | ---: |
| `ALIASING` firings, 257 program corpus | 6 aggregates + 2 portfolios | 0 |
| `ALIASING` false positives | all of them | none |
| genuine wrap still flagged | | `SignedPortfolioPair`, 94% of the mean lost at a 1.2e-13 deficit |
| `INFEASIBLE` firings, 255 building programs | n/a | 4, all already failing their severity mean |
| programs carrying a feasibility reading | n/a | 132 |

`VALIDATION_BASELINE` moved in both directions, which is what a baseline asserted in each direction is for: `SignedPremiumMinusLoss`, `SignedPortfolioMixed` and `ThinThinPortfolio` left it (they were never aliasing), and `CurvePareto`, `GrossCatXOL`, `HeavyTailValidation` and `ThickThickPortfolio` gained `INFEASIBLE`.

### Verification

Full gate, `pytest -m 'slow or not slow'`, at each of the three bumps. Thirty one new cases in a new `tests/test_feasibility.py` and twelve in `tests/test_validation.py`, plus the config and library baseline edits. The five failures in `tests/test_agg_libraries.py` and `tests/test_library_entries.py` throughout are pre-existing and unrelated: the author's uncommitted `library.agg` edits (`ISOMixedExponential` renamed to `CommAutoMixedExponential`, the rewritten `LayerPicks`, the new `LayerPicksMED`) have not reached the baselines or the canonical layout yet.

### Still open, recorded here so it is not lost

The far tail raw moment observation in the section above is now `[Far-Tail-Raw-Moment-Inflation]` in `dev/TODO.md`, as the plan asked. It is untouched by any of this.
