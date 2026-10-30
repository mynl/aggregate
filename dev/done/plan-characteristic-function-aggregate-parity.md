# Plan: bring AggregateCHF's surface into line with Aggregate's

`[Characteristic-Function-Aggregate-Parity]`. Drafted 2026-10-29 against
aggregate 1.34.0; revised the same day with the author's rulings (below).
**Status: draft for the author's review**, with no open questions. It
follows
`dev/done/plan-characteristic-function-engine.md` (`[Characteristic-Function-Engine]`,
1.27.0 to 1.34.0), which built `AggregateCHF` and the `chfagg` statement.

**Size.** Roughly 1,900 new or changed lines including docstrings and tests,
over six library steps, each a minor bump and each useful alone. No
`Aggregate` code path changes speed or output. The shared edits are one new
`Validation` member, two small extensions of `explain_validation`, one helper
factored out of `_validation.validation_explanation` (its output for
`Aggregate` and `Portfolio` pinned by a test), and `AggregateCHF` added to
existing exhibit helpers. Step `[CHF-Arithmetic-Form]` is a **breaking change
to the `chfagg` DecL**, for declarations written since 1.30.0 (2026-10-29).

## Goal

A `chfagg` should behave as though `AggregateCHF` were a subclass of
`Aggregate` with no frequency, no severity and no reinsurance:

1. the same DecL arithmetic for scale and location as an `agg` severity:
   `chfagg G 25 * gamma 4`, not `chfagg G gamma 4 25`;
2. the same frames in the same shapes: `summary_df`, `validation_df` and
   `stats_df` with Aggregate's rows and columns, the frequency and severity
   rows empty (`NaN`);
3. the same verdict vocabulary and the same opening to the explanation, with
   the characteristic function checks added after it;
4. the same names for the same quantities (`actual_m`, not `agg_m`), and every
   `Aggregate` member that means something for a law on a grid, `sharpen`
   included.

## Rulings

Author, 2026-10-29, on the first draft:

1. **Shapes only, everywhere.** Passing a parameter a family does not take is
   an error, in a `chfagg` and in an `agg` severity alike: a shape for a
   zero-shape family, two shapes for a one-shape family, and so on. So
   `norm 100 30` is refused in favor of `30 * norm + 100`. The `agg` side is
   its own plan, `dev/plan-severity-parameter-count.md`
   (`[Severity-Parameter-Count]`).
2. **Follow scipy.** Where scipy has the law, its shape parameters are the
   DecL shapes: `norminvgauss a b`, `invgauss mu`.
3. **`NaN`** for the frame cells that do not apply.
4. **No separate check table.** The rows of today's `validation_df` fold into
   the `info` string, as an `elt` aggregate's loss table checks do, so they
   reach the Loss Lab with no work there; the prose goes in
   `validation_explanation`.
5. **Aggregate's rule for the stable mean**: `fails agg mean` until
   `[CHF-Tail-Completion]`, with the closed-form gap in the explanation.
6. **`sharpen` makes sense**, or an analog of it: step `[CHF-Sharpen]`.

## Why not a subclass

The engine plan ruled it out (author, 2026-10-29, first round, ruling 2: "An
`Aggregate` with its moment machinery switched off is therefore out"). On
review the ruling holds, and the reasons are concrete:

- **`Aggregate.__init__` builds a compound.** It makes the frequency, the
  severities, the moment store from their moments, and the reinsurance
  attributes. A `chfagg` has none of them, so a subclass either passes a
  fake frequency and severity (the moment store would then describe a law
  that is not the one computed) or skips `super().__init__` and stubs the
  71 instance attributes an `Aggregate` sets, most of them about
  frequency, severity or reinsurance.
- **73 of Aggregate's 105 public class members** do not exist on
  `AggregateCHF` today, and 42 of those have no meaning for it (the other 31
  are added here or deferred, below):
  `sharpen`, `pnl_program`, `reins_program`, `make_pnl`, `occ_bivariate`,
  `palm_kappa`, the `reins_*` family, `sev`, `q_sev`, `tvar_sev`,
  `sev_density_df`, `picks`, `unwrap`, `freq_pmf`, `create_frequency`,
  `as_tweedie`, `pollaczeck_khinchine`, `wiener_hopf`, `eventual_ruin`, the
  `elt_*` family and more. A subclass inherits every one.
- **Overriding them does not hide them.** The Loss Lab (`aggregate-api`)
  lights its leaves with `hasattr`: `pnl_program` lights P&L, `reins_program`
  lights reinsurance, `sharpen` lights sharpening. An override that raises
  still makes `hasattr` return `True`, so a subclass shows the P&L,
  reinsurance and sharpening leaves for a `chfagg` and fails when one is
  opened.
- **Dispatch follows the class.** Fourteen `singledispatch` registrations on
  `Aggregate` (the aggregate, structure, reinsurance, ruin, approximation and
  bivariate charts, and the reinsurance, statistics, summary, tail,
  validation and sharpen exhibit treatments) and fourteen
  `isinstance(x, Aggregate)` tests in `src` (Portfolio unit acceptance,
  `_dhistogram_from_object`, the underwriter, the structure chart, `bounds`,
  `tweedie`, `utilities`) would send a `chfagg` down the compound paths.

`AggregateNTT` is different in kind: it is a real compound, `dfreq` and `dsev`,
which `Aggregate.__init__` builds unchanged; only the convolution in `update`
differs, so every inherited member is right. That is why subclassing fit there
and does not fit here.

**What went wrong instead.** The engine plan required that "names follow
`Aggregate` wherever the meaning is the same", and then specified the frames
from the `chfagg`'s own logic: a one-row summary, a check table under the name
`validation_df`, a four by three `stats_df`, and the theoretical moments under
`agg_m`, a spelling `Aggregate` retired at 1.0.0a149. Parity was specified for
the accessors and not for the frames or the DecL. The remedy is a contract,
not a base class: the same names, shapes and vocabulary, enforced by parity
tests that build an `Aggregate` and a `chfagg` side by side and compare.

## Current behavior, 1.34.0 (measured 2026-10-29)

- **DecL** (`src/aggregate/decl.lark:75`, `parser.py:616` on,
  `characteristic_function.py:70`): the positional numbers include scale and
  location, `gamma shape scale`, `stable alpha beta [scale [loc]]`,
  `norm mean sd`, `invgauss mean shape`, `norminvgauss alpha beta delta mu`,
  `variancegamma sigma nu theta mu`, `temperedstable C G M Y [mu]`; `tweedie`
  and the `mean cv` forms are as an `agg` has them. The language reference
  (`docs/4_dec_Language_Reference.rst:179`) and `decl-testers.agg:1543`
  write `chfagg G gamma 4 25` beside `agg G dfreq [1] sev 25 * gamma 4`, two
  spellings of one law.
- **`summary_df`** (`characteristic_function.py:2227`): one `Agg` row. CV is
  not blanked at mean zero: `stable 1.5 0.5 2 0` shows CV of minus 677.6,
  where `Aggregate.summary_df` uses `_cv_or_nan`.
- **`stats_df`** (`:2263`): index `mean`, `sd`, `cv`, `skew`; columns
  `theoretical`, `grid`, `rel error`. Aggregate's is 26 rows by
  `(component, measure)` and, for one component, nine columns.
- **`validation_df`** (`:2352`): a check table, columns `value`, `target`,
  `status`, `note`, 8 to 12 rows. Aggregate's is three rows, `Freq`, `Sev`,
  `Agg`, by eight columns `EX | Est EX | Err EX | CV | Est CV | Err CV | Sk |
  Est Sk` (the SD trio in place of CV for a signed law).
- **`validation_description`** (`:2370`): `not unreasonable` or `flags <row
  names>`. Aggregate's comes from `explain_validation`: `not unreasonable`,
  `fails agg cv`, `pmf deficit 4.3e-06`, `n/a, not updated`.
- **`validation_explanation`** (`:2378`): one sentence per check row, with none
  of Aggregate's opening.
- **Tolerances**: every moment at `eps`; `valid_aggregate` uses `eps` for the
  mean, `10 eps` for the CV and `100 eps` for the skewness.
- **Theoretical moments** are `agg_m`, `agg_cv`, `agg_skew` (`:2145`). The
  Loss Lab looks for `actual_m` (`aggregate-api/src/aggregate_api/routes/objects.py:1014`,
  whose note records the 1.0.0a149 rename with no alias), so a `chfagg`'s
  theoretical mean does not reach its summary bar. `tests/test_fcc_surface.py:366`
  asserts `Aggregate` and `Portfolio` have no `agg_m`.
- **`density_df`** has `loss, p_total, p, F, S, lev, exa`; Aggregate's adds
  `log_p, exlea, e, exgta, exeqa`.
- **Verdicts on the shipped declarations:**

| declaration | verdict at 1.34.0 | cause |
|---|---|---|
| `gamma 100 cv 0.5` | not unreasonable | |
| `stable 1.5 1 2 10` | flags mean rel error | the window leaves out the tail mean above it |
| `stable 1.5 0.5 2 0` | flags mean rel error | the same, two tails |
| `stable 1.1 1` | flags outside mass, mean rel error | the same, and coverage at log2 16 |
| `tweedie 1.5 100 2` | flags most negative bucket, truncation error | see below |

  The stable mean gap is exactly the closed-form tail mean beyond the window,
  $A\alpha x^{1-\alpha}/(\alpha-1)$ measured from $\delta_1$ at the top bucket
  edge, less the corresponding term for the mass below the window, which sits
  in the first bucket:

| declaration | grid mean shortfall | closed form |
|---|---|---|
| `stable 1.5 1 2 10` | 0.05308 | 0.05303 |
| `stable 1.1 1` | 2.86458 | 2.86494 |
| `stable 1.5 0.5 2 0` | 0.02319 | 0.02319 |

  `Aggregate` behaves the same way on a power-law severity:
  `agg P 10 claims sev 10 * pareto 1.5 - 10 poisson` at `bs=1/4, log2=18`
  reads `grid infeasible for this severity; fails sev mean, agg mean`. Under
  parity the stable laws therefore keep failing the mean check; what changes
  is that the explanation quantifies the gap. Completing the tail is
  `[CHF-Tail-Completion]` in `dev/TODO.md`, not this plan.

  The Tweedie truncation flag is a defect: `truncation_error` is
  $|\varphi(\pi/h)|$ of the whole law, which tends to the atom
  $e^{-\lambda} = 4.54\times10^{-5}$, while the inversion subtracts the atom
  first. The continuous part's value at the same frequency is
  $1.13\times10^{-7}$, under the $10^{-6}$ flag. The negative buckets are
  real ringing at the jump of the continuous density at $0^+$: 1,382
  negative buckets, the worst $-1.8\times10^{-8}$ against a largest bucket of
  $7.3\times10^{-5}$, total $-9.3\times10^{-8}$.

## Findings outside this plan

1. **`agg` severities drop numbers silently** (2026-10-29, built with
   `update=False`, the frozen law's mean and standard deviation; now its own
   plan, `dev/plan-severity-parameter-count.md`):

   ```
   sev norm 100 30        mean   0    sd 1
   sev expon 5            mean   1    sd 1
   sev uniform 0 10       mean   0.5  sd 0.289
   sev gamma 4 25         mean   4    sd 2       the 25 is dropped
   sev norm 100 cv 0.3    mean 100    sd 1       with a warning: achieved cv 0.01
   sev expon 10 cv 1      mean  11    sd 1
   ```

   `sev0_two_params` and `sev0_one_param` (`parser.py:2575`) put the numbers in
   `sev_a` and `sev_b`; the zero-shape branch of `SeverityScipy._build`
   (`_severity.py:2873`) uses only `sev_loc` and `sev_scale`, and the
   one-shape branch never uses `sev_b`. `sev0_mean_cv` sets `sev_scale` to
   1.0, so the zero-shape branch's `sev_scale == 0` test, which would derive
   the scale from the CV, never passes. The corpora use only `sev expon 1`
   and `expon 1` (four scalar clauses and five mixed vectors), where the
   dropped number is the default. `[Severity-Parameter-Count]` refuses a
   number a family cannot take, with the arithmetic spelling in the message,
   and repairs the zero-shape `mean cv` path. The two plans state one rule, so
   the teaching comparison of one law computed two ways uses one spelling.
2. **The `stable` conversion statement** keeps `stable alpha beta [scale
   [loc]]`, the `Stable` class's parameter list. After this plan
   `stable 1.5 1 2 10` (conversion) and `chfagg S 2 * stable 1.5 1 + 10` name
   the same law in two spellings, which differ in meaning only at
   $\alpha = 1$ in S1. Recommend leaving the conversion statement as it is and
   adding the `chfagg` spelling to its printed equivalents later, as a
   `dev/TODO.md` entry.

## Design

### Inventory of Aggregate's public members

The table groups Aggregate's 105 public class members and its moment
attributes by what this plan does with each on `AggregateCHF`.

| group | members | plan |
|---|---|---|
| already shared, same meaning | `q`, `tvar`, `cdf`, `sf`, `pdf`, `pmf`, `snap`, `var_dict`, `prob_loss_assets`, `price_pentagon`, `calibrate_distortions`, `update`, `with_hints`, `bounded`, `density_df`, `summary_df`, `tail_df`, `tail_periods_df`, `validation_df`, `validation_description`, `validation_explanation`, `info`, `plot`, the label, help and program mixins, `bs`, `log2`, `x_min`, `xs`, `agg_density`, `est_m`, `est_sd`, `est_cv`, `est_skew` | keep; frames and verdict reshaped (`[CHF-Aggregate-Frames]`, `[CHF-Validation-Verdict]`) |
| same quantity, different name | `agg_m`, `agg_cv`, `agg_skew` | rename to `actual_m`, `actual_cv`, `actual_skew`; add `actual_sd`, `actual_var`, `est_var`; no alias, as at 1.0.0a149 |
| missing, meaningful, cheap | `valid`, `validation_score`, `validation_eps`, `value_type`, `prob_eq_0`, `spec`, `json`, `density`, `sample`, `price`, `price_ccoc`, `price_pentagon_ex`, `evaluate`, `apply_distortion`, `__str__`, the five extra `density_df` columns | add (`[CHF-Validation-Verdict]`, `[CHF-Accessor-Parity]`) |
| missing, meaningful, a report | `bs_window_df`, `bs_description`, `bs_explanation`, `tail_class`, `tail_behavior_df`, `tail_description`, `tail_explanation` | add (`[CHF-Grid-And-Tail-Reports]`) |
| missing, meaningful, an analog | `sharpen`, `sharpen_df`, `sharpen_description`, `sharpen_explanation`, `sharpen_program` | add (`[CHF-Sharpen]`), closing that `dev/TODO.md` entry |
| meaningful, deferred | `approximate`, `approximation_df`, `approximation_density_df`, `entropy_fit`, `rescale`, `as_severity` | not here; a follow-up each |
| no meaning without a frequency, severity, cession or loss table | `sev`, `q_sev`, `tvar_sev`, `sev_density_df`, `discretize`, `update_work`, `picks`, `unwrap`, `freq_pmf`, `create_frequency`, `one_claim`, `base_mean`, `as_tweedie`, `dsev_bucket`, `reins_bucket`, `apply_occ_reins`, `apply_agg_reins`, `reins_*` (12 members), `pnl_program`, `make_pnl`, `occ_bivariate`, `occ_joint`, `palm_kappa`, `pollaczeck_khinchine`, `wiener_hopf`, `eventual_ruin`, `elt_df`, `elt_description`, `elt_explanation`, `severity_error_analysis`, `aggregate_error_analysis`, `center_window`, `spec_ex` | omit; the frames hold `NaN` where they would appear |

### [CHF-Arithmetic-Form] scale and shift as arithmetic; families keep their shapes

**DecL**, the rule an `agg` severity follows:

```
chfagg NAME [as "LABEL"] [c *] FAMILY shapes [s0|s1] [+ b | - b] TRAILER
chfagg NAME [as "LABEL"] [c *] FAMILY mean cv v [+ b | - b] TRAILER
```

The positional numbers are the family's shape parameters only, in scipy's
order where scipy has the law. The law declared is $cX + b$ with $X$ the
family's standard law and $c > 0$.

| family | shapes | standard law | example | refused spelling and its pointer |
|---|---|---|---|---|
| `gamma` | shape $k$ | Gamma, scale 1 | `25 * gamma 4 + 10` | `gamma 4 25`: write `25 * gamma 4` |
| `norm` | none | N(0, 1) | `30 * norm + 100` | `norm 100 30`: write `30 * norm + 100` |
| `invgauss` | scipy's `mu` | mean `mu`, shape 1 | `400 * invgauss 0.25` | `invgauss 100 400`: write `400 * invgauss 0.25` |
| `stable` | $\alpha$ $\beta$, then `s0` or `s1` | $S(\alpha, \beta, 1, 0)$, S1 by default | `2 * stable 1.5 1 + 10` | `stable 1.5 1 2 10`: write `2 * stable 1.5 1 + 10` |
| `norminvgauss` | scipy's `a` `b` | scipy's law at scale 1 | `3 * norminvgauss 6 1.5 + 1` | `norminvgauss 2 0.5 3 1`: write `3 * norminvgauss 6 1.5 + 1` |
| `variancegamma` | $\sigma$ $\nu$ $\theta$ | Madan, Carr and Chang, location 0 | `variancegamma 0.2 0.5 -0.1 + 0.05` | four numbers: the `+` form |
| `temperedstable` | $C$ $G$ $M$ $Y$ | CGMY, location 0 | `temperedstable 1 5 10 0.5 + 2` | five numbers: the `+` form |
| `tweedie` | `p mean dispersion`, `p mean cv c` | unchanged | `2 * tweedie 1.5 100 2` | none |

The `mean cv` forms are unchanged for `norm`, `gamma`, `invgauss` and
`tweedie`, and the arithmetic applies on top of them, so
`2 * gamma 100 cv 0.5` has mean 200, as `sev 2 * gamma 100 cv 0.5` does.
`variancegamma` and `temperedstable` are not scipy laws and their published
parameters mix scale with shape (the variance gamma's $\sigma$ and $\theta$
scale together, CGMY's $G$ and $M$ are inverse scales), so they keep their
published parameters less the location. That is the rule's one stated
exception, and the language reference states it.

A refused spelling raises a `ValueError` naming the equivalent new spelling,
computed from the numbers given. Nothing is reinterpreted silently, unlike the
`agg` severities in finding 1.

**Stable laws and $\alpha = 1$.** The arithmetic is the law of $\gamma X + \delta$,
as `sev` and scipy's `levy_stable(loc, scale)` define it. Under S0 that is
$S(\alpha, \beta, \gamma, \delta; 0)$ for every $\alpha$. Under S1 it is
$S(\alpha, \beta, \gamma, \delta; 1)$ for $\alpha \ne 1$, and at $\alpha = 1$
it is $S(1, \beta, \gamma, \delta + (2/\pi)\beta\gamma\log\gamma; 1)$. The
pointer for a refused `stable 1 beta g d` therefore shifts $\delta$ by
$-(2/\pi)\beta g\log g$, so the suggested spelling is the law the old one
declared. A side effect, and a fix: the scipy reference of a DecL stable law
is now right at $\alpha = 1$ (the engine plan's finding on `Stable.fz`),
because it is built through scipy's own `loc` and `scale`.

**Implementation: one wrapper.** `FamilyAffine(base, scale, shift)` implements
the family protocol from the base's closed forms, so the engine (window, tilt,
split, inversion) does not change:

- $\varphi_Y(s) = e^{ibs}\varphi_X(cs)$; $K_Y(s) = K_X(cs) + bs$ on
  $(s_-/c, s_+/c)$;
- `power_tails` $(Ac^\alpha, Bc^\alpha, \alpha)$; `location` $c\delta_1 + b$;
  `support` and `atom` mapped; `scale` $c$ times the base's; `moments`
  $(cm + b, c^2v, \text{skew})$; `tilt_side` unchanged;
- `reference`: the base's scipy frozen law rebuilt with `loc` $c\cdot$loc$ + b$
  and `scale` $c\cdot$scale; `None` stays `None`;
- the stable members, delegated through the map: `halves()` wraps each half
  in the same scale, with the shift on the plus half; `reflected()` is
  `FamilyAffine(base.reflected(), c, -b)`; `tail_mass_above(x)` is
  `base.tail_mass_above((x - b) / c)`; `density_bound` is the base's over $c$;
- `to_decl()` renders `c * <base> + b` as written, omitting a unit scale and
  a zero shift.

The map $c = 1$, $b = 0$ is not wrapped. The Python API is unchanged:
`AggregateCHF('G', 'gamma', shape=4, scale=25)` and
`FamilyStable(dict(alpha=..., scale=..., loc=...))` still accept full
parameters, as scipy does. Two new keywords of `AggregateCHF.__init__`,
`chf_scale` and `chf_loc`, hold the arithmetic; they are also the spec keys,
mirroring the `sev_scale` and `sev_loc` an `agg` spec holds for the same
arithmetic. Python callers that want the DecL meaning pass them.

**Out of scope here:** a negative multiplier and `b - X` (reflection, which
`sev` supports with `ssev`) are refused with a message; `stable alpha -beta`
is the reflection of a stable law. A follow-up if wanted.

**Touchpoints.** `decl.lark` (`chf_body` gains the `expr TIMES` prefix and the
`PLUS expr` and `MINUS expr` suffix around the law, scalar `expr` rather than
`numbers`, since a `chfagg` has no vector form); `parser.py` (the `chf_*`
callbacks, `_CHFAGG_FORM`, and the refusals); `characteristic_function.py`
(`FamilyAffine`, `FAMILY_PARAMETERS` and `FAMILY_REQUIRED` as shapes, defaults
`mean 0, sd 1` for `norm`, `mu 0` for the variance gamma); `decl_writer.py`
(`_render_chfagg`); `src/aggregate/data/decl-help/anatomy.html` and
`card.html`; the corpora (`_test_suite.agg` 14 `chfagg` lines,
`decl-testers.agg` 14, `library.agg` one entry), with
`tests/capture_spec_snapshot.py` rerun and the diff read; the
`chfagg` section of `docs/4_dec_Language_Reference.rst` and the regenerated
grammar reference; `tests/test_characteristic_function.py` and
`tests/test_decl_parser.py`.

**Acceptance.**

- `chfagg G 25 * gamma 4 hints{log2=12; bs=1/4}` and
  `agg G dfreq [1] sev 25 * gamma 4` agree bucket for bucket (the existing
  oracle, now with one spelling); `25 * gamma 4 + 10` has `est_m` 110 to
  $10^{-9}$.
- Every refused spelling raises `ValueError` whose message contains the new
  spelling, and that spelling builds the same masses as the old one did at
  1.34.0, to $10^{-12}$, including `stable 1 0.5 2 3` in S1.
- `2 * stable 1 0.5 + 3`: the grid cumulative agrees with scipy
  `levy_stable(1, 0.5, loc=3, scale=2).cdf` at the bucket edges to the
  tolerance the existing scipy oracle tests use.
- Every corpus `chfagg` line round trips: `program` to `build` to `pprogram`
  unchanged.

### [CHF-Aggregate-Frames] the moment store, summary and validation frames in Aggregate's shape

**`stats_df`.** Index `_STATS_ROW_INDEX` (`_aggregate.py:887`, 26 rows),
columns `e0.m0, mixed, independent, after_occ, empirical, occ_impact,
agg_impact, gross_empirical, error`, all float, built at `__init__` and
refreshed by `update`, as `Aggregate` does it:

- `agg` rows `ex1` to `skew`: the theoretical values in `e0.m0`, `mixed` and
  `independent`; the grid's in `empirical` and `gross_empirical`; `error` the
  noise-aware relative error of `empirical` against `mixed`;
- `meta` rows: the values a one-component `Aggregate` with no premium has
  (limit `inf`, attachment 0, `el` the theoretical mean, premium 0, `mix_cv`
  0, `wt` 1), and `NaN` for `lr` and `sevcv_param`, which have no meaning
  here;
- `freq` and `sev` rows, and the `after_occ`, `occ_impact` and `agg_impact`
  columns: `NaN`, as on an `Aggregate` with no reinsurance;
- a moment the law does not have is `inf` or `nan`, the conventions the store
  already uses for a power-law severity.

**`summary_df`.** Index `Freq`, `Sev`, `Agg` (named `X`), columns `Mean | SD |
CV | Skew | P01 | Median | P99`. The `Freq` and `Sev` rows are `NaN`; `qd`
already prints `summary_df.fillna('')`, so they show blank, as Aggregate's
frequency percentiles do. CV through `Aggregate._cv_or_nan`. Before `update`,
the theoretical moments, as on `Aggregate`.

**`validation_df`.** Aggregate's `_describe` shape: three rows; before
`update` the columns `EX | CV | Sk`, after it the eight. The SD trio replaces
the CV trio when the law is signed, `support[0] < 0`, matching Aggregate's
`_signed()` gate (so `norm`, two-sided and $\beta = 1$, $\alpha > 1$ stable laws,
the normal inverse Gaussian, variance gamma, tempered stable and Tweedie
$p \le 0$ use SD). Built from the moment store, with `_snap_noise` on the
value columns as `_describe` applies it.

**The checks fold into `info`** (ruling 4), as an `elt` aggregate's do: its
`info` has a one-line `loss table` row and its `elt_description` a `checks`
row counting the checks by status and naming each warning, with every check
written out in `elt_explanation`. The private `_validation_rows()` stays as
the one source of the check values. `info` becomes Aggregate's rows in
Aggregate's order, `n/a` where they do not apply (claim count, frequency and
severity distribution, loss table, padding, `sev_calc`, `dsev_bucket`,
premium, loss ratio), `normalize` `False` (the grid is never renormalized),
`reinsurance` `none`; then the `chfagg` block: family, parameters as written,
support, scale, total mass, outside mass below and above, tilt, $\theta$,
edge error, truncation error, transfer $\sigma$, most negative bucket,
buckets per scale, and a `checks` row in `elt_description`'s form (`8 ok`, or
`7 ok, 1 flag: outside mass, use hints{log2=17; bs=1/16}`); then the three
tail lines, `bounded` and `id`. Aggregate's `info` already lists every row
with `n/a` before `update`; this applies the same rule. The Loss Lab serves
`info` as a string, so the rows reach it with no work there.

**Names.** `agg_m`, `agg_cv`, `agg_skew` become `actual_m`, `actual_cv`,
`actual_skew`, with `actual_sd`, `actual_var` and `est_var` added, no alias.
Callers: `tests/test_characteristic_function.py` (six lines) and the module
itself; nothing outside reads the old names (`rg` 2026-10-29).
`test_actual_and_est_families_on_aggregate_and_portfolio` extends to a
`chfagg`.

**Exhibits** (`exhibits/_characteristic_function.py`). The four
registrations keep their own captions, which describe a law with no count or
severity, and gain the insurer treatments Aggregate's frames get:
`_summary_flags` on `summary`, `_stats_insurer_moment_store` on `stats` (the
raw moment rows dropped). The validation emphasis waits for `valid`, in the
next step.

**Acceptance.**

- A parity test builds `chfagg` laws (`25 * gamma 4`, `30 * norm + 100`,
  `2 * stable 1.5 1 + 10`, `stable 1.5 0.5`) and `Aggregate` counterparts (an
  unsigned one, and a signed `ssev` one for the signed laws), and asserts
  `summary_df`, `validation_df` and `stats_df` have equal `index` and
  `columns`, before and after `update`.
- `stats_df.loc[('agg', 'mean'), 'mixed'] == actual_m` and
  `stats_df.loc[('agg', 'mean'), 'empirical'] == est_m`.
- For every corpus law, `info` has a row holding the value of each row of
  the 1.34.0 `validation_df`, and its labels equal Aggregate's up to the
  `chfagg` block.
- Exhibit snapshots recaptured (`tests/capture_exhibit_snapshots.py`) and the
  diff read.

### [CHF-Validation-Verdict] Aggregate's verdict, vocabulary and opening

**`valid`.** A `Validation` flag set from a new
`_validation.valid_characteristic_function(obj)`, beside `valid_aggregate` and
`valid_portfolio` (the module's convention: similar bodies kept side by side,
not merged). It sets:

- `AGG_MEAN`, `AGG_CV`, `AGG_SKEW` from the moment store at `eps`, `10 eps`
  and `100 eps`, each live only when the theoretical value is finite and
  above the noise floor: Aggregate's rule, so a window statistic of a moment
  the law lacks never fails;
- `DEFECTIVE` when the deficit $1 - \sum p$ exceeds `DEFICIT_MATERIALITY`
  ($10^{-4}$): the `outside_above` a tilted grid leaves out;
- `ALIASING` when an untilted grid's outside mass exceeds
  `1 - bucket_sizing_p`, or an untilted heavy tail's edge error exceeds
  `EDGE_ERROR_FLAG`: mass that wrapped in, the flag's existing meaning;
- `INFEASIBLE` when `recommended_grid` needs a log2 above
  `RECOMMENDED_LOG2_MAX`;
- **`TRUNCATION`**, a new member appended to `Validation` (existing values
  unchanged; `Aggregate` never sets it), when `truncation_error` exceeds
  `TRUNCATION_FLAG`.

The most negative bucket and buckets per scale stay advisory `info` rows,
outside the verdict: `Aggregate`'s verdict has no negative mass
or resolution test either, and a coarse grid shows up in the moment errors.

**`truncation_error` fix.** For a law with an atom $(x_a, w)$ it becomes
$|\varphi(\pi/h) - we^{i\pi x_a/h}|$, the continuous part the inversion
actually truncates. `tweedie 1.5 100 2` goes from $4.54\times10^{-5}$ to
$1.13\times10^{-7}$ and stops flagging.

**`validation_description`** is `explain_validation(obj.valid, deficit)`, so a
`chfagg` uses Aggregate's words. `explain_validation` gains a `TRUNCATION`
branch (`characteristic function truncated at pi / bs, lower bs`) and a
`subject` keyword, default `'severity'`, so the infeasible prefix can read
`grid infeasible for this law`. `validation_score` and `validation_eps` are
added; `_validation.validation_score` applies unchanged, the `sev` terms
dropping out as `NaN`.

**`validation_explanation`.** The opening, factored into a helper shared with
`Aggregate` and `Portfolio` (their text unchanged, pinned by a test):
"Validation of the realized grid against the analytic moments: fails agg
mean. Each of mean, CV and skewness is compared for the aggregate, and a
relative error above 0.0001 fails; ..." with the components named by the
caller, and a sentence that a law computed from its characteristic function
has no frequency or severity. The `DEFECTIVE`, `ALIASING`, `INFEASIBLE`
paragraphs follow, worded for a law, then a paragraph per check of the `info`
block: the window and the mass outside it below and above against
the budget, with the `recommended_grid` hint when coverage fails; the tilt
pair, $\theta$ and the edge error; the truncation; buckets per scale; the
transfer $\sigma$ of a two-sided law; and, for a power tail with a mean, the
closed-form tail mean beyond the window with the grid mean completed by it
against the theoretical (the measurement above: 9.99995 against 10 for
`2 * stable 1.5 1 + 10`).

**Exhibits.** The validation exhibit gains `_moment_validation_emphasis` with
`obj.valid`, as Aggregate's has.

**Acceptance.**

- `validation_description` is `not unreasonable` for `gamma 100 cv 0.5`,
  `25 * gamma 4`, `30 * norm + 100` and `tweedie 1.5 100 2`; `fails agg mean`
  for `2 * stable 1.5 1 + 10`; contains `characteristic function truncated`
  for `gamma 0.3` on a coarse grid; `n/a, not updated` before `update`.
- `validation_explanation` of every corpus law starts with the shared opening,
  and Aggregate's and Portfolio's are unchanged for the corpus programs.
- The stable explanation's completed mean agrees with `actual_m` to
  $10^{-5}$ relative for the three stable rows of the table above.

### [CHF-Sharpen] a closed-form sharpen

**Why Aggregate's probe does not transfer.** `_bucket_window.sharpen`
re-updates on neighboring cells, scores each with `validation_score` (the
moment errors in tolerance units) and walks the bucket while the score
improves. On a `chfagg` with a power tail the only live moment term is the
mean, and its shortfall is the tail beyond the window, which shrinks as the
window widens whatever the body costs. Measured 2026-10-29 on today's
`stable 1.5 1 2 10` (`2 * stable 1.5 1 + 10`) at log2 16:

| bs | mean rel. error | buckets per scale | edge error |
|---|---|---|---|
| 1/16, the center | $-5.3\times10^{-3}$ | 32 | $1.6\times10^{-4}$ |
| 1/8 | $-3.8\times10^{-3}$ | 16 | $3.9\times10^{-4}$ |
| 1/4 | $-2.7\times10^{-3}$ | 8 | $9.4\times10^{-4}$ |
| 1/2 | $-1.9\times10^{-3}$ | 4 | $2.2\times10^{-3}$ |
| 1 | $-1.2\times10^{-3}$ | 2 | $1.3\times10^{4}$, the tilt collapses to 0 |

Every step improves the score, so the walk runs to `bs_limit` and lands on a
grid with two buckets per scale and an edge error of $10^4$.

**The analog uses the closed forms instead of a walk.** A `chfagg`'s errors
have closed forms, and `recommended_grid` already returns the smallest
`(log2, bs)` meeting coverage, truncation and resolution.

- **Score**, in tolerance units, the power mean of
  `_validation.combine_score_terms` (the `power` argument) over the live
  terms:
  - `u_agg_mean`, `u_agg_cv`, `u_agg_skew` from `validation_score_terms`,
    live by Aggregate's rule, except that a law with a power tail drops
    `u_agg_mean`: its shortfall is the window's tail, which ruling 5 accepts
    and no grid under the cap removes;
  - `u_coverage`, outside mass over `1 - bucket_sizing_p`;
  - `u_edge`, edge error over `EDGE_ERROR_FLAG`, live for a power tail;
  - `u_truncation`, truncation error over `TRUNCATION_FLAG`;
  - `u_resolution`, `MIN_BUCKETS_PER_SCALE * bs / scale`, live where the
    scale is finite.

  Measured at log2 16, power 2:

| declaration today | u_coverage | u_edge | u_resolution | score | recommended `(log2, bs)` |
|---|---|---|---|---|---|
| `gamma 100 cv 0.5` | $10^{-7}$ | n/a | 0.003 | 0.002 | (9, 2) |
| `stable 1.5 1 2 10` | 0.43 | 0.16 | 0.31 | 0.28 | (15, 1/8) |
| `stable 1.5 0.5` | 0.41 | 0.35 | 0.63 | 0.41 | (8, 1/16), a bug, below |
| `stable 1.1 1` | 6.4 | 0.09 | 0.63 | 3.2 | (19, 1/16) |
| `stable 0.7 1` | 579 | 0.01 | 0.16 | 290 | (24, 1/64) |

- **Gate**, Aggregate's: a center at or under `good_enough` that is sound is
  confirmed and nothing runs. The first three rows above are confirmed; the
  last two run.
- **Cells**: the center and the `recommended_grid` cell, its `log2` capped at
  the center's plus one and at `log2_cap`. Like Aggregate's probe it grows the
  grid by at most one `log2` per call, and a re-run continues; a recommended
  `log2` below the center's is taken as it is. Each cell is a real `update`
  (5 to 12 ms at log2 16), so `sharpen_df` holds realized values rather than
  the model's.
- **Pick**: `_bucket_window._pick_cell` unchanged (sound, then thrifty, then
  score), so a recommended cell smaller than the center wins on thrift when a
  forced probe (`good_enough=0`) finds it passes.
- **Outcome**, Aggregate's: `_program.pin_sharpen` pins a moved grid as
  `hints{log2=...; bs=...}` and a confirmed one as `note{sharpen: ...}`;
  `sharpen` returns `self`; `execute=False` restores the center.
- **Signature**, Aggregate's: `sharpen(bs=None, log2=None, *, log2_cap=20,
  bs_limit=16, power=2, good_enough=0.5, execute=True)`. `bs_limit` is
  accepted and inert, since there is no bucket walk, and the docstring states
  it.
- **Frames**: `sharpen_df` with Aggregate's columns in Aggregate's order, the
  `u_sev_*` terms and `aliasing` `NaN`, and the four inversion terms
  appended. `d_bs` is the base 2 logarithm of the bucket ratio, an integer on
  `Aggregate` and not always one here, since `recommended_grid` moves along
  the `round_bucket` ladder. `sharpen_description`, `sharpen_explanation` in
  `chfagg` words (what the score is, which terms are live, why the mean of a
  power tail is not), and `sharpen_program` from `_program.sharpen_program`.

**`recommended_grid` fix, a prerequisite.** For a two-sided law
`grid_window` returns `x_min=None`, and both `recommended_grid` and the
coverage advice in `_validation_rows` treat `None` as covered, so
`stable 1.5 0.5` is advised log2 8, where 1.8% of the mass is outside the
window against a budget of $10^{-5}$. `_outside_at` already estimates a
two-sided law's outside mass without `x_min`; using it gives log2 16 (log2 15
estimates $1.16\times10^{-5}$).

**The Loss Lab.** `can_sharpen` and `has_sharpen`
(`aggregate-api/src/aggregate_api/capability.py:178`, `:211`) test
`hasattr(obj, 'sharpen')` and `sharpen_df`; the route calls
`obj.sharpen(log2_cap=...)` and serves `sharpen_program` and
`sharpen_description` (`routes/objects.py:3788` on). The Sharpen leaf
therefore lights with no api edit once the kind is admitted there, and the
Narrative leaf collects `sharpen_*` by suffix. The `[CHF-Sharpen]` entry in
`dev/TODO.md` closes.

**Acceptance.**

- `gamma 100 cv 0.5`, `2 * stable 1.5 1 + 10` and `stable 1.5 0.5` at log2 16
  are confirmed: nothing runs, `note` gains `sharpen: ...`, `hints` unchanged.
- `stable 1.1 1` moves one `log2` per call and reaches the recommended log2
  19 in at most three calls, scoring under 0.5 there; `program` holds each grid in
  `hints`, and `build(program)` rebuilds it bucket for bucket.
- `recommended_grid` of `stable 1.5 0.5` is log2 16, and the built grid's
  outside mass is within the budget.
- A forced probe on `25 * gamma 4` picks a cell with log2 at or below the
  center's.
- `sharpen_df` columns are Aggregate's followed by the four inversion terms.

### [CHF-Grid-And-Tail-Reports] the grid sizing and tail reports

**`bs_window_df`**, Aggregate's columns exactly (`applies, selected, x_min,
x_max, W, bs, log2, log2_need, coverage, clipped, note`, dtypes as at
1.0.0a275), indexed by `method`, rows: `rule A` (edges from the closed-form
tails at `bucket_sizing_p`), `cap` (the tilted bucket cap, when it binds),
`given` (rule C), `recommended` (`recommended_grid`), and `used`. `coverage`
is the string `1-<outside mass>`, `clipped` the `outside_above` of a tilted
grid, `log2_need` from `recommended_grid`. `bs_description` and
`bs_explanation` in Aggregate's form. The `bs_window` exhibit registers
`AggregateCHF`.

**`tail_class`, `tail_behavior_df`, `tail_description`, `tail_explanation`**,
Aggregate's frame columns (`family, min, max, left_tail, right_tail, bounded,
concentrated, cv, note`) with rows `frequency`, `severity` (`NaN`, note "none:
a characteristic function law") and `aggregate`. Each side's class derives
generically from the family protocol: `POWER_LAW` where the side has a power
tail constant, `BOUNDED` at a finite support end, `SUPER_EXPONENTIAL` where
the cumulant generating function is finite on that whole side (the normal),
else `EXPONENTIAL`. `bounded` becomes derived from it, as on `Aggregate`. The
`tail_behavior` exhibit registers `AggregateCHF`. The three tail lines of
`info` (in place since `[CHF-Aggregate-Frames]`, reading `n/a`) now come from
it.

**Acceptance.** Parity tests on columns and dtypes for both frames against an
`Aggregate`; `bs_window_df` is `None` before `update`; the tail classes of the
eight families (stable one-sided, two-sided and $\alpha = 2$ included) as
listed.

### [CHF-Accessor-Parity] the cheap members

`value_type` (`'loss'`, read-only), `prob_eq_0` (`pmf(0)`, the Tweedie atom),
`spec` (the spec dict), `json`, `density` (`density_df` where `p_total > 0`),
`sample(n, replace=True)`, `price`, `price_ccoc`, `price_pentagon_ex`,
`evaluate` (premium given explicitly; there is no expected premium) and
`apply_distortion` (no reinsurance views), each the thin delegation to
`_pricing` or `density_df` that `Aggregate` uses; `__str__` as Aggregate's
(`info` then `summary_df`); `density_df` gains `log_p`, `exlea`, `e`,
`exgta` and `exeqa` with Aggregate's guards. Every name is vetted against the
class before it is added.

**Acceptance.** For `25 * gamma 4` on `hints{log2=12; bs=1/4}` and the
matching `agg`, `density_df` columns equal, `price`, `price_ccoc`,
`price_pentagon_ex` and `evaluate` agree to $10^{-10}$, `sample(1000)` draws
grid points, and `prob_eq_0` of `tweedie 1.5 100 2` equals the atom to
$10^{-9}$.

## Names vetted (rg over `src/aggregate`, 2026-10-29)

`FamilyAffine`, `chf_scale`, `chf_loc`, `valid_characteristic_function`,
`Validation.TRUNCATION` and the four score terms `u_coverage`, `u_edge`,
`u_truncation`, `u_resolution` occur nowhere. `actual_m`, `validation_eps`,
`bs_window_df`, `tail_behavior_df` and the five `sharpen` members occur on
`Aggregate` and `Portfolio` with the meanings adopted here, and nowhere on
`AggregateCHF`, so nothing is shadowed. The plain words `shift` and `affine`
appear in comments and in unrelated `pnl` code; neither becomes an attribute.

## Order and bumps

| step | bump | depends on |
|---|---|---|
| `[CHF-Arithmetic-Form]` | 1.35.0 | none |
| `[CHF-Aggregate-Frames]` | 1.36.0 | none |
| `[CHF-Validation-Verdict]` | 1.37.0 | frames |
| `[CHF-Sharpen]` | 1.38.0 | verdict |
| `[CHF-Grid-And-Tail-Reports]` | 1.39.0 | verdict |
| `[CHF-Accessor-Parity]` | 1.40.0 | none |

The version numbers assume this plan lands alone; whichever of this plan and
`[Severity-Parameter-Count]` executes second renumbers. Each bump runs the
full suite (`uv run pytest -m 'slow or not slow'`) and updates
`CHANGELOG.md`, `dev/TODO.md` (this plan's entry, and the stable conversion
entry at the first bump) and the docs `.rst` in lockstep; the docs build is
left to the author. The first step's CHANGELOG entry has the second paragraph
a breaking change earns: the refused spellings and their replacements.

## Execution log

Approved by the author 2026-10-30 ("Yes this is approval"); executed from 1.35.0. On the sharpen soundness question raised at review, the author ruled 2026-10-30: "the method needs some tweaks but the concept stands", leaving the analog to the executor.

### [CHF-Arithmetic-Form], 1.35.0

Landed as specified, with these divergences.

1. **The alpha = 1 shift has the other sign.** The plan said the pointer for a refused `stable 1 beta g d` shifts delta by `-(2/pi) beta g log g`. The law `S(1, beta, g, d; 1)` is `g Z + d + (2/pi) beta g log g` for the standard law `Z` (the S1 characteristic function picks up `log g` under scaling with that sign; `FamilyStable.tail_mass_above` already said so), so the pointer adds the term: `stable 1 0.5 2 3` is `2 * stable 1 0.5 + 3.44127120031`. Measured against masses captured from 1.34.0 before any edit: every refused spelling's pointer rebuilds the same grid, with masses within 5.4e-14 (the worst, `stable 1 0.5 2 3` and its `s0` form); `test_a_refused_spelling_points_at_the_same_law` checks each against the unchanged Python full-parameter construction at 1e-12.
2. **scipy's S1 `loc` is already Nolan's delta 1 at alpha = 1, so mapping it is the wrong reference.** The plan expected the reference rebuilt through scipy's `loc` and `scale` to fix the engine plan's finding on `Stable.fz`. Measured: the grid of `2 * stable 1 0.5 + 3` (the law `2 Z + 3`) agrees with scipy's S1 `levy_stable(1, 0.5, loc=2.5587, scale=2)` to 8e-9 and with S0 `loc=3` to 8e-9, and misses S1 `loc=3` by 0.068. scipy's S1 is not equivariant at alpha = 1, so `FamilyAffine.reference` rebuilds a stable base through `Stable` at the mapped scale and S1 location; every other base maps scipy's `loc` and `scale`. The plan's acceptance line (`levy_stable(1, 0.5, loc=3, scale=2)`) holds in scipy's S0; the test checks the S1 law with the corrected `loc`, and the reference. **For the author:** this also suggests the engine plan's finding that `Stable.fz` is shifted at alpha = 1 was itself mistaken, since `Stable.fz` passes delta 1 as scipy's S1 `loc`; not pursued here.
3. **`FamilyAffine.location` is not `c delta1 + b` at alpha = 1.** It is the transformed law's S1 location, computed through S0, which is equivariant: `Family.location_under(c, b)`, overridden on `FamilyStable` as `s0_to_s1(alpha, beta, c gamma, c delta0 + b)`. With the plan's formula the window edges sat 0.44 off and the grid differed from 1.34.0's.
4. **`Family.to_decl` renders the arithmetic form** through a new `_affine_parts()` per family, `(c, b, body)`, which `FamilyAffine` composes. Without it a law built from Python with full parameters displayed a body the parser now refuses, and `arithmetic_spelling` would have needed a second renderer. Numbers are written to twelve significant digits.
5. **A shifted atom.** Not in the plan: `2 * tweedie 1.5 100 2 + 10.3` puts the atom at 10.3, which the inversion requires to be a grid point. `_plain_window` now anchors the grid on the atom (0 for every law without one, so nothing else moves).
6. **New public names** beyond the plan's list, vetted by `rg` over `src`: `LEGACY_PARAMETERS`, `arithmetic_spelling`, `render_affine`, `Family.location_under`; `FamilyAffine` keeps `base`, `c` and `b` as attributes.
7. **More files carried old spellings** than the plan's touchpoints: `tests/test_charts_ir.py`, `tests/test_exhibits.py`, `tests/test_fcc_surface.py`, `tests/capture_exhibit_snapshots.py` and `dev/regen_features.py`. `anatomy.html` needed no change (it names the keyword only); `card.html`'s example now shows the arithmetic. Corpus: `CHF.VG` drops its zero `mu`; new testers `CFE.Shifted`, `CFE.Lowered`, `CFE.StableOne`, `CFE.StdNorm`, `CFE.InvGaussMu`, `CFE.TweedieAt`. The exhibit snapshot did not move.
8. **The Loss Lab ripple, not in the plan.** `aggregate-api/tests/conftest.py:73` probes `chfagg Gate.Probe gamma 4 25`, which now fails to parse, so the API's `chfagg` tests skip rather than fail; three fixtures use the same spelling (`test_capability.py:52`, `test_objects.py:2846`, `:2899`). `gamma 100 cv 0.5` parses on both sides of the change. Reported, not edited.

9. **A glued minus is ambiguous, and stays so.** The tier 3 gate's `test_no_new_grammar_ambiguities` caught `CHF.VG`, `variancegamma 0.2 0.5 -0.1`: `NUMBER` absorbs a glued minus, so `-0.1` is the third shape or `MINUS 0.1`. The shipped resolver reads it as the sign, which the variance gamma's negative theta needs; a spaced `- 10` is the shift. `sev` resolves the same text the other way (`sev 25 * gamma 4 -10` is a shift), so the refusal of `chfagg G 25 * gamma 4 -10` says a glued minus is a sign and gives the spaced spelling. Making it unambiguous needs a new shift terminal, which would ripple into `parser_errors._TERMINAL_LABELS` and the app's `decl-keywords.json`; instead `CHF.VG` joins `KNOWN_AMBIGUOUS` with the reason, as `UM.Scaled` did, `_NAME_RE` there learns `chfagg`, and a test pins both readings.

### [CHF-Aggregate-Frames], 1.36.0

Landed as specified, with these divergences.

1. **The empirical moments are window statistics, with no deficit relocation.** `Aggregate` builds its `empirical` column through `xsden_to_mwrangler`, which puts a deficit at the implied maximum loss. A tilted `chfagg`'s deficit is `outside_above`, a closed-form tail rather than truncation, and `[CHF-Validation-Verdict]` completes the mean with that tail, which needs the window statistic. So `empirical` is the sums over the grid, the same numbers `est_m`, `est_sd` and `est_skew` were at 1.35.0, and `stats_df.loc[('agg', 'mean'), 'empirical'] == est_m` holds exactly. Relocating would have moved the window mean of `2 * stable 1.5 1 + 10` from 9.9469 to about 9.965.
2. **The `stable 1.5 0.5` CV of minus 677.6 stays.** The plan cited it as the case `_cv_or_nan` repairs. It does not: the window mean is minus 0.0116 against an SD of 7.86, a ratio of 1.5e-3, above `CV_MEAN_REL_TOL` (1e-3), so `Aggregate` would print the same. The theoretical CV, `nan`, is in `validation_df` and `stats_df`.
3. **`info` holds the check values, not the moment rows.** Of the 1.34.0 check table, the inversion rows (total mass, most negative bucket, outside mass, edge error, tilt, transfer sigma, truncation error, buckets per scale) are lines of `info`, `outside mass` with its split, `outside mass 4.34e-06, below 0, above 4.34e-06`; the moment rows are the `Err` columns of `validation_df`, and `checks` counts the inversion checks only. A private `_checks_frame()` keeps the old table as the one place the tests and the explanation read the check values from, beside `_validation_rows()`.
4. **The `meta` block follows the plan's list** (`mix_cv` 0, `attachment` 0 in every theoretical column), where a measured one-component `dfreq [1]` `Aggregate` has `mix_cv` `NaN` and `attachment` `NaN` in `e0.m0`.
5. **Snapshots.** The recapture also moved `economic_tail/raw/Tower` in the last digit (257.10242225105003 to 257.10242225105), floating noise unrelated to this step; the committed entry is kept and only the six `AggregateCHF` entries move. The summary, stats and validation captions are rewritten for the new shapes.
6. **`dev/FEATURES.csv`** marks `actual_m`, `actual_sd`, `actual_cv`, `actual_skew`, `actual_var` and `est_var` on `AggregateCHF`. The library's `StableLaw` note now points at `info` for the tilt and edge error.

### [CHF-Validation-Verdict], 1.37.0

Landed as specified, with these divergences.

1. **The mean check is not gated on the noise floor.** The plan made each moment flag live only where the theoretical value is finite and above the noise floor, calling that `Aggregate`'s rule. `valid_aggregate` gates only CV and skewness; its mean test is the noise-aware error, absolute at a zero mean. `valid_characteristic_function` follows the code, so `stable 1.5 0.5` (mean 0, window mean minus 0.0116) reads `fails agg mean`, as its 1.34.0 row did and as ruling 5 has it; a mean that does not exist (alpha at most 1) gives a `nan` error and no flag.
2. **The atom is removed from every truncation measure, not only `truncation_error`.** With the whole law's value, `recommended_grid` of `tweedie 1.5 100 2` returned (24, 1/32768), since no bucket reduces the atom's `exp(-lambda)`, which would have set `INFEASIBLE` falsely. The new `truncation_at(family, h)` serves `truncation_error`, `recommended_grid` and the rule A bucket walk. The walk's grid moves for a law whose atom pushed it: `tweedie 1.8 1000 5` keeps bs 1/4 where it took 1/8 (no corpus line). `tweedie 1.5 100 2` keeps its grid.
3. **The completed mean uses the asymptotic series on a one-sided law.** With the leading term alone `stable 1.1 1` completed to 3.6e-4 of the scale, the plan's own 2.86458 against 2.86494, outside the 1e-5 acceptance. `FamilyStable.tail_mean_above` integrates the series `tail_mass_above` sums, `sum_k t_k u^(1 - k alpha) k alpha / (k alpha - 1)`, which brings it to 4.4e-6; `FamilyAffine` delegates. A two-sided law keeps the leading term (its tails come from both halves, and the second term vanishes at alpha = 1.5), 3.9e-7. The acceptance's "relative" is read against `max(|actual_m|, scale)`, since three of the four laws have mean 0.
4. **`ALIASING` is worded for a law.** `explain_validation`'s phrase for it, `agg mean lost in the convolution, possible FFT wrap; raise log2`, is wrong for a law with no convolution; with `subject='law'` it reads `mass outside the window wrapped in; raise log2`. `Aggregate` and `Portfolio` pass no subject and are unchanged.
5. **The opening helper is `validation_opening(short, eps, components)`**, a list of the two sentences, so each caller appends. Measured: `validation_description` and `validation_explanation` of all 157 `agg` and `port` lines of `_test_suite.agg` are identical before and after; a test pins the opening literal for one `Aggregate` and one `Portfolio`.
6. **`INFEASIBLE` is computed only when coverage, truncation or resolution fails on the grid at hand**, since `recommended_grid` walks log2 8 to 24 and most laws need none of it; the verdict is cached on `_valid` and cleared by `update`. The coverage advice in the checks keeps its coverage-only test, so a law short of truncation is not told coverage needs log2 above 24.

### [CHF-Sharpen], 1.38.0

Landed as specified, with these divergences.

1. **Soundness is the coverage budget.** The plan's gate was `Aggregate`'s, which marks a cell unsound at a deficit above 1e-12 (`VALIDATION_NOISE`); a tilted window always leaves its tail out (`2 * stable 1.5 1 + 10` at log2 16, 4.3e-6), so no stable center would ever have been confirmed, against the plan's own acceptance. Raised at review; the author ruled 2026-10-30 that the concept stands with the method tweaked. The analog: `defective` in `sharpen_df` is `u_coverage > 1`, the outside mass over `1 - bucket_sizing_p`, which `_pick_cell` and the gate read unchanged.
2. **`stable 1.1 1` at log2 19 scores 0.615, not under 0.5.** It reaches log2 19 in exactly three calls (17, 18, 19) and is sound there (`u_coverage` 0.92); a grid that just meets the recommended coverage carries `u_coverage` near 1 and `u_resolution` 0.625, so its score sits near 0.6 by construction. The test checks the three calls, soundness, and the bucket for bucket rebuild from `program`. A fourth call finds the recommended cell equal to the center and confirms it.
3. **`sharpen_explanation`'s `reason`, `recommended` and the one or two cells** are this probe's; `sharpen_description` reuses `_bucket_window`'s `_fmt_bs`, `_move_phrase` and `_sharpen_fmt`, so the two classes print the same phrases.
4. **Not registered for the `sharpen` exhibit.** The plan does not ask for it, and the app's Sharpen leaf reads `sharpen_program` and `sharpen_description`, not the exhibit. Registering `AggregateCHF` there is one call and a caption when wanted, and would add to the Loss Lab's exhibit list.
5. **`dev/FEATURES.csv`** marks the five sharpen members on `AggregateCHF`.

### [CHF-Grid-And-Tail-Reports], 1.39.0

Landed as specified, with these divergences.

1. **`tail_class` is `TailClasses(None, None, agg)` with `agg` the heavier side.** The plan named the method without its value. `Aggregate`'s triple is frequency, severity and aggregate rungs; a law has no first two, and its aggregate rung is taken as the heavier of its two sides (`_tail._heaviest`), so a signed law with a power left tail reports `power-law`.
2. **A bare characteristic function (`from_chf`) is `undetermined` on both sides**, `TailClass.UNKNOWN`, since it has no closed forms to read; its `bs_window_df` has rule A inapplicable and the grid `given`.
3. **`bs_window_df`'s `selected` marks the method the grid came from** (`rule A`, `cap` or `given`), as `Aggregate`'s marks the winning method, and `used` is never selected. `coverage` on `rule A` and `recommended` is the budget the window is sized to (`1-1e-05`), on `cap`, `given` and `used` the mass actually outside (`1-4.34e-06`). The `recommended` row of a two-sided law has no `x_min` (the crop places it), so its edges are `nan`.
4. **`bs_description` says `wraps in` for an untilted grid and `leaves out` for a tilted one**, and adds `(raise log2 to N)` when the recommended `log2` is larger, in `Aggregate`'s form.
5. **`dev/FEATURES.csv`** marks the seven members on `AggregateCHF`; the two exhibit lists in the tests grow to six.

