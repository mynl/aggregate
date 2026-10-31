# Note: grid sizing for lattice severities, with reinsurance, windows and the infeasible case

`[Lattice-Grid-Sizing]`. Written 2026-10-30 against `dev` at 1.43.1 (`03aaec6`). A plan for a
plan: the facts, measured on 858 generated programs against an independent oracle, the defects
they expose, and the rule a plan should implement. Nothing in the repository changed except this
note and the study scripts under `dev/lattice-study/`. It supersedes the TODO entry
`[Layered-Dsev-Lattice]` and, from `D:\Projects\Using\bugs\aggregate-review-answers-1.43.qmd`,
item 1 and the items `[Portfolio-Window-Severity-Reach]` and `[Aggregate-Ceded-At-Limit]`
merged into it there. The plan is `dev/done/plan-lattice-grid-sizing.md` (2026-10-31).

## The question

A discrete severity whose atoms are whole multiples of a dyadic step $h$ has an aggregate on the
lattice $h\mathbb Z$ whatever the frequency, and every reinsured view of it (gross after a layer,
ceded, net, the ceded and net of an aggregate cover) is on a lattice too. An exact grid exists:
every atom of every view on a grid point, the grid just large enough to hold the law. The sizer
sometimes finds it, sometimes picks a coarser bucket that puts an atom between grid points, and
sometimes picks a window that does not hold a claim. The three examples that started this:

| Program | Today | Right |
|:--|:--|:--|
| `agg T 100 claims dsev [1000 1001] fixed` | `bs` 2, `log2` 16 from 0: atom 1001 split, variance 75 for 25, `fails sev cv, agg cv` | `bs` 1, `log2` 10, window from 99,543 |
| `port P agg A dfreq [1] dsev [3]` | `bs` 1, `log2` 1, window from 3: the claim of 3 is off the unit's severity grid, mass 0, `fails pmf deficit 1` | `bs` 1, `log2` 2: a grid of width 4 holds the claim wherever its origin is |
| `agg AWK dfreq [1:100] dsev [1000 1001]` | `bs` 2, `log2` 16, `fails sev cv, agg skew`, no reason given | infeasible at `log2` 16: needs 17 at `bs` 1, and `log2=17` builds exactly |

The third is the important one: **infeasible at the requested `log2` is a legitimate answer**,
and the library should say so and say what `log2` would do, instead of coarsening in silence and
leaving the verdict to notice the symptom.

## The specification the oracle checks

For a program whose severities are discrete, let $h$ be the dyadic step of the pooled atoms of
**every view** (gross after any layer; ceded and net of each occurrence layer; for an aggregate
cover `share po y xs a`, the points `share * atoms`, `share * a` and `share * y`), let $R$ be the
width of the hull of the gross atoms after the layer and 0 (for `dsev [1000 1001]` that is 1001;
for `dsev [-100:50]` it is 150; for `dsev [-5 -4]` it is 5), and let $[lo, hi]$ be the support
of the law that `density_df.p_total` holds. The severity is discretized on its own grid of
`2**log2` points, `xs_sev = (arange(N) - i0) * bs`, with physical 0 at index `i0`: 0 for a
nonnegative severity, `ceil(-lo_sev / bs)` for a signed one, so the grid starts at the severity's
lower endpoint (`_severity_negative_buckets`, `_aggregate.py` 4225). Whatever the output origin,
that grid must span $R$ as well as the law's support. Two grids are available at `bs = h`:

- from 0: `log2` $\ge \lceil \log_2(\max(hi, R)/h + 1) \rceil$;
- windowed at $lo$: `log2` $\ge \lceil \log_2(\max(hi - lo, R)/h + 1) \rceil$.

A program is **feasible at a cap** when the smaller of the two fits it, and the library should
then return the exact law (total variation under $10^{-4}$, which admits FFT rounding on
thousand-claim books; the real failures are at 0.5 and above). When neither fits, the program is
**infeasible at that cap**: whatever grid is used is wrong by construction, the verdict must say
`fails`, and a warning should name the smallest `log2` that works. A program whose views have no
dyadic step (`30% po` of integer atoms, decimal atoms) is outside the specification and keeps the
continuous estimator.

The oracle in `dev/lattice-study/oracle.py` computes the law by convolution on the integer lattice
(numpy, exact to rounding), the step by exact rational gcd (no depth limit), and the two needs.
It shares no code with the sizer.

## The program sets

`dev/lattice-study/sweep.py`, seeded, 858 programs in eight sets. Atoms from six families (small
integers, hundreds, near atoms such as `[1000 1001]` and `[10002 10007]`, thousands, dyadic
fractions, signed); counts 1 to 1000 fixed, `dfreq` lists and ranges, Poisson 1 to 100; one
occurrence layer with share 100%, 50% or 25%, net or ceded; one aggregate cover; a layer on the
`dsev`; one-unit and two-unit Portfolios drawn from the rest. The sweep runs in a process pool,
about 15 seconds per variant on this machine.

## Results

Each program is classed by the specification: `exact`; `WRONG` and whether the verdict contains
`fails` (`flagged`) or not (`silent`); `infeasible` at `log2` 16 and whether the library warned or
only the verdict flagged it. The variants are the prototype patches in
`dev/lattice-study/patches.py`, monkeypatches used for measurement only. All row sums are 858.

| Variant | exact | WRONG flagged | WRONG silent | infeasible, warned | infeasible, flagged only | no lattice |
|:--|--:|--:|--:|--:|--:|--:|
| 1.43.1 as it stands | 699 | 62 | **68** | 0 | 27 | 2 |
| `[views]` only | 782 | 42 | 5 | 0 | 27 | 2 |
| `[window-holds-severity]` only | 722 | 37 | 70 | 0 | 27 | 2 |
| the four patches together | **817** | 12 | **0** | 0 | 27 | 2 |

Figures from the 2026-10-31 run with probabilities printed at full precision; see defect 7 for
why the earlier run, with six-digit probabilities, differed.

The 68 silent wrong answers today are all reinsured: 44 of 80 occurrence programs, 14 of 60
aggregate-cover programs, and 10 Portfolios holding such a unit, return a wrong distribution
with the verdict `reinsurance; subject not unreasonable`, because the verdict covers the subject
only. `agg X 1 claims dsev [1000 2000 3000] occurrence net of 1500 xs 250 fixed` is the
smallest: net is 250, 500 or 1500, the library takes `bs` 1000 and returns 0, 1000 or 2000, mean
right, law wrong.

Grid size does not grow: the median `log2` of an exact program is 7 before and after, and the
count at `log2` 16 goes from 7 to 13 of about 800.

**The twelve that remain wrong under the four patches**, all flagged, all reinsured:

- Five Aggregates that need a windowed grid **and** carry occurrence reinsurance or an aggregate
  cover (`100 claims dsev [1000 1001] occurrence net of 25% po 1500 xs 200 fixed` and the like):
  from 0 they need `log2` 16 to 20, windowed 12 to 16, and the sizer refuses to window a
  reinsured book. That refusal is right today (the occurrence machinery carries the severity on
  the output grid) and lifting it is a separate plan.
- Seven Portfolios: the one-unit twin of one of those Aggregates (drawn twice into set G), four
  two-unit books with a reinsured unit, and one two-unit book the Portfolio coarsens to `bs`
  0.5 where the oracle fits `bs` 0.25 at `log2` 16.

The three named examples under the four patches: `T` builds at `bs` 1, `log2` 10 from 99,543
and is exact; `P` at `bs` 1, `log2` 2 from 3 and is exact; `AWK` stays at `bs` 2, `log2` 16,
wrong and flagged, which is the case the warning is for.

**The 27 infeasible programs.** Every one is flagged by the verdict (`fails sev cv` and the
like) and none is warned. Rebuilt at the `log2` the oracle names, 22 become exact, today and
with the patches; the five that do not are two-unit Portfolios, which at the named `log2` still
take a bucket coarser than the step (`bs` 2 at `log2` 18 where the step is 1 and fits), the
Portfolio sizer defect of the next section. The suggested `log2` is therefore reliable for an
Aggregate and a lower bound for a Portfolio until that sizer follows the same rule.

## A unit alone and in a Portfolio

A one-unit Portfolio adds nothing to the mathematics, so building every single-unit program of
sets A to F twice, as `agg` and as `port P agg ...`, at the same cap, isolates the Portfolio
sizer and the combine from the program (`dev/lattice-study/port_vs_agg.py`, 718 programs):

| Variant | same grid | laws agree (TV < 1e-9) | laws differ materially (TV > 1e-4) |
|:--|--:|--:|--:|
| 1.43.1 as it stands | 117 | 550 | 127 |
| the four patches | 138 | 657 | 22 |

**The combine is sound.** Whenever the two grids both hold the law, the Portfolio reproduces the
Aggregate to 1e-9; the roll-combine in `_portfolio.py` (about line 2834) loses nothing. Of the
657 that agree under the patches, 558 agree on different grids: the Portfolio takes a larger
`log2` in 405 (its window covers 12 nines on both edges where the Aggregate trims the low edge
to 6), a different origin in 154, a different bucket in 2.

**The Portfolio sizer is a second implementation and it chooses differently.** `port_best_window`
(`_bucket_window.py` 1795) sizes a three-moment window on the total (`estimate_agg_window` on
`actual_m`, `actual_sd`, `actual_skew`), picks a bucket in `_port_lattice_bucket`, and has no
severity guard; a unit's own `_bs_window` is run only to read its width. Of the 22 material
disagreements under the patches, 18 are infeasible books where both are wrong and the Portfolio
takes the coarser bucket in 14; in 3 the Aggregate is exact and the Portfolio wrong (`dfreq [1:3]
dsev [10002 10007]` fits `bs` 1 at `log2` 15 alone and gets `bs` 2 in a Portfolio; two
`3 claims dsev [10002 10007] fixed aggregate ...` books are windowed by the Portfolio under a
cover); in 1 the reverse. The same sizer is why five infeasible two-unit books do not cure at the
named `log2`.

**What the plan should require.** A one-unit Portfolio takes its unit's grid verbatim: `bs`,
`log2` and origin from the unit's `_bs_window`, so `port P agg A ...` and `agg A ...` are the same
object on the same grid. A Portfolio of several units applies the rule of the next section with
the total's band as the usual window, the step over the pooled views of every unit, and the
largest unit hull as `R`. The one-unit comparison becomes a test (every program of sets A to F,
same grid and TV below 1e-9) alongside the two-unit set H in `[Correctness-Oracle]`.

## What `nttagg` already does, and what it settles

`AggregateNTT` (`ntt.py`) is the exact engine: `dfreq` times `dsev` on nonnegative integers,
no layer, reinsurance, Poisson count or Portfolio, `bs` pinned to 1 from 0, `log2` trimmed to
`bit_length(max N * max s)`. Three things follow for this note.

- **It is the precedent for the infeasible answer.** A cap below what the support needs
  raises: `the exact support 0..100100 needs log2 >= 17, got 16. Pass build(..., log2=17) or add
  hints{log2=17} to the declaration`. The same program spelled `agg` coarsens to `bs` 2 and
  returns a wrong law with `fails sev cv`. That inconsistency is a finding in itself, and the
  lattice warning should carry the same sentence.
- **It never windows and never takes a lattice step**, so it is a reference answer, not a
  sizing model: `nttagg T dfreq [100] dsev [1000 1001]` needs `log2` 17 where the windowed grid
  needs 10, and `dsev [1000 2000 3000]` runs at `bs` 1 over the whole support. The sizing rule
  below is for `agg`; `nttagg` keeps its own.
- **It validates the oracle.** `dev/lattice-study/cross_check_ntt.py` builds every integer
  program of sets A and B as an `nttagg` with exact fractional probabilities and compares
  `p_exact_float` with the enumeration: total variation under $10^{-13}$ on every program that
  fits its cap. The `[Correctness-Oracle]` plan should use `nttagg` as the reference wherever a
  program is in its scope and the enumeration for layers, reinsurance, Poisson counts and
  Portfolios.

## The defects, by mechanism

1. **The lattice step is taken from the unlayered gross atoms only.** `Aggregate._severity_atoms`
   (`_aggregate.py` about line 7005) returns `support_atoms`, which `SeverityDHistogram._build`
   stores before the layer; `_exact_discrete_window` (about 7290) pools the same and takes the
   support of the gross sum even when `p_total` is the net or ceded sum; `_pooled_lattice`
   (`_bucket_window.py` 2227) pools the same for a Portfolio and for `sharpen`. So a layer, an
   occurrence layer or an aggregate cover whose attachment or share is off the gross lattice
   sizes on the wrong step, and the exact-support row sizes the wrong law. Pooling the views
   cures 47 programs on its own and is where the silent failures live.
2. **A windowed grid must span the severity, and only the Aggregate has the hull.** The
   severity grid has `N` points whatever the output origin, so `N*bs` must cover the hull of
   the gross support and 0 (from 0 for a nonnegative severity, from the lower endpoint for a
   signed one). `bs_window` has the guard
   (`sev_hi < extent`, about line 1298) and **rejects** the window when it fails, falling to a
   coarser grid from 0, where it could grow `log2` to the need, up to the cap, and keep the
   exact bucket. `port_best_window` and `_port_lattice_bucket` have no guard at all:
   `port P agg A dfreq [1] dsev [3]` takes the nice divisor 1 of the step 3 at `log2` 1, a grid
   of width 2 that cannot hold the claim of 3, and loses all its mass. Growing `log2` to the
   need (and, when even the cap cannot hold it, taking the grid from 0 if that fits) cures the
   `100 claims dsev [1000 1001] fixed` family at `bs` 1, `log2` 10. The docstring of
   `_severity_high_estimate` (`_aggregate.py` 7053) states the severity is discretized on
   `[0, N*bs]` with physical 0 at index 0; that holds only for the nonnegative severities the
   windowed row admits, and the plan should restate it in terms of `i0`.
3. **Windowing under a cover.** The Aggregate refuses to window under occurrence reinsurance
   (line 1258) but not under an aggregate cover, whose ceded amount can sit below the window's
   origin: `3 claims dsev [8 10] fixed aggregate ceded to 2 xs 4` cedes 2 always and returns 12.
   The Portfolio windows a book with a reinsured unit of either kind. Both should keep the
   0-based grid when it fits the cap.
4. **Off-lattice coarsening is silent.** When no lattice candidate fits the cap, `_size`
   drops the lattice and the resolution estimate runs; the only trace is the note `lattice 1
   too fine for the log2 cap, bs 2 off it` in `bs_window_df`. No warning, no verdict word, no
   number. The need is known at that moment (`_need_log2` of the window at the step), so the
   warning can say `needs log2 17 at bs 1; log2 16 was given, so bs 2 is used and the atoms do
   not sit on the grid`.
5. **The exact support row is laid from 0, never at its own origin.** For a fixed or `dfreq`
   count on a lattice `_exact_discrete_window` returns the exact support (100,000 to 100,100 on
   step 1 for the `T` example), but `_size` puts the origin at `x_lo` only when
   `signed or force_origin`, and only the windowed row passes `force_origin=True`. Sized from
   0 the width 100 becomes 100,100, step 1 needs `log2` 17, the bucket coarsens to 2, and the
   corner-reachability guard (`EXACT_DISCRETE_REACH_LOGP`, written for exactly that overstated
   extent) rejects the row. Meanwhile the windowed row has the right origin and bucket
   (100,026, `bs` 1, `log2` 6) and is rejected by the severity-fit guard (defect 2). The cure is
   one argument, `force_origin=True` on the exact row, plus the hull growth of defect 2; the
   corner guard then matters only as the fallback when the exact window does not fit the cap at
   the lattice bucket. A prototype that added the exact window as a separate selection step
   with its own slack arithmetic (`patch_exact_windowed`) broke four small books and was not
   kept; the fix belongs inside `_size`, not beside it.
6. Minor: `LATTICE_DEPTH` is fixed at 16 (`utilities.py` 343) while the cap is a parameter. The
   depth answers "could the largest atom fit on a grid at this step", which is the cap's
   question; it should follow `log2`.

7. **`dsev` probabilities that do not sum to 1 are used as given in the moments and
   normalized in the density.** `SeverityDHistogram` keeps `pk` as passed (`_severity.py` 3110)
   and its moments are the sums over them, so a deficit is an atom at 0 the law does not have;
   the discretized `sev_density` is normalized to 1. Probabilities printed to six digits,
   `[0.333333 0.222222 0.444444]`, leave 1e-6 unaccounted, and for `100 claims dsev [999 1000
   1001]` that takes the aggregate sd from 8.75 to 13.29 and the skew from -0.02 to -42.6; the
   three-moment window from those moments runs from 29,648 to 100,015 for a law whose support
   is 100 wide. The first run of this study carried that artifact in every program with
   weighted atoms, which is why its tables differed (747 and 689 for the single patches
   against 782 and 722 now). Recommendation: normalize at construction when the sum is within
   a tolerance of 1, and raise outside it; the sizer should never see moments of a law other
   than the one it discretizes.

## The rule the plan should implement

Stated by the author, 2026-10-31. Run the usual method. If the severities are on a lattice,
take the lattice bucket `h` (its nice divisor first). If `h` is finer than the usual bucket, the
lattice cannot fit the cap: keep the usual grid and warn with the lattice warning. Otherwise snap
`x_min` down to a multiple of `h` and shrink `log2` by `log2(h / bs_usual)`, retaining at least
the range the usual grid had. The range a grid must retain is the larger of the aggregate band
and the severity hull with 0 (`R`), so the shrink stops while **both** hold.

In `_size` this is one change: `need = max(_need_log2(x_hi - x0, b), _need_log2(R, b))` in the
candidate walk and in the free-bucket branch, with `R` passed in. The `windowed` row then comes
out at `bs` 1, `log2` 10 for the `T` example instead of `log2` 6, and its severity-fit guard
(`sev_hi < extent`) has nothing left to reject. The same `need` in `port_best_window` gives the
Portfolio the guard it never had. The rest follows:

1. **Views**: `h` is taken over every view (defect 1), so ceded and net bucket on the lattice.
2. **No window** under occurrence reinsurance (as now), under an aggregate cover (defect 3), or
   for a signed book, until their own plans.
3. **Infeasible**: when no candidate fits the cap, keep the usual grid as today **and** warn with
   a sibling `LatticeGridWarning` naming the step, the `log2` needed at it, the `log2` given and
   the bucket used; put the same words in `bs_description` and the verdict
   (`Validation.INFEASIBLE` precedent, `_validation.py` 177).
4. **Portfolio**: a one-unit Portfolio takes its unit's grid verbatim; otherwise the same
   rule with the total's band as the usual window, the step over the pooled views, and `R` the
   largest unit hull.

Not in this plan: the `exact_discrete` row at its own origin (defect 5, one residual where the
moment band sits off center on a near-degenerate book), windowing a reinsured book (5 of the 12
residuals), a signed book sized from its realized origin (`[Signed-Portfolio-Origin]`), and
atoms with no dyadic step.

## Decisions

Rulings by the author, 2026-10-31, on the questions the first draft raised.

1. **Infeasible: warn and build.** `nttagg` raises; the continuous `InfeasibleGridWarning` warns
   and builds. `agg` warns and builds, with the verdict word and `nttagg`'s sentence naming the
   `log2` to pass. A build that completes with a flagged answer is worth more than an exception
   when the `agg` is one unit of something larger, and the aLL status strip shows the verdict.
2. **No growth past the cap** beyond what `WINDOW_LOG2_GROWTH` already allows the exact step in
   the windowed method. The cap is the user's; the warning names the number to pass.
3. **A sibling warning class.** `InfeasibleGridWarning` is worded for a thick continuous law
   ("add an occurrence limit"); a lattice book's remedy is the opposite, a larger `log2`.
   `LatticeGridWarning`, its own text, the same `warn_once` keying.
4. **The `exact_discrete` row at its own origin is not needed** once the `windowed` row is sized
   on both conditions (withdrawn; defect 5 records the mechanism).

## Suggested shape of the plan

Five bumps, in this order, each with the sweep and the one-unit comparison rerun and their
tables recorded:

- `[Dsev-Probability-Normalization]`: `SeverityDHistogram` normalizes probabilities whose sum
  is within a tolerance of 1 and raises outside it (defect 7). Small, first, so every later
  measurement is of the law the user wrote.
- `[Lattice-Views]`: one function returning the atoms of every view, used by
  `_severity_atoms`, `_exact_discrete_window`, `_pooled_lattice`. Expect 782 exact, 5 silent.
- `[Window-Holds-Severity]`: `_size` sizes on both conditions and the severity-fit guard goes;
  no window under an aggregate cover. Expect `T` at `bs` 1, `log2` 10 and every Aggregate set
  at its four-patch figure.
- `[Portfolio-Follows-Unit-Grid]`: a one-unit Portfolio takes its unit's grid; a Portfolio of
  several units sizes by the same rule on the total's band, the pooled views and the largest
  unit hull; no window with a reinsured unit. Expect 817 overall, every single-unit program on
  the same grid alone and in a Portfolio, and the five two-unit infeasible books to cure at
  the named `log2`.
- `[Lattice-Infeasible-Warning]`: `LatticeGridWarning` with `nttagg`'s sentence, the verdict
  word, `bs_description`, and `LATTICE_DEPTH` tied to the cap. Expect the 27 to read
  `infeasible, warned`, with a test that the named `log2` builds each one exactly.

Then the oracle, the sets and the one-unit comparison move into `tests/` as
`[Correctness-Oracle]` (the review's item 1 step 3), seeded and fast: the whole sweep is 10
seconds in a pool and about a minute serial.

## How to rerun

```
.venv/Scripts/python.exe dev/lattice-study/sweep.py base     # the library as it stands
.venv/Scripts/python.exe dev/lattice-study/sweep.py four     # the four prototype patches
.venv/Scripts/python.exe dev/lattice-study/sweep.py views    # or window: one patch alone
.venv/Scripts/python.exe dev/lattice-study/port_vs_agg.py base   # or four: a unit alone and as a one-unit Portfolio
.venv/Scripts/python.exe dev/lattice-study/cure.py four      # the 27 infeasible at their named log2
.venv/Scripts/python.exe dev/lattice-study/cross_check_ntt.py   # the oracle against nttagg, sets A and B
```

`res_base.pkl` and `res_four.pkl` hold the frames behind the tables above (one row per program:
program text, oracle step and needs, the grid taken, total variation, verdict, class).
