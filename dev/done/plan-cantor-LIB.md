# Execution notes: [Cantor-Severity]

Companion to `dev/plan-cantor.md`. One section per phase, and every divergence
from the plan recorded at the moment it is made.

## Review, 2026-09-25

Verified against the tree at `b4e2108`, version `1.0.0a345`.

Confirmed sound: `_classify_sev` and the `Severity._registry` dispatch are as
described; the DecL transformer rules `sev0_zero_params` and `sev0_one_param`
both set `sev_scale = 1.0` and leave `sev_a` at its `np.nan` default, so the
shape resolution order in the plan works; `moms()` takes the precomputed
`sev1/sev2/sev3` fast path exactly when `attachment == 0` and
`detachment == inf`; `_bucket_window.bs_window` honors an explicitly passed
`bs` verbatim and never rounds it to a binary fraction; `_severity_lattice()`
returns `None` for any continuous component, so a Cantor severity cannot
disturb bucket selection. Name vetting: `rg -i cantor src/` and
`rg natural_bs src/ tests/` are both empty, and `bs` is not an attribute or
method on `Severity` or its parents.

The plan's mathematics checks out: the moment recursion follows from the
self-similar identity, the cdf digit iteration puts `F(a) = F(1-a) = 1/2` in
the gap branch as it should, and the characteristic function product reduces
to the classical `exp(it/2) prod cos(t/3**k)` at `c = 1/3`.

## Divergences

### D1. The corpus file is `_test_suite.agg`

The plan (and `CLAUDE.md`) name `src/aggregate/agg/test_suite.agg`. The file
was renamed to `_test_suite.agg` and `tests/capture_spec_snapshot.py` reads
that name. Corpus lines go there.

### D2. `cantor_bs` is `scale / q**m`, and that is the right lattice

`hacks/cantor_fft.py` returns `bucket_size = 2 / 3**m`, the gcd of the level-m
atoms `2j / 3**m` and so the coarsest exact lattice, which invited the worry
that the plan's `scale / q**m` was needlessly fine. Measured, the plan is
right and the hack's lattice is the worse choice. A level-m cylinder has width
`1/q**m`, and the discretization centers its buckets on a half-bucket offset,
so with `bs = 1/q**m` each cylinder straddles the bucket pair `(2j, 2j+1)` and
its mass `2**-m` lands **exactly half in each**, bit for bit, with nothing
anywhere else and an aggregate mean of exactly `0.5`. With `bs = 2/3**m` the
whole cylinder lands in one bucket at its **left endpoint**, which reproduces
the level-m atoms literally but biases the mean low by `a**m / 2` and fails
validation at coarse m. The acceptance check "the discretized severity equals
`cantor_pmf` masses" therefore holds in the paired form, `d[2j] + d[2j+1] ==
ps[j]`, which is what `tests/test_cantor_severity.py` asserts.

### D3. Attainable cv range is closed at the low end

The plan writes the cv range as the open interval `(1/sqrt(3), 1)`. Since
`c = 0` is admitted and gives the uniform law exactly, `cv = 1/sqrt(3)` is
attained: the range is `[1/sqrt(3), 1)`. The validation message says so.

### D4. `cantor_pmf` takes the base `q`, contradicting decided question 2

Section `[cantor-module]` specifies `cantor_pmf(log2_points, q=3)` generalized
to any integer base, while decided question 2 records "ship it in the module
(c = 1/3 only, documented)". The acceptance checks require
`cantor_pmf(m, q=4)` for the `c = 1/2` library entry, so the design section and
the acceptance checks agree against the summary line. Implemented with the `q`
parameter defaulting to 3.

### D5. `V:/worktrees/dev-files.md` no longer exists

The `execute-plan` skill and the session memory both point at a control list at
`V:/worktrees/dev-files.md` carrying a status row per plan. That file is gone;
the only file at `V:/worktrees/` is `BETA-MERGE.md`, which is a merge reference
card and not a plan status list. No control-list row was updated.

### D6. Oracle agreement is exact on generic points, Holder-bounded on boundaries

The plan's first acceptance check asks the shipped cdf to agree with the exact
ternary oracle "to within a few ulp" on a grid that includes ternary rationals.
Measured: agreement is **bit for bit** (zero ulp, 50,000 uniform draws plus gap
midpoints, subnormals down to `5e-324`, and approaches to 1), and the two part
company only at exact ternary rationals and at Cantor-set atoms, where the
general iteration's rescaling by `1/a` can place a boundary point in the
neighboring cylinder. The observed gap there is `2.9e-11`, which is the Holder
bound the plan itself predicts (exponent `log 2 / log 3` about 0.631, so one
ulp of coordinate error buys about `1e-10` of cdf error) and is far below
anything a discretization can resolve. `tests/test_cantor.py` therefore splits
the check in two: bit-for-bit equality on the generic grid, and a `1e-9`
Holder tolerance on the boundary grid, with an assertion that the boundary grid
really is where the two differ.

## Phase 1, [cantor-module], a346

`src/aggregate/cantor.py` and `tests/test_cantor.py`. 43 unit tests.

Implementation notes beyond the plan:

- **Leading-zero accumulator.** Both the staircase and the quantile loop carry
  an integer count of leading zero digits beside the float accumulator, and
  return `ldexp(frac, -shift)`. Halving a weight 54 times from the first
  nonzero digit would lose a result of order `2 ** -200` entirely; counting the
  leading zeros separately keeps full *relative* accuracy at every magnitude.
  `cdf(5e-324)` returns `7.97e-205`, the right order for `F(x) ~ x ** 0.631`.
- **`_sf` is the mirrored recursion, not `1 - _cdf`.** The two branches swap
  roles, which keeps a small survival probability's relative accuracy in the
  right tail where the complement would have lost every digit. The default
  discretization differences the survival function, so this is the path that
  matters.
- **`_stats` is vectorized over `c`**, with skewness returned as an exact zero
  by the reflection symmetry rather than as the difference of two nearly equal
  third moments.

## Phase 2, [severity-wiring] and the DecL surface, a347

### D7. `sev1` / `sev2` / `sev3` are withheld from a reflected or spliced severity

The plan says to set the precomputed moments from the frozen distribution
unconditionally. `_apply_reflect`'s own docstring warns why that is unsafe:
under plain `sev` the answer is the *clamped* law `max(Y, 0)`, and a populated
`sev1` diverts `moms()` into its fast path and returns the unclamped value
(the bug `sev -lognorm 10 cv 0.5` once had). The same holds for a splice, whose
conditioning `_apply_lb_ub` applies after `_build`. `SeverityCantor._build`
therefore fills the three only when `not sev_reflect and sev_lb == 0 and
sev_ub == inf`, and otherwise leaves them `None` so `_numerical_moms` does the
work. Verified correct against a hand calculation on the layer `0.3 xs 0.2`,
where the conditional layer mean is exactly `(61/360) / (3/4)`.

### F1. The auto-sized grid fails validation on a bare single-claim Cantor

`build('agg C 1 claim sev cantor fixed')` with no grid chooses
`bs = 2**-16, log2 = 16`, a window of exactly `1.0`, which is the severity's
whole support. The top half bucket is then cut, and a Cantor severity keeps
`S(1 - bs/2) = F(bs/2)` about `6e-4` of its mass there because `F` is Holder
with exponent `0.631`, so the mean comes out `2.4e-4` low and validation reports
"fails sev mean, agg mean". The identical grid on `sev uniform` loses only
`4e-6` and passes. This is the severity-agnostic sizer meeting a law whose mass
piles up against its upper endpoint, not a defect in the Cantor code: one more
`log2`, or any of the hinted grids, makes the mean exact to `1e-16`. Left alone
deliberately, since the plan rules `recommend_bucket` out of scope and parks the
general fix as `[Natural-Lattice-Snap]`. Worth the author's attention: it is the
first thing a reader trying `sev cantor` at the prompt will see.

### F2. A new `RuntimeWarning` in `_validate_moments`, found and fixed

`Severity('cantor', sev_cv=0.7)` reached `_validate_moments` with `sev_mean = 0`
and computed `sev_cv * sev_mean / (sev_mean + loc)`, which is `0/0` and warned
`invalid value encountered in scalar divide`. No scipy severity can reach it:
the one-shape branch raises when both `sev_mean` and `sev_scale` are zero,
whereas the Cantor path defaults the scale to 1 and needs no mean. The fix is
general, in `_validate_moments`: with no declared mean there is nothing for the
`loc` shift to restate the target cv against, so the target is the declared cv
as written. The library's no-`RuntimeWarning` property (`[RuntimeWarning-Census]`,
`1.0.0a220`) is preserved.

### D8. The docs carry no severity-name catalogue to update

`[docs-lockstep]` asks for the `.rst` severity-name lists to gain `cantor`.
There are none: `dhistogram` and `chistogram` appear only in two incidental
prose sentences, and the scipy zoo is never enumerated in the doc tree. Two
substantive touches were made instead. `docs/3_reference/3_x_Distribution.rst`
now says that one named severity is the library's own rather than scipy's, and
`docs/2_aggregate_overview/bucket-selection.rst` records that an explicitly
passed `bs` is honored as written, which is what lets the Cantor severity ask
for a base-`q` bucket. Docs are pending a rebuild, per the standing rule.

### D9. `decl-testers.agg` gains a `CAN.` section, not a new letter prefix

The skill requires every DecL program used in a pytest case to live in
`decl-testers.agg`. Its documented scheme is letter prefixes `P..X` continuing
`A..O`, and every single letter is taken, so the eight Cantor programs go under
the descriptive prefix `CAN.`, following `ASV.`, `MXT.`, `HINT.` and the other
later sections that already departed from single letters.

### F3. A pre-existing `RuntimeWarning` from the `RenewalDeterministicWait` library entry

The numerics gate, `pytest -m 'slow or not slow' -W error::RuntimeWarning`,
fails on `test_every_library_entry_builds[agg:RenewalDeterministicWait]` with
`invalid value encountered in sqrt` at `moments.py:500`, where
`MomentWrangler.mcvsk` takes the square root of a negative central variance.
Confirmed **pre-existing**: the same test fails the same way with `_severity.py`
and `library.agg` checked back out to `1.0.0a346`, before any of this phase's
edits. Left alone, since fixing it would move numbers on an unrelated entry.
It means the `[RuntimeWarning-Census]` property of `1.0.0a220` has since
lapsed on the renewal path, which is the author's to triage.

### F4. One flaked gate run

The first tier 3 run failed
`test_reins_bivariate.py::test_netceded_refuses_a_pin_it_cannot_honor`. It
passes in isolation and the immediately following full rerun was green at 5,209
passed, which is the shape of the nondeterminism `CLAUDE.md` records for the
bivariate suites. Both runs are reported rather than only the green one.
