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

### D2. `cantor_bs` is `scale / q**m`, finer than the hack's lattice

`hacks/cantor_fft.py` returns `bucket_size = 2 / 3**m`, which is the gcd of
the level-m atoms `2j / 3**m` and so the *coarsest* exact lattice. The plan
specifies `cantor_bs = scale / q**m`, twice as fine at `q = 3` and three times
as fine at `q = 4`, and writes the resulting library hint values (`1/59049`,
`1/4096`) explicitly. Both choices align the level-m cylinder sets with
buckets, so both are exact; the plan's is implemented as written. The cost is
that every other bucket carries zero mass at `q = 3`.

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
