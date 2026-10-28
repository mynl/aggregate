# Plan: an exact bucket size for discrete severities on a fractional lattice

`[Fractional-Lattice-Bucket]`. Drafted 2026-10-28 against `dev` at 1.19.0 (`8f75495`), from the
PMIR project (`D:\Projects\PMIR`), for the aggregate session. Design settled with Steve on
2026-10-28 against `dev` at 1.23.0 (`93640a7`); see Methodology and Decisions. Reviewed the same
day; the four review rulings are folded into the rule and listed under Decisions. **Status: executed,
1.24.0 to 1.26.0 on 2026-10-28; see the execution log.**

## Goal

When every severity atom is a whole multiple of a common dyadic step, choose a dyadic `bs` so that
each atom lands exactly on a grid point and the grid is sized to the support, for fractional and
signed atoms as already happens for integer atoms. One lattice-step routine and one bucket rule
serve the univariate sizer, the Portfolio and the bivariate severity.

## Why

A discrete severity on a lattice has an aggregate on the same lattice, whatever the frequency, so
an exact and small grid is available. Today aggregate finds it only when the atoms are integers.
The first example below, three atoms at quarter points, gets a grid about 2,000 times larger than
it needs.

The immediate user is PMIR, the book *Pricing Multi-period Insurance Risk* and its package `pmir`
(`V:\dev\PMIR`). Its DMC pricing method sums the mean-zero increments of a Doob martingale, which
are signed fractions, as a Portfolio of signed `dsev` units. The legacy PMIR package ran its own
FFT convolution partly to get an exact grid for these sums; the rule for `pmir` is that grids are
aggregate's job.

The bivariate discrete severity is in worse shape (below): it takes a floating-point gcd as the
axis bucket, which gives a non-dyadic `bs` on decimal atoms and an unusably fine one on noisy
atoms. Steve: "that is a horrible situation at present". It is fixed here under the same rule.

## Current behavior

Measured on `dev` at 1.19.0 (and identically on the 1.11.0 release) with
`build('agg T 3 claims dsev [atoms] [.25 .5 .25] fixed')`, with the later rows measured at 1.23.0:

| atoms | count | `_severity_lattice()` | `bs` | `log2` |
|:--|:--|:--|:--|:--|
| 0.25, 0.5, 1.75 | 3 fixed | `None` | $2^{-13}$ | 16 |
| -1.75, 0.25, 0.5 | 3 fixed | `None` | $2^{-13}$ | 16 |
| 0.75, 1.5, 2.25 | 3 fixed | `None` | $2^{-13}$ | 16 |
| 3, 6, 9 | 3 fixed | 3.0 | 1 | 5 |
| 3, 6, 9 | 3 Poisson | 3.0 | 3 | 6 |
| 0.1, 0.2, 0.3 | 3 fixed | `None` | $2^{-16}$ | 16 |

- `Aggregate._severity_lattice` (`_aggregate.py`, line 6991) returns the gcd of the atoms when all
  are integers (to `atol=1e-9`) and `None` otherwise. `_bucket_window._sharpen_lattice` and
  `_bucket_is_exact` (lines 2057 and 2073) delegate to it, taking an integer gcd across units for a
  Portfolio, and the sizer `_size` (line 962) uses the result as `lattice_bs`.
- Quarter-point atoms therefore fall to the generic estimator: $2^{-13}$ divides 0.25, so the grid
  is exact by accident, but 65,536 buckets cover a support that needs about 22 at `bs = 0.25`.
- Decimal atoms are not multiples of $2^{-16}$, so each is spread over two buckets.
- The 3, 6, 9 split between counts is an inconsistency, not a decision. `_exact_discrete_window`
  (line 7269) returns a hard-coded `bs_lattice` of `1.0` (line 7383), and the `exact_discrete` row
  wins for a fixed or `dfreq` count; under a Poisson count that row does not apply, and the
  `moment` row uses the gcd, 3.
- Portfolio: `port_best_window` (`_bucket_window.py`, line 1651) sets
  `bs = round_bucket(max(min_k bs_k, W / N))` and takes no gcd across units. Measured: Poisson
  units on steps 3 and 5 get `bs = 4`, on neither lattice; a signed Portfolio of quarter-point
  units gets $2^{-10}$ at `log2` 16.
- Bivariate: `bivariate._lattice_bs` (line 1169) is a floating-point Euclid gcd (`_float_gcd`,
  line 1155), used as the axis `bs` in discrete mode (line 1903) and treated like a pinned bucket.
  Atoms are placed by `np.round(x / bs)` (line 2140). Measured: axis atoms 0, 0.1, 0.2 get
  `bs = 0.1`; 0.1, 0.2, 0.3 give `0.09999999999999998`; 0, 0.1, 0.3000001 give about $10^{-7}$,
  and the build had not finished after 90 seconds. `_netceded_exact_bs` (line 995) uses the same
  `_float_gcd`, with a budget test that rejects a lattice too fine to afford.
- `_renewal._lattice_step` (line 287) is a gcd of `Fraction.limit_denominator(10**9)`
  approximations verified to `1e-9` relative. It sizes the renewal wait grid, which is internal to
  the claim count, and recovers 1/10 and 1/3 by design.
- Policy that constrains any change: `utilities.round_bucket` (line 236) keeps a sub-unit `bs` a
  power of two, deliberately binary exact ("0.2 / 0.4 / 0.8 are not"), and `snap_bs_to_reference`
  (`_bucket_window.py`, line 712) keeps the grid commensurable with a referenced object's `d` by
  snapping to $d \cdot 2^m$.

## The legacy algorithm

`D:\Projects\PMIR\Python\pmir_package\src\pmir_legacy\working.py`: `dyadic_step_exact` and
`dyadic_step_tol`, with the helpers `_gcd_many` and `_ctz`. Read-only reference; the code is marked
there as slated to move here.

- **Exact.** Every float is exactly a dyadic rational $p_i / 2^{k_i}$ (`float.as_integer_ratio`).
  Put all of them over $2^K$, $K = \max k_i$, take the gcd $g$ of the integer numerators and cancel
  its factors of two. The step $b = m / 2^h$, $m$ odd, generates the additive group generated by the
  atoms, so every $x_i / b$ is an integer in exact binary arithmetic.
- **Tolerant.** The smallest $h$ such that every $x_i$ lies within `tol` of $2^{-h}\mathbb Z$
  (optionally all but a fraction `robust_frac`), followed by the same gcd reduction.

On the examples above, and on $1/3, 2/3$:

| atoms | exact | tolerant, `tol=1e-6` |
|:--|:--|:--|
| 0.25, 0.5, 1.75 | 0.25 | 0.25 |
| -1.75, 0.25, 0.5 | 0.25 | 0.25 |
| 3, 6, 9 | 3 | 3 |
| 0.1, 0.2, 0.3 | $2^{-55}$ | $2^{-19}$ |
| 1/3, 2/3 | 1/3 (as a float, height 54) | $2^{-19}$ |

Both find only dyadic lattices, which is what this plan wants. The routine below is the tolerant
form with a noise-level tolerance and no `robust_frac`.

## Methodology

**Principle** (Steve, 2026-10-28). aggregate is a numerical library, not an exact-arithmetic one.
`bs` is always a dyadic float, $n / 2^s$ with $n$ an odd integer: a non-dyadic `bs` has caused
repeated grief. Within that, the chosen `bs` must look reasonable and hold the true aggregate in as
small a vector as possible, with no aliasing, no loss of precision and minimal discretization
error. When those pull apart, a nice bucket, in the sense of `round_bucket`, is preferred to the
exact step: 0.25 rather than 0.75, and 1 rather than 3. It is the more natural grid, and the one a
later operation (a reinsurance layer, a referenced object's `d`) is likely to force anyway.

**The rule.**

1. **Lattice step.** Pool every atom of every severity component (of every unit, for a Portfolio),
   with 0 included, so the lattice is linear. Search $r = 0, 1, 2, \dots$ while
   $2^{-r} \ge 2^{-D} \max_j |x_j|$, with depth $D = 16$, and take the first $r$ at which every
   atom is within $\epsilon \max_j |x_j|$ of $2^{-r}\mathbb Z$. Round each $x_i 2^r$ to the
   integer $p_i$ and set $h = \gcd_i(p_i) / 2^r$, written $h = n / 2^s$ with $n$ odd ($s$ is
   negative for an even integer step: $1000 = 125 \cdot 2^3$). If the search ends without a hit,
   there is no lattice and the standard method runs. Decimal atoms such as 0.1 and thirds never
   hit, so they have no lattice. $\epsilon = 10^{-12}$; all atoms 0 gives $h = 1$, as today.

   The depth is what makes a hit mean something. Searching down to $2^{-r} \approx \epsilon
   \max|x|$, as first drafted, always hits: at that depth the tolerance covers half a step, so
   every atom is "on" the lattice. Measured on the draft rule, atoms 0.1, 0.2 hit at $r = 41$ with
   $h = 0.09999999999990905$, which then passes the fit, while 0.1, 0.2, 0.3 get
   $h \approx 4.5 \times 10^{-13}$ and fail it, the split decided by rounding noise. At $D = 16$
   the step is at least $2^{24}$ times the tolerance. $D = 16$ matches the default `log2`: a
   lattice finer than $2^{-16} \max|x|$ is not a sensible bucket, and a power-of-two `bs` from the
   standard estimator usually lands a fine dyadic lattice exactly anyway.
2. **Candidates.** First the largest nice bucket that divides $h$, then $h$ itself. Nice means
   `round_bucket`'s set: powers of two below 1, and the ladder $\{1, 2, 4, 5, 8\} \cdot 10^k$ from
   1 up (`_BUCKET_LADDER`). Below 1 the nice divisor is $2^{-s}$, the odd part $n$ replaced by 1
   (0.75 to 0.25). From 1 up it is the largest ladder value dividing $h$ (3 to 1, 6 to 2, 12 to 4,
   15 to 5, 300 to 100, 1000 stays 1000); a non-integer step above 1 gets a power of two (1.5 to
   0.5).
3. **Fit.** Take the first candidate $b$ at which the method's own window fits:
   `_need_log2(span, b) <= log2`, the requested `log2` (default 16) acting as the cap. A candidate
   that does not fit the cap is not nice: the nice divisor is held to the cap on every row,
   including `windowed`. Only the exact step $h$ may use `grow_cap` on the `windowed` row, as the
   lattice step does today. Then shrink `log2` to the need. If no candidate fits there is no
   lattice, and the standard estimator runs exactly as today.

Steve's framing of step 3, for a fraction, is to replace $n$ by 1 when $\log_2 n$ is no more than
the overhead left, $16 - \text{need}(h)$. Since $\text{need}(2^{-s}) \approx \text{need}(h) + \log_2
n$, that is the fit test on the first candidate, so the cap is the only margin. From 1 up the same
test applies with the ladder divisor in place of $2^{-s}$. A very fine exact step ($s$ near
55) fails the fit by itself, so no separate height threshold is needed: once the lattice is that
fine, the standard methods apply.

**Worked examples** (cap 16 unless noted; `log2` by the inclusive count of `_need_log2`):

| atoms | count | today `bs`, `log2` | $h$ | rule `bs`, `log2` | reason |
|:--|:--|:--|:--|:--|:--|
| 0.25, 0.5, 1.75 | 3 fixed | $2^{-13}$, 16 | 1/4 | 0.25, 5 | $n = 1$ |
| -1.75, 0.25, 0.5 | 3 fixed | $2^{-13}$, 16 | 1/4 | 0.25, 5 | signed, $n = 1$ |
| 0.75, 1.5, 2.25 | 3 fixed | $2^{-13}$, 16 | 3/4 | 0.25, 5 | 0.25 fits |
| 0.75, 1.5, 2.25, `log2=4` | 3 fixed | | 3/4 | 0.75, 4 | 0.25 needs 5 |
| 1.5, 3, 4.5 | 3 fixed | $2^{-12}$, 16 | 3/2 | 0.5, 5 | power of two |
| 3, 6, 9 | 3 fixed | 1, 5 | 3 | 1, 5 | unchanged |
| 3, 6, 9 | 3 Poisson | 3, 6 | 3 | 1, 8 | ladder divisor |
| 3, 6, 9 | 1,000,000 Poisson | 3, 15 | 3 | 3, 15 | `windowed`: 1 needs 17 |
| 5, 10, 15 | 3 fixed | 1, 6 | 5 | 5, 4 | on the ladder |
| 1000, 2000, 5000 | 3 fixed | 1, 14 | 1000 | 1000, 4 | on the ladder |
| 1000, 2000, 5000 | 3 Poisson | 1000, 7 | 1000 | 1000, 7 | unchanged |
| 0.1, 0.2, 0.3 | 3 fixed | $2^{-16}$, 16 | none | as today | not dyadic |

Today's `bs` comes from `round_bucket`'s ladder only when `bs` is free. An integer lattice is used at
its gcd under a random count and at 1 under a fixed count (the hard-coded `1.0`), so the rule moves
the integer rows that are off the ladder under a random count, or on it under a fixed count. Each
move keeps the same probabilities at the atoms. The integer rows were measured at 1.23.0; the rule's
`log2` is the need at the rule's `bs` on the same window. The million-claim row is concentrated, so
the `windowed` row wins; its band is about 75,000 wide (the count, not the atoms, sets the width),
so `bs = 1` would need `log2` 17, past the cap, and the exact step 3 stays, at 15.

## Decisions

Each design question, answered by the methodology.

- **[Decimal-Lattices]** Dyadic lattices only, option (a). Decimal atoms have no lattice and go to
  the standard estimator as today; the rational step is not used for `bs`. Options (b) and \(c),
  a non-binary `bs` or an integer grid in units of the step with a scale on the labels, are out.
  \(c) is noted as a candidate for the integer (NTT) engine, `nttagg`, as a separate future plan.
- **[Lattice-As-Bucket]** Settled by the rule: the bucket is the largest nice divisor of the step
  when that fits, else the step itself, below 1 and above alike. The hard-coded `1.0` in
  `_exact_discrete_window` is replaced by the lattice step, so the fixed and random counts agree:
  3, 6, 9 gets `bs = 1` and 5, 10, 15 gets `bs = 5` under either.
- **[Lattice-Fits-Grid]** The fit test of step 3, against the same cap and `grow_cap` the sizer
  already uses. No height threshold.
- **[Lattice-Search-Depth]** (review, 2026-10-28) The search stops at depth $D = 16$ below
  $\max|x|$, a module constant `LATTICE_DEPTH` beside `LATTICE_RTOL`: find a sensible step if one
  exists, otherwise run the standard method. `_dyadic_lattice_step` takes `depth` as a keyword.
  `_netceded_exact_bs` passes a depth that reaches its gross `bs`, $\lceil \log_2(\max|x| /
  \text{gross bs}) \rceil$ when that exceeds 16, because its atoms are the images of the whole
  gross grid, so $\max|x|$ is the grid top, and the gross `bs` must stay findable when the gross
  `log2` exceeds 16.
- **[Windowed-Nice-Cap]** (review, 2026-10-28) The nice candidate must fit the requested `log2`
  on every row. `grow_cap` is only for the exact step on the `windowed` row, as today, so `log2`
  never grows past the request just to make `bs` nicer.
- **[Bivariate-Axis-Cap]** (review, 2026-10-28) In discrete mode an axis fits its candidates
  against its measured width (from `_size_axes`) with the cap `total_log2 - min_axis_log2`, 16 at
  the defaults, the most the free split gives one axis. When both axes fit, there is no further
  joint budget test, as today. An axis with no fitting candidate is left free.
- **[Bivariate-Off-Lattice-Scheme]** (review, 2026-10-28) Same approach in 1-D and 2-D: an off
  lattice joint matrix is scattered with the unit's `dsev_bucket` scheme, `'linear'` by default
  (`config.py`), which is what builds `g0` and `g1`. The bilinear split's margins are exactly the
  1-D linear scatter, so the row and column sums of `S` equal `g0` and `g1`. The draft's
  `'nearest'` did not match the default and would have broken that check.
- **[Lattice-Tolerance]** The tolerance exists only to absorb floating-point noise from arithmetic
  that should have landed on the lattice (a conditional mean, a scaled atom), never to move an atom
  a real distance. Relative, $\epsilon = 10^{-12}$ of $\max |x|$, as a module constant
  `LATTICE_RTOL` beside `round_bucket`. One tolerant routine covers the exact case. The largest
  residual $\max_i |x_i - p_i / 2^r|$ goes in the bucket table note. `robust_frac` is dropped:
  every atom lands, or there is no lattice.
- **[Signed-Lattice-Origin]** `_size` already sets the origin to $\lfloor x_{lo} / b \rfloor b$,
  which is exact for a dyadic $b$ once the fit holds (the integer factor is below $2^{53}$). Tests
  confirm every atom lands on a grid point after the window's pad balancing and after
  `snap_bs_to_reference`. The lattice is linear (0 included); a shifted lattice such as
  $0.1 + \mathbb Z$ helps only a fixed count and is not pursued.
- **[Lattice-Portfolio]** The Portfolio step is the step of all units' pooled atoms (their dyadic
  gcd), and the rule picks the bucket once, against the Portfolio's window and cap. The units'
  separately chosen buckets are not combined. Any continuous unit means no lattice, as now.
- **[Lattice-Numbers-Move]** Numbers move, to the exact answer: a minor bump per step and a
  "numbers move" sentence in each CHANGELOG entry. No opt-out setting; pinning `bs` is the opt-out.
  Moves expected: programs with fractional dyadic `dsev` atoms (coarser exact `bs`, smaller grid);
  integer lattices under a random count whose gcd is off the ladder (Poisson 3, 6, 9: `bs` 3 to 1);
  integer lattices under a fixed or `dfreq` count whose ladder divisor is above 1 (5, 10, 15: `bs`
  1 to 5); Portfolios of lattice units; bivariate discrete axes with decimal or noisy atoms. The
  integer moves keep the same probabilities at the atoms on a different grid, so `density` values
  scale with `bs`.
- **[Lattice-Pinned-Bucket]** A pinned `bs`, or a pinned `log2`, is left alone. A pinned `bs` that
  misses the lattice gets a note in the bucket table, not a warning.
- **[Lattice-Routine-Name]** Private. `_dyadic_lattice_step(atoms, rtol=LATTICE_RTOL,
  depth=LATTICE_DEPTH)` returns `(h, residual)` or `None`; `_lattice_candidates(h)` returns the
  candidate buckets in preference order. Both live in `utilities.py` beside `round_bucket`, where
  `bivariate.py` can import them. None of the four names (`LATTICE_DEPTH` included) occurs in
  `src/aggregate` today (checked with `rg` at 1.23.0, `LATTICE_DEPTH` at review); recheck at
  implementation.
- **[Renewal-Excluded]** `_renewal._lattice_step` is left alone. Its grid is internal to the claim
  count, it needs the rational steps 1/10 and 1/3, and its results must not move. The plan's former
  `[Lattice-Routines-Consolidated]` step is dropped.

## Proposed change

Three steps, each a minor version bump with its own commit.

1. **[Severity-Lattice-Fractional]**
   - Add `LATTICE_RTOL`, `_dyadic_lattice_step` and `_lattice_candidates` to `utilities.py`, with
     NumPy docstrings whose Notes give the rule and the reason for the power-of-two preference.
   - `_severity_lattice` returns the routine's $h$, so dyadic fractional atoms have a lattice.
   - `_exact_discrete_window` accepts atoms on a dyadic lattice, not only integers, and returns $h$
     in place of the hard-coded `1.0`.
   - `_size` receives $h$ and picks the bucket by the rule, per row, with that row's cap
     (`grow_cap` for $h$ only, on the `windowed` row). `_size` also keeps a forced-bucket input
     with no candidate search, for the reference snap (`snap_bs_to_reference`), which re-runs
     `_size` with $d \cdot 2^m$ and must not have it replaced by a nice divisor.
   - The `exact_discrete` row says "bs coarsened" only when no candidate fits, not whenever the
     chosen bucket differs from $h$.
   - `_sharpen_lattice` and `_bucket_is_exact` use the pooled dyadic step across units in place of
     the integer gcd.
   - The linear `dsev_bucket` path (`discretize_severities`, `_aggregate_compute.py`, with
     `_rebucket_to_grid`) treats a fractional grid position below the tolerance as on the grid, so
     noise leaves no sliver of mass in the next bucket. The default path already snaps each atom to
     its nearest bucket.
   - Bucket table: the `exact_discrete` description becomes "the exact discrete (lattice) support",
     and the note shows $h$, the chosen bucket and the residual, or that a pinned `bs` misses the
     lattice.
2. **[Portfolio-Lattice-Bucket]** `port_best_window`: when every unit has a lattice, the rule picks
   the bucket from the pooled step against the Portfolio's window and cap, ahead of the
   `round_bucket(max(min_k bs_k, W / N))` formula, which stays for every other case. For a signed
   Portfolio the window is the wrap-safe span, $\max(W, \max_k W_k)$, as the current formula
   floors it, and `log2` shrinks to the need there too (today it stays at the cap).
3. **[Bivariate-Dyadic-Lattice]**
   - `bivariate._lattice_bs` and `_netceded_exact_bs` use `_dyadic_lattice_step` and the rule;
     `_netceded_exact_bs` keeps its budget test as the fit. `_float_gcd` is deleted (its only
     callers are these two); `tests/test_bv_discrete.py` and `tests/test_netceded_sizing.py`
     import `_lattice_bs` and `_float_gcd`, and those tests are rewritten against the new
     routine.
   - Discrete mode, an axis with a lattice that fits the axis cap ([Bivariate-Axis-Cap]): the
     rule's bucket and the exact index placement, as now.
   - Discrete mode, an axis with no lattice or no fitting candidate: the axis `bs` is left free
     and measured by `_size_axes`, as for a copula axis, and the joint matrix is placed with
     `scatter_bivariate` (`scatter_bivariate_sparse` on the massive path) using the units'
     `dsev_bucket` scheme ([Bivariate-Off-Lattice-Scheme]), offset by the signed lay-in `i0`, so
     the row and column sums of `S` equal `g0` and `g1` on the grid.

## Acceptance

- The worked examples give the rule's `bs` and `log2`, every atom on a grid point.
- Decimal atom sets (0.1, 0.2; 0.1, 0.2, 0.3; 0.1, 0.2, 0.4) and thirds have no lattice and
  size exactly as today.
- Atoms perturbed by a few units in the last place get the same `bs` as the clean atoms, and each
  atom's mass sits in one bucket under both `dsev_bucket` settings; the bucket table shows the
  residual.
- A signed Portfolio of three units with dyadic `dsev` atoms gets the rule's bucket for the pooled
  step, with $(x - x_{min}) / bs$ an integer for every atom; Poisson units on steps 3 and 5 get
  `bs = 1` (today 4).
- Bivariate: axes on 0, 1, 2 and on 0.25, 0.75 keep today's `bs`; an axis on 0, 0.1, 0.2 gets a
  dyadic `bs` with margins matching `g0` and `g1`; an axis on 0, 0.1, 0.3000001 builds in normal
  time; the `MV.DBV*` programs in `decl-testers.agg` are unchanged.
- The suite passes at every tier the bump calls for. Over `src/aggregate/agg/reproductions.agg`,
  `recitations.agg` and `library.agg`, every program whose `bs` changes is listed and checked, and
  each falls in one of the moves listed under [Lattice-Numbers-Move].

## Files

`src/aggregate/utilities.py` (the routine, the candidates, `LATTICE_RTOL`),
`src/aggregate/_aggregate.py` (`_severity_lattice`, `_exact_discrete_window`),
`src/aggregate/_aggregate_compute.py` (the linear `dsev_bucket` path),
`src/aggregate/_bucket_window.py` (`_size`, `_sharpen_lattice`, `_bucket_is_exact`,
`port_best_window`, the bucket table notes), `src/aggregate/bivariate.py` (`_lattice_bs`,
`_netceded_exact_bs`, discrete-mode placement), tests, `CHANGELOG.md`, `dev/TODO.md` (no
`[Fractional-Lattice-Bucket]` entry exists at review; one is added and ticked at landing). Not
`src/aggregate/_renewal.py`.

The plan moves to `dev/done/` in the 1.24.0 commit, carrying Steve's uncommitted edits to it along
with the review's (agreed 2026-10-28).

## Usage doc

Wanted (Steve, 2026-10-28): `hacks/using/using-lattice-buckets.qmd`, outside version control and
never committed, written after the final step. Usage docs now live in `hacks/using/`, not at the
top of `hacks/`. Follow `using-stable.qmd` and `using-tweedie.qmd` there: the same front matter
(`format: html` with `toc` and `embed-resources`, the jupytext and kernelspec block), and
`from using_common import wprint, GT` for printing and tables. A few paragraphs on the rule (the
dyadic step, the nice candidate, the fit, the depth), then short blocks, one idea each: the worked
examples, with the 1.23.0 `bs` and `log2` from the table above beside the new ones; a signed Portfolio of dyadic units; noisy
atoms landing in one bucket with the residual in the bucket table; decimal atoms with no lattice; a
pinned `bs` that misses the lattice; a bivariate axis on and off a lattice. Every block is run once
in `.venv` and each sentence checked against what it prints; the doc is not rendered.

## Downstream

- `pmir_legacy` keeps its copy of the legacy routine while its own callers (`aggregate_fft` and the
  `fft` method of `emergence`) remain, and drops it when they are replaced.
- `pmir`'s emergence code builds signed Portfolios and relies on steps 1 and 2 for an exact grid.
- The PMIR record is `V:\dev\PMIR\dev\state-of-play.md`, under "Slated for aggregate".

## Execution log

Executed 2026-10-28 from `dev` at 1.23.0 (`88ccc2a`). A baseline grid for every `agg`, `port` and `bivariate` program in `reproductions.agg`, `recitations.agg` and `library.agg` (340 programs) was captured at 1.23.0 before any code moved, for the acceptance diff.

### Rulings at execution (Steve, 2026-10-28)

- **[Lattice-Search-Start]** The rule's search "r = 0, 1, 2, ..." never tries a step coarser than 1, so an atom set with $\max|x| > 2^{16}$ finds no lattice: `Homer2003.HC` (`dsev [200000 400000 600000 800000 1000000]`, 1.23.0 `bs` 200000 at `log2` 7) would have fallen to the generic estimator at `log2` 16. Ruled: the search starts at the coarsest power of two at or below $\max|x|$ ($r$ may be negative) and runs $D = 16$ levels below it, so the test is scale free and integer lattices behave as at 1.23.0.
- **[Signed-Portfolio-Origin]** `Portfolio.update` places a signed grid at $\min(\sum_k x_{\min,k}, \min_k x_{\min,k})$, not at `port_best_window`'s $x_{lo}$. Measured at 1.23.0: 1000 signed fixed-count units on `[-1.75 .25 .5]` realize an origin of $-1750$ at width 512 and hold 17.7% of the mass, a pre-existing bug in exactly PMIR's shape. Ruled: step 2 measures the `log2` need from a conservative estimate of that realized origin, $\lfloor \sum_k x_{\min,k} / b \rfloor b$ from the phase 1 unit rows, not from $x_{lo}$. The non-lattice signed bug is left for a separate plan.

### Divergences

Step 1, `[Severity-Lattice-Fractional]`, 1.24.0:

- **Search start** ([Lattice-Search-Start], ruled above). `_dyadic_lattice_step` searches $r$ from $-\lfloor \log_2 M \rfloor$ to $\lfloor D - \log_2 M \rfloor$, $M = \max|x|$, so `Homer2003.HC` keeps `bs` 200000 at `log2` 7.
- **Two private helpers not named in the plan.** `Aggregate._severity_atoms()` pools the components' `support_atoms` (or `None`), shared by `_severity_lattice`, the sizer (which needs the residual as well as the step) and the Portfolio pool. `_bucket_window._pooled_lattice(aggs)` is the pooled fit across units, used by `_sharpen_lattice` now and by `port_best_window` at step 2. Neither name occurred in `src/aggregate` (checked with `rg`).
- **Sharpen wording.** `sharpen_explanation`'s "bs is finer than it needs to be" fired whenever `bs` was below the lattice step, which would now argue against the rule's own choice (1 on a lattice of 3). It fires only when `bs` is below the nice divisor, and names that bucket. The pinned `bs = 1` on a lattice of 5 still gets it.
- **The lattice note goes on every row sized against the lattice** (`moment`, `exact_discrete`, `bounded_small`, `windowed`), not only the winner: `; lattice 0.75 (residual 0), bs 1/4 on it`, `; pinned bs 1/2 misses the lattice 0.75`, or `; lattice 1/4 too fine for the log2 cap, bs ... off it`. The `exact_discrete` row's own note reads `dfreq/fixed x dsev on a lattice` (was `... integer lattice`).
- **"bs coarsened" is a divisibility test**, `h / bs` a positive integer, so a nice divisor is exact and a pinned `bs` finer than the step is not called coarsened.
- **Integer atoms with a step deeper than $2^{-16} M$** (for example `dsev [0 1 100000]`) now have no lattice and no `exact_discrete` row; at 1.23.0 they had both, with `bs` coarsened off the lattice anyway. No corpus program is of this shape.
- **Exhibit snapshots** (`tests/data/exhibit_snapshots.json`) regenerated: 44 keys move, the `bs_window` notes on every fixture and the `bs` 1 to 10 move of the reinsurance fixtures (`dsev [10 20 30]` under `dfreq`), which reformats their downstream numbers. `economic_tail/raw/Tower` also moved in the last place on recapture, and a fresh capture at 1.23.0 moves it identically, so it is environment noise; its old entry was kept.
- **Corpus diff at step 1** (329 programs built, 12 move, all in [Lattice-Numbers-Move]): `Wang2002.HDD` (half-point atoms) $2^{-5}$, 16 to 0.5, 12; `Klugman2019.KPW.9.15.w` and `.cp` (step 25, random count) 25, 6 to 5, 8; `Klugman2019.ee.1` 15 to 5 and `.ee.2` 16 to 8 (Bernoulli); `Frees2018a.Revised` (step 6, mixed) 6, 10 to 2, 12; `Frees2018a.lda.5.3.2` (`dfreq`, step 100) 1, 11 to 100, 5; `DiceThreeEvenDice` (`dfreq`, step 2) 1, 6 to 2, 5. Four Portfolios move through their units' resolution floor before step 2 touches `port_best_window`: `Klugman2019.KPW.9.15.p` 50, 6 to 5, 10 (at 50 the 75 and 150 atoms of unit B were off the grid), `Exam.CASq15` 1, 11 to 5, 9, `BodoffTwo` 1, 13 to 50, 7, `BodoffThree` 1, 13 to 5, 11. Decimal and third atom sets size exactly as at 1.23.0.
- **Finding, not fixed: layers are not in the pool.** `support_atoms` are the unlayered atoms, so a layered `dsev` sizes on the wrong lattice: `Frees2018a.Revised` (`inf xs 10 sev dhistogram xps [6 12 18 30]`) sat at `bs` 6 at 1.23.0 with its layered values 2, 8 and 20 off the grid, and lands on 2 now only because 2 divides 10. Pooling attachments and limits into the atoms would fix it; it is the same at 1.23.0 for integer atoms and is outside this plan.
- **TODO entry.** The plan says a `[Fractional-Lattice-Bucket]` entry "is added and ticked at landing", but `dev/TODO.md` holds open items only and deletes a finished one. So 1.24.0 adds the entry with steps 2 and 3 remaining, plus two new open items from this step, `[Signed-Portfolio-Origin]` and `[Layered-Dsev-Lattice]`, and the final step deletes the plan's entry.
- **Gates at 1.24.0**: tier 3, `uv run pytest -m 'slow or not slow'`, 7082 passed; numerics, the same with `-W error::RuntimeWarning`, 7082 passed.

Step 2, `[Portfolio-Lattice-Bucket]`, 1.25.0:

- **Signed need from the realized origin** ([Signed-Portfolio-Origin], ruled above). `_port_lattice_bucket` estimates the origin `update` will realize as $\sum_k \min(\lfloor s_k / b \rfloor b, 0)$, $s_k$ each unit's phase 1 window low edge, which is at or below the realized $\min(\sum_k x_k, \min_k x_k)$. The need is the larger of the span from $\min(\text{estimate}, x_{lo})$ to $x_{hi}$, so never below the plan's wrap-safe window, and each unit's own window at $b$. Measured: 1000 signed fixed-count units on `[-1.75 .25 .5]` get `bs` 0.25 at `log2` 13 and hold all their mass with the exact mean $-187.5$ (1.24.0: 17.7% of the mass). The non-lattice signed case is untouched and still has the bug.
- **Names not in the plan.** `_bucket_window._port_lattice_bucket(...)` holds the rule for a Portfolio; `Portfolio._bs_lattice` (`dict(h, residual, bs)` or `None`, set by `port_best_window`) feeds a `bs_explanation` sentence and the `used` row note. Neither name occurred in `src/aggregate`.
- **The many-units regression needs many units.** With a few units the skewness widens the moment window past the summed support, so sizing from $x_{lo}$ happens to hold (20 skewed units: window from $-356$ against an origin of $-200$). `test_many_signed_units_keep_their_mass` uses 200 units (window about $[-136, 45]$, origin $-350$; from $x_{lo}$ the grid would be $[-350, -94)$) and takes about 8 s, so it is tagged `slow` and runs at tier 3.
- **Corpus diff against step 1**: one program moves, `DiscretePortfolio` (signed integer units), `bs` 1 at `log2` 16 to `bs` 1 at `log2` 6, mass 1 and mean exact.
- **Exhibit snapshots**: the four Portfolio `bs_window` keys gain the lattice note on the `used` row; nothing else moves (the Tower noise entry kept as at step 1).
- **Gates at 1.25.0**: tier 3 went 7088 passed and 1 failed, `test_decl_words.py::test_module_entry_point_prints_the_list_as_json` (`'1.25.0' == '1.24.0'`), because `__version__` was bumped while the run was in flight and the test's subprocess imported the new value. The numerics gate, the same selection with `-W error::RuntimeWarning`, started after the bump and passed 7089; `tests/test_decl_words.py` rerun alone passed 260. Lesson: bump after the gate, not during it.

Step 3, `[Bivariate-Dyadic-Lattice]`, 1.26.0:

- **`_lattice_bs(xs)` kept as the axis step**, now `_dyadic_lattice_step`'s `h` or `None`; `_float_gcd` is deleted and its two tests rewritten (`test_lattice_bs` gains the decimal cases, `test_float_gcd_is_the_one_implementation` becomes `test_netceded_lattice_is_the_dyadic_rule`).
- **The axis fit moved into `_size_axes`** (new keyword `lattice_axes`), because the measured width it fits against is only known there; `update` no longer pins the axis `bs` to the step before sizing. An axis with a pinned `log2` fits its candidates against that `log2` rather than the free-split cap.
- **Placement by axis.** `BivariateAggregate._on_lattice` (new, per axis) is true when the axis `bs` divides its step. Both true: today's exact index placement. Either false: `scatter_bivariate` (`scatter_bivariate_sparse` massive) with the off-lattice unit's `dsev_bucket` scheme, the lattice axis rounded to its exact index first so it does not split, offset by `i0`. A pinned `bs` that misses an axis lattice is scattered too, where 1.25.0 rounded it to the nearest index. New private helper `bivariate._divides(bs, h)`.
- **Netceded depth against the power-of-two factor of the gross `bs`.** The plan's $\lceil \log_2(\max|x| / \text{gross bs}) \rceil$ assumes a power-of-two gross `bs`; since 1.24.0 one can be 0.75, whose lattice needs the 0.25 level, so the depth is measured to the largest power of two dividing the gross `bs`. `_dyadic_lattice_step` now also stops at $2^{53}$ above the largest atom, where $x 2^r$ stops rounding exactly (a gross `bs` of `1e-6` in `test_exact_bs_early_exit_matches_the_full_gcd` would otherwise ask for about 77 levels). The netceded pinned-`bs` `exact` flag tests every image and the gross grid against the pinned bucket, since the chosen bucket can now be a nice divisor of the step rather than the step.
- **Acceptance, measured.** Axes on `[0 1 2]` and `[0.25 0.75]` keep `bs` 1 and 0.25; `[0 0.1 0.2]`, `[0.1 0.2 0.3]` and `[-0.3 0 0.2]` get a power-of-two `bs` with the margins of `S` equal to `g0` and `g1` to `1e-14`; `[0 0.1 0.3000001]` builds in 0.2 s (1.25.0: not finished after 90 s). The nine `MV.DBV*` programs in `decl-testers.agg` and the 18 `bvagg` programs in the corpora (six discrete) give the same `bs`, axis lengths, mass and mean as 1.23.0, and so do the 46 `pnl`, `xpnl` and `nttagg` programs. No exhibit snapshot moves.
- **First gate at 1.26.0 was red**: 7099 passed, 1 failed in both runs, `test_decl_unparser.py::test_roundtrip[FLB.BVQuarter]`. The tester `dbvsev [0.25 0.75] [0 1 2]` has no matrix, so the default is uniform 1/6, which the unparser writes as `0.16666666666666669` and reads back as `0.16666666666666666`: not idempotent. A 3 by 3 default (1/9) round-trips, as `MV.DBVUniform` does, so the tester and its test case became `[0 0.25 0.75]` (axis `bs` still 1/4). The 1/6 non-idempotence is pre-existing unparser behavior, not touched here.
- **Correction to 1.25.0.** Its CHANGELOG entry credited it with Poisson units on steps 3 and 5 going from `bs` 4 to 1, and a signed quarter-point book from $2^{-10}$ to 0.25. Measured at 1.24.0 (`3a6f18a`), both moves had already happened at step 1 through the units' resolution floor (1 at `log2` 9, and 0.25 at `log2` 16). What 1.25.0 changed for them is the signed book's `log2` 16 to 8. A case only step 2 changes is units whose nice buckets do not divide each other: steps 2 and 5 got `bs` 2 at 1.24.0, with the atoms at 5 and 15 off the grid, and get 1 at 1.25.0. The 1.25.0 entry is corrected in the 1.26.0 commit, the test docstring fixed, and `test_portfolio_units_whose_buckets_do_not_divide` and the tester `FLB.Steps25` added.
- **Usage doc** at `hacks/using/lattice-buckets.qmd`, not the plan's `using-lattice-buckets.qmd` with `using_common`: the folder was reorganized during the session (`<feature>.qmd`, `style.py`, `_stamp.yml`, house formats), and the doc follows the current pattern. Every block was run once in order in `.venv`; not rendered.
- **Gates at 1.26.0**, after the tester fix: tier 3 7102 passed; numerics, `-W error::RuntimeWarning`, 7102 passed. The version was bumped after both finished.
