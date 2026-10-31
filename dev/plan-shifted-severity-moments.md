# Plan: layer moments without far-tail cancellation, and shifted severities on the closed form

`[Shifted-Severity-Moments]`. Drafted 2026-10-31 against aggregate 1.43.1 (`03aaec6`). Two
minor bumps: `[Analytic-Layer-Tail-Cancellation]` (already on `dev/TODO.md`, lines 72 to 77),
then `[Shift-Removed-Moments]`. Agreed by the author on 2026-10-30
(`D:\Projects\Using\bugs\aggregate-review-answers-1.43.qmd`, item 12: "your method is right";
the far-tail fix goes first). Versions are assigned at execution; the plan runs second, after
`[Lattice-Grid-Sizing]`.

## Goal

A severity written `c * law + b` has the moments of that law, exactly, with the shift applied
after: `100 * gamma 2 + 10000` has skewness $\sqrt 2$ and `100 * gamma 2 + 1000000` has the same
skewness, not 1.387 and `nan`. A layer `y xs a` on any of the six closed-form families has its
moments right wherever it attaches: `10 xs 1000` on `lognorm 1` has a second moment below 100,
the largest possible, and a positive third moment. The verdict then stops failing correct
discretizations of shifted and far-tail-layered severities (13 of 30 shifted programs and 3 of 7
shifted layers today).

## The method (author, 2026-10-30)

Let $X$ be the law before the shift, $b$ the shift, and a layer $y$ xs $a$ on $X + b$, with
$a' = a - b$. If $a' \ge 0$ the layer is $y$ xs $a'$ on $X$, which the closed form handles. If
$a' < 0$, let $d = -a'$: the layer is the constant $y$ when $y \le d$, and otherwise $d + W$ with
$W = \min(y - d, X)$, so

$$
\mathsf{P}[(d + W)^n] = \sum_{k=0}^{n} \binom{n}{k} d^{\,n-k}\, \mathsf{P}[W^k].
$$

No layer is $a = 0$, $y = \infty$. The closed forms stay in use; the shift never enters a
partial expectation. A family with no closed form applies the same reduction on its unshifted
quantile, and a law with support below zero integrates that quantile over the range where
$X + b > a$.

## Current behavior

All anchors are `src/aggregate/_severity.py` unless stated.

- **A shifted law never reaches the closed form.** The gate `_uses_closed_form_moments`
  (2708 to 2738) requires `sev_loc == 0` (or the Lomax-as-shifted-Pareto case), no reflection,
  no splice. Everything else goes to `_numerical_moms` (1524 to 1655), quadrature of `isf`,
  `isf**2`, `isf**3` over `[sf(detachment), sf(attachment)]` through `_safe_integrate` at
  `epsrel` 1e-6 for the first moment and 1e-4 for the others (1497), accepted at a relative
  error of 1e-3 (1571), then expanded about the attachment. The raw moments are about 0 with
  the shift inside the integrand, so the central moments are differences of large numbers:
  measured, `100 * gamma 2 + 10000` gives variance 20,004.83 against 20,000 and skewness
  1.3870610; `+ 1000000` gives variance -72,596.8, skewness `nan`, and the log line
  `weird var < 0` from `moments.py:399`.
- **The partial expectations read the shift as part of the mean.** `_partial_e` (202 to 332)
  takes `fz.stats('m')` as the lognormal's, exponential's and gamma's scale (261 to 291), which
  includes `loc`; Pareto, Lomax and beta decline to quadrature when `loc` is not 0 (293 to 332).
  So the closed form cannot simply be opened to shifted laws: it has to be given the unshifted
  law.
- **The far-tail cancellation.** `_moms_analytic` (335 to 394) differences lower partial
  expectations and expands about 0: `sum((-1)**(m-k) * binom(m,k) * attachment**(m-k) *
  (pe_detach[k] - pe_attach[k]))` at 385 to 389. A thin layer far up the tail subtracts two
  nearly equal numbers of size $a^m$. Measured:

  | Layer | Path | $m_1$ | $m_2$ | $m_3$ |
  |:--|:--|--:|--:|--:|
  | `10 xs 1000` on `lognorm 1`, scale 1 | closed form | 9.6409 | 111.211 | -14,784.1 |
  | same | quadrature at 1e-13 | 9.65674 | 95.4381 | 948.779 |
  | `10 xs 1e5` on `100 * lognorm 1` | closed form | 8.2844 | 171,385.8 | -1.7136e10 |
  | same | quadrature at 1e-13 | 9.99648 | 99.9530 | 999.472 |

  The same expansion is in `_moms_retention` (1253 to 1307, lines 1302 and 1304), so a far-tail
  franchise or disappearing deductible cancels the same way (`100 xs 1000 to 1100 disappearing`
  on `lognorm 1`: third moment -6.48e6 against 953,304). `_lomax` expands about $-\lambda$ with
  alternating signs (174, 197 to 198) and its docstring admits the loss of digits.
- **Two other consumers of `_partial_e` bypass the gate.** `_layered_lev` (415 to 438) calls
  `_moms_analytic` directly for `size_biased_cdf` (441) and `feasible_bucket` (520), and
  `feasible_bucket` (581 to 589) checks neither `sev_loc` nor a splice. On
  `Severity('gamma', sev_a=2, sev_scale=100, sev_loc=10000)` it returns the reading of an
  unshifted gamma with scale 5,100, `_layered_lev(s, 10000)` is 13,189.6 against 10,000, and
  `size_biased_cdf(s, 10000)` is 0.3127 against 0. The consumer is `severity_feasibility`
  (`_bucket_window.py` 650, calls at 720 to 736), stored as `_bs_feasibility`, which drives
  `InfeasibleGridWarning` and the verdict's `INFEASIBLE` reading.
- **A side defect.** `_moms_analytic` with `limit == 0` returns 3 entries, not 4 (359 to 360),
  and `moms()` raises `ValueError: not enough values to unpack` on
  `exp_attachment=10, exp_limit=0`.
- **The signed path is the control.** `ssev` uses scipy's `fz.moment` with the shift and is
  exact at a shift of 1e4; at 1e6 the conversion of raw to central moments in `_stats` (2601
  to 2606) loses six digits (skewness 1.414304). That loss, about $\epsilon (b/\sigma)^3$, is
  the limit of any method that carries raw moments, on the grid as in the severity.

## Steps

### `[Analytic-Layer-Tail-Cancellation]`

**Change.** `_partial_e` gains a sibling `_partial_e_upper(sev_name, fz, a, n)` returning
$\mathsf{P}[X^k; X > a]$ for $k = 0..n$ in closed form for the six kinds: lognormal and gamma
through the complementary regularized functions (`erfc` and `gammaincc` in log form), the
exponential directly, beta through `betaincc` (scipy 1.11 and later; the version floor is
checked at execution), Pareto Type I from its tail and Lomax from the same series with the
tail factored out. `_moms_analytic` differences the upper partial expectations and expands about
$a$ rather than 0 when $F(a) > 1/2$:

$$
\mathsf{P}[(X-a)^m; a < X \le a+y] = \sum_k \binom{m}{k} (-a)^{m-k}\bigl(U_k(a) - U_k(a+y)\bigr),
$$

with the same mass term `limit**m * sf(detachment)`. Below $F(a) = 1/2$ the lower form stays,
so nothing moves on ordinary layers. `_moms_retention` takes the same switch on its two
differences. `_layered_lev` follows automatically. The `limit == 0` return becomes `n + 1`
zeros.

**Files.** `src/aggregate/_severity.py`; `tests/test_partial_e_pareto.py`,
`tests/test_partial_e_beta_lomax.py` (the survival reference `_layer_reference` at 41 is the
oracle), a new `tests/test_layer_tail_moments.py`.

**Acceptance.** The four rows of the table above agree with the quadrature column to 1e-8
relative; `10 xs 1e5` on `100 * lognorm 1` builds with the verdict `not unreasonable`. Every
layer in the two `test_partial_e` files keeps its value to 1e-12 (the switch is inactive
there). The six kinds at ten attachments from $F(a) = 0.5$ to $S(a) = 10^{-12}$ agree with the
survival reference to 1e-8. `moms()` on `exp_limit=0` returns zeros. `test_moms_never_reaches_quadrature`
still passes.

### `[Shift-Removed-Moments]`

**Change.**

1. `Severity` keeps an unshifted frozen law, `fz0`, beside `fz` whenever `sev_loc != 0`: the same
   family and shapes with `loc = 0`. For a reflected law the base is already built at 0 (2870 to
   2871) and nothing changes.
2. `_uses_closed_form_moments` drops the `sev_loc == 0` condition. `moms()` (2682 to 2702) calls
   a new `_moms_shifted(fz0, limit, attachment, shift, n)` that applies the method: $a' \ge 0$
   is `_moms_analytic(fz0, y, a', n)`; $a' < 0$ with $y \le d$ is the constant; otherwise the
   binomial expansion over `_moms_analytic(fz0, y - d, 0, k)`. The conditional divisor stays
   `pattach`, now `fz0.sf(a')`.
3. `_numerical_moms` takes the same reduction: integrate the unshifted `isf` and expand, so a
   shifted family without a closed form (`weibull_min`, `invgamma`, `gumbel_r`, `norm`) also
   loses the shift from its integrand. A law with support below zero (the normal) integrates
   over the range where $X + b > a$, as the method requires.
4. `_retention_moms` passes `fz0` and the shifted attachment and retention end through the same
   reduction.
5. `feasible_bucket` declines a law with `sev_loc != 0` (adds it to the checks at 581 to 589)
   until `_layered_lev` is given `fz0` and the reduction, which it is in this step; the
   measurement above (13,189.6 against 10,000) becomes a test.
6. Out of scope and unchanged, each named in the docstring of `moms`: splices (`_apply_lb_ub`,
   2161 to 2201, quadrature on the spliced law), reflected clamped laws (`_apply_reflect`),
   signed `ssev` (`_apply_signed`, scipy's own moments), `cantor` (its own `_build`), `SeverityElt`
   and `SeverityCopy`.

**Files.** `src/aggregate/_severity.py`; `tests/test_shifted_severity_moments.py` (new);
`tests/test_feasibility.py` (the decline).

**Acceptance.** The ten severities of the review's table (`100 * gamma 2 + 10000`,
`+ 100000`, `+ 1000000`, `100 * lognorm 1 + 10000`, `100 * weibull_min 2 + 10000`,
`+ 100000`, `100 * beta 2 3 + 10000`, `100 * invgamma 5 + 10000`, `100 * gumbel_r + 10000`,
`100 * norm + 10000`) have skewness within 1e-6 of the exact values 1.41421, 1.41421,
1.41421 (within 1e-4 at a shift of 1e6), 6.18488, 0.63111, 0.63111, 0.28571, 3.46410,
1.13955, 0; the seven shifted layers of the review agree with quadrature at 1e-13 to 1e-8 on
the third moment; `30 * norm + 100` keeps `sev_m` 100.0034 (the clamp, not this plan). Of the
review's 30 shifted programs, 26 or more read `not unreasonable`; the four at a shift of
1,000,000 are the rounding limit and are listed in the log. `bs` and `log2` do not move on the
six sample programs of the review. Tier 3 with the numerics gate.

## Not in this plan

- Carrying central moments through the library to beat the $\epsilon (b/\sigma)^3$ limit at
  shifts near $10^4$ times the scale. The four failures at 1,000,000 stay, no longer hidden by
  `nan`.
- `[Feasibility-Lomax-Beta]` (TODO 89 to 93), which keeps those kinds out of
  `_FEASIBILITY_KINDS`.
- The verdict's skewness threshold and `nan` handling, in `[Validation-Verdict]`.

## Execution log

(filled at execution: versions, the measured tables after each bump, divergences)
