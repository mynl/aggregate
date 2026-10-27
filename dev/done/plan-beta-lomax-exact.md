# Plan: closed-form layered moments for beta and Lomax severities

Label `[Beta-Lomax-Exact-Moments]`. Drafted 2026-10-27 at 1.16.0. Executed
2026-10-27 as 1.17.0; see the execution log at the end.

## Goal

`Severity.moms()` computes layered moments in closed form for `lognorm`, `gamma`,
`pareto` and `expon`, and by quadrature for everything else. Add `beta` and `lomax`
to the closed-form list.

- **beta** matters because of the ELT work: ELT rows produce badly skewed betas
  (tiny `a`, huge `b`), and quadrature returns nan or crashes on them.
- **lomax** is the shifted Pareto, `lomax(α, scale=λ)` = `pareto(α, loc=-λ,
  scale=λ)`. Pareto is exact, so Lomax is exact. The reason is consistency;
  today's quadrature is already accurate for it.

## Current behavior

All in `src/aggregate/_severity.py`.

- `moms()` path 2 sends the four kinds to `_moms_analytic`, which differences the
  partial expectations `_partial_e` returns and expands `(x - a)^m` binomially.
  Everything else goes to `_numerical_moms`, quadrature of `fz.isf` over
  probability. The kind list is written out four times: the guard in
  `_partial_e`, `_FEASIBILITY_KINDS`, `moms()` path 2, `_retention_moms()`.
- **beta is numerical.** Measured at 1.16.0, scale 100 unless stated, reference
  40-digit mpmath:

  | shapes `a b` | layer | today | closed form |
  |---|---|---|---|
  | 2.625 2.625 | none | 1.1e-7 on `m2`, 17 ms | 1e-14 |
  | 0.005 50 | none | 1.9e-6 on `m2`, 38 ms | 1e-14 |
  | 1e-4 1e4, scale 1e8 | none | 3e-8, 88 ms | 1e-14 |
  | 1e-5 1, scale 1e8 | none | **nan**, all three | 1e-14 |
  | 1e-6 10, scale 1e8 | none | **ZeroDivisionError** in `_numerical_moms` | 1e-14 |
  | 0.02 20 | 50 xs 1 | 1.8e-5 on `m3`, 27 ms | 1e-14 |

  A closed-form partial expectation costs about 6 µs.
- **lomax is numerical**, about 1e-12 accurate, 6 to 45 ms a severity. It is used
  across `library.agg` (12 lines) and `decl-testers.agg` (6).
- `_partial_e` already has a shifted-Pareto branch (`loc == -scale`), but
  `moms()` never reaches it: the path-2 gate requires `sev_loc == 0`, so
  `sev 120 * pareto 2.5 - 120` is numerical too. Only `feasible_bucket` reaches
  it. The branch divides by `i - α` and returns **nan at integer shapes**:
  `ss.pareto(2, scale=120, loc=-120)` gives nan for `k = 2, 3` at a finite limit.
- `_partial_e_pareto_type_1` forms $\alpha\lambda^\alpha(a^d - \lambda^d)/d$, and
  $\lambda^\alpha$ overflows once $\alpha\log_{10}\lambda > 308$ (λ = 1e8, α = 60).

## Mathematics

Write $\mathsf P[X^k; X \le x] = \int_0^x u^k f(u)\,du$, the partial expectation
`_partial_e` returns for `k = 0..n`.

**Beta.** scipy `beta(p, q, loc=0, scale=s)` is $X = sT$, $T \sim
\mathrm{Beta}(p, q)$. Since $t^k f_{p,q}(t) = r_k f_{p+k,q}(t)$ with

$$r_k = \frac{B(p+k, q)}{B(p, q)} = \prod_{j=0}^{k-1}\frac{p+j}{p+q+j},$$

$$\mathsf P[X^k; X \le x] = s^k\, r_k\, I_{\min(x/s,\,1)}(p+k,\, q),$$

with $I$ the regularized incomplete beta, `scipy.special.betainc`. Valid for all
$k \ge 0$ and all $p, q > 0$: the same shift-the-shape-and-read-the-cdf device the
gamma branch uses. A limit and attachment need nothing new, because
`_moms_analytic` already turns partial expectations into layer moments.

**Lomax.** scipy `lomax(α, loc=0, scale=λ)` is $X = Y - \lambda$, with $Y$
Type-I Pareto on $[\lambda, \infty)$. Expanding $x^k$ about $-\lambda$,

$$\mathsf P[X^k; X \le x] = \sum_{i=0}^{k}\binom{k}{i}(-\lambda)^{k-i}\,
\mathsf P[Y^i; Y \le \lambda + x],$$

and the inner terms are `_partial_e_pareto_type_1(α, λ, λ + x, k)`, which already
handles the removable case $i = \alpha$. That fixes the integer-shape nan. At
$x = \infty$, if any inner term up to $k$ is infinite the answer for $k$ is
`inf` (the moment does not exist); returning it directly avoids `inf - inf` and
its `RuntimeWarning`.

**Type-I overflow.** $\alpha\lambda^\alpha(a^d - \lambda^d)/d =
\alpha\lambda^k\,\mathrm{expm1}(d\log(a/\lambda))/d$ with $d = k - \alpha$. Same
value, no $\lambda^\alpha$.

A scratch prototype of both forms matched mpmath to 1.1e-14 (beta, every shape
in the table, `x` from 1e-3 to past the support) and to about 1e-9 (Lomax layer
moments, shapes 0.737 to 60, unlimited and ordinary layers). The Lomax sum
alternates, so a single partial moment at $x \ll \lambda$ loses about
$(\lambda/x)^k$ ulps (5e-6 relative on $k = 3$ at $x = \lambda/1000$). That
washes out of the layer moments, where the $x^k S(x)$ term dominates, but it sets
where the tests below may check partial moments directly.

## Changes

**[analytic-kinds-single-source]** Define `_ANALYTIC_KINDS = ('lognorm',
'gamma', 'pareto', 'expon', 'lomax', 'beta')` beside `_FEASIBILITY_KINDS`, and
use it in the `_partial_e` guard, `moms()` path 2 and `_retention_moms()`.
`_FEASIBILITY_KINDS` keeps the original four (see the decision below), and its
comment gives the reason.

**[beta-closed-form]** A `beta` branch in `_partial_e`: shapes from
`fz.args[:2]`, `lo, hi = fz.support()`; when `lo != 0`, fall back to
`_partial_e_numeric`, as the pareto branch does for an unexpected `loc`. Call
`scipy.special.betainc` directly; a frozen `ss.beta(...).cdf` costs about half a
millisecond to construct.

**[lomax-closed-form]**
- New `_partial_e_lomax(alpha, lam, a, n)` per the mathematics above. `_partial_e`
  sends `lomax` (`lam = fz.kwds.get('scale', 1.0)`; a nonzero `loc` falls back
  to numeric) and `pareto` with `loc == -scale` to it. Delete the inline binomial
  loop.
- Rewrite the `k != alpha` line of `_partial_e_pareto_type_1` with `expm1`.
- Relax the `moms()` path-2 gate (and the matching one in `_retention_moms()`) to
  admit `pareto` with `sev_loc == -sev_scale`, so `sev 120 * pareto 2.5 - 120` and
  `sev 120 * lomax 2.5` take the same path. Checked at 1.16.0: that DecL builds
  with `sev_scale = 120`, `sev_loc = -120`, `sev_lb = 0`, `sev_ub = inf`, so the
  other clauses of the gate already pass.

**Docstrings**: `_partial_e` (list the six, drop "beta?" from its to-do line),
`moms()` Notes path 2, `_retention_moms()` Notes ("the four kinds"),
`_FEASIBILITY_KINDS` comment, `feasible_bucket` Notes ("the four `_partial_e`
handles" becomes the four of `_FEASIBILITY_KINDS`). The kind list in
`docs/2_aggregate_overview/bucket-selection.rst:244` is the feasibility list and
stays correct.

**[tests]** New `tests/test_partial_e_beta_lomax.py`:

1. Beta partial expectations match `_partial_e_numeric` on benign shapes (rel
   1e-8), match `fz.moment(k)` at `x = inf` on every shape in the table (rel
   1e-12), and `k = 0` is the cdf.
2. `Severity('beta', sev_a=1e-6, sev_b=10, sev_scale=1e8).moms()` and the `1e-5 1`
   case are finite and equal `fz.moment(k)`.
3. Beta `50 xs 1` on `0.02 20`, scale 100, conditional and not, matches an
   independent reference to 1e-10: `quad` of $m(x-a)^{m-1}S(x)$ over the layer
   (smooth, unlike the density near 0), divided by $S(a)$ when conditional.
4. Lomax partial moments match `_partial_e_numeric` at $x \in \{\lambda,
   10\lambda, 100\lambda\}$ (rel 1e-8; not below $\lambda$, see the note above),
   and `ss.lomax(...).moment(k)` at $x = \infty$. Shapes 1, 2 and 3 with
   `1000 xs 0` and `750 xs 250` give finite layer moments that match the
   survival-function reference of item 3. Unlimited `k >= α` is `inf`. All under
   `warnings.simplefilter('error', RuntimeWarning)`.
5. `lomax(α, scale=λ)` and `pareto(α, loc=-λ, scale=λ)` give identical
   `_partial_e`, and `agg T 1 claim sev 120 * lomax 2.5 fixed` and the same with
   `sev 120 * pareto 2.5 - 120` give identical `moms()`, with and without a
   `750 xs 250` layer (written before `sev` in DecL).
6. Dispatch: `moms()` for beta, lomax and shifted pareto, layered and not, never
   calls `_numerical_moms` (monkeypatch it to raise).
7. `feasible_bucket` still returns `None` for lomax and beta, pinning the decision.

In `tests/test_partial_e_pareto.py`, update the docstring of
`test_shifted_pareto_still_uses_its_own_closed_form`, since the branch is now
shared with Lomax; the assertion is unchanged.

**[release]** One minor bump, through the version-bump skill: `__version__`, a
one-paragraph CHANGELOG entry, the `[Analytic-Layer-Tail-Cancellation]` TODO
entry (below), and this plan moved to `dev/done/`.

## Acceptance

- The new tests pass. Tier 1 while iterating, tier 2 before calling it done,
  tier 3 and the numerics gate (`-m 'slow or not slow' -W
  error::RuntimeWarning`) at the bump.
- Existing tests that pin beta or lomax numbers may move:
  `tests/test_baseline.py` at rtol 1e-12 (its corpus has `sev 500 * beta 5 2`),
  and beta cases in `test_bucket_sizing.py`, `test_approximate.py` and
  `test_chart_severity.py`. Every divergence must be a beta, lomax or shifted
  pareto theoretical moment, or a number computed from one (a chosen `bs`
  included), moving toward the closed form. Anything else is a finding. For the
  baseline, re-capture with `tests/baseline/capture.py` once the divergences are
  accounted for.
- No new warnings on the `library.agg` lomax and beta lines, since feasibility is
  unchanged.
- Unchanged: a conditional beta layer attaching above the support (`pattach = 0`)
  returns nan with a `RuntimeWarning` today, on either path. It stays as is.

## Known limitation, shared with the existing kinds

`_moms_analytic` differences lower partial expectations,
$\mathsf P[X^k; X \le d] - \mathsf P[X^k; X \le a]$, and then expands about 0. For a thin layer far in
the tail both steps cancel catastrophically. That is already true of the
existing kinds: lognormal σ = 1, scale 100, `10 xs 1e5` has its third moment off
by 1.7e7 relative. Lomax and beta inherit it. Lomax 2.5, scale 120, `10 xs 1e6`
($S \approx 10^{-10}$ at the attachment) goes from 1e-5 by quadrature to wrong,
while quadrature already returns nan for lomax 20 `750 xs 250`. Layers attaching
where $S$ is above about 1e-6 are good to about 1e-8. ELT betas usually carry no
layer. TODO entry **[Analytic-Layer-Tail-Cancellation]**: difference upper
partial expectations (survival function, `betaincc`, the Type-I tail) once
$F(a) > 1/2$, for all six kinds.

## Feasibility reading: deferred

By the same consistency argument, `lomax` belongs in `_FEASIBILITY_KINDS` beside
`pareto`. Doing it changes warnings: an unlimited lomax needs `log2_required` 17.0
at shape 2.5, 24.8 at 1.8 and 35.2 at 1.5, so every unlimited lomax in
`library.agg` with shape up to about 2.5 starts raising `InfeasibleGridWarning`
at `log2 = 16`. The warning is true, but it moves a dozen library lines and their
tests. Author ruling 2026-10-27: deferred to TODO entry
**[Feasibility-Lomax-Beta]**, already in `dev/TODO.md`.

## Out of scope

Weibull, Burr, invgamma and invgauss closed forms; the tail cancellation above;
the feasibility reading.

## Execution log

2026-10-27, one bump, 1.17.0. All seven changes landed as specified, except the divergences below.

**Divergences.**

- **[analytic-gate-helper]** The plan relaxed the path-2 gate in `moms()` and the matching one in `_retention_moms()` in place. Both now call one private method, `Severity._uses_closed_form_moments()`, which holds the `_ANALYTIC_KINDS` test, the reflect, splice and shift clauses, and the shifted-Pareto admission, so the seven-clause condition is written once, the way the kind list now is. No collision: `rg _uses_closed_form` found nothing before the change.
- **[tests] item 5.** `_partial_e` for `lomax(α, scale=λ)` and `pareto(α, loc=-λ, scale=λ)` is identical, bit for bit. The DecL `moms()` are identical unlimited, but in a `750 xs 250` layer they agree only to 1 ulp (`48447552.24239163` against `...162`), because `_moms_analytic` adds `limit^m S(d)` from each law's own scipy `sf`. The test asserts equality or `rel=1e-12`.
- **[tests] item 6.** Besides `_numerical_moms`, the dispatch test also monkeypatches `_partial_e_numeric` to raise, which catches a silent fallback inside `_partial_e` too, and asserts `_uses_closed_form_moments()`. Layer used: `50 xs 10`.
- **[tests] item 1.** Benign shapes `(2.625, 2.625)`, `(2, 5)`, `(5, 2)`, `(3, 1.5)` at scale 100 against `_partial_e_numeric` at `x = 30, 60, 90`. Past the support (`x = 2 * scale`) is checked against `fz.moment(k)` in the unlimited test, for every table shape. The cdf check runs at `x / scale = 1e-3, 0.1, 0.5` on every table shape, at rel 1e-12.
- **[lomax-closed-form]** `_partial_e_lomax` returns zeros for `a <= 0`, mirroring the `a <= lam` guard of the Type-I helper. `_partial_e` already returns zeros at `a == 0`, so this only matters to a direct caller.
- **Docstrings.** `_partial_e` was rewritten in NumPy style rather than edited, and its Notes carry the beta derivation; the to-do line now reads Weibull, Burr, invgamma, invgauss.
- **`dev/TODO.md`.** The author's uncommitted `[Feasibility-Lomax-Beta]` entry, which this plan cites as already present, ships in this commit with the new `[Analytic-Layer-Tail-Cancellation]` entry.

**Acceptance measurements.**

- `tests/test_baseline.py` moved on one case only, `Port.Bounded` (`sev 500 * beta 5 2`, `sev 800 * beta 7 2`), in `stats_df` and `describe`. Diffing the full `stats_df` between a HEAD snapshot and the change, along the harness path (`build(..., update=False)` then `update(log2=16, bs=1/8, padding=0)`): every moved cell is a theoretical moment in the A, B and total columns, an `error` row computed from one, or `('meta', 'el')` in the `empirical` column, which is the theoretical expected loss. The densities are bit-identical. Every move is toward the closed form: the sev mean error falls from `-4.7e-11` to `8.9e-16`, and the theoretical A moments now equal `fz.moment(k)`, against `357.14285714688947` (mean) and `52083331.875` (third moment) by quadrature. Re-captured with `tests/baseline/capture.py`; only the two `Port.Bounded` parquet files and the manifest (version and commit) changed.
- `test_bucket_sizing.py`, `test_approximate.py`, `test_chart_severity.py`: no movement.
- The 16 `library.agg` statements using `lomax` or `beta` build with identical results and identical warning sets at HEAD and after the change.
- Beta on the plan's table: `moms()` equals `fz.moment(k)` to within 3.3e-16 relative on all five unlayered shapes, including `1e-5 1` (nan before) and `1e-6 10` (`ZeroDivisionError` before).

**Gates.** Tier 1 on the new and affected files while iterating; tier 2 (`uv run pytest`) 6645 passed; the numerics gate (`uv run pytest -m 'slow or not slow' -W error::RuntimeWarning`), the tier 3 selection under the stricter filter, 6852 passed.
