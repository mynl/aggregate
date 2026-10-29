---
bibliography: D:/Projects/Biblio/uber-library.bib
csl: D:/Projects/Biblio/journal-of-risk-and-uncertainty.csl
---

# Plan: distributions computed by inverting a characteristic function

`[Characteristic-Function-Engine]`. Drafted 2026-10-28 against aggregate 1.18.2;
revised three times on 2026-10-29 against 1.26.0 with the author's rulings folded
in. **Status: final, awaiting the author's go-ahead.** Every design question is
ruled; there are no open questions. Line references are for 1.26.0. It absorbs
`[Power-Variance-Family]` (`dev/TODO.md`, "Grammar, parser and unparser"). It
builds on `[Tweedie-Statement]` and `[Stable-Statement]`
(`dev/done/plan-tweedie-statement.md`, `dev/done/plan-stable-statement.md`),
which supply `TweedieConversion`, the `conversion` answer kind and `Stable.chf`,
and it follows the pattern `nttagg` set for a new declaration kind
(`dev/done/plan-nttagg.md`, `dev/done/plan-nttagg-composability.md`).

**Size.** A large plan: roughly 3,200 new lines including docstrings and tests,
over eight library steps, each a minor bump and each useful on its own. Nothing
on an existing code path gets slower; the only edits to existing paths are the
severity reference hooks (step `[Characteristic-Function-Severity]`), one hint
key, and the FourierTools shim (step `[Characteristic-Function-Pedagogy]`).
Execution may stop after any step.

## Goal

A declaration and object kind, `chfagg`, that computes a distribution on a grid
by sampling its characteristic function and inverting it with one FFT, instead
of discretizing a severity and applying a frequency generating function. Heavy
tails are computed by inverting along a shifted contour (exponential tilting),
chosen automatically; a stable law with two heavy tails is split into its two
one-sided parts, each tilted toward its own light side. The families are infinitely
divisible laws with closed-form characteristic functions: the $\alpha$-stable
laws, the whole power variance (Tweedie) family, and the normal, gamma, inverse
Gaussian, normal inverse Gaussian, variance gamma and tempered stable laws. The
purposes, in the author's order:

1. **Teaching.** Illustrate the direct characteristic function method, and show
   one law computed two ways: `agg X tweedie 1.5 100 2` by the compound engine
   beside `chfagg X tweedie 1.5 100 2`, and a Pareto book beside its stable
   limit.
2. **Accurate stable distributions**, replacing scipy where it fails: in the far
   tail and in speed.
3. **The PMIR emergence application**, which needs an accurate stable `cdf`,
   including skewed laws with two heavy tails.
4. **`[Power-Variance-Family]`**: Tweedie for every $p$, including the members
   with no frequency and severity form.

## Rulings

- Author, 2026-10-28: the engine is wanted, after the Tweedie and stable
  statements (both landed: 1.22.0, 1.23.0).
- Author, 2026-10-29, first round:
  1. The keyword is **`chfagg`**, mirroring `nttagg`. No reinsurance, no limits.
  2. No special cases, and a natural presence in the Loss Lab (aLL,
     `aggregate-api`). An `Aggregate` with its moment machinery switched off is
     therefore out.
  3. FourierTools folds into the new engine, and pedagogy calls the engine.
  4. Portfolio units and reinsurance are out of scope. Distortion pricing is in
     scope, since it needs only a grid distribution.
  5. A `chfagg` is usable as a severity, as `sev agg.NAME` is today.
  6. Families: comprehensive from the start, partly for teaching; certainly
     normal, gamma, normal inverse Gaussian and variance gamma. No compound
     Poisson family.
- Author, 2026-10-29, second round:
  1. **Window: rule A** (log2 is the budget, the window follows
     `bucket_sizing_p`) by default; **rule C**: when the user sets `bs` and
     `log2`, they are used. A grid of $2^{34}$ is never an option.
  2. **Tilting is required**, and automated: a `tilt=True|False|float` keyword,
     default `True`. Design below.
  3. **Class name `AggregateCHF`.** It is not a subclass of `Aggregate`; the
     docstring says so. This departs knowingly from the `Base<Kind>` reading
     of the house naming rule, in favor of the `AggregateNTT` parallel.
  4. **Long family names** (`norminvgauss`, `variancegamma`, `temperedstable`).
  5. **Keep a forwarding shim** for `aggregate.ft` for now, with an
     explanation in the module and in the warning.
- Author, 2026-10-29, third round:
  1. **Bucket cap** for tilted laws agreed: the bucket is capped and the window
     ends short, with the mass beyond reported, rather than coarsening the body
     to reach `bucket_sizing_p`.
  2. **Two-sided tilting is in version one** (wanted for PMIR): a stable law
     with $|\beta|<1$ is split into its two totally skewed parts, each tilted
     toward its light side. Design and measurements below.
  3. **Shim removal has no date**: when the three outside documents have moved
     off `FourierTools`, or at the next major release.

## Why a new engine

Aggregate's engine discretizes a severity, takes its FFT, applies the count's
probability generating function, and inverts. It needs a finite expected count
and a severity. A Tweedie with $1<p<2$ has both, which is why the `tweedie`
clause expands into an ordinary aggregate and the engine never changes. A stable
law with $\alpha<2$ is the distribution of a Lévy process with infinite Lévy
measure: there is no expected count and no severity, so nothing to expand into.
The same holds for the Tweedie members with $p<0$ and $p>2$, which are
exponentially tilted, totally skewed stable laws [@Jorgensen1997], and for the
normal inverse Gaussian, variance gamma and tempered stable laws. What all of
these have is a closed-form characteristic function, and sampling it on the
FFT's frequency grid replaces the first two steps of the compound engine.

**Why "agg" is honest.** Every family in scope is infinitely divisible: it is
the time-one law of a Lévy process, and therefore a limit of compound Poisson
aggregates. The keyword describes what the laws are, not how they are computed.

## Current behavior, 1.26.0

- `FourierTools` (`src/aggregate/ft.py:340`) inverts a characteristic function
  with `irfft`: `invert(log2, x_min, bs, x_max, s)` (`:458`) samples the Fourier
  transform and returns **density values** on the grid, not bucket masses;
  `invert_simpson` (`:517`), `compute_exact` (`:578`) and the plots (`:622` on,
  bodies in `src/aggregate/plots/_fourier.py`) compare with a scipy object.
  `make_levy_chf` (`:761`) is the standard S1 stable characteristic function.
  The same module holds compound engine figure helpers unrelated to
  characteristic function inversion: `poisson_example` (`:28`),
  `fft_wrapping_illustration` (`:66`), `recentering_convolution_example`
  (`:168`), `recentering_convolution` (`:265`) and their helper `stats`
  (`:323`).
- `FourierTools` consumers: `Tweedie.ft` (`src/aggregate/tweedie.py:837`),
  `Tweedie.audit` (`:853`) and `Tweedie.test` (`:1059`); the docs page
  `docs/3_reference/3_x_Auxiliary.rst:88`; and, outside this repository,
  `D:/projects/aggregate/monograph/posts/3_user_guides/3_x_10mins.qmd`,
  `D:/projects/aggregate/monograph/posts/2_technical_guides/_2_x_nm_ft_conv_algo.qmd`
  and `D:/projects/aggregate/notes/math/resources/pedagogy.py`.
  `src/aggregate/cantor.py:620` mentions it in a docstring.
- `aggregate.ft` is **outside the 1.0 stability promise**: the stable tier in
  the CHANGELOG preamble lists `Aggregate`, `Portfolio`, `PnL`, `Severity`,
  `Frequency`, `Distortion`, `BivariateAggregate`, `Underwriter`, `build`, `qd`
  and `decl.lark`, and nothing else.
- `Tweedie.chf` (`tweedie.py:954`) is `exp(K(1j * t))`; `kappa` (`:909`) and `K`
  (`:946`) give the cumulant functions for every $p$.
- **`Tweedie.fz` is wrong for the stable members** (`tweedie.py:718`, measured
  2026-10-28). For $2<p<\infty$ with $\theta<0$, the usual case since it gives a
  finite mean, the code constructs a `NotImplementedError` without raising it
  and `fz` returns `None`: `Tweedie(2.5, 1, 1).fz is None`, and likewise $p=4$.
  For $p<0$ it returns the untilted `ss.levy_stable(alpha, -1, ...)` whatever
  $\theta$ is: `Tweedie(-1, 1, 1).fz.mean()` is 0, not the declared mean 1. The
  Cauchy branch ($p=\infty$) prints `set up Cauchy`. For $1<p<2$ it returns a
  built `Aggregate`, so `fz` is already "a distribution object", not strictly a
  scipy frozen law.
- `Stable` (`src/aggregate/stable.py:48`) gives `chf` (`:216`), `mean`,
  `tail_constant` (`:281`) and `left_tail_constant` in closed form, and `pdf`,
  `cdf`, `sf`, `ppf`, `isf` from scipy's `levy_stable` (`fz`, `:292`), which is
  unreliable in the far tail and slow (class Notes). `Stable.chf` is written
  with `abs` and `sign`, so it is correct on the real line only.
- `GridDistribution` (`src/aggregate/_grid_distribution.py:222`) is an
  immutable value type over `(x, p, bs)`. It provides `q`, `tvar`,
  `tvar_threshold`, `cdf`, `sf`, `pmf`, `mean`, `lev`, `tvar_of_limited`,
  `prob_loss_assets`, `pdf` and `snap` (these two need `bs`), `cap`,
  `return_period`, `to_series`, `info` and `help`. It has no variance or
  skewness, no plot, no `density_df`, no DecL program, no `update`, and is not
  a first-class class. Its accessors work off the realized cumulative, so a
  grid whose mass is below 1 is handled: `sf` is `1 - cdf` and so includes
  the missing mass. `Aggregate` holds one (`_grid_distribution`,
  `src/aggregate/_aggregate.py:7651`) and delegates `q`, `tvar`, `cdf` and the
  rest to it.
- **First-class contract.** `FIRST_CLASS_CLASSES` and `FCC_REQUIRED`
  (`src/aggregate/constants.py:145`, `:159`) require `info`, `help`, `note`,
  `hints`, `tags`, `program`, `pprogram`, `summary_df`, `validation_df`,
  `stats_df`, `density_df` and `plot`; `tests/test_fcc_surface.py` checks
  every listed class. `PnL` (`src/aggregate/_pnl.py:1040`) and
  `BivariateAggregate` (`src/aggregate/bivariate.py:1217`) are first-class
  peers that do not subclass `Aggregate`; they mix in `HelpMixin`,
  `LabeledMixin` and `ProgramMixin`.
- **How aLL decides what to show** (surveyed 2026-10-29 in
  `V:/dev/aggregate-api`). There is no per-kind table: `capability_for`
  (`src/aggregate_api/capability.py:787`) builds the navigation from the
  library's exhibit and chart registries (`singledispatch`, so a registered
  class gets the leaves) and from `hasattr` on the object: `price_pentagon`
  lights pricing, `pnl_program` and `reins_program` light P&L and reinsurance,
  `with_hints` lights hints, `sharpen` lights sharpening. The build strip
  reads `bs`, `log2`, `est_m`, `est_cv`, `validation_description`, `note`,
  `tags`; the frame routes read `summary_df`, `tail_df`, `validation_df`,
  `stats_df`, `density_df`. Admitting `nttagg` took four api touchpoints
  (`_KIND_OF_CLASS`, `SUPPORTED_KINDS`, `_KIND_ORDER`, the `BuildResponse.kind`
  literal) plus front-end labels; everything else rode on the library.
- **Pricing is duck-typed.** `calibrate_distortions` (`src/aggregate/_pricing.py:740`)
  reads `density_df['p_total']`, `bs`, `q`, `cdf`, `snap`, `_is_loss_value` and
  `name`; `price_pentagon` (`:149`) reads `q` and `bs`; `_canonical_grid`
  (`:479`) slides a windowed or signed grid onto the canonical loss axis;
  `guard_unbounded_anchor` (`:29`) reads `bounded`.
- **Severity references.** `sev agg.NAME` parses to a symbolic `sev_ref`
  (`src/aggregate/parser.py:2316` on; grammar `sev_ref_leaf`,
  `src/aggregate/decl.lark:594`, token `BUILTIN_AGG`, `:1023`).
  `Underwriter._resolve_sev_ref` (`src/aggregate/underwriter.py:2105`) requires
  the referenced entry to carry `hints{log2=...; bs=...}`, rebuilds it, calls
  `update`, and converts it with `_dhistogram_from_object`
  (`src/aggregate/_severity.py:664`), which accepts only `Aggregate` and
  `Portfolio` (an `isinstance` switch). `tail.output_support_max`
  (`src/aggregate/tail.py:1039`) supplies the reference's theoretical upper
  support. `_safe_lookup` (`underwriter.py:2733`) checks that a dotted
  reference's kind matches its prefix, so `agg.NAME` cannot name an `nttagg`
  today.
- **Hints.** `_HINT_KEYS` (`underwriter.py:318`) is the allow-list of update
  knobs a `hints{}` clause may set; `_resolve_hints` (`:456`) passes the extra
  keys to `update` as keywords, and `Aggregate.update` absorbs unknown ones in
  `**kwargs`. Hint values coerce to int, float, fraction, `True`/`False` or a
  string.
- Window and bucket sizing for the compound engine
  (`src/aggregate/_bucket_window.py`) is moment based, with
  `BUCKET_SIZING_P = 0.99999` (`:77`, from `config.py`) as the probability the
  window covers, and `utilities.round_bucket` (`src/aggregate/utilities.py:236`)
  rounding the bucket **up** to the next rung of a nice ladder.
  `InfiniteVarianceError` (`constants.py:474`) refuses an infinite variance.
- **Tilting precedent.** The compound engine does not tilt, but
  `_renewal.renewal_count_pmf` (`src/aggregate/_renewal.py`) does, with
  `tilt_total`, the tilt times the window length (default 20), "circular wrap
  is damped by `exp(-r * tilt_total)`". The `tilt` keyword below uses the same
  dimensionless convention.

## Mathematics

**Bucket masses, not density samples.** Take the grid $x_k = x_0 + kh$,
$k = 0,\dots,n-1$, window $W = nh$, and frequencies $t_j = 2\pi j/W$. Let $U$ be
uniform on $[-h/2, h/2]$, independent of $X$. Then
$\Pr(X\in[x_k-h/2, x_k+h/2)) = h f_{X+U}(x_k)$, and
$\varphi_{X+U}(t) = \varphi_X(t)\,\mathrm{sinc}(th/2)$ with
$\mathrm{sinc}(u) = \sin(u)/u$. Inverting $\varphi_X\cdot\mathrm{sinc}$ therefore
returns the bucket probabilities of the "round" discretization aggregate already
uses, rather than density values. The masses are exact for a lattice law whose
lattice contains the grid too, so one kernel serves continuous and atomic laws.
Because they are the round discretization's masses, a `chfagg` and an `agg` on
the same grid agree bucket for bucket up to the two errors below, which is both
the teaching comparison and the strongest test oracle.

Two errors remain. **Frequency truncation**: frequencies beyond $\pi/h$ are
dropped, and the error is of the order of $|\varphi(\pi/h)|$. For a stable law
$|\varphi(t)| = e^{-(\gamma|t|)^\alpha}$, negligible once $h \ll \gamma$ when
$\alpha$ is not small. For a gamma law $|\varphi(t)| = (1+\theta^2t^2)^{-k/2}$
decays only polynomially, so a shape below 1 (a pole at zero) rings near zero;
the same holds for variance gamma with a large $\nu$. **Aliasing**: the result
is the law wrapped onto the window, so all mass outside the window lands inside
it.

**Aliasing is the binding constraint for heavy tails.** The wrapped mass is about
$(1+\beta)C_\alpha\gamma^\alpha(x_0+W)^{-\alpha}$ from the right tail, plus the left
tail. Without tilting, holding it to $10^{-6}$ with $h=\gamma/50$ needs about
$2^{18}$ buckets at $\alpha=1.5$, $2^{23}$ at $\alpha=1.1$, and $2^{34}$ at
$\alpha=0.7$ (computed 2026-10-28, $\beta=1$). Worse, the wrapped mass lands
where it does the most relative damage: the mass at $x+W$ folds onto $x$, and
near the right edge of the window $p(x+W)/p(x) \approx 2^{-\alpha-1}$, so the
untilted relative error in the right half of the window is of order one
whatever the total wrapped mass (measured below).

**Measured accuracy, untilted** (2026-10-28, $\alpha=1.5$, $\beta=1$, S1, unit
scale, $x_0=-20$; references are scipy `sf` in the body and the tail asymptotic
at 1000):

| log2 | $h$ | $W$ | time | sf(10) rel. error | sf(100) rel. error | sf(1000) rel. error |
|---|---|---|---|---|---|---|
| 16 | 0.05 | 3,277 | 3 ms | $8\times10^{-6}$ | $-4\times10^{-4}$ | $-8\%$ |
| 18 | 0.02 | 5,243 | 13 ms | $9\times10^{-7}$ | $-1\times10^{-4}$ | $-3\%$ |
| 20 | 0.01 | 10,486 | 52 ms | $3\times10^{-7}$ | $-2\times10^{-5}$ | $-0.5\%$ |

**Window edges from the tails.** Every family in scope has, on each side, one
of two kinds of tail, and each gives a closed-form edge for a target mass
$\varepsilon$ beyond it on that side.

- *Light side* (a cumulant generating function $K(s) = \log\mathsf{P}e^{sX}$
  finite for $0<s<s_+$). Chernoff: $\Pr(X>x) \le \exp(K(s) - sx)$ for each such
  $s$, so the right edge is $x_R = \inf_{0<s<s_+} (K(s) - \log\varepsilon)/s$, a
  one-dimensional minimization; the left edge uses $K(-s)$. The bound is
  conservative, never optimistic. Normal, gamma, inverse Gaussian, normal
  inverse Gaussian, variance gamma and tempered stable laws are light on both
  sides; every Tweedie with a finite mean is a natural exponential family with
  $\theta$ interior to its domain, so its $K$ is finite near 0 and it is light
  on both sides too.
- *Power side* (stable, $\alpha<2$): $\Pr(X>x) \sim A x^{-\alpha}$ with
  $A = (1+\beta)C_\alpha\gamma^\alpha$ (`Stable.tail_constant`), so
  $x_R = \delta_1 + (A/\varepsilon)^{1/\alpha}$, an asymptote rather than a bound.
  The other side of a totally skewed stable ($\beta=1$; $\beta=-1$ by
  reflection) is light, through the Laplace transform
  $\Lambda(s) = \log\mathsf{P}e^{-sX} = -\gamma^\alpha\sec(\pi\alpha/2)s^\alpha - \delta_1 s$
  for $\alpha\ne1$ and $(2\gamma/\pi)s\log s - \delta_1 s$ for $\alpha=1$,
  $s\ge0$; for $\alpha<1$ the support is bounded below at $\delta_1$.

**Tilting is inversion along a shifted contour.** For $\beta=1$ the
characteristic function $\varphi(s) = \mathsf{P}e^{isX}$ is analytic on the upper
half plane $\mathrm{Im}\,s>0$, where $|e^{isX}| = e^{-\mathrm{Im}(s)X}$ and the left
tail is light. On the real line the bucket mass is
$p_k = (1/2\pi)\int \varphi(t)\,h\,\mathrm{sinc}(th/2)\,e^{-itx_k}\,dt$; the
integrand is analytic in the strip $0\le\mathrm{Im}\,s\le\theta$, so by Cauchy's
theorem the line can be moved to $\mathrm{Im}\,s=\theta$, where
$e^{-isx_k} = e^{-itx_k}e^{\theta x_k}$. Hence

$$
e^{-\theta x_k}p_k = \frac{1}{2\pi}\int \varphi(t+i\theta)\,h\,\mathrm{sinc}\big((t+i\theta)h/2\big)\,e^{-itx_k}\,dt ,
$$

a sequence that decays like the tilted law $e^{-\theta x}f(x)$, exponentially on
the right. Discretizing that integral wraps this damped sequence, not the
heavy tail itself; multiplying the result by
$e^{\theta x_k}$ recovers the untilted masses. No Laplace transform value is
needed, the sinc factor is the same function at the complex frequency, and
$\theta=0$ is the untilted kernel, so there is one kernel:

```python
t = 2 * np.pi * np.arange(n // 2 + 1) / (n * h)
s = -t + 1j * theta                      # irfft sums with e^{+}, hence -t
c = phi(s) * csinc(s * h / 2) * np.exp(1j * t * x0)   # csinc(z) = sin(z) / z, csinc(0) = 1
x = x0 + h * np.arange(n)
p = irfft(c, n) * np.exp(theta * x)
```

`phi` must be the analytic continuation, not `Stable.chf`. For $\beta=1$, S1:
$\log\varphi(s) = -\gamma^\alpha\sec(\pi\alpha/2)(-is)^\alpha + i\delta_1 s$
($\alpha\ne1$) and $(2\gamma/\pi)(-is)\log(-is) + i\delta_1 s$ ($\alpha=1$),
principal branches; both reduce to the S1 form on the real line and to
$\Lambda(\theta)$ at $s=i\theta$.

The masses beyond the window are no longer folded in: they are **left out** and
reported. For $\beta=1$ they all lie above the window, so the grid's total is
$1 - \Pr(X > x_0+W)$, the deficit is the closed-form tail mass, and
`sf = 1 - cdf` stays correct across the window. Mass left out *below* the window
($\beta=-1$, and every two-sided law) is handled by the convention under
`[Characteristic-Function-Object]`.

**The cost of tilting, and the automatic choice.** Write the tilt as the
dimensionless $\tau = \theta W$, the number of e-folds of damping across the
window (the convention `_renewal` uses). Two errors set it, both measured
relative to the bucket mass $p(x_R)$ at the right edge, where they are worst:

- *wrap*, damped: $\rho\,e^{-\tau}$, where
  $\rho = \sum_{m\ge1} p(x_R+mW)/p(x_R) = q^{\alpha+1}\zeta(\alpha+1, 1+q)$,
  $q = (x_R-\delta_1)/W$, from the power tail ($\zeta$ the Hurwitz zeta,
  `scipy.special.zeta(s, q)`);
- *amplification*: the FFT's absolute error, roundoff plus frequency
  truncation, is spread evenly over the tilted masses and multiplied by
  $e^{\theta x_k}$, so relative to $p(x_R)$ it is
  $\varepsilon_{\mathrm{eff}}\,\kappa\,e^{\tau}$, where
  $\varepsilon_{\mathrm{eff}} = 2^{-52} + |\varphi(\pi/h)|$ and
  $\kappa = \bar f\,(x_R-\delta_1)^{\alpha+1}/(A\alpha)$ is the dynamic range of
  the masses across the window, with the density bound
  $\bar f = (1/2\pi)\int|\varphi| = \Gamma(1+1/\alpha)/(\pi\gamma)$.

The sum $\rho e^{-\tau} + \varepsilon_{\mathrm{eff}}\kappa e^{\tau}$ is least at

$$
\tau^* = \max\Big(0, \tfrac12\log\frac{\rho}{\varepsilon_{\mathrm{eff}}\kappa}\Big),
\qquad E^* = 2\sqrt{\rho\,\varepsilon_{\mathrm{eff}}\,\kappa},
$$

all in closed form, no extra FFT. $E^*$ is the reported edge error. Because
$\kappa\ge1$ and $\rho$ is of order one, $\tau^*$ cannot exceed about
$\frac12\log 2^{52}\approx18$. Tilting replaces a relative error of order one
by one of order $\sqrt{\varepsilon\kappa}$: it halves the number of digits the
window's dynamic range costs.

**Two consequences for the grid.** Because truncation error is amplified by
$e^{\tau}$, a tilted grid needs $|\varphi(\pi/h)|$ at roundoff level:
$h \le h_{\mathrm{trunc}} = \pi\gamma/(52\log 2)^{1/\alpha}$, which is $0.29\gamma$
at $\alpha=1.5$, $0.087\gamma$ at 1, $0.019\gamma$ at 0.7 and $0.0024\gamma$ at
0.5. And because a tilted grid no longer has to reach far into the tail to
avoid wrapping, it can keep a fine body: $h \le \gamma/10$. Where rule A's
bucket would exceed $h_{\max} = \min(\gamma/10, h_{\mathrm{trunc}})$, the bucket is
capped and the window ends at $x_0+nh$; the mass beyond is reported, not
wrapped. Left edge: the support bound for $\alpha<1$; otherwise the Chernoff
edge for $10^{-40}$, because mass below the window folds onto the top of it and
is amplified by $e^{\tau}$ (the super-exponential left tail makes this cost a
few $\gamma$).

**Measured accuracy, tilted against untilted on the same grid** (2026-10-29,
prototype of the kernel above, $\beta=1$, S1, unit scale, rule A with
`bucket_sizing_p` 0.99999, the cap, and $\tau^*$; references: the untilted
kernel on a window 256 times longer for $\alpha\ge1$, the exact convergent
series
$\Pr(X>x) = -(1/\pi)\sum_{k\ge1}(-1)^k\Gamma(k\alpha)\sin(k\pi\alpha)(x/c)^{-k\alpha}/k!$,
$c^\alpha = \gamma^\alpha\sec(\pi\alpha/2)$, for $\alpha<1$; under 10 ms each at
log2 16):

| $\alpha$ | log2 | $h$ | $W$ | $\tau^*$ | $E^*$ | rel. error at $W/2$, tilted / untilted | at right edge, tilted / untilted | mass beyond window |
|---|---|---|---|---|---|---|---|---|
| 1.5 | 16 | 0.028 | 1,864 | 8.4 | $1.5\times10^{-4}$ | $1.3\times10^{-5}$ / $0.10$ | $4.2\times10^{-5}$ / $0.34$ | $5.0\times10^{-6}$ |
| 1.1 | 16 | 0.1 | 6,554 | 8.9 | $1.5\times10^{-4}$ | $1.1\times10^{-5}$ / $0.20$ | $2.1\times10^{-5}$ / $0.56$ | $3.8\times10^{-5}$ |
| 1.0 | 16 | 0.087 | 5,712 | 9.5 | $9.7\times10^{-5}$ | $2.1\times10^{-6}$ / $0.23$ | $7.0\times10^{-6}$ / $0.64$ | $1.1\times10^{-4}$ |
| 0.7 | 16 | 0.019 | 1,229 | 12.1 | $1.1\times10^{-5}$ | $8.3\times10^{-7}$ / $0.42$ | $3.9\times10^{-6}$ / $1.0$ | $5.1\times10^{-3}$ |
| 0.7 | 20 | 0.019 | 19,670 | 9.8 | $1.2\times10^{-4}$ | $8.8\times10^{-6}$ / $0.42$ | $1.7\times10^{-5}$ / $1.1$ | $7.3\times10^{-4}$ |
| 0.5 | 20 | 0.0024 | 2,536 | 12.2 | $1.7\times10^{-5}$ | $1.0\times10^{-6}$ / $0.69$ | $3.0\times10^{-6}$ / $1.6$ | $1.6\times10^{-2}$ |

Every tilted grid had all masses positive, and its window total equaled one
less the tail mass beyond it to $10^{-7}$ (rechecked 2026-10-29, third round),
**provided** the tail mass for $\alpha<1$ comes from the exact series above
(agreement within $3\times10^{-8}$; the leading term $Ax^{-\alpha}$ alone is off
by $3\times10^{-5}$ at $\alpha=0.7$, log2 16). For $\alpha\ge1$ the leading term
agrees to $10^{-7}$ at these windows. `outside_mass` uses exactly this rule:
the series for $\alpha<1$, the leading term for $\alpha\ge1$, on each side
(every one-sided half of a two-sided law is $\beta=\pm1$, so the series applies
there too). $E^*$ is conservative by a factor of
3 to 10. The remaining limit is coverage, not accuracy: for $\alpha$ near 0.5 the
truncation cap keeps the window short, and the mass beyond it is large but
known.

**Two heavy tails: split, tilt each half, convolve.** No single contour serves
a law with two heavy tails: $\mathsf{P}e^{\theta X}$ is infinite for every
$\theta\ne0$, so $\varphi$ has no continuation off the real line. The S1 stable
law splits exactly instead. With $w_\pm = (1\pm\beta)/2$ and
$\gamma_\pm = \gamma\,w_\pm^{1/\alpha}$,

$$
X = X_+ + X_-,\qquad X_+\sim S(\alpha, 1, \gamma_+, \delta_1; 1),\qquad X_-\sim S(\alpha, -1, \gamma_-, 0; 1),
$$

independent, for every $\alpha$ including 1, because $\log\varphi$ is linear in
$\beta$ with coefficient proportional to $\gamma^\alpha$ (the S1 form at
$\alpha=1$ has no $\gamma\log\gamma$ term). $X_+$ has one heavy tail, on the
right, and is tilted as above; $X_-$ is computed as the reflection of
$S(\alpha, 1, \gamma_-, 0; 1)$, so it is tilted toward the left. The halves are
combined by a linear convolution, so nothing wraps. Three details make it
accurate, each found by prototype on 2026-10-29:

- *The bucket factor is applied once.* The convolution of two sets of bucket
  masses gives the masses of $X+U_1+U_2$, a second, spurious rounding with
  relative error of order $(h/\gamma)^2/24$. So $X_+$ is inverted with the
  $\mathrm{sinc}$ factor and $X_-$ without, and the product of the two
  transforms is $\varphi\,\mathrm{sinc}$ exactly on the band.
- *Resolution transfer.* For $\beta$ near $\pm1$ the light half is narrow: at
  $\alpha=1.5$, $\beta=0.99$, $\gamma_-=0.03\gamma$, under one bucket. Its own
  truncation $|\varphi_-(\pi/h)|$ is then of order one, and the tilt amplifies
  it; the plain split failed outright there (relative errors of $10^8$). The
  fix moves a normal law from the wide half to the narrow one, which leaves the
  product unchanged:

  $$
  \varphi(s) = \big(\varphi_+(s)\,e^{\sigma^2s^2/2}\big)\big(\varphi_-(s)\,e^{-\sigma^2s^2/2}\big),
  \qquad \frac{\sigma^2\pi^2}{2h^2} = \max\big(0,\ 52\log2 - (\pi\gamma_-/h)^\alpha\big),
  $$

  the least $\sigma$ that brings the narrow half's truncation to $2^{-52}$. The
  wide half becomes a deconvolved, signed sequence, which is harmless as an
  intermediate. Since $|\varphi| = |\varphi_+||\varphi_-|$, both halves reach
  $2^{-52}$ whenever the whole law reaches $2^{-104}$ (and at $\beta=0$ only
  then), so the **two-sided
  bucket cap** is $h\le\min(\gamma/10,\ \pi\gamma/(104\log2)^{1/\alpha})$,
  smaller than the one-sided cap by $2^{1/\alpha}$. A law at $\beta=\pm1$ keeps
  the one-sided path and cap. Measured $\sigma/h$ ran from 0 (both halves
  resolved on their own) to 2.7.
- *Each half on the full grid, then crop.* Mass of one half beyond its own
  window re-enters the window through the other half's heavy tail, so halves of
  $n/2$ buckets each lose accuracy near both edges (measured: $10^{-2}$ to
  $10^{-1}$ in the outer 1% of the window at $\alpha=1$, $\beta=0.5$, log2 14).
  Each half is therefore inverted on all $n$ buckets,
  the linear convolution has $2n$, and the $n$ kept are placed so that the mass
  left out below equals the mass left out above (one cumulative sum and an
  argmin; this is rule A's "split equally between the tails"). The left-out
  masses are then known: the $2n$ result's mass outside the kept window plus
  each half's closed-form tail beyond its own window. Placing the window from
  the closed-form tails instead, balancing $A_-d_-^{-\alpha}$ against
  $A_+d_+^{-\alpha}$, was tried and rejected: it ignores where the body sits,
  lands up to 6% of the grid off for skewed laws, and at $\beta=0.99$ broke
  the edge error bound ($3\times10^{-4}$ against $2\times10^{-4}$ at
  $\alpha=1$; $10^{-2}$ against $1.6\times10^{-3}$ at $\alpha=0.7$).

Each half's $\tau$ is the one-sided $\tau^*$ of that half: its tail constant
$A_\pm = (1\pm\beta)C_\alpha\gamma^\alpha$, its truncation, and its density
bound, which for the narrow half is
$\min\big(\Gamma(1+1/\alpha)/(\pi\gamma_-),\ 1/(\sigma\sqrt{2\pi})\big)$ once the
normal is added. The halves' own $E^*$ overstate the assembled law's error by up
to 300 times near $\beta=\pm1$, so the **reported edge error** of a two-sided law
is instead $\max(E_L, E_R, \text{mass left out})$, where $E_L$ and $E_R$ are the
one-sided formula applied to the whole law (its density bound
$\Gamma(1+1/\alpha)/(\pi\gamma)$) with that side's tail constant and the distance
from $\delta_1$ to that edge. For $\alpha<1$ the mass left out dominates: the
worst error is 0.4 to 1.4 times the mass left out on one side, which is the
re-entry effect above, at the kept window's edges. Cost: two inversions of $n$ and one convolution of
$2n$, about 12 ms at log2 16, against 5 ms for a one-sided law.

**Measured accuracy, two-sided** (2026-10-29, prototype, S1, unit scale,
$\delta_1=0$, log2 16, rule A with `bucket_sizing_p` 0.99999 and the two-sided
cap; reference: the untilted kernel on a window 256 times longer, centered on
the kept one; "worst" is the largest relative bucket error over the stated part
of the kept window; every grid had all masses positive):

| $\alpha$ | $\beta$ | $h$ | set by | $W$ | $\sigma/h$ | $\tau_+$, $\tau_-$ | worst, middle half | worst, whole window | reported edge error | untilted worst, same grid | left out, each side |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.5 | 0 | 0.036 | rule A | 2,335 | 0 | 7.6, 7.6 | $9.0\times10^{-6}$ | $3.3\times10^{-5}$ | $6.4\times10^{-5}$ | 1.2 | $3.2\times10^{-6}$ |
| 1.5 | 0.5 | 0.035 | rule A | 2,265 | 0 | 8.0, 7.0 | $1.3\times10^{-5}$ | $3.5\times10^{-5}$ | $7.9\times10^{-5}$ | 1.3 | $3.0\times10^{-6}$ |
| 1.5 | 0.9 | 0.031 | rule A | 2,043 | 0 | 8.3, 5.8 | $1.7\times10^{-5}$ | $4.1\times10^{-5}$ | $1.0\times10^{-4}$ | 1.5 | $1.8\times10^{-6}$ |
| 1.5 | 0.99 | 0.029 | rule A | 1,901 | 2.5 | 8.4, 3.9 | $2.2\times10^{-5}$ | $4.6\times10^{-5}$ | $1.2\times10^{-4}$ | 1.6 | $7.6\times10^{-7}$ |
| 1.1 | 0 | 0.064 | cap | 4,214 | 0 | 8.3, 8.3 | $5.7\times10^{-6}$ | $1.1\times10^{-5}$ | $7.0\times10^{-5}$ | 1.4 | $3.5\times10^{-5}$ |
| 1.1 | 0.5 | 0.064 | cap | 4,214 | 1.9 | 8.7, 7.7 | $6.9\times10^{-6}$ | $1.9\times10^{-5}$ | $6.1\times10^{-5}$ | 1.4 | $3.1\times10^{-5}$ |
| 1.1 | 0.99 | 0.064 | cap | 4,214 | 2.7 | 9.0, 5.3 | $1.2\times10^{-5}$ | $1.5\times10^{-5}$ | $8.5\times10^{-5}$ | 1.6 | $5.2\times10^{-6}$ |
| 1.0 | 0 | 0.044 | cap | 2,856 | 0 | 9.2, 9.2 | $2.5\times10^{-6}$ | $2.6\times10^{-5}$ | $2.2\times10^{-4}$ | 1.5 | $1.1\times10^{-4}$ |
| 1.0 | 0.5 | 0.044 | cap | 2,856 | 1.9 | 9.6, 8.5 | $5.1\times10^{-6}$ | $3.4\times10^{-5}$ | $1.9\times10^{-4}$ | 1.5 | $9.6\times10^{-5}$ |
| 1.0 | 0.99 | 0.044 | cap | 2,856 | 2.7 | 9.8, 6.0 | $3.9\times10^{-6}$ | $2.2\times10^{-5}$ | $4.3\times10^{-5}$ | 1.6 | $1.5\times10^{-5}$ |
| 0.7 | 0 | 0.007 | cap | 457 | 0 | 11.8, 11.8 | $2.4\times10^{-4}$ | $1.2\times10^{-3}$ | $6.2\times10^{-3}$ | 1.9 | $3.1\times10^{-3}$ |
| 0.7 | 0.5 | 0.007 | cap | 457 | 1.9 | 12.3, 10.9 | $3.3\times10^{-4}$ | $1.3\times10^{-3}$ | $5.3\times10^{-3}$ | 2.0 | $2.6\times10^{-3}$ |
| 0.7 | 0.99 | 0.007 | cap | 457 | 2.7 | 12.6, 8.0 | $1.2\times10^{-4}$ | $4.7\times10^{-4}$ | $6.6\times10^{-4}$ | 3.1 | $3.3\times10^{-4}$ |

The reported edge error bounded the worst error in every row. For $\alpha\ge1$
the error anywhere in the window is below $5\times10^{-5}$ for every $\beta$, as
good as the one-sided laws. For $\alpha=0.7$ the limit is coverage again: the
two-sided cap gives a window of $457\gamma$ at log2 16, the left-out mass is
$3\times10^{-3}$ a side, and the coverage row recommends a larger `log2`.

**Atoms.** For $1<p<2$ the Tweedie has an atom $e^{-\lambda}$ at zero, and
$\varphi(t)\to e^{-\lambda}$ as $t\to\infty$, so frequency truncation leaves
ringing near zero. Subtract the atom from $\varphi$, invert the continuous part,
and add the atom back to the bucket at zero. The grid is anchored so that 0 is
a grid point ($x_0$ a multiple of $h$), as on every aggregate grid.

## Design

### [Characteristic-Function-Object] the class `AggregateCHF`

**What a `GridDistribution` gives, and why it is not enough on its own.** A
`GridDistribution` gives every probability accessor (`q`, `tvar`, `cdf`, `sf`,
`pmf`, `mean`, `lev`, `prob_loss_assets`, `pdf`, `snap`, `cap`). It does not
give variance, skewness, a plot, a `density_df`, a DecL program, `update`, or
first-class status, and it is an immutable value type whose holder rebuilds it
when the grid changes. So `AggregateCHF` **holds** a `GridDistribution` and
delegates to it, exactly as `Aggregate` does, and is otherwise a first-class
peer like `PnL` and `BivariateAggregate`. It is **not a subclass of
`Aggregate`**, so no `Aggregate` code path needs a special case and no consumer
reading `frequency`, `sev` or the reinsurance attributes is ever handed one.
The class docstring opens its Notes with exactly that: the name parallels
`AggregateNTT` and the `chfagg` keyword, `isinstance(obj, Aggregate)` is
`False` by design, and the shared surface is by duck typing.

**Natural aLL presence comes from the contract, not from special cases.** The
api builds its navigation from registries and `hasattr`. The class therefore:

- meets `FCC_REQUIRED` and is added to `FIRST_CLASS_CLASSES`;
- registers the `summary`, `tail`, `validation` and `stats` exhibits and one
  chart emitter by `singledispatch`, as `PnL` does (`exhibits/_pnl.py`,
  `charts/_emit_pnl.py`);
- carries `price_pentagon` and `calibrate_distortions` (pricing lights up);
- does **not** carry `pnl_program`, `reins_program` or `sharpen`;
- carries `with_hints` from step `[Characteristic-Function-Severity]`.

The api work is then the four kind touchpoints `nttagg` needed and the
front-end labels; a separate plan in `aggregate-api`.

**Module.** `src/aggregate/characteristic_function.py`, reached by submodule
(`from aggregate.characteristic_function import AggregateCHF`), as `Tweedie`,
`Stable` and `AggregateNTT` are. Vetted 2026-10-29: `AggregateCHF`,
`characteristic_function`, `chfagg` and `tilt` (as an attribute or keyword)
occur nowhere in `src`, `tests` or `docs`; nor do `tilt_theta`,
`outside_below`, `outside_above`, `outside_mass`, `transfer_sigma`,
`edge_error`, `truncation_error`, `grid_window` or `recommended_grid` in
`src`.

**Surface.** Names follow `Aggregate` wherever the meaning is the same, so the
duck-typed pricing and chart code applies unchanged.

| member | meaning |
|---|---|
| `__init__(name, family, form=None, note='', tags=(), hints='', label=None, **parameters)` | the underwriter's `AggregateCHF(**spec)`; `parameters` as written in DecL |
| `from_family(name, family, **trailer)` | classmethod from a `Stable`, a `Tweedie`, or a family object |
| `from_chf(name, chf, *, x_min, bs, log2, **trailer)` | classmethod for a user-supplied characteristic function with an explicit grid (FourierTools' arbitrary `chf`; the Cantor law); never tilted |
| `family` | the family object (below) |
| `update(log2=16, bs=0, bucket_sizing_p=BUCKET_SIZING_P, tilt=True, **kwargs)` | compute the grid; other hint keys are accepted and ignored with a logged warning |
| `grid_window(log2=16, bucket_sizing_p=BUCKET_SIZING_P, tilt=True)` | `(bs, log2, x_min, tilt)` the rules would choose, without computing; `x_min` is `None` for a two-sided law, whose window is placed by the crop during the computation (hints carry only `log2` and `bs`, so nothing downstream needs it) |
| `recommended_grid(epsilon)` | the smallest `(log2, bs)` whose window reaches mass `epsilon` beyond it at the capped bucket; advice for `validation_df` and the hygiene message |
| `log2`, `bs`, `x_min`, `xs`, `agg_density` | the grid and the masses, named as on `Aggregate` |
| `tilt`, `tilt_theta` | the realized tilts, always a pair `(left, right)`: $\tau$ in e-folds across the window, and $\theta=\tau/W$. $(0, \tau)$ for $\beta=1$, $(\tau, 0)$ for $\beta=-1$, $(\tau_-, \tau_+)$ two-sided, $(0, 0)$ untilted. As with `bs`, the attribute holds the realized value of the `update` keyword of the same name |
| `outside_below`, `outside_above`, `outside_mass` | the law's mass below and above the window, and their sum: closed form, plus for a two-sided law the computed mass of the $2n$ convolution outside the kept window. Tilted grids leave it out; untilted grids wrap it in |
| `transfer_sigma` | $\sigma$ of the normal moved between the halves of a two-sided law; 0.0 otherwise |
| `edge_error`, `truncation_error` | the modeled worst relative bucket error ($E$ at the realized $\tau$, about $\rho$ untilted; two-sided, $\max(E_L, E_R, \text{outside\_mass})$); $\vert\varphi(\pi/h)\vert$ |
| `density_df` | columns `loss`, `p_total`, `F`, `S`, `lev`, `exa`, indexed by `loss`, as on `Aggregate` |
| `q`, `tvar`, `cdf(x, kind='previous')`, `sf`, `pdf`, `pmf`, `snap`, `var_dict`, `prob_loss_assets` | delegates to the held `GridDistribution`, same signatures as `Aggregate` |
| `agg_m`, `agg_cv`, `agg_skew` | theoretical moments from the cumulants; `inf` or `nan` where they do not exist |
| `est_m`, `est_cv`, `est_skew` | grid moments; for a law without the moment they are statistics of the window, and `validation_df` says so |
| `bounded`, `support` | `False` for every family in scope; `(lower, upper)` of the law |
| `info`, `help`, `note`, `hints`, `tags`, `label`, `program`, `pprogram` | the mixins, as on `PnL` |
| `summary_df`, `tail_df`, `stats_df`, `validation_df`, `validation_description` | as below |
| `plot()` | three panels: density (mass over `bs`), log density, and survival on log axes with the power or Chernoff tail drawn in; the density panels start at the second bucket when the first holds `outside_below` |
| `price_pentagon`, `calibrate_distortions` | thin wrappers over `_pricing`, as `Aggregate` has them |

`summary_df` and `tail_df` use the same columns and return-period ladder as
`Aggregate`'s. `stats_df` is one block, theoretical against grid moments.
`validation_df` rows: total mass against `1 - outside_above` (tilted; the mass
below is in the first bucket) or 1 (untilted); most negative bucket;
`outside_mass` against `1 - bucket_sizing_p`; `edge_error`; `truncation_error`;
buckets per body scale (`scale / bs`, flagged below 10, where `scale` is the
standard deviation when finite and $\gamma$ otherwise); the tilt pair, and
`transfer_sigma` for a two-sided law; each finite moment's relative error; and
"not finite: window statistic" for each infinite one. A flagged coverage row
carries ready-to-paste `hints{log2=...; bs=...}` from `recommended_grid`.

**Conventions.** Loss orientation (`_is_loss_value` True). Masses are the
round discretization's, so `cdf(x_k)` is the law's $F(x_k + h/2)$, exactly as
on `Aggregate`; a caller wanting the cdf at an arbitrary point to second order
(PMIR) interpolates the cumulative masses at the bucket edges $x_k + h/2$. The
grid is never renormalized: outside mass is reported, not hidden. **Mass left
out below the window is added to the first bucket** and mass left out above is
not added anywhere, so the cumulative masses, and with them `cdf`, `q` and
`sf = 1 - cdf`, are correct throughout the window for every tilt: this is what
lets a $\beta=-1$ or two-sided law use the `GridDistribution` accessors
unchanged. The first bucket then means "the left edge and everything below
it", as the zero bucket of a clamped severity already does; `lev`, `mean` and
the grid moments are window statistics that include it. `tvar` is the
window's, which for a tilted heavy tail omits the mass above; completing it from
the power tail is `[CHF-Tail-Completion]`.

### Window and bucket rule

The grid has three numbers, the left edge $x_0$, the bucket $h$ and the size
$n = 2^{\text{log2}}$.

- **Rule A, the default** (`bs=0`). `bucket_sizing_p` sets the mass
  $\varepsilon = 1-p$ allowed beyond the window, split equally between the
  tails; the edges come from the closed forms above; $h$ is
  `round_bucket`$((x_R-x_L)/n)$ and $x_0 = h\lfloor x_L/h\rfloor$.
- **Rule A, tilted law.** As A, then the bucket is capped at $h_{\max}$ (the
  largest rung of `round_bucket`'s ladder not above
  $\min(\text{scale}/10, h_{\mathrm{trunc}})$; a small private helper beside
  `round_bucket`), the window ends at $x_0+nh$, and $\tau = \tau^*$. The left
  edge is the support bound or the $10^{-40}$ Chernoff edge.
- **Rule A, two-sided law.** The edges come from each side's power tail at
  $\varepsilon/2$, $h$ is `round_bucket`$((x_R-x_L)/n)$ capped at the two-sided
  $h_{\max} = \min(\gamma/10,\ \pi\gamma/(104\log2)^{1/\alpha})$ by the same
  helper, each half is inverted on $n$ buckets from its own light edge, and the
  $n$ buckets kept from the $2n$ convolution are placed to balance
  `outside_below` against `outside_above`, so $x_0$ is an output of the
  computation rather than an input.
- **Rule C** (`bs` given, or `hints{log2=...; bs=...}`). Used as given; $x_0$
  from the left edge rule (two-sided: from the balanced crop), the window
  $n\cdot bs$ long, the tilt still automatic unless `tilt` says otherwise. A
  `bs` above the cap shows up in `truncation_error` and `edge_error`, since
  $\varepsilon_{\mathrm{eff}}$ includes the truncation term and $\tau^*$ shrinks
  accordingly; two-sided, it also forces a larger `transfer_sigma`.

When `outside_mass` exceeds `1 - bucket_sizing_p`, the coverage row is flagged
with the `recommended_grid` text; when that needs log2 above 24 it says so
plainly rather than suggesting it.

### [Characteristic-Function-Tilting] the `tilt` keyword

`update(..., tilt=True)` and `hints{tilt=...}`:

- **`True`** (default): automatic. Every stable law with $\alpha<2$ is tilted.
  With $|\beta|=1$ it is one-sided ($\beta=-1$ computed as the reflection of
  $\beta=1$, so a tilt is always "toward the heavy tail"), $\tau=\tau^*$ and the
  one-sided cap applies. With $|\beta|<1$ it is split, each half gets its own
  $\tau^*$, and the two-sided cap applies. Every other law gets $\tau=0$ and
  rule A unchanged: light-tailed laws do not need it (their $\rho$ is
  negligible, so $\tau^*$ would be 0 anyway).
- **`False`**: never tilt, never split, and no cap: the law is inverted
  directly and wraps. For teaching: the same declaration with
  `hints{tilt=False}` shows the wrap the tables above measure.
- **a number $\tau\ge0$**: tilt by exactly $\tau$ e-folds across the window,
  cap applied; a two-sided law gives both halves the same $\tau$. `tilt=8`
  means "damp the wrapped tail by $e^{-8}\approx3\times10^{-4}$ and amplify
  roundoff at the edge by $e^{8}\approx3\,000$", which is the sentence the
  docstring gives. A number on a law with no power tail (any family but
  `stable`) is a `ValueError` naming why: such a law barely wraps, so there is
  nothing for a tilt to damp. `bool` is tested before `float`,
  so `tilt=1` means one e-fold and `tilt=True` means automatic; the docstring
  states it.

`info` shows the realized tilt as `tilt (0, 8.4), theta (0, 0.0045), edge error
1.5e-04`, adding `transfer sigma 0.12` for a two-sided law; `validation_df`
carries the same. `tilt` joins `_HINT_KEYS`; on any other kind it reaches
`Aggregate.update`'s `**kwargs` and is inert, as `padding` is on a `chfagg`.

The family object supplies what the tilt needs: `tilt_side`, the side of the
heavy tail, one of `'right'`, `'left'`, `'split'` (both) or `None` (neither);
for `'right'`, a `chf` valid on the closed upper half plane
$\mathrm{Im}\,s\ge0$, where the light left tail makes it analytic. For the
stable adapter these are the analytic forms above, and a test checks they
equal `Stable.chf` on the real line and $e^{\Lambda(\theta)}$ at $s=i\theta$.
`'left'` is computed by reflection and needs no second form. For `'split'`,
the family supplies `halves()`, returning the two one-sided family objects
with $\gamma_\pm$ (the $\delta_1$ goes to the right half). Every tilted family
supplies the constants $A$, $\alpha$, $\delta_1$, $\gamma$ for $\rho$,
$\kappa$, $h_{\mathrm{trunc}}$ and the edge error. The split, the transfer, the
convolution and the crop live in the engine, not in the family, so any later
family that can supply `halves()` reuses them.

### [Characteristic-Function-Families] the families

A family is a small object with: `name`; `parameters` (as written); `chf(s)`;
`cgf(s)` for real $s$ and its domain `(s_minus, s_plus)` (empty on a power side);
`power_tails`, the right and left constants and the index, or `None`;
`tilt_side` and, for a split law, `halves()`; `support`; `atom`, `(x, w)` or
`None`; `scale`, the body scale;
`moments`, mean, variance and skewness with `inf`/`nan` allowed; `reference`,
a scipy frozen law where one exists, used by the oracles and the pedagogy
comparisons; and `to_decl()`. `Stable` and `Tweedie` are wrapped by adapters in
the new module rather than changed.

Version one (spellings ruled: long forms; scipy's names where scipy has the
law, spelled out otherwise):

| family | DecL body | parameters | $K(s)$, light sides | tails | support | oracle |
|---|---|---|---|---|---|---|
| stable | `stable alpha beta [scale [loc]] [s0\|s1]` | Nolan S1 or S0 [@Nolan2020] | Laplace transform on the short side when $\vert\beta\vert=1$ | power, $A$ and $B$ from `Stable`; tilted when $\vert\beta\vert=1$, split and each half tilted when $\vert\beta\vert<1$ | $\mathbb{R}$, or $[\delta_1,\infty)$ for $\alpha<1$, $\beta=1$ | scipy in the body; exact series ($\alpha<1$, $\beta=1$); `ss.levy` at $\alpha=1/2$; `ss.cauchy` at $\alpha=1$, $\beta=0$; the long untilted window |
| tweedie | `tweedie p mean dispersion`, `tweedie p mean cv c` | reproductive $(p,\mu,\sigma^2)$, any $p\le0$ or $p>1$ | $\kappa(\theta+s)-\kappa(\theta)$ from `Tweedie` | light both sides | $\mathbb{R}$ for $p\le0$, else $[0,\infty)$ | normal, gamma, inverse Gaussian at $p=0,2,3$; compound engine and `tweedie_density` for $1<p<2$; mean $\mu$ and variance $\sigma^2\mu^p$ throughout |
| norm | `norm mean sd`, `norm mean cv c` | $\mu,\sigma$ | $\mu s+\sigma^2s^2/2$ | light | $\mathbb{R}$ | `ss.norm` |
| gamma | `gamma shape scale`, `gamma mean cv c` | $k,\theta$ | $-k\log(1-\theta s)$, $s<1/\theta$ | light | $[0,\infty)$ | `ss.gamma` |
| invgauss | `invgauss mean shape`, `invgauss mean cv c` | $\mu,\lambda$ | $(\lambda/\mu)(1-\sqrt{1-2\mu^2s/\lambda})$, $s<\lambda/2\mu^2$ | light | $[0,\infty)$ | `ss.invgauss(mu/lam, scale=lam)` |
| norminvgauss | `norminvgauss alpha beta delta mu` | $\vert\beta\vert<\alpha$, $\delta>0$ | $\mu s+\delta(\gamma-\sqrt{\alpha^2-(\beta+s)^2})$, $\gamma=\sqrt{\alpha^2-\beta^2}$, $-\alpha-\beta<s<\alpha-\beta$ | light | $\mathbb{R}$ | `ss.norminvgauss(a=alpha*delta, b=beta*delta, loc=mu, scale=delta)` |
| variancegamma | `variancegamma sigma nu theta mu` | Madan, Carr and Chang [@Madan1998] | $\mu s-\nu^{-1}\log(1-\theta\nu s-\sigma^2\nu s^2/2)$ between the roots | light | $\mathbb{R}$ | `ss.laplace` at $\theta=0$, $\nu=1$; cumulants otherwise |
| temperedstable | `temperedstable C G M Y [mu]` | CGMY [@Carr2002], $Y<2$, $Y\notin\{0,1\}$ | $\mu s + C\Gamma(-Y)[(M-s)^Y-M^Y+(G+s)^Y-G^Y]$, $-G<s<M$ | light | $\mathbb{R}$ | cumulants $\kappa_n = C\Gamma(n-Y)(M^{Y-n}+(-1)^nG^{Y-n})$ |

Moments: normal inverse Gaussian mean $\mu+\delta\beta/\gamma$, variance
$\delta\alpha^2/\gamma^3$; variance gamma mean $\mu+\theta$, variance
$\sigma^2+\nu\theta^2$; tempered stable from the cumulants. The Tweedie members
with $p>2$ are the one-sided tempered stable laws with index
$(p-2)/(p-1)\in(0,1)$, already tilted by their own $\theta$, so they need no
further tilt. Tweedie $p=1$ is a lattice law with the dispersion as its step;
it is refused with a message pointing at `agg ... poisson dsev [phi]`, as is
$0<p<1$, where no Tweedie law exists. Candidates for later, each one family
object and its tests: Student t, logistic, Laplace by name, Meixner,
generalized hyperbolic, the one-sided and generalized tempered stable laws
[@Rosinski2007]. Added to the bibliography and needed for the docstrings:
Barndorff-Nielsen (1997), "Normal inverse Gaussian distributions and stochastic
volatility modelling", *Scandinavian Journal of Statistics* 24, 1 to 13; the
author added it through archivum at @BarndorffNielsen1997.

### [Characteristic-Function-Statement] DecL

```
chfagg S stable 1.5 1 2 0
chfagg L stable 0.7 1 hints{tilt=False}  # the same law, wrap left in, for teaching
chfagg T tweedie 2.5 100 2               # any p, including p < 0 and p > 2
chfagg C tweedie 1.5 100 2               # the compound engine's law, computed directly
chfagg G gamma 100 cv 0.5 hints{log2=18}
chfagg V variancegamma 0.2 0.5 -0.1 0 note{a variance gamma return}
```

Grammar (`src/aggregate/decl.lark`), mirroring `nttagg_out`:

```
answer: ... | chfagg_out -> answer_chfagg | ...
chfagg_out: CHFAGG name as_label chf_body trailer
chf_body: STABLE numberl                -> chf_stable
        | STABLE numberl ID             -> chf_stable_param
        | TWEEDIE expr expr expr        -> chf_tweedie_dispersion
        | TWEEDIE expr expr CV expr     -> chf_tweedie_cv
        | ID numberl                    -> chf_family
        | ID expr CV expr               -> chf_family_cv
CHFAGG.2: /chfagg(?![a-zA-Z0-9._:~\-])/
```

The family words other than `stable` and `tweedie` are ordinary identifiers
validated by the transformer, so `chfagg` is the only new reserved word. The
spec stores the parameters as written, keyed by name, with `family` and, for
the two-form families, `form` (`'cv'` or positional), so the unparser renders
back exactly what was written; the class converts. Not a `port_item`: a
`chfagg` cannot appear inside a portfolio, `pnl` or bivariate, exactly as
`nttagg`.

Touchpoints, from the `nttagg` precedent: `decl.lark` (answer, rule, `CHFAGG`,
`chfagg` in the `ID` exclusion list); `parser.py` (`answer_chfagg`, the
`chf_*` callbacks, refusals as `ValueError` naming the form);
`parser_errors.py` (token descriptions); `decl_words/__init__.py`
(`_DECLARATIONS`); `underwriter.py` (`_KIND_WRITE_ORDER` beside `nttagg`, a
`kind == 'chfagg'` branch in `_factory` with a local import);
`decl_writer.py` (`_render_chfagg`); `agg.sublime-syntax`;
`src/aggregate/data/decl-help/anatomy.html` and `card.html`; the corpora
`_test_suite.agg` and `decl-testers.agg`, with
`tests/capture_spec_snapshot.py` rerun and the diff read; `library.agg`
entries so the kind is visible from the front door (the
`[NTT-Composability]` lesson); the grammar reference
(`docs/4_agg_language_reference/ref_include.rst`) regenerated; and a section
in `docs/4_dec_Language_Reference.rst` beside "Exact Compounds (nttagg)".
`python -m aggregate.decl_words` output changes, which `aggregate-api`'s
`tests/test_decl.py` checks.

### Pricing

Distortion pricing needs only the grid: `calibrate_distortions` and
`price_pentagon` are duck-typed and `_canonical_grid` already handles a
windowed or signed grid. The class carries both as thin wrappers, as
`Aggregate` does. Two cautions belong in their docstrings, not in code. Every
price is at a finite asset level $a$, so it is finite even for $\alpha\le1$, but
it needs $a$ inside the window, with $S(a)$ large against `outside_mass` when
the grid is untilted. And `guard_unbounded_anchor` refuses $p=1$ because
`bounded` is `False`, which is correct for every family here.

### [Characteristic-Function-Severity] a `chfagg` as a severity

`sev chfagg.NAME` and `ssev chfagg.NAME`, with the existing hygiene rule (the
referenced `chfagg` must carry `hints{log2=...; bs=...}`):

- grammar: `BUILTIN_CHFAGG.3: /chfagg\.[a-zA-Z][a-zA-Z0-9._:~\-]*/`, a
  `sev_ref_leaf` alternative, and `chfagg\.` added to the `ID` prefix
  lookahead beside `agg\.` and `port\.`;
- `parser.py`: `sev_ref_chfagg`, the same `_sev_ref_spec`;
- `underwriter._resolve_sev_ref`: a branch calling `inner.update(log2=, bs=,
  bucket_sizing_p=, **kw)` for this class (it already branches on
  `Portfolio`), and `_sev_ref_hygiene_message` taking its suggestion from
  `grid_window`;
- `_dhistogram_from_object`: replace the `isinstance` switch with a private
  `_output_xs_ps()` on `Aggregate`, `Portfolio` and `AggregateCHF`, each
  returning the arrays the switch reads today. This removes a special case
  rather than adding a third;
- `tail.output_support_max`: `AggregateCHF` returns `support[1]`;
- `with_hints()` on the class, returning the program with `hints{}` from
  `grid_window`, the same contract as `Aggregate.with_hints`.

A law with mass below zero used through `sev` is clamped onto the zero atom
with the existing warning; `ssev` keeps it. A tilted heavy-tailed reference
has a deficit equal to `outside_mass`; the existing
`REFERENCE_DEFICIT_MATERIALITY` warning covers it and its advice (raise the
inner's `log2`) is the right one.

### [Characteristic-Function-Pedagogy] FourierTools folds in, behind a shim

Every capability of `ft.py` gets a new home:

| 1.26.0 | new home |
|---|---|
| `FourierTools(chf, fz)` with a known law | `build('chfagg ...')` or `AggregateCHF.from_family` |
| `FourierTools(chf, 'continuous')`, arbitrary `chf` | `AggregateCHF.from_chf` |
| `invert` (density samples) | `update` (bucket masses); `pdf` is mass over `bs` |
| `invert_simpson` [@Wang2008] | `pedagogy.chf_simpson_inversion(obj)` |
| `compute_exact` | `pedagogy.chf_exact_comparison(obj)`, against `obj.family.reference` |
| `plot`, `plot_wraps`, `plot_simpson`, `plot_fourier3d`, `plot_fourier3da` | `pedagogy.chf_inversion_figure`, `chf_wrapping_figure`, `chf_simpson_figure`, `chf_fourier3d_figure`, with bodies added to `plots/_fourier.py` beside the old ones |
| `make_levy_chf` | `Stable(alpha, beta).chf` |
| `poisson_example`, `fft_wrapping_illustration`, `recentering_convolution_example`, `recentering_convolution`, `stats` | moved unchanged to `pedagogy.py` |
| `Tweedie.ft` | kept, returning the shim's `FourierTools` with a `DeprecationWarning`; `Tweedie.fz` returns an `AggregateCHF` for the stable members |
| `Tweedie.test`, `Tweedie.audit` | rewritten over `AggregateCHF` |

**The shim.** `aggregate.ft` stays importable. Its docstring opens with the
explanation: the module is superseded by `aggregate.characteristic_function`,
whose `AggregateCHF` returns bucket masses rather than density samples, sizes
its own window, tilts heavy tails and is a first-class object with a DecL
form; the table above, as a short list, says where each name went.
`FourierTools` keeps its current body unchanged, so old notebooks reproduce
their numbers, and its constructor raises a `DeprecationWarning` naming
`AggregateCHF.from_chf` and `from_family`. The moved functions and
`make_levy_chf` are forwarded through a module `__getattr__` (PEP 562) that
warns once per name and returns the new object. Nothing inside the library
imports `aggregate.ft` any more, which a test checks (`rg` in the test, or an
import-graph check). Removal is tracked as `[FourierTools-Shim-Removal]` in
`dev/TODO.md`, with no date: when the three outside documents have moved, or at
the next major release, whichever is first. `docs/3_reference/3_x_Auxiliary.rst`
documents the new module and keeps one sentence on the shim; `cantor.py:620`
points at `from_chf`.

### Out of scope

Portfolio units, `pnl` and `xpnl` engines, reinsurance and limits (ruled),
bivariate laws, a compound Poisson family (ruled), `sev nttagg.NAME` (the same
mechanism would admit it; separate item), the api kind (separate plan), and
these follow-ups, each a `dev/TODO.md` entry when the plan lands:

- `[CHF-Tail-Completion]`: `tvar` and `lev` beyond the window from the power
  tail, $\int_{x_R}^\infty x\,dF \approx A\alpha x_R^{1-\alpha}/(\alpha-1)$; and,
  for a two-sided law, the closed-form correction for each half's left-out mass
  re-entering the window through the other half's tail, which sets the
  interior error at $\alpha<1$.
- `[CHF-Sharpen]`: `sharpen` on the class through `recommended_grid`.

## Steps

Each step is one minor bump and one commit, with code, tests, `__version__`,
the CHANGELOG paragraph, `dev/TODO.md` and `dev/FEATURES.csv` (via
`dev/regen_features.py`) together. The plan moves to `dev/done/` in the first
step's commit. Tier 2 (`uv run pytest`) gates each step; tier 3 and the
numerics gate (`-W error::RuntimeWarning`) gate each bump. No docs build.

1. **`[Characteristic-Function-Object]`.** New
   `src/aggregate/characteristic_function.py`: the kernel with $\theta=0$ (bucket
   masses, atom split), rule A and rule C, the Chernoff and power edges,
   `recommended_grid`, the family protocol with `stable`, `norm`, `gamma` and
   `invgauss`, and `AggregateCHF` with the full surface above except the tilt,
   pricing, `with_hints` and the exhibit and chart registrations. Add
   `'AggregateCHF'` to `FIRST_CLASS_CLASSES`. New
   `tests/test_characteristic_function.py`. Acceptance: the kernel, normal,
   gamma, inverse Gaussian and untilted stable oracles; `test_fcc_surface.py`
   passes with `FCC_CONTRACT_EXCEPTIONS` still empty.
2. **`[Characteristic-Function-Tilting]`.** The shifted contour in the kernel,
   the analytic stable `chf`, $\tau^*$, the one-sided bucket cap and left edge,
   `outside_below` and `outside_above` (series for $\alpha<1$), the convention
   that puts the mass below into the first bucket, the `tilt` keyword and
   attributes, `tilt` in `_HINT_KEYS`, the validation rows. Acceptance: the
   one-sided tilting oracles.
3. **`[Characteristic-Function-Two-Sided]`.** The split for $|\beta|<1$:
   `halves()` on the stable adapter, the resolution transfer and
   `transfer_sigma`, the bucket factor on one half, each half on $n$ buckets,
   the $2n$ linear convolution and the balanced crop, the two-sided cap and
   rule A edges, the two-sided edge error. A separate bump from step 2 so that
   each lands, and can be bisected, alone. Acceptance: the two-sided oracles.
4. **`[Characteristic-Function-Statement]`.** The `chfagg` declaration and
   every touchpoint listed under DecL. Acceptance: each corpus line parses and
   round-trips through `decl_writer`; the snapshot diff contains only the new
   lines; `build('chfagg S stable 1.5 1 2 0')` returns an updated, tilted
   object, and `build('chfagg S stable 1.5 0.5 2 0')` a split one;
   `hints{tilt=False}` and `hints{tilt=8}` reach `update`;
   `hasattr(obj, 'pnl_program')` and `hasattr(obj, 'reins_program')` are
   `False`.
5. **`[Characteristic-Function-Families]`.** `tweedie` for every admissible $p$
   with the atom split, `norminvgauss`, `variancegamma`, `temperedstable`; the
   `Tweedie.fz` repair (stable members return an updated `AggregateCHF`, the
   dead `NotImplementedError` and the `print` go). Closes
   `[Power-Variance-Family]`. Acceptance: the Tweedie, normal inverse Gaussian,
   variance gamma and tempered stable oracles; `Tweedie(2.5, 1, 1).fz` is not
   `None` and `Tweedie(-1, 1, 1).fz.est_m` is 1 to $10^{-6}$.
6. **`[Characteristic-Function-Pricing]`.** `price_pentagon`,
   `calibrate_distortions`; exhibit registrations (`summary`, `tail`,
   `validation`, `stats`) in a new `exhibits/_characteristic_function.py`; a
   chart emitter registered as `chfagg` in
   `charts/_emit_characteristic_function.py` over `outcome_doc`, with the
   family's reference law as the companion series where one exists.
   Acceptance: the pricing oracle; `available_exhibits` and `available_charts`
   list the new entries for the class and not `reins` or `pnl`;
   `tests/test_charts_ir.py` covers the emitter.
7. **`[Characteristic-Function-Severity]`.** `sev chfagg.NAME` and `ssev
   chfagg.NAME`, `_output_xs_ps`, the resolver and hygiene branches,
   `output_support_max`, `with_hints`. Acceptance: the severity oracle; the
   hygiene error on a `chfagg` without hints names `with_hints` and gives
   paste-ready text.
8. **`[Characteristic-Function-Pedagogy]`.** The moves and the shim as
   described; pedagogy figures on `AggregateCHF`; `Tweedie.test` and `audit`
   rewritten; docs page and `cantor.py` docstring;
   `[FourierTools-Shim-Removal]` added to `dev/TODO.md`. Acceptance: no module
   under `src/aggregate` imports `aggregate.ft`; `from aggregate.ft import
   FourierTools, poisson_example` works and warns; each new figure function
   runs on its documented example (one smoke test each, no image diff).

The api kind is a separate `aggregate-api` plan, written after
`[Characteristic-Function-Pricing]` lands.

## Acceptance oracles

- **Kernel against the round discretization.** On one shared `(x0, bs, log2)`
  grid, `chfagg G gamma 4 25` and `agg G dfreq [1] sev 25 * gamma 4` (or its
  current spelling) agree bucket for bucket to $10^{-10}$ absolute; the same
  for `norm` (with a signed `ssev`) and `invgauss`.
- **Exact references.** Normal, gamma (shape at least 1), inverse Gaussian and
  normal inverse Gaussian bucket masses against
  `ref.cdf(x + h/2) - ref.cdf(x - h/2)` to $10^{-9}$ absolute at the default
  grid; gamma with shape 0.5 is flagged by the truncation row instead.
- **Stable, untilted** (`tilt=False`): the 2026-10-28 table reproduced within a
  factor of 2.
- **Stable, tilted.** The 2026-10-29 one-sided table reproduced: relative
  bucket error below $10^{-4}$ at $W/2$ and at the right edge for each row,
  below the reported `edge_error`, and the window total within $2\times10^{-7}$
  of `1 - outside_above`; all masses positive. References as in that table,
  plus `ss.levy` exactly at $\alpha=1/2$, $\beta=1$. $\beta=-1$ equals the
  reflection of $\beta=1$ to $10^{-12}$ in every bucket but the first, which
  also holds `outside_below`; its `cdf` is correct across the window (against
  the reflected `sf`). The analytic `chf` equals `Stable.chf` on the real line
  and $e^{\Lambda(\theta)}$ at $s=i\theta$ to $10^{-12}$ relative. Closure: the
  grid of `Stable(1.5, 1, 1) + Stable(1.5, 1, 2)` equals `from_chf` with the
  product of the two characteristic functions on the same grid to $10^{-12}$.
- **Stable, two-sided.** The split is exact: $\varphi = \varphi_+\varphi_-$ on
  the real line to $10^{-12}$ in $\log\varphi$ for $\alpha\in\{0.7, 1, 1.5\}$
  and $\beta\in\{0, 0.5, 0.99\}$ (measured $2\times10^{-13}$ at most). The
  2026-10-29 two-sided table reproduced: in every row the worst relative bucket
  error over the whole window is below the reported `edge_error`, and below
  $10^{-4}$ for every row with $\alpha\ge1$; all masses positive;
  `outside_below` and `outside_above` within a factor of 2 of each other; grid
  total plus `outside_above` equal to 1 within the reported `edge_error`. A
  case with $\sigma=0$ ($\alpha=1.5$, $\beta=0.5$) and one with $\sigma/h>2$
  ($\alpha=1$, $\beta=0.99$) are both in the parametrized set. Reference: the
  untilted kernel on a window 256 times longer, centered on the kept one (a
  `slow` test for the table at log2 16; the fast suite runs the row
  $\alpha=1.5$, $\beta=0.5$ at log2 14 against the same reference, held to its
  own `edge_error`).
  `tilt=False` on the same law wraps, as the table's untilted column shows.
- **Tweedie.** $1<p<2$ against the compound engine on a shared grid and
  against `tweedie_density`; the atom at 0 equals $e^{-\lambda}$ to
  $10^{-12}$; $p=0,2,3$ against normal, gamma and inverse Gaussian; mean and
  variance to $10^{-6}$ relative for $p\in\{-1, 2.5, 4\}$.
- **Variance gamma and tempered stable.** Laplace special case exact; grid
  mean, variance and skewness against the closed forms to $10^{-6}$ relative.
- **Pricing.** `calibrate_distortions(0.10, p=0.99)` on the `chfagg` gamma and
  on the matching `agg` agree to $10^{-8}$ in every distortion's shape, and
  `price_pentagon(p=0.99)` likewise in every pentagon entry.
- **Severity.** `agg X dfreq [2] sev chfagg.G` and `agg X dfreq [2] sev 25 *
  gamma 4` agree to $10^{-10}$ on a shared grid.
- Total mass as above and no bucket below $-10^{-12}$ except where the
  truncation row is flagged; `-W error::RuntimeWarning` clean.

## Downstream

- `aggregate-api`: admit the `chfagg` kind (`_KIND_OF_CLASS`,
  `SUPPORTED_KINDS`, `_KIND_ORDER`, the `BuildResponse.kind` literal,
  `KIND_LABEL` in `main.js` and `lite.js`, `DENSITY_COLS_BY_KIND`, regenerated
  `decl-keywords.json`). A separate plan after
  `[Characteristic-Function-Pricing]`.
- PMIR's emergence application (`zeta_from_pattern`, `ay_counts`) on an accurate
  stable `cdf`, skewed or two-sided, via `build('chfagg ...')`. Below
  $\alpha=1$ a two-sided law wants log2 18 or more for coverage; the coverage
  row says so.
- Teaching: `D:/projects/aggregate/education` (modules 03, 04 and 08), the
  monograph's Tweedie pages and the two monograph posts that use
  `FourierTools`, and the `fft` writing project
  (`D:/projects/aggregate/fft`: grids, windows, and the aliasing budget), whose
  subject this engine illustrates; the tilted and untilted pair on one
  declaration is a ready figure for it. The wrapping figures already serve
  [@Mildenhall2024].
- `Stable`'s class Notes point at `chfagg` for far-tail values once
  `[Characteristic-Function-Statement]` lands.

## Execution log

Execution started 2026-10-29 at 1.26.0; the plan moved here in the first step's commit.

### [Characteristic-Function-Object], 1.27.0

Landed as `src/aggregate/characteristic_function.py` with `tests/test_characteristic_function.py` (31 cases), `AggregateCHF` in `FIRST_CLASS_CLASSES`, a `chf` fixture in `tests/test_fcc_surface.py` and `dev/regen_features.py`, and a duck-typed branch in `utilities.qd`. Gate: `uv run pytest -m 'slow or not slow' -W error::RuntimeWarning`. Divergences:

- **Rule A places the rounding slack between the sides.** The plan's `x_0 = h\lfloor x_L/h\rfloor` puts the whole of the width `round_bucket` adds (up to 2x) on the right, so the left tail's wrap landed at the far right of the window, where it showed as a skewness of 7e-4 on a normal law. The slack is now split between the sides, cut at a finite support bound; 0 stays a grid point.
- **A finite support bound replaces a Chernoff edge within 5% of the window of it** (`SUPPORT_EDGE_SHARE`). The Chernoff left edge of `gamma 4 25` at mass 5e-6 is 1.77, and the 1e-6 below it wrapped onto the top of the window; the 113 buckets saved were not worth it. With the bound, the gamma and inverse Gaussian masses match scipy to 1e-12 rather than 3e-8.
- **The gamma oracle holds from shape 3, not shape 1.** `|\varphi(\pi/h)| = (1+\theta^2\pi^2/h^2)^{-k/2}` is about 1e-4 for the exponential at the default grid (a density jump at 0 decays as 1/t), 1e-8 at shape 2 and 1e-12 at shape 3; the truncation row flags shape 1, as the plan says it flags shape 0.5. The oracle test runs shapes 3 and 4.
- **The untilted stable table is reproduced through `from_chf` at `x_0 = -20`**, which is how it was measured. Under rule A the left edge of `stable 1.5 1` sits at the mass-5e-6 Chernoff edge near -5.5, and the mass below it wraps onto the top of the window, so `sf(10)` carries a 3e-5 relative error whatever the `log2`; the tilted path's 1e-40 left edge removes this in the next step.
- **`grid_window` returns `(bs, log2, x_min)` and `update` has no `tilt` keyword yet**; both gain the tilt in `[Characteristic-Function-Tilting]`. `update(log2=None, ...)`: `None` means 16, or the grid given to `from_chf`.
- **`recommended_grid` is defined by resolution**: the smallest `log2` whose rule A bucket gives at least `MIN_BUCKETS_PER_SCALE` (10) buckets per body scale, since rule A always covers; the tilted bucket cap joins it in the next step.
- **`validation_explanation` is added beside `validation_description`**, because `test_narrative_pairs_are_symmetric` requires both halves.
- **`dev/FEATURES.csv` gets its `AggregateCHF` column at the last step**, when the surface is complete; the matrix is hand curated and the regen script only audits it (it reports OK now).
- `pedagogy.tilted_aggregate_density` already takes a `tilt` keyword, so the plan's vetting sentence was overstated; it is a teaching function, not a member of the new class, and there is no collision.

### [Characteristic-Function-Tilting], 1.28.0

Landed as the one-sided path in `characteristic_function.py` (`_one_sided_window`, `_edge_terms`, `_tau_star`, `_bucket_cap`, the analytic `FamilyStable.chf`, `tail_mass_above`, `reflected`), `utilities._round_bucket_down`, `tilt` in `underwriter._HINT_KEYS`, and fourteen new cases, six of them the plan's table rows at log2 16 and 20 marked `slow` (their `alpha >= 1` references need `2 ** 24` buckets) with the same rows at log2 14 in the fast suite. The table reproduced: mid-window and right-edge relative errors 1.5e-5 and 4.0e-5 at `alpha = 1.5`, 5e-6 and 3e-6 at 1.1, 3e-7 and 1.3e-5 at 1.0, 7e-7 and 1.4e-6 at 0.7, every grid positive, every total within 2e-7 of `1 - outside_above`, and the coverage row flagged from `alpha = 1.1` down, as the plan's "mass beyond window" column says. Divergences:

- **The tail mass above the window is the series at every `alpha`, not the leading term from `alpha = 1`.** On a log2 14 window the leading term is 0.4% off at `alpha = 1` and 0.16% at 1.1, which breaks the 2e-7 total-mass check the plan sets (its 1e-7 agreement was measured on log2 16 windows). Above `alpha = 1` the series is asymptotic (`gamma ** alpha sec(pi alpha / 2)` is negative) and is summed to its smallest term, 2e-5 relative at `alpha = 1.1`; at `alpha = 1` the expansion is `a / v + a ** 2 (log v + gamma_E - 1) / v ** 2`, derived in the method's Notes from the Laplace transform and checked against scipy. Totals now agree to 4e-9 at log2 14 for every `alpha` and scale.
- **A series stopping bug at `alpha = 1/2`**: `sin(k pi alpha)` is zero at every even `k`, which the first draft read as the end of the series; the test against `scipy.stats.levy` caught it (3e-6 relative), and the stop now reads the term's magnitude before the sine.
- **The plan's untilted 256x reference wraps too**, by about `(257) ** -(alpha + 1)` of the edge mass, 1.5e-5 at `alpha = 1`, the same order as the tilted error at log2 14, where `E*` is 1.7e-5. The tests allow the reference its own closed-form wrap, `rel < edge_error + rho_ref`.
- **The heavy side takes the whole `1 - bucket_sizing_p` budget** on a tilted grid, since the light side is at 1e-40; the plan's "split equally" applies to the untilted and two-sided windows.
- **The validation mean row flags every stable law with a finite mean** (`est_m` is a window statistic; `stable 1.5 1` at log2 16 reads -0.027 against 0): honest, and `[CHF-Tail-Completion]` is the fix. The total mass row's tolerance stays 1e-6 absolute.
- **Finding for the author, outside this plan:** `Stable.fz` at `alpha = 1` with `scale != 1` is not the law `Stable.chf` describes. scipy's `levy_stable(1, beta, scale=g)` is `g` times the standard law, while Nolan's S1 with scale `g` is that plus `(2 g / pi) beta log g` (the `|t| log|t|` term picks up `log g` under scaling); `Stable.fz` passes `scale=self.scale` straight through, so its `cdf`, `sf`, `ppf` and `pdf` are shifted by `(2 g / pi) log g` at `alpha = 1`. Measured: scipy at scale 2 gives `sf(60)` equal to its own scale 1 `sf(30)`. `tail_mass_above` carries the shift; `FamilyStable.reference` inherits the discrepancy, which matters to the pedagogy comparisons of `[Characteristic-Function-Pedagogy]`.

The 2026-10-29 one-sided and two-sided measurements came from
scratch prototypes of the kernels in Mathematics (nine scripts, the last six on
the two-sided method: plain split, resolution transfer, full halves and crop,
density bound and balanced crop, rule A buckets, edge error prediction). They
are not kept; steps 2 and 3 rebuild the methods under test. The prototype
history is in Mathematics: the plain split failed at $\beta$ near $\pm1$ (fixed
by the resolution transfer), and halves of $n/2$ buckets failed near the edges
(fixed by full halves and the crop).

## References

::: {#refs}
:::
