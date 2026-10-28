---
bibliography: D:/Projects/Biblio/uber-library.bib
csl: D:/Projects/Biblio/journal-of-risk-and-uncertainty.csl
---

# Plan: distributions computed by inverting a characteristic function

`[Characteristic-Function-Engine]`. Drafted 2026-10-28 against aggregate 1.18.2.
**Status: parked design, not scheduled.** It records the analysis and
measurements of 2026-10-28 so they survive until the work is taken up; the open
questions below must be ruled before it executes, and the line references
rechecked. It absorbs `[Power-Variance-Family]`, which is in the archived
`dev/done/TODO-2026-10-07.md` and was never carried into `dev/TODO.md`. It builds
on `[Tweedie-Statement]` (`dev/plan-tweedie-statement.md`) and `[Stable-Statement]`
(`dev/plan-stable-statement.md`), which supply `TweedieConversion`, the
`conversion` kind and `Stable.chf`.

## Goal

A statement and object kind that computes a distribution on a grid by sampling
its characteristic function and inverting it with one FFT, instead of
discretizing a severity and applying a frequency generating function. The first
families are the $\alpha$-stable laws and the whole power variance (Tweedie)
family. The purposes, in the author's order:

1. **Teaching.** Illustrate the direct characteristic function method, and show
   one law computed two ways: `agg X tweedie 1.5 100 2` by the compound engine
   beside its characteristic-function counterpart, and a Pareto book beside its
   stable limit.
2. **Accurate stable distributions**, replacing scipy where it fails: in the far
   tail and in speed.
3. **The PMIR emergence application**, which needs an accurate stable `cdf`.
4. **`[Power-Variance-Family]`**: Tweedie for every $p$, including the members
   with no frequency and severity form.

## Rulings

Author, 2026-10-28: the engine is wanted; it is kept warm and done in due course,
after the Tweedie and stable statements. Everything else below is proposal.

## Why a new engine

Aggregate's engine discretizes a severity, takes its FFT, applies the count's
probability generating function, and inverts. It needs a finite expected count
and a severity. A Tweedie with $1<p<2$ has both, which is why the `tweedie`
clause expands into an ordinary aggregate and the engine never changes. A stable
law with $\alpha<2$ is the distribution of a Lévy process with infinite Lévy
measure: there is no expected count and no severity, so nothing to expand into.
The same holds for the Tweedie members with $p<0$ and $p>2$, which are
exponentially tilted, totally skewed stable laws [@Jorgensen1997]. What all of
these do have is a closed-form characteristic function, and sampling it on the
FFT's frequency grid replaces the first two steps of the compound engine.

## Current behavior, 1.18.2

- `FourierTools` (`src/aggregate/ft.py:340`) inverts a characteristic function
  with `irfft`: `invert(log2, x_min, bs, x_max, s)` samples the Fourier transform
  and returns density values on the grid; `invert_simpson`, `compute_exact` and
  plots compare with a scipy object. `Tweedie.ft` (`tweedie.py:691`) uses it.
  `make_levy_chf` (`ft.py:761`) is the standard S1 stable characteristic function.
- `Tweedie.chf` (`tweedie.py:808`), `kappa` (`:763`) and `K` (`:800`) give the
  cumulant and characteristic functions for every $p$.
- **`Tweedie.fz` is wrong for the stable members** (measured 2026-10-28). For
  $2<p<\infty$ with $\theta<0$, the usual case since it gives a finite mean, the
  code constructs a `NotImplementedError` without raising it and `fz` returns
  `None`: `Tweedie(2.5, 1, 1).fz is None`, and likewise $p=4$. For $p<0$ it
  returns the untilted `ss.levy_stable(alpha, -1, ...)` whatever $\theta$ is:
  `Tweedie(-1, 1, 1).fz.mean()` is 0, not the declared mean 1. The Cauchy branch
  ($p=\infty$) prints `set up Cauchy`.
- scipy's `levy_stable` is unreliable in the far tail and slow; see
  `dev/plan-stable-statement.md`, "Current behavior".
- Window and bucket sizing (`src/aggregate/_bucket_window.py`) is moment-based,
  and `InfiniteVarianceError` (`constants.py:474`) refuses an infinite variance.
  Validation compares grid moments with theoretical moments.
- `GridDistribution` (`src/aggregate/_grid_distribution.py:222`,
  `__init__(x, p, bs=None, name='', is_loss_value=True)`) computes `q`, `var`,
  `tvar`, `cdf`, `sf`, `pmf`, `mean` and `lev` from the grid and probabilities
  alone, with no assumption about spacing or sign.
- aggregate has no exponential tilting on its compound engine today.

## Mathematics

**Bucket masses, not density samples.** Take the grid $x_k = x_0 + kh$,
$k = 0,\dots,n-1$, window $W = nh$, and frequencies $t_j = 2\pi j/W$. Let $U$ be
uniform on $[-h/2, h/2]$, independent of $X$. Then
$\Pr(X\in[x_k-h/2, x_k+h/2)) = h f_{X+U}(x_k)$, and
$\varphi_{X+U}(t) = \varphi_X(t)\,\mathrm{sinc}(th/2)$ with
$\mathrm{sinc}(u) = \sin(u)/u$. Inverting $\varphi_X\cdot\mathrm{sinc}$ therefore
returns the bucket probabilities of the "round" discretization aggregate already
uses, rather than density values. In numpy (`np.sinc` is $\sin(\pi u)/(\pi u)$):

```python
t = 2 * np.pi * np.arange(n // 2 + 1) / W
p = np.fft.irfft(phi(-t) * np.sinc(t * h / (2 * np.pi)) * np.exp(1j * t * x0), n)
```

Two errors remain. **Frequency truncation**: frequencies beyond $\pi/h$ are
dropped. For a stable law $|\varphi(t)| = e^{-(\gamma|t|)^\alpha}$, so the error is
negligible once $h \ll \gamma$. **Aliasing**: the result is the law wrapped onto
the window, so all mass outside $[x_0, x_0+W)$ lands inside it.

**Aliasing is the binding constraint for heavy tails.** The wrapped mass is about
$(1+\beta)C_\alpha\gamma^\alpha(x_0+W)^{-\alpha}$ from the right tail, plus the left
tail. To hold it to $10^{-6}$ with $h=\gamma/50$ needs about $2^{18}$ buckets at
$\alpha=1.5$, $2^{23}$ at $\alpha=1.1$, and $2^{34}$ at $\alpha=0.7$ (computed
2026-10-28, $\beta=1$). One uniform grid is practical for $\alpha$ above about
1.3, marginal near 1, and impossible below 1 at this accuracy.

**Measured accuracy** (2026-10-28, $\alpha=1.5$, $\beta=1$, S1, unit scale,
$x_0=-20$; references are scipy `sf` in the body and the tail asymptotic at 1000):

| log2 | $h$ | $W$ | time | sf(10) rel. error | sf(100) rel. error | sf(1000) rel. error |
|---|---|---|---|---|---|---|
| 16 | 0.05 | 3,277 | 3 ms | $8\times10^{-6}$ | $-4\times10^{-4}$ | $-8\%$ |
| 18 | 0.02 | 5,243 | 13 ms | $9\times10^{-7}$ | $-1\times10^{-4}$ | $-3\%$ |
| 20 | 0.01 | 10,486 | 52 ms | $3\times10^{-7}$ | $-2\times10^{-5}$ | $-0.5\%$ |

Total mass was 1 to twelve places and every bucket probability positive. The
error at 1000 shrinks as the window grows, as the aliasing estimate predicts.

**Exponential tilting.** A spectrally positive law ($\beta=1$) has a Laplace
transform: for $\alpha\ne1$ in S1,
$\log\mathsf{P}e^{-sX} = -\gamma^\alpha s^\alpha\sec(\pi\alpha/2) - \delta_1 s$,
$s\ge0$. Tilting by $e^{-\theta x}$ makes the right tail exponentially light, and
the tilted law's characteristic function is $\varphi(t+i\theta)/\mathsf{P}e^{-\theta X}$
by analytic continuation. Two uses follow. The Tweedie members with $p<0$ and
$p>2$ are exactly such tilted laws, so their characteristic functions come from
`Tweedie.chf` with no wrapping problem in the right tail. And an untilted
spectrally positive stable can be computed tilted and then untilted by
multiplying the masses by $e^{\theta x_k}\mathsf{P}e^{-\theta X}$, at the cost of
amplifying relative error toward the right of the window; choosing $\theta$ is a
tradeoff between wrapped mass and that amplification. Laws with two heavy tails
have no such remedy.

**Atoms.** For $1<p<2$ the Tweedie has an atom $e^{-\lambda}$ at zero, and
$\varphi(t)\to e^{-\lambda}$ as $t\to\infty$, so frequency truncation leaves
ringing near zero. Subtract the atom from $\varphi$, invert the continuous part,
and add the atom back to the bucket at zero. The compound engine is already
accurate for these members, so the characteristic-function route is for
comparison and teaching.

**Grid sizing without moments.** Use the closed-form tails: the right edge where
$(1+\beta)C_\alpha\gamma^\alpha x^{-\alpha}$ equals the target survival, the left
edge likewise (or from the light tail when $\beta=1$, $\alpha>1$), and the bucket
from the body, $h = \gamma/k$ with $k$ about 50. Report the estimated wrapped mass
in the object's information, as aggregate reports aliasing today.

## Proposed design

### [Characteristic-Function-Object] the object

A `GridDistribution` subclass holding the characteristic function, the family
parameters, the grid, the masses and the wrapped-mass estimate, with `update(log2,
bs)` and the window rules above. It inherits quantiles, VaR, TVaR (finite only
for $\alpha>1$), `cdf` and `sf`. It is deliberately not an `Aggregate`: moment
validation, moment-based sizing, pricing and reinsurance all assume finite
moments or a frequency and severity. Name to be chosen and vetted under the
`Base<Kind>` rule, for example `GridDistributionFourier`.

### [Characteristic-Function-Statement] DecL

A declaration with its own keyword, stored in the recipe base and taking the
usual trailer, so that the bare `stable` and `tweedie` statements stay
calculators:

```
chf S stable 1.5 1 2 0
chf T tweedie 2.5 100 2          # any p, including p < 0 and p > 2
chf C tweedie 1.5 100 2          # the compound engine's law, computed directly
```

The keyword is an open question (`chf`, `fourier`, `direct`); `chf` is the
standard abbreviation and the name of the existing methods.

### [Characteristic-Function-Families] the families

Version one: `stable` (from `Stable.chf`) and `tweedie` for every $p$ (from
`Tweedie.chf`, with the atom split for $1<p<2$). This replaces `Tweedie.fz` for
$p<0$, $p>2$ and $p=\infty$, fixing the defects above, and lifts the $1<p<2$
restriction of the `tweedie` clause for this engine only. Candidates for later:
tempered stable laws (finite moments, so possibly more useful for insurance than
stable laws themselves), normal inverse Gaussian, variance gamma, and a compound
Poisson whose severity has a closed-form characteristic function, computed as
$\exp(\lambda(\varphi_X(t)-1))$ with no severity grid.

### [Characteristic-Function-aLL] the Loss Lab

A new object kind in aggregate-api, as `nttagg` was
(`V:/dev/aggregate-api/dev/done/plan-1.1.0-nttagg.md`), serving the distribution
and quantile views only, with moment columns blank and pricing views refused; the
api already refuses unsupported views per kind. This is a separate api plan when
the time comes.

## Steps, provisional

1. `[Characteristic-Function-Object]` with stable only, $\alpha>1$ first, and the
   accuracy oracles below. One minor bump.
2. `[Characteristic-Function-Statement]`. One minor bump.
3. Tweedie for every $p$ and the `Tweedie.fz` repair (`[Power-Variance-Family]`).
   One minor bump.
4. Tilting for spectrally positive laws, and $\alpha\le1$ where it then becomes
   practical. One minor bump.
5. The api kind, in aggregate-api.

## Acceptance oracles, provisional

- Stable: the closed-form tail constant at large $x$ within the reported wrapped
  mass; scipy in the body; the closure algebra checked on the grid
  (`chf` of a sum computed directly and as a sum).
- Tweedie $1<p<2$: the compound engine and `tweedie_density` (the exact series)
  agree with the direct route to a stated tolerance.
- Tweedie $p=3$: `ss.invgauss`; $p=0$ and $p=2$: normal and gamma.
- Total mass 1 and no negative bucket beyond a stated tolerance.
- `-W error::RuntimeWarning` clean.

## Open questions

1. The keyword, and the class name.
2. Whether the object should be a `GridDistribution` subclass (proposed) or an
   `Aggregate` with the moment machinery switched off, which would give aLL more
   views for free at the price of special cases throughout.
3. The default accuracy target for the wrapped mass, and whether the window
   follows a survival target or a log2.
4. Whether `FourierTools` is refactored onto the new object or left as the
   pedagogy and paper-figure tool it mostly is today.
5. Portfolio units, reinsurance, distortion pricing and bivariate laws are out of
   scope for version one; confirm.

## Downstream

- PMIR's emergence application (`zeta_from_pattern`, `ay_counts`) on an accurate
  stable `cdf`.
- Teaching: `D:/projects/aggregate/education` (modules 03, 04 and 08), the
  monograph's Tweedie pages, and the `fft` writing project
  (`D:/projects/aggregate/fft`: grids, windows, and the aliasing budget), whose
  subject this engine illustrates. The wrapping figures in `ft.py` already serve
  [@Mildenhall2024].

## Execution log

Not started.

## References

::: {#refs}
:::
