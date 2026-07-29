---
title: "Laws, types, and the scipy.stats shape/loc/scale paradigm"
bibliography: C:/s/TELOS/Biblio/uber-library.bib
csl: C:/s/TELOS/Biblio/journal-of-risk-and-uncertainty.csl
---

`scipy.stats` parameterizes every one of its hundred-odd continuous
distributions the same way: some **shape** parameters, a **location**, and a
**scale**. The first encounter is disorienting — *why does my lognormal not
take $\mu$ and $\sigma$? why does a Pareto need a* `loc`*?* — but the design
is not arbitrary. It is a direct implementation of how probability theory
itself organizes distributions, and like ISO 8601 dates, it is the convention
the whole world should adopt: briefly alien, then obviously right, then
impossible to live without. This note explains the underlying mathematics and
the payoff.

## Laws

What most applied work calls a *distribution*, the classical literature calls
a **law** — Loève's *Probability Theory* [@Loeve1955] uses the term
throughout (it is the French *loi*): the law of a random variable $X$ is the
probability measure $P(X \in \cdot)$ on the real line, equivalently its
distribution function $F$.

Every law on $\mathbb{R}$ decomposes (Lebesgue) into a mixture of at most
three pure kinds:

1. **discrete** — all mass on a countable set of atoms (claim counts, a die,
   an empirical sample);
2. **absolutely continuous** — has a density $f = F'$ (the parametric
   severity curves of daily practice);
3. **singular continuous** — no atoms *and* no density (the Cantor law);
   mathematically real, practically ignorable.

Insurance work lives in mixtures of the first two: a lognormal severity
censored at a policy limit has a continuous part and an atom at the limit; an
aggregate distribution has an atom at zero (no claims) and a continuous body.
Software that takes distributions seriously must handle laws, not just
densities — one reason `aggregate` represents everything as a discrete law on
a grid, atoms welcome.

## Types

Group the laws into equivalence classes under affine maps: two laws belong to
the same **type** if one is a positive-scale-and-shift of the other,

$$Y \overset{d}{=} aX + b, \qquad a > 0 .$$

The type is the *shape of the law with the units and origin forgotten*. This
is classical: the convergence-of-types theorem [@Loeve1955] says a
nondegenerate limit law is determined exactly up to type — which is why the
central limit theorem names "the normal type" rather than some particular
$N(\mu, \sigma^2)$, and why extreme-value theory has exactly three types.

Types matter in practice because the affine map is the actuary's daily
algebra:

- **scale** is units — currency conversion, and severity **trend**: inflating
  losses 10% multiplies scale by 1.1 and touches nothing else. "Severity
  inflation preserves the curve's shape" *is* the statement that trend acts
  within a type.
- **location** is origin — a shift: a deductible reimbursed, a premium offset
  against a loss, the shift in a shifted lognormal or shifted gamma fit.

## What scipy.stats does

`scipy.stats` implements precisely this two-level structure. Each continuous
family `dist` is defined by a **standardized** member with distribution
function $F_Z$ (its `loc=0, scale=1` form), possibly indexed by shape
parameters; the general member is

$$F(x) = F_Z\!\left(\frac{x - \mathrm{loc}}{\mathrm{scale}}\right),
\qquad
f(x) = \frac{1}{\mathrm{scale}}\,
f_Z\!\left(\frac{x - \mathrm{loc}}{\mathrm{scale}}\right).$$

Read through the lens of the last section:

- **shape parameters select the type** (which equivalence class of curves);
- **`loc` and `scale` select the member within the type** (origin and
  units).

A family with *no* shape parameters is a single type: there is only one
normal type (`norm(loc, scale)` sweeps it out), one exponential type
(`expon(scale)` — this is *why* "the exponential distribution" has only a
rate), one uniform type. A family with one shape parameter, like
`gamma(a)` or `pareto(b)` or `lognorm(s)`, is a one-parameter *curve of
types*, each type swept out by `loc`/`scale`.

The machinery — densities with their Jacobians, distribution and quantile
functions, random variates, moments, fitting — is written **once**, against
the standardized member, and every family inherits the affine algebra
correctly. No per-family shift parameters, no forgotten $1/\mathrm{scale}$ in
a density, no special-cased "three-parameter" versions of anything. That is
the brilliance: a theorem-shaped API.

## The decoder ring, applied

The places newcomers stumble are exactly the places the convention is doing
its job.

**Lognormal.** Practice writes $X \sim \mathrm{LN}(\mu, \sigma^2)$, meaning
$\log X \sim N(\mu, \sigma^2)$. Factor it: $X = e^{\mu}\, e^{\sigma Z}$ with
$Z$ standard normal. So $e^{\mu}$ is *pure scale* and $\sigma$ is the shape —
$\mu$ is not a location of $X$ at all (it is the log of its scale). Hence

```python
lognorm(s=sigma, scale=np.exp(mu))
```

which looks odd precisely once, and is then permanently clearer than the
textbook form: it says out loud that a lognormal's $\mu$ is a unit choice,
which is why trend acts on $e^\mu$ and leaves $\sigma$ alone.

**Pareto.** `scipy.stats.pareto(b)` is the single-parameter Pareto with
survival $x^{-b}$ on $[1, \infty)$. The actuarial Pareto (Lomax) with
survival $(\theta/(x+\theta))^b$ on $[0, \infty)$ is *the same type*, shifted
to start at the origin:

```python
theta * pareto(b) - theta        # i.e. pareto(b, scale=theta, loc=-theta)
```

Two industries' parameterizations, one distribution, zero special cases — the
affine map absorbs the difference. (DecL accepts the algebra literally:
`sev 100 * pareto 1.3 - 100`.)

**Gamma and friends.** `gamma(a, scale=theta)` — shape $a$, and the rate
convention is just $\mathrm{scale} = 1/\lambda$. Where a "three-parameter
gamma" appears in the literature, scipy needs no new family: that is
`gamma(a, loc=c, scale=theta)`, the shifted gamma used for aggregate
moment-matched fits.

## How `aggregate` leans on it

The `aggregate` package [@Mildenhall2024] sits directly on this paradigm and
removes the remaining friction for actuarial users:

- **Moment-form parameters.** Practitioners think in means and CVs, not shape
  parameters. DecL's `sev lognorm 100 cv 2` solves for the shape and scale
  with the stated unlimited mean and CV — any scipy family, parameterized the
  way pricing thinks [@Klugman2012].
- **The affine algebra is language.** `sev 100 * pareto 1.3 - 100` and
  `ssev 100 - lognorm 80 cv 0.025` *are* `scale * X + loc` written down;
  what scipy expresses in keyword arguments, DecL expresses as arithmetic.
- **Discrete laws are first-class.** `dsev [2 5 12] [.5 .3 .2]` is a pure
  atomic law (the discrete corner of the Lebesgue decomposition), with exact
  moments and quantiles — no continuous impersonation.

One convention from the measure theory down to the policy form: laws,
organized into types by affine maps; shape to pick the type, scale for the
units, location for the origin. Learn it once; read every distribution in the
scientific Python ecosystem — and every DecL severity clause — at sight.

## References

::: {#refs}
:::
