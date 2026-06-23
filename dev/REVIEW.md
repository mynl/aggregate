# `aggregate` beta-gate review — summary

Condensed findings from the 2026-06-21 objective review. Full verbatim text:
[`dev/beta-review-2026-06-21.md`](beta-review-2026-06-21.md). This page is the
working checklist; we'll come back to it.

## Verdict in one line
The engine and the pricing/allocation science are beta-ready and genuinely
distinctive; the gap to a *confident* beta is (1) n-unit portfolio dependence
and (2) a finished experience layer (guides, examples, docstrings, API freeze).

## Strengths (lean on these)
1. **FFT engine + validation framework** — simulation accuracy at parametric
   speed, and `explain_validation`/moment-matching flags catch misspecified
   grids that other FFT codes silently get wrong.
2. **DecL with `decl.lark` as single source of truth** — concise, testable DSL;
   every grammar line is a parametrized test.
3. **Integrated distortion / spectral / IME-2022 bounds** — coherent pricing,
   Euler allocation, Bodoff layers, pricing+allocation bounds in one model.
   Nothing else open-source combines these.

## Top 3 weaknesses
1. **Monolithic core.** `distributions.py` ~9,900 LOC; `Aggregate` alone ~125
   methods / ~6,500 LOC (severity + FFT + reinsurance + pricing + validation +
   sampling + plotting). Split into `_frequency` / `_severity` / `_aggregate`
   (+ fits) **before 1.0 freezes import paths** — free now, breaking later.
2. **Fragmented, incomplete portfolio dependence.** Units combine by independent
   convolution ("no portfolio-level frequency-mixing analog", portfolio.py:319).
   Dependence is scattered: 2-peril copula, sample-only Iman–Conover,
   allocation-only comonotonic. For a capital-allocation tool this is the most
   material functional limitation. Clean fix: a **shared mixing variable across
   units** (common shock), composing with the existing PGF machinery for
   principled n-unit dependence without copulas. _(Author: this is where Iman
   Conover / "switcheroo" come in — on the list.)_
3. **On-ramp maturity vs. API churn.** Placeholder user guides (reserving,
   capital, strategy), thin example library, pending docstring sweep, breaking
   renames still landing at a89. Beta is when users expect the surface to settle.

## In-scope functional gaps (fair to raise)
- **Reinstatements** — absent from grammar and engine; table stakes for treaty
  pricing. Most conspicuous omission given otherwise-strong reinsurance vocab.
- **Richer deductibles** — corridor, franchise/disappearing, annual aggregate
  deductible as first-class constructs.
- **Esscher / exponential premium principle** — the one classical method the
  distortion framework doesn't subsume.

## Correctly out of scope (don't add)
Loss development / IBNR / triangles, stochastic reserving, credibility, GLM /
experience rating, multi-year dynamics, inflation/trend, cat-model internals.
Cat output enters cleanly via empirical `dsev`/histogram severities — the right
seam. **State this scope explicitly in the README.**

## Possibly over-built
- Exotic distortion zoo (CLL, CLin, LEP, LY) — segregate "production" vs
  "research" distortions in docs / namespace.
- `bivariate.py` (~2k LOC for exactly two perils) — cost/benefit weakens if the
  shared-mixing dependence story lands.

## Other / smaller items
- **Publish an API-stability / deprecation policy at b1** — the "rename cleanly,
  no shims" stance is right now, must flip at beta.
- **Ship-blocker: ZT/ZM frequencies (B4)** — documented-but-broken is worse than
  absent; fix or hide before beta.
- **Extend gold-standard testing to allocation/bounds** — add closed-form checks
  where analytic allocations exist; add `hypothesis` invariants (q monotone,
  TVaR ≥ VaR, allocation sums to total, net ≤ gross).
- **PIR reproducibility** — "exhibits only via `pip install aggregate==0.30.1`"
  is tribal knowledge book readers will distrust; make it loud and documented.
- **Two expressive surfaces (DSL + objects)** — document the blessed path per
  task (DSL to construct, objects to analyze).
- **README "why / who / what-it's-not"** — best defense against unfair
  missing-feature critiques.

## Suggested priority order
1. Shared-mixing (common-shock) n-unit dependence design.
2. Pre-beta hygiene: ZT/ZM fix-or-hide; API-stability policy; README scope.
3. `distributions.py` split (background, deadline = import-path freeze).
4. Experience layer: guides, example library, docstring sweep.
5. Then: reinstatements; allocation test hardening; distortion namespace tidy.

_Not obviously in `dev/TODO.md` yet: reinstatements, shared-mixing dependence
design, ZT/ZM status, API-stability policy. Reconcile when we return to this._
