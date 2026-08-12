# Plan [NetCeded-Natural-Allocation]: allocate a gross premium to occurrence ceded and net

> **Status: DRAFT for author review, 2026-08-11.** Written up from the author's
> specification in the oversight session of the same day; nothing is
> implemented. Line anchors are as of `1.0.0a249`.

> **Release status: additive.** Two new methods on `BivariateAggregate`, which
> is in the 1.0 stable list; nothing existing changes behavior. Does not gate
> `1.0.0b1`. The author slots it relative to the 3D surface work
> (`dev/plan-3d-plot.md`), with which it shares mathematics (the kappa curve)
> but no code dependency in either direction.

## The calculation

G = C + N splits gross into ceded and net through an **occurrence** program, so
N is not comonotonic with G (under an aggregate program it would be). A
distortion g is calibrated on the gross basis so that rho_g(G) = P, with rho_g
the Choquet integral. The natural allocation of P to the components is

    A_C = E[C g'(S_G(G))],   A_N = E[N g'(S_G(G))],   A_C + A_N = P

and conditioning on G reduces it to the kappa curve:

    kappa_C(s) = E[C | G = s],   A_C = E[kappa_C(G) g'(S_G(G))]

kappa_C cannot come out of the 1-D FFT machinery, because C and N are dependent
(shared claim count, comonotone per-claim cession). It comes out of the
bivariate joint, which the library already builds.

The discrete form, and why the usual "conditions apply" on g'(S) disappears: on
the lattice, use increments

    Delta_gS_i = g(S(g_{i-1})) - g(S(g_i))

off the joint's own G marginal, and A_C = sum_i kappa_C(g_i) Delta_gS_i. That is
the Lebesgue-Stieltjes statement directly: no derivative of g is evaluated,
atoms are handled exactly, and A_C + A_N = rho_g(marginal) holds to floating
point by construction because kappa_C + kappa_N is the identity.

## What exists and what is missing (audited 2026-08-11, a249)

| piece | where | verdict |
|---|---|---|
| the joint | `build_netceded_joint(agg, views=...)`, `bivariate.py:791`; any two of gross / ceded / net; the engine behind `Aggregate.occ_bivariate` and the DecL `grossceded` / `grossnet` / `netceded` routes; both axes share one common `bs` by construction | exists |
| pointwise conditionals | `slice(x=..., y=...)`, `bivariate.py:3354`, the normalized conditional law as a `GridDistribution`, one conditioning value per call | exists, insufficient |
| the kappa curve over the grid | nowhere: zero `exeqa` or kappa hits in `bivariate.py` | missing |
| a distortion applied to a bivariate | nowhere: `Distortion.price` (`spectral.py:1898`) is 1-D; Portfolio's `exeqa_` machinery computes kappa by the FFT trick under unit independence, which is exactly what fails here | missing |
| variable features | `contract_terms.py` is 1-D transforms phi(x) throughout (retro, swing, slide, profit commission, corridor, reinstatements); checked and confirmed not this calculation | not relevant |

## Names, vetted per the house rule

- **`BivariateAggregate.exeqa_df`**, the kappa frame. Follows Portfolio's
  `exeqa_` vocabulary (its `density_df` carries `exeqa_<unit>` columns), so the
  two surfaces speak one language. No collision: `exeqa` appears nowhere in
  `bivariate.py` and `exeqa_df` nowhere in `src`. `kappa_df` was considered and
  rejected: `Tweedie.kappa` (`tweedie.py:763`) already uses the word for the
  cumulant function, and one word carrying two meanings across classes is what
  the vetting rule exists to catch.
- **`BivariateAggregate.natural_allocation`**, the allocation method. No
  `natural_allocation` callable or attribute exists anywhere in `src`.
  `allocate` was rejected as too generic a claim on the namespace.

## Phase 1 [Bivariate-Exeqa]: the kappa curve

`exeqa_df(axis=0)` on `BivariateAggregate`. Precondition: the object carries a
joint (the existing `_require_density` pattern, `bivariate.py:1964`).

Returns a frame indexed by the conditioning axis grid with columns `p` (the
marginal mass), `F`, `S`, `exeqa_<other>` (the conditional mean of the other
axis), and `exeqa_self` (the identity, carried so the additivity check is a
column subtraction, mirroring Portfolio carrying `exeqa_total`).

Implementation is one matrix vector product per direction: numerator, the other
axis coordinates against the joint; denominator, the marginal. Zero-mass rows
mask to NaN with the `p` column saying why. Reads the joint through existing
accessors; no new state.

Tests:

1. **Quota share exactness.** A pure share cession, `occurrence net of 50% po
   inf xs 0`, has kappa_C(g) = 0.5 g exactly at every lattice point with mass.
   One assert pins sign conventions, axis order and grid alignment together.
2. **Tie to the moment store.** The mass weighted mean of `exeqa_ceded` equals
   the ceded el that `reins_stats_df` reports, at a tolerance driven by the
   joint's deficit.
3. **Pointwise agreement with `slice`.** `exeqa` at a grid value equals
   `slice(x=value).mean()`.
4. **Additivity.** `exeqa_self + exeqa_other` equals the index to scatter
   tolerance: reported, not asserted at machine precision, because the
   rebucketing scatter can put mass epsilon off the line c + n = g. The linear
   scatter is mean preserving, which is why the tie is tight.

## Phase 2 [NetCeded-Natural-Allocation]: the allocation

`natural_allocation(distortion, P=None)` on a netceded-mode
`BivariateAggregate` built with views `('gross', 'ceded')`, so conditioning on
G is a straight axis slice and kappa_N is g minus kappa_C. The method takes an
already calibrated `Distortion` and does no calibration of its own:
calibration is `_pricing`'s business and the caller's choice.

Weights are the Delta_gS increments on the joint's own G marginal, the same
convention as `Distortion.price`, so the total ties to the 1-D answer on the
same grid, same number.

Return a tidy frame, rows gross, ceded, net; columns L (component mean), P
(allocated premium), M (P less L), LR. The gross row is the total; ceded plus
net ties to it by construction. Alongside it, report `rho_gap`.

**The calibration grid mismatch, settled here so it is not re-derived later.**
g is calibrated on the aggregate's fine 1-D gross density; the joint's G
marginal is a coarser rebucketed cousin with its own deficit, so rho_g of it
will not hit P to the bit. The method computes allocation **fractions** on the
joint's grid and applies them to the caller's P (defaulting to rho_g of the
joint marginal when P is not given). Robust to discretization, preserves
additivity exactly, and the gap between the two rho_g readings is reported as
`rho_gap` rather than silently absorbed.

Tests:

1. The identity distortion recovers the component means.
2. Quota share: the fractions are q and 1 less q under every distortion.
3. Fractions sum to one and the returned premiums to P, exactly.
4. The gross row ties to `Distortion.price` evaluated on the joint's marginal.
5. `rho_gap` is small on the standard test programs and is reported.

Docstring Notes carry the mathematics: the increment form and the fraction
rule are the two decisions a reader cannot recover from the code.

## Phase 3 [Bivariate-Total-Exeqa], deferred, author gate

The generic form for any bivariate: E[X | X + Y = s] and E[Y | X + Y = s] on
the total's own grid. Serves copula-mode pairs (allocation of a dependent two
unit total, which no Portfolio machinery can do) and is the same quantity the
3D surface's total cut and kappa dots read client side today. Where the two
axes share a `bs` the anti-diagonal is lattice aligned and the sums are exact;
where they differ, route value weighted mass and plain mass through the
existing scatter internals (`_scatter_1d`, the pushforward machinery) onto the
total grid and take the ratio. Deferred because the netceded ask does not need
it; recorded because it unifies three consumers (allocation, the 3D chart,
dependency diagnostics) and should be designed once, deliberately.

## Out of scope

- Any app or API surface. When this is wanted on screen it becomes an
  additional view of the `pricing.allocate` exhibit for occurrence-reinsured
  Aggregates; `dev/plan-pricing-exhibits.md` (the Pricing pane redesign,
  2026-08-12) records the slot. This plan is the library capability only.
- Portfolio n unit dependent allocation. Nothing touches Portfolio's `exeqa_`
  machinery or its independence assumption.
- `contract_terms` and reinstatement interactions.
- Allocation bounds off the bivariate kappa. The pricing bounds machinery
  consumes kappa columns, so the fit is natural; noted for later, not planned
  here.

## Cadence

Each phase lands with its own `1.0.0aNNN` bump, a one line commit
(`[Bivariate-Exeqa] aNNN: ...`, `[NetCeded-Natural-Allocation] aNNN: ...`),
the CHANGELOG section as the real description, `dev/FEATURES.csv` regenerated
to pick up the new members, and NumPy docstrings with the why in Notes. On
landing, move this plan to `dev/done/` and tick the `dev/TODO.md` entry the
author files for it.
