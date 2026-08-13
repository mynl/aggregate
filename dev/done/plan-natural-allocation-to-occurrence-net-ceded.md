# Plan [NetCeded-Natural-Allocation]: allocate a gross premium to occurrence ceded and net

> **Status: EXECUTED 2026-08-13.** Phase 1 [Bivariate-Exeqa] at `1.0.0a273`,
> phase 2 [NetCeded-Natural-Allocation] at `1.0.0a274`. Phase 3
> [Bivariate-Total-Exeqa] stays deferred behind its author gate and is now
> tracked in `dev/TODO.md` under "Numerics & pricing core". The plan text below
> is the design as approved; the execution notes at the foot record what the
> code does where it differs, with reasons. Read them before working on either
> method.
>
> Drafted 2026-08-11 from the author's specification in the oversight session
> of the same day; line anchors in the audit table are as of `1.0.0a249` and
> have all moved, though nothing structural changed.

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

## Execution notes, 2026-08-13

Everything above stands as designed. What follows is what the shipped code
does where it departs from the text, and the two measurements that decide how
the tests are written. Nine items.

**1. One defect in the plan, corrected. Phase 1 test 4 as written is
vacuous.** It asks that `exeqa_self + exeqa_other` equal the index. Under axis
conditioning `exeqa_self` **is** the index by definition, so the statement
reduces to `exeqa_other == 0`. The wording came across from Portfolio, where
conditioning is on the **total** and the units genuinely sum to it. The
statement that carries the intended content is taken across two joints:
`E[C | G = g] + E[N | G = g] = g`, with the `(gross, ceded)` and
`(gross, net)` pairs built at one pinned `bs` so their gross grids coincide.
That is the version shipped, and it is the one that actually exposes the
rebucketing scatter the plan wanted reported. Phase 3
[Bivariate-Total-Exeqa] would make it a single-object check.

**2. `slice` is on `MassiveBivariateDistribution` only,** not on the dense
`BivariateDistribution` an in-core joint hands back, so phase 1 test 3 as
written cannot run on the object under test. The shipped test takes the same
route `slice` takes (read the row, normalize, hand it to a
`GridDistribution`) against the in-core joint. Adding `slice` to the dense
container would have closed the gap and was declined as out of scope. The
audit table's "exists, insufficient" verdict is right about the capability
and imprecise about where it lives.

**3. Exactness measured, and the tests follow the measurement.** Phase 1 test
1 claims a share cession gives `kappa_C(g) = g/2` **exactly**. True only when
the cession lands on the joint lattice. On `dfreq [3] dsev [2:20:2]` with
`50% po inf xs 0` at `bs=1` the bilinear scatter never splits and the error is
`1.8e-15`; on the same cession with a continuous severity the scatter smears
and the error is real. The smear is an **absolute** quantity of order `bs`, so
both it and the two-joint check above are asserted in **buckets** (measured:
0.67 and 0.19 of a bucket respectively). A relative tolerance looks terrible
near the origin, which is the one place a bucket of error does not matter.

**4. Column names are the real unit names, not the literal `exeqa_self`.**
`exeqa_<conditioning axis>` and `exeqa_<other axis>`, so on a netceded joint
they read `exeqa_Gross` / `exeqa_Ceded`. This mirrors Portfolio exactly, whose
`exeqa_total` is a real unit name and not a keyword, and it means two frames
from different conditioning directions can be joined without collision.

**5. `natural_allocation` accepts any netceded joint carrying a gross axis,**
not only `('gross', 'ceded')`. `('gross', 'net')` works (the kappa curve is
then the net one and ceded follows by subtraction), and gross on axis 1 works.
It costs one `index` call and makes `grossnet` a first-class input. Refused:
`('net', 'ceded')`, which has no gross axis to condition on, and copula mode,
which wants phase 3. Both refusals name what to do instead.

**6. The returned frame is the pentagon octet, not a bespoke `L / P / M / LR`.**
`reins_price_df` moved to `PENTAGON_STATS` at `a262`, after this plan was
written, and these two tables sit beside each other on a screen. The frame
goes through `complete_pentagon` with `Q = NaN`, so `a` is infinite and `Q`,
`PQ` and `ROE` are blank: the same convention `reins_price_df` uses for an
unlimited quote, and `L`, `M`, `P`, `LR` are populated exactly as planned.

**7. `rho_gap` is the gap between the two `rho_g` readings, as the plan says,
and it does not move with `P`.** The method always takes both readings:
`rho_joint` on the joint's gross marginal and `rho_fine` on the source
aggregate's fine 1-D gross density, with `rho_gap = rho_joint - rho_fine`. All
three ride in `.attrs`. `P` still defaults to `rho_joint`, which is what makes
the plan's phase 2 test 4 (the gross row ties to `Distortion.price` on the
joint's marginal) true. Measured on the plan's own example at `bs=4`, the gap
is 2e-3 relative under `ph 0.7` and smaller under the others.

**8. Two behaviors documented rather than engineered away.** Under a grid
deficit the identity distortion prices **above** the mean, by
`deficit * top_value`, because `choquet_weights` runs forwards and parks
unrepresented mass at the largest represented outcome. The plan's phase 2 test
1 ("the identity distortion recovers the component means") is therefore run on
a lattice discrete program with no deficit, and a second test pins the parked
quantity on the deficit-carrying one so the difference is never mistaken for
an allocation error. Separately, the reading is unlimited, so a mass
distortion (`ccoc`) on an unbounded support charges the top grid bucket; the
docstring says so and points at passing a finite-`a` price in as `P`. The
fractions, being ratios on one grid, are far steadier than the level.

**9. Structural refusals come before `_require_density`,** the ordering
`reins_price_df` uses and states ("after the structural refusal, so an object
with no cession is told that rather than told about its tail"). A caller
holding the wrong kind of object is told so without first spending a 2-D FFT
to find out.

**Reuse, for the record.** Nothing here reimplements existing machinery:
`choquet_weights` for the increments, `Distortion.effective_g` for the view
and value-type resolution, `GridDistribution` for `F` / `S`,
`complete_pentagon` for the octet, and `exeqa_df` itself for the curve.

**Tests** are `tests/test_bivariate_exeqa.py` (10 cases) and
`tests/test_natural_allocation.py` (20 cases), both in the **fast** tier: the
programs are small and the joint grids pinned, which is where a correctness
check belongs. The three existing bivariate suites are `slow` at module level
and new work should not disappear into them.
