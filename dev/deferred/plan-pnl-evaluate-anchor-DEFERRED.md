# Plan a388 (DEFERRED): an asset anchor for `PnL.evaluate`, and an honest walk ([PnL-Evaluate-Anchor])

> **DEFERRED by the author, 2026-10-05, the day after it was drafted, and
> superseded in approach.** Nothing here was built. The text below is kept
> unedited because its analysis of `ccoc` stands and the successor plan reuses
> it; the *mechanism* it proposes is the part that was rejected.
>
> **Why it was deferred.** The plan rests on trimming a position's margin at
> the capital standing behind it, which presumes the actual-capital reading:
> losses above assets go unpaid, and the haircut has to be routed around a
> reinsurance recoverable that survives the cedent's insolvency (its
> Decision 3 and Decision 5, and the whole `role`-based carve-out). The margin
> walk is not in that regime. It reports **allocated** capital: a unit inside a
> bigger whole, managed to a 100-year standard, where every loss is assumed
> paid somehow and there is no haircut anywhere. Under that reading the trim is
> simply wrong, and the awkward net-versus-ceded distinction the plan spends
> three sections on never needed to exist.
>
> **What the author still wants, and the correct reason it is hard.** `ccoc`.
> The obstacle is a domain question, not an anchor question:
> `rho_g(X) = ∫₀^∞ g(S(x)) dx` with a mass at the origin, `g(0+) = d > 0`, is
> bounded below by `∫₀^∞ d dx` and so is `+∞` for **every** unbounded `X`,
> however thin the tail. Hence `dom(rho_ccoc) = L^∞` exactly, and **no larger
> subset of `L¹`**. Every other family under consideration is continuous at
> `s = 0` and so is defined on a `g`-dependent space strictly containing
> `L^∞`: all of `L¹` for `tvar`, and for `ph` with exponent `α` whatever makes
> `S^α` integrable. The exclusion recorded at `_pricing.py:985`, that `ccoc`
> "needs an asset level", is operationally true and states the wrong reason,
> and the wrong reason is what produced this plan: an anchor threaded through
> the evaluation API, rather than the question of whether the variable is in
> the functional's domain.
>
> **The superseding approach (author, 2026-10-05).** Pre-process the aggregate
> into a **capped** aggregate, then run it through the standard machinery
> unchanged. The cap is a projection into `L^∞`, not a solvency statement, so
> it applies to whatever subject was chosen and raises no net-versus-ceded
> question. Verified the same day, and it needs no new keyword:
>
> * `aggregate net of inf xs a` caps exactly: on
>   `agg 10 claims sev lognorm 50 cv 1.5 poisson aggregate net of inf xs 1400`,
>   `q(1) = 1400.0` and `est_m = 496.2567 = E[X ∧ 1400]`.
> * **Unanchored `evaluate` on a capped aggregate is byte-for-byte identical to
>   `evaluate` anchored at the cap**, across all five families. The solve
>   already locates the essential supremum itself (`_calibration_datum`'s
>   `ess_sup`), so once the variable is in the domain there is nothing for an
>   anchor keyword to do.
> * `c.bounded = True` followed by `c.calibrate_distortions(0.10, p=1)` returns
>   all five families with `ccoc r = 0.100000` at `a = 1400.0`. One line of
>   user code, against the shipped library.
> * A `pnl` over a capped engine carries **zero** mass strictly below `P - a`,
>   with an atom of `0.010014` at `-835.5`. Its unanchored panel gives
>   `ccoc r = 0.123716`, and `E[M]/(a - P) = 103.36444/835.5 = 0.123716`,
>   which is the figure this plan's own first table produces for the aggregate
>   anchored at `p = 0.99`. So capping the engine at the level the book is
>   managed to makes the walk's **untrimmed** margin the limited margin, and
>   `SA CoC` equals `ccoc` with no trim at all.
>
> **The successor, when it is picked up.** Two small items, neither a keyword,
> logged in `dev/done/TODO-2026-10-07.md`:
>
> 1. **`bounded` is not honest.** The capped aggregate reports `q(1) = 1400.0`
>    and `bounded = False` in the same breath, which is a contradiction in the
>    object's own answers and the only reason the certification above has to be
>    done by hand. An aggregate cession whose top layer is unlimited caps the
>    support, and the tail classifier should see it. `PnL` carries no `bounded`
>    at all and should derive one from its rows.
> 2. **The family set keys off the anchor instead of the domain.**
>    `EVAL_FAMILIES` / `EVAL_FAMILIES_ANCHORED` should collapse to one
>    predicate on `bounded`, which *is* the domain condition. `ccoc` then
>    appears on every surface for any bounded object with nothing else
>    changing. Optionally report it on an unbounded object as a row whose
>    `status` says `rho = inf` rather than omitting it silently; the
>    degenerate-row contract already produces that shape.
>
> Two caveats belong in the successor's docs, both inherent to the approach.
> Capping moves **all five** families rather than just enabling `ccoc`
> (`tvar` 0.2607 uncapped against 0.2724 capped at the 1-in-100), which is
> correct and is the point, one random variable and five mutually comparable
> functionals, but a capped panel is not the old panel plus a row. And the
> capped object's mean is the capped mean (496.64 against 500.00), so the
> ledger's `L` and every loss ratio move with it: a capped P&L's loss leg is
> the capped loss.
>
> Capping at the object's own `q(p)` is self-referential and needs two passes.
> That is trivial in Python and awkward in a `.agg` file, and since sweeping
> the level is the whole point (the `agg limit` ruling in "Rejected
> alternatives" below), it does not want a keyword either.
>
> ---

Status: DEFERRED 2026-10-05. Drafted 2026-10-04 against 1.0.0a387, never built.
Five design points were put to the author and resolved the same day: anchored by
default, one probability resolved per step, **trim the walk's capital ratios
too**, **drop `a=`** from the P&L surface, and **never trim a ceded row**,
because reinsurance recoverables stand through the cedent's insolvency. Those
rulings are folded into the text below; section "Decisions, as resolved" records
them, including the two places the author overruled the draft.

## Goal

Two things, and they are one change.

1. Give `PnL.evaluate` the asset anchor that `Aggregate.evaluate` and
   `Portfolio.evaluate` have carried since 1.0.0a261, so the `ccoc` family
   joins the Cherny and Madan acceptability panel on a P&L. The panel today
   reports `ph` / `wang` / `dual` / `tvar` and nothing else, because `ccoc` is
   excluded from the unanchored family set by construction and `PnL.evaluate`
   has no way to name an anchor.
2. Make the margin walk's three cost-of-capital columns consistent with
   themselves on the rows that bear risk: each divides an expected margin by a
   capital figure, and the numerator is currently the **unlimited** margin while
   the denominator is capital. Trim the numerator at that basis's own capital on
   a risk-bearing row, and leave a ceded row alone.

With both, the walk's standalone cost of capital and the panel's `ccoc` become
the same number on every risk-bearing row, which closes the round trip the a261
anchor work set out to close.

## Why (standalone summary of the analysis)

### `ccoc` is an accounting identity, not a shape fit

With assets `a` the maximum loss is `a`, so for the cost-of-capital distortion
`g(s) = d + (1 - d)s`,

```
rho_g(X ∧ a) = ∫₀ᵃ [d + (1-d) S(x)] dx = d·a + (1-d)·L,   L = E[X ∧ a]
```

Setting that equal to a premium `P` and solving gives `d = (P - L)/(a - L)`,
that is `d = M/(M + Q)` with `M = P - L` and `Q = a - P`, hence

```
r = d/(1 - d) = M/Q
```

Measured on `agg EE 10 claims sev lognorm 50 cv 1.5 poisson` against `P = 600`,
three anchors, `Aggregate.evaluate` against the identity computed by hand:

| p | a | L | M | Q | M/Q | `ccoc` param |
|---|---|---|---|---|---|---|
| 0.99 | 1435.50 | 496.636 | 103.364 | 835.50 | 0.123716 | 0.123716 |
| 0.995 | 1639.50 | 498.099 | 101.901 | 1039.50 | 0.098029 | 0.098029 |
| 0.999 | 2215.50 | 499.478 | 100.522 | 1615.50 | 0.062224 | 0.062224 |

Exact to six decimals at every anchor. Two consequences follow.

**`ccoc` has no degrees of freedom beyond the anchor.** Over those same three
anchors the `tvar` parameter moves 0.2724 to 0.2625, about 4% relative, while
`ccoc` moves by a factor of two. The four shape families are fitted by the body
of the distribution and barely notice the cap; `ccoc` is the anchor and nothing
else. Its `gini_p` is likewise `d = r/(1+r)`, a pure function of `r`, carrying
no information about the risk.

**Unanchored `ccoc` is meaningless rather than imprecise.** Forcing it today
(`pnl.evaluate(names=('ccoc', ...))`) runs and returns `r = 0.005048` on that
book, because `_evaluate_margin_arrays` (`_pricing.py:1247`) passes
`assets = ess_sup or z[-1]`, the top of the FFT grid. That is `M` divided by a
capital number set by `log2`. The exclusion recorded at `_pricing.py:985` is a
correct ruling, not a limitation to route around, and the fix is to supply an
anchor rather than to widen the family set.

### Where the limited-liability haircut lands, with reinsurance in place

This is the piece that decides which rows may be trimmed, and it is worth
writing out because the answer is not the obvious one.

Write `a` for assets, `X` for the gross loss, `R` for the recovery and
`N = X - R` for the net. A reinsurance recoverable is an **asset of the ceding
company that survives its insolvency**: well-established insurance law, and the
reason a cut-through is a negotiated exception rather than the default. So the
cedent's resources in any state are `a + R`, and what it pays policyholders is

```
min(X, a + R) = R + min(N, a)
```

Three readings come straight off that identity.

* The solvency test is `N <= a`. It is the **net** that has to fit inside
  assets, not the gross.
* The haircut is `(N - a)^+`, and policyholders bear it.
* `R` is paid **in full in every state**. There is no solvency haircut on a
  recovery, ever.

So the capital-reduced view applies to net positions and to a gross position
read standalone; it never applies to a cession. Author ruling, 2026-10-04,
overruling the draft, which had reached the right exclusion for the wrong
reason (see Decision 3).

One caveat a reader of the walk needs, and the docstring will carry it: a gross
row's limited margin is a **standalone counterfactual**, the "what if I kept it
all" reading in which `N = X` and the row's own level is the right one. It is
not what policyholders actually recover under the program, because with
reinsurance inuring their haircut is `(N - a)^+` and the recovery adds to
resources. The book-level statement is `[Waterfall-Limited-Margin]`.

### The walk divides a limited denominator into an unlimited numerator

`PnL.evaluation_df` carries `SA CoC`, `Div CoC net` and `Div CoC gross`,
computed by `_capital_ratio` (`_pnl.py:826`) as `margin / -M01`. On the
reference program used throughout this plan,

```
xpnl PP 600 premium less agg EE 10 claims sev lognorm 50 cv 1.5
     occurrence net of 200 xs 100 rate 0.12 poisson
```

the walk reads

```
Step              Margin   M01 standalone   M01 div net   M01 div gross
Gross            100.000           -835.5     -805.2155        -832.0000
occ 200 xs 100    15.634            360.0      261.2155         312.1264
All              115.634           -544.0     -544.0000        -519.8736
```

and every ratio takes the **untrimmed** `Margin` over one of those capital
figures. That is not defensible once stated: the denominator treats the capital
figure as the capital standing behind that row, and the numerator then reports
the result the row would produce if it could lose more than that. The ratio
already asserts "this is the row's capital"; the numerator has to agree.

Trimmed on the risk-bearing rows, numerator `E[max(M, -Q)]` at each basis's own
`Q`, and left alone on the cession:

| Step | basis | Q | M | M limited | old r | new r | put |
|---|---|---|---|---|---|---|---|
| Gross | standalone | 832.000 | 100.000 | 103.4021 | 0.12019 | 0.124281 | 3.4021 |
| Gross | div net | 805.216 | 100.000 | 103.6875 | 0.12419 | 0.128770 | 3.6875 |
| Gross | div gross | 832.000 | 100.000 | 103.4021 | 0.12019 | 0.124281 | 3.4021 |
| occ 200 xs 100 | standalone | -360.000 | 15.634 | n/a | -0.04343 | unchanged | n/a |
| occ 200 xs 100 | div net | -261.216 | 15.634 | n/a | -0.05985 | unchanged | n/a |
| occ 200 xs 100 | div gross | -312.126 | 15.634 | n/a | -0.05009 | unchanged | n/a |
| All | standalone | 544.000 | 115.634 | 118.9185 | 0.21256 | 0.218600 | 3.2846 |
| All | div net | 544.000 | 115.634 | 118.9185 | 0.21256 | 0.218600 | 3.2846 |
| All | div gross | 519.874 | 115.634 | 119.1735 | 0.22243 | 0.229235 | 3.5396 |

Three readings off that table.

**The put is the difference, and it grows with the tail.** `M limited - M` is
the value of the limited liability put at that capital level: 3.40 on the gross
standalone basis, 3.28 to 3.54 on the closing row. As a fraction of the margin
that is 3.40% here and 7.88% on `lognorm 50 cv 4`, since the gap is
`0.01 × (mean excess beyond the 1-in-100) / M`.

**The anchor symmetry survives the trim.** `walk_df`'s docstring records that on
the Gross row the gross cell coincides with that row's standalone `M01`, and on
the closing row the net cell does. Both coincidences carry through to the limited
margins (103.4021 twice on Gross, 118.9185 twice on All), because identical
capital gives an identical floor. Worth pinning.

**The standalone basis now equals the panel.** `0.124281` and `0.218600` are
exactly what the anchored `evaluate` panel reports for `ccoc` on those rows.

### What the trimmed div bases do and do not say

The div-basis ratio reads "the limited return on the capital this basis
allocates to this row". It is **not** "this row's share of the book's limited
result". The two agree on the standalone basis, where the capital figure is a
genuine asset level for the row considered alone, and part company on the div
bases, where the capital figure is an allocation and the honest book-level
statement would allocate the book's own haircut `(N - a)^+` across the ledger
under equal priority. That is a larger piece of work, logged as
`[Waterfall-Limited-Margin]`; this plan delivers the self-consistent ratio and
says in the docstring which of the two questions it answers.

### The implementation falls out of the existing solve

In payoff orientation the cap is a floor: a position with capital `a` behind it
settles at `max(M, -a)`, the mass below `-a` collapsing onto an atom at `-a`.
That is `_cap_loss` (`_pricing.py:1107`) mirrored, which is why the plan reuses
it through a negate-reverse sandwich rather than writing a second collapse.

No special casing is needed downstream. `_canonical_grid` reflects a payoff and
shifts by `c = max(M)`, which for a margin row is the premium (verified: the
gross row's grid tops out at exactly 600.0), so the canonical obligation is
`Z = c - M` with maximum loss `c + a` and consideration `c`. Then
`d = (c - E[Z])/((c + a) - E[Z])` with `E[Z] = c - E[M]`, which reduces to
`d = E[M]/(a + E[M])` and `r = E[M]/a`. The capital is `a` and the margin is the
floored one, which is exactly the identity.

Prototyped against the live private functions on the reference program:

```
Gross result           role=sell  a=832.000  E[Mfloor]=103.4021  M/a=0.124281  ccoc r=0.124281  tvar=0.2722
margin                 role=net   a=544.000  E[Mfloor]=118.9185  M/a=0.218600  ccoc r=0.218600  tvar=0.4243
occ 200 xs 100 result  role=buy   not anchored (recoverables stand)             ccoc r=nan       tvar unchanged
```

The identity holds on both risk-bearing rows. `tvar` moves by about 4% relative
on each (0.2604 to 0.2722, 0.4061 to 0.4243), which is the panel's visible
number change. The cession keeps its unanchored solve, so its four shape
families are unchanged from a387 and its `ccoc` cell is blank.

### `role` decides both the side and whether to floor

`_waterfall_frames` (`_pnl.py:2394`) reads a risk-bearing row's own left tail
(`gd.q(0.01)`) and a ceded row's own **right** tail (`gd.q(0.99)`, the writer's
1-in-100), and the acceptability solve flips a `buy` row to the seller's frame
(`_eval_sign`, `_pricing.py:1013`). Both surfaces already classify every row,
the walk by its `ceded` flag and the panel by `_row_role`, and that one
classification now carries both jobs: which side the capital state is read from,
and whether there is a haircut at all. `EVAL_SIGN` enumerates exactly the three
roles the rule needs, so it reads as **floor `sell` and `net`, never `buy`**.

The domain knowledge stays at the caller. `_pricing` is generic math and should
not know insurance law, so `evaluate_margin` floors whatever it is given and
`PnL.evaluate` simply does not pass `assets` for a `buy` row. That is the
layering `_pnl.py`'s own module docstring states ("All the domain magic happens
at the **caller** level").

## Decisions, as resolved

**Decision 1: the anchored reading is the default.** `PnL.evaluate` resolves
`p = 1.0 / WATERFALL_RETURN_PERIOD` and `p=None` is the unanchored reading,
which is the same meaning `None` carries on `Aggregate.evaluate` and
`Portfolio.evaluate`. The default is anchored because a P&L is a **held
position** whose own ledger already names the capital state on every row
(`walk_df`'s `M01` columns), where an `Aggregate` is an obligation that can
reasonably be considered without stated capital. A P&L measured against its
whole distribution is measured against the top of the FFT grid, the reading
`guard_unbounded_anchor` exists to discourage everywhere else.

**Decision 2: one probability, resolved per step.** The caller names a single
`p`; each ledger row resolves it against its own `GridDistribution`, so each
risk-bearing step gets its own asset level, exactly as the walk's standalone
column already does and as `Portfolio.evaluate` does across units
(`_portfolio.py:3382-3401`). A single absolute level applied to every row would
floor a small position at the whole book's capital, which is not a position
anyone holds.

**Decision 3: the walk is trimmed on risk-bearing rows, and a ceded row is never
trimmed.** Two overrules, in sequence.

The draft first recommended leaving the walk alone, on the grounds that only the
standalone basis maps onto an asset level. The author overruled it: the
untrimmed walk is double think, because the ratio already asserts the capital
figure is the row's capital. The resolution is to floor each risk-bearing row's
margin at **that basis's own** capital, which is uniform, needs no new
mathematics, and makes the standalone basis agree with the panel.

The draft then excluded cessions by a **numerical** test, `Q <= 0`, since a cover
releases capital and a negative `Q` names no level. The author overruled the
reasoning: the exclusion is substantive, not numerical. Recoverables stand
through the cedent's insolvency, so a ceded row takes **unlimited recoveries**
and the right declaration is on the row's **role**.

The distinction is not academic even though the two rules agree on everything
measurable. A remote cover has positive `Q` on all three bases
(`xpnl ... aggregate net of inf xs 2000 deposit 6` gives
`M01 = -6.0` and so `Q = +6.0` on each), so the sign rule would try to floor it,
and the floor lands at `-6.0`, which is exactly the result's natural lower bound
(a guaranteed-cost cession cannot lose more than its ceded premium, since the
recovery is non-negative). The trim is therefore a measured no-op:
`E[max(M, -6)] = E[M] = -5.192208`. **The rules agree by coincidence, not by
design**, and a coincidence is the wrong thing to encode: it holds because of a
support bound that a reinstatement schedule or a loss-sensitive cession need not
respect, and a future reader extending the code would have nothing to reason
from. The `Q <= 0` guard stays as a numerical safety net on a degenerate capital
figure; the role test is what excludes cessions, and the docstring carries the
legal reason.

**Decision 4: `a=` is dropped from the P&L surface.** `PnL.evaluate` takes `p`
only. An absolute asset level cannot be shared across a multi-row walk, which
Decision 2 already settles, so a surviving `a=` would exist mainly to carry the
unanchored sentinel, which `p=None` carries better and consistently with the
sibling methods. The escape for an absolute level on a single distribution is
`_pricing.evaluate_margin(gd, assets=...)`, which this plan gives that keyword
(step [Evaluate-Margin-Assets]), and `Aggregate.evaluate` keeps its `a=`
unchanged. `EvaluationResult` still has an `a` field: `PnL.evaluate` populates
it with the resolved level when the panel holds exactly one step, and leaves it
`None` otherwise, mirroring `EvaluationResult.premium`, which is `None` on a P&L
"where every ledger row carries its own and no single number stands for them"
(`results.py:418`). `p` is populated always, since it is one shared input. The
per-row capital is not duplicated onto the panel; `walk_df` publishes it.

**Decision 5: the panel does not anchor a ceded row either.** The panel reads a
cession from the seller's side, so flooring it would apply a solvency haircut to
the recovery leg, which is the thing Decision 3 forbids. A ceded row therefore
keeps its unanchored solve: its four shape families report exactly as they do at
a387, and its `ccoc` cell is `NaN` with a `status` naming the reason (suggested
wording, `unlimited: recoveries stand`). This is the mixed panel it sounds like,
and the `status` column is what makes it readable; the degenerate-row contract
(`_eval_panel`, `_pricing.py:1031`) already produces exactly that shape. A
welcome side effect: the whole ceded-tower block of `tests/test_pnl.py` keeps
its numbers.

**Sub-decision, for the author: the new `walk_df` columns.** Every input to a
published ratio should itself live in a published frame (`MSD`'s standard
deviation is in `economic_df`, `CR` is in `economic_ratios_df`), and after this
change the limited margins would live nowhere. Recommended: `walk_df` gains
**three** columns, one per basis, mirroring the existing `M01` triple, so the
frame reads as three capital states beside three limited margins and every ratio
is reproducible from what is printed. Suggested names `M lim standalone` /
`M lim div net` / `M lim div gross`; the spelling is the author's call. The
narrower alternative is one column on the standalone basis only, since that is
the one closing the round trip with the panel, leaving the two div numerators
implicit and documented.

## Design, dependency order

New public names, vetted against the existing surface 2026-10-04: the keyword
`assets` on `evaluate_margin` (matching `evaluate_constant_premium`, which
already has it), the keyword `p` on `PnL.evaluate` (matching `Aggregate.evaluate`
and `Portfolio.evaluate`), and three `walk_df` columns. No new module-level
names, no new DecL vocabulary, no new stored state.

1. **[Margin-Floor]** `src/aggregate/_pricing.py`. A module-private
   `_floor_margin(x, p, assets)` beside `_cap_loss` (`:1107`), collapsing the
   mass below `-assets` onto an atom at `-assets` and returning the collapsed
   `(x, p)`. Implement it as `_cap_loss` on the negated, reversed arrays and
   negate back, so there is exactly one collapse in the module. Pure array work,
   no lattice assumption, so an irregular grid (`bs = None`, which is what every
   exact P&L row is) is handled by construction. Both consumers below use this
   one helper; the walk takes the mean of its output, which equals
   `sum(max(x, -a) · p)` exactly, so there is one definition of the floor rather
   than a mean formula beside a collapse.

2. **[Evaluate-Margin-Assets]** `src/aggregate/_pricing.py`, `evaluate_margin`
   (`:1037`) and `_evaluate_margin_arrays` (`:1216`). Add `assets=None` to both.
   In `_evaluate_margin_arrays` the floor is applied **immediately after** the
   `_eval_sign` flip (`:1237`) and **before** the degeneracy block (`:1242`), in
   that order and for a reason: the flip puts the position in the frame the
   anchor is resolved in, and the floor raises `E[M]`, so a row that has no
   breakeven unlimited may have one limited, and that is the correct reading once
   the capital is named. Default the family set the way
   `evaluate_constant_premium` already does, `EVAL_FAMILIES` when
   `assets is None` and `EVAL_FAMILIES_ANCHORED` otherwise.
   This function stays **generic**: it floors whatever it is handed, at any
   `role`, and holds no view about which positions may be floored. That decision
   is domain knowledge and belongs to the caller (step 3).
   Edge ruling: `assets <= 0` means the position does not lose at the level
   asked, so there is no level to floor at. Treat it as unanchored for the solve
   and record it in the row's `status` (suggested wording,
   `no capital at p = <p>`), rather than flooring at a positive level, which
   would truncate the body. This is a numerical guard, not the cession rule.

3. **[PnL-Evaluate-Anchor]** `src/aggregate/_pnl.py`, `PnL.evaluate` (`:2756`).
   Signature becomes
   `evaluate(self, *, p=1.0 / WATERFALL_RETURN_PERIOD, names=None)`, with the
   docstring naming the constant so the default and the walk's capital state
   cannot drift apart. No `a=` (Decision 4), so no both-at-once error to raise.
   Guard `p = 0`: on a payoff it is the mirror of the `p = 1` the pricing
   surface refuses, it resolves to the bottom of the grid (measured: `gd.q(0)`
   is `-32167.5` on a book whose 1-in-100 is `-835.5`), and `PnL` carries no
   `bounded` property for `guard_unbounded_anchor` to read, so this needs its
   own check in the same voice. Per row: if `_row_role(...)` is `'buy'`, pass no
   `assets` (Decision 5) and let the row's `ccoc` cell come back blank with its
   status; otherwise take the row's `gd`, resolve `a_row = -gd.q(p)` and pass it
   as `assets=`. Populate `EvaluationResult.p` always and `.a` on a single-step
   panel (Decision 4). Since the family set is now row-dependent, `result.names`
   is the **union** in reporting order, which is what the exhibit's reindex
   expects (`exhibits/_pricing.py:450`).

4. **[Walk-Limited-Margin]** `src/aggregate/_pnl.py`, `_waterfall_frames`
   (`:2368`). Each row already has its `gd` (`:2388`) and its `ceded` flag
   (`:2391`) and its three capital figures. For a row that is **not** ceded,
   and for each basis with `Q = -bad_outcome > 0`, compute the limited margin via
   `_floor_margin`, carry it into the three new `walk_df` columns and use it as
   the numerator of that basis's `_capital_ratio` call. For a ceded row, and for
   any basis with `Q <= 0`, the column is `NaN` and the ratio keeps the
   untrimmed margin, which is the a387 behavior. `_capital_ratio` itself is
   unchanged: it takes a margin and a bad outcome, and the caller now hands it
   the right margin. `walk_df`'s `Margin` column stays the booked expected
   result, untouched, so the walk continues to foot against `economic_df` and
   `economic_ratios_df`.

5. **[Waterfall-Docstrings]** `src/aggregate/_pnl.py`. On `evaluation_df`
   (`:2540`) and `walk_df` (`:2452`): a risk-bearing row's CoC columns divide the
   margin **floored at that basis's capital** by that capital; a **ceded row is
   never floored, because a reinsurance recoverable is an asset that survives
   the cedent's insolvency, so there is no solvency haircut on a recovery**; the
   gap between a floored and a booked margin is the value of the limited
   liability put; the div-basis reading is the limited return on **allocated**
   capital, not the row's share of the book's limited result; and a gross row's
   limited margin is the standalone counterfactual, not what policyholders
   recover under the program, since with reinsurance inuring their haircut is
   `(N - a)^+`. On `PnL.evaluate`: the anchor, the per-step resolution, that
   `p=None` is the unanchored reading, that `ccoc` joins only when anchored and
   why (its parameter is `M/Q`), that a ceded row is not anchored and reports a
   blank `ccoc` with a status, and that `walk_df` publishes the per-row capital.
   This is the only place any of it is stated; none of it goes in the CHANGELOG
   as well.

6. **[Evaluate-Caption]** `src/aggregate/exhibits/_pricing.py`,
   `_evaluate_insurer` (`:425`). The `anchor` sentence (`:454`) branches on
   `result.a is None`, so a multi-row P&L panel with `a = None` and `p = 0.01`
   would wrongly read "measured against the whole distribution". Branch on `p`
   as well, with wording for the per-step case and for the ceded exception
   (suggested: "each position measured with its own 1-in-100 capital behind it;
   a cession is measured unlimited, since recoverables stand"). In
   `src/aggregate/exhibits/_pnl.py`, `_economic_waterfall_frames` (`:319`): one
   sentence in each caption for the new columns, the floored numerator and the
   ceded exception. These are the only exhibits edits: the waterfall arithmetic
   was promoted out of the exhibit layer at a253 precisely so that exhibits serve
   published frames, and both frames here are published.

## What deliberately does not change

- `Aggregate.evaluate`, `Portfolio.evaluate` and `evaluate_constant_premium`:
  already anchored since a261, unanchored by default, untouched. `Aggregate`
  keeps its `a=`; only the P&L surface drops it.
- `EVAL_FAMILIES` / `EVAL_FAMILIES_ANCHORED` and the `ccoc` exclusion rule. The
  fix supplies the anchor the rule asks for; it does not relax the rule.
- `_capital_ratio` (`:826`). It takes a margin and a bad outcome and divides;
  which margin is the caller's business.
- Every ceded row's numbers, on both frames and in the panel's four shape
  families (Decisions 3 and 5).
- `walk_df`'s `Margin` column, which must keep footing against the ledger.
- `WATERFALL_RETURN_PERIOD` stays a module constant and stays the single place
  the capital state is named, which is the 2026-08-05 ruling declining a
  configurable capital-level framework. This plan **reads** that constant twice,
  once for the walk and once for the panel default; it does not add a second way
  to set the level.
- `Portfolio.evaluate`'s rule for `p` / `a` on a multi-step result. The P&L
  diverges, with the reason recorded in its docstring (Decision 4).

## Rejected alternatives

**A DecL `assets` clause on `agg` / `port`.** Rejected 2026-10-04. The anchor is
a reading choice made at the call, not a fact about the declaration; a declared
level would compete with `WATERFALL_RETURN_PERIOD` for the same job; and it
costs a reserved word (the full mirror list is in
`dev/done/plan-derived-premium.md`, section "Design, dependency order") for no
capability the keyword argument lacks.

**A DecL `agg limit` clause, mirroring the occurrence limit.** Rejected by the
author, 2026-10-04, and the reason is the better argument of the two: you want
to **sweep** asset levels, and `Portfolio.augmented_df` already serves every
level at once from one O(n) sweep (`_portfolio_common.build_augmented`:
`exag_total(a) = cumsum(loss·gp) + loss·gS`, and per unit
`cumsum(kappa_i·gp) + a·gS(a)·TAIL_i(a)`, the tail collapsed onto an atom at `a`
and split by the tail share). `_cap_loss` does the same thing on a single
distribution post-FFT with no re-convolution. Declaring one limit locks down a
question whose whole value is in being asked repeatedly. Recorded here because
the idea reads as an obvious symmetry with `occurrence net of` and will
otherwise be re-proposed cold.

For the record on that symmetry: an aggregate cap **is** already declarable as
reinsurance, `aggregate net of inf xs a`, verified exact (`E = 429.0799` against
a direct `E[min(X, 600)] = 429.0799` on
`agg 10 claims sev lognorm 50 cv 1.5 poisson`). The post-FFT route in
`_cap_loss` agrees with it and costs nothing, because capping is a pushforward
on the already-computed aggregate law. What neither route can do is be followed
by another convolution, which is correct: a capital statement is the last thing
that happens.

**Leaving the walk untrimmed**, and **excluding cessions by the sign of `Q`.**
Both were the draft's own recommendations, both overruled by the author
2026-10-04. Recorded under Decision 3 with the arguments that defeated them.

**Document the identity and stop.** Considered and rejected: `M/Q` is trivial
arithmetic, but `Q = a - P` and nothing in the acceptability panel supplies an
`a`, so there is nothing for a reader to compute. Pointing them at `SA CoC`
instead would have been pointing at the untrimmed number.

## Tests

Extend `tests/test_evaluate_anchor.py`, which already holds the a261 Aggregate
and Portfolio anchor suite and whose stated acceptance criterion is "the round
trip closes". The reference program and every figure in the tables above are the
fixtures.

Panel:

- **the identity**: on the gross row and the closing row, the `ccoc` parameter
  equals `E[max(M, -a)] / a` computed off the row's own `gd`, to `1e-9`.
  Expected `0.124281` at `a = 832.0` and `0.218600` at `a = 544.0`.
- **ccoc joins only when anchored**: a risk-bearing row carries five families,
  `p=None` carries four, mirroring
  `test_ccoc_joins_the_families_only_when_anchored` (`:80`).
- **a cession is not anchored**: its `ccoc` cell is `NaN` with the status naming
  recoveries, and its four shape parameters equal the a387 values to `1e-9`.
- **the anchor is two-sided by role on the walk**: the ceded row's `M01
  standalone` is unchanged at `360.0`, read off its right tail.
- **the default is the walk's return period**: `evaluate()` and
  `evaluate(p=1.0 / WATERFALL_RETURN_PERIOD)` agree cell for cell.
- **the floor places the tail rather than dropping it**: total mass is preserved
  and `E[max(M, -a)] > E[M]`, the mirror of
  `test_capping_places_the_tail_rather_than_dropping_it` (`:169`).
- **`p = 0` is refused** with a message naming the fix.
- **no capital at the level asked**: a risk-bearing position that cannot lose at
  `p` reports the `no capital` status rather than flooring at a positive level.
- **`a=` is gone**: `evaluate(a=500)` raises `TypeError`, pinning Decision 4 so a
  future reader does not restore the keyword by accident.

Walk:

- **the round trip closes between the two frames**: `SA CoC` equals the panel's
  `ccoc` parameter on every risk-bearing step, to solver tolerance. This is the
  acceptance check for the whole bump.
- **the nine cells**: the table above, cell for cell.
- **a cession is untrimmed whatever the sign of `Q`**: two fixtures, the
  reference program's responsive cover (`Q < 0` on all three bases) and the
  remote cover
  `xpnl ... aggregate net of inf xs 2000 deposit 6` (`Q = +6.0` on all three),
  both reporting all three `M lim` cells `NaN` and all three ratios unchanged
  from a387. The second is the pin that makes the rule role-based rather than
  sign-based; without it a later refactor could swap the test back with no
  failure.
- **the anchor symmetry survives the trim**: on the Gross row the div-gross
  limited margin equals the standalone one, and on the closing row the div-net
  one does.
- **`Margin` is untouched**: it still equals `economic_df`'s `EX` for each step,
  so the walk foots.
- **the put is the gap**: `M lim - Margin` equals `E[(-Q - M)^+]` on each
  trimmed cell.

Existing pins to update, each keeping its unanchored coverage by naming
`p=None` (the a265 pattern, where each pin kept its lifted coverage by passing
`allocation='lifted'` explicitly):

- `tests/test_pnl.py:361`
  `test_evaluate_matches_the_constant_premium_price_form`, described in its own
  docstring as "the regression anchor for the whole margin route". Its four a105
  values are the unanchored reading and must be kept as such under `p=None`,
  with an anchored companion added beside it.
- `tests/test_pnl.py` lines 343, 379, 389, 415, 433, 442: audit each for whether
  it asserts shape (no change) or numbers (name `p=None`). `:343` asserts
  `list(...distortion...) == _FAMS`, four families, so it moves under
  Decision 1.
- `tests/test_pnl.py:486` and the ceded-tower block: **expected to be
  unchanged** under Decision 5. A moved number there is a finding, not a pin to
  update.
- `tests/test_pnl_peel.py`, `tests/test_create_pnl.py`, `tests/test_exhibits.py`:
  grep for `evaluate`, `walk_df` and `evaluation_df`, and triage the same way.
  The peel suite is the one most likely to carry walk numbers, and a peeled
  walk is mostly ceded rows, so most of it should hold.
- `tests/data/exhibit_snapshots.json`: both the two `pricing.evaluate` entries
  and the six `economic_waterfall` entries move. Regenerate via
  `tests/capture_exhibit_snapshots.py` and read the diff deliberately: the
  waterfall entries should show three new columns and moved CoC cells on
  risk-bearing rows **only**, with every cession row's ratios unchanged. A moved
  cession ratio is a finding.

## Docs

- `docs/2_aggregate_overview/pipeline-pnl.rst`: the `evaluate` prose gains the
  anchor and the `ccoc` row; the waterfall prose gains the floored numerator and
  the ceded exception.
- `docs/2_aggregate_overview/features.rst`: a version-headed subsection (the
  capability table ends at a195), with an ipython example showing the
  five-family panel beside the walk, since the two agreeing is the point.
- No DecL grammar change, so no `ref_include.rst` regen and no keyword mirror
  edits. No cheat-sheet edit.
- Per CLAUDE.md, do **not** build the doc tree in the verification loop; note
  that docs are pending a rebuild.

## Release hygiene

One bump to `1.0.0a388`. CHANGELOG section `[PnL-Evaluate-Anchor]`, one
paragraph: `PnL.evaluate` takes `p=` and defaults to the walk's 1-in-100 so
`ccoc` joins the panel, `p=None` is the unanchored reading, `a=` is not accepted
on a P&L; **in bold** that the four existing families' parameters move by a few
percent on risk-bearing rows and that the walk's three cost-of-capital columns
now divide the margin floored at each basis's capital, so every risk-bearing
row's figure rises (the reference program's gross standalone goes 0.12019 to
0.124281); that **ceded rows are unchanged on both frames**, since recoverables
stand through insolvency; `walk_df` gains three columns; and
`tests/data/exhibit_snapshots.json` needs a sync. No rationale in the entry; the
accounting belongs in the docstring and the argument belongs here.

Add `[Waterfall-Limited-Margin]` to `dev/TODO.md` under "After the cut". The div
bases now report the limited return on **allocated** capital, and the remaining
question is the book-level one: with assets `a`, gross `X`, recovery `R` and net
`N = X - R`, the cedent pays `R + min(N, a)`, so the haircut is `(N - a)^+`,
policyholders bear it and `R` is paid in full. Allocating that haircut across the
ledger under equal priority is the honest book-level limited view, and it is what
would let the div bases answer "this row's share of the book's limited result"
rather than "this row's limited return on allocated capital". The machinery it
needs is the conditional-mean ladder at every state below the boundary, which
the per-atom route can give and the stitched Palm route costs one transform per
state; the mass involved is 1%.

Regenerate `dev/FEATURES.csv` via `uv run python dev/regen_features.py`. Move
this plan to `dev/done/`. One-line commit,
`[PnL-Evaluate-Anchor] a388: PnL.evaluate takes an asset anchor and the walk trims to capital`.

Test tiers per CLAUDE.md: `pytest -n0 --dist no --testmon-forceselect` in the
edit loop, `uv run pytest` before declaring done, and
`uv run pytest -m 'slow or not slow'` at the bump. This touches the quadrature,
so also run the numerics gate,
`uv run pytest -m 'slow or not slow' -W error::RuntimeWarning`.

## API ripple

None required. The API calls `pnl.evaluate()` with no arguments and picks up the
anchored panel, with `ccoc` appearing in the `pricing.evaluate` exhibit's
INSURER pivot automatically (it reindexes on `result.names`,
`exhibits/_pricing.py:450`, which is why step 3 returns the union of the
per-row family sets), and the waterfall exhibit picks up the new columns and the
moved ratios from the published frames. Two optional follow-ups on that side:
exposing `p` as a control so a reader can re-evaluate at a different capital
level, which is the sweep the rejected DecL clause would have prevented, and a
note in the UI that the risk-bearing rows' CoC columns are now limited figures
while the cessions' are not.
