# Plan a394: the junior leg of the priority ladder ([Priority-Junior-Leg])

Status: executed. Ruled and numbered by the author on 2026-10-06 ("NNN will be
a394. This is to put back lower priority analysis."), against 1.0.0a393.
Landed in two bumps, `1.0.0a394` and `1.0.0a395`; see the execution log at the
foot of this file.

## Goal

Restore the second-priority (subordinated) expected recovery that 0.30.1 carried
as `e2pri_{unit}`, under the name **`ex_junior_{unit}`**, and surface it with the
two legs already present so the whole priority ladder reads in one place. Add
the conditional-mean ladder and the conditional law that let the subordination be
studied the way `dev/done/plan-a3xx` style notes study reinsurance credit, and
that the companion note
`D:\Projects\aggregate\notes\credit-impact-of-reinsurance\reinsurance-and-priority-2.qmd`
consumes.

No part of the old `analysis_priority` / `priority_capital_df` /
`epd_2_assets` surface comes back. See "What does not come back".

## Why (standalone summary of the analysis)

### The question

An insurance company writes direct insurance and assumes reinsurance. In a
number of US receivership schedules a ceding company's reinsurance recoverable
ranks **below** the direct policyholder's claim: the policyholder is a senior
creditor of the estate and the cedent is junior. Quantifying the gap needs three
expected recoveries at an asset level `a`, for a unit `i` against the pool of the
other units `-i`:

| priority rule | recovery of unit `i` | column |
|---|---|---|
| senior | `min(X_i, a)` | `lev_{i}` |
| equal (pro rata) | `X_i · min(1, a/X)` | `exa_{i}` |
| junior | `min(X_i, (a − X_{-i})^+)` | **`ex_junior_{i}`** (missing) |

The first two are already in `density_df`, written by
`_portfolio_density.add_exa`. The third is the whole subject of this plan.

The shortfall decomposition that follows is worth stating, because it is the
dual of the identity in the companion note 1. With `T = X_i + X_{-i}` the total:

```
D_junior  = min((T − a)^+, X_i)          the junior tranche absorbs the
D_senior  = ((T − a)^+ − X_i)^+          whole default first, up to its own size
```

### Why it is cheap

For independent units the junior recovery is a single convolution,

```
P min(X_i, (a − X_{-i})^+) = sum_y lev_i(a − y) · p_{-i}(y)
                           = ift( ft(lev_i) · ft(p_{-i}) )  evaluated at a,
```

which gives the **whole curve in `a` at once**. The 0.30.1 implementation was
exactly this (`portfolio.py:2427` at tag `Version-0.30.1`). In 1.0 the second
operand is already built: `_portfolio_density._ft_nots` constructs the
leave-one-out transforms `ft(p_{-i})` for the kappa numerator, and each unit's
`Aggregate.ftagg_density` **persists on the object after `update`**. Measured on
a two-unit book at `bs=1/4, log2=20`:

* rebuilding `_ft_nots` from the persisted `ftagg_density` reproduces `p_total`
  to `2.2e-16`;
* `lev_A(a) + ex_junior_B(a) − lev_total(a) = −5.9e-11`, both ways round;
* the equal-priority legs foot, `sum_i exa_i(a) − lev_total(a) = −6.9e-07` on a
  three-unit book at `log2=21`.

So the engine is one `ift(ft(lev_i) * nots[i])` per unit over state that is
already in memory.

### The structural facts the numbers show

Two units, `Direct` 40 claims lognormal 50 cv 2, `Assumed` 12 claims lognormal
90 cv 3, both Poisson, `a = q(0.99) = 7038`, `bs=1/4 log2=20`. EPD ratios
`P D_i / P X_i` (Butsic's measure, `@Butsic1994`):

```
                              Direct (A)   Assumed (B)
senior                          0.000408      0.008717
equal priority                  0.002725      0.015306
A senior, B junior              0.000408      0.019597
B senior, A junior              0.006283      0.008717
total EPD                       0.007137
```

1. **The three rules distribute the same pot.** All sum to `min(T, a)`, so at
   fixed assets subordination is a pure redistribution, zero sum in expected
   recovery. There is no asset-basis dilemma of the kind that occupies note 1.
2. **The senior leg does not see the junior book**, since `min(X_1, a)` does not
   involve `X_2`. A's EPD is identical in rows 1 and 3. Writing an assumed book
   costs the direct policyholder nothing at fixed assets, and the premium it
   brings raises `a`, so it makes them strictly better off.
3. **The redistribution is asymmetric, and the asymmetry is size.** Moving A
   from equal priority to senior improves A by 6.7x and costs B only 1.28x,
   because A is roughly twice B.
4. **EPD is monotone in rank** for every unit, senior < equal < junior. On the
   three-unit check: Direct 0.000316 < 0.002593 < 0.006309. This is the
   cheapest acceptance test.

### The conditional-mean ladder, and why Palm does not reach it

The companion note wants `P[D_junior | T = x]`, the analogue of note 1's
conditional-mean line. `Aggregate.palm_kappa` does **not** deliver it, for the
same structural reason note 1 documents: the target is a nonlinear function of
an *aggregate*, so there is no per-claim `c` with `D = sum_j c(X_j)` and nothing
for the Palm identity to grip. That holds whether the two books are independent
(separate event streams, so Palm does not apply at all) or share one event
stream (the `clash` model, where Palm gives the first-order split `P[X_i | T=t]`
exactly and then stalls).

Independence supplies an exact 1-D route instead. Split the senior pool's pmf at
the asset level, `p_{-i}^{<a}` and `p_{-i}^{>=a}`; then for `x > a`

```
P[ D_junior · 1{T = x} ] = (u·p_i) * p_{-i}^{>=a} (x)
                         + (x − a) · ( p_i * p_{-i}^{<a} ) (x)
```

two convolutions per asset level, exact on the full grid, with
`P[D_senior | T=x] = (x−a)^+ − P[D_junior | T=x]` free by subtraction.
Verified: integrating the ladder against `p_total` reproduces
`P X_i − ex_junior_i(a)` to four parts in `1e5`, and a 4M-path Monte Carlo
agrees within its standard error.

The **conditional law** (not just its mean) is also exact and needs no 2-D FFT:
the law of `X_i` given `T = t` is `p_i(u)·p_{-i}(t−u)/p_total(t)`, one slice at
`O(n)` per row, capped at `(t−a)^+`. A couple of hundred rows across the tail
gives an exact cloud on the fine grid, where note 1 had to accept a budget
`2048^2` joint. This is the computational payoff of Portfolio independence, and
it is worth saying out loud in the docstring.

## Where the flag lives: nowhere

**Not an `update` kwarg.** `ftagg_density` persists on each `Aggregate` after
`update`, so the junior leg is computable on an already-built `Portfolio` with no
new state and no FFT recomputation (verified above). A lazy cached property is
therefore strictly better than a kwarg threaded through `update` into `add_exa`:

* `density_df` stays as lean as `[Portfolio-Refactor]` left it. The EPD and
  eta-mu family was deleted for leanness (`dev/done/plan-meta.md` meta.7,
  `dev/done/plan-numerics-0-meta.md` D4) and nothing here reverses that
  decision.
* It works post-`build`, which is what a note or a notebook actually does.
* No plumbing. `update` already carries nine keyword arguments.

The cost is one `ft` of the `lev` column plus one `ift` per unit, on a buffer
already allocated.

## Deliverables

### [engine] `_portfolio_density.junior_recovery`

New module-level function beside `add_exa`, single owner of the convolution:

```python
def junior_recovery(port, unit):
    """E[min(X_i, (a - X_{-i})^+)] as a function of assets a, on the output grid."""
```

Reads `port.density_df['lev_{unit}']` and the leave-one-out transform from
`_ft_nots({a.name: a.ftagg_density for a in port.agg_list})`, returns the first
`n_out` entries of the inverse transform.

Two guards, both raising rather than returning quiet nonsense:

* **Signed or windowed grids.** `lev_{unit}` is evaluated at the output grid's
  loss values from the unit's *native* grid; on a windowed book the two differ,
  and on a signed P&L grid "assets" and a receivership waterfall do not mean the
  same thing (`exa_{unit}` is already NaN there). Raise `NotImplementedError`
  with the same wording style as `Aggregate.palm_kappa`.
* **`padding >= 1` required**, and the result is read only on the first `n_out`
  entries. `lev` does not decay (it rises to `E[X]`), so the statement that
  matters is that `p_{-i}` has support inside `[0, n)` and the convolution at
  `a < n` therefore touches `lev` only at indices `<= a`. No circular
  contamination in the range read. Say this in `Notes`.

### [engine] `Portfolio.priority_df`

Cached property. Index `loss` (read as the asset level `a`), columns

```
ex_senior_{unit}   alias of lev_{unit}
ex_equal_{unit}    alias of exa_{unit}
ex_junior_{unit}   new
ex_total           alias of lev_total
```

A reporting view, in the spirit of `reins_density_df`: it duplicates values, not
logic, and carrying all three legs under uniform names is what makes the ladder
legible and the footing checks one-liners. `ex_senior` / `ex_equal` are aliases
and the docstring says so, pointing at `add_exa` as the owner.

### [engine] `Portfolio.priority_epd_df(assets)`

The headline table: rows `MultiIndex (unit, rule)` with `rule in
{senior, equal, junior}`, columns `mean`, `recovery`, `shortfall`, `epd`, plus a
`total` block. One row group per unit, read off `priority_df` at `assets`.
This is the readable kernel of the old `analysis_priority`.

### [engine] `Portfolio.priority_kappa(assets, unit)`

The two-convolution conditional-mean ladder, returning
`(kappa_junior, kappa_senior, p_total)` on `density_df.loss`, NaN where
`p_total` is below the validation noise floor. Named to rhyme with
`Aggregate.palm_kappa`, and its `Notes` says plainly that Palm cannot do this
and why.

### [engine] `Portfolio.priority_conditional(assets, unit, totals)`

The exact conditional law of the junior shortfall given `T = t` for each `t` in
`totals`, by direct slices. Returns a list of `GridDistribution`, so the note's
cloud figure and any distortion pricing of the shortfall come free. Optional:
drop it if `priority_kappa` plus the EPD table carry the note.

### [tests] `tests/test_priority.py`

1. `lev_i(a) + ex_junior_{-i}(a) == lev_total(a)` for a two-unit book, both ways
   round, to `1e-9` relative.
2. `sum_i exa_i(a) == lev_total(a)` (regression guard on the existing legs).
3. EPD monotone in rank, senior <= equal <= junior, every unit, three-unit book.
4. `sum_t kappa_junior(t) · p_total(t) == P X_i − ex_junior_i(a)` to `1e-4`
   relative; `kappa_junior + kappa_senior == (t−a)^+` pointwise.
5. A `dfreq` / `dsev` discrete book where all three legs are exact by hand, so
   one case is pinned to arithmetic rather than to another FFT.
6. `NotImplementedError` on a signed (`pnl`) book and on a windowed book.

### [docs] and release hygiene

`docs/` reference page for the three new methods; grep for stale `:meth:`
references. `CHANGELOG.md` one paragraph, `dev/TODO.md` entry marked, plan moved
to `dev/done/` on the author's word.

## What does not come back

* **`epd_2_assets` / `assets_2_epd`.** Dictionaries of `scipy.interpolate.interp1d`
  splines wrapped in `minus_arg_wrapper` / `minus_ans_wrapper`, which swallowed
  `ValueError` and returned a literal `999` sentinel. Where an inverse is wanted
  (what assets restore the junior creditor to its equal-priority EPD?), root-find
  with `scipy.optimize.brentq` on the `priority_df` curve at call time.
* **`analysis_priority`.** Six scenarios with labels like
  `'thought buying (line 2pri epd = base not line eq pri epd'`, emitted as an
  HTML `<ul>` story. One of the six is economically interesting, the cedent that
  priced its recoverable as if equal priority, and it belongs in the note's
  mispricing section, not in the library.
* **`priority_capital_df`** and the `full_report_list` hooks that printed it.
* **The eta-mu column family.** `ημ_{unit}` was the materialized not-unit
  density; nothing here needs it materialized, only its transform, which
  `_ft_nots` already holds.

## Scope boundaries

* **Independence.** Everything rests on `Portfolio` units being independent, which
  they are by construction. A direct and an assumed book sharing cat events is
  the `clash` form (`bivariate.py:147`), and then the junior recovery is a
  genuine 2-D pushforward on `BivariateDistribution`, identical to note 1. Out
  of scope here; the docstrings must say that the columns assume independence.
* **Two tiers, not k.** `lev_i` is "unit `i` senior to everything else" and
  `ex_junior_i` is "unit `i` junior to everything else pooled". Both are 2-tier
  readings of a k-unit book. A real receivership schedule has more tiers
  (administrative expenses senior to all, then policyholder claims, then general
  creditors, then surplus notes), which is the sequential
  `min(X_j, (a − sum_{i<j} X_i)^+)`. That is a genuine generalization and a
  separate plan; note the formula and stop.
* **Naming.** "Waterfall" already means the P&L margin walk in this repository
  (`PnL.walk_df`, `[Waterfall-Net-Of-Tier]`, `[Waterfall-Gross-Basis]`). The
  liability-side ladder is **priority** / **subordination** throughout, never a
  waterfall.
* **Equal priority is incurred-amount pro rata**, `P[X_i · min(1, a/X)]`, which is
  the modeling convention and matches `@Phillips1998`. An actual receivership
  pro-rates allowed claims with timing and discounting. One honesty sentence in
  the docstring, no more.
* **No grid heroics.** The senior leg is a limited expected value and is
  unforgiving about severity discretization: the first proof-of-concept run used
  the auto grid, `bs=40` against a severity mean of 50, and lost 5% of the
  represented mean while every internal identity still closed to `1e-11`. The
  tests must assert the represented means against the declared means, or they
  will certify a wrong book as consistent.

## Names vetted

`rg` over `src/`, `tests/`, `docs/` found no existing `ex_junior`, `ex_senior`,
`ex_equal`, `priority_df`, `priority_epd_df`, `priority_kappa`,
`priority_conditional`, or `junior_recovery`. Per the CLAUDE.md shadowing rule,
none is assigned as an instance attribute in any `__init__`, and the three
`Portfolio` additions are a property plus two verbs, so no noun/verb collision.
The rejected 0.30.1 name `e2pri_` is exactly the cryptic-code form the house
rules prohibit.

## Acceptance

```
UV_PROJECT_ENVIRONMENT=.venv uv run pytest tests/test_priority.py
UV_PROJECT_ENVIRONMENT=.venv uv run pytest                     # fast gate
UV_PROJECT_ENVIRONMENT=.venv uv run pytest -m 'slow or not slow'   # at the bump
```

and the companion note renders with its EPD table, cloud, and ladder figures
reproducing the four structural facts above.

## References

`Butsic1994` (the EPD measure), `Phillips1998` (the equal-priority pricing
baseline that fact 2 contradicts), `NAIC2007` (Insurer Receivership Model Act),
`Cai2017` (joint insolvency allocation), `Chen2015c` (reinsurance network
contagion). All five are in `D:/Projects/Biblio/uber-library.bib`.

**Open legal question, for the author, not for the model.** Whether a ceding
insurer's claim sits with policyholders in Class 2 or drops to general creditors
in Class 5 varies by state adoption of the model act. The plan asserts only that
the split exists; no count of states should be written without checking.

## Execution log

Executed 2026-10-06 against `1.0.0a393`. Two phases, two bumps, one commit each.

### Review findings before starting

* **The plan's facts hold against today's code.** The plan was written against
  `1.0.0a387` and six bumps landed since (`a388` to `a393`), none of which
  touched `_portfolio_density.py` or the combine in `_portfolio.py`. `add_exa`
  still writes `lev_{unit}` / `exa_{unit}` / `lev_total` / `e_{unit}`,
  `_ft_nots` is still the single owner of the leave-one-out products, and
  `Aggregate.ftagg_density` still persists after `update`.
* **Every name in "Names vetted" is still free.** No `ex_junior`, `ex_senior`,
  `ex_equal`, `priority_df`, `priority_epd_df`, `priority_kappa`,
  `priority_conditional` or `junior_recovery` anywhere in `src/`, `tests/` or
  `docs/`.
* **The four structural facts reproduce exactly.** The EPD table the plan quotes
  came back from a proof of concept before any code was written: Direct
  `0.000408 / 0.002726 / 0.006284`, Assumed `0.008718 / 0.015308 / 0.019599`,
  total `0.007138`, against the plan's `0.000408 / 0.002725 / 0.006283`,
  `0.008717 / 0.015306 / 0.019597`, `0.007137`.
* **The windowed guard needs no special construction.** `_combine_x_min` is set
  only on the auto-size path (`bs == 0`), so a fixture with an explicit `bs` is
  never windowed and `use_roll` is exactly
  `port._signed() or port._combine_x_min is not None`.
* **No TODO item existed** for `[Priority-Junior-Leg]`; one was added as a DONE
  entry rather than opened and immediately closed.

### Divergences

1. **`junior_recovery` takes an optional `state`.** The plan's signature is
   `junior_recovery(port, unit)`. Building the leave-one-out transforms is the
   expensive part and `priority_df` loops over every unit, so the signature
   became `junior_recovery(port, unit, state=None)` and the construction moved
   into a new sibling, `priority_state(port, caller=...)`, which also owns the
   guards. `priority_df` builds the state once and passes it down. The
   single-owner property the plan asked for is intact: one convolution, in one
   place.

2. **A third guard: the frame must be the independent combine.**
   `priority_state` reconstructs `p_total` from the units' persisted
   `ftagg_density` and raises `ValueError` if it does not match
   `density_df['p_total']` to the validation noise floor (measured difference on
   the normal update path: `0.0`). The plan did not call for this. The hazard is
   real and was found while reading `_portfolio_sample.swap_density_df`, which
   replaces `density_df` and recomputes the unit transforms **locally** without
   touching `agg.ftagg_density`: reading the persisted transforms on a swapped
   or sample-based frame would silently compute every column against the wrong
   senior pool. The check turns the independence assumption the docstrings state
   into something measured.

3. **The junior leg is computed by convolution, not by subtraction, and the
   docstring says why.** The identity
   `min(X_i, (a - X_{-i})^+) = min(X, a) - min(X_{-i}, a)` holds pointwise, so
   `lev_total - lev_{-i}` would be one `ift` as well. It cancels
   catastrophically when the junior unit is small against the senior pool, which
   is exactly the interesting case, so the convolution stays and the identity
   serves as the test.

4. **The `padding >= 1` justification is stated more weakly than the plan states
   it.** The plan says "`p_{-i}` has support inside `[0, n)`". It does not,
   strictly: each unit's pmf occupies `[0, n)` of a buffer of length
   `n << padding`, so the pooled senior density genuinely reaches above `n`, and
   the circular convolution at index `a < n` therefore picks up
   `Σ_{y > a + n} lev_i(a - y + 2n) p_{-i}(y)`. That term is senior-pool mass
   above the top of the grid, so it is the same far-tail aliasing the combine
   itself accepts, and it is bounded by what the reproduction check in
   divergence 2 measures. The `Notes` say that rather than the stronger claim.

5. **`assets` is joined by a `p=` alternative.** The reading methods take
   `assets=None, *, p=None` and require exactly one, matching
   `Portfolio.reins_price_df` and `allocation_bounds`. The plan's signature
   named only `assets`.

6. **No `.rst` edit was needed in the reference.**
   `docs/3_reference/3_x_Portfolio.rst` is `autoclass`-driven, so the new
   methods appear there without a per-method entry. The narrative paragraph went
   into `docs/2_aggregate_overview/pipeline-portfolio.rst` instead. No stale
   `:meth:` references, since nothing was renamed or deleted.

7. **Test tolerances are looser than the plan's in one place, and the reason is
   pre-existing.** The junior footing closes to `1e-9` relative as the plan
   predicted. The `sum_i exa_i = lev_total` regression guard needs `1e-6`
   relative, and the pointwise `senior >= equal >= junior` bracket needs one
   part in `1e6` of the unit's own expected loss: the equal leg divides kappa
   through by the total and so carries share noise two orders of magnitude above
   the junior convolution's. That is the state of `exa_{unit}` today, not
   something this plan introduced.

### Phase 1, `1.0.0a394`, the three legs and the deficit table

`_portfolio_density.priority_state` and `junior_recovery`;
`Portfolio.priority_df` (cached, invalidated in `update`), `priority_epd_df` and
the private `_priority_assets` resolver. Plan tests 1, 2, 3, 5 and 6, plus the
shape / cache / alias / argument-contract cases and the stale-frame guard:
`tests/test_priority.py`, 18 cases. Fast gate green, 5,388 passed.

### Phase 2, `1.0.0a395`, the conditional ladder and the conditional law

`_portfolio_density.priority_conditional_mean` and `priority_conditional_law`,
with `GridDistribution` newly imported there (it is a leaf, so the module keeps
its near-leaf property); `Portfolio.priority_kappa` and
`Portfolio.priority_conditional`. Plan test 4 plus the bracket, the
below-assets, hand-computed and guard cases on both, and a cross-check of the
law's mean against the ladder, which compares two genuinely different routes
(direct slices of the joint against two convolutions).
`tests/test_priority.py` is 31 cases. Fast gate green, 5,401 passed; full gate
`-m 'slow or not slow'` green, 5,608 passed in 136 s.

Two further divergences, both recorded at the moment they were made:

8. **`priority_conditional` was kept, not dropped.** The plan marked it
   optional. It is thirty lines on top of machinery phase 2 already builds, and
   the companion note's cloud figure consumes it. It also earns its keep as a
   test: the law's mean reproduces the ladder's `kappa_junior` to seven figures
   by a completely separate route.

9. **The ladder blanks at machine epsilon, not at the validation noise floor.**
   The plan says `NaN` where `p_total` is below the validation noise floor
   (`1e-12`). Measured: that drops rows carrying `3.5e-4` of `E[D_junior]` on
   the smaller unit, so the plan's own acceptance check (the ladder integrates
   to `E[X_i] − ex_junior_i(a)` to `1e-4` relative) **fails** at `1e-12` and
   passes comfortably at `eps`: `-4.2e-6` for the smaller unit, `-9.3e-8` for
   the larger. Machine epsilon is also the cut `add_exa` already applies to
   `exeqa` in the same module, so this is the consistent choice as well as the
   accurate one. The conditional mean is a ratio of two sub-epsilon numbers out
   there, but its product with `p_total` is still material, and the docstring
   says so for a reader integrating the ladder by hand.

### Left for the author

* **`docs/update_extract_bib.py` defaults to the KOLMOGOROV path.** Its
  `--uber` default and docstring both name `C:/S/TELOS/Biblio/uber-library.bib`,
  which no longer exists; `CLAUDE.md` says the library is at
  `D:/Projects/Biblio/uber-library.bib`. `docs/extract.bib` was regenerated here
  with `--uber D:/Projects/Biblio/uber-library.bib`, which added `Butsic1994`
  and incidentally picked up a missing comma in the `Major2026` author field.
  Changing the default is a one-line, no-behavior-change tidy and so does not
  bump; it was left uncommitted rather than swept into either bump.
* **The docs have not been rebuilt.** Per `CLAUDE.md`, the build is not part of
  a verification cycle. The new material is one section of
  `docs/2_aggregate_overview/pipeline-portfolio.rst` plus the three
  `autoclass`-collected methods, and it cites `Butsic1994`, now present in
  `docs/extract.bib`.
* **The open legal question stands unanswered, as the plan directs.** Whether a
  ceding insurer's claim sits with policyholders or drops to general creditors
  varies by state adoption of the model act. Nothing written here counts states.
