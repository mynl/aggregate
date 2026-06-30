# Plan — `create_pnl`: a domain-agnostic P&L API over the pushforward engine

> **Status: DRAFT — not executed.** Author insight (2026-06-30): DecL is
> *expressly* an insurance language and will **not** soon express power
> generation, crop revenue, or ALM. The cross-domain vehicle is the **Python
> `PnL` API**, not the grammar. So: keep DecL insurance-only (**no DecL changes
> in this plan**), and build a small, honest, domain-agnostic constructor —
> `create_pnl(source, …)` — that any domain can call, with insurance as one
> *caller* among many.
>
> This **supersedes the a121 leg wrappers** (`legs.py`, `_insurance_view.py`):
> the pushforward *engine* stays, but `create_pnl` is a better surface than
> `Leg` / `LegSet` / `InsuranceView`, so those come out. The a121 rewiring was
> not wasted — it proved reinstatement and variable rating already share one
> compute path, which is exactly what makes this refactor safe.

---

## The one idea

A P&L position is **money in minus money out**, as a function of a random state:

```
result(state) = consideration(state) − obligation(state),   state ~ source   [role='sell'; 'buy' flips it]
```

Group each map's values over the source by output value and you get three
`GridDistribution`s — consideration, obligation, result — exact, with all their
moments. **That's the whole object.** Gross/ceded/net insurance is one *preset*
of leg names; the kernel never needs to know the word "ceded."

Two things make it cross-domain where the a121 `InsuranceView` was not:

1. **Labels are data, not code.** The caller names the legs (`'revenue'`,
   `'fuel cost'`, `'operating margin'`) via the **dict keys**; there is no
   built-in `perspective`/`category`/`kind` taxonomy.
2. **The constructor takes an opaque source slot.** A GD, an `Aggregate` (with or
   without reins), a `BivariateDistribution`, or a bare `(values, probs)` pair —
   the maps decide what it means.

---

## Philosophy: a P&L is a lightweight value object

A P&L **consumes and throws away** its stochastic engine — it neither *is* an
`Aggregate` (not a subclass) nor *has* one (no retained reference). `create_pnl`
reads the probabilities and the component values off whatever engine you hand
it, builds the three `GridDistribution`s **eagerly**, and **discards the source**.
A P&L is then just three signed distributions + their exact moments + the
caller's labels: an *accounting* object, light enough to build in two lines
without ever touching DecL. Deliberate consequences:

- **No `self.agg`.** The obligation is a GD, not a live aggregate. *(Breaking: the
  current `PnL.agg` and the `value_type`-read-from-agg surface go — audit `p.agg`
  usages; a caller who wants the engine kept it themselves.)*
- **No `update()`.** The GDs are fixed at construction; you update the *engine*
  first, then snapshot. A reins P&L snapshots **all** its legs eagerly (so the
  cession tower needs no live agg either).
- **No `validation_df`** (see §1) — a P&L has nothing of its own to validate.

---

## §1 The core API and the FCC report contract

```python
create_pnl(source, *, consideration, obligation, role='sell',
           result_name='result', name=None, probs=None) -> PnL
```

- **`source` — an opaque slot.** Whatever you pass is handed to the component
  maps; `create_pnl` itself only needs a **probability vector** off it. The maps
  decide what `source` *means*: pass a `GridDistribution` and read `source.x`; an
  `Aggregate` (with or without reins) and read `density_df` / `reins_density_df`;
  a `BivariateDistribution` and read both axes; or a bare `(values, probs)` pair.
  Probabilities **default from the slot** (a GD's `p`, an aggregate's `p_total`,
  a joint's flattened mass, the second element of a pair) and can be overridden
  with `probs=`.
- **`consideration`, `obligation` — magnitudes; the role supplies the sign.**
  Each is a constant, a callable `f(source) -> value array`, or an ordered dict
  `{name: …}` of such (multi-component — `obligation={'loss': …, 'expense': …}`
  so the parts surface as **rows**). They are **non-negative magnitudes**; you
  never hand-flip a sign (this kills the awkward ceded-loss negation).
- **`role` — are you *buying* or *selling* the obligation?** `'sell'` (you're
  paid the consideration and owe the risky leg): `result = consideration −
  obligation` — the gross insurance book (`+premium − loss`). `'buy'` (you pay the
  consideration and receive the risky leg): `result = obligation − consideration`
  — every reinsurance leg (`+recovery − ceded premium`). The result is always a
  **payoff**, and the cession sign-flip falls out of the role, not a hand edit.
  (Long/short is ambiguous — "long the risk" vs "long the position" — which is
  what tangled the `cons−obl` call; buy/sell *the obligation* is unambiguous.)
- **Names come from the keys.** A dict component is named by its key; a bare
  `consideration=100` is named `'consideration'`; the net is `result_name`; the
  object is `name`. No separate `name_info` — keys *are* the names, and a keyed
  component is never renamed.

`create_pnl_tower(legs, *, delta_names=None, name=None) -> PnLTower` stacks legs
into the inuring waterfall — every **one-step delta** *and* a **final total
delta** (see §3).

**How a leg becomes a distribution — exact, no rebucketing.** Evaluate each
component map over the source's atoms, then **group by output value and sum
probability** → a `GridDistribution` on the (irregular) set of realized values,
`bs=None`. Because nothing is fed to an FFT, there is **no regular grid and no
rebucketing** — so the GD is *exact* and is the single source of every moment
**and** percentile (no EX-vs-Est split — the other reason a P&L needs no
`validation_df`). `result` is `cons − obl` (per role) computed **per atom, then
grouped** — never arithmetic on the two marginal GDs (they are dependent; the
per-atom difference carries the covariance, so means add and SDs don't, for
free).

### The FCC report contract (four reports, **no** `validation_df`)

`validation_df` exists where we hold *exact* moments and check a **simulation**
against them. A P&L has no simulation of its own — it inherits whatever the
upstream engine already validated and is otherwise pure deterministic accounting.
So a P&L deliberately has **no `validation_df`**. The four reports are:

**1. `summary_df` — the short headline** (like today's no-reins `PnL.summary_df`).
Rows, in order: each **consideration** component (named *and ordered by the keys*
of the `consideration` dict), **Total consideration** (only if >1 component);
each **obligation** component (keys of `obligation`), **Total obligation** (only
if >1); the **result** (named by `result_name`). Columns:
`EX`, `% Consid`, `SD`, `CV`, `Skew`, `P01`, `Median`, `P99` — where `% Consid` is
the row's `EX` as a fraction of `E[Total consideration]` (the loss-ratio / margin
family: an obligation row reads like an LR, the result like a margin %).

**2. `stats_df` — the detailed table** (the `stats × legs` companion, a slice of
today's `gcn_df`). Every statistic (`EX`, `SD`, `CV`, `Skew`, and the *full*
percentile ladder) for every leg, **each paired with the same value as a fraction
of a fixed consideration scale** — so two value columns per leg
(`value`, `% of <scale>`). The scale **defaults to Total consideration** and may
be set to any single consideration element via `scale=`. **The scale must be
deterministic** (`SD == 0`): scaling a *distribution* by a stochastic
consideration would need the joint we deliberately never form, so a non-fixed
scale **warns** and divides by the scale's `EX`. (Use it to read everything "per
unit of a fixed gross premium".)

**3. `density_df` — a dict of GDs, not a staple** *(the hobgoblin question,
settled your way).* `density_df` is an ordered dict `{leg_name: GridDistribution}`
— consideration(s), obligation(s), result — each GD a Series-like `outcome → prob`
on its **own** (irregular, exact) grid. The main customer is `plot`, which just
iterates and calls each GD's Series view, so there is **no lossy staple** onto a
shared grid. (GD gains a thin `to_series()` / `_repr` for this; "a foolish
consistency…" — we do *not* force the `Aggregate.density_df` single-frame shape
onto a P&L.)

**4. `plot` — net first.** Two panels: the **density** and the **distribution
(CDF)** of the net result. Component overlays and the cession waterfall come later
and live on the **tower** (§3), not the single leg.

(The exact percentile ladder and any further ratio rows are pinned in
`[Reporting-Guidelines]`, the next plan; this plan fixes the structure above.)

## §2 What it sits on — and what it replaces

**Kept:** `GridDistribution` (the value type — now carrying everything, since the
leg distributions are exact), `BivariateAggregate` / `occ_bivariate` (joint
construction), and the scatter engine (`pushforward_1d` /
`BivariateDistribution.pushforward`) — but **`create_pnl` no longer uses the
scatter**. A leg is an exact **group-by-value, sum-prob** over the source atoms
(§1); the regular-grid scatter survives only for the cases that genuinely want a
uniform grid (feeding a further convolution) or optional compaction of a very
large support. `transformed_moments` is subsumed — the exact GD reports its own
moments.

**Removed (superseded by `create_pnl` / `create_pnl_tower`):**

| Out | Why |
|---|---|
| `legs.py` — `Leg`, `LegSet`, `GraphSource` | components + the raw 1-D `(scenarios, probs)` path replace them; `GraphSource`'s `Y=κ(X)` collapse is just "compute κ inside the obligation map over the 1-D gross grid" |
| `_insurance_view.py` — `InsuranceView`, `perspective_of`/`category_of`/`kind_of`, `LEG_LABELS` | vocabulary becomes the dict keys / labels (data); the gross/ceded/net leg set becomes a small **insurance preset** function |
| `_pnl.py` — `gcn_assemble_column`, `PnL._gcn_perspective_rows`, `_gcn_marginal`, `_gcn_magnitudes`, `_gcn_impact`, `_quantile` | the hand-rolled GCN column builder → `create_pnl_tower` |
| per-class `gcn_df` bodies in `reinstatement.py` / `variable_rating.py` | both become **tower builders** (§3) |
| `tests/test_legs.py`, `tests/test_insurance_view.py` | replaced by `tests/test_create_pnl.py` |

**PnL surface changes (deliberate, breaking):** with the engine discarded
(Philosophy), `PnL` also loses **`.agg`**, **`update()`**, and **`validation_df`**;
`pnl_df` → `density_df` (net). `gcn_df` is no longer a property the single net P&L
hand-rolls — it is the **tower** view: a reins P&L snapshots its gross/ceded/net
legs eagerly at construction and exposes the waterfall as `.gcn_df` (§3), so
`build('pnl …').gcn_df` stays, same output, different home.

Net module change: **−2 modules** (`legs`, `_insurance_view`); the P&L surface
consolidates into `_pnl.py` (+ a small examples/test module). DecL: **0 changes**
— the underwriter glue that builds `pnl` objects calls `create_pnl` /
`create_pnl_tower` under the hood.

## §3 The insurance side, rewired (and the cascade made incremental)

`PnL.gcn_df` and both analysis classes stop hand-rolling columns and instead
**build a tower of legs**. A leg is one `create_pnl`; the tower stacks them and
emits the running net plus the inuring **benefit deltas** — which *is* the GCN
waterfall, now general and incremental:

```python
gross = create_pnl(agg, role='sell',                       # write the risk
                   consideration={'premium': 1000},
                   obligation={'loss': loss_map, 'expense': exp_map},
                   result_name='underwriting result')

occ_re = create_pnl(agg, role='buy',                       # buy protection
                    consideration={'ceded premium': 120},
                    obligation={'occ recovery': occ_recovery_map},
                    result_name='occ cession')

agg_re = create_pnl(agg, role='buy',
                    consideration={'ceded premium': 30},
                    obligation={'agg recovery': agg_recovery_map},
                    result_name='agg cession')

# incremental: add a layer, read its benefit; the list also yields the total
create_pnl_tower([gross])
create_pnl_tower([gross, occ_re],         delta_names=['occ benefit'])
create_pnl_tower([gross, occ_re, agg_re], delta_names=['occ benefit', 'agg benefit'])
```

All magnitudes are **positive**; the role does the signs (`sell` gross:
`+premium − loss − expense`; `buy` reins: `+recovery − ceded premium`). The tower
columns are each leg + the **running net** after it + each **one-step delta**
(the layer's benefit) + a **final total delta** (net-of-all vs gross), and the
running net adds because means add. **"Try this or that"** is literally adding or
dropping a leg from the list — the cascade is a one-line edit, not a bespoke
5-column code path. The final frame equals today's `build('pnl …').gcn_df`.

- **`ReinstatementAnalysis`** becomes a builder: its `_leg_functions` shrink to
  *component maps* (`loss`, `recovery = A(R)`, `ceded premium = D + h(R)`, the
  agg-cover legs) fed to `create_pnl` + `create_pnl_tower`. `summary_df` /
  `tail_df` / `validation_df` / `plot` / `bs_*` **stay** but read the new legs.
- **`VariableRatingAnalysis`** likewise — and it *gains* `summary_df` / `tail_df`
  for free (it inherits the `PnL` exhibits instead of having none).
- **`PnL.gcn_df`** = the insurance preset over `create_pnl_tower`.

## §4 Worked examples

> Sketches double as the `tests/test_create_pnl.py` spec. Maps are vectorized;
> all amounts are magnitudes with the sign set by `role`.

### Part A — insurance (the parity set; build first, must reproduce `gcn_df`)

The refactor's regression bar — each is `create_pnl` legs fed to
`create_pnl_tower`, and `build('pnl …').gcn_df` must stay byte-identical:

- **Basic gross** — `sell`: `consideration={'premium': P}`,
  `obligation={'loss': …, 'expense': …}`, result the underwriting result. One
  leg, no tower.
- **Gross + fixed-cost reinsurance** — gross `sell` + a `buy` leg (ceded premium
  / recovery); the 3-column tower *is* today's `gcn_df`.
- **Variable clauses** — each fills exactly **one component** with a φ-map over
  the source (the only feature-specific step, as today):
  - *swing* → ceded-premium component `= φ_swing(ceded loss)`
  - *slide / profit commission* → commission component `= φ(ceded LR)·P_C`
  - *loss corridor* → ceded-loss component `= φ_corr(ceded LR)·P_C`
  - *reinstatements* → a `buy` leg whose **source is the `(L,R)` joint**:
    ceded-premium `= D + h(R)`, ceded-loss `= A(R)`
- **Gross + retro rating** — gross `sell` with the premium component
  `= φ_retro(net account loss)`.

### Part B — cross-domain (new; proves the API is not insurance-shaped)

Each is one `create_pnl(source, consideration=…, obligation=…, role=…)` whose
*result* genuinely needs the joint.

#### B1 — the six coupled-driver cases

1. **Crop revenue insurance** — `obligation = max(guarantee − Y·P, 0)` over a
   joint `(Y, P)` with **negative** dependence (the natural hedge). Tests that
   the marginals would overstate risk.
2. **ALM / surplus** — `result = A(equity) − L(rates)` over a coupled
   `(equity, rates)` joint; the correlation drives funded-status risk.
3. **Parametric basis risk** — `result = payout(index) − actual_loss` over a
   coupled `(loss, index)` joint; the residual is unreachable from the loss
   marginal.
4. **Reinsurer credit (wrong-way)** — `recovery = ceded · 1{solvent}` over an
   *adversely* coupled `(loss, solvency)` joint; the cover evaporates in the bad
   tail.
5. **Spark spread [financial 1]** — `result = max(Power − HR·Gas, 0)` over a
   coupled `(Power, Gas)` joint (the dispatch option).
6. **Dual-index weather derivative [financial 2]** —
   `payout = rate · HDD · GasPrice` over a positively-coupled `(HDD, price)`
   joint; cold-and-dear is the product.

#### B2 — Electricity merchant generator (the author's example, refined)

A fossil station on a grid supply contract; `(T, W)` = (temperature, wind) joint.
Residual (non-green) volume `v(T,W) = d(T)·(1 − g(W))`; price clears on residual
demand, so `c = c(T,W)` (the refinement — price carries the wind channel, the
source of the upside tail); fuel cost `f` per unit; the unit dispatches only when
in the money:

```python
op = create_pnl(
    weather_biv,                                   # (T, W) BivariateDistribution
    consideration=lambda t, w: v(t, w) * c(t, w),  # revenue (state-dependent!)
    obligation   =lambda t, w: v(t, w) * f,        # fuel cost
    names={'+':'revenue', 'obligation':'fuel cost', 'result':'operating margin'})
op.summary_df       # E/SD/CV/Sk + percentiles of revenue, fuel cost, margin
```

The dispatch option is the one-line variant
`result = v(t,w) · max(c(t,w) − f, 0)` (a per-state leg). This is the worked
proof that **consideration is itself a function of the state** — the assumption
insurance hard-codes away.

#### B3 — Fixed cost per count + variable cost (the `dsev[1]` count-axis trick)

You pay a **fixed cost each time you fire up a generator** (per *event*) **plus a
cost for the amount used** (per *unit*). Make a bivariate whose two axes are the
**event count `N`** and the **amount `X`** by giving one unit the degenerate
severity `dsev[1]` — each event contributes 1, so that unit's aggregate **is the
count**:

```python
# one axis is the count N (severity always 1), the other is the amount X
gen = build('bivariate Plant 5 claims '
            'agg Starts dsev [1] '                  # aggregate == N  (the count)
            'agg Fuel  sev lognorm 100 cv 1 '       # aggregate == X  (amount used)
            'copula independent poisson')           # (illustrative; shared count)

plant = create_pnl(
    gen.bivariate,                                  # joint (N, X)
    consideration=PRICE,                            # contract revenue
    obligation=lambda n, x: STARTUP * n + FUEL * x, # fixed-per-start + per-unit
    names={'+':'contract', 'obligation':'generation cost', 'result':'margin'})
```

This shows the bivariate machinery isn't only for two *losses* — `dsev[1]`
promotes the **claim count** to a first-class axis, so any "per-occurrence fixed
charge + per-amount charge" structure (deductibles per claim, claims-handling
expense, policy fees, ceded reinstatement counts) becomes a plain `f(N, X)`.

## §5 Names to vet (per CLAUDE.md naming rule)

- `create_pnl` (free function) vs the existing `Aggregate.make_pnl` — keep
  `make_pnl` as object sugar that calls `create_pnl`; the free function is the
  general entry. One concept, two entry points (like `build` vs methods) — called
  out, not a synonym for the *object*.
- `create_pnl_tower` → returns a `PnLTower`. Confirm no collision (none today).
- `consideration` / `obligation` — the magnitude args; `role` (`sell`/`buy`)
  the sign; `result_name` the net's label. Confirm none shadow a `PnL` member.
- `PnL` is **reshaped** into a value object (component GDs + exact moments +
  labels, no engine reference). Audit every existing member: keep `evaluate`
  / `plot` / `q` / `cdf` / `sf` / `prob_loss` (read the result GD); rename
  `pnl_df` → `density_df` (net); **remove** `agg` / `update` / `validation_df` /
  the hand-rolled `_gcn_*`. `scale=` is the `stats_df` parameter naming the fixed
  consideration denominator (default `'Total'`).

## §6 Workstreams and order

1. **[Core-API]** — `create_pnl` + the reshaped `PnL` value object (components →
   exact group-by GDs + the four FCC reports, `role`-signed). Unit-test on a raw
   `(values, probs)` and a GD slot. *Pure addition; nothing removed yet.*
2. **[Tower]** — `create_pnl_tower` + `PnLTower` (running net, one-step deltas,
   total delta).
3. **[Insurance-Parity]** (Part A — the regression bar) — the gross/ceded/net
   **preset** + rewire `PnL.gcn_df`, then `ReinstatementAnalysis` /
   `VariableRatingAnalysis` as tower builders (keeping their bespoke exhibits).
   `build('pnl …').gcn_df` **byte-identical**; the full reinstatement /
   variable-rating suites stay green. **Delete** `gcn_assemble_column` + the
   hand-rolled `_gcn_*` here.
4. **[Cross-Domain]** (Part B — proves generality) — `tests/test_create_pnl.py`:
   the six coupled-driver cases + electricity + the `dsev[1]` count-cost trick.
5. **[Rip-Out]** — delete `legs.py`, `_insurance_view.py`, `GraphSource`, and the
   two a121 test modules; confirm `GridDistribution` + the joint constructors
   remain and the suite is green.

Each workstream bumps `1.0.0a*`; behavior-frozen where it touches insurance
output (`gcn_df` unchanged) **except** the deliberate new `summary_df` structure.

## §7 Tests / regression bar

- `tests/test_create_pnl.py` (new): the §4 cases — degenerate-vs-joint agreement,
  means-add-SDs-don't, exact-vs-rebucketed moments, the count-axis trick.
- **FCC contract:** `summary_df` rows (named/ordered by the dict keys, totals only
  when >1) and its 8 columns; `stats_df` value/`%-of-scale` pairing; the
  **non-fixed-scale warning** (a stochastic `scale=` warns and divides by `EX`);
  `density_df` is the net; **no `validation_df`** on a P&L.
- Insurance parity: `build('pnl …').gcn_df` byte-identical to today across the
  reinstatement / variable-rating suites (the tower reproduces the cascade).
- `VariableRatingAnalysis` *gains* `summary_df` / `tail_df` (new green tests).
- `uv run pytest` green; `regen_features.py` audit re-run and `FEATURES.csv`
  updated **by hand or with `newline=''`** (never `csv.DictWriter`→`write_text` —
  it doubles line breaks on Windows): the `PnL` row gains the four-report surface
  and loses `agg` / `update` / `validation_df`; the analysis columns drop
  `view` / `distributions`.

## §8 Housekeeping

Plan-based change → bump `1.0.0a*`; `CHANGELOG.md` section per workstream;
`dev/TODO.md` active sequence updated (**[PnL-API]** lands **before**
**[Reporting-Guidelines]**); move this plan to `dev/done/` at close. **Docs:** the
§4 examples graduate to a "P&L as a pushforward" docs page (generic
`state → result` notation, the cross-domain cast). No DecL doc changes.

## Relation to the active sequence

- **Replaces** the would-be `[PnL-From-Bivariate]` and folds it into
  `[PnL-First-Class]` → renamed **[PnL-API]**; it delivers the missing generic
  entry point *and* the first-class surface in one pass.
- **Feeds** `[Reporting-Guidelines]`: the single-column `summary_df` and the tower
  are the report shapes the guidelines then standardize (rows fixed, columns
  pure). So `[PnL-API]` is now **first** in the active sequence.
- **Unblocks** `[Portfolio-of-PnL]`: a book is a `create_pnl_tower` (or a
  `create_pnl` over a portfolio total) — the constant-consideration v1 is nearly
  free once legs compose.
- DecL untouched; `[Plotting-Punchups]` and the independent fixes unchanged.
