# Plan — property-cat reinstatement premiums: stochastic ceded premium as a first-class GCN analysis

> **Status: DRAFT — not executed.** The last new feature for v1.0. Models occurrence
> reinsurance with paid/free reinstatements, where **ceded premium is stochastic**
> (reinstatement premium `h(R)` is a function of unlimited annual occurrence recovery
> `R`), and produces complete gross/ceded/net underwriting exhibits from one FFT2 joint
> distribution by deterministic pushforward.
>
> The math, worked examples, and numerical/audit design live in **`dev/pre-plan-reinstatements.md`**
> (spec) and **`dev/reinstatements.md`** (background, correct arithmetic). This plan does
> **not** repeat them; it adds the four things the pre-plan omits — **DecL**, **PnL/cedednet
> reuse**, **naming**, **first-class-citizen surface** — plus the execution order, the test
> wrinkles, and the corrections to fold in.

---

## Decisions locked (author, 2026-06-27)

1. **DecL: a `pnl` block carries the reinstatement clause.** Reuse the **existing `pnl`
   output kind** (which already has premium − loss and `occ_reins` in its production). The
   reinstatement terms decorate a single occurrence layer; gross premium and the consideration
   leg come from the `pnl` premium clause. This is the "wow" — one declarative block does
   gross volume, the cat layer, the reinstatement schedule, and `build()` hands back a `PnL`
   whose `gcn_df`/`summary_df`/`plot` are the exhibit:

   ```
   pnl Cat
       10000 premium - 85% lr                              # gross volume + expected loss
       sev lognorm 50 cv 3
       occurrence net of
           95% po 100 xs 100
               reinstatements 1 free and 1 at 50% and 2 at 100% deposit 10
       poisson
   ```

2. **Exhibit: the returned `PnL`'s own `gcn_df` IS the exhibit.** Reuse the existing
   doubly-additive Gross/Ceded/Net × Consideration/Obligation/Margin table and its SD-not-CV
   convention; the only delta is that the Consideration column becomes a **distribution** on
   the Ceded/Net rows (stochastic `D + h(R)`), so add spread (SD/CV) columns. Do **not** invent
   a parallel exhibit format. `ReinstatementAnalysis` is the **engine** (joint `(L,R)`,
   pushforward, the 11 legs, audit) that the `PnL` delegates to; it is not a second face.

3. **Recovery cap `(m+1)y` lives in `ReinstatementTerms`.** The source aggregate must carry
   the occurrence layer (`occ_reins`) but **not** an aggregate cap; `agg_reins` on the source
   is a hard error (would double-count the cap).

4. **Names:** `Aggregate.reinstatement_analysis() -> ReinstatementAnalysis` (the programmatic
   path); the DecL `pnl` block is the headline path. Full words (house default); `reins` stays
   canonical for rein**surance** — "reinstatement" is a distinct word and is spelled out.

**Open decisions resolved by the author's syntax:**

- **`[decl-rol]`** — `deposit` accepts **currency** (`deposit 10`) **or** a rate-on-line
  percent (`deposit 12% rol` → `D = 0.12·limit`). Base rate `r = D / limit` (linked); no
  independent rol for v1.0.
- **`[decl-return]`** — `build()` returns a **`PnL`** (analysis-backed); gross premium = the
  `pnl` consideration.
- **`[share]`** — non-100% occurrence share is **allowed** (`95% po 100 xs 100`); validate
  exactly one reinstated layer.
- **Missing `reinstatements` clause = free + unlimited reinstatements** (an ordinary
  occurrence layer; existing behavior). The `reinstatements [...]` clause is the opt-in to
  finite, paid reinstatement capacity.

Plus the pre-plan §24 decisions: payoff sign convention throughout (exhibits add); percentile
rows show the "bad" (adverse) tail; deposit entered in currency (or `% rol`); custom premium
functions must be vectorized (`ValueError` if not); `BivariateDistribution.pushforward` is
**public**.

---

## What already exists (ground-truthed — do not rebuild)

- **The canonical `(L, R)` joint is already one call.** `Aggregate.occ_bivariate(views=('gross','ceded'))`
  (`_aggregate.py:1027`) returns a `BivariateAggregate` (mode `netceded`) whose `.bivariate`
  property (`bivariate.py:1201`) is a `BivariateDistribution` (`bivariate.py:1765`). The gross
  view is the identity, the ceded view is `occ_ceder` summed per occurrence — i.e. **`R` is the
  unlimited occurrence recovery, before any annual cap.** FFT2 with shared frequency is already
  wired (`build_netceded_joint`, `bivariate.py:269`, FFT2 at `:412`). **Pre-plan "Stage 1" is done.**
- **`BivariateDistribution`** holds `density` (matrix), axis grids (stored positionally as
  `.ceded`, `.net`), `bs_ceded`/`bs_net`, and `meta['axis_names']`. In `netceded` mode the two
  bucket sizes are **forced equal** (comonotone coupling) — which is exactly right here (`R ≤ L`
  on a shared grid). It has `marginals`, `moments`, `corr`, `contour`. It has **no** `pushforward`.
- **`make_ceder_netter`** (`_reinsurance.py:83`) → piecewise-linear `ceder`/`netter`;
  `apply_occ_reins`/`apply_agg_reins` populate `sev_density_{gross,ceded,net}` /
  `agg_density_{gross,ceded,net}`. Occurrence and aggregate reins are independent stages.
- **`PnL`** (`_pnl.py:33`) already has `gcn_df` (`:270`) — the doubly-additive
  Gross/Ceded/Net × Consideration/Obligation/Margin table — and `summary_df` (`:225`) reporting
  **SD not CV** near break-even (`:231`). This is the exhibit template (decision 2).
- **DecL has no per-layer premium today.** `occ_reins`/`agg_reins` are pure loss structure
  (`(share, limit, attach)` tuples + `occ_kind`/`agg_kind`); premium enters only via exposure
  (`exp_premium`/`exp_lr`) or a `pnl` consideration. **Reinstatement premium is the first
  pricing term embedded in the loss grammar** — handle with care.

---

## The accounting maps onto `gcn_df` exactly (decision 2)

With payoff signs (premium received +, paid −; loss as a negative obligation; recovery +), for
each grid point `(L, R)`, `A = terms.recovery(R)`, `RP = terms.reinstatement_premium(R)`,
`D = terms.deposit`, `P_G` = gross premium:

| | Consideration (premium) | Obligation (loss) | Margin |
|---|---|---|---|
| **Gross** | `P_G` (fixed) | `−L` | `P_G − L` |
| **Ceded** | `−(D + RP)` | `+A` | `A − D − RP` |
| **Net** | `P_G − D − RP` | `−(L − A)` | `P_G − D − RP − L + A` |

Rows add, columns add — **in expectation**. The only thing reinstatements add over `PnL.gcn_df`
is that the Consideration column is now a **distribution** on the Ceded/Net rows. So the exhibit =
`gcn_df` means + spread (SD/CV) columns + a note that means add but SDs/percentiles do not.
`U_ceded = A − D − RP` is the **reinsurer's** underwriting result — the same object serves cedant
and reinsurer.

---

## Workstreams

### `[decl]` — DecL `pnl` block + reinstatement clause (the new, riskiest piece)

**Syntax** — reuse the existing `pnl` output kind; decorate a single occurrence layer:

```
pnl Cat
    10000 premium - 85% lr            # gross volume; expected loss = 8500
    sev lognorm 50 cv 3
    occurrence net of
        95% po 100 xs 100             # one reinstated layer (share allowed)
            reinstatements [0 0 .5 1 1] deposit 10
    poisson
```

- The `pnl` block already parses `name <numbers> premium - <loss> … occ_reins freq …` and builds
  a `PnL(agg, consideration=...)`. **Gross premium `P_G` = the stated premium** (10000);
  expected loss from the `lr`/loss clause — both already handled by the `pnl` path. No new
  premium concept at the block level.
- `reinstatements <n> free and <n> at <p>% and …` — the **human form** (treaty language),
  chained with `and`. `one free and three at 100%` → α `[0 1 1 1]`;
  `one free and one at 50% and three at 100%` → `[0 .5 1 1 1]`;
  `1 free and 1 at 50% and 2 at 100%` → `[0 .5 1 1]`. `free` = `at 0%`; `at p%` is a multiplier of
  the base rate `r`. Counts accept either **digits** (any count) **or** the number-words
  `one…five` as lexer sugar mapping to ints (digits stay canonical; `7 at 100%` is still digits).
  **No hard cap** on the total count.
- `reinstatements [α₁ … α_m]` — the **explicit list** escape hatch (same `reinstatement_rates`
  tuple), for programmatic or irregular schedules. `[1 1 1]` = three full-rate; `[0 1]` = one
  free + one full.
- **Omitting the clause = free + unlimited** reinstatements (an ordinary occurrence layer —
  existing behavior); the `reinstatements …` clause is the opt-in to finite, paid capacity.
- `deposit 10` — deposit premium `D` in **currency**, or `deposit 12% rol` → `D = 0.12·limit`.
  Base rate `r = D / limit` (`[decl-rol]` resolved). The reinstatement premium is
  `h(R) = r · Σ αⱼ[(R − bⱼ)₊ ∧ wⱼ]`.
- The occurrence layer stays in `occ_reins = [(share, limit, attach)]` with `occ_kind`
  (`'net of'`/`'ceded to'` both allowed — the analysis computes all views regardless). The cap
  `(m+1)·limit` is **derived in `ReinstatementTerms`**, never added to `occ_reins`/`agg_reins`.

**Grammar (`decl.lark`).** Add an optional reinstatement decorator to a **single `reins_clause`**
(not `reins_list`/`tower` — reinstatements need exactly one layer). New terminals
New terminals `REINSTATEMENTS.2`, `DEPOSIT.2`, `ROL.2`, `FREE.2`, and a number-word terminal
`one…five` (transformer maps to ints — bounded sugar); **reuse** existing `AT` and `AND`.
Two sub-productions both feed the `reinstatement_rates` tuple: the `[α…]` list (a `doutcomes`/list),
and a `<count> free | <count> at <pct>` group chain (counts are a number literal **or** a
number-word; `at <pct>` and the `deposit … % rol` value reuse the existing `_PercentNumber`
machinery). Watch lexer priority on the number-words so they don't shadow identifiers/distribution
names; keep `D3` (grammar-reference-from-`decl.lark`) in mind — the production must be
self-documenting.

**Transformer (`parser.py`).** New spec keys (Lark spec keys are a sanctioned carve-out from the
`reins`-canonical rule, but spell these out): `reinstatement_rates` (tuple of α),
`reinstatement_deposit` (`D` in currency, after resolving `% rol`). Validate at transform/build
time: **exactly one occurrence layer** bears the clause, **no `agg_reins`** when reinstatements
present (decision 3), rates nonnegative.

**`build()` return.** A `PnL` (decision `[decl-return]`). The underlying `Aggregate` carries the
terms as attributes (`self.reinstatement_terms`); the `PnL` detects them and, when present, backs
its `gcn_df`/`summary_df`/`plot` with a `ReinstatementAnalysis` (the stochastic-ceded pushforward).
Do **not** auto-build the (expensive) pushforward on `agg.update()` — build it lazily on first
exhibit access and cache. `pnl.reinstatement_analysis` exposes the engine object for the deep dive.

**Examples to add** to `src/aggregate/agg/test_decl.agg` (per the keep-in-sync rule) under a new
reinstatement section: the block above, the free+50%+100% schedule, an `occurrence ceded to`
variant, and a no-`reinstatements` layer (asserting free+unlimited default). See `[tests]` for
the snapshot wrinkle.

### `[engine]` — generic pushforward on `BivariateDistribution` (public, decision 5)

`bivariate.py`. Add:

- `pushforward(self, function, *, bs=None, log2=None, window=None, scheme='linear', name=None, chunk_size=None)`
  → a 1-D grid-distribution object (reuse the existing `GridDistribution`/grid-dist machinery so
  the result has `q`/`cdf`/`sf`/moments and plays with `Aggregate`/`PnL` reporting). `function`
  receives broadcast axis arrays `function(x_axis[:,None], y_axis[None,:])` and must be vectorized.
  NumPy broadcasting + two `np.bincount` calls for linear rebucketing (pre-plan §9.4); row chunking
  (pre-plan §9.7); retain total mass and report deficit/clipped.
- `transformed_moments(self, function, max_order=2)` → **exact** moments on the source joint grid
  (`Σ pᵢⱼ f(lᵢ,rⱼ)^k`), the headline-driving numbers (pre-plan §15). These are the "EX" column;
  the rebucketed pushforward gives "Est" and the quantiles.
- `[axis-names]` cleanup: `pushforward`/`transformed_moments` must read axis0/axis1 by role, not
  by the misleading positional `.ceded`/`.net` names. Either thread `axis_names` first-class or
  pass explicit grids internally; at minimum document that for `('gross','ceded')` views axis0=`L`,
  axis1=`R`. Small, self-contained.

NumPy only for v1.0; benchmark `2^10…2^12` (pre-plan §19); Numba only if profiling demands and only
behind an identical-results guard.

### `[terms]` — `ReinstatementTerms` (owns recovery + premium; cap lives here)

New module `src/aggregate/reinstatement.py` (no FFT code). Frozen dataclass:

```python
@dataclass(frozen=True)
class ReinstatementTerms:
    limit: float            # y
    rates: tuple            # (α₁ … α_m) price multipliers; len == m
    deposit: float          # D, currency (DecL resolves `12% rol` → 0.12·limit)
```

- Derived: `rol` (`= deposit / limit`, `[decl-rol]` — not an independent field),
  `n_reinstatements` (`m`), `reinstatement_capacity` (`m·y`),
  `total_recovery_capacity` (`(m+1)·y`), `maximum_reinstatement_premium`.
- Methods (vectorized, no Python loops): `recovery(R) = R ∧ (m+1)y`,
  `reinstatement_premium(R) = h(R)` (pre-plan §5.1 tranche sum), `ceded_premium(R) = D + h(R)`.
- Alt constructors `from_tranches(...)` (general widths) and `from_callable(...)` (arbitrary
  vectorized increasing premium function; `ValueError` if it fails to broadcast — decision 4).
- Validation (pre-plan §16.6): recovery finite/nonneg/≤ capacity; `h` finite/nonneg/nondecreasing.

**Name vetting** (CLAUDE.md hard rule): `recovery`, `reinstatement_premium`, `ceded_premium`,
`limit`, `rates`, `deposit`, `rol` are new on a new class — no collision. Stored values are nouns,
actions are verbs; no `__init__` attribute shadows a method.

### `[analysis]` — `ReinstatementAnalysis` engine + first-class surface (PnL-backed)

New class in `reinstatement.py`, owning the joint + pushforward + audit. Two entry points,
**one engine**:

- **Headline (DecL):** the `pnl` block → a `PnL` that builds a `ReinstatementAnalysis` lazily and
  delegates `gcn_df`/`summary_df`/`plot` to it; `gross_premium` = the `pnl` consideration.
- **Programmatic:** `Aggregate.reinstatement_analysis(gross_premium=None, terms=None,
  percentiles=(.90,.95,.99,.995,.996,.999))` (`_aggregate.py` concern):

1. require `occ_reins` present, **reject `agg_reins`** (decision 3);
2. `gross_premium` defaults from `self.exp_premium` if present, else required;
3. `terms` defaults from the DecL `reinstatement_*` attributes if present, else required;
4. build/reuse `occ_bivariate(views=('gross','ceded'))` (cache the joint independently of `terms`
   so multiple schedules reuse one FFT2 — pre-plan §11.4);
5. pushforward the 11 accounting legs; return the analysis.

The `PnL` is the user-facing object; `ReinstatementAnalysis` is its engine (not a second face).
The surface members below live on whichever object is canonical — `gcn_df`/`summary_df`/`plot`
on the `PnL` (delegating), the joint/audit/legs on the analysis — but exposed consistently so
`qd(pnl)` shows the GCN exhibit.

**First-class surface** (FEATURES.csv conventions — note the rename from the pre-plan's `exhibit`/
`audit`/`statistics`/`metadata`):

| Member | Kind | Notes |
|---|---|---|
| `gcn_df` | property | means GCN table, modeled on `PnL.gcn_df` (rows/cols add) |
| `summary_df` | property | headline: premium/CV, loss/CV, UW/SD, percentile rows (pre-plan §13) — reads from `stats_df` |
| `stats_df` | attribute | canonical moment store (single source of truth), as on `Aggregate` |
| `validation_df` | property | the "audit": EX (direct joint-grid moment) vs Est (rebucketed), mass + premium/loss/UW + covariance identities (pre-plan §16) |
| `validation_explanation` | property | long narrative on the audit (no `_description`, mirrors `Aggregate`) |
| `density_df` | property | per-bucket frame for a selected leg / all legs |
| `tail_df(periods=None)` | method | **return-period table of net underwriting loss** — the killer capital exhibit (pre-plan only had it in plotting §18.4) |
| `distributions` | attribute (dict) | the 11 named 1-D legs (`net_uw`, `net_loss`, `reinstatement_premium`, …) |
| `terms`, `source` | attributes | the `ReinstatementTerms` and the `(L,R)` `BivariateAggregate` |
| `reins_description` / `reins_explanation` | properties | treaty narrative: layer, `m`, rol, deposit, capacity, max RP |
| `bs_description` / `bs_explanation`, `bs_window_df` | — | joint-grid sizing audit (reuse) |
| `info`, `_repr_html_`, `plot()` | — | structured summary; Jupyter display; one multi-panel mosaic |
| `qd(analysis)` support | — | quick display |

`plot()` is one mosaic of pre-plan §18: (a) `(L,R)` log-density contour with `R=my`, `R=(m+1)y`,
and premium breakpoints overlaid; (b) `A(R)`, `h(R)`, `D+h(R)`, `D+h−A`; (c) gross vs net UW-loss
survival/return-period; (d) the impact curve `qₚ(W_gross) − qₚ(W_net)`.

### `[pnl-share]` — factor shared 1-D P&L presentation into `_pnl.py`

Extract the reusable bits of `PnL.gcn_df`/`summary_df` (signed-additive table builder, SD-not-CV
margin rule) into helpers that both `PnL` and `ReinstatementAnalysis` call. **Do not** make `PnL`
carry the joint matrix (pre-plan §12). Keep the change surgical; `PnL`'s public surface unchanged.

---

## Corrections to fold in (do not propagate the pre-plan's errors)

- **Pre-plan §7 worked examples are numerically corrupted** (`A=76`, `A=151`/`RP=152`,
  `A=211`/`RP=212` — paste artifacts). The correct values are in `reinstatements.md`
  (`75`; `150`/`10`; `200`/`10`). Regenerate §7 / §21.1 from `reinstatements.md` **before** these
  become test fixtures.
- **Pre-plan §13 Impact-column directions are inconsistent** (some rows Net−Gross, some Gross−Net,
  CV row a third rule). Derive Impact **uniformly** from the `gcn_df` payoff-sign convention
  (decision: exhibits add) — don't redefine per row.
- **Axis-name lie** (`[axis-names]`): don't trust the positional `.ceded`/`.net` attributes for
  `('gross','ceded')` views.

---

## Open decisions (author)

All resolved (see "Decisions locked"): `[decl-rol]` (deposit currency or `% rol`, `r = D/limit`),
`[decl-return]` (`build()` → `PnL`), `[share]` (allow non-100% share, validate single layer),
missing-clause default (free + unlimited), `[decl-shorthand]` (human `<n> free and <n> at <p>%`
form **and** the `[…]` list escape hatch; counts as digits **or** number-words `one…five`; no hard
cap), `[pnl-division]` (`gcn_df`/`summary_df`/`plot` live on the **`PnL`**, delegating to the
engine — one face, one engine).

**Forward hook (not v1.0):** sliding-scale / profit-commission premium ("swings and slides") is the
same stochastic-consideration pattern — a deterministic premium function of the same `(L, R)` joint —
so it drops onto `PnL` later through the identical pushforward path, no new abstraction. Keep the
`[engine]` pushforward and the PnL-delegation seam general enough that it lands cleanly.

---

## Implementation order

1. **`[engine]`** — `pushforward` + `transformed_moments` on `BivariateDistribution`, with the
   `[axis-names]` cleanup. Unit-test against hand-computed pushforwards before any contract logic.
2. **`[terms]`** — `ReinstatementTerms` + validation; pure, fast to test (pre-plan §21.1–21.3,
   using the **corrected** numbers).
3. **`[analysis]`** — `ReinstatementAnalysis` + `Aggregate.reinstatement_analysis()`; wire the
   pushforwards; accounting identities at every occupied cell **before** rebucketing (pre-plan §21.4).
4. **`[pnl-share]`** — factor the shared presentation; build `gcn_df`/`summary_df`/`validation_df`.
5. **`[decl]`** — grammar + transformer + `build()` plumbing **last** (it only sugars the Python
   terms, so the engine must be proven first).
6. First-class trimmings: `tail_df`, narratives, `plot()`, `_repr_html_`, `info`, `qd`.
7. Update `dev/FEATURES.csv` (+ rerun the introspection cross-check) for the new `ReinstatementAnalysis`
   column and any new `BivariateDistribution` rows.

## Tests

New files (pre-plan §20): `tests/test_reinstatement_terms.py`, `tests/test_reinstatement_pushforward.py`,
`tests/test_reinstatement_analysis.py`, `tests/test_reinstatement_exhibit.py`. Cover pre-plan
§21.1–21.8: deterministic one-event cases (corrected numbers), many-small-losses + order invariance,
free/differently-priced schedules at every breakpoint, pointwise accounting identities, distribution
mean/covariance identities, grid invariance across `bs`, and a Monte-Carlo cross-check for a
Poisson/lognormal cat model.

- **DecL snapshot wrinkle (call out explicitly):** `tests/data/expected_specs.json` is a **frozen
  SLY snapshot**; the new reinstatement lines have **no** SLY reference, so the
  `test_spec_matches_snapshot` path cannot cover them. Add the reinstatement DecL lines to
  `test_decl.agg` with **hand-written spec assertions** (parse → assert `reinstatement_rates`/
  `_deposit`/`_rol` and the single-layer/`no-agg_reins` guards), not via the frozen snapshot.
- **Regression bar:** every existing `test_suite.agg` line still parses and snapshot-matches
  (the grammar addition must be purely additive); `uv run pytest` green.

## Housekeeping

Plan-based change → bump `1.0.0a*` in `pyproject.toml`; add a `CHANGELOG.md` section (feature:
reinstatement premiums; stochastic ceded premium; `BivariateDistribution.pushforward` public;
`ReinstatementAnalysis`; DecL `reinstatements …` clause). Add a `dev/TODO.md` entry under the
appropriate track and move this plan to `dev/done/` at close. Docs: new reinstatement page is a
follow-up (ties to `D7` reinsurance case-study rewrite); keep any `:meth:`/`:class:` refs in lockstep
but **do not** build the doc tree in the iteration loop.
