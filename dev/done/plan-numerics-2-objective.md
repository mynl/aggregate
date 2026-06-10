# Plan numerics-2 — objective spine (no distortion)

> Part of the numerics program; see `plan-numerics-0-meta.md`. Depends on
> `plan-numerics-1-unit-density` (accessors in place). **No distortion surface
> here** — this plan owns kappa and the objective allocation columns only.
> Implements `shifted-calc-method.md` (Task A) + the objective half of
> `portfolio-calc-methods.md`, and absorbs the *objective* half of the former
> `plan-portfolio-neg-x-pricing` draft (folded in and removed; in git history).

## Goal

Compute `kappa_i(x) = E[X_i | X=x]` on the portfolio total grid via the
shifted-support FFT method, derive all objective allocation columns from `kappa` +
`p_total` by **direct sums that carry `x0`**, drop the `p_{unit}` write and the
entire EPD family, and bring the `Aggregate` objective columns onto the same
windowed-safe footing. Signed-support objective columns fall out for free.

## Step 0 — the audit (do before editing; primary discipline)

Build a small **signed + windowed + disjoint-support** book (e.g. two `ssev`
lognormal-minus-shift lines (cv <= 0.5) plus one ordinary loss line so the total straddles 0;
plus a two-tight-positive-unit book whose total window sits near their sum), run
`update(add_exa=True)` with the signed path temporarily enabled, and produce the
table **column → assumption → signed/windowed verdict → fix**, each verdict
*measured* against a brute-force / Monte-Carlo reference, not guessed. Known
suspects (verdicts TBD by the audit):

- `exeqa_{line}` — the `ft(df.loss · p_i)` weight (priority column).
- `exa_*` / `exa_total`, `lev_*`, `exlea_*`, `exgta_*` — zero-origin `cumsum·bs`.
- the `loss_max` blanking heuristic and the `mult∈{1,10,100}` bucket.
- every `df.loc[0, …]` / `slice(0, a)` / `exeqa/loss` site.

## Step 0b — regression baseline (before editing)

The captured corpus (`tests/baseline/corpus.py`) already snapshots the **key**
columns — portfolio `p_*` / `exeqa_*` (plus `loss/F/S/p_total`) and Aggregate
`p_total/F/S/lev` — and key columns are the regression anchor (D5: everything else
is derived). Before touching `add_exa`: (a) confirm the baseline is freshly
captured on pre-change code; (b) capture **spot-check values** for the derived
columns (`exa_* / exlea_* / exgta_* / exi_xgta_*` and Aggregate `exa/exlea/exgta`
at a handful of sampled rows per corpus case) so the direct-sum rewrite has a
measured target. Gate: key columns and spot checks to **1e-14 relative**;
byte-equal only where the computation path is untouched (e.g. `p_total`).

## Deliverables

### 1. Shifted-support kappa (`shifted-calc-method.md`)

- **`update` persists per-unit state** (G1): `origin_i`, `p_i_native`, `ft_p_i`,
  and the new `ft_xp_i = FFT((origin_i + r·bs)·p_i_native)`. Cannot be reconstructed
  after rebasing — must be captured at combine time.
- **Lifecycle (D6): transient within `update`.** The FT state is computed with the
  unit at combine time and consumed by `add_exa` immediately afterwards in the same
  `update` call; the padded FT arrays are **freed before `update` returns** (no
  memory tax on many-unit / high-`log2` books). Only scalars (`origin_i`) and the
  native pmfs persist — numerics-4 needs the origins.
- **kappa numerator** `n_i = ifft(ft_xp_i · ft_not_i)`; `kappa_i = n_i / p_total`,
  then rebase onto the total output window (`s = k + round((o − Σorigin)/bs)`).
- **`ft_not_i` via prefix/suffix products** (G2), O(mM); spectral division
  `ft_all/ft_p_i` only behind a robust nonzero-bin test.
- Replaces the `df.loss * df[p_i]` form (which needs `p_{unit}` on the total grid),
  so `add_exa`'s dependence on `p_{unit}` is gone.

### 2. Direct-sum objective columns (`portfolio-calc` core formulas)

For positive total loss with `share_i,k = kappa_i,k / x_k`:

```
cum_xi_i,k     = Σ_{j≤k} kappa_i,j · p_j                 # fwd cumsum
tail_share_i,k = Σ_{j>k}  share_i,j · p_j                # rev cumsum  (= exi_xgta·S)
alpha_i,k      = tail_share_i,k / S_k                    # = exi_xgta   (S>0 guard)
exa_i,k        = cum_xi_i,k + x_k · tail_share_i,k       # carries x0
exlea_i,k      = cum_xi_i,k / F_k
exgta_i,k      = (e_i − cum_xi_i,k) / S_k
```

`exa_total` is the same with `kappa_total = x`, `share_total = 1`. Retire
`cumsum(S)*bs`, `df.loc[0:loss_max]`, the `mult` heuristic; replace with explicit
`F<tol / S<tol / |x|<tol` denominator guards (meta steering 1). Keep the
`kappa/loss` ratio behind a **positive-loss guard** (steering 6): on signed grids,
objective `alpha` is not a recovery share and is left undefined/blanked rather than
divided.

### 3. Remove the EPD/second-priority family (D4)

Delete `add_exa_details` (orphaned: `epd_0_* / epd_1_* / e1xi_1gta_*`), the
portfolio `epd_*` columns, and **`Aggregate.density_df['epd']`**. (`e2pri_*` is
already absent.) Keep `e` only where `exgta` needs it.

### 4. Drop the `p_{unit}` write (D1)

Once kappa no longer reads `p_{unit}`, stop writing it into `Portfolio.density_df`
(both paths). Stand-alone unit quantities (`lev_i`, `e_i`, percentiles) come from
the **native** unit pmf via the numerics-1 accessors / native prefix sums
(`lev_i(a) = unit_cum_xp[idx] + a·(1 − unit_cum_p[idx])`), valid whether or not unit
support overlaps the total window.

**Sampling/switcheroo cluster (found in review; design deferred to its own
future plan).** `Portfolio.sample`, `swap_density_df` (method + module twin),
`add_exa_sample`, and `make_awkward` still read `p_{unit}` and are structurally
premised on unit columns living in `density_df`. Their *redesign* is explicitly
not this plan — but dropping the write must not leave them broken: give them the
**minimal mechanical re-source** (unit pmf via the numerics-1 accessors where
they read `port.density_df[p_{unit}]`; `swap_density_df` keeps accepting a
user-supplied `p_{line}` frame as input) so pytest stays green, and flag any
deeper semantic question (e.g. `add_exa_sample` on a windowed/signed book) for
the separate sampling plan rather than solving it here.

### 5. Aggregate-side objective columns

`Aggregate.density_df` `lev/exa/exlea/exgta` to direct sums carrying `x0`
(windowed/signed safe); drop the `epd` column (D4). The Aggregate **distortion**
surface (`apply_distortion`, `exag`) is numerics-3.

## Invariants / tests (small exact finite laws)

- `Σ_i kappa_i(x) == x` on every row with material `p_total` (the shifted-method key
  invariant — the core correctness anchor).
- `Σ_i exa_i,k == exa_total,k` at every `k`.
- shifted kappa == brute-force convolution for small discrete units **incl. negative
  origins and nonzero positive origins**; force a zero in one line spectrum and
  confirm prefix/suffix still works.
- disjoint-window two-unit book: `p_total` on the total window, each
  `unit_density(u)` on its own window, **no core calc reads `p_{unit}`**;
  `e_i` (native) reconciles `Σ kappa_i · p_total` within represented-window tol.
- stand-alone `lev_i(a)` == exact capped native sum for `a` below / inside / above
  the unit support; independent of total-window overlap.
- signed book: `exeqa_{line}(a)` matches MC `E[X_i | X≈a]` at several `a` incl `a<0`.
- **legacy regression (D5)**: a standard zero-origin book reproduces today's
  `p_total` byte-equal (path untouched) and the key `exeqa_*` columns to 1e-14
  relative (captured baselines); derived `exa, exlea, exgta` match the Step-0b spot
  checks to 1e-14 — fp drift from `cumsum(S)*bs` → direct sums is expected and fine.
  This is a refactor for the legacy case.

## Files

- `src/aggregate/portfolio.py` — `update` (per-unit state, drop `p_{unit}` write),
  `add_exa` (shifted kappa + direct sums), delete `add_exa_details`, native
  stand-alone `lev/e`, re-enable signed objective path (remove warn+fallback).
- `src/aggregate/distributions.py` — Aggregate objective columns + drop `epd`.
- `tests/` — the invariants above; signed/windowed/disjoint corpus; mirror any DecL
  into `test_decl.agg`.
- `docs/` — `.rst` lockstep for the removed columns (`epd`, `p_{unit}` semantics):
  `5_x_portfolio_calculations.rst` is the big one. Fix where unambiguous (derive
  replacements by simple math from the new columns); flag the unfixable kernel for
  the author (meta rule 11). No doc build in the loop.

## Out of scope

- Distortion / `gp` / `exag` / `T.*`/`M.*` / pricing (numerics-3).
- `value_type` consumption (numerics-3).
- Windowed *combine* sizing (numerics-4) — this plan makes allocation correct *given*
  a combine; the windowed sizing/routing of `p_total` is numerics-4. (They meet at
  the per-unit-origin state captured in `update`.)
- **Sampling/switcheroo redesign** (`sample` / `swap_density_df` /
  `add_exa_sample` semantics) — separate future plan; deliverable 4 only keeps
  them mechanically working when the `p_{unit}` write goes.

## Housekeeping

Version bump; CHANGELOG (shifted kappa; direct-sum objective columns; **breaking:**
`p_{unit}` removed from `Portfolio.density_df`, EPD family + `Aggregate.epd` removed;
signed objective columns now available); `dev/TODO.md` N2 done. Move to `dev/done/`.
(The objective half of the former `plan-portfolio-neg-x-pricing` draft is folded in
here; that draft was already removed at drafting time.)
