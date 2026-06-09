# Plan numerics-3 — distortion spine

> Part of the numerics program; see `plan-numerics-0-meta.md`. Depends on
> `plan-numerics-2-objective` (kappa + objective columns on the total grid).
> Implements `choquet-calc-method.md` (Task B) + the distorted half of
> `portfolio-calc-methods.md`, and absorbs the distortion / `value_type` half of the
> former `plan-portfolio-neg-x-pricing` draft (folded in and removed; in git
> history). **This is where linear and lifted become one engine and `T.*`/`M.*` die.**

## Step 0 — the audit (do before editing)

Run the numerics-2 signed/windowed book through `apply_distortion` + `price` for
both `allocation` methods and a mass distortion, and produce the table **distorted
column → assumption → verdict → fix**, measured against `Distortion.price` and a
brute-force reference. Confirm the consistency target (below) currently fails only
where the notes predict (missing `x0`, cache key, mass guard). Also **pre-capture**
`pricing_at` / `pentagon_at` / `price` scalar readouts for the corpus distortions on
pre-change code (the augmented baseline already snapshots the key distorted columns
`loss/S/gS/gp_total/exag_*`) — these are the regression targets for deliverable 4.

## Deliverables

### 1. One exact-discrete Choquet helper (`choquet-calc-method.md`)

In `spectral.py`, a single helper (build on / fold in `Distortion.price`'s
`method='dx'/'ds'` and `make_q`):

```
S_k = P(X>x_k);  T_k = S_k + p_k;  T_0=1, S_{-1}=0
gp_k = g(T_k) - g(S_k)                      # exact distorted atom weights, a pmf
rho      = dot(x, gp)                        # orientation-agnostic
rho(X∧a) = dot(min(x,a), gp)
returns S, T, gS, gp   (+ deficit, layer-form reconciliation assert)
```

- Layer form `x0 + Σ (x_{k+1}-x_k) g(S_k)` survives **only** as a reconciliation
  assert (steering 1), not the computation.
- **Deficit policy** (G3): clean fuzz → `deficit = 1 − Σp` → renormalize/absorb if
  within tol, else raise `DefectiveDistributionError` unless an explicit truncation
  policy is passed.
- **Gatekeeping stays in Aggregate/Portfolio** (steering 5): the helper computes
  unconditionally; the caller that knows boundedness rejects
  `has_mass and not bounded` (G6 — move this guard *down* from `price(lifted)` to the
  builder so `exag_total` can't build an unstable frame).
- **Public surface persists:** `Distortion.price` and `make_q` keep their signatures
  and route through the helper — the spectral calculation producing `gp` exists in
  **exactly one place**.
- **`S_calculation` survives only as the deficit-parking direction** (D7): forwards
  parks unrepresented mass at the top atom, backwards zeroes it. On a clean law the
  two agree to tol — assert it. A material difference means the distribution is
  defective and the answer is not knowable; at that point you are *choosing* whether
  the missing mass goes up or down to the tail, and the deficit policy (above)
  governs whether that choice is even allowed.

### 2. `view × value_type` — the 2×2 (D3)

Two orthogonal inputs; each toggles `g ↔ g_dual`; compose by XOR — use the dual iff
`(view==bid) XOR (value_type==payoff)`. No negate-the-variable path: the
`dot(x, gp)` engine is orientation-agnostic, so `value_type` only selects the
effective `g`.

| | loss | payoff |
|---|---|---|
| **ask** | `g` | `g_dual` |
| **bid** | `g_dual` | `g` |

(Sanity, author-checked: ask-of-loss puts the heaviest distortion weight on the
large-loss tail = upside for the insurer; ask-of-payoff sorts ascending so the
largest *good* outcome is down-weighted = dual. ✓) A dedicated `value_type` DecL
keyword stays deferred — but note the *attribute* already exists on `Aggregate`
(`distributions.py:3290`; `pnl` sets it), so this deliverable **consumes** the
existing attribute rather than inventing the axis.

**Portfolio-level `value_type` (D8 — settled).** A Portfolio's units must have a
consistent `value_type` and the Portfolio must know what it is. **Homogeneous books
adopt the units' common level** (default `loss`). **A mixed book raises** — a clear
error naming the offending units and telling the user to redeclare them explicitly
(reflect / `pnl` at declaration). No magic, no invisible reversal: auto-reversal
was rejected because it acts at *combine* time (it changes the total law, not just
the pricing) and drags in the deferred straddling-P&L allocation semantics; it
could return later as an explicit opt-in flag if ever wanted. The 2×2 table above
is unaffected — a separate, happy circumstance.

### 3. Unified linear/lifted builder at `apply_distortion`

Both methods, same column schema, computed across **all** asset levels in one O(n)
sweep (author-confirmed; the two-line formula in meta §1):

```
exag_i = cumsum_k(kappa_i · gp)  +  x · gS · TAIL_i
   lifted: TAIL_i = exi_xgtag_i   (beta,  distorted tail share)
   linear: TAIL_i = exi_xgta_i    (alpha, objective tail share)
```

- Compute the collapsed-atom term as `x·gS·alpha` (not `x·(gS/S)·tail_share`): the
  `Seq0`-zeroing on `alpha`/`beta` already handles the `S→0` right edge cleanly.
- `exi_xgtag_i` (beta) = distorted tail share via `gp`; reuse the numerics-2
  `exi_xgta_i` (alpha) for linear. So linear adds **no new machinery** beyond
  objective + distorted columns.
- `price` reads rows from this frame for **both** methods; the separate
  `_collapsed_exeqa` linear pricing engine is **deleted**. The per-`a`
  "collapse iff `sf(a) > deficit`" heuristic dissolves into the all-levels sweep +
  the global deficit policy.
- **`AllocationBounds` moves onto this builder** (D2): the convex-hull slicing reads
  the unified surface; `_collapsed_exeqa` is absorbed/retired (no downstream
  primitive kept). Regression-gate bounds against its captured baseline.

### 4. Kill `T.*` / `M.*`; explicit pentagon columns

Replace the 46 `T.*`/`M.*` sites with explicit per-line `L/M/P/Q` (and the layer
curves only inside the diagnostic frame, deliverable 6):

```
L_i = exa_i ;  P_i = exag_i ;  M_i = P_i − L_i ;  Q_total = a − P_total
```

computed by direct sums (carry `x0`). **No persistent per-line `Q` column** (D7 —
nothing consumes the full curve). Line capital is computed **on demand** at the
requested `a` by the layer-ROE construction,

```
Q_i(a) = Σ_{k: x_k < a} (gS_k·β_i,k − S_k·α_i,k) · (1 − gS_k)/(gS_k − S_k) · Δx_k
```

(line layer margin ÷ total layer ROE, integrated; a `gS==S` layer has zero margin
and contributes zero capital — guard the ratio), sourced from the same
`alpha/beta/gS` columns, inside `pricing_at`/`pentagon_at`/`price`. This is the one
legitimately layer-based quantity (capital *is* allocated by layer); reconcile
`Σ_i Q_i(a) == a − exag_total(a)` as the assert (zero-origin exact; the `x0` term on
windowed grids goes through the Step-0 audit). Migrate the real consumers:
`pricing_at`, `pentagon_at`, `price(lifted/linear)`, **`pentagon.py:345–346`** —
each reads one row at one `a`, exactly the on-demand shape.

### 5. Aggregate distortion surface + consistency target

`Aggregate.apply_distortion`/`price` route through the helper (currently
`exag = hstack((0,gS[:-1])).cumsum()*bs`, zero-origin, no bid/gp/mass-guard). The
six surfaces must agree on total premium for a normalized finite law at the same `a`
(the `choquet` note's consistency target):

```
Distortion.price(.,a,'ask','dx')  ==  (.,'ds')  ==  Portfolio exag_total(a)
  ==  Portfolio.price(...).price   ==  Aggregate exag(a)  ==  Aggregate.price(...).P
```

### 6. `plot_twelve` adapter (finish "not the boss")

Explicit `Portfolio.allocation_diagnostics(distortion, surface='lifted'|'linear')`
returning the layer curves (`S·alpha`, `gS·beta`, layer margin `gS·beta − S·alpha`,
cumulative margin/capital) + `F/gF/S/gS` + `kappa/alpha/beta`. `plot_twelve` reads
that frame and the numerics-1 native unit pmfs; **remove the efficient→full
cache-pop hack** (`pedagogy.py:1408–1410`). The core pricing frame no longer carries
`efficient=False` diagnostic columns.

### 7. Cache key; `efficient` removed

Widen the `apply_distortion` cache key from `name` to
`(name, view, value_type, S_calculation, allocation)` so bid/ask, loss/payoff,
linear/lifted, and forward/backward frames coexist. **`efficient` is removed
entirely** (D7): with diagnostics out of the core frame (deliverable 6) and the
methods it gated gone, there is no lean/full split — one frame shape, no flag.

## Invariants / tests

- **identity distortion** ⇒ `beta == alpha`, `exag == exa` (premium==loss),
  `M ≈ 0` to float tol, for both linear and lifted.
- one-line portfolio: line == total for identity and non-identity distortions.
- **linear vs lifted**: agree when `a ≥ max support`; differ predictably when `a`
  cuts a tail; `Σ_i exag_i == exag_total` (= `rho(X∧a)`) for **both**.
- the consistency target (deliverable 5) holds across all six surfaces.
- **mass-on-unbounded raises in the builder**, not just in `price` (G6).
- `value_type='payoff'` prices as the dual of the matching `loss` object (round-trip).
- signed P&L total prices via `dot(kappa, gp)` (steering 6); equal-priority share
  rejected on signed grids unless an explicit non-negative payment law is given.
- cache: ask/bid, loss/payoff, linear/lifted, fwd/bwd-S frames do not overwrite each
  other.
- forwards vs backwards `S_calculation` agree to tol on a clean law (D7); a material
  difference is a defective-distribution diagnostic, not a free parameter.
- mixed-`value_type` book **raises** with the D8 error (no combine, no reversal);
  homogeneous `payoff` book adopts the common level.
- **legacy regression (D5)**: zero-origin book reproduces today's lifted `exag_*`
  (key distorted columns, captured augmented baseline) to 1e-14 relative, and the
  `pricing_at`/`pentagon_at`/linear-price scalars from the Step-0 pre-capture to
  1e-14 — fp order-of-ops drift is expected and fine.
- `AllocationBounds` reproduces its captured baseline on the new builder.

## Files

- `src/aggregate/spectral.py` — the Choquet helper; `view×value_type` → effective g.
- `src/aggregate/portfolio.py` — `_build_augmented` (unified linear/lifted, `gp`,
  pentagon cols, drop `T.*`/`M.*`), `price` (read both, delete `_collapsed_exeqa`
  engine), `pricing_at`/`pentagon_at`, `allocation_diagnostics`, cache key.
- `src/aggregate/distributions.py` — `Aggregate.apply_distortion`/`price` via helper.
- `src/aggregate/bounds.py` — `AllocationBounds` onto the unified builder.
- `src/aggregate/pentagon.py` — `T.M_`/`T.Q_` reads → explicit `M`/`Q`.
- `src/aggregate/pedagogy.py` — `plot_twelve` onto `allocation_diagnostics`; drop hack.
- `tests/` — invariants above; mirror DecL into `test_decl.agg`.
- `docs/` — `.rst` lockstep for the `T.*`/`M.*`/linear-engine removal:
  `5_x_portfolio_calculations.rst`, `5_x_distortions.rst`, `5_x_bodoff.rst`,
  `2_x_10mins.rst` + the problems pages. Fix where unambiguous (replacement columns
  derivable by simple math); flag the unfixable kernel for the author (meta rule
  11). No doc build in the loop.

## Out of scope

- Windowed *combine* sizing (numerics-4).
- `value_type` DecL keyword and the deeper "capital allocation on a straddling P&L"
  semantics (deferred — a short research note if needed).

## Housekeeping

Version bump; CHANGELOG (**breaking:** `T.*`/`M.*` removed, linear/lifted unified,
`apply_distortion` cache key changed; new: `value_type` pricing axis,
`allocation_diagnostics`); `dev/TODO.md` N3 done. Move to `dev/done/`.
(The distortion / `value_type` half of the former `plan-portfolio-neg-x-pricing`
draft is folded in here; that draft was already removed at drafting time.)
