# Plan: PricingBounds — cross-pricing ranges, and the uniform/Gini lens

## Context

`AllocationBounds` (1.0.0a36) answers: given the total `X` is priced to `P`,
what is the range of the *consistent allocation* to each unit? The same
geometry answers a strictly more general question (similar-risks paper, next
step): given `X` is priced to `P` by some distortion `g`, what is the range of
the **price of another risk `Y`** under the same `g`?

A distortion with Kusuoka measure `μ` prices *any* risk by
`ρ_g(Z) = ∫ TVaR_p(Z) μ(dp)`. So with `X`, `Y` fixed:

- constraint `∫ TVaR_p(X) μ(dp) = P`,
- objective: range of `∫ TVaR_p(Y) μ(dp)`.

This is the `AllocationBounds` slicing on the curve `(TVaR_p(X), TVaR_p(Y))`,
with the additivity constraint dropped (it was only an audit, never used by the
engine). Extreme points are biTVaRs, exactly as before.

**Special case — uniform reference (the Gini lens).** With `X = U[0,1]`,
`TVaR_p(X) = (1+p)/2` is affine in `p`, so the x-axis *is* `p` and the
constraint collapses to a moment condition `E_μ[p] = 2P − 1 =: π`. The object
of study becomes the convex/concave envelope of `Y`'s own TVaR-vs-`p` curve,
sliced at `p = π` — an X-free, Gini-indexed reading of *why* distortions
pricing `Y` disagree. This "turns the question around": fix the mean Kusuoka
level `π`, read off the remaining pricing latitude.

## Key technical resolution — unmatched p-grids

`X` and `Y` are independent objects with different CDF breakpoints. Within a
single atom of `X`, `TVaR_p(X)` is affine in `u = 1/(1−p)`; within a single
atom of `Y`, `TVaR_p(Y)` is affine in the *same* `u`. On the intersection of an
X-atom and a Y-atom — i.e. between consecutive points of the **union of the two
breakpoint sets** — both coordinates are affine in `u`, so the curve is exactly
a line segment in `(T_X, T_Y)` space. Therefore:

> Merge the two breakpoint grids, evaluate both TVaRs at every union point, and
> the curve is exactly piecewise linear with those vertices — no interpolation,
> no resampling, exact for the discretized risks.

Each object's `tvar(p)` / `make_var_tvar` evaluates exact TVaR at arbitrary `p`;
breakpoints are the `F` column (cumsum of `p_total`). Reuses the `s_floor`
material-mass truncation from `AllocationBounds`.

## Design: shared engine + axis abstraction

### 1. Refactor `AllocationBounds` into engine + vertex builder

Today the hull/slice core has the allocation extractor welded to its front.
Split into:

- **Vertex table**: `(p_vertices, x = T_X(p), {y_j(p)})` — the only thing that
  varies across the three applications.
- **Engine** (unchanged behaviour): `_monotone_hull`, `_slice`, `bounds(P)`,
  `bitvars(P)`, `distortion`, `check`, `plot`, `p_star`. Keyed purely on the
  vertex table.

Mechanics: extract a shared base (or a module-level helper holding the vertex
arrays + hulls and the query methods); `AllocationBounds` and `PricingBounds`
become thin front ends that build the vertex table and hand it to the engine.
`AllocationBounds` public behaviour must be unchanged — verify its existing
test module still passes byte-for-byte on `curve_df`/`bounds`/`bitvars`.

### 2. Axis abstraction — a "TVaR source"

A small adapter exposing:

- `tvar(p)` — vectorised exact TVaR at `p` (required),
- `breakpoints` — array of `p` where VaR jumps, or `None` if closed-form/smooth,
- `tvar_inv(t)` — inverse, optional; needed only for `p_star`/calibration on the
  x-axis. For a risk, synthesised from the per-atom `1/(1−p)` structure exactly
  as `AllocationBounds.p_star` does now; for a closed form, supplied directly.

Adapters:

- **risk source** — wraps `Aggregate` / `Portfolio` / pmf `Series`; `tvar` and
  breakpoints from the object (`make_var_tvar`, `F` column).
- **callable source** — wraps a `(T, T_inv)` pair; `breakpoints = None`.
  Uniform is the canonical instance: `T(p) = (1+p)/2`, `T_inv(t) = 2t − 1`
  (provide a ready-made constructor / sentinel so callers say
  `reference='uniform'` rather than hand-coding it).

Both axes accept either source type (symmetric), so uniform can be `X` *or* `Y`.

### 3. `PricingBounds` class

`PricingBounds(x_source, y_sources, *, a=np.inf, s_floor=..., n_grid=...)`:

1. Collect breakpoints from all sources that have them → union (drop
   `< s_floor` mass). If none (all closed-form), use a dense `n_grid` p-grid
   (approximate; smooth curves converge fast).
2. `x = x_source.tvar(p_vertices)`; one y-column per `y_source` (vectorised —
   x-vertices are shared, only y-evaluations multiply, so many `Y`s are cheap).
3. Hand to the engine. `bounds(P)` → range of price(`Y_j`) given `ρ(X)=P`;
   `bitvars`, `distortion`, `check` (reprice `Y` directly with the achieving
   biTVaR) as before.
4. Asset cap `a`: bound `X ∧ a` (and/or `Y ∧ a`) by routing risk sources
   through the existing `_collapsed_exeqa` / capped `tvar` — **defer to a
   follow-up** unless trivial; ship unbounded first.

`y_sources` accepts one risk or several (also a reinsurance layer, an excess
aggregate, a unit standalone — anything with a `tvar`).

### 4. Uniform/Gini convenience

A thin wrapper (or `reference='uniform'` arg) that sets `x_source` to the
uniform callable, reports `π = 2P − 1` alongside the bounds, and optionally a
Gini/dispersion summary (envelope gap of `Y`'s TVaR curve at `π`). No new
engine code — it is `PricingBounds` with special arguments.

## Naming

`Bounds` (total-price distortion cloud) / `AllocationBounds` (consistent splits)
/ **`PricingBounds`** (price ranges of other risks). All in `bounds.py`; extend
the module-docstring comparison table to three columns. (Note: `hacks/pb.py`'s
prototype class was provisionally named `PricingBounds` before it became
`AllocationBounds` — the gitignored hack is superseded, no conflict.)

## Files

- `src/aggregate/bounds.py` — engine refactor; `PricingBounds`; TVaR-source
  adapters; uniform constructor; docstring table → 3 classes. Add to `__all__`.
- `src/aggregate/portfolio.py` — `Portfolio.pricing_bounds(Y, p=...)`
  convenience entry point (parallel to `allocation_bounds`).
- `docs/3_reference/3_x_Bounds.rst` — autoclass `PricingBounds`. Build deferred.
- `tests/test_pricing_bounds.py` — new (see below).

## Tests

- **Hand-checked discrete**: two tiny independent risks; union-of-breakpoints
  vertex table, hulls, bounds, bitvars, `p_star` asserted exactly.
- **Engine parity**: a `PricingBounds` where `Y = X` must give the trivial
  range `[P, P]` (every consistent distortion prices `X` to `P`) — a strong
  invariant.
- **Uniform reference**: `x = U[0,1]`; check `π = 2P − 1`, slice location, and
  the envelope-gap (Gini) reading on a known `Y`.
- **Cross-check vs `Bounds`**: at the bracketing biTVaR knots from a `Bounds`
  object on `X`, the price of `Y` equals `w·TVaR_{p1}(Y)+(1−w)·TVaR_{p0}(Y)` —
  independent confirmation of the hull extremes.
- **`check(P)`** reprices `Y` from first principles to ~1e-12.
- **AllocationBounds regression**: its existing tests still pass unchanged
  after the engine refactor.

## Housekeeping

- New feature from a plan → **version bump** (`1.0.0a36 → a37`).
- `CHANGELOG.md` section; `dev/TODO.md` note; move this plan to `dev/done/`.

## Verification

- `uv run pytest` (full suite + both bounds test modules).
- Smoke: `Y = X` gives `[P, P]`; uniform reference reproduces the Gini reading
  on a worked example; one continuous cross-pricing example plotted.
