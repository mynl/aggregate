# Plan: AllocationBounds — natural-allocation pricing ranges (TODO N5)

## Context

`Portfolio.pricing_bounds` has raised `NotImplementedError` since 1.0.0a11
(written against the legacy `Bounds` API). The replacement was prototyped and
validated in `hacks/pb.py` (2026-06-07 hackathon): given total premium P, the
range of natural allocations to each unit over all distortions pricing X to P
is the vertical slice at T = P through the convex hull of the curve
`(TVaR_p(X), a_i(p))`. On the discrete FFT grid that curve is *exactly
piecewise linear* with vertices at CDF breakpoints (both coordinates are
affine in `u = 1/(1-p)` within an atom), so the hull — built from plain
conditional tail expectations via reverse cumsums — is exact. O(n) per unit,
P-independent construction, cheap slicing per P.

## Decisions (agreed with author)

- Class name **`AllocationBounds`**, in `bounds.py` next to `Bounds`
  (names state the distinction: `Bounds` bounds *prices of the total*;
  `AllocationBounds` bounds *allocations given the total's price*).
- Entry point **`Portfolio.allocation_bounds()`** returning the object;
  old `pricing_bounds` method **deleted** (with `PricingBoundsResult`).
- Text comparison table Bounds vs AllocationBounds goes in the
  `bounds.py` module docstring.
- Unbounded total only; asset cap `a` deferred (TODO note — interacts with
  linear-vs-lifted NA at default; linear NA preferred).
- Keep author's uncommitted tweaks: `Bounds` `n_p=256` default + docstring
  (bounds.py), `ipython-autotime` in pyproject notebook extras.

## Changes

1. **`src/aggregate/bounds.py`** — extend module docstring (two-class
   overview table); add `_monotone_hull` helper + `AllocationBounds` class
   (port of `hacks/pb.py` `PricingBounds` with one numerics improvement:
   `check()` repricing uses exact vertex tail probabilities `S_vert` for the
   hinge `g(s) = min(1, s/S)` instead of `1 - p` — avoids cancellation for
   p near 1). Add to `__all__`.
2. **`src/aggregate/portfolio.py`** — delete `pricing_bounds` (lines
   ~624-733); add `allocation_bounds(units=None, s_floor=1e-14)` thin
   wrapper; drop `PricingBoundsResult` import.
3. **`src/aggregate/results.py`** — delete `PricingBoundsResult` dataclass,
   header mention, and now-unused `Bounds` TYPE_CHECKING import.
4. **`docs/3_reference/3_x_Bounds.rst`** — add `autoclass` for
   `AllocationBounds`; grep docs for stale `pricing_bounds` refs.
   Doc build deferred to author (standing rule).
5. **`tests/test_allocation_bounds.py`** — new. Hand-checkable discrete
   portfolio (2 indep units, dsev [0 8]/[0 2], four total outcomes) with
   exact vertex table, hulls, bounds, bitvars, p_star asserted; plus
   self-audits on a continuous portfolio (additivity, check() repricing,
   width monotone in P, feasible-range errors, degenerate distortion).
6. **Housekeeping** — version 1.0.0a35 → 1.0.0a36; CHANGELOG section;
   TODO N5 marked done (this file → dev/done/).

## Verification

- `uv run pytest` (full suite + new test module).
- Smoke: rebuild hackathon example, compare `AllocationBounds` output to
  validated `hacks/pb.py` results.
