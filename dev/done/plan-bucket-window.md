# Plan B — non-zero aggregate *output* window for high-mean / thin-tail aggregates

## Status

**Landed in 1.0.0a51** (Steps 1, 2, 2b). Split out of the former
`dev/plan-bucket-sizing.md` (now deleted). This is the deeper,
**bivariate-relevant** half. It landed after Plan A (`dev/plan-bucket-combine.md`,
a49).

**What landed vs. the original plan.** The deepest discovery during execution:
the benign-wrap *compute* and output roll (Step 2's hard part — periodic FFT,
the `round(x_min/bs)` relabel, and straddle handling) were **already built** in
`_fft_aggregate` for the negative-x / `pnl` work — `np.roll(a, -j0)` is modular,
so a 15M-bucket roll and any period straddle are automatic. The real work was
just **Step 1 (windowed sizing in `_bs_window`)** plus relaxing the `x0=0` gate.
Two guards were added that the draft did not anticipate: a **severity-fit guard**
(a single severity must fit `[0, N·bs]`, else fall back — the `fixed`-1 /
`approximate` trap) and **occ-reins suppression** (the occ-reins severity path
rides the output grid `xs == xs_sev`, which a non-zero origin breaks; deferred,
see `dev/TODO.md` W5). Step 3 (portfolio extension) was **not** done in this pass
— single-aggregate windowing is the shipped scope; the portfolio combine on
windowed units is follow-up (ties W2 / Track M).

This plan adds a genuinely new computational capability (computing an aggregate
on a grid *smaller* than its absolute support by exploiting benign periodic
wrap). Treat it carefully and review the freeze diff closely.

## The hard invariant (read first)

**The severity grid ALWAYS contains physical 0 on its lattice.** It need not be
*based* at 0 — `[-10, 10]`, `[-10, 0]`, `[0, 10]` are all fine — but a window
like `[10, 20]` that excludes 0 is **illegal**. In code:
`xs_sev = (arange(N) − i0)·bs`, where `i0` is the index of physical 0
(`distributions.py:4150`, `_severity_offset`).

**You may never push a per-occurrence constant into the severity.** The aggregate
is a sum of a *random* number `N` of severities, so a severity shift `c` becomes
`N·c` — a random shift, not a constant one (the "sev + const = const×N" trap).

→ **Windowing is exclusively an aggregate *output* operation**, achieved by
relabeling the periodic FFT output, with the constant added **once** on the final
aggregate axis. The severity underneath stays 0-containing.

## Motivation

```python
agg Window 10_000_000 claims dsev [1 2] poisson
```

Reproduced 2026-06-08: builds at `bs=600, log2=16, x_min=0, x_max≈39.3M`, but the
mass sits at `m=15,000,000 ± sd≈5,000` — a band ~60k wide. The grid spans
`[0, 39M]`, so only ~100 buckets land on the actual mass: the resolution is
wasted on an empty `[0, 14.97M]`. We want `bs=1`, window `[≈14.97M, 15.03M]`.

This matters most in the **bivariate** case (the reason this plan exists): a
2¹⁶×2¹⁶ severity grid is unaffordable, so each axis must be windowed to the
region that actually carries mass. Plan B is the 1-D enabler the bivariate work
will consume; it reuses the signed-axis / output-roll machinery (see
`[[project_negx_multivariate_plans]]`).

## The technique (and why it is new)

The FFT aggregate is **periodic** with period `P = M·bs = 2**log2 · bs`. Compute
on the small grid `[0, P)` at `bs=1`; the thin band at 15M wraps to `15M mod P`.
Because the band width (~60k) `< P` (65,536) and essentially **no mass lies
outside the band** (coverage `1 − 1e−WINDOW_NINES`), the periodization
`Σ_k f(y + kP)` produces a **single clean, non-overlapping copy** of the true
density. Relabel the output **up by the integer number of periods** (the output
roll `j0 = round(x_min/bs)`, reusing the `i0`/`j0`/`shift` machinery at
`distributions.py:2247-2267`) so the displayed axis reads `[x_min, x_min + P]`
with `x_min ≈ 14.97M`. The constant is added once, on the output axis only.

**Why this is not just "un-gate the existing roll":** today's `pnl`/affine output
roll (`_apply_agg_affine`, `_pnl_window`) computes the loss on a grid that
**absolutely holds** it (`[0, x_max]` at the loss `bs`/`log2`), then relabels for
*display*. Plan B deliberately computes on a grid **smaller than `x_max`** and
relies on the wrap being benign — that is the new capability. It requires:

1. **A feasibility guard.** The wrap is benign only if `band_width < P` **and**
   the out-of-band mass `< tolerance`. Otherwise fall back to the current
   behaviour (coarser `bs` that holds `[0, x_max]`). Refuse — do not silently
   alias — when the guard fails.
2. **Straddle handling.** If the band `mod P` crosses the buffer boundary
   (wraps around the end), rotate the buffer so the band is contiguous before
   relabeling. The offset is chosen from the mean; the existing `j0` roll is the
   tool, but here `j0 = round(x_min/bs)` is large (≈ 15M/bs) and reduced mod `M`.

## Design constraints (carried from the former plan)

1. **Discrete distributions: heavily favour `bs = 1`.** Only coarsen upward when
   a `dsev` carries genuinely large atoms. The high-mean example must end at
   `bs=1` precisely because window *width*, not `x_max`, drives sizing.
2. **Output window need not start at 0.** Size off the realised **window width**
   `x_max − x_min`, not the absolute `x_max`. This is what lets `bs=1` / a small
   `log2` survive for high-mean / thin-tail cases.
3. **Lower `log2` when the cap is not needed.** The band may need far fewer than
   2¹⁶ buckets; propagate the smaller `log2` instead of always filling the cap.

## What already exists (reuse, don't reinvent)

- **`_severity_offset` / `i0`** (`distributions.py:4150`) — input roll; keeps 0
  on the severity lattice. Untouched (it is the invariant above).
- **The output roll `j0` / `shift = i0 + j0`** (`distributions.py:2247-2267`) —
  the relabel mechanism. Currently reachable only on the signed / affine path
  (`_size` forces `x0 = 0.0` for non-signed, `distributions.py:6241`). Plan B
  relaxes that gate for the high-mean non-signed case.
- **`estimate_agg_window(m, sd, skew, p)`** (`distributions.py:442`) — the
  two-sided window already used for signed aggregates. Reuse it to size the
  high-mean window from width.
- **`_bs_window`'s `_size` / `_row` / `used`-row machinery**
  (`distributions.py:6218-6371`) — extend, don't rewrite.

## Plan (staged)

### Step 1 — window-width sizing for non-signed high-mean

- In `_bs_window`, detect the high-mean / thin-relative-spread non-signed case
  (mass band far from 0, `band_width ≪ m`). Size it from the **two-sided**
  `estimate_agg_window` (width), choose `x_min` near the band, and set `bs` from
  width not `x_max`.
- Add the **feasibility guard** (`band_width < P`, out-of-band mass `<` tol).
  When it fails, keep today's `[0, x_max]` behaviour.
- Record a new `_bs_window_df` method row (e.g. `windowed`) so the decision is
  inspectable, consistent with `moment` / `exact_discrete` / `bounded_small`.

### Step 2 — un-gate the output roll for non-signed; benign-wrap compute

- Relax the `x0 = 0.0 if not signed` rule for the windowed case. **Two gates,
  not one** — both must learn the new "non-signed but windowed" state or they
  cancel:
  1. the `_size` origin computation (`x0 = ... if signed else 0.0`,
     `distributions.py:6299` and the over-cap branch at `:6310`); and
  2. the return gate `ret_x0 = sel_x0 if self._signed_severity() else 0.0`
     (`distributions.py:6404-6406`) — if only `_size` is relaxed, the return
     stomps the non-zero `x0` back to 0.
- Drive the output roll `j0` in `update` / `update_work` for this case (the roll
  exists; the gating relaxes). Add the **straddle rotation** so the band is
  contiguous in the buffer before relabeling. (This is the one piece with no
  existing analog — review its freeze diff most closely.)
- Verify the benign-wrap compute is mass-exact: the periodized copy is the true
  density to within the coverage tolerance.

### Step 2b — plotting graceful from a positive origin

- `Aggregate.plot` / `_limits` must start gracefully from `x_min > 0` rather
  than assuming a 0 origin. The a50 window-aware `_limits(stat='range')`
  thin-tail branch (`xs[0] > 0`) already anchors the left edge at the realised
  support min — confirm it now fires (it was forward-looking / unreachable until
  this plan lands) and that the density / distribution / Lee panels read sensibly
  on a non-zero-origin grid.

### Step 3 — portfolio extension (high-mean non-signed books)

- Extend Plan A's `best_window` so `W_tot` is computed from realised window
  **widths with non-zero origins**, and the shared grid carries a non-zero output
  `x_min`. Reuses Plan A's per-unit window table; drives each unit on the shared
  windowed grid.
- Guard the combined feasibility (`Σ widths < P`) the same way.

### Step 4 — validation + freeze review

- Confirm moments (`agg_m/cv/skew` vs `est_*`), mass conservation, and the
  feasibility/straddle guards on a spread of high-mean cases.
- Freeze-before / check-after: high-mean cases **will** move (that's the point) —
  review the diff, confirm each is the finer windowed grid.

## Verification

- **`uv run pytest`** — green.
- **`scripts/freeze_knowledge.py`** — high-mean cases move; review surface.
- **`scripts/bucket_baseline.py`** — diff the high-mean rows.
- **Targeted asserts** appended to `tests/test_bucket_sizing.py`:
  - `agg Window 10_000_000 claims dsev [1 2] poisson` → `bs == 1`, sensible
    `log2`, `est_m/cv/skew` match analytic, mass `== 1`;
  - a high-mean **continuous** case (e.g. large-mean lognormal, thin relative
    spread) → windowed, moments match;
  - a **straddle** case (band crossing the period boundary) → correct after
    rotation;
  - an **infeasible** case (`band_width ≥ P`) → falls back, no silent aliasing.

## Housekeeping (standing rules)

- Bump `pyproject.toml` `1.0.0a*` (own bump, separate from Plan A).
- `CHANGELOG.md`: describe window-width sizing + the non-zero output window;
  call out that high-mean / thin-tail aggregates now resolve at `bs=1`.
- `dev/TODO.md`: one line; mark on landing.
- Append the targeted asserts to `tests/test_bucket_sizing.py`.
- Move this plan to `dev/done/plan-bucket-window.md` on landing.

## Out of scope

- **The bivariate solver itself.** Plan B is the 1-D windowing enabler that the
  bivariate work (2-D severity grid) consumes; the 2-D machinery is its own plan.
- **Severity-grid changes.** The severity always contains 0 (the invariant
  above); only the aggregate *output* window moves.
- **Plan A's combine rule** (resolution + span) — landed separately first.

## Resolved decisions (author-confirmed)

- **Detection threshold = the self-tuning rule: window when it is feasible and
  strictly finer than the 0-based grid.** No `band_width / m` ratio is needed.

  *Why this is self-limiting (not "always finer").* The window is
  `estimate_agg_window` at `p = 1 − 1e−12`, i.e. `[m − z·sd, m + z·sd]` with
  `z = norm.isf(1e−12) ≈ 7.03`. Its lower edge clears 0 — so the window is
  genuinely narrower than the 0-based `[0, x_max]` — **only when**
  `m − z·sd > 0`, equivalently

  ```
  agg_cv = sd / m  <  1/z  ≈  0.14.
  ```

  So the rule fires precisely on concentrated aggregates (very high claim count,
  or thin severity × high frequency — the `10M claims dsev[1 2]` case has
  `agg_cv ≈ 3e−4`). For an ordinary `agg_cv ≳ 0.14` aggregate the two-sided
  lower edge falls at or below 0, `x_min` **clamps to 0**, and the grid stays
  byte-identical to the legacy 0-based grid. The clamp at 0 is what bounds the
  blast radius — the rule is equivalent to "the mass band clears 0 by `z·sd`,"
  which is exactly the intended target rather than "almost everything."

- **`q` / `F` / quantiles / plots are defined only on the window `[x_min, x_max]`
  for a windowed aggregate.** This is accepted and deliberate. Sizing at
  `1 − 1e−12` two-sided guarantees coverage of the probability *mass* to 1e-12
  per edge; what is given up is the `[0, x_min)` *axis* region (sub-tolerance
  far tail, low-attachment layer losses, the from-0 severity overlay / plot
  view). A user who needs the full `[0, x_max]` axis can pass `x_min=0` to
  `update` to force the legacy grid back. Plotting must start gracefully from a
  positive origin (see Step 2b).

- **Fallback policy when infeasible: quiet fallback to `[0, x_max]`** with an
  inspectable `_bs_window_df` row recording the decision (a `windowed` /
  `infeasible` note), consistent with `moment` / `bounded_small`. No
  `explain_validation` warning.
