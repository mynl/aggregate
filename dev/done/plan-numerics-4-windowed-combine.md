# Plan numerics-4 — integrating 1A + 1P windowing into multivariate

> **STATUS: COMPLETE (2026-06-19) — delivered via `dev/done/plan-mv.md`.** This
> plan's deliverable (Part B, bivariate per-axis windowing) was superseded and
> shipped by the bivariate firm-up: MV-2/MV-3 (a72/a76) deleted both private
> sizers (`_size_axis`, module `size_axis`) and route every axis through the
> measured `balanced_window` primitive; Part A (windowed portfolio combine) had
> already moved to 1P in `plan-bucket-window-2`; P0 (occ-reins × windowing) is
> covered by netceded sizing on `reins_density_df` (MV-3). The marginal-
> reproduces-standalone invariant is plan-mv §8. **One item deliberately deferred
> post-1.0:** custom *per-axis `bs`* rebucketing (numerics-4's "ride `xs_sev`"
> framing of P0) — netceded keeps the single gross `bs` for 1.0 (plan-mv §5.2);
> windowing for occ-reins books is nonetheless re-enabled. Tracked as W2 (now
> done). Moved to `dev/done`.
>
> Part of the numerics program; see `plan-numerics-0-meta.md`. **This plan
> consumes the univariate windowing primitives built in
> `plan-bucket-window-2.md` (Step 1: 1A Aggregate symmetric window, 1P Portfolio
> windowed combine) and integrates them into `multivariate.py`** — bivariate
> per-axis windowing. It also re-bases onto the kappa-ready foundation that
> `plan-numerics-2-objective` establishes. Depends on numerics-2 (and numerics-3
> for the priced/allocated views of a windowed book) **and on bucket-window-2
> landing first** (it owns the 1-D sizer and the portfolio combine). **Last in
> the program; `plan-multivariate-punchup.md` is the follow-on MV work.**
>
> **Scope moved out (2026-06-16):** the **windowed portfolio combine** (former
> Part A) is now **1P in `plan-bucket-window-2.md`** — this plan *consumes* it,
> it no longer owns it. What remains here is the genuinely multivariate piece:
> per-axis windowing of the 2-D severity grid, which reuses 1A's per-axis sizer
> and 1P's origin-sum / roll-combine machinery.

## Why it changes after numerics-2/3

The former `plan-window-port-bv` draft was written assuming the *legacy* `add_exa`
(kappa via `df.loss · p_{unit}` on a shared grid, `p_{unit}` columns in
`density_df`). After numerics-2:

- kappa is already shifted-support and carries **per-unit origins** — the windowed
  combine no longer needs a special allocation path; it inherits one.
- `p_{unit}` is no longer in `Portfolio.density_df`; the windowed-combine's
  "roll each unit column by `−j0_tot`" + masking subtlety (old Part A step 4) is
  reframed in terms of `unit_density` accessors, not core-frame columns.
- objective columns are direct-sum / windowed-safe, so the "off-window unit column"
  caveat becomes a property of `aligned_unit_density_df`, not a correctness risk in
  `density_df`.

So this plan keeps that draft's **sizing + routing** core (the compute machinery
largely already exists: the signed path *is* the non-zero-origin combine;
`_affine_axis` *is* the per-axis output relabel) but drops the bits that
double-handled `p_{unit}` and bespoke kappa.

## Scope (carried from the former window-port-bv draft, revised)

### Part A — windowed portfolio combine → **moved to `plan-bucket-window-2.md` (1P)**
Built in Step 1 alongside the 1A Aggregate sizer: `best_window` sums per-unit
origins (`x_min = Σ x_min_k`, floored) and widths, pads once at the total, and
routes the windowed non-signed book through the existing roll-combine. This plan
**consumes** the result (uniform per-unit two-sided windows + the origin-sum /
roll machinery); it is no longer a deliverable here. See bucket-window-2 §"1P".

### Part B — bivariate per-axis windowing (the goal of this plan)
- `multivariate.size_axis` windowed variant, parametrized by coverage nines (keep
  the `WINDOW_NINES` vs `_WINDOW_NINES` split for 2-D memory).
- per-axis output roll (`np.roll(T, −j0_i, axis=i)`), the pure-roll specialisation of
  `_affine_axis`; relabel axis coords.
- **marginal-consistency anchor**: marginalising a windowed axis reproduces the 1-D
  windowed aggregate of that margin (same `x_min_i, bs_i`, moments) — the core
  correctness assert.

### P0 — occurrence reinsurance × windowing
Re-folded here rather than parked: make the occ-reins severity path ride the
**severity** grid (`xs_sev`), not the output grid, then re-enable windowing for
occ-reins books. This now also aligns with numerics-2/3's native-grid reinsurance
views (windowed-world note §Reinsurance: gross/ceded/net carry their own origins).
Until it lands, keep the interim discoverability note (`applies=False`, one
`logger.info`).

## Step 0 — audit / re-base

The Scope section below is the authoritative record of this work (the original
`plan-window-port-bv` draft was untracked and is gone — there is nothing to recover).
Re-base it against the landed numerics-2/3 code, expanding the step list where
kappa / `p_{unit}` / objective columns are referenced, and read the then-current code
as the source of truth (any line numbers carried in from the old draft are stale).
Confirm which
of its decisions still hold (mixed-book windowing, coverage-nines split) and which
are now moot (per-column masking).

## Invariants / tests (carried + extended)

> The portfolio-combine invariants below (2-unit high-mean book, mixed book,
> `Σ kappa_i == x` on the windowed combine) now belong to **1P in
> bucket-window-2**; listed here as the upstream guarantees this plan relies on.

- *(1P)* 2-unit high-mean book → shared `x_min>0`, `bs` finer than 0-based, `p_total` mass 1,
  moments match; vs forced `x_min=0` legacy combine → moments agree.
- *(1P)* mixed high-mean/ordinary → total windows, `p_total` right; ordinary unit's
  `aligned_unit_density_df` column off-window/empty (not wrapped garbage), its own
  `density_df` correct on its native window.
- *(1P)* **`Σ_i kappa_i(x) == x`** on a *windowed* combine (ties numerics-2's anchor into the
  windowed grid — the key cross-plan check).
- bivariate: each axis windowed, grid far smaller than 0-based `[0,vmax]²`, both
  marginals match their 1-D windowed aggregates; mixed bivariate windows one axis,
  0-bases the other; memory check (resolves a high-mean joint the `cap_log2=14` grid
  cannot).
- occ-reins windowed book (post-P0): mass/moments right; gross/ceded/net combine via
  native origins.
- legacy regression byte-stable for non-windowed books.

## Files

- `src/aggregate/multivariate.py` — `size_axis` windowed variant, per-axis roll.
- `src/aggregate/distributions.py` — occ-reins on `xs_sev` (P0), per-unit window
  metadata.
- `tests/test_multivariate*.py`.
- *(upstream, bucket-window-2 / 1P:* `src/aggregate/portfolio.py` `best_window`,
  `update` windowed routing; `tests/test_bucket_sizing.py`.*)*

## Housekeeping

Version bump; CHANGELOG (windowed portfolio combine + bivariate per-axis windowing;
occ-reins windows again if P0 lands); `dev/TODO.md` W2/M-track + the former
window-port-bv deferred follow-ups closed. Move this plan to `dev/done/` on landing.
(The window-port-bv draft was already removed at drafting time — folded in here.)

## Out of scope

- Copula/dependence beyond current `multivariate`; 3+ dimensions (ship 2-D first).
- The user's negative/subset-grid **allocation** math that originally parked the
  window-port-bv draft — settle that (it informs how a windowed/subset book
  *allocates*, which is a numerics-2/3 concern) before finalizing Part B's allocated
  views.
