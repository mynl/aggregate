# Plan numerics-4 — windowed portfolio combine + bivariate per-axis windowing

> Part of the numerics program; see `plan-numerics-0-meta.md`. **Supersedes the
> former `plan-window-port-bv` draft** — that draft was untracked and has been
> removed, so this plan (the Scope section below in particular) is now the **sole
> record** of its Part A (windowed portfolio combine), Part B (bivariate per-axis
> windowing), and P0 (occ-reins × windowing), re-based onto the kappa-ready
> foundation that `plan-numerics-2-objective` establishes. Depends on numerics-2
> (and numerics-3 for the priced/allocated views of a windowed book). **Last in the
> program.**

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

### Part A — windowed portfolio combine
- `best_window`: non-signed windowed origin `x_min = Σ x_min_k` (floored), `bs` from
  summed band **widths** — generalise the hard-coded `x_min=0` non-signed branch.
- `update`: route a windowed non-signed book through the existing roll-combine
  (discriminator `signed or windowed`); drive each unit with its **explicit phase-1
  origin** (not `x_min='auto'`), since a pinned shared `bs` won't re-fire the
  strictly-finer gate.
- Mixed books: window the total whenever it clears 0; off-window unit *views* read
  empty/native via `unit_density` (no false wrapped mass in any core column —
  reframed from the old per-column masking, which no longer applies since
  `p_{unit}` is gone from `density_df`).

### Part B — bivariate per-axis windowing (the goal)
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

- 2-unit high-mean book → shared `x_min>0`, `bs` finer than 0-based, `p_total` mass 1,
  moments match; vs forced `x_min=0` legacy combine → moments agree.
- mixed high-mean/ordinary → total windows, `p_total` right; ordinary unit's
  `aligned_unit_density_df` column off-window/empty (not wrapped garbage), its own
  `density_df` correct on its native window.
- **`Σ_i kappa_i(x) == x`** on a *windowed* combine (ties numerics-2's anchor into the
  windowed grid — the key cross-plan check).
- bivariate: each axis windowed, grid far smaller than 0-based `[0,vmax]²`, both
  marginals match their 1-D windowed aggregates; mixed bivariate windows one axis,
  0-bases the other; memory check (resolves a high-mean joint the `cap_log2=14` grid
  cannot).
- occ-reins windowed book (post-P0): mass/moments right; gross/ceded/net combine via
  native origins.
- legacy regression byte-stable for non-windowed books.

## Files

- `src/aggregate/portfolio.py` — `best_window`, `update` windowed routing.
- `src/aggregate/multivariate.py` — `size_axis` windowed variant, per-axis roll.
- `src/aggregate/distributions.py` — occ-reins on `xs_sev` (P0), per-unit window
  metadata.
- `tests/test_bucket_sizing.py`, `tests/test_multivariate*.py`.

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
