# Plan: support signed loss severity in a `pnl` aggregate (option B)

## Goal

Make `pnl` (premium-minus-loss, the aggregate-level affine `PnL = shift − A`)
correct when the **loss severity is itself signed** (a `dsev` with a negative
atom, or `ssev`). Today this silently produces garbage:

```python
ap = build('pnl GP 5 premium - dfreq[3] dsev[-1 1]', bs=1)
# describe "Est EX" = -32763.75   (should be 5)
# stats_df empirical col = ±32768  (≈ 2**15 grid-index garbage)
# DefectiveDistributionWarning: PMF deficit 5.000e-01
# final P&L support {2,4} (mass 0.5); the {6,8} states are dropped
```

Correct result: `P&L ∈ {2,4,6,8}` with probs `{⅛,⅜,⅜,⅛}`, mass 1.0,
`mean = 5`, `sd = √3 ≈ 1.732`.

## Root cause (from the investigation)

A `pnl` is built on the design assumption that **the loss severity is
non-negative** — the affine handles the sign at the *aggregate* level. Two
signedness notions coexist:

- `_signed_severity()` — the *convolution-grid* gate (negative-atom severity).
- `_agg_affine_active()` — the `pnl` reflect/shift wrapper.

The bug is the interaction **both true**. At grid build
(`distributions.py:4107-4112`) the affine path **hard-codes a 0-based loss grid**:

```python
if self._agg_affine_active():
    xs = np.arange(0, N, dtype=float) * bs     # 0-based, ignores signed severity
```

So a signed loss (`L ∈ {−3,−1,1,3}`) has its negative atoms **wrapped to the top
of the FFT grid** (indices ~65535). Consequences, both observed and arithmetic-
matched:

1. **Empirical moments** (`est_*`, computed pre-affine at ~4259-4284 from
   `self.xs`×`agg_density`) weight the wrapped negatives by their index:
   `est E[L] = .375·1 + .125·3 + .375·65535 + .125·65533 = 32768.75`; the affine
   then flips it (`est_m = shift − est_m = 5 − 32768.75 = −32763.75`). Both match
   the reported numbers exactly.
2. **Mass deficit.** `_apply_agg_affine`'s reverse-and-roll keeps only
   `valid = (t>=0)&(t<N)`; the wrapped mass maps to `t<0` and is dropped → the
   0.5 deficit and the truncated `{2,4}` support.

The **plain** `agg L dfreq[3] dsev[-1 1]` is already correct (signed grid,
`x_min=−8`, mass 1, `est_m=0`) — proof that the signed-severity convolution path
works; only the `pnl` affine forces the wrong grid.

## Design: separate the *loss* grid from the *P&L display* window

For a signed-loss `pnl` there are genuinely **two two-sided grids**:

- the **loss convolution grid** `[loss_lo, loss_hi]` (negative origin), exactly
  what the plain signed agg already builds; and
- the **P&L display window** `[pnl_lo, …]` produced by `_apply_agg_affine` +
  `_pnl_window`.

The current code conflates them (the affine path assumes loss origin 0 and lets
`_bs_window` hand back the P&L origin in `x_min`). The fix is to thread the loss
origin through and let the affine relabel do its existing reverse-and-roll from a
*signed* loss grid (its formula already reads `loss_x0 = self.xs[0]` and
`j_loss = round(loss_x0/bs)`, so it is **already origin-agnostic** — it just
never receives a signed loss grid today).

### Steps

1. **Grid build (`update`, ~4107-4112).** Gate the 0-based force on
   *non-signed* loss only:
   ```python
   if self._agg_affine_active() and not self._signed_severity():
       xs = np.arange(0, N) * bs               # unchanged: ordinary pnl
   elif self._agg_affine_active():             # signed-loss pnl (new)
       xs = loss_lo + np.arange(0, N) * bs     # genuine signed loss grid
   else:
       xs = x_min + np.arange(0, N) * bs
   ```
   This needs the **loss** origin `loss_lo`, not the P&L origin. So:

2. **`_bs_window` for affine-active** must return (or expose) the *loss* grid
   origin for the signed-loss case, alongside the P&L display origin. Cleanest:
   have the affine branch of `_bs_window` compute the loss window (two-sided, via
   the same `_size`/`estimate_agg_window` path the plain signed agg uses) and
   return that as `x_min`; `_apply_agg_affine` already computes the P&L window
   independently via `_pnl_window`. (Confirm `_bs_window`'s affine branch and the
   `_size(signed=…)` call so the loss window is built on loss moments, not P&L.)

3. **`_pnl_window` (4410-4456)** already sizes from the **affine moments**
   (`m = shift − agg_m`, `sd`, `skew`) via `estimate_agg_window`, which is
   two-sided — so it should already cover the reflected signed-loss spread *once
   the loss grid is correct*. Verify the `valid` mask in `_apply_agg_affine`
   (4514) now keeps **all** mass (the dropped-tail branch should only ever shed a
   genuine far-tail residual, never half the distribution). Add an assertion /
   validation tie-in if the kept mass falls below `1 − WINDOW noise`.

4. **Empirical moments.** No change needed *if* steps 1–2 land: with a signed
   loss grid, the pre-affine `est E[L] = 0`, and the affine flip gives
   `est_m = shift − 0 = 5`, `est_sd = √3`, `est_skew = 0`. The garbage disappears
   as a consequence, not via a separate patch. Confirm the est-moment block reads
   the corrected `self.xs`.

5. **Non-signed `pnl` is byte-for-byte unchanged** — every new branch is gated on
   `_signed_severity()`, which is `False` for an ordinary pnl. This is the
   invariant the freeze/check harness guards.

## Sequencing — this lands FIRST, standalone (decision)

**Ship this before the bucket work.** It does *not* depend on
`dev/plan-bucket-sizing.md`: the two-sided loss-window sizing it reuses is the
**existing** signed-severity path (`_bs_window` / `_size(signed=True)` /
`estimate_agg_window`) that already produces correct results for the plain
`agg L dfreq[3] dsev[-1 1]` today. Nothing in steps 1–5 requires the bucket-plan
changes; they only touch how the *affine* path picks up that already-working loss
grid.

Known overlap to keep on the radar (handled later by the bucket plan, **not** a
blocker here):

- A **Portfolio of pnl units** that includes a signed-loss pnl will size its
  shared grid through the same combine logic the bucket plan is fixing (the RMS
  `best_bucket` bug). A *single* signed-loss `pnl` aggregate — the target of this
  plan — does not go through that path and is fixed independently. Books of such
  units get fully correct sizing when the bucket plan lands; until then they are
  no worse than any other portfolio.
- The bucket plan's width-based sizing (its constraint 2) and this plan touch the
  same `_bs_window`/`_pnl_window` surface, so whoever does the bucket work second
  should re-baseline `scripts/bucket_baseline.py` and re-run freeze/check after.

## Verification

1. **Targeted correctness** — the motivating program:
   ```python
   ap = build('pnl GP 5 premium - dfreq[3] dsev[-1 1]', bs=1)
   ```
   assert: no `DefectiveDistributionWarning`; `ap.agg_density.sum() ≈ 1`; support
   `{2,4,6,8}` with probs `{.125,.375,.375,.125}`; `describe` "Est EX" ≈ 5,
   "Est SD" ≈ √3, "Est Sk" ≈ 0; `stats_df` empirical column sane (loss `E[L]≈0`,
   `sd≈√3`). Also a second case with a non-symmetric signed `dsev` and a
   continuous `ssev` pnl.
2. **Non-signed pnl unchanged** — build an ordinary `pnl` (non-negative loss) and
   assert identical `bs`/`log2`/density to pre-change (this is what freeze/check
   covers globally).
3. **Portfolio of pnl** including a signed-loss unit — assert mass conservation
   for the unit's own density. (A fully-correct *shared* combine grid tracks with
   the bucket plan; here just confirm the signed-loss unit itself is no longer
   garbage when placed in a book.)
4. **`uv run pytest`** green.
5. **`scripts/freeze_knowledge.py`** freeze-before / check-after — expect
   **all-match** (no existing test_suite object is a signed-loss pnl; if one is,
   it's currently garbage and *should* change — confirm which).
6. **`scripts/bucket_baseline.py`** diff if the bucket-window-width work lands in
   the same branch.

## Housekeeping (standing rules)

- Bump `pyproject.toml` `1.0.0a*`.
- `CHANGELOG.md`: "`pnl` now supports a signed loss severity (`dsev` with a
  negative atom / `ssev`): the loss convolves on its genuine signed grid before
  the affine relabel, fixing the previous mass deficit and ±2¹⁵ empirical-moment
  garbage." Note it's a bug fix, not breaking.
- `dev/TODO.md`: one-line under the appropriate track; cross-reference the bucket
  plan.
- Append the motivating program (and the second signed-loss case) to
  `src/aggregate/agg/test_decl.agg` under the matching section (DecL-sync rule).
- Move this plan to `dev/done/plan-pnl-signed-severity.md` on landing.

## Resolved (author decisions)

1. **Sequence, not coordinate** — land this **first**, standalone (see Sequencing
   above). Get signed-loss `pnl` working soon; the bucket plan follows.
2. **Far-tail spread > N·bs stays a *warning*** (the existing `_pnl_window`
   deficit branch), not an error — but **keep it on the radar**: add a `dev/TODO.md`
   line so the genuinely-unrepresentable case isn't forgotten.

## Out of scope

- The FFT convolution core and the plain signed-severity path (already correct).
- Option A (reject signed-loss pnl) — superseded by this plan.
- Pricing/allocation columns on signed support (`add_exa`) — already a separate
  known gap (`dev/plan-portfolio-neg-x-pricing.md`); this plan only fixes the
  density/moments.

---

## Execution notes (landed 1.0.0a45)

The diagnosis and design held exactly; the implementation of steps 1–2 was
**simpler than written** and is recorded here.

1. **Steps 1–2 collapsed: `_bs_window` already computes the loss origin.** The
   plan proposed a new 3-way branch in `update` plus threading a separate
   `loss_lo`. In fact `_bs_window` *already* sizes a correct signed loss window
   for a signed severity — its selected origin `sel_x0` is the loss origin (e.g.
   `exact_discrete` returns `x_min = -3` for `dfreq[3] dsev[-1 1]`). The affine
   branch was simply **discarding** it (`ret_x0 = x0_pnl`, the P&L display
   origin) and `update` then hard-coded a 0-based loss grid. The fix:
   - `_bs_window` affine branch returns `ret_x0 = sel_x0 if _signed_severity()
     else 0.0` — the loss origin for a signed-loss pnl, **0 for an ordinary
     pnl** (byte-for-byte the legacy 0-based grid, and ignoring any `x_min_in`
     override exactly as before). `x0_pnl` is still computed for the `used`
     display row.
   - `update`'s affine special-case is **removed**: the loss grid is now built
     uniformly as `xs = x_min + arange(N)*bs` (for an ordinary pnl `x_min == 0`,
     so identical to the old `arange(N)*bs`). `_apply_agg_affine`, already
     origin-agnostic (`loss_x0 = self.xs[0]`), relabels the signed loss onto the
     P&L window with no change.
   This is strictly less surface area than the plan and keeps the non-signed
   path provably unchanged (freeze: all 146 knowledge-base objects match 1e-12).

2. **Step 3 mass check added as a warning.** `_apply_agg_affine` now compares
   `loss_d.sum()` to `pnl_d.sum()` after the reverse-and-roll and emits a
   `DefectiveDistributionWarning` when more than `VALIDATION_NOISE` is dropped
   off the P&L window — the same threshold/treatment as the loss-FFT deficit
   check. This surfaces two **pre-existing, previously-silent** far-tail drops on
   *non-signed* continuous pnls (`test_mv_pnl_marginal…` ~3e-8;
   `test_describe_finite_for_mean_zero_pnl` ~1.7e-7) — exactly the
   "far-tail spread stays a *warning*" case the author resolved (item 2). Both
   tests still pass.

Verification: motivating program yields support `{2,4,6,8}`, probs
`{.125,.375,.375,.125}`, mass 1, mean 5, sd √3, skew 0, **no warning**; plus a
non-symmetric signed `dsev` (mean closed form) and a continuous `ssev` pnl, and a
signed-loss pnl unit inside a Portfolio (own marginal mass-conserving) — all in
`tests/test_pnl.py`, mirrored in `src/aggregate/agg/test_decl.agg`. Full suite
**1091 passed**; freeze/check **all 146 match within 1e-12**.
