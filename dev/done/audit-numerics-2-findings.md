# Step-0 audit findings — plan-numerics-2-objective

Produced by `dev/audit-numerics-2.py` (run on pre-change code, 1.0.0a55,
commit 50f1d0a) which force-runs today's `add_exa` on signed / windowed books
and measures every objective column against an **exact brute-force
convolution** that carries unit origins explicitly. Pre-change anchors: full
pytest green (1452 passed), `test_baseline.py` green against the a27 capture,
spot-checks captured to `tests/baseline/data/spotchecks.json`.

## Audit books

- **A — signed discrete** (exact reference): three `dfreq×dsev` units, two with
  negative atoms; total window `[-8, 247]`, `bs=1`.
- **B — signed lognorm-shift mix**: two `ssev lognorm − shift` units + one
  ordinary loss line; total window `[-335.75, 3760]`, `bs=1/4`. (Top of the
  combined support exceeds the window, so far-tail rows carry ~1e-8 wrap mass —
  a *combine sizing* artifact owned by numerics-4, noted where it contaminates.)
- **C — two tight positive units** (cv .05/.04) through the forced signed
  combine. The auto-sizer still chose a zero-origin grid (true nonzero-origin
  *positive* windows arrive with the numerics-4 combine), but the tight units
  expose the **spectral-division noise** path measurably.
- **D — disjoint-scale units** (means 1e6 and 100): same zero-origin outcome;
  division-path noise again visible on the dominant unit.

## Column → assumption → verdict → fix

| Column | Hidden assumption | Measured verdict | Fix (this plan) |
|---|---|---|---|
| `p_total` | — | ok everywhere (A: 1.7e-16; C: 2.5e-12) | untouched (byte-equal gate) |
| `exeqa_{line}` | `ft(df.loss·p_unit)` needs the unit pmf *on the total grid* | A: **correct** (1.9e-15) — the rolled `p_{unit}` happens to compensate the total origin; B: far-tail rows contaminated (see correction 2 below); C: ~2e-9 (conditioning limit at `p≈1e-9` rows — the initial 4e-7 verdict was a *reference* artifact, correction 1) | shifted-support kappa from **native** pmfs (`ft_xp` captured at combine time); `ft_not` via prefix/suffix where a spectrum has exact zero bins, division otherwise |
| `exa_total` / `lev_total` | `cumsum(S)·bs` integrates from 0 | A: **wrong, rel err 1.33** (missing `x0`) | direct sum `Σ x·p + x·S` |
| `exlea_total` | `(exa − loss·S)/F` + `loss_max` blanking | A: **wrong, 1.3e2**; C: 2.3e-8 (blanking overshoot) | `cum_x/F` with explicit `F ≤ tol` guard |
| `exgta_total` | `loss + (e − exa)/S` | A: **wrong, 3.4e2**; C: 3.6e-3 right-edge `S→0` noise | `(e − cum_x)/S` with `S ≤ tol` guard |
| `exlea_{line}` | `loc[0:loss_max] = 0` blanking assumes left edge = noise region | A: **wrong, 0.67** — blanks *material* rows on a signed grid; C/D: ~4e-4 (`mult·bs` buffer overshoots into live region) | `cum_xi/F` + `F ≤ tol` guard; heuristic deleted |
| `exgta_{line}` | sound given aligned `p_{unit}` (`e_col` from total grid) | A: ok (2.2e-12); breaks when a unit window clips | `e_i` from the **native** unit pmf |
| `lev_{line}` | `(1−cumsum p_unit)` cumsum`·bs` from 0 | A: **wrong, 3.8–12×** | native capped sums carrying the origin: `cum_xp[i] + a·(1−cum_p[i])` |
| `exi_xeqa/exi_x/exi_xlea/exi_xgta_{line}`, `exa_{line}` | `kappa/loss` is a recovery share | undefined on signed grids (divides through 0); C/D: 1.1–2.7e-3 right-edge noise (last-row NaN + `S` division) | positive-loss guard: **NaN on signed grids** (steering 6); direct tail sums + `S ≤ tol` guard otherwise |
| `loss_max` / `mult ∈ {1,10,100}` heuristic | zero-origin index; left edge = junk | see `exlea` rows above | replaced by explicit `F/S/|x| ≤ tol` guards |
| `df.loc[0, …]` sites | 0 is on the grid | `KeyError` on a positive-origin window (by inspection; C kept origin 0) | masks on `|loss| < bs/2` |
| `epd_0_*`, `epd_1_*`, `e1xi_1gta_*` (`add_exa_details`) | zero-origin `1/loss` | no callers anywhere | **deleted** (D4) |
| `Aggregate.density_df` `lev/exa/exlea/exgta` | same `cumsum(S)·bs` / `(e−exa)/S` idioms | same failure modes as the portfolio totals | direct sums carrying `x0`; `epd` column **deleted** |

Key confirmations:

- the shifted-method invariant `Σ_i κ_i(x) = x` holds to 7e-15 on the exact
  signed book once kappa is right — it is the correctness anchor the tests use;
- `exa_{line}` computed as layer integral agrees with the direct sum on
  zero-origin books (C/D: ~2e-11), so the rewrite is a pure refactor there.

## Corrections found during execution (measure, don't guess — twice)

1. **The spectral-division "noise" verdict on C/D was a reference artifact.**
   The first brute-force reference trimmed unit pmfs at 1e-13; un-trimmed
   (full-vector `fftconvolve`), division and prefix/suffix agree with the
   reference equally well (1.5e-9 vs 1.9e-9, the conditioning limit at
   `p ≈ 1e-9` rows). Division `(a·b)/a = b(1+O(eps))` is per-bin
   well-conditioned even on deeply underflowed spectra. The exact-zero-bin
   test is therefore retained (preserving byte-parity of the legacy pricing
   surface — an experimental `min|ft| < FT_NOISE_FLOOR` switch moved captured
   `exag`/`pricing_at` readouts by ~5e-7 via tail-noise reorganization with
   zero accuracy benefit); prefix/suffix replaces only the `O(m²·M)` rebuild
   on exact-zero spectra.
2. **Book B's residuals are a unit-window deficit, not an allocation error.**
   Unit A's own window truncates 1.2e-8 of real left-tail mass (Poisson can
   stack 9+ claims × the −25 shift below `x_min(A)`); the portfolio combine
   retains that mass in the wrapped buffer, so `p_total` and the brute
   reference disagree by exactly the deficit, amplified in the
   ratio/conditional columns. The unit already warns
   (`DefectiveDistributionWarning`); window sizing is numerics-4's problem.
   On clean-window books (A, C, D) the post-change columns are exact to
   conditioning limits.

## Post-change verification (1.0.0a56)

Book A (signed discrete): every column ≤ 2.3e-12 vs exact brute force,
`Σκ = x` to 3.8e-16. Books C/D: kappa ~1e-9 (at `p≈1e-9` rows), `exa_*`
1.6e-11, totals/lev exact; residual 1e-5..1e-4 entries sit at rows where
`F`/`S ≈ 1e-7` (intrinsic conditioning of the ratio, identical for any
implementation). Regression: baseline key columns byte-stable except the
expected ≤2.8e-12 Aggregate `lev` drift (recaptured); derived spot-checks
≤7.5e-12 (`exgta` cancellation) / ≤2.5e-14 (everything else); full pytest
1483 passed.

## Out-of-scope notes (flagged, not fixed here)

- Unit-window deficit on Book B (combine/window sizing) → numerics-4.
- `_build_augmented` / `apply_distortion` on signed frames: still zero-origin
  (`df.loc[0, …]`, `cumsum·bs`); numerics-2 adds a clear refusal on signed
  books, the rewrite is numerics-3.
- The genuinely windowed (nonzero positive origin) *combine* never triggers
  from `update` today (the auto-sizer kept origin 0 even for Books C/D); the
  new add_exa handles arbitrary origins and is unit-tested against a
  hand-built positive-origin frame, but live windowed books arrive with
  numerics-4.
