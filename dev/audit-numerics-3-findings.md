# Step-0 audit findings — plan-numerics-3-distortion

Produced by `dev/audit-numerics-3.py` (run on pre-change code, 1.0.0a56,
commit e20b131) which measures today's distorted surface
(`_build_augmented`, `price` linear+lifted, `Aggregate.apply_distortion`)
against the exact-discrete Choquet reference of `choquet-calc-method.md`
(`gp = g(T) − g(S)`, `ρ(X∧a) = Σ min(x,a)·gp`). Pre-change anchors: full
pytest green (1487 passed; 3 pre-existing `test_style.py` failures from the
a56 style-sheet edit, unrelated), `test_baseline.py` green, and the corpus
`pricing_at` / `pentagon_at` / `price` scalars pre-captured to
`tests/data/numerics3_precapture.json` (regression targets for
deliverable 4).

## Audit books

- **Bodoff** — bounded zero-origin discrete two-unit book (exact, hand
  checkable); distortions `dual 1.85`, `tvar 0.65`, `ccoc 0.10` (mass),
  `tvar 0` (identity).
- **One-line dice port + matching Aggregate** — the six-surface
  consistency target.
- **CNC** — unbounded zero-origin (gamma/lognorm mixed) book; cache and
  mass-guard probes.
- **Synthetic shifted dice** — positive-origin atom table (`x0 = 101`);
  live positive-origin *total* grids do not arise until numerics-4
  (Plan A clamps non-signed origins to 0), so the exposure is shown on
  the formula directly.
- **Signed discrete book** (numerics-2 Book A): three units, total
  straddles 0.

## Column → assumption → verdict → fix

| Distorted column / surface | Hidden assumption | Measured verdict | Fix (this plan) |
|---|---|---|---|
| `gS`, `gp_total` (`-diff(gS, prepend=1)`) | `S` strict tail, sorted rows | **ok everywhere** (Bodoff 0; CNC ≤ 4e-16) — already the exact `g(T)−g(S)` atom weights | keep formula, route through the one Choquet helper; document as exact atom weights |
| `exag_total = gS.shift(1).cumsum()·bs` | `loss[0] == 0` | zero-origin: ok (≤ 6.6e-15 Bodoff, 6.4e-13 CNC); positive-origin: **wrong, rel err 0.97**; signed: **wrong, rel err 1.00**; `+x0` repairs both to ≤ 9e-16 | direct sums carrying `x0`: `cumsum(x·gp) + x·tail(gp)` |
| `exi_xgtag_{line}` (beta, shift-trick reverse cumsum / `gS`) | positive loss; far tail reliable | Bodoff: exact (≤ 1e-15); CNC: **2.2e-8 absolute** noise in far-tail rows where `gS ≲ 1e-7` (ill-conditioned legacy formula; `exag_*` built from it still fine at 1e-13) | direct tail sums `Σ_{j>k} share_j·gp_j / gS_k` with `Seq0` guard; far-tail beta moves at ~1e-8 absolute (derived column, spot-check gate) |
| `exag_{line}` (lifted, `(beta·gS).shift.cumsum()·bs`) | `loss[0] == 0`, `kappa/loss` share | zero-origin: ok (≤ 6.8e-15 Bodoff, 7e-13 CNC) vs exact `cumsum(κ·gp) + x·tail(share·gp)` | one O(n) sweep `cumsum(κ·gp) + x·gS·TAIL`, TAIL = beta (lifted) / alpha (linear) |
| `price(allocation='linear')` (`_collapsed_exeqa` + layer integrals) | zero-origin, uniform grid | **no convention gap**: total and per-line P match the exact collapsed-law `ρ` / `Σκ·gp + a·share·gp_atom` to ≤ 3.9e-13 (Bodoff exact, CNC fp-drift) | engine deleted; `price` reads rows of the unified frame; 1e-14 regression vs pre-capture is feasible |
| `_build_augmented` truncation `idx = int(index/bs + 1)` | `index == k·bs` | benign today (zero-origin only reachable); breaks positional addressing on any shifted window | positional `iloc` indexing |
| `df.loc[0, 'T.Q_*'] = 0` | 0 on the grid | benign today; adds a phantom row on a windowed index (by inspection) | origin-row mask |
| cache key (`distortion.name` only) | one view/S-calc per name | **confirmed stale**: `ask` then `bid` returns the identical cached frame | key `(name, view, role-flag, S_calculation, allocation)` |
| mass guard (in `price(lifted)` only) | — | **confirmed G6**: `apply_distortion(ccoc)` on unbounded CNC builds the unstable frame with no complaint | guard moves down into the builder (lifted); linear stays available (collapsed law is bounded by construction — see note below) |
| `Aggregate.apply_distortion` (`hstack((0,gS[:-1])).cumsum()·bs`) | zero-origin; no bid view, no `gp`, no mass guard, cached nowhere | dice: ok (3.4e-16) — zero-origin only | route through the Choquet helper; full `view × value_type`, mass guard |
| signed total pricing | — | `dot(x, gp) == Distortion.price(dx) == Σ_i dot(κ_i, gp)` to 1.7e-15 on the signed book (steering 6 verified: the engine is orientation-agnostic) | builder prices signed totals via `dot(κ, gp)`; equal-priority share columns stay positive-loss-guarded |

## Consistency target (six surfaces)

On the legacy zero-origin one-line dice book, **all six surfaces already
agree to ≤ 5.6e-16** (`Distortion.price` dx/ds, `Port exag_total(a)`,
`Port.price` lifted/linear, `Agg exag(a)`, `Agg.price P`). The blockers
are confined exactly where the notes predict: the missing `x0` term
(positive-origin/signed grids — unreachable or refused today), the
name-only cache key, and the mass-guard placement. The plan's target is a
consolidation, not a numerical repair, for every reachable legacy book.

## Notes settled during the audit

1. **The old linear engine has no bucket-convention gap.** Its
   double-reverse-cumsum layer integral reproduces the exact discrete
   collapsed-law Choquet values to fp precision on both an exact discrete
   book and a 64k-row FFT book. The 1e-14 regression gate on the linear
   `price` scalars is therefore achievable by the direct-sum rewrite.
2. **Mass distortion + unbounded support invalidates only the *lifted*
   tail split.** `ρ(X∧a) = Σ_{k≤a} x·gp + a·g(S(a))` never depends on
   *where* in the tail the mass-atom weight lands, so `exag_total` and the
   alpha (linear) split are stable; beta integrates `gp` across tail
   states and inherits the top-bucket artifact. Hence the builder-level
   guard (G6) applies to the **lifted** frame; the **linear** frame stays
   available with mass distortions (preserving the captured
   `Port.CNC × ccoc × linear` baseline) with beta columns blanked.
3. **Far-tail beta noise on unbounded books.** The legacy shift-trick
   `exi_xgtag` carries ~2e-8 absolute noise where `gS ≲ 1e-7`; the
   direct-sum replacement changes those rows. `exag_*` (the integrals) are
   unaffected at 1e-13. Spot-check gates accordingly.

## Post-change verification (1.0.0a57)

- **Six-surface consistency target** holds to ≤ 1.2e-16 on the one-line
  dice book for dual and tvar (`Distortion.price` dx == ds == Portfolio
  `exag_total(a)` == `price` lifted == linear == Aggregate `exag(a)` ==
  `Aggregate.price P`).
- **Lifted surfaces vs the a56 pre-capture**
  (`tests/data/numerics3_precapture.json`): Bodoff ≤ 1.1e-14; CNC /
  Bounded ≤ 9e-13 (`pricing_at`, `pentagon_at`, `price(lifted)` per-line
  L/M/P/Q and totals) — fp order-of-ops drift only.
- **Linear totals** ≤ 2.3e-12 vs a56; linear **per-line** cells moved as
  predicted by the convention unification (Bodoff M up to a few units on
  small margins; CNC/Bounded Q 0.1–6% — the old `rcoc` capital engine and
  the `X ≥ a` atom merge are gone). Corpus `price__*__linear` frames
  recaptured; everything else in the corpus baseline passed against the
  a56 capture before recapture (augmented frames, `pricing_at`, lifted
  price all within rtol 1e-12 / atol 1e-14).
- Two execution findings worth recording (measure, don't guess — again):
  (1) zeroing tiny *negative* p fuzz before the cumsum broke the exact
  `S = 0` cancellation at the essential sup, which a mass distortion
  amplifies by `g(0+) − g(0)` — the fuzz is now kept (validated only);
  (2) for the same reason an `S ≤ tol → 0` snap is wrong on bounded books
  whose genuine tail probabilities pass through the dust range — the
  helper renormalizes only when the sum is not exactly 1 and never snaps.
- The per-line beta tail sums close with a collapsed atom at the
  exeqa-reliability cut (`share_cut · g(S_cut)`), the exact analogue of
  the legacy fill-value trick; without it the distorted tail mass beyond
  the cut lands on junk shares (measured: 13–17% errors on bounded-book
  per-line `exag` under ccoc).
- Items for the author:
  - `DEFICIT_MATERIALITY = 1e-4` is a judgment call (1e-6 tripped on a
    casual log2=14 lognorm book with deficit 1.5e-5; the Def.Pareto
    showcase at 7.5e-3 still raises). Review the level.
  - Under the identity distortion the margin-proportional layer
    construction allocates **zero** per-line capital (no margin to
    allocate) while total capital is real; the old code emitted NaN
    there. Documented in `_line_capital_at`.
  - `analyze_distortions` now *skips* mass-on-unbounded members of a
    sweep with a `UserWarning`; doc exhibits that swept the calibrated
    quintet on unbounded books will lose their `ccoc` rows on rebuild.

## Pre-captured regression targets

`tests/data/numerics3_precapture.json`: for each corpus portfolio
(`Port.CNC`, `Port.Bounded`, `Port.Bodoff`) × distortion (`ccoc`, `dual`,
`tvar`): `pricing_at(p=0.99)` L/M/P/Q per line, `pentagon_at` L/M/P/Q per
line, and `price` (price + per-line L/M/P/Q) per allowed allocation
method. `pricing_at`/`pentagon_at` skipped for `Port.CNC × ccoc` (the
post-change lifted builder refuses mass-on-unbounded by design). The
augmented-frame baselines (`loss/S/gS/gp_total/exag_*`) were already
captured by the standing harness (`tests/baseline/`).
