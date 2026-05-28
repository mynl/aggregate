# Plan: before/after baseline consistency harness

> Status: **draft for review.** No code yet. The companion narrative is in
> `dev/pipeline-portfolio.rst` (Proposal — before/after consistency harness);
> the aggregate and portfolio refactor plans both sequence "harness first" and
> point here. **The thing to review is the DecL corpus in §3** — grids are a
> first guess and want the author's eye.

---

## 1. Purpose

Lock in the *current* numbers in the big dataframes (`stats_df`, `describe`,
`density_df`, `augmented_df`) and the headline risk measures **before** the
under-the-hood refactor starts, so any unintended drift is caught immediately.
The computation is deterministic (no RNG, no calibration in the baseline path —
see §4), so we can capture a golden baseline from `REFACTOR` HEAD and assert
equality to a **very tight** tolerance.

This is a **characterisation** test (freeze behaviour), distinct from the
existing correctness tests (theoretical-vs-empirical validation). When the
refactor *intends* to move a number (e.g. Portfolio adopting the de-fuzzed
`xsden_to_mwrangler` convention, or the `_build_augmented` ROE-fallback fix), the
baseline is regenerated **deliberately**, in the same commit, with the change
called out.

---

## 2. Determinism rules (so "very very close" is achievable)

1. **Pin `(log2, bs, padding)` per case** — never call the recommender (it may
   itself be refactored). `bs` is always an exact binary fraction.
2. **No RNG in the baseline.** `Portfolio.sample` draws random numbers — do
   **not** use it. The switcheroo case uses a small **fixed, hand-built** sample
   (committed as data), not a random draw.
3. **No calibration root-solve in the primary snapshots.** Apply **fixed-shape**
   distortions (below), not `calibrate_distortions`, so there is no Newton
   iteration whose last ULPs can wander across SciPy versions. (A *separate*,
   looser-tolerance calibration snapshot can come later if wanted.)
4. **Record environment** in the manifest: `aggregate`, `numpy`, `scipy`,
   `pandas` versions. FFT / `interp1d` can move by ULPs across releases.

Fixed distortions used everywhere:

| role | distortion | mass? |
|---|---|---|
| with a mass | `Distortion('ccoc', 0.10)` (cost of capital 10%) | **yes** (jump at `s=0`) |
| no mass | `Distortion('dual', 1.85)` (dual) | no |
| no mass | `Distortion('tvar', 0.65)` (tvar) | no |

Asset level for readouts: `p = 0.99` (and `0.999`); `a = q(p)` is a deterministic
index lookup.

---

## 3. The corpus  *(review me — grids are first-guess)*

Small on purpose: max-digit precision of a deterministic pipeline needs coverage
of code *paths*, not statistical breadth. **5 aggregates + 3 portfolios.** Written
as the `corpus.py` that the capture script will consume.

```python
# tests/baseline/corpus.py  — THESE LOOK REASONABLE AS A START

# Each agg case: (program, dict(log2=, bs=, padding=, normalize=))
# Each agg case: (program, dict(log2=, bs=, padding=, normalize=))
AGG_CASES = {
    # 1a/1b: fixed-1 shortcut vs full-FFT path — must agree to ~eps.
    "Base.FixedOne":
        ("agg Base.FixedOne 1 claim sev lognorm 100 cv 0.5 fixed",
         dict(log2=16, bs=1/32, padding=1, normalize=True)),

    "Base.DfreqOne":
        ("agg Base.DfreqOne dfreq [1] sev lognorm 100 cv 0.5",
         dict(log2=16, bs=1/32, padding=1, normalize=True)),

    # 2: symmetric discrete — zero skew, fuzz-sensitive (validation regression).
    "Sym.Dice":
        ("agg Sym.Dice dfreq [1] dsev [1:6]",
         dict(log2=5, bs=1, padding=1, normalize=True)),

    # 3: thick tail — exercises tail accuracy / wide grid.
    "Tail.LN":
        ("agg Tail.LN 5 claims sev lognorm 100 cv 3 poisson",
         dict(log2=16, bs=5, padding=2, normalize=True)),

    # 4: occurrence + aggregate reinsurance — the subject/after reporting case.
    "Re.Both":
        ("agg Re.Both 100 claims 5000 xs 0 sev lognorm 50 cv 1.5 "
         "occurrence net of 3500 po 4000 xs 1000 poisson "
         "aggregate net of 2000 xs 3000",
         dict(log2=16, bs=1/4, padding=2, normalize=True)),

    # 5: defective — heavy tail truncated, normalize=False so sum(p) < 1.
    #    (Alt to also hit the no-2nd-moment path: sev 100 * pareto 1.5.)
    "Def.Pareto":
        ("agg Def.LN 1 claim sev 1000 * pareto 1.5 - 1000 fixed",
         dict(log2=16, bs=3125/8192, padding=1, normalize=False)),

    # 6: bounded mixture
    "Mixture":
        ('agg Mix [10 20] claims [100 400] xs 0 sev lognorm [20 40 100] cv [.5 .6 .7] wts=3 mixed gamma .2',
        dict(log2=16, bs=1/8, padding=1, normalize=True)),
}

# Each port case: (program, grid, allocation_methods to snapshot)
# also reasonable. no need to test switcheroo at this point.
PORT_CASES = {
    # P1: unbounded (gamma/lognorm, fixed-1) — PIR CNC. LINEAR focus
    #     (lifted+CCoC here is the unbounded+mass case the refactor will refuse,
    #      so snapshot CCoC under linear; PH is fine under either).
    "Port.CNC":
        ("port Port.CNC "
         "agg CNC.NonCat 25 claim sev gamma   80 cv 0.15 mixed gamma .2 "
         "agg CNC.Cat    5  claim 200 xs 0 sev lognorm 40 cv 1.50 mixed ig .2",
         dict(log2=16, bs=1/4, padding=1),
         ["linear"]),                      # + "lifted" for dual, tvar

    # P2: bounded but a bit silly
    "Port.Bounded":
        ("port Port.Bounded "
         "agg A dfreq[1:3 ] sev 500 * beta 5 2 "
         "agg B dfreq[1:5 ] sev 800 * beta 7 2 ",
         dict(log2=16, bs=1/8, padding=0),
         ["linear", "lifted"]),

    # P3: bounded discrete (max 199, fixed-1) — Bodoff. LIFTED focus
    #     (bounded ⇒ lifted+CCoC is stable, no refuse).
    "Port.Bodoff":
        ("port Port.Bodoff "
         "agg wind  1 claim sev dhistogram xps [0,  99] [0.80, 0.20] fixed "
         "agg quake 1 claim sev dhistogram xps [0, 100] [0.95, 0.05] fixed",
         dict(log2=8, bs=1, padding=1),
         ["lifted", "linear"]),
}
```

Notes / to confirm:

* **Grids** (`log2`, `bs`) for `Tail.LN`, `Re.Both`, `Def.LN` are guesses sized
  off the mean — please sanity-check they contain the distribution (and, for
  `Def.LN`, that the deficit is the *intended* small one, not a too-small grid).
* **`Re.Both` clause order** follows the test-suite convention: `occurrence net
  of … <freq>` (occ before frequency) and `aggregate net of …` after frequency.
* **`Def.LN`** uses a very high-CV lognormal + `normalize=False` to force
  `Σp<1`; swap to `sev 100 * pareto 1.5` if you also want the genuinely
  no-second-moment path (then `recommend_bucket` would raise — fine, we pin `bs`).
* **`Port.Sample`** needs a small fixed multivariate sample DataFrame committed
  alongside (e.g. a few hundred rows for two lines) so the switcheroo is
  reproducible.

---

## 4. What to snapshot (per case)

After `update` (pinned grid):

* `stats_df` (full) and `describe` (full).
* `density_df.filter(regex='p_|exeqa_')` **plus** `loss` and `S` — we definitely
  want some of the probabilities in the snapshot.
* scalar readouts: `q(0.9)`, `q(0.99)`, `q(0.999)`, `tvar(0.99)`.

For the **portfolios**, additionally, for **each** of the two fixed distortions
(`ccoc` mass, `ph` no-mass) and each listed allocation method:

* the post-distortion columns from `augmented_df` (lifted) / the linear readout:
  `gS`, `gp_total`, and the `exag_*` (risk-adjusted premium) columns;
* the `pricing_at` / `analyze_distortions` result frame at `p=0.99`.

Exclude volatile fields: `last_update`, object `repr`s, timestamps.

---

## 5. Storage & manifest

* **Frames → parquet** (`tests/baseline/data/<case>__<frame>.parquet`) —
  preserves float64 exactly. (Feather is fine too; parquet chosen for
  portability.)
* **Manifest → `tests/baseline/data/manifest.json`**: per case, the program
  text, grid, `normalize`, the list of snapshotted frames + column filters, the
  scalar readouts, and the environment block (`aggregate`/`numpy`/`scipy`/
  `pandas` versions) and the capture commit SHA.

---

## 6. Comparison harness

`tests/test_baseline.py`:

* Load the manifest; rebuild each case at its pinned grid; recompute each
  snapshot; load the stored parquet.
* Per `(case, frame, column)` assert `np.allclose` at a **single strict tier**
  `rtol=1e-12, atol=1e-14` to start, and **see how far that holds** (the stated
  initial goal: very, very close). Only *introduce* a looser tier where a genuine
  `interp1d` / reinsurance step / distortion root-solve forces it — do not start
  loose.
* On failure, report the **first** divergent `(case, frame, column, index)` with
  the max abs and rel diff — a targeted signal, not a wall of numbers. (A small
  helper that returns the worst-offending cell.)
* Object/meta rows (e.g. `('meta','name')` while it still exists) compared as
  equality, not `allclose`. (After the planned all-float `stats_df` change this
  simplifies.)

---

## 7. Workflow

* **Capture now**, from `REFACTOR` HEAD, before any refactor edit; commit the
  parquet + manifest; record the SHA in the manifest.
* A `--regenerate-baseline` pytest flag (or `AGG_REGEN_BASELINE=1`) re-captures;
  otherwise the test is read-only.
* When a refactor step **intends** to change numbers (Portfolio moment
  convention; `_build_augmented` ROE fix; any tail-mass change), regenerate in
  the same commit and note the diff in the message.

---

## 8. File layout

```
tests/
  baseline/
    corpus.py        # §3 — the DecL specs, grids, distortions, snapshot manifest  ← THE key file
    capture.py       # build each case, snapshot → parquet + manifest.json
    data/            # committed baselines
      manifest.json
      <case>__<frame>.parquet …
      sample_switcheroo.parquet     # fixed sample for Port.Sample
  test_baseline.py   # §6 — load data/, rebuild, compare
```

---

## 9. Open / to confirm

1. **The DecL specs and grids in §3** — the main review item.
2. Include the no-second-moment `pareto` variant for `Def.LN`, or keep the
   lognormal-cv-6 form only?
3. Switcheroo: hand-build the fixed sample, or derive it once from a *seeded*
   `Port.CNC.sample(...)` and commit the result (still reproducible because
   committed)?
4. Do we want a **separate, looser-tolerance** snapshot that exercises
   `calibrate_distortions` (Newton solve), or keep calibration out of the
   baseline entirely?
