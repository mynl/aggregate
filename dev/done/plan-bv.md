# plan-bv — massive (disk-backed) bivariate distributions: out-of-core update, dict pushforwards, exploration-grade visualization

> **STATUS: EXECUTED — shipped `1.0.0a126`, 2026-07-02.** All six phases
> landed; see `CHANGELOG.md` and the run results in §12 below. Design settled
> with the author 2026-07-02 (all §11 decisions resolved; see the decision
> log there).
> Techniques proven in the sibling exploration repo `T:/ai/big2conv` (bigbiv)
> — pointers to the proof studies throughout. This is the money application:
> a `bv` updated at `log2 = (14, 14)` and beyond, living on disk, reduced to
> the `GridDistribution`s the analyst actually wants, and explorable visually
> at full resolution.

## 1. Goal

Three deliverables, dropped into the existing architecture:

1. **Massive update.** `BivariateAggregate.update` at `log2 ~ (14, 14)` up to
   `(16, 16)+`. The realized joint density lives **on disk only** (zarr store);
   RAM stays bounded by a band regardless of grid size. New argument: the
   backing directory.
2. **Dict pushforward.** `pushforward(self, functions={k_i: f_i}, bs=...)` on
   the (already-updated) bv: stream the pushforward of every `f_i(X, Y)` —
   and, when there are two or more, of their **total** `t = Σ f_i` — in
   **one** pass over the on-disk density, returning a `GridDistribution` per
   key plus `'total'`. The total is accumulated **pre-bucketing** (evaluated
   exactly per cell, then bucketed once at the end with its own required
   `bs_total`) — never as a sum of bucketed results. A single function (the
   usual case) skips the total entirely; constant functions (common — a
   premium) are detected and short-circuited before the sweep. `X`, `Y` are
   recovered from indexes via the bv's measured windows (`axis_xs`); output
   bucketing (`bs`) is caller-supplied.
3. **Visualization.** A new plots-subsystem module for exploring the massive
   joint: zoom to full resolution at constant cost, decimation with
   mean/max/min channels so edges, ridges and atoms survive coarsening, plus
   an optional interactive tier. Maximum power and wow factor.

Placement: the compute kernels in a **new leaf module**
`_aggregate_compute_massive.py` (sibling of `_aggregate_compute.py`, same
pure-function contract); the plotting in `plots/_bivariate_massive.py` under
the existing three-layer plotting architecture.

## 2. Proven foundations (bigbiv, `T:/ai/big2conv`)

Everything below is measured, not guessed (studies in `studies/`, tests in
`tests/` of that repo):

- **The 3-pass zarr corner-turn works and is exact.** The separable 2-D
  compound `iFFT2(pgf(FFT2(S)))` runs as: (1) row-FFT streamed to a 2-D-tiled
  zarr store; (2) *read the store back column-wise* — the transpose is just
  reading tiles in orthogonal order — FFT down columns, apply the pgf
  (elementwise, nothing moves), inverse column-FFT, write to a second store;
  (3) inverse row-FFT streamed out, folding reductions band by band.
  Bit-identical to the in-core engine (`atol 1e-12`, all chunk shapes);
  at 16384² the streamed `E[X+Y]` matched theory to `3e-12` with peak RSS
  2.4× below the naive in-core footprint and the joint never materialized
  (`studies/zarr_out_of_core.py`). Wall time ~50 s at 16384² on the dev
  machine, complex128, including a streamed reduce.
- **dask's in-memory shuffle is NOT the road.** `dask.distributed`'s P2P
  rechunk holds transpose buffers it cannot spill (it spills completed keys
  only, not a running task's working set); workers died repeatedly at 16384²
  under 2–4 GB budgets. The explicit zarr staging worked first try, single
  process, no cluster. A dask "sequencer" is therefore **optional future
  parallelism** (pass-2 column bands are independent), not a dependency:
  scipy's `workers=-1` already threads the per-band FFTs and the passes are
  I/O-bound.
- **`scipy.fft` is the right backend** (already aggregate's choice): ~2×
  faster than `numpy.fft` for 2-D at these sizes, threaded via `workers=`.
- **Multi-reduce folds free.** Streaming N pushforwards in one pass costs one
  pass: the expensive part (FFTs + corner-turn) is function-independent
  (bigbiv `compound_zarr(reduces={...})`, `zpushforward`).
- **Windowing is bookkeeping, not data movement.** The `j0` output-window
  roll is a *cyclic relabelling*; out-of-core it is applied per band at
  pass-3 write time (≤ 2 contiguous reads per band + an in-RAM `np.roll`
  along axis 1). No global transpose, no extra pass. (§4.3.)
- **float32 rejected** (author decision, 2026-07-02): ~2e-5 relative error on
  means for a 2× saving isn't worth it; also several `Frequency.freq_pgf`
  implementations silently upcast. Everything below is float64/complex128.

## 3. Where it slots into aggregate

The in-core pipeline today (`bivariate.py`):

- `update` (`bivariate.py:1141`) — parses sizing, calls `_size_axes`
  (`:1020`, **measure-don't-guess**: standalone marginals + `balanced_window`
  give per-axis `(bs, log2, x_min)`), derives per-axis `_i0` (severity
  negative lay-in), `_j0` (output-window origin), `_mlog2` (FFT buffer
  length, **decoupled from output length** so a tight window far from 0
  doesn't alias, `:1229`).
- `update_work` (`:1305`) — forms the per-claim severity `S` (copula
  `rectangle_pmf`, the given discrete matrix, or netceded's comonotone
  scatter), lays it into the `(M0, M1)` buffer (`_lay_signed_2d`, `:1284`),
  `rfft2 → freq_pgf (ravel/reshape) → irfft2`, then
  `np.roll(a, (-j0_0, -j0_1))[:N0, :N1]`.
- `BivariateDistribution` (`:1945`) — the result container; `pushforward`
  (`:2004`) already scatters via `_scatter_1d` with an optional `chunk_size`;
  `transformed_moments` (`:2090`) provides the exact "EX" audit numbers.
- 1-D precedent: `_aggregate_compute.freq_sev_convolution` — a **leaf**
  pure-function kernel with exactly the same `i0`/`j0` semantics, extracted so
  the core is testable without a full `update`. The massive kernel is its 2-D,
  out-of-core sibling and should mirror its argument vocabulary.

**Key observation:** all three bv modes differ only in how `S` is formed.
`S` is small (per-*claim*, its support is tiny relative to the aggregate's).
So the massive path needs exactly **one** kernel taking `(S, freq_pgf, en,
sizing/window parameters, store_dir)` — the sizing machinery (`_size_axes`,
`_i0/_j0/_mlog2`) is reused *unchanged*. Nothing about windowing needs to be
re-derived; it is already correct, and out-of-core it degenerates to label
bookkeeping.

## 4. Deliverable 1 — the massive update kernel

### 4.1 Module and API

New leaf `src/aggregate/_aggregate_compute_massive.py`. Same contract as
`_aggregate_compute.py`: pure functions; numpy / scipy.fft / zarr only; never
imports `_aggregate` or `bivariate`. `zarr` is imported **lazily inside the
functions** with a clear error naming the optional extra (§9) if missing.

```
massive_bivariate_convolution(
    S, freq_pgf, en, *,
    N0, N1, bs0, bs1,            # output grid (the measured window)
    i0=(0, 0), j0=(0, 0),        # severity lay-in / window origin, per axis
    mlog2,                       # per-axis FFT buffer log2 (from update sizing)
    store_dir,                   # backing directory (REQUIRED — this is the point)
    row_chunk=512, col_chunk=512,
    keep_transform=False,        # keep z1/z2 staging after completion
    progress=None,               # optional callback(pass_no, frac) for long runs
) -> MassiveResult              # handles + accumulators, see §5
```

plus `massive_pushforward(...)` (§6) and `build_pyramid(...)` (§7), all
driven by the same band loops.

### 4.2 The three passes (with aggregate's windowing folded in)

Stores are 2-D-tiled zarr arrays, chunks `(row_chunk, col_chunk)`, complex128
for staging, float64 for the final density. `M0 = 1 << mlog2[0]`, etc.;
`nf1 = M1 // 2 + 1` (rfft width along axis 1).

- **Pass 1 — severity lay-in + row-FFT → `z1` (M0 × nf1, complex).**
  The 2-D lift of `freq_sev_convolution`'s signed lay-in: `S` occupies rows
  `[0, nS0 - i0_0)` and `[M0 - i0_0, M0)` (negatives wrapped to the top),
  columns likewise within each row. Because `S` is per-claim and therefore
  *tiny* relative to `M`, **only those ~nS0 row bands are ever written**;
  zarr leaves unwritten chunks at fill value 0, so `z1` is *sparse on disk*
  (a few MB even at `(16, 16)`), and pass 1 is near-free.
- **Pass 2 — the corner-turn: column bands of `z1` → FFT(axis 0) → pgf →
  iFFT(axis 0) → `z2` (M0 × nf1, complex, dense).**
  For each band of `col_chunk` columns: read `z1[:, c0:c1]` (full height —
  unwritten chunks materialize as zeros for free), `fft` down axis 0, apply
  `freq_pgf(en, block.ravel()).reshape(...)` (the empirical pgf assumes 1-D —
  same ravel/reshape as `update_work:1359`), `ifft` down axis 0, write.
  This is the expensive pass: full disk write of `z2` and the whole FFT work.
- **Pass 3 — inverse row-FFT + window relabel + fold-everything → outputs.**
  Iterate the **output** rows `r ∈ [0, N0)` in bands (not the buffer's `M0`
  rows — the window crop saves `(M0 − N0)/M0` of the read). Output row `r`
  lives at buffer row `(r + j0_0) mod M0`: a band maps to ≤ 2 contiguous
  `z2` row reads. `irfft` along axis 1 (full `M1` width — the inverse needs
  the whole complex row), then the axis-1 window is an **in-RAM**
  `np.roll(band, -j0_1, axis=1)[:, :N1]`. The band now carries true value
  labels `axis_xs[0][r0:r1]` / `axis_xs[1]`, in monotone physical order.
  Fold, per band, in ONE sweep:
  - write `density.zarr[r0:r1, :]` (float64, N0 × N1) — *the realized bv on
    disk, in physical order* (post-hoc consumers never see `j0` again);
  - marginals (exact), total mass → `deficit`;
  - mixed raw moments `E[X^a Y^b]`, `a + b ≤ 3` (per-band float64 partials,
    pairwise-summed) → means, sds, `corr`, the info/validation surface;
  - the visualization pyramid's base level (§7);
  - any pushforwards requested at update time (§6 applies post-hoc too).

After pass 3, `z2` (the largest store) is **deleted by default**
(`keep_transform=False`); the durable artifacts are `density.zarr`, the
pyramid, and the small accumulator outputs.

### 4.3 Correctness of the streamed relabel

`np.roll` of a finished array is a cyclic *renaming*: buffer index `i` holds
physical value `x_min + ((i − j0) mod M)·bs`. Pass 3's reductions bin by
**value**, and binning is order-independent, so streaming bands in buffer
order with rolled *labels* is exactly equivalent to rolling the array — the
identical argument as `freq_sev_convolution`'s notes ("relabelling a finished
array carries no `N·s` shift", `_aggregate_compute.py:79`), lifted to 2-D.
Writing `density.zarr` at the rolled row offsets (each input band lands at ≤ 2
contiguous output row ranges) makes the on-disk density physically ordered at
zero extra cost. The guard is unchanged from 1-D: exact when the aggregate
support width `< M·bs` per axis; the nines-window sizing guarantees that to
`10^-window_nines`, and `deficit` reports what wrapped.

### 4.4 Entry point and modes

`BivariateAggregate.update(log2=0, bs=0, padding=1, store_dir=None, ...)`:

- `store_dir=None` → today's in-core path, byte-for-byte untouched.
- `store_dir='D:/scratch/bv1'` → after `_size_axes` (unchanged) and the
  per-axis `_i0/_j0/_mlog2` derivation (unchanged), delegate to
  `massive_bivariate_convolution` instead of `update_work`'s in-core FFT.
  All three modes route through it, because each mode's `S` formation is
  unchanged and small (copula `rectangle_pmf` on the per-event grids; the
  discrete lattice scatter; netceded's comonotone scatter).
- Result surface: `self.bivariate` returns a `MassiveBivariateDistribution`
  (§5). `marginals`, `moments`, `corr`, `deficit`, `info` all work from the
  pass-3 accumulators — same numbers as in-core, no disk read.

**Padding for massive (SETTLED 2026-07-02): `padding=0`.** In-core default is
`padding=1` (doubles each axis → 4× cells); for massive that is 4× disk and
time for no useful protection: the measured window already covers the support
to `10^-window_nines`, so at `padding=0` only ~1e-9 of mass wraps, and
`deficit` reports it. The massive path defaults **`padding=0`**, documented,
with `deficit` as the guard. (bigbiv runs padding=0 throughout; deficits
observed ~1e-12 with measured windows.)

### 4.5 Budgets (complex128 staging + float64 density, padding=0, x_min=0)

| grid (per axis) | cells | `z2` staging | `density.zarr` | pyramid (f32, 3ch) | steady-state disk | est. update wall |
|---|---|---|---|---|---|---|
| 2^13 = 8192  | 67 M  | 0.5 GB | 0.5 GB | 0.27 GB | ~0.8 GB | ~15 s |
| 2^14 = 16384 | 268 M | 2.1 GB | 2.1 GB | 1.1 GB  | ~3.3 GB | ~1 min |
| 2^15 = 32768 | 1.1 G | 8.6 GB | 8.6 GB | 4.3 GB  | ~13 GB  | ~5 min |
| 2^16 = 65536 | 4.3 G | 34 GB  | 34 GB  | 17 GB   | ~52 GB  | ~15–30 min |

(Wall times extrapolated ~N log N from the measured 50 s at 16384² including
a streamed reduce; I/O ~2× the store sizes per pass on NVMe. Peak transient
disk = staging + density + pyramid ≈ steady-state + `z2`. zarr's default
zstd compression will shrink real densities substantially — vast near-zero
regions — the table is uncompressed worst case.) RAM: `max(row_chunk·M1,
M0·col_chunk)·16 B` ≈ 0.5–1 GB at `(16,16)` with 512-bands — constant in
grid size. Advise `store_dir` on the Dev Drive (async Defender + many small
tile files) with ≥ 150 GB free for `(16,16)` transients.

## 5. The `MassiveBivariateDistribution` container

Sibling of `BivariateDistribution` (same module), same role accessors
(`axis0`, `axis1`, `axis_names`) so pushforward/moment code addresses axes by
role. Holds: the `density.zarr` handle (lazy), `axis_xs`, `bs`, marginals,
moments, `deficit`, pyramid handle, `meta` (provenance incl. `store_dir`,
grid/window parameters, creation stamp).

- `.density` returns the **zarr array** (duck-types for slicing:
  `bv.density[1000:1010, :]` works; a full `[:]` read is the user explicitly
  asking for the whole thing and getting what they asked for). Documented.
- `.marginal(i)` → `GridDistribution` of axis i (exact, precomputed).
- `.slice(x=...)` / `.slice(y=...)` → conditional `GridDistribution`
  `P(Y | X≈x)` — one row/column-of-tiles read, instant; the analyst's probe.
- `transformed_moments(f)` — streamed version of `:2090` (one band sweep),
  same "EX" audit semantics.
- Persistence: the store directory is self-describing (an `attrs.json` /
  zarr attrs with axes, bs, meta), so a **`reopen(store_dir)`** classmethod
  reconstructs the container in a later session without re-running the FFT.
  This falls out almost free and is worth a lot for a 30-minute update.

## 6. Deliverable 2 — `pushforward(functions={k: f}, bs=...)`

On `MassiveBivariateDistribution` (and, dispatching to the existing in-core
path, on `BivariateDistribution` for API uniformity):

```
def pushforward(self, functions, bs, *, bs_total=None, total_key='total',
                windows=None, scheme='linear', is_loss_value=True)
    -> dict[str, GridDistribution]     # one per key; plus total_key when n >= 2
```

- **Semantics.** Each `f_i` is vectorized `f(x, y)` on broadcast label
  arrays (role-addressed axes, same contract as `:2020`). With two or more
  functions, a **`'total'`** entry is also returned: the random variable
  `t(X, Y) = Σ_i f_i(X, Y)`, accumulated **cell-wise pre-bucketing** per band
  (evaluate every `f_i`, sum the value arrays, then scatter `t` once with its
  own bucket) — one bucketing error, not `n` compounding ones. The per-`f_i`
  scatters and the total scatter share the same band and the same flattened
  weights: `n+1` `bincount`s per band, one disk read for all.
- **Single-function fast path (the usual case).** `len(functions) == 1` ⇒
  no total is computed or returned, no accumulation overhead — just the one
  streamed pushforward. `bs_total` must be omitted (raise if given: nothing
  for it to size).
- **Constant functions (common — e.g. a premium).** Detected **before the
  sweep** via a coarse probe lattice over the label ranges (a constant on
  the probe is treated as the constant `c`; genuinely non-constant functions
  that fool a probe are pathological and out of scope — documented). A
  constant `f_k = c` never enters the band loop: its `GridDistribution` is a
  point mass at `c` carrying the total mass, built directly from the stored
  `total_mass`; its contribution to the total is a scalar shift folded into
  the accumulation for free. With constants excluded, a call like
  `{loss: f, premium: 150e6}` costs one function's sweep, not two.
- **`bs` (required — the caller knows their output scale).** Scalar (applied
  to every `f_i`) or array of length `n` in `functions` iteration order
  (dicts are ordered). **`bs_total` is required whenever `n >= 2`** (raise
  with a clear message if missing) — explicit is better than implicit; no
  derived default. *(SETTLED 2026-07-02: separate required kwarg, not a
  length-`n+1` array.)*
- **Windows.** The output grid per function needs a global `(lo, hi)` before
  scattering. Default: a **label-only pre-sweep** — evaluate each (non-
  constant) `f_i` over the label grid band-by-band, track min/max; pure CPU,
  no disk I/O (~seconds at `(14,14)`, ~a minute at `(16,16)`).
  `windows={k: (lo, hi)}` skips it per key (the caller "has a good idea" —
  their words). Values falling outside a pinned window clip to the edge
  buckets and are reported via `.clipped_mass`, matching the existing
  convention (`:395`).
- **Reserved key (SETTLED: `'total'`).** The sum is returned under
  `total_key` (default `'total'` — the house word; use it consistently in
  the audit frame, plot labels and docs). Collision with a user key raises.
- **Output.** `GridDistribution` per key (trimmed to realized support,
  `is_loss_value` passed through), each carrying `.pushforward_source` and
  `.clipped_mass` per the existing pattern. Mean-preserving `'linear'`
  scatter by default; `'nearest'` available.
- **Cost.** One full read of `density.zarr` + one evaluation/scatter per
  non-constant function (+1 for the total when `n >= 2`) per band. At
  `(16,16)`: ~34 GB read ≈ 1–3 min on NVMe; repeatable at will because
  update is decoupled (the FFT never reruns). Prototype: bigbiv
  `zpushforward` / `compound_zarr(reduces=...)` — already validated against
  full-joint pushforwards to `1e-11`.
- **Audit (SETTLED: yes — the exact numbers matter).** For each key, the
  streamed `transformed_moments` "EX" numbers come from the *same* band
  sweep at ~zero cost; report Est-vs-EX alongside, mirroring the
  reinstatements audit pattern. Linearity check for free:
  `mean(total) == Σ mean(f_i)` exactly (pre-bucket accumulation makes this a
  hard invariant, and it doubles as the unit test).

## 7. Deliverable 3 — visualization (`plots/_bivariate_massive.py`)

Two tiers within the existing three-layer plotting architecture (matplotlib
boundary, Layer-1 panel workers + Layer-2 compositor, class `.plot()` stub
delegating).

### 7.1 Data layer: the decimation pyramid (built during pass 3, free-ish)

A map-tile pyramid over the density, stored in the same directory
(`pyramid.zarr`): level k is a 2×2 reduction of level k−1, down to ~256².
Three channels per level, because one reduction lies:

- **sum** — probability aggregates additively; a 2×2 block-sum *is* the pmf
  on the coarser grid. The "body" channel; what a heatmap should show.
- **max** — preserves what sum washes out at coarse zoom: thin ridges
  (netceded's comonotone curve is a *one-cell-wide* filament carrying finite
  mass), atoms, kinks at reinsurance attachment/limit lines. At `(16,16)`
  rendered to a 1k image, one pixel covers 64×64 cells — a filament simply
  vanishes from a sum channel; the max channel keeps it lit.
- **min** — reveals interior holes / exact-zero regions inside apparently
  solid mass (support boundaries, lattice artifacts).

Base level folds from pass-3 bands (a row band reduces its own 2×2 blocks);
higher levels cascade in RAM once small. float32 (viz-only), ~17 GB
uncompressed at `(16,16)`, far less compressed.

### 7.2 Tier 1 — static matplotlib compositor: `plot_bivariate_massive`

The **constant-cost invariant**: every render reads only the pyramid level
whose resolution matches the pixel budget (and, at extreme zoom, raw
`density.zarr` tiles) — never the full grid. Pan/zoom from the whole
distribution down to individual buckets at the same speed.

Panels (mosaic per house style):

- **Main heatmap**: `log10` density **by default** (SETTLED — the author
  almost always wants log scale here; mass spans many decades and linear
  hides everything but the mode; `log=False` opt-out), percentile-clipped
  color range, zeros masked to background. **Dual encoding**:
  sum-channel as the color field, max-channel composited as a luminance
  boost where `max ≫ block-sum/cells` — sub-pixel structure *glows*. This is
  the single biggest wow-per-line feature.
- **Atom decomposition**: zero-inflated bivariates carry finite mass ON the
  axes (`P(X=0, ·)`, `P(·, Y=0)`) and at the origin. Naive heatmaps either
  hide these or let them saturate the colormap. Render them as separate 1-D
  strips flanking the heatmap + an origin badge (`P(0,0)`), with the interior
  shown clean. (Insurance-native: probability of zero claims is a headline
  number, not a color-scale nuisance.)
- **Marginal panels** top/right (from the exact stored marginals), aligned
  to the heatmap axes; log/linear toggle.
- **Contours** of log-density at decade levels overlaid on demand; on the
  coarse pyramid, exceedance contours `P(X > x, Y > y)` (2-D suffix sums are
  cheap at pyramid resolution) — the joint-tail view a risk analyst reads.
- **Window/zoom API**: `plot(window=((x0, x1), (y0, y1)))` — same call
  contract as everything else in the subsystem; picks the pyramid level,
  reads the tiles, done.
- **Slices**: `plot_slice(x=...)` conditional strip charts from §5's
  one-read slices.

### 7.3 Tier 2 — interactive explorer (optional dependency, the wow tier)

`explore()` on the container: a **datashader + holoviews/bokeh** app over the
pyramid — Google-Maps-style pan/zoom of a 4-gigacell pmf in JupyterLab, with
dynamic re-aggregation, hover readout (x, y, p, log10 p, cdf), channel
toggle (sum/max/min), linked marginal panels that re-window to the viewport,
and click-to-slice conditionals. Lazy-imported from a separate module so the
matplotlib-only boundary of `plots/__init__` is untouched; guarded with a
clear "install `aggregate[viz]`" error. matplotlib Tier 1 is always
available; Tier 2 is where the demo lives.

*(Alternative considered: matplotlib-native zoom callbacks re-reading tiles —
zero new deps, workable, but bokeh's server-side re-aggregation is the
qualitative jump. Proposal: Tier 1 always ships; Tier 2 behind the extra.)*

## 8. Validation

House pattern: cross-check everything against an independent computation.

1. **Kernel == in-core, bit-level.** Small grids (≤ 2^10 per axis), all three
   modes, `massive_bivariate_convolution` vs `update_work` density:
   `atol 1e-12`. Sweep chunk shapes incl. non-divisors (bigbiv precedent:
   `tests/test_zarr_backend.py`).
2. **Windowing bookkeeping** (the risk concentrate): signed axes
   (`i0 ≠ 0`), shifted windows (`x_min > 0` ⇒ `j0 ≠ 0`), both together;
   massive marginals vs the standalone 1-D `Aggregate` marginals (thinning
   exactness), and vs in-core `update` on the same sizing.
3. **Moments**: streamed mixed moments vs `MomentAggregator` theory and vs
   in-core `moments()`.
4. **Pushforward**: streamed vs `BivariateDistribution.pushforward` on the
   materialized small-grid density (`atol 1e-11`); `mean(total) == Σ means`
   (exact, pre-bucket invariant); Est-vs-EX audit numbers; window clipping
   accounting; **single-function fast path** (no `'total'` key, `bs_total`
   raises if supplied); **constant detection** (point mass at `c` with the
   stored total mass, zero band-loop evaluations — assert via a counting
   wrapper — and the correct scalar shift in `'total'`); missing `bs_total`
   with `n >= 2` raises.
5. **Deficit guard**: deliberately undersized window ⇒ wrap contamination
   visible in `deficit`; documented failure mode (matches 1-D notes).
6. **Persistence round-trip**: `reopen(store_dir)` reproduces marginals,
   moments, pushforwards.
7. **Scale smoke** (not in CI): `(14, 14)` end-to-end incl. plot + explore;
   RSS sampled < 2 GB; timings recorded to the plan.

## 9. Dependencies & packaging

- `zarr>=3` as optional extra `aggregate[massive]`; lazy import, actionable
  error. (zarr 3 requires py ≥ 3.11 — matches `requires-python`.)
- `aggregate[viz]` extra: `datashader`, `holoviews`, `bokeh` (Tier 2 only).
- **No dask dependency.** Single-process explicit band loops + threaded
  scipy FFTs; the measured failure of the distributed shuffle is the
  documented rationale. Revisit only if profiling shows CPU-bound pass 2.
- No numba; `bincount`/vectorized numpy throughout (matches `:2052`).

## 10. Phasing

| phase | scope | exit criterion |
|---|---|---|
| 1 | `_aggregate_compute_massive.py` kernel + tests §8.1–.2 | bit-match in-core, all modes, signed/windowed |
| 2 | `update(store_dir=)` + `MassiveBivariateDistribution` + marginals/moments/info + `reopen` | §8.3, .6 green; `(14,14)` smoke |
| 3 | `pushforward(functions, bs)` + sum + audit | §8.4 green |
| 4 | pyramid + Tier-1 plot + slices + atom strips | visual review on netceded (ridge) + copula (smooth) cases |
| 5 | Tier-2 `explore()` | the demo |
| 6 | docs (feature page, 5-min intro segment), CHANGELOG, version bump | — |

## 11. Decision log (SETTLED with the author, 2026-07-02)

1. **Entry point**: `update(store_dir=...)` kwarg on the existing `update` —
   no separate `update_massive()`. `store_dir=None` keeps today's in-core
   path untouched.
2. **Padding**: massive path defaults **`padding=0`** (§4.4) — the measured
   window makes further padding pointless; `deficit` is the guard.
3. **Total bucketing**: **`bs_total` is a separate, explicit kwarg, REQUIRED
   when `n >= 2`** — no derived default ("explicit is better than
   implicit"). Reserved key is **`'total'`** (not `'sum'`) — use "total"
   consistently across the audit frame, plot labels and docs. Single
   function ⇒ no total at all (the usual case); constant functions are
   detected pre-sweep and short-circuited (§6).
4. **`.density`**: lazy zarr view (slice-friendly; a full `[:]` read is the
   user asking for exactly that). Documented.
5. **Staging**: `z2` **deleted by default** after pass 3
   (`keep_transform=False`); the 34 GB at `(16,16)` is not worth keeping for
   a cheap-ish re-run capability.
6. **Tier-2 stack**: the sophisticated group — **datashader + holoviews +
   bokeh** behind the `aggregate[viz]` extra. A lot of effort goes into this
   feature; the pictures coming out should match.
7. **No DecL surface**: API-only in v1 (constructor / `update` / methods).
   "You have to know what you are doing to use this."

## 12. Run results (execution, 2026-07-02, `1.0.0a126`)

- **Tests:** 34 cases in `tests/test_massive_bivariate.py` — kernel bit-match
  vs in-core across all three modes (`atol 1e-12`), signed axes (`i0 != 0`),
  shifted windows (`j0` positive and negative), chunk sweep incl. non-divisor
  shapes, sparse-S equivalence, en=0 point mass, staging lifecycle, update
  surface parity, container probes, reopen round-trip, deficit guard, dict
  pushforward (streamed == in-core `1e-11`, pre-bucket total linearity exact,
  single-function fast path, constant short-circuit with a call counter,
  window clipping, reserved-key collision), pyramid channel exactness +
  cascade, plot/plot_slice/explore smokes. Full bivariate regression files
  unchanged-green (167 total with the pre-existing suites).
- **§8.7 scale smoke, `(14, 14)` = 16384² copula book (this machine):**
  update **142 s** wall (includes the two 2^20 measurement updates and the
  trimmed `rectangle_pmf`), deficit **2.1e-8** at `padding=0`; density
  **0.77 GB** on disk compressed (2.1 GB logical); 2-function dict
  pushforward (one constant, short-circuited) **43 s** — audit `Err EX`
  ~2e-16, `mean(total) = Σ means` exact; `reopen` **0.02 s**; full-view plot
  **0.68 s**, deep 10%-window zoom **0.35 s** (constant-cost confirmed);
  peak RSS **2.4 GB** (vs <2 GB hoped — the 2^20 in-core measurement
  marginals and matplotlib own the excess, not the kernel bands).
- **Deviations from the letter of the plan** (all behavior-preserving):
  the `_size_axes` *measurement* grid is capped at 2^20 on the massive path
  (a `(16,16)` budget would otherwise demand a 2^32-bucket 1-D update);
  copula-mode `S` is trimmed to the per-event support (< 2e-15 mass) since
  the full-grid rectangle matrix is exactly the allocation the plan avoids;
  netceded severity goes through a new sparse scatter
  (`scatter_bivariate_sparse`); pyramid level selection is area-based so
  anisotropic grids (netceded 16384×64) keep their short axis resolved;
  the default plot window is the marginals' realized support box, not the
  full measured grid (half-empty canvas otherwise); atom strips render as
  filled lines with a mass annotation and auto-hide below 1e-6.
