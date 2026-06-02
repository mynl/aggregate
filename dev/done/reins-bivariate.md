# reins-bivariate — joint (ceded, net) occurrence distribution via 2D FFT

**Forward-looking research feature.** Phase 1 adds a method to `Aggregate`;
the Portfolio extension is sketched only. Downstream of `reins-buckets.md` (it
reuses the `reins_bucket` scatter). Bumps the `a*` version per the
one-subproject-one-bump convention.

## Context

For an occurrence reinsurance program, each claim `X` splits deterministically
into ceded `c(X)` and net `n(X) = X − c(X)`. But the **aggregate** ceded
`C = Σ c(X_i)` and **aggregate** net `N = Σ n(X_i)` are *not* deterministic
functions of one another — the random claim count `N_claims` decouples them. We
currently compute only the two univariate margins
(`reins_density_df['p_agg_ceded_occ' | 'p_agg_net_occ']`); the *joint* law of
`(C, N)` is unavailable.

> **Naming note (post `reins-reporting`, 1.0.0a19).** The density frame was
> renamed `reinsurance_df → reins_density_df`. The `p_agg_ceded_occ` /
> `p_agg_net_occ` columns (aggregate of the ceded / net *occurrence* severity)
> are unchanged and are exactly the univariate margins this feature joins. These
> are the **unconditional** occ aggregates — the same objects whose moments
> appear as the `('occ', 'Ceded')` / `('occ', 'Net')` `agg` rows of
> `reins_stats_df` and the occ-block ceded/net rows of `reins_describe`. (The
> *conditional* per-layer `layer.k` columns of `reins_stats_df` are a different
> basis and are **not** used here.)

The compound-distribution FFT machinery extends to the joint law **with no new
math**. The per-claim bivariate severity `(c(X), n(X))` is supported on the curve
`c + n = X`. Its 2D FFT is `Ŝ`, and the joint aggregate characteristic function
is `freq_pgf(n, Ŝ)`. Because `freq_pgf(n, z)` is elementwise in `z`
(e.g. Poisson `exp(n(z−1))`, `distributions.py:983`), it applies unchanged to a
2D array. This is a genuine advance for FFT methods and directly useful: joint
ceded/net risk, reinsurer-vs-cedent dependency, and co-moments.

Decisions locked with the author:
- Return a dedicated **`BivariateDistribution`** class in a new
  `src/aggregate/bivariate.py` submodule (mirrors `ft.py` / `tweedie.py`; no
  top-level re-export, per CLAUDE.md).
- **Auto** per-axis bucket/window sizing from the univariate margins, with
  explicit overrides.
- **Reuse the `reins_bucket`** linear/nearest scatter (from `reins-buckets.md`),
  applied per axis.
- Phase 1 covers **occurrence** reinsurance only. The aggregate-reins bivariate
  is trivial — at the aggregate level ceded and net are deterministic functions
  of the aggregate, so their joint is degenerate on a curve — and is explicitly
  out of scope.

## Goal

`Aggregate.occ_bivariate(...)` returns a `BivariateDistribution` for
`(C, N) = (aggregate occ ceded, aggregate occ net)`, computed by 2D FFT, with
contour plotting, mixed moments `E[C^i N^j]`, and a normalized
correlation / co-moment readout. Requires the object to carry occurrence
reinsurance and to be updated.

## Why it works (math)

Per claim, `(c(X), n(X))` lives on the line `c + n = X`. Placing the gross
severity mass `p_sev_gross[k]` at the 2D point `(c(x_k), n(x_k))` builds the
bivariate severity `S`. The joint aggregate density is
`iFFT2( freq_pgf(n, FFT2(S)) )` — exactly the univariate `_fft_aggregate`
(`distributions.py:3632`) with 1D → 2D transforms. Marginalizing the resulting
aggregate over one axis recovers the corresponding univariate occ-ceded /
occ-net aggregate, which gives exact validation targets.

## Inputs already available

*(Line numbers are as of 1.0.0a19 and drift; method/attribute names are the
stable anchors.)*

- Grid / freq: `self.xs`, `self.bs`, `self.n` (instance attributes set in
  `update_work`).
- Per-claim densities: `self.sev_density_gross / _ceded / _net` (set in
  `apply_occ_reins`, `distributions.py:4103`).
- Cession maps: `make_ceder_netter(self.occ_reins)` → `ceder, netter`
  (`utilities.py:276`).
- Univariate occ ceded/net aggregates for sizing + validation:
  `reins_density_df['p_agg_ceded_occ' | 'p_agg_net_occ']` (`distributions.py:1835`).
- FFT / PGF: `freq_pgf` (`distributions.py:983`); 2D via `numpy.fft.rfft2` /
  `irfft2` (real-input, matching the 1D `rfft` pattern in `utilities.py:129`).

## Algorithm (`Aggregate.occ_bivariate`)

1. **Preconditions.** Require `self.occ_reins is not None` and that the object is
   updated (sev densities present); else raise `ValueError`. Any aggregate reins
   is ignored (we operate on the per-occurrence ceded/net) — documented in the
   docstring.
2. **Decide buckets + windows.** From `p_agg_ceded_occ` / `p_agg_net_occ` compute
   each axis's effective max (e.g. the `1 − 1e-9` quantile × a small pad) and pick
   `bs_ceded`, `bs_net` (via `round_bucket`, `utilities.py:164`) and per-axis
   `log2_ceded`, `log2_net` (default ≈ 10, memory-capped). Windows are
   `[0, (N−1)·bs]` per axis (zero offset in phase 1). All four overridable by
   kwargs `bs_ceded=`, `bs_net=`, `log2_ceded=`, `log2_net=`.
3. **Build bivariate severity `S` (N_c × N_n).** `cv = ceder(self.xs)`,
   `nv = netter(self.xs)`; scatter `sev_density_gross[k]` onto `(cv[k], nv[k])`
   using the `reins_bucket` method per axis:
   - `nearest` → one cell;
   - `linear` → bilinear split over the ≤ 4 surrounding cells with weights
     `(1−fc)(1−fn)`, `fc(1−fn)`, `(1−fc)fn`, `fc·fn`, preserving **both** marginal
     means.
   Clip to grid; overflow piles at the edge (documented).
4. **2D convolution.** `Z = rfft2(S, s=(N_c<<pad, N_n<<pad))`;
   `A = self.frequency.freq_pgf(self.n, Z)`;
   `density = irfft2(A, s=...)[:N_c, :N_n]`; take the real part, zero sub-eps
   dust, renormalize. Reuse `self.padding`.
5. **Re-center / trim.** Compute the bounding box where row/column marginal mass
   exceeds a tolerance; expose it for contouring (the full grid is retained on the
   object).
6. **Wrap** in `BivariateDistribution(density, ceded_grid, net_grid, bs_ceded,
   bs_net, meta)`.

## `BivariateDistribution` (new `src/aggregate/bivariate.py`)

Lightweight container + methods:

- `.density` (2D ndarray), `.ceded`, `.net` (1D grids), `.bs_ceded`, `.bs_net`.
- `.marginals()` → `(ceded_density, net_density)` by axis-summing (validation).
- `.moments(max_order=3)` → DataFrame of `E[C^i N^j]`, `i, j ∈ 0..max_order`,
  via vectorized `Σ density · C^i · N^j` (`np.einsum`).
- `.corr()` → Pearson correlation (plus optional normalized co-skew / co-kurt
  from the mixed central moments).
- `.contour(ax=None, log=False, **kw)` → filled contour over `(ceded, net)`;
  honors the `FIG_W` / `FIG_H` + constrained-layout plotting conventions.
- `_repr_html_` / `__repr__` summarizing means, correlation, grid shape.

Memory guard: warn if `(N_c<<pad) · ((N_n<<pad)//2 + 1)` complex128 entries
exceed a threshold; recommend `log2 ≤ ~11` per axis.

## Files

- `src/aggregate/bivariate.py` — new: `BivariateDistribution` + scatter/contour
  helpers.
- `src/aggregate/distributions.py` — `Aggregate.occ_bivariate(...)` (sizing,
  scatter, 2D FFT, wrap). Reuse `make_ceder_netter`, `freq_pgf`, `round_bucket`,
  and the `reins_bucket` attribute.
- Docs: a short experimental subsection in
  `docs/2_user_guides/2_x_re_pricing.rst` (pending rebuild per CLAUDE.md).

## Verification

- **Marginals** of `occ_bivariate` match `reins_density_df['p_agg_ceded_occ']`
  and `['p_agg_net_occ']` (compare means / cv to a `VALIDATION_NOISE`-ish tol;
  full density when `bs_ceded == bs_net == self.bs`).
- **Cross-check against the reporting frames** (free, exact targets from the
  same densities): the ceded marginal's `(mean, cv, skew)` equal
  `reins_stats_df.loc[('agg', m), ('occ', 'Ceded')]` (and `('occ', 'Net')` for
  net), and equal the `('occ', 'ceded'/'net', 'agg')` `Est` cells of
  `reins_describe`. The anti-diagonal `C + N` matches the `('occ', 'Gross')`
  `agg` column / the `('occ', 'gross', 'agg')` `EX` reference.
- **Additivity**: `E[C] + E[N] == E[gross aggregate]`;
  `Var(C+N) == Var(gross)`.
- **Anti-diagonal**: the distribution of `C + N` equals the gross aggregate
  (moment-level check across differing `bs`; density check on a shared grid).
- `.corr()` finite and in `[-1, 1]`; a small synthetic Dice / excess case
  eyeballed via `.contour()`.
- New `tests/test_reins_bivariate.py`; sync any DecL into `test_decl.agg`.
- `uv run pytest` green (`UV_LINK_MODE=copy`).

## Portfolio sketch (NOT implemented in phase 1)

Under unit independence (consistent with the existing portfolio totals): align
all units to a common `(bs_ceded, bs_net)` grid; each unit's joint occ
`(C_u, N_u)` density is computed as above (units without occ reins are degenerate
on the net axis: `ceded = 0`). The portfolio joint `(ΣC_u, ΣN_u)` is the 2D
convolution across units → multiply the per-unit 2D aggregate-density FFTs and
invert. Open issues for that phase: common-grid sizing across heterogeneous
units, memory (2D × number of units), and whether to expose a
`Portfolio.occ_bivariate` mirroring the univariate aggregation. Left as a
forward sketch.

## Close-out

- Bump `pyproject.toml` to the next `1.0.0a*` (exact number depends on execution
  order relative to the two reins plans).
- README.rst bullet: experimental 2D-FFT bivariate occ ceded/net on `Aggregate`.

## Open / watch

- **Aliasing / window tension** is the main technical risk: the FFT grid must
  cover each aggregate's full effective support (else wrap-around aliasing),
  while small `bs` per axis is wanted for resolution. The auto-sizing from the
  univariate margins is what reconciles these and keeps the 2D array feasible.
  Validate with the marginal / additivity checks above; surface a deficit warning
  analogous to the univariate path.
- Offset windows (a linear-phase shift to model a non-zero-floor net) are a
  possible later refinement; phase 1 keeps zero-offset `[0, max]` axes.

## What actually shipped (2026-06-02, 1.0.0a20)

Built essentially as planned. Notable points and divergences:

- **`freq_pgf` is *mathematically* elementwise but not always *implemented*
  that way.** `FrequencyEmpirical.freq_pgf` (which backs `dfreq`) sums over the
  frequency support via a `matmul` that assumes a 1-D argument, so it raises on
  a 2-D `z`. Fix: `freq_pgf(self.n, z.ravel()).reshape(z.shape)` — universally
  safe (closed-form PGFs are elementwise) and makes every frequency type work.
- **Marginal *means* match the univariate aggregates exactly** (the linear
  scatter preserves the first moment), independent of bucket size. **Marginal
  *cv* matches only at the matched grid** (`bs_ceded == bs_net == self.bs`):
  linear rebucketing adds `bs²·f(1-f)` to a density's second moment, so when a
  ceded mean is comparable to the model bucket the *univariate* (model-grid)
  ceded cv is inflated and the bivariate's auto-sized **finer** ceded axis is the
  more accurate one. Tests assert means on the auto grid and the full cv identity
  on a forced matched grid. (This was the one real surprise — the cross-check
  against `reins_stats_df`/`reins_describe` is mean-level on auto grids.)
- **Correlation is positive even for a fixed count.** The per-claim `(c, n)` are
  already positively associated through the cession map (both rise with claim
  size in the layer), so a deterministic `N` does *not* give ~0 correlation; a
  random (Poisson) count *adds* a common-shock coupling on top. The test asserts
  the real invariant: `corr(Poisson) > corr(fixed) > 0` on the same book.
- **No renormalisation.** Mirroring `_fft_aggregate`, the raw 2-D density is kept
  (sub-1e-15 dust zeroed) and the lost tail mass is reported as `meta['deficit']`;
  `BivariateDistribution.moments`/`corr` normalise by the on-grid total.
- **Files as planned:** new `src/aggregate/bivariate.py`
  (`BivariateDistribution`, `size_axis`, `scatter_bivariate`),
  `Aggregate.occ_bivariate` in `distributions.py` (lazy import of the submodule
  to avoid a cycle; `scipy.fft as sfft` for `rfft2`/`irfft2`),
  `tests/test_reins_bivariate.py` (31), DecL section Z (`BV.*`). 777 pytest pass.
  The docs subsection in `2_x_re_pricing.rst` is still pending a manual rebuild.
- **Portfolio `occ_bivariate` not implemented** — left as the forward sketch
  above.
