# Notes [NetCeded-Kappa-Band]: the conditional band around the kappa curve, in core or on disk

> **Status: NOTES, 2026-08-14, not a plan and nothing implemented.** Written from a working session that built the netceded joint at full granularity, on disk, and drew the chart the author wants. Every number below was measured in that session against LIB `1.0.0a275`; line anchors are the same version. The follow-on work has an obvious shape but the phases here are a proposal, not an approved plan. Companion documents: `dev/done/plan-natural-allocation-to-occurrence-net-ceded.md` (the plan that shipped `a273` `exeqa_df` and `a274` `natural_allocation`), `dev/done/plan-pricing-natural-allocation.md` (the app-facing four subtabs, symlinked from the API repo), and `dev/done/plan-bv.md` (the massive route).

## 1. The rationale

`exeqa_df` gives the kappa curve, the conditional mean cession given the gross outcome, and `natural_allocation` prices with it. Both are means, and a mean is exactly the wrong summary for the question a cedent actually asks about an occurrence program, which is not "what do I cede on average when the year comes in at 500" but "having come in at 500, how much of that could I have been ceding, and how much am I actually ceding". The gross outcome does not determine the cession: the same 500 can arrive as one claim of 500, ceding 50, or as five claims of 100, ceding 250. The kappa curve averages that away by construction, and the allocation inherits the averaging, which is correct as pricing and silent as description.

The conditional distribution of `C` given `G = g` is already sitting in the joint, one row at a time, and it is cheap. Reading two percentiles off each row turns the kappa curve into a curve with a band, and the band is where the story is. Measured on `agg Demo 10 claims sev lognorm 50 cv 1.5 occurrence net of 50 xs 50 poisson`, at a gross outcome of 500 the mean cession is 103.6 while the first and ninety ninth percentiles are 48.0 and 172.5, so the realized share of that outcome ceded runs from under a tenth to over a third. At `g = 200` the band runs from 0.0 to 67.0, meaning a year of that size can cede nothing at all. No summary built out of means says that.

There is a second reading that belongs next to the band, the deterministic ceiling. For a single occurrence layer `limit xs attach` the largest cession achievable with a gross total of `g` comes from splitting `g` into claims of exactly `attach + limit`, each ceding the full limit, so the ceiling is a comb with teeth every `attach + limit` and a maximum share of `limit / (attach + limit)` at each tooth. Drawing it next to the ninety ninth percentile says how much of the theoretically available cession the program actually delivers: on the demo the ceiling peaks at 0.5 and the measured p99 share peaks at 0.400 (at `g = 125`), 0.364 (at `g = 275`) and 0.353 (at `g = 400`), so even the good tail of the conditional law gets nowhere near the ceiling, because getting several claims to land exactly at the top of the layer is a lot to ask.

## 2. What works today, and the one hole

Measured by calling every analytic surface on a disk backed netceded joint (65,536 x 2,048 at `bs = 0.5`):

| Surface | In core | Massive |
|---|---|---|
| `bs_window_df`, `axis_support_df`, `summary_df`, `stats_df`, `validation_df`, `dependency_df`, `tail_df`, `plot` | works | works |
| `marginals`, `moments`, `corr` | works | works, off the pass 3 accumulators, no disk read |
| `density_df` | works | refuses, by design, a full frame would read the store |
| `exeqa_df` | works | **refuses** (`bivariate.py:2210`) |
| `natural_allocation` | works | **refuses**, only because it calls `exeqa_df` |

So the massive route is already first class for everything except the kappa curve and the allocation that rides on it. That is the hole, and it is one loop wide. The kappa curve is a row wise reduction of the joint, `d @ y` against `d.sum(1)`, and a row band is exactly what the massive store hands out cheaply. This was anticipated: `[Gross-Anchored-Kappa-Insight]` in `dev/done/plan-pnl-consolidated-xpnl-walk.md` records that G slices are axis aligned row averages and therefore one pass even on the massive route, and `dev/TODO.md` `[Massive-Kappa-Second-Sweep]` expects that insight to dissolve most of its own scope.

## 3. The machinery, one version for both worlds

### 3.1 [Joint-Row-Bands], the single iterator

Everything below is a row wise fold. Give `BivariateAggregate` one private band iterator and each consumer stops caring where the density lives.

```python
def _row_bands(self, axis=0, band_rows=None):
    """Yield ``(r0, r1, block)`` row bands of the joint, in core or on disk.

    Parameters
    ----------
    axis : {0, 1}, default 0
        The conditioning axis. ``1`` yields bands of the transpose, so a
        consumer always folds along rows.
    band_rows : int, optional
        Rows per band. Defaults to the whole grid in core (one band) and to
        the store's row chunk on the massive route, which is the read size
        the chunking was chosen for.

    Yields
    ------
    (int, int, ndarray)
        Half open row range and the dense float64 block for it.

    Notes
    -----
    The in core case yields exactly one band, so a consumer written against
    this iterator costs nothing there: no copy, no chunk arithmetic, one
    pass either way. On the massive route the band is the unit of disk read,
    and peak memory is ``band_rows * n1 * 8`` bytes regardless of grid size.
    A disk backed transpose (``axis=1``) reads column bands instead, which
    the zarr chunking supports at the same cost, so the transpose is not a
    special case for the caller.
    """
```

### 3.2 [Kappa-Band-Columns], the frame

`exeqa_df` grows one keyword and loses its massive refusal. The columns follow the existing naming, which already carries the axis name (`exeqa_Ceded`), so the percentile columns do too.

```python
def exeqa_df(self, axis=0, levels=None):
    """The kappa curve, optionally with conditional percentiles.

    Parameters
    ----------
    axis : {0, 1}, default 0
        The conditioning axis, as today.
    levels : sequence of float, optional
        Probability levels. Each one adds a ``q<pp>_<other axis>`` column
        holding the conditional quantile of the other axis given the
        conditioning one. ``None`` (default) keeps today's columns exactly,
        so this is additive.

    Notes
    -----
    The quantiles route through :class:`GridDistribution` on the row's own
    normalized mass, one per live row, so the probability vocabulary stays
    the library's one implementation and a lattice law's atoms are handled
    the way every other quantile in the package handles them. Quantiles
    commute with the monotone map ``c -> c / g`` at fixed ``g``, so a share
    band is the value band divided by the index and needs no separate pass.

    A row with no mass yields ``NaN`` for every quantile column, matching
    the existing treatment of a conditional mean given a null event.
    """
```

Measured cost on the 65,536 row grid: the kappa fold alone is a few seconds, and adding two levels takes the sweep to 38.6 s, all of it in the per row `GridDistribution` construction rather than the disk. Three ways to cut it if that matters, in order of preference: build the quantile kernel only on rows inside a caller supplied CDF range, since the plotted range is a few thousand rows rather than sixty five thousand; hand the whole band to one vectorized `searchsorted` on the row cumulatives, which is what `make_var_tvar` does per row anyway; or leave it, because 38 s once per joint is not the bottleneck in any workflow that just spent 72 s building the joint.

`natural_allocation` needs no change beyond calling the new `exeqa_df`: it consumes the kappa column and the gross marginal, both of which the band sweep already produces.

### 3.3 What the caller writes

```python
kb = big.exeqa_df(axis=0, levels=(0.01, 0.99))
#   columns: p, F, S, exeqa_Gross, exeqa_Ceded, q01_Ceded, q99_Ceded
alloc = big.natural_allocation(dist, P=600)
```

Identical in core and on disk, which is the point of section 3.1.

## 4. [NetCeded-Exact-Lattice], why both axes share a bucket size

The netceded joint forces one `bs` on both axes, hardwired at `bivariate.py:924` (`bs_x = bs_y = bs`). Copula mode takes a per axis `bs` tuple and is right to. The constraint is worth writing down because it is not obvious and it is not about the conditioning: slicing a row works perfectly well on mismatched lattices. Three reasons, in order of force.

**The third view is a subtraction on the index.** The three views satisfy `ceded + net = gross` pointwise, so whichever view is not on an axis is read as `g` minus the kappa curve rather than measured again. `natural_allocation`'s Notes say so directly, under "What the third view costs". That is what makes `kappa_C + kappa_N` the identity exactly, and in turn what makes `A_C + A_N = P` exact by construction rather than to within a rounding. It requires `g - c` to be a lattice value, which requires one lattice.

**The comonotone curve wants to land on the lattice.** Per claim the cession map is flat, then unit slope through the layer, then flat. On a shared lattice whose `bs` divides the layer boundaries, and whose gross sample points are lattice points, every per claim point `(x, c(x))` is a cell center and the bilinear scatter never fires, so the joint carries no rebucketing smear at all and the kappa curve is exact rather than accurate to the scatter. Measured on the demo, maximum off lattice offset of the sampled view images:

| joint `bs` | gross axis | ceded axis | scatter |
|---|---|---|---|
| 4.0 | 5.0e-01 | 5.0e-01 | splits, every point |
| 0.5 (the gross `bs`) | 0.0e+00 | 0.0e+00 | never fires |

**One `bs` keeps the sizing honest across entry points.** The comment at `bivariate.py:987` records the third reason: sizing the common `bs` from the 2-D budget rather than inheriting the gross `bs` is what makes `occ_bivariate` and the DecL `grossceded` prefix agree regardless of how fine the gross aggregate happens to be.

The exactness above is a property of excess of loss layers on an aligned lattice, not a general one. A **share** cession multiplies by a non integer factor and generally cannot be made exact at any usable bucket size: measured on `agg ... 500 xs 0 sev ... occurrence net of 0.3 po 100 xs 100`, at the gross `bs` of 0.125 the ceded images sit half a bucket off the lattice, and `_lattice_bs` of those images is 0.000375, so an exact grid would need a gross axis of absurd length. The rule to write into the sizing code is therefore "prefer the exact lattice when it is affordable, report when it is not, never pretend", not "always be exact".

`_lattice_bs` (`bivariate.py:1042`) already computes exactly the quantity wanted here, the largest bucket size placing every atom on the grid, and is currently used only by discrete mode (`bivariate.py:1711`). The netceded default should reuse it rather than grow a second implementation.

## 5. [NetCeded-Measured-Sizing], size from the support, not from a square

The two windows in a netceded joint are structurally asymmetric and the sizing should say so out loud. Ceded is capped by the cover, gross is whatever the severity's tail is. Measured on the demo at the bivariate window setting of nine nines: gross window 28,906, ceded window 710.5, against a gross mean of 500. The gross aggregate's own grid is `bs = 0.5`, `log2 = 16`.

The grid that follows from those three numbers is `bs = 0.5` with `log2_x = 16` and `log2_y = 11`, and it is the grid at which nothing is lost: the bucket size matches the gross grid so the scatter never fires, and each axis covers its own measured window with no more room than it needs. The realized ceded support runs to 1,500 buckets, so `log2_y = 11` (2,048) has headroom and `log2_y = 10` would clip.

The square grid is a mistake here, and the session measured how much of one. At `16 x 16`, the same `bs`, every answer is identical to `16 x 11` to every digit printed (deficit 5.666e-09, correlation 0.794125, ceded mean 94.16629224, against an exact `reins_stats_df` occ ceded mean of 94.16629), while the ceded axis carries mass on 2.29% of its buckets. The cost of that identical answer was 1,491 s against 71.9 s and roughly 65 GB of transient staging against 3 GB. Final stores are the same size (0.35 GB) because zarr compresses the structurally zero chunks to nothing, so the price is entirely in the dense complex staging arrays, which is the general shape of the tradeoff on the massive path: what hurts is `2**(log2_x + log2_y - 1)` complex cells written twice, not the density that is kept.

The proposed default, which reproduces `16 x 11` without being told:

```
bs_exact = the largest bucket size placing the gross grid and every ceded
           image on one lattice (_lattice_bs over the images, folded with
           the gross bs)
bs       = bs_exact when the implied cell count fits the budget, else the
           coarsest round_bucket value that does fit, with the loss of
           exactness reported rather than silent
log2_i   = ceil(log2(hi_i / bs + 1)), measured per axis, independently
```

Two consequences worth stating. The budget stops being a square split and becomes a total, which is what `total_log2` already means everywhere else. And the recommendation should be readable before paying for the build, so that a caller can see "this wants 2**27 cells and 0.35 GB on disk" and decide, rather than discovering it afterward. The existing `bs_window_df` and `bs_explanation` are the right shape for that report but currently exist only on a built object.

## 6. [Chart-Kappa-Band], the picture

The chart is two panels and reads entirely off the frame from section 3.2 plus the layer parameters for the ceiling. It should be a registered chart in `charts/_emit_bivariate.py` alongside `joint_surface`, following that file's pattern, with a predicate that accepts a massive joint (the `joint_surface` predicate is `density is not None`, which excludes them) since the whole point of the band is that it survives the disk route.

**Left panel, the kappa curves.** `kappa_C(g)`, `kappa_N(g) = g - kappa_C(g)`, the identity line, and both bands. The net band is the ceded band reflected in the identity, `[g - q99, g - q01]`, because `N = G - C` pointwise. Two shaded regions of equal width, one hugging zero and one hugging the diagonal, is the conservation statement in one picture, and it is the panel to lead with.

**Right panel, the share.** The band and the mean divided by the index, plus optionally the deterministic ceiling from section 1. The ceiling is a single closed form for one occurrence layer and should be drawn only when the program is one layer, since a tower has no such simple envelope.

Three decisions the emitter should make rather than the caller.

The plotted range is a probability window on the conditioning marginal, not a mass floor. A raw threshold like `p > 1e-4` means different things at different bucket sizes: on the `bs = 0.5` frame the largest row mass is 8.9e-04, so that threshold discards most of the picture, while on the `bs = 4` frame it keeps nearly all of it. A CDF range of `1e-3` to `0.999` means the same thing on every grid and here gives `g` in [33, 2216].

Smoothing of the band edges is display only and should default to off. The edges are lower quantiles of a lattice law and step by whole buckets, so a centered rolling mean over a few tens of buckets tidies them, but the visible structure below `g = 500` is real: it is the comb of section 1, the k claims at the top of the layer boundaries, and smoothing it away removes the mechanism rather than noise.

The band is a percentile band, so the legend should say so and never say "confidence interval". Nothing here is an estimate with sampling error; the joint is the law.

## 7. Findings from the session, unrelated to the band but blocking it

Three defects, all reproduced, none fixed, all in the way of anyone trying to build a joint big enough to want the disk route.

**[Budget-Clip-Destroys-Axis]**, `bivariate.py:1017` to `1032`. `total_log2` defaults to 20. When a caller pins both axes over that budget the clip rule cuts "the wider axis", and with both pinned equal it cuts axis 0 to `_MIN_AXIS_LOG2`:

```
bs=0.5, log2_x=16, log2_y=16, default budget  ->  n0=16, n1=65536, clipped=True
```

A 16 bucket gross axis is not a tail deficit, which is what the warning calls it. When both axes are pinned and the pair exceeds the budget, raising is the honest answer, since the caller has stated both numbers and neither can be honored.

**[Occ-Bivariate-Passthrough]**, `_aggregate.py:1520`. `occ_bivariate` accepts `bs`, `log2_x` and `log2_y` and nothing else, so it can pass neither `total_log2` nor `store_dir`, and the two are coupled: without the first no grid can be large enough to want the second. The massive netceded path exists and works (`bivariate.py:1775` to `1805`) but is unreachable from the public method; the session drove it by constructing `BivariateAggregate(mode='netceded', ...)` directly. This is also how the plain call goes wrong: `a.occ_bivariate(views=('gross','ceded'), bs=0.5)` returns a 512 x 2,048 object with a **deficit of 0.535** and a correlation of **-0.2225** for a comonotone pair, warning once and then answering questions.

**[Netceded-Update-Log2-Noop]**, `bivariate.py:1754` to `1774`. The warning above advises "Raise `update(log2=...)`", which does nothing on a netceded joint: `_update_netceded` forwards `log2` only to the inner aggregate, and only when the bv built that aggregate itself, while sizing always reads `_nc_kwargs`. Verified: `update(log2=27)` and `update(log2=(16, 11))` both leave the shape and the deficit unchanged. Either the parameter should work or the message should stop recommending it.

A fourth, smaller: `BivariateDistribution` has no `slice`, so the analyst probe for one conditional law (`slice(x=500)` returning the normalized `GridDistribution`) exists only on the massive container (`bivariate.py:3714`). The in core case can do it in two lines, which is exactly why it should be the same two lines, on the same method name, in both containers.

## 8. A proposed order of work

Small, and each step is independently useful.

**Phase [Sizing-And-Passthrough]**. Fix the three findings in section 7 together, since they are one story: `occ_bivariate` gains `total_log2`, `store_dir` and the chunk arguments; the pinned and over budget case raises instead of clipping; the warning text stops advising a parameter that does nothing. Add the measured default of section 5 so that `occ_bivariate(views=('gross','ceded'))` picks the exact lattice when it is affordable. This phase is a prerequisite for anyone reproducing the pictures.

**Phase [Joint-Row-Bands]**. The iterator of section 3.1, plus moving `slice` down to the shared container. No behavior change, one new private method, and it is what makes the next phase small.

**Phase [Kappa-Band-Columns]**. `exeqa_df(levels=...)` on the iterator, which simultaneously removes the massive refusal from `exeqa_df` and from `natural_allocation`. Tests: the quantile columns bracket the kappa column; the mass weighted mean of the kappa column reproduces the other axis's marginal mean (already the `a273` invariant, measured at 94.16629224 against 94.16629224 in this session); the in core and massive routes agree cellwise, which the session measured at 1.5e-14 maximum absolute difference on identical sizing.

**Phase [Chart-Kappa-Band]**. The registered chart of section 6, with the predicate accepting massive joints.

Left out deliberately: `[Bivariate-Total-Exeqa]` (`dev/TODO.md`), the generic conditioning on `X + Y` rather than on an axis, stays deferred and untouched. The band described here conditions on an axis, which is the easy case and the one the netceded question asks.

## 9. Session measurements, for reference

All on `agg Demo 10 claims sev lognorm 50 cv 1.5 occurrence net of 50 xs 50 poisson` at LIB `1.0.0a275`, on 8 cores and 32 GB.

| Build | Wall | Store | Deficit | Correlation |
|---|---|---|---|---|
| `bs=4`, 13 x 8, in core | 0.69 s | n/a | 9.567e-11 | 0.793062 |
| `bs=4`, 13 x 8, massive | 4.33 s | 0.01 GB | 6.091e-11 | 0.793062 |
| `bs=0.5`, 16 x 11, massive | 71.9 s | 0.35 GB | 5.666e-09 | 0.794125 |
| `bs=0.5`, 16 x 16, massive | 1,491 s | 0.35 GB, 65 GB transient | 5.666e-09 | 0.794125 |

In core against massive at identical sizing: maximum absolute cellwise difference 1.5e-14, maximum marginal difference 2.9e-13, correlations identical to six places. Note the two paths ran at different padding (1 in core, inherited from the aggregate; 0 on the massive route, its documented default) and it made no difference at a nine nines window.

Band sweep over the 16 x 11 store, kappa plus two levels, 38.6 s for 65,536 rows. Streamed pushforward of the ceded share `c / g` over the same store, one pass, 19.0 s, giving the unconditional law of the share: mean 0.178 (against `E[C] / E[G] = 0.188`), median 0.186, 95th 0.299, 99th 0.343. That is a different and complementary reading to the conditional band and comes free with machinery that already exists.

An observation, not a finding: the staging stores `z1` and `z2` hold dense complex FFT data, which does not compress, so the default zarr compressor is pure cost on them while being a large win on `density.zarr` (0.35 GB against 1.07 GB dense at 16 x 11, and against 32 GB dense at 16 x 16). Worth a measurement before anyone tunes it, since the massive path is I/O bound and the observed throughput was around 35 MB/s with roughly half a core busy.

Standing note for the examples: this demo program uses an **unlimited** lognormal severity, which is why the gross window is 28,906 against a mean of 500 and why the gross axis needs `log2 16` at all. Real policies carry limits. A bounded severity shrinks every number in this document and the sizing question with it, so any example written from these notes should put a limit on the severity.
