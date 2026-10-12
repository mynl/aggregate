# Note: fuzz removal and the empirical skewness, second occurrence

Status: note, 2026-10-11, written from the `reproductions` book session. Not a plan; no code
changed. It extends the TODO entry **[Fuzz-Removal-Phantom-Tail-Mass]** (`dev/TODO.md`, diagnosed
2026-10-07, full account in `dev/done/plan-a400-v1-loose-ends.md` phase 3), which was parked
because the fix moves `est_skew` corpus-wide. This is the second time the defect has surfaced in
real work, and the measurements below change the recommendation in that entry.

## The second occurrence

The `Pittarello2024` chapter of the reproductions book reruns GEMAct's Table 2 (Pittarello,
Luini and Marchione, AAS 2024), which computes on GEMAct's own grid, $2^{22}$ buckets of 500:

```python
SEV = 'sev lognorm 84541.68 cv 2.1022565437545127'
a = build(f'agg PLM.Gross 3 claims {SEV} poisson', bs=500, log2=22)
```

`est_skew` is **11.916** against a theoretical **7.2841** (`validation_df.loc['Agg', 'Sk']`).
The default grid (`bs=4000`, `log2=16`) gives 7.2717 and is what the chapter reports, with one
sentence on the defect.

The mechanism is the one in the TODO entry, unchanged. `remove_fuzz` (absolute threshold
`eps = 2.22e-16`) zeroes 3,714,008 buckets carrying `2.002e-11` of genuine tail mass. Then
`xsden_to_mwrangler` reads `pg = 1 - den.sum() = 2.002e-11` as a truncation deficit, so it puts
that mass at `xs[-1] + bs = 2.097e9` and adds `pg * edge**3 = 1.847e17` to an `ex3` of
`3.933e17`. The raw density gives 7.2841.

| `log2` | `est_skew` | raw density | buckets zeroed | mass removed |
|---:|---:|---:|---:|---:|
| 18 | 7.2257 | 7.2255 | 0 | 0 |
| 20 | 7.3403 | 7.2822 | 568,974 | 1.976e-11 |
| 22 | 11.916 | 7.2841 | 3,714,008 | 2.002e-11 |

At `log2=18` the grid genuinely truncates (`pg = 4.3e-12`, no buckets zeroed), and relocating
the deficit to the edge is the intended behavior.

## The new finding: the raw density is not the fix either

The parked entry measured the raw density as better at every grid tried and held back only
because the case that motivated `remove_fuzz` was never re-found. The case is easy to find: put
a small-support distribution on a grid far wider than it needs, and the raw density goes wrong
too.

| model | `log2` | theory | `est_skew` | raw density | noise floor, k = 1 |
|---|---:|---:|---:|---:|---:|
| `agg Dice dfreq [10] dsev [1:6]` | 16 | 0 | 7.9e-4 | -1.8e-4 | 6.8e-7 |
| same | 20 | 0 | **4.875** | **2.010** | -2.4e-15 |
| `agg PoiDie 10 claims dsev [1:6] poisson` | 20 | 0.39351 | **1.3528** | **0.54944** | 0.39351 |
| `agg LN 1 claim sev lognorm 200 cv 2 fixed`, `bs=6.427` | 21 | 14 | **142.51** | 14.000 | 14.000 |
| `agg LN2 5 claims sev lognorm 200 cv 2 poisson`, `bs=6.427` | 21 | 5 | **15.304** | 4.9999 | 4.9997 |
| GEMAct gross, `bs=500` | 22 | 7.2841 | **11.916** | 7.2841 | 7.2826 |

All `bs=1` except where stated. "Raw density" and "noise floor" are skewness computed directly
from the density, normalized by its own sum, with no deficit relocation.

The two failures are mirror images:

- **The absolute threshold is too high for a long genuine tail.** A lognormal aggregate on a tall
  grid has real density between `1e-26` and `2e-16` over millions of buckets, and `eps` throws it
  away. The relocation of the deficit to the grid top then turns that loss into a large third
  moment.
- **The absolute threshold is too low for noise on a wide grid.** The inverse FFT leaves
  sign-indefinite noise of order `1e-17` to `1e-18` in every bucket the distribution never
  reaches. That is below `eps`, so some of it survives, and the raw density keeps all of it.
  Weighted by `x**3` out to `x = 1e6` it dominates the third moment of a distribution whose
  support ends at 60.

## A threshold measured from the noise itself

The noise is sign-indefinite and the true density is non-negative, so the negative values
measure the noise. Take `floor = max(-raw[raw < 0])`, or 0 when nothing is negative, and zero
every bucket below `k * floor`. The rightmost column of the table above is `k = 1`. It is within
1e-6 for both die models (exact to rounding except ten dice at `log2=16`, 6.8e-7), matches the raw
density on the pure lognormal cases, and comes within 2e-4 relative on the GEMAct case. `k = 4`
makes ten dice at `log2=16` exact too (6.5e-16), and `k = 4` and `k = 16` lose a little more of the
lognormal tail (7.2813 and 7.2792 at `log2=22`). The measured noise floors were between
4e-19 and 1.4e-17, always well below `eps`.

Measured on these five models only. Three things to settle before it is a plan:

- **No negatives, no information.** When nothing is negative the floor is 0 and the density is
  used raw. That held in every case here where it occurred, but a grid with positive-only noise is
  conceivable. A fallback such as `eps * max(raw)` keeps the current behavior there.
- **One extreme value sets the floor.** A percentile of the negative magnitudes may be steadier
  than the maximum. Neither was tested.
- **The removed mass is not a truncation deficit.** Whatever threshold is used, `pg` should come
  from the raw sum (`1 - raw.sum()`), so that only a genuine shortfall is relocated to the grid
  top. On the GEMAct case at `log2=22`, the current cleaned density with no relocation gives
  7.2691 (0.2% low, the genuine tail mass being lost) instead of 11.916. This is the cheap interim
  the TODO entry names, and it is independent of the threshold. It does nothing for the die rows,
  where the error is noise that survives the threshold, not mass that the threshold removes.

## Blast radius

`remove_fuzz` is called by `update_work` for the moments (`_aggregate.py:4568`), by
`density_df.p_total` (`_aggregate.py:1850`), at `_aggregate.py:4684` and `:7904`, by
`reins_density_df`, and by `Portfolio.remove_fuzz` (`_portfolio.py:574`). Only the moments
call feeds `x**3` and the deficit relocation; the others are display and probability views,
where the absolute threshold does little harm. A first plan could change the moments call alone,
which limits the corpus-wide movement to `est_skew`, `est_cv` and any validation flag that reads
them.

The two comments the TODO entry says are owed are still owed. The comment above
`_aggregate.py:4568` says the aggregate "has no negative density even under aliasing", but
812,365 buckets are negative in the GEMAct case. The `remove_fuzz` docstring says zeroing
`|x| < eps` is "safe and lossless", but it removed 2e-11 of genuine mass there.

## Tests that would pin it

The six rows of the table above, each against `validation_df.loc['Agg', 'Sk']`, with tolerance
set at 1e-6 absolute for the die models and 1e-3 relative for the others. The lognormal rows are
the regression for this defect, and the die rows are the regression for the one `remove_fuzz` was
written to fix.

The measuring scripts are short; the core is:

```python
raw = a.agg_density
neg = raw[raw < 0]
floor = -neg.min() if len(neg) else 0.0
clean = np.where(raw < floor, 0.0, raw)
p = clean / clean.sum()
m = (a.xs * p).sum(); v = ((a.xs - m)**2 * p).sum()
skew = ((a.xs - m)**3 * p).sum() / v**1.5
```
