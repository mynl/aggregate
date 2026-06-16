# multivariate.py punch-up — sizing / coverage reconciliation (starter)

> **STATUS: NOT STARTED — starter notes. This is the MV follow-on AFTER
> `plan-numerics-4` lands** (numerics-4 integrates the 1A/1P windowing primitives
> from `plan-bucket-window-2.md` into `multivariate.py`; this plan then
> reconciles the two axis-sizing code paths and settles the `[multivariate]`
> config knobs on top of that windowed foundation). Captured 2026-06-05 while
> scoping `dev/plan-config.md`. A seed for the tuning pass, not a finished plan.
> The one config item that *was* clean (the `_WINDOW_NINES` de-dup →
> `[multivariate] window_nines`) ships with the config work; everything below is
> deferred to this effort so we don't pin half-tuned values into config.

## Why this exists

`multivariate.py` sizes the 2-D axis grids with **two separate code paths that
disagree** on grid size, memory cap, and tail-coverage notion. The 2-D grid
memory scales as the *square* of the per-axis length, so these choices matter a
lot. Reconciling them (share one scheme, or keep two but make them deliberate)
is the core of the tuning pass; only then should the resulting knobs land in
`config`'s `[multivariate]` section with settled defaults.

## The inconsistency

Two axis-sizing paths:

- **copula mode** — `MultivariateAggregate._size_axis` (`multivariate.py:565`),
  called from `update` (`:623`). Moment-based: `estimate_agg_window(m, sd, skew,
  p)` with `p = 1 - 10**-_WINDOW_NINES`.
- **netceded mode** — module `size_axis` (`multivariate.py:71`), called from
  `build_netceded_joint` (`:256`/`:258`). Empirical: upper `quantile` of the
  computed margin cdf.

| | copula (`_size_axis`) | netceded (`size_axis`) |
|---|---|---|
| target `default_log2` | **9** | **10** |
| `cap_log2` (memory guard) | **11** (≈ 2048² ≈ 4M cells) | **14** (≈ 16384² ≈ 268M cells) |
| coverage basis | moment window (`estimate_agg_window`) | empirical-cdf quantile |
| coverage value | **12 nines** (`_WINDOW_NINES`) | **9 nines** (`quantile = 1 − 1e-9`) |

Open question for the tuning pass: **should the two modes share one sizing /
coverage scheme, or stay deliberately distinct?** (They use genuinely different
inputs — moment estimate vs. realised margin — so a single scheme is not
obviously right.) The answer determines whether `[multivariate]` gets one shared
`{default_log2, cap_log2, coverage}` triple or per-mode fields
(`copula_*` / `netceded_*`).

## Tuning knobs (genuine — candidates for `[multivariate]` once settled)

| Knob | Location | Value | Role |
|---|---|---|---|
| `default_log2` (copula) | `_size_axis` :565 | 9 | target axis grid |
| `cap_log2` (copula) | `_size_axis` :565 | 11 | axis memory cap |
| `default_log2` (netceded) | `size_axis` :72 | 10 | target axis grid |
| `cap_log2` (netceded) | `size_axis` :72 | 14 | axis memory cap |
| `quantile` (netceded) | `size_axis` :72 | 1 − 1e-9 | empirical coverage (reconcile vs `window_nines`) |
| `window_nines` (copula) | `_WINDOW_NINES` :68 | 12 | moment-window coverage — **already → `[multivariate] window_nines` in config** |

## Guards / cosmetics (leave as literals — NOT config; exposing = speculative magic)

| Literal | Location | Value | Role |
|---|---|---|---|
| `8.0 * sd` | `_size_axis` :575/:576, `build_netceded` (size region) :343 | 8.0 | fallback window width when `estimate_agg_window` raises |
| `2.0 * m` | `_size_axis` :578 | 2.0 | degenerate-margin (point-mass) fallback |
| `1e-9` | `_size_axis` :579 | 1e-9 | `vmax` floor (degenerate guard) |
| `1e-15` | netceded :283, copula :712 | 1e-15 | 2-D density dust floor (`|density| < 1e-15 → 0`) — could tie to a shared noise floor if ever exposed |
| `levels=14` | `plot` :942, `contour` :1114 | 14 | contour count (cosmetic; could be `[plotting]` but fine as a method default) |
| `padding=1` | `update` :590 | 1 | FFT zero-pad (already a kwarg, mirrors 1-D build) |

## Suggested approach (for when the pass starts)

1. Decide shared-vs-per-mode sizing. Benchmark copula vs netceded memory/accuracy
   at a few `(default_log2, cap_log2)` settings on representative books.
2. Reconcile the two coverage notions — likely express both as "nines" so the 9
   (netceded) vs 12 (copula) gap is a deliberate, visible choice rather than an
   accident of two code paths.
3. Land the settled knobs in `config`'s `[multivariate]` section (the
   `MultivariateSettings` dataclass is already provisioned to grow).
4. Consider the `1e-15` denoise: keep as a local literal, or expose as
   `[multivariate] denoise_floor` if tuning shows the 2-D dust floor matters.

## Cross-references

- `dev/plan-config.md` — owns `[multivariate] window_nines` (the clean de-dup);
  `MultivariateSettings` is the home these knobs will join.
