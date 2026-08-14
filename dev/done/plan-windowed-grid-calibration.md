# Plan: `[Windowed-Grid-Breaks-Calibration]`, the pricing calibration ignores the window offset

> **Status: LANDED (1.0.0a289).** Implemented in the classic branch of
> `_pricing.calibrate_distortions`; the per-kind `Distortion.<kind>.calibrate`
> math is untouched and stays pure and 0-based. The receipt convention matches
> the signed branch: `L`, `P`, `a` slide, `M`, `Q`, `coc` are shift invariant.
> Tests: `tests/test_windowed_calibrate.py` (11 cases). A grid already at the
> origin takes `x0 = 0` and is byte-for-byte unchanged.
>
> **Goal (acceptance):** a windowed build and the same program forced onto a
> zero-based grid calibrate to the same distortion shapes and the same
> pentagon.
>
> Raised from the terminal 2026-08-14 against LIB `1.0.0a285`, API `1.0.0a104`,
> with the author's suspicion that the windowed `bs` was the cause. It was.

---

## 1. The report

This program "doesn't calibrate right":

```
agg LimitProfile
  [10000 20000 5000] premium at [0.8 0.7 0.5] lr
  [1000 2000 5000] xs 0
  sev lognorm 50 cv 1.5
  poisson
```

**Confirmed, and the limit profile is incidental.** Any program whose grid gets
windowed hits this, and the error is always exactly the window offset.

The build itself is clean: mean, CV and skew all validate to 1e-9 and the
expected loss is 24,500 as declared. The grid is windowed, `bs = 1/2`,
`log2 = 16`, `x_min = 13,759.5`, `x_max = 46,527`.

The calibration is not clean. At `coc=0.10, p=0.99`, against the same program
forced onto a zero-based grid:

| | windowed (default) | zero based | |
|---|---|---|---|
| `L` | 10,732.41 | 24,491.92 | short by 13,759.50 |
| `M` | 1,695.69 | 444.83 | 3.8x too big |
| `P` | 12,428.10 | 24,936.75 | |
| `LR` | 0.864 | 0.982 | |
| `ph` | 0.411 | 0.784 | |
| `wang` | 0.872 | 0.229 | |
| `dual` | 3.079 | 1.306 | |
| `tvar` | 0.543 | 0.126 | |

The shortfall in `L` is `13,759.50`, which is `x_min` to the last digit:
`10,732.414357 + 13,759.5 = 24,491.914357`, and `E[min(X, a)]` read straight
off the pmf is `24,491.914357`.

## 2. Why

`_pricing.py` was written against a zero-based grid and never learned about
`_bucket_window`. The word `x_min` did not appear in the module. Both docstrings
said so out loud:

- `_limited_ev`: "`E[min(X, assets)] = bs · Σ_{x < assets} S(x)` on the full
  contiguous bs-grid". The body is `float(bs * S[S.index < assets].sum())`. On a
  windowed grid this omits the region `[0, x_min)`, where `S` is identically 1,
  whose area is exactly `x_min`.
- `_calibration_survival`: "from a 0-based `p_total` density (the full
  contiguous bs-grid)". It returns `S` over the window only, so the layer
  integral `∫₀^a g(S(x)) dx` that `Distortion.calibrate_set` runs is short by
  the same rectangle, because `g(1) = 1` for every distortion.

`_bucket_window.py:477` documents the window as deliberate ("so `q` / `F` /
plots are defined on the window, not from 0"), so this is the pricing side not
being told, rather than the windowing side misbehaving.

### 2.1 Two failure modes, and the quiet one is the `Aggregate`

**`Aggregate` loses both legs**, and they partly mask each other: the premium
target is computed from the understated `L`, and the layer integral it is
matched against is understated by the same rectangle, so the families converge
on plausible-looking numbers with small reported errors. Nothing warns. This is
the reported case.

**`Portfolio` loses only the second leg**, because `add_exa` caches an
`exa_total` that already handles the window and `calibrate_distortions` prefers
it. So `L` is right and the target is right, but the integral is capped at
`a - x_min`, and the calibration fails loudly and returns garbage. On
`port WindowTest` (two Poisson lognormal units, grid starting at 32,440):

| | windowed | zero based |
|---|---|---|
| `L` | 48,988.28 (correct) | 48,988.28 |
| `ph` | **-0.555** | 0.785 |
| `wang` | **inf** | 0.227 |
| `dual` | **inf** | 1.302 |
| `tvar` | **220.02** | 0.124 |
| error | -25,965.71 | ~1e-11 |

A negative PH index, infinite Wang and Dual parameters, and a TVaR at a
probability of 220. The error is arithmetic: the truncated integral cannot
exceed `a - x_min = 56,110 - 32,440 = 23,670`, against a target of 49,635.71,
so `23,670 - 49,635.71 = -25,965.71` exactly.

### 2.2 The real bug is a frame mix, not a missing term

`_limited_ev` is consistently in frame and `_calibration_survival` is
consistently in frame. What is wrong is that `a` and `P` are out of frame while
`exa` and `S` are in frame, and the coc inversion `P = ν·exa + δ·a` mixes them.
That is why the two arms fail differently, and why patching only `exa` makes
things worse.

The two arms also disagree about `exa` today, and the fix has to normalize
them: an `Aggregate` reaches `_limited_ev` (in frame), a `Portfolio` takes the
cached `exa_total` (out of frame, and correct). Hence the `+ x0` on one side and
not the other.

## 3. The fix: slide the window, do not pad

Author's framing, 2026-08-14. A spectral risk measure is translation
equivariant, so for `X' = x_min + X`, `ρ(X') = x_min + ρ(X)`. Calibrating `X'`
to a premium `P` is calibrating `X` to `P - x_min`. The same slide carries the
expected loss and the asset level, and `Q = a - P` and therefore `M = coc·Q`
are shift invariant, so the margin is untouched.

**Padding `S` with `x_min / bs` leading ones is not an option** (author): the
count is unbounded. A `Po(1e9)` at `bs=1` windows a billion buckets away from
the origin. The shift is O(1).

**LIB already had this machinery.** `_canonical_loss_frame` states the argument
in its own docstring, and the signed and payoff branch calibrates in frame and
un-shifts the receipt (`exa_z - c, P_z - c, a_z - c`). The windowed grid is the
same transform with `c = -x_min`. The classic branch just never applied it,
because `transform` triggers on `index.min() < 0` and the shift is clamped by
`c = max(0, -min(support))`.

### 3.1 Why not widen the canonical frame instead

Changing `c = max(0, -min(support))` to `c = -min(support)` and triggering
`transform` on `index.min() != 0` would route windowed builds through the
existing shifted branch. Three obstacles, so the targeted edit is safer:

- The `reins_view` refusal is keyed on `transform`, and a windowed reinsured
  aggregate is entirely normal, so legitimate calls would start raising
  `NotImplementedError`.
- `_canonical_loss_frame` trims FFT padding at `VALIDATION_NOISE`, so `c` would
  key off the genuine support rather than the exact `x_min`, moving numbers on
  every windowed build.
- `kind` is honored only on the classic path.

Keeping `transform` to mean "signed or payoff" leaves those three alone.

## 4. The edit

In the classic branch of `calibrate_distortions`. `x0` is read **after** the
`reins_view` selection, which is correct and general: the views share the
object's grid, checked on a reinsured aggregate.

```python
x0 = float(density.index[0])
if x0:
    density = pd.Series(density.to_numpy(), index=density.index - x0)
a_z = a - x0
if reins_view is None and 'exa_total' in obj.density_df.columns:
    exa = obj.density_df.loc[a, 'exa_total']      # already out of frame
else:
    exa = _limited_ev(density, obj.bs, a_z) + x0  # in frame, brought out
if lr is not None:
    coc = _coc_from_lr(exa, a, lr)                # out of frame, see 4.1
delta = coc / (1 + coc)
nu = 1 - delta
P = nu * exa + delta * a
S, ess_sup = _calibration_survival(density, obj.bs, a_z)
dists = Distortion.calibrate_set(
    S=S, dx=obj.bs, premium_target=P - x0, ess_sup=ess_sup,
    assets=a_z, el=exa - x0, names=names)
distortion_df, calibration_df = _calibration_frames(
    dists, coc, p_val, cdf_(a), exa, P, a)        # report out of frame
```

`ess_sup` needs no adjustment: `_calibration_survival` reads it off the density
index, which is now in frame. `pd` was already imported. `P` is computed out of
frame and the target passed in frame as `P - x0`, which is exact rather than
approximate: `ν + δ = 1`, so `ν·exa_z + δ·a_z = P - x0` identically.

The docstrings of `_limited_ev` and `_calibration_survival` were rewritten to
state 0-based as a **contract on the caller** rather than as an assumption about
the grid, which is what let this sit unnoticed.

### 4.1 `lr=` resolves out of frame (author ruling, 2026-08-14)

A loss ratio is scale free but not shift free, so the frame has to be chosen
rather than inherited. The signed branch resolves `_coc_from_lr` **in frame**,
reasoning that a canonical shift `c` stands for genuinely negative outcomes and
the in-frame premium is the one that means anything. For a windowed grid the
opposite holds: the window is an artifact of the grid, not of the risk, so a
reader who writes `lr=0.7` means the loss ratio on the premium they are shown.

**Ruling: out of frame on the classic path, in frame on the signed path.** The
two branches therefore read differently on the same `lr=` line, deliberately,
and both docstrings say so.

Verified: on the windowed `Aggregate`, `lr=0.90` reports `LR = 0.90000000` on
the real `L` and `P`, where the in-frame reading would have delivered
`0.79772731`.

One consequence worth knowing: the admissible range of `lr` differs by frame.
`_coc_from_lr` refuses when `P >= a`, so out of frame the constraint is
`lr > L/a = 0.8335` on the reported program, where in frame it would have been
`lr > 0.6868`. Out of frame is the range the reader can reason about, since both
ends are numbers they are shown.

## 5. Verification

The edit against the same programs forced onto a zero-based grid:

| case | before | after | reference |
|---|---|---|---|
| Aggregate windowed, `ph` | 0.410947 | 0.783835 | 0.783836 |
| Aggregate windowed, `wang` | 0.872481 | 0.228821 | 0.228820 |
| Aggregate windowed, `dual` | 3.078979 | 1.306208 | 1.306206 |
| Aggregate windowed, `tvar` | 0.542856 | 0.126160 | 0.126159 |
| Aggregate windowed, `L` | 10,732.41 | 24,491.91 | 24,491.92 |
| Portfolio windowed, `ph` | -0.555062 | 0.784813 | 0.784813 |
| Portfolio windowed, `wang` | `inf` | 0.226807 | 0.226807 |
| Portfolio windowed, `dual` | `inf` | 1.301734 | 1.301734 |
| Portfolio windowed, `tvar` | 220.015580 | 0.123577 | 0.123577 |

Fitted errors come back around 1e-11. The `Portfolio` reference is the arm that
now emits a `questionable convergence` notice (error 3.2e-05 on a target of
49,635, relative 6e-10) while the fixed windowed path does not, so the windowed
build is the better converged of the two.

Both already-zero-based cases, one `Aggregate` and one `Portfolio`, are
unchanged: `x0 == 0` makes `if x0:` false, so the density is not even copied,
and `± 0.0` is an exact no-op on a float.

## 6. The defect was confined to the calibration

Everything that *applies* a distortion is already window aware, so nothing else
needed the same treatment. Holding the distortion fixed and varying only the
grid:

- `Portfolio.price(0.99, PH(0.785))` agrees to 8 significant figures windowed
  against zero based, per unit and in total, and its total reproduces
  `layer integral over the window + x_min` exactly.
- `Aggregate.price` likewise: `L` 24,491.914 windowed against 24,491.920 zero
  based, `P` agreeing to 7 figures.
- `add_exa`'s `exa_total` is correct on a windowed grid (gap 0.0000).

So once `calibrate_distortions` hands back a correct distortion, the
allocation, the natural allocation, and the stand-alone frames are all right
with no further change.

## 7. Tests

`tests/test_windowed_calibrate.py`, 11 cases. There was no test asserting a
windowed calibration before, which is how an `inf` Wang parameter survived.

- windowed against zero based on both arms, shapes and full pentagon
- every family finite, converged, and in range on a windowed grid (the direct
  guard on the `Portfolio` garbage)
- the reported `L` equals `E[min(X, a)]` read off the pmf, rectangle included
- `M`, `Q`, `coc` shift invariant and the accounting identities close
- `lr=` out of frame, at two ratios, including that the in-frame reading is a
  different number
- a zero-based grid untouched
- the `a=` anchor takes the same route as `p=`
- a guard that both fixtures really do window, or the file is vacuous

## 8. Notes for the author

**`build(program, x_min=0)` is an `Aggregate`-only workaround.** It is the
documented escape hatch and it works on the reported program, picking
`bs=1, log2=16` on its own with no accuracy cost. It does **not** work on a
`Portfolio`: `x_min` is an `Aggregate.update` kwarg and `Portfolio.update`
raises `TypeError: Portfolio.update() got an unexpected keyword argument
'x_min'`. The whole-book test pins an explicit `log2`/`bs` instead. Whether
`Portfolio.update` should accept `x_min` for symmetry is a separate question,
not raised here.

**Nothing is owed on the API side.** The app calls `calibrate_distortions` and
renders what it is served, so the Calibrate, Stand-alone, Allocate and Evaluate
leaves were all reporting these numbers faithfully. With LIB fixed the app is
correct with no change, once it syncs (`uv sync --all-extras`, agreement 7 in
the oversight charter).

**Deferred, diagnosed but not fixed: `cdf` is `nan` below the window.** Tracked
in `dev/TODO.md` as `[Windowed-Cdf-Nan]`, awaiting the author's ruling. Same
root, different site and different blast radius. `Aggregate.cdf`
(`_aggregate.py:6357`) and `Portfolio.cdf` (`_portfolio.py:2073`) both build
`interpolate.interp1d(..., kind='previous', bounds_error=False,
fill_value='extrapolate')`, and scipy's `previous` kind has no previous knot
below the first, so it fills `nan` rather than 0. Confirmed in isolation on
scipy 1.17.1. The answer should be 0: the distribution has no mass below
`x_min` by construction. `q` is unaffected, and `sf = 1 - cdf` inherits the
`nan`.
