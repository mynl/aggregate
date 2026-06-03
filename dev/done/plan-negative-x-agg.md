# Plan: negative-support severity & output window — **Aggregate** scope

> **Status:** ✅ IMPLEMENTED (1.0.0a21, 2026-06-02/03). All five stages landed;
> 799 tests pass (28 in `tests/test_negative_x.py`); default path byte-for-byte
> unchanged.
>
> **Rev 6 (2026-06-03) refinements (author):**
> - **`signed=` removed from `update`/`update_work`.** Signedness is purely the
>   severity declaration (`ssev` / negative `dsev` → `Severity.signed`); a runtime
>   override is incoherent (a built `Severity` is already clamped or not). The
>   `ppf` auto-detect probe is gone; `_signed()` = `any(s.signed)`.
> - **Unified bucket+window estimator + `_bs_window_df`.** `update` runs up to
>   three sizing methods and records them in the inspectable `self._bs_window_df`,
>   then selects: **exact_discrete** (`dfreq`/`fixed` × `dsev` on an integer
>   lattice → exact finite support, `bs=1`, minimal `log2`) > **bounded_small**
>   (bounded severity → `[0, N_hi·s_max]` from a high freq quantile; chosen only
>   when tighter than moment, so LLN concentration hands large counts back to
>   moment) > **moment** (legacy 3-moment; reproduces `recommend_bucket` exactly
>   for non-negative, two-sided `estimate_agg_window` for signed). `log2` is a
>   **cap** (input or 16): exact may use fewer buckets but never more — *unless*
>   the user pins `bs`, which means "I control the grid" and the given `log2` is
>   honoured (this preserves pinned baselines). Origin: `x_min=0` for a
>   non-negative aggregate (legacy), `floor(x_lo)` (tracks the mass, ± far from 0)
>   when signed. Explicit `x_min`/`bs` honoured (a `used` row in the df).
> - **`sev_density_df`.** Severity columns (`p_sev`/`F_sev`/`S_sev`/`log_p_sev`)
>   moved out of `density_df` onto the severity's own grid `xs_sev` — because a
>   windowed/signed aggregate and its severity no longer share a grid. `plot`,
>   `q_sev`, `tvar_sev`, `cramer_lundberg`, `severity_error_analysis` re-sourced;
>   baseline corpus updated. `density_df.p_sev` is gone.
> - **`info`** renders discrete/signed severities by their support
>   (`atoms [-2 5]`, shortened `N atoms [..]` for many) instead of the bogus
>   `5 xs 0` layer form, and warns when the severity support is **outside** the
>   output window (read `sev_density_df`). Open: how to *plot* an off-window
>   severity (TODO #15).
>
> Key decisions taken during initial execution:
> - **Plan §4 was wrong that signed severity is "free."** The Severity layering
>   deliberately clamps `x<0→0` (e.g. `norm` piles its sub-zero tail at 0) and
>   `validate_discrete_distribution` clamped negative `dsev` atoms to 0. Both
>   fixed: `allow_negative` flag (dfreq still clamps; dsev preserves),
>   `SeverityDHistogram` negative-atom placement, and a raw-`fz` (un-clamped)
>   read in `discretize` when signed.
> - **Opt-in is a DecL keyword (author decision, rev 5):** signedness is a
>   *parse-time property of the severity*, which dissolves the moments/clamping
>   chicken-and-egg (analytic moments are correct before any FFT). The severity
>   family is now `sev` (continuous, clamps at 0 — unchanged), `dsev` (discrete,
>   never clamps — auto-signs on a negative atom), and **`ssev`** (continuous,
>   never clamps — the signed/P&L variant). `Severity.signed` records it; the
>   aggregate derives `_signed_sev = any(component.signed)`. The `signed=`
>   argument on `update` survives only as an override; the auto-detect `ppf`
>   probe is gone. `ssev` is **orthogonal to `value_type`** — it does *not*
>   imply `payoff`.
> - **Window default falls out:** `update(x_min='auto')` (the default) resolves
>   to `None` (auto two-sided window) for a signed aggregate and `0` otherwise,
>   so a P&L declared in DecL just works from `build()` with no extra argument.
> - **`estimate_agg_window(m, sd, skew, p)`** takes the analytic **sd** (not
>   cv) so the mean-zero case (where cv blows up) works. Lives in
>   `distributions.py` (not `utilities.py` as §11.3 said) because it reuses the
>   MoM fits there; re-homing + unifying with `bivariate.size_axis` is a
>   follow-up.
> - **`plot` is signed-aware:** both the discrete and continuous branches use a
>   two-sided loss range so the x-axis covers negative support (previously
>   `f(q(0.999))` reversed/clipped when the support was negative).
> - **Deferred (non-blocking):** two-sided deficit split (`deficit_lo/hi`); the
>   `ft.py` recentering helpers calling the core path + the equivalence test;
>   occurrence reinsurance on a *signed severity* grid; a DecL keyword for
>   `value_type` (the member exists; only the keyword is deferred).
>
> Split out of the former combined `plan-negative-x.md`; the Portfolio half is now
> [`plan-negative-x-port.md`](plan-negative-x-port.md) (a draft to be **refreshed
> after this Aggregate work lands**). Sibling: [`plan-multivariate.md`](plan-multivariate.md)
> — **build negative-x (this) first**; multivariate reuses the signed-axis /
> window machinery.
>
> Scope of *this* document: everything at the **`Aggregate`** level — negative
> severity, the output window, window estimation, signed reporting. Portfolio
> combine and the `density_df`/pricing-column audit are deliberately deferred to
> the sibling plan so we can implement, ship, and learn from the Aggregate work
> before designing Portfolio.
>
> Two intertwined features, designed together: (i) **negative x** and (ii) **the
> output window**, both roll/offset bookkeeping on the FFT lattice. **0 is always
> in the severity**; there is **no "subtract k" term** (we compute `Σ Xᵢ` as
> usual). The output window is the `recentering_convolution` method already in
> `ft.py` / ch. 5 of the help.

---

## 1. Motivation

Every aggregate today lives on a non-negative loss grid. The natural next object
is a **profit/loss (P&L) distribution**: premium − loss, ceded commission −
ceded loss, a reinsurance contract's economic result, a hedged book. These take
negative values (a profit is a "negative loss"). We want `build('agg …')` to
produce a P&L aggregate at the usual speed/accuracy, including the motivating
case

```
agg PnL 1e6 claims dsev [-1 10] [15/16 1/16] poisson
```

(DecL note: the count goes before `claims` and the frequency type is last;
`dsev … poisson 1e6` is not valid grammar.) A binary per-risk P&L summed over
Poisson(10⁶) — a thin, near-Gaussian lump at
≈ (−15 + 10) / 16 × 10⁶ (a *negative* mean). This is **TODO #3 / #4** (negative
`xs`; integrated aliasing + movable window) — the most-wished-for feature.
(Portfolio-combine, TODO #6, is the sibling plan.)

The `dsev` form above auto-signs (a negative atom has no other reading). The
continuous counterpart uses the new **`ssev`** keyword — the signed,
never-clamp sibling of `sev`:

```
agg PnL 100 claims ssev 50 * norm + 10 poisson      # a normal-ish P&L per risk
```

`sev` keeps clamping its sub-zero tail at 0 (unchanged); `ssev` keeps the
negative support. The keyword is the only DecL addition the feature needs.

---

## 2. Two features, designed together

| Feature | Side | What it does |
|---|---|---|
| **F1 — negative-support severity** | input | severity may be < 0; **0 is in the severity grid** (starts at, ends at, or brackets 0). Negatives sit at the wrapped (high) end of the FFT buffer via an input offset `i0_sev`. **No shift term** — `Σ Xᵢ` exactly as today. |
| **F2 — the output window** | output | place the aggregate on a window `[x_min_out, x_min_out + N·bs)` *where the mass actually is*, so a tight far-from-0 lump (Po(10⁶) example) uses a small `bs` over a narrow window instead of paying for `[0, mean]`. |

They are separate capabilities but **tightly intertwined**: both reduce to
"array index 0 ↔ *this* physical value", both are a single `np.roll`, and they
share the window-estimation and reporting plumbing. We build them in one cycle.

The severity grid **need not be symmetric about 0** — there may be more mass /
range to the left or to the right; that only changes *how much* roll happens.
(We use `np.roll`, **not** `np.fft.fftshift`, precisely because `fftshift`
hard-codes a centred `N/2` shift while `roll` takes an arbitrary integer offset.)

### Why F2 is *not* the failed "shift" approach
A constant per-claim shift `s` fails for random `N` (the aggregate shifts by the
random, correlated `N·s`). **F2 does not do this.** The FFT computes the exact
compound distribution **modulo the period** `N·bs`; F2 merely **relabels** the
already-correct output by rolling it onto the right window. Relabeling a
finished, exact array has no `N·s` term — so F2 works for *any* frequency,
random or fixed. This is the crucial clarification.

---

## 3. The method we base on: `recentering_convolution` (review, don't copy)

`ft.py:260` (`recentering_convolution`, with `recentering_convolution_example`
at `:166`) and ch. 5 of the help ("Numerical methods / FFT") already implement
F2 — **as a toy / pedagogy helper, not production code.** We take the *method*
and write a **fresh, industrial-strength, reviewed** implementation for the core
path; we do **not** lift the code as-is. (Nothing is wrong with it as an
illustration; it just isn't designed/executed to core-path standards.) The
method:

1. Compute the aggregate on a grid long enough to hold the **supported width** of
   the distribution (where density ≳ 1e-15).
2. Roll **left** by the analytically-known mean `E[A] = E[N]·E[X]` (in buckets,
   `round(E[A]/bs)`, mod N).
3. Roll by the chosen window offset (the toy uses a centred `N/2`; the core uses a
   **general** offset via `np.roll`, since the window need not be mean-centred —
   see §5.2).
4. Re-index the x-axis to the window's physical values.

The core `update`/`update_work` path absorbs this (generalised to an arbitrary
window start); the `ft.py` versions remain documented illustrations (and, per
§11.5, are refactored to *call* the core path so there is one source of truth).

**Correctness condition (no aliasing):** recentering is exact **iff** the
effective support width `W < N·bs` (else distinct parts of the support collide on
the lattice). The two-sided quantile estimate (§5.2) returns both the window and
`W`, so we size the grid to guarantee `W < N·bs` and **warn** if a user-forced
grid violates it.

---

## 4. Current state (grounding, file:line)

| Concern | Where | Today | Change |
|---|---|---|---|
| Grid | `distributions.py:3493` `update`, `:3550` `update_work` | `xs = bs·arange(N)`, starts at 0 | grid `= x_min_out + bs·arange(N)`; add `x_min_out`, `i0_sev` |
| Severity discretization | `discretize` `:3385`, `:3445` `adj_xs[0]=−inf` | folds all sub-`bs/2` mass into bucket 0 (correct for positive RV) | signed mode: **finite** left endpoint `xs[0]−bs/2` + normalize, symmetric with the right end (`:3429`) |
| FFT core | `_fft_aggregate` `:3745`, `_freq_sev_convolution` `:3808` | `ft`/`ift`, 0-based buffer | roll in by `i0_sev`, roll out by window start; zero-risk/fixed-1 shortcuts unchanged |
| Recentering | `ft.py:166/260` (pedagogy) | standalone illustration | re-implement in core (generalised window start) |
| Bucket / window | `recommend_bucket` `:4742`, `_estimate_agg_percentile` `:273` | **upper** percentile only; `skew≤0`→normal | two-sided → `(bs, x_lo, x_hi, W)`; reflected `sln`/`sgamma` for negative skew |
| MoM fits | `sln_fit` `:91`, `sgamma_fit` `:114` | positive-skew (right tail) | reflected path: fit `−A` for the left tail; shared symmetric/normal fallback |
| Reins scatter | `_rebucket_to_grid` `:4083` (+ `scatter_bivariate` in `bivariate.py`) | index `= v/bs` (assumes `xs[0]=0`) | index `= (v − x_min_out)/bs` |
| Deficit | `:3650` `1 − Σp` + warning `:3651` | mass off the **right** | split `deficit_lo` / `deficit_hi`; name the side |
| `density_df` index | from `self.xs` | starts at 0 | starts at `x_min_out` (may be < 0); `cumsum`/`F`/`S`/VaR/TVaR stay valid left-to-right |

**Already negative-aware (free):** `discretize` comments *"allow severity to have
real support … from −inf"* (`:3444`); `_estimate_agg_percentile` already branches
on `skew≤0` (`:303`); `Severity` wraps negative-capable scipy RVs (`norm`, any
`sev … + shift`). The work is grid/window/offset plumbing and reporting.

---

## 5. Math

### 5.1 Offsets and rolls (input + output)
Input severity on `xs_sev[j] = (j − i0_sev)·bs`, `i0_sev` = index of physical 0.
Lay into FFT order and convolve:

```
g     = roll(sev_signed, −i0_sev)        # severity 0 at FFT index 0; negatives wrap up
z     = FFT(g, padding)                   # padding as today
ftagg = freq_pgf(n, z)                    # unchanged — PGF is elementwise in z
a     = real(iFFT(ftagg))                 # a[k] = mass at physical (k·bs mod N·bs)
```

Output window: choose `x_min_out` (snapped to a multiple of `bs`); displayed grid
`xs[i] = x_min_out + i·bs`, and

```
agg = roll(a, −round(x_min_out / bs))
```

- Full non-negative default: `x_min_out = 0`, `i0_sev = 0` → **no rolls, identical
  to today**.
- Negative-bracketing window: `x_lo < 0 < x_hi`.
- Tight far-from-0 window: `0 < x_lo` (mean-centred `x_min_out = E[A] − N/2·bs` is
  one option, reproducing `recentering_convolution`, but is not the default).

Both offsets require integer bucket positions ⇒ snap `x_min_out` / the severity
zero to multiples of `bs` and log if moved.

### 5.2 Two-sided window estimation
We estimate `x_lo` by the **same process** used for the upper bound in
`_estimate_agg_percentile`, only taking a **low-`p` quantile** of the estimated
aggregate. The window is **not** forced to be mean-centred — for a strongly
skewed aggregate the mass sits off-centre and we let the two quantiles place it.
Generalise `_estimate_agg_percentile` to return the **window and width** from the
shifted-lognormal / shifted-gamma MoM fits:

- **Symmetric case first.** Whenever `skew ≈ 0` (within a tight tolerance) — which
  includes every genuinely symmetric P&L — neither `sln` nor `sgamma` is defined;
  use a **normal** (or other symmetric) approximation for *both* edges. This is
  the default fallback, not an afterthought.
- **Upper edge** `x_hi`: as today (`p_hi = 1−10⁻ᵖ`; positive skew → `sln`/`sgamma`).
- **Lower edge** `x_lo`: low quantile of `A`. For `skew(A) < 0` (heavy left tail)
  **reflect**: fit `−A` (mean `−m`, same `cv` on `|m|`, `skew = −skew(A) > 0`),
  take its upper `p_hi` quantile `q`, set `x_lo = −q`.
- **Width** `W = x_hi − x_lo`; **window** `x_min_out = floor_bucket(x_lo)`,
  `x_max = ceil_bucket(x_hi)`.
- **bs from N (the usual direction).** The normal workflow is *input N, determine
  bs*: given `N = 2**log2`, set `bs = round_bucket(W / N)` (so `N·bs ≥ W`, i.e. the
  alias-free condition holds by construction). If the caller instead fixes `bs`,
  we check `N·bs > W` and warn otherwise. This mirrors `size_axis`
  (`bivariate.py:41`), which we **unify** into one helper (see §8).

New helper `estimate_agg_window(m, cv, skew, p)` → `(x_lo, x_hi, W)` beside
`_estimate_agg_percentile`; `sln_fit`/`sgamma_fit` gain a `reflect` path and a
shared symmetric/normal fallback.

### 5.3 Signed-mode discretization
With negative support, the leftmost bucket must **not** absorb the whole lower
tail (the asymmetric `adj_xs[0]=−inf`). Symmetric with the right end:

```
adj_xs = hstack([xs[0] − bs/2, xs + bs/2])    # signed 'discrete'; no −inf
```

The **left bucket is treated exactly like the right bucket** — no extra catch
mass — and we then **optionally normalise to sum 1** (same `normalize` flag and
semantics as today). Gate on signed mode so the non-negative path is byte-for-byte
unchanged.

### 5.4 Two-sided deficit
`deficit = 1 − Σp` can now leak off either end → report `deficit_lo` / `deficit_hi`
and have `DefectiveDistributionWarning` (`:3651`) name the side.

### 5.5 Sign convention — a tracked member variable
The aggregate **does** carry a sign-convention member (the earlier "sign-agnostic"
position is superseded): a value in `{loss, payoff}`, **default `loss`**
(provisional member name `value_type`; confirm at implementation). It records how
the variable should be read — actuarial **loss** ("more is worse") vs. **payoff**
/ asset ("more is better").

- It is **set at construction** (default `loss`), surfaced in `info`, and carried
  on the object. A DecL keyword to set it can follow later (out of scope here).
- It is **inert for the distribution itself** — density, moments, quantiles,
  deficit, plotting do **not** depend on it; F1/F2 are unaffected.
- It **comes into play when applying distortions / pricing**: distortions assume
  the actuarial loss orientation, so a `payoff` object is negated (or the
  distortion's dual is applied) at the pricing layer. The pricing/distortion
  consumption of `value_type` is specified in the sibling Portfolio plan and the
  downstream pricing work, not implemented here.

So: track it now (cheap, one member + `info` line), apply it later.

---

## 6. API surface (Aggregate only)

`Aggregate`:
- `update(..., x_min=0, x_max=None)` / `update_work(..., x_min=0, x_max=None)`.
  - `x_min=0, x_max=None` → **today's behaviour exactly**.
  - `x_min=None` → auto two-sided window (`estimate_agg_window`) = `[floor(x_lo),
    ceil(x_hi)]`; may be negative and/or far from 0 (off-centre for skewed
    aggregates — **not** forced to mean-centre).
  - explicit `x_min`/`x_max` → snap to bucket, use, and check `W < N·bs` (warn).
- New members: `x_min`, `x_max`, `i0` (severity zero index), `window =(x_min,
  x_max)`, and the `value_type` sign convention (§5.5, default `loss`). `self.xs`
  is the signed grid; `bs`/`log2` unchanged.
- `density_df` index = signed `xs`. `q`/`tvar`/`var`/`sf`/`cdf`/`plot` consume
  `xs`/`F`/`S` and stay valid on signed support.
- `info`/`describe`: show `window`, `W`, split deficit, `value_type`, and whether
  the output was recentred.

**Out of scope for this plan (→ `plan-negative-x-port.md`):** the Portfolio
combine with offsets; the **audit of every `Portfolio.density_df` column** on
signed support (esp. the **price** column / `add_exa`, `portfolio.py:2122`); and
the distortion/pricing *consumption* of `value_type`. Pricing/allocation
*semantics* on signed support (where the sign flips the risk reading, and
`lev`/`exa`/allocation may need re-derivation) is a downstream research item the
Portfolio plan opens.

---

## 7. Staged implementation (Aggregate)

1. **Audit spike (first action).** Grep the blast radius of non-negativity
   assumptions: `xs[0]==0`, `loss>=0`, cumsum-from-0, log-scale plots, `snap`,
   `_rebucket_to_grid`, `lev`/`exa`. ~15 min; confirms steps 2–5 scope, no
   design decisions.
2. **Offset core (F1+F2 plumbing).** `x_min_out`/`i0_sev` in `update_work`; roll
   in/out of `_fft_aggregate`; signed `discretize`; signed `_rebucket_to_grid`
   index; add `value_type` member. Default `x_min=0` keeps the whole suite green.
   *Gate: full suite unchanged.*
3. **Re-implement recentering, production-grade.** A fresh, reviewed core
   output-window roll based on the `ft.py:recentering_convolution` *method* (not
   its code; §3), arbitrary window start; fixed-N exact; random-N exact when
   `W < N·bs`. Refactor the `ft.py` helpers to call it.
4. **Window estimation.** `estimate_agg_window`, reflected `sln`/`sgamma`,
   symmetric/normal fallback, `x_min=None` auto path, `W < N·bs` guard +
   two-sided deficit/warning.
5. **Reporting.** signed `density_df`; `info`/`describe` window + deficit +
   `value_type`; `plot` on signed axis; P&L worked examples (incl. Po(10⁶)).

After this lands we **refresh `plan-negative-x-port.md`** from what we learned.

---

## 8. Interaction with `plan-multivariate.md`

- **Shared window sizer.** `bivariate.py:size_axis` ≡ `estimate_agg_window` +
  per-axis sizing. **Unify** into one helper in **`utilities.py`** (gathered with
  the FFT routines `ft`/`ift` and `round_bucket`), used by 1D aggregates and each
  multivariate axis.
- **Signed axes for free.** A multivariate *line* may be a P&L (negative
  support); the per-axis offset `i0_k` is exactly F1. Negative-x first ⇒
  multivariate inherits signed axes.
- **Reflected MoM fits** shared by both window estimators.

This is the concrete reason to do **negative-x first**.

---

## 9. Testing

`tests/test_negative_x.py`:
- **Identity:** `x_min=0, x_max=None` reproduces current `agg_density`/moments
  bit-for-bit across a basket of programs (roll/offset regression guard).
- **Symmetric:** `dsev [-1 1] [.5 .5] poisson λ` → mean 0, `est_skew ≈ 0`,
  `q(0.5) ≈ 0`.
- **Binary P&L lump (negative mean):** `dsev [-1 10] [15/16 1/16] poisson 1e6` →
  per-claim `E[X] = (−15 + 10)/16 = −5/16`, so `E[A] = −5/16 × 10⁶ ≈ −3.125×10⁵`;
  `Var(A) = λ·E[X²] = 10⁶·(15·1 + 1·100)/16 ≈ 7.19×10⁶`, `sd ≈ 2681`; a thin,
  near-Gaussian lump at a **negative** mean. Auto window is a tight band around
  `−3.125×10⁵` (off-centre / negative), not `[0, …]`; deficit ≈ 0.
- **Recentering equivalence:** core output-window path matches a known-good
  reference on the `ft.py:recentering_convolution` example (`xs=[3,4,7,34]`,
  `ps=[1/8,1/8,1/8,5/8]`, en=5000, log2=15).
- **`W ≥ N·bs` guard:** forcing too small a grid warns and the deficit/aliasing
  is detected.
- **Fixed-N exactness:** `dsev [-2 5] [.5 .5] fixed 3` vs the closed-form 4-point
  convolution.
- **Window estimation / reflected fits:** negative-skew agg recovers `x_lo < 0`
  within a bucket; reflected `sln`/`sgamma` matches the un-reflected fit on `−A`;
  symmetric agg uses the normal fallback.
- **`value_type`:** default `loss`; member set/read; `info` shows it (no effect on
  the distribution).
- Append DecL programs to `src/aggregate/agg/test_decl.agg` under a new section.

`uv run pytest` (`$env:UV_LINK_MODE='copy'`); no docs build in-loop.

---

## 10. Risks / watch

- **Off-by-one in `i0_sev` / window roll** — the classic FFT-offset bug; pinned by
  the identity test, the recentering-equivalence test, and the fixed-N closed
  form.
- **Aliasing (`W ≥ N·bs`)** — caught by the two-sided deficit + the `W < N·bs`
  guard.
- **Negative-skew fit robustness** — reflected fits can misplace the shift at
  extreme skew; fall back to normal as today.
- **Non-negativity assumptions downstream** — the step-1 audit spike; pricing /
  Portfolio columns deferred to the sibling plan.
- **`density_df` consumers** indexing by absolute position vs label — find & fix.

---

## 11. Resolved decisions (author)

1. **Window default (`x_min=None`):** low-`p` quantile for `x_lo` analogous to the
   upper bound; window `= [floor(x_lo), ceil(x_hi)]`, **not** mean-centred. (§5.2)
2. **Signed-mode left bucket:** treated **identically to the right bucket** — no
   extra catch mass — then **optionally normalise to sum 1**. (§5.3)
3. **Home for the shared window helper:** `utilities.py`, with the FFT routines.
   (§8)
4. **Portfolio + pricing/`density_df` audit:** **split out** to
   `plan-negative-x-port.md`, to be designed/refreshed **after** this Aggregate
   work lands. (§6)
5. **`ft.py` recentering helpers:** base the core on their *method* but write
   **industrial-strength** code (the existing code is a toy — reviewed/rewritten,
   not copied); then refactor the `ft.py` helpers to *call* the core path. (§3)
6. **Version:** next `1.0.0a*` (`a21` or whatever is next at scheduling).
7. **Sign convention:** **tracked member** `value_type ∈ {loss, payoff}`, default
   `loss`; inert for the distribution, **consumed when applying distortions /
   pricing** (actuarial loss orientation enforced there). (§5.5)
