# Plan — calibrate (and price) distortions on signed distributions

> **Status: LANDED (1.0.0a99).** Implemented in `_pricing.calibrate_distortions`
> + `_pricing._canonical_loss_frame`; per-kind `Distortion.<kind>.calibrate` math
> unchanged. Receipt convention (author-approved): un-shift only — `M`/`Q`/`coc`
> shift-invariant, `L`/`P`/`a` may go negative together when net-beneficial; the
> accounting identities hold with `M, Q >= 0`. `names=` added to both class
> delegators. Tests: `tests/test_signed_calibrate.py`; DecL `SC.*` in
> `decl-testers.agg`. Classic non-negative path byte-for-byte unchanged.
>
> **Status: DRAFT — not executed.** Caller-side bookkeeping only; the per-kind
> `Distortion.<kind>.calibrate` math stays pure and 0-based. No new numerics, no
> change to any existing non-negative path (must stay byte-for-byte).
>
> **Goal (acceptance):** after this plan, **every** distortion calculation —
> calibration *and* pricing — is correct for **every** support: non-negative
> loss, signed loss (straddles 0), and payoff (more-is-better), with or without
> negative outcomes.
>
> **Release mechanics (CLAUDE.md).** Behavior change (new capability + tests) →
> version bump + `CHANGELOG.md` entry at close. `uv run pytest` green before any
> commit; the frozen baseline must not move.

---

## 0. Findings — what is already sign-safe, and what is not

The library computes distorted expectations **two structurally different ways**,
and the survey (current `src/aggregate/` tree) shows the support-safety splits
cleanly along that line. This is the fact that scopes the whole plan.

### 0.1 Already sign-agnostic — **no change needed** (the "outcome × adjusted-probability" path)

Every routine that *applies* a distortion to *price* already weights outcomes by
exact distorted atom weights `gp = g(T) − g(S)` and forms `Σ xₖ·gpₖ`. Because the
*outcome* `x` carries its own sign and origin, these are correct on any monotone
grid, negative buckets included:

| Routine | File:line | Form |
|---|---|---|
| `choquet_weights` | `spectral.py:109` | core engine; `gp = gT − gS`, reconciliation `x[0] + Σ diff(x)·gS` **carries the origin** `x[0]` |
| `Distortion.price` | `spectral.py:1698` | `ask = dot(xa, gp)`, `bid = dot(xa, g_dual gp)` |
| `Aggregate.apply_distortion` | `_aggregate.py:3427` | `exag = cumsum(x·gp) + x·gS` |
| `Portfolio._build_augmented` / `apply_distortion` | `_portfolio_common.py:90`, `_portfolio.py:2452` | `exag_total = cumsum(loss·gp) + loss·gS`; per-unit kernel likewise |
| `Distortion.effective_g` | `spectral.py:772` | routes `g ↔ g_dual` by `view`(ask/bid) XOR `is_loss_value` |
| `GridDistribution.q / cdf / sf / mean / snap` | `_grid_distribution.py:220+` | `searchsorted` on `self._x`; index-agnostic |

**Conclusion: the pricing/application surface needs no shift/reverse.** It is
"free" exactly as expected for the outcome × adjusted-probability form. The plan
must not touch it.

### 0.2 NOT sign-safe — the **layer-integral calibration path** (this is the entire fix)

Calibration solves for a distortion *shape* by integrating the **layer / Lee
form** `∫₀^∞ g(S(x)) dx ≈ Σ g(Sₖ)·bs`, which is valid **only for `X ≥ 0`**. The
correct two-sided measure is

```
ρ_g[X] = ∫₀^∞ g(S(x)) dx − ∫_{−∞}^0 (1 − g(S(x))) dx
```

so on a grid with negative buckets the bare sum adds `g(S)` where it must subtract
`(1 − g(S))`. Every routine below assumes the one-sided form:

| Routine | File:line | Assumption |
|---|---|---|
| `_pricing.calibrate_distortions` | `_pricing.py:214` | **the entry point**: `a = q(p)` (large-x tail), `exa = E[min(X,a)]`, `density.loc[0:assets]` |
| `_calibration_survival` | `_pricing.py:154` | `S = 1 − density.loc[0:assets].cumsum()`; asserts `S > 0` weakly decreasing |
| `_limited_ev` | `_pricing.py:178` | `bs · Σ S[x < a]`, cumsum from 0 |
| `GridDistribution.lev` | `_grid_distribution.py:279` | 0-based, deliberately `add_exa`-parity |
| `Distortion.<kind>.calibrate` (PH, Wang, TVaR, Dual, CLL, CLin, LEP, LY) | `spectral.py` (each subclass) | Newton on `Σ g(Sₖ)·bs` (+ `ess_sup` mass term for `clin`/`lep`/`ly`) |

**Both `Aggregate.calibrate_distortions` (`_aggregate.py:4845`) and
`Portfolio.calibrate_distortions` (`_portfolio.py:2405`) delegate to the single
`_pricing.calibrate_distortions`.** That one function is the only place the
transform must be installed. *(The old `Portfolio.calibrate_distortion` singular
referenced by the previous draft no longer exists — the refactor consolidated
calibration into `_pricing`.)*

---

## 1. Approach — translation-equivariance + the duality already wired into pricing

Distortion risk measures are translation-equivariant and comonotone-additive, and
a constant is comonotone with everything, so the corrections are **exact**, not
approximations. Two transforms, addressing two **orthogonal** issues:

### 1.1 Negative support (loss-oriented) → **SHIFT** (this is the whole job here)

`c = max(0, −min(support)) ≥ 0`, `Z = X + c ≥ 0`. Then `ρ_g[Z] = ρ_g[X] + c`. The
point of the shift is precisely that the **one-sided** layer integral over the
non-negative `Z` *equals* the correct **two-sided** measure of `X`, plus `c`. So:
calibrate the unchanged subclass `calibrate` on `S_Z` against `premium_target + c`;
the recovered shape is identical to the one for `X` (the Newton `f` sees only the
*shape* of `S`, slid onto a non-negative axis). Subtract `c` to read any amount
back. For a loss with `X ≥ 0`, `c = 0` and the path is byte-for-byte unchanged.

### 1.2 Payoff orientation (more-is-better) → **REVERSE**, and it is **forced**, not a convenience

A concave `g` loads the small-probability (large-`x`) tail — the **bad** tail of a
*loss*. A payoff's bad tail is on the **left**, so the same family loads the wrong
end. The fix is `X → −X` (reverse) so the bad tail is on the right again, then
shift to clear the new negatives.

**Why reversal is required (not optional), and how it stays consistent with
pricing.** Pricing already flips orientation: `effective_g(view, is_loss_value)`
returns `g_dual` for a payoff ask (`spectral.py:772`). So the **applied** price of
a payoff is `ρ_{g_dual}(X)`. For calibration to land the shape that pricing will
actually use, it must solve

```
ρ_{g_dual}(X) = premium_target.
```

By duality `ρ_{g_dual}(X) = −ρ_g(−X) = −ρ_g(Y)` with `Y = −X`. So calibrating
the **primal** `g` on the reversed-and-shifted loss `Z = Y + c` to the mapped
target reproduces exactly the shape whose **dual** prices the payoff to target.
**The reversal at calibration and the `g_dual` flip at pricing are the matched
pair** — they compose to the identity on *price*, not a double-flip. State this
invariant in the code so nobody later "fixes" one half:

> **Invariant.** Calibration emits a shape in the canonical **non-negative
> loss frame**; pricing's `effective_g` restores the caller's orientation. The
> two orientation flips cancel on price. The stored `Distortion` is frame-free
> (a pure `g`, no offset, no reversal baked on).

**Physical reverse, not a per-kind dual-calibrate.** Routing payoff through the
dual *inside* calibration would force every kind to grow a second integrand
(`Σ g_dualθ(S)·bs`); physically reversing the survival keeps every subclass
`calibrate` untouched and is kind-agnostic. Preferred.

**This 1.2 reversal is the one place "good vs bad" enters.** It is *orthogonal*
to bid/ask: bid/ask is the `g`-vs-`g_dual` choice `effective_g` already owns at
pricing time, on top of whatever frame calibration produced. Do not entangle them.

---

## 2. Where it lives — inside `_pricing.calibrate_distortions`

Per the user's intent ("ask `calibrate_distortions` and it works regardless of
sign and support"), the transform is installed **in the one entry point**, as a
pre/post wrapper around the unchanged subclass `calibrate`:

1. **Read the orientation flag** `obj._is_loss_value` and the **full signed
   grid** `(x, p_total)` from `obj.density_df` — *not* the `0:assets` slice.
2. **Choose the frame transform** `T = (reverse?, c)`:
   - `reverse = not obj._is_loss_value` (payoff);
   - after the optional reverse, `c = max(0, −min(support with mass))`.
   For the default non-negative loss, `reverse=False, c=0` → existing path.
3. **Map the capital anchor and targets into the transformed frame** (§3), build
   the 0-based `S_Z`, recompute `exa` and `ess_sup` **on the transformed grid**,
   invert the cost-of-capital → `P` exactly as today but in-frame.
4. **Call `Distortion.calibrate_set` unchanged** with the in-frame `S_Z`,
   `premium_target`, `assets`, `el`, `ess_sup`.
5. **Invert `T` at the boundary**: the returned shapes are frame-free; the
   `distortion_df` / `calibration_df` receipt reports `a`, `p`, `exa`, `P`, `Q`
   in the **caller's original convention** (un-shift, un-reverse). The
   `Distortion` objects are stored with no offset on them.

Gate the entire transform on `reverse or c > 0` (equivalently `obj._signed()` plus
the payoff flag) so the non-negative loss path is provably untouched.

**Touch list (current files):**
- `_pricing.calibrate_distortions` (`_pricing.py:214`) — install the wrapper.
- `_pricing._calibration_survival` (`_pricing.py:154`) — accept a transformed,
  full-grid survival; keep its `S > 0` / weakly-decreasing assert but **on the
  transformed (non-negative) grid**.
- `_pricing._limited_ev` (`_pricing.py:178`) — compute `exa` in-frame; do **not**
  read the stored `exa_total` shortcut for a signed/payoff object (that column is
  the loss-frame LEV of the *original* grid and is wrong post-transform).
- `Distortion.<kind>.calibrate` — **unchanged.**
- Pricing/application surface (§0.1) — **unchanged** (already sign-safe).

---

## 3. Bookkeeping wrinkles (the actual work)

The integral is exact; the index/target mapping is the labor.

- **`assets`, `el`, `ess_sup` are invariant under *shift* but not under
  *reversal*.**
  - *Shift:* all move by `+c`; the ccoc rate `r = (P − el)/(a − P)` is
    shift-invariant (`((P+c)−(el+c))/((a+c)−(P+c)) = (P−el)/(a−P)`), so the
    cost-of-capital inversion `P = ν·exa + δ·a` needs only consistent `+c` on
    `exa` and `a`. Verified.
  - *Reversal:* the asset cap flips meaning — for a loss `a` truncates the
    **large-loss** (right) tail; reversed, the bad tail is the other end. Re-express
    the quantile so it still caps the bad tail: under `Y = −X`,
    `q_Y(p) = −q_X(1 − p)`. So a payoff anchored at probability `p` maps to
    `a_Y = −q_X(1 − p)`, then `a_Z = a_Y + c`. This is the fiddly step, not the
    integral.
- **Recompute `exa`, do not trust `exa_total`.** `calibrate_distortions` currently
  shortcuts `exa = density_df.loc[a, 'exa_total']` when present (`_pricing.py:240`).
  That column is the original-frame LEV; for signed/payoff recompute `exa` from the
  transformed `S_Z` via the in-frame `_limited_ev`.
- **Mass-at-zero kinds (`ly`, `clin`, `lep`).** Their `calibrate` adds an
  `ess_sup`-weighted atom at the origin; the "0" reference moves under the shift, so
  thread `ess_sup → ess_sup + c` (and reversed first if payoff).
- **bid/ask relabel is *pricing*, not calibration.** Because calibration now always
  emits a loss-frame shape and `effective_g` restores orientation at pricing
  (§1.2), there is **no** bid/ask relabel to do in the calibration wrapper. The
  only obligation is the invariant: the stored shape must be the loss-frame one.
  (Confirm by a round-trip test, §4, rather than by relabeling here.)
- **`S > 0 / weakly decreasing` assert.** Still valid — on the transformed
  (non-negative, possibly reversed) grid, not the raw signed one.

---

## 4. Tests

- **Shift-exactness round-trip (loss).** A non-negative law calibrated directly ==
  the same law shifted negative (`ssev` / explicit `x_min < 0`) then calibrated
  through the wrapper: identical shapes; prices identical after un-shift.
- **Payoff round-trip (the orthogonality guard).** A `pnl` / payoff aggregate
  calibrated to a target: confirm the stored shape is the loss-frame shape, and
  that **pricing it back** through `apply_distortion(view='ask')` (which applies
  `g_dual`) reproduces `premium_target`. This is the test that fails if reversal
  and `effective_g` ever double-flip or fail to flip.
- **Both anchors.** Anchor on `p` and on `a` for a signed/payoff object; the
  `calibration_df` reports `a, p, exa, P, Q` in the **caller's** convention.
- **Per-kind sweep.** Every family (incl. mass-at-zero `ly`/`clin`/`lep` and
  closed-form `ccoc`) calibrates on a signed grid; `ess_sup`/`a` threading covered.
- **Regression (frozen baseline).** Every existing non-negative calibrate/price
  case byte-for-byte unchanged; `reverse=False, c=0` provably no-ops.

---

## 5. Out of scope

- **No change to the pricing/application surface** (§0.1) — it is already
  sign-safe; touching it risks the frozen baseline for zero benefit.
- **No change to the layer-integral subclass math** — the shift makes the
  one-sided form compute the two-sided measure; the kinds stay pure and 0-based.
- **bid/ask (`g` vs `g_dual`) is a separate, already-solved axis** (`effective_g`,
  `spectral.py:772`); this plan only guarantees calibration produces a shape
  consistent with it.
- **No signed support pushed through the FFT/convolution** — that is the `pnl`
  affine-relabel story already landed in `dev/done/plan-negative-x-agg.md`
  (`_signed_severity` vs `_signed`, `_apply_agg_affine`).
