# Plan — calibrate (and price) distortions on signed distributions

> **Status: DRAFT — not executed.** Caller-side bookkeeping only; the per-kind
> `Distortion.calibrate`/`price` math stays pure and 0-based. No new numerics,
> no change to any existing non-negative path (must stay byte-for-byte).
>
> **Release mechanics (CLAUDE.md).** Behaviour change (new capability + tests)
> → version bump + `CHANGELOG.md` entry at close. `uv run pytest` green before
> any commit; the frozen baseline must not move.

---

## 0. Problem

`Distortion.calibrate` and `Distortion.price` integrate the **layer / Lee form**
of the distorted expectation,

```
E_g[X] = ∫₀^∞ g(S(x)) dx        # PHDistortion.calibrate: np.sum(S**rho) * bs
```

which is valid **only for `X ≥ 0`**. The correct two-sided form is

```
ρ_g[X] = ∫₀^∞ g(S(x)) dx − ∫_{−∞}^0 (1 − g(S(x))) dx
```

so on a grid with negative buckets the existing sum adds `g(S)` where it should
subtract `(1 − g(S))`. The caller `Portfolio.calibrate_distortion`
(`portfolio.py:3469-3494`) reinforces the assumption: it slices
`density_df.loc[0:assets]` from 0 and asserts `S > 0 and weakly decreasing`, so a
signed distribution is silently truncated rather than flagged.

Two ways a distribution arrives signed:
- **payoff convention** (`value_type='payoff'`, e.g. a `pnl` aggregate) — more is
  better, so the bad tail is on the *left*;
- **negative support** (signed severity / shifted law) — outcomes straddle 0.

## 1. Approach — the "easy route" (translation-equivariance + comonotone duality)

Distortion risk measures are translation-equivariant and comonotone-additive, and
a constant is comonotone with everything, so the corrections below are **exact**,
not approximations.

- **Shift (negative support).** `c = −min(support) ≥ 0`, `Z = X + c ≥ 0`. Then
  `ρ_g[Z] = ρ_g[X] + c`. Calibrate on `S_Z` against `premium_target + c`; the
  recovered shape is identical to the one for `X` (the Newton `f` sees only the
  *shape* of `S`, and `S_Z` is `S_X` slid onto a non-negative axis). Recover any
  price by subtracting `c`.
- **Reverse (payoff convention).** A distortion loads the *right* tail = the bad
  tail of a **loss**; a payoff's bad tail is on the *left*. So `X → −X` puts the
  bad tail back on the right before calibrating — exactly the **bid/ask duality**.
- **Signed payoff** = reverse-then-shift: `X → −X`, then add `c = max(X)` to clear
  the new negatives.

**Physical reverse, not analytic dual.** `price` already exposes the dual via
`kind='bid'`/`g_dual`, but routing payoff through the dual would need every kind
to grow a dual-calibrate. Physically reversing the grid is kind-agnostic and
uniform — preferred.

## 2. Where it lives

A thin pre/post **wrapper at the caller**, around the unchanged subclass
`calibrate`/`price`:

1. build `x`, `S` from the **full signed grid** (not the `0:assets` slice);
2. record the frame transform `(reverse?, c)` and apply it → non-negative,
   loss-oriented `x`, `S`;
3. call the subclass `calibrate` with `premium_target` mapped into the
   shifted/reversed frame;
4. **the caller owns the `(reverse?, c)` transform and inverts at the boundary** —
   the `Distortion` object stays frame-free (no offset baked onto it), so a
   calibrated distortion remains a pure `g`. Prices come back in the caller's
   original convention because the caller un-shifts/un-reverses the result, not
   the distortion. Feasible because the calling logic is already largely
   centralized.

Touch points: `Portfolio.calibrate_distortion` (`portfolio.py:3410`), the
`price`/`price_stand_alone` path (`portfolio.py:4204`, `4297`), and the analogous
`Aggregate.price` (`distributions.py:8024`). Gate the whole transform on
`_signed()` so the non-negative path is untouched.

## 3. Bookkeeping wrinkles (the actual work)

- **`assets`, `el`, `ess_sup` are not invariant under *reversal*.** Under the
  *shift* they all move by `+c` (and `ccoc`'s `r = (P−el)/(a−P)` is shift-invariant
  — verified). Under *reversal* the asset cap flips from "truncate the large-loss
  tail" to a floor; the wrapper must re-express `a` so it still truncates the bad
  tail. This is the fiddly bit, not the integral.
- **Mass-at-zero kinds (`ly`, `clin`, `lep`).** They key off `ess_sup` and an atom
  at 0; the "0" reference moves under the shift, so thread `ess_sup → ess_sup + c`.
- **bid/ask relabel.** After a reversal an `ask` on `−X` is a `bid` on `X`. The
  wrapper must relabel so the returned side matches what the user asked for.
- **`S > 0 / weakly decreasing` assert.** Still valid — but on the transformed
  (non-negative, reversed) grid, not the raw signed one.

## 4. Tests

- Round-trip: a known non-negative law priced directly == the same law shifted
  negative then calibrated/priced through the wrapper (shift-exactness).
- Payoff (`pnl`) aggregate: calibrate to a target, confirm bid/ask come back on the
  intended side and match the hand-computed dual.
- Regression: every existing non-negative calibrate/price case unchanged
  (baseline frozen).
- Per-kind sweep so the `assets`/`ess_sup` threading is covered for the
  mass-at-zero kinds and `ccoc`.

## 5. Out of scope

No new distortion kinds; no change to the layer-integral math; no signed support
pushed through the FFT/convolution (that stays the `pnl` affine-relabel story —
see `_signed_severity` vs `_signed`, `dev/done/plan-negative-x-agg.md`).
