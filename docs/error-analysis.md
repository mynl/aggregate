# Floating-point error in FFT aggregates: a worked case

This note analyzes the floating-point (FP) noise floor of the FFT aggregate
algorithm, using a near-normal life-insurance book as the worked example. It
also draws out a modeling lesson: how you *parameterize* a signed aggregate
decides whether it buckets exactly or fights the grid.

## The setup

A block of `n = 1,000,000` one-year term policies. Each policy collects premium
`P` and pays face `F` on death (probability `q = 0.019`). Working in units where
`gcd(P, F)` is divided out gives `P = 1`, `F = 50`, so per-policy **profit** is

```
X = +1   with prob 1 - q   (survives, keeps premium)
X = -49  with prob q        (dies, pays 50, nets 1 - 50)
```

modeled directly as a signed severity:

```python
a = build('agg LIFE.INS dfreq[1000000] dsev[-49 1] [0.019 0.981]', bs=1, log2=17, padding=0)
```

Per-policy mean `0.05`, variance `46.60`; aggregate mean `50,000`, sd `≈ 6,826`.
The aggregate is exactly supported on the **spacing-50 sublattice**
(`profit = n - 50·deaths ≡ 0 (mod 50)`), so at `bs = 1` every non-multiple-of-50
bucket is mathematically zero. Yet those "zero" buckets carry noise at
**~1e-13**, not the usual **~1e-15**.

## Why the noise is 1e-13, not 1e-15

The FFT path computes `agg_ft = freq_pgf(n, sev_ft)` then inverts. For a fixed
count the frequency PGF is `freq_pgf(n, z) = z**n`
(`aggregate/_frequency.py`, `FrequencyFixed`), so the aggregate transform is
literally

```
agg_ft = sev_ft ** 1_000_000.
```

**1. Baseline FFT roundoff.** A radix-2 transform of length `M = 2^17` carries
relative error `~ u·log2(M)` per coefficient, with unit roundoff
`u ≈ 1.1e-16`. Here `17·u ≈ 1.9e-15` — the "usual 1e-15."

**2. The nth-power amplifier.** Write `sev_ft[k] = c_k`, computed as
`ĉ_k = c_k(1 + δ_k)` with `|δ_k| ~ 1.9e-15`. Then

```
ĉ_k**n = c_k**n · (1 + δ_k)**n ≈ c_k**n · (1 + n·δ_k),
```

so the **relative** error is multiplied by `n`. With `n = 1e6`,
`n·δ ≈ 1e6 · 1.9e-15 ≈ 1.9e-9`. This is the compounding effect: raising the
characteristic function to the millionth power amplifies each coefficient's
roundoff a millionfold.

**3. Which frequencies survive.** `|c_k|**n` is negligible except where
`|c_k| ≈ 1`. Because the severity lives on a spacing-50 lattice, `|c(θ)| = 1`
*exactly* at the 50 reciprocal-lattice frequencies `θ_m = 2π m / 50`. Around each,
`|c_k|**n = exp(-σ_agg²θ²/2)` is a bump of half-width
`M / (2π·σ_agg) ≈ 3` grid points. So `~50 × 3 ≈ 150` frequencies each carry the
`~1e-9` amplified error.

**4. Back to real space.** The inverse FFT's `1/M` normalization plus incoherent
(random-phase) summation over `N_sig ≈ 150` significant frequencies gives

```
noise(x) ≈ (1/M)·√(N_sig)·(n·u·log2 M)
        ≈ (1/131072)·12·1.9e-9  ≈  2e-13,
```

matching the observed ~1e-13. **The `n = 1e6` power is the amplifier; the inverse
FFT's `1/M` normalization is the brake** — which is why the floor lands at 1e-13
rather than the un-normalized 1e-9.

**Scaling (testable).** The floor grows approximately **linearly in `n`** (the
number of convolutions), weakly like `√(lattice count)`, and like `1/M`. So
`n = 1e5 → ~1e-14`, `n = 1e7 → ~1e-12`.

The off-lattice grass is pure FP roundoff — it does not affect `q`/`cdf`/`tvar`
or the moments (the spurious mass totals `~150·1e-13 ≈ 1e-11`). It is a
density-cosmetics and deep-tail-fidelity issue, not a correctness one.

## The bucketing dilemma

The severity atoms `-49` and `+1` are coprime to the aggregate's spacing-50
lattice, forcing a choice:

* **`bs = 1`** — severity exact, aggregate values exact (on `50ℤ`), but the grid
  is 50× oversampled: 49 of every 50 buckets are known-zero, and they fill with
  the amplified FP grass above.

* **`bs = 50`** (the aggregate's natural lattice) — no wasted cells, no grass,
  but now the *severity* atoms `-49, +1` do **not** sit on the `{…, -50, 0, 50, …}`
  grid, so mass-preserving (linear) bucketing spreads them:

  ```
  dsev[-49 1] [q, 1-q]  ->  dsev[-50 0 50] [ q·49/50,  (q + (1-q)·49)/50,  (1-q)/50 ]
  ```

  This preserves the mean exactly but inflates `E[X²]` by exactly
  `d₁·d₂ = 1·49 = 49` (the standard linear-split variance leak), so the
  severity sd goes `6.826 → 9.777` (+43%) and validation fails on sd while
  passing the mean. (At the default `bs = 2` the same spreading gives a milder
  ~1% sd error.)

### Why one grid can't win: phase accumulation

You cannot get both the severity `{-49, +1}` and its `n`-fold aggregate exact on
a single `bs = 50` grid, *at any origin*. Both atoms are `≡ 1 (mod 50)`, so a
`bs = 50` grid phased at origin `≡ 1` holds the severity exactly — but the
`n`-fold convolution of phase-1 points lands at phase `n·1 mod 50 = 0`, off that
grid. The deterministic per-policy offset (`+1`) **accumulates `n` times** under
convolution (`n·1 = 1,000,000`), and a single-origin grid cannot carry that
`N·s` shift. This is the same shift the windowed FFT-wrap machinery is careful to
avoid.

## The resolution: change of variables (P&L)

The clean fix is to pull the deterministic part out of the stochastic severity —
which is exactly the P&L decomposition. Model profit as *premium minus losses*,
with the loss severity on the benefit lattice `{0, F}`:

```python
b = build('pnl LIFE.INS.PNL 1000000 premium less agg L dfreq[1000000] dsev[0 50] [0.981 0.019]')
```

Now:

* the loss severity `{0, 50}` is phase-0 and buckets **exactly** at `bs = 50`;
* the `n`-fold loss aggregate stays on `50ℤ≥0` — phase-0 is closed under
  convolution, so no phase drift, no grass, no wasted cells;
* the total premium `n·P = 1,000,000` enters as an **affine shift** (which is
  where the accumulated `N·s` term correctly lives), exact because
  `1,000,000 ≡ 0 (mod 50)`.

The result reproduces the sd exactly: Margin sd `6826.236063` vs analytic
`6826.236152` (agreement to ~7 significant figures; the residual is FP, not
discretization), against the signed form's 1% error.

**The lesson.** `dsev[-49 1]` is a mis-parameterization: it smuggles a
deterministic premium into the stochastic severity, coupling a fine severity
lattice to a coarse aggregate lattice. The P&L change of variables separates the
deterministic affine shift (carrying the `n·offset`) from the phase-0 stochastic
lattice, and the numerics fall out exact. Model net underwriting results as
`pnl premium less agg losses`, not as a signed severity with premium folded in.

## Mitigations, in order of preference

1. **Reparameterize as P&L** (above) — exact, no wasted grid, no grass. The
   right answer whenever a signed aggregate is really *deterministic ± losses*.

2. **Bucket at the natural lattice.** When a clean change of variables is not
   available but the aggregate genuinely lives on a sublattice of spacing `d`,
   `bs = d` removes both the waste and the grass by construction (the amplified
   error then perturbs only real support cells, invisibly).

3. **Structural (positional) defuzz.** When a fine `bs` is forced, zero every
   bucket *not* on the known sublattice `d` (residue `r = N·a_ref mod d`). This
   removes the grass by *position*, never by magnitude, so — unlike a threshold
   defuzz — it can never clip genuine deep-tail mass. It applies only to the
   fully-discrete integer-lattice path (which `Aggregate._exact_discrete_window`
   already detects), needs `bs | d` and an aligned origin, and cleans only the
   off-lattice cells (the on-lattice cells keep their ~1e-13 amplified error,
   harmlessly). A niceness / tail-fidelity polish, not a correctness fix.

## References

The growth of FFT-convolution error with the number of convolutions, and the
aliasing of compound distributions, are treated by Grübel & Hermesmeier
(*ASTIN Bulletin*, 1999, 2000) and by Embrechts & Frei (*Journal of Computational
Finance*, on FFT versus Panjer recursion). Both are in the author's
`uber-library.bib` (`Grubel1999`, `Grubel2000`); the Embrechts–Frei key should be
confirmed before this note is promoted to a citable Quarto page.
