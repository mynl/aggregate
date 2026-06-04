# Mean-preserving (`linear`) bucketing for discrete-histogram severities

**Status:** DONE (2026-06-04), shipped in 1.0.0a28. See README a28 section and
`tests/test_dsev_bucket.py`. Implementation notes vs this plan:
- `_rebucket_to_grid` gained **two** new params, not one: `scheme=None` (as
  planned) **and** `origin=None`. The `origin` fix was essential and is the one
  thing the plan's revision note got wrong: it claimed `x_min == xs_sev[0]`
  always holds for signed `dsev`, so the existing `self.x_min` scatter origin
  would work. That is only true on the *auto* window; a **forced** `x_min` below
  the smallest atom (e.g. `x_min=-8` with atoms `{-2,5}`) shifts every atom. The
  bed is indexed on `xs_sev`, so the scatter must use `origin=xs_sev[0]`.
- Blast radius (revision note point 7): both discrete corpus cases moved, not
  zero — `Sym.Dice` **and** `Port.Bodoff` (its `{0,99,100}` atoms are on-grid,
  but linear's exact mass placement beats cdf-diff+normalize by ~1 ULP). Both
  re-captured; moves are at the FP floor (`<= 1.5e-11` on moment-derived stats,
  `<= 6e-14` on densities), within `rtol=1e-12`/`atol=1e-14` except the one
  `Sym.Dice` skew row that nudged just over `atol`.


**Date:** 2026-06-03; revised 2026-06-04 after the `_DiscreteRV` cleanup landed
(1.0.0a26) and per author direction below.
**Related:** `dev/done/plan-discrete-severity-fz.md` (the `_DiscreteRV` cleanup —
now shipped, so the atom masses already live on `sev.fz.xk`/`sev.fz.pk`);
reuses the reinsurance rebucketing machinery (`_rebucket_to_grid`).

> ## Revision note (2026-06-04) — decisions + corrections that OVERRIDE the body
>
> Author decisions:
> 1. **Name `dsev_bucket`** (not `sev_bucket`): the knob applies only to discrete
>    point-mass severities (`dhistogram`/`fixed`); continuous severities are
>    already exact via the cdf-difference. Sibling of `reins_bucket`.
> 2. **Default `'linear'`** (NOT `'nearest'`). Rationale: mean-preservation is the
>    correct default for an accuracy-focused library, and it matches the existing
>    `reins_bucket` default (`'linear'`). The `nearest`-default "zero blast radius"
>    argument in §4/§7 is therefore **rejected**; see the blast-radius revision
>    below.
>
> Corrections to the body (the plan was drafted before `_DiscreteRV` and has two
> bugs):
> 3. **§3a is obsolete.** `_DiscreteRV` already stores sorted `xk`/`pk`; read
>    `sev.fz.xk` / `sev.fz.pk` directly. No new `support_ps` attribute needed.
> 4. **§3b guard is WRONG.** `sev.detachment == np.inf` is *never* true for a
>    discrete severity (`_build` truncates `limit = min(limit, max(xs))`, so
>    `detachment = max(xs)`, finite). The correct "truly unlayered" test is
>    **`sev.exp_attachment is None`** (verified: unlayered `dsev` has
>    `exp_attachment is None`, `attachment == 0`, `detachment == max(xs)`). Same
>    trap that bit the moments path in the `_DiscreteRV` work.
> 5. **`_rebucket_to_grid` needs a `scheme` parameter.** It currently hard-reads
>    `self.reins_bucket`. Generalise to `_rebucket_to_grid(values, mass,
>    scheme=None)` (default `self.reins_bucket`, so the two reinsurance call
>    sites are unchanged); the `dsev` call passes `scheme=self.dsev_bucket`.
> 6. **§6 signed-grid concern is RESOLVED, not deferred.** Verified
>    `x_min == xs_sev[0] == xs[0]` for `dsev [-2 5]`, so the existing scatter onto
>    `self.xs` already places signed atoms correctly. This *must* work in Phase 1
>    anyway, because `linear`-default means `dsev [-2 5]` (already in the suite)
>    hits the scatter on day one. On-grid signed atoms give `f==0` ⇒ linear ==
>    nearest, so it is exact.
> 7. **Blast radius is NOT zero (and that's fine).** With `linear` default,
>    on-grid integer-atom / `bs==1` cases (dice, `dsev [-2 5]`, the PIR discrete
>    ports) are byte-for-byte unchanged (`f==0`). Only genuinely *off-grid*
>    discrete cases move — and they move *toward* correctness (mean-preserving),
>    so any baseline/golden that shifts is **re-captured** as the new truth
>    (exactly as we did for the `_DiscreteRV` moments). First step of execution:
>    run the suite and see which, if any, baselines move; expectation is none,
>    since the corpus `dsev` are all on-grid.
> 8. **Layered discrete stays on the cdf-diff path in Phase 1** (per §5), but
>    note `linear`-default makes the unlayered-vs-layered asymmetry *visible*:
>    document that `dsev_bucket` linear applies to unlayered discrete atoms; a
>    layered discrete severity discretises via the standard cdf-difference. The
>    layer value-map needed for Phase 2 is now trivial — it's the same
>    `clip(atoms-a, 0, limit)` map `_DiscreteRV.layer_moments` already uses.
> 9. **Close-out:** bump `pyproject.toml` to 1.0.0a28; README bullet; surface
>    `dsev_bucket` in `info()`/`describe` beside `reins_bucket`/`sev_calc`;
>    regenerate `ref_include.rst` only if the grammar changes (it does **not** —
>    `dsev_bucket` is a build/update kwarg, not DecL syntax) so no grammar regen;
>    keep the `.rst` severity/discretization docs in lockstep; move this file to
>    `dev/done/` when green.

## 1. The problem

When an `Aggregate` is discretized (`Aggregate.discretize`, distributions.py
~3731), the severity is sampled onto the model grid `xs = bs·arange` at the
half-bucket boundaries `xs ± bs/2`:

```python
adj_xs = np.hstack((xs_sev - bs/2, xs_sev[-1] + bs/2))
appx   = -np.diff(fz.sf(adj_xs))      # (or np.diff(fz.cdf), or max of both)
```

For a **discrete** severity (`dsev` / `dhistogram` / `fixed`), `fz` is a step
CDF, so `diff` over `[xᵢ−bs/2, xᵢ+bs/2)` collects exactly the atoms that fall in
that window — i.e. **each atom snaps to its nearest grid point.** That is the
`nearest` scheme. When the atoms are off-grid (empirical samples with a
non-integer `bs`), nearest-rounding biases the first moment by up to `bs/2` per
atom; the discretized mean ≠ `Σ xₖ pₖ`.

The fix the user wants: offer a **`linear`** option that splits each atom's
mass between its two bracketing grid points `k`, `k+1` with weights `1−f`, `f`
(`f = xₖ/bs − k`). Because `(1−f)·k·bs + f·(k+1)·bs = xₖ`, the discretized
**first moment is preserved exactly**, total mass is preserved, and on-grid
atoms (`f = 0`) reduce to nearest — so the common `bs = 1`, integer-atom case is
unchanged.

> Note: more often than not the atoms are integers and `bs = 1`, so `nearest`
> and `linear` coincide and this is a non-issue. It bites only when the atoms
> are empirical samples and `bs` is not an integer.

## 2. The machinery already exists — reinsurance rebucketing

The reinsurance code already does exactly this scatter, with the same two
schemes, in `Aggregate._rebucket_to_grid` (distributions.py ~4568):

```python
scaled = (np.asarray(values) - self.x_min) / bs
if self.reins_bucket == 'nearest':
    idx = np.clip(np.round(scaled).astype(int), 0, n-1)
    np.add.at(out, idx, mass)
else:  # 'linear' -- mass split preserves E[X] exactly
    k   = np.clip(np.floor(scaled).astype(int), 0, n-1)
    f   = np.clip(scaled - k, 0.0, 1.0)
    kp1 = np.clip(k + 1, 0, n-1)
    np.add.at(out, k,   mass * (1 - f))
    np.add.at(out, kp1, mass * f)
```

The toggle is the `reins_bucket` property (`'linear'` default for reins,
validated to `{'linear','nearest'}`, with cache invalidation on set — see
~1843-1863 and the `REINS_BUCKET_DEFAULT` constant threaded through `__init__`
~3138). So both the algorithm **and** the vocabulary are already in the class.

## 3. How easy is it? — LOW / MODERATE

The arithmetic is solved; the work is plumbing and one scoping decision. Three
pieces:

### 3a. Retain the atom masses on the Severity (1 line)
`SeverityDHistogram._build` already stores `self.support_atoms = xs` but drops
`ps`. Add alongside it:

```python
self.support_ps = np.asarray(ps, dtype=float)
```

`support_atoms` + `support_ps` then fully describe the discrete law for the
scatter. (`SeverityFixed` inherits this for free — single atom, mass 1.)

### 3b. Take the linear branch in `discretize` for discrete components
`discretize` loops `for fz in self.sevs:` where each element is a **`Severity`**
(the name is misleading — it exposes `cdf`/`sf` via `rv_continuous`, and carries
`sev_kind`, `support_atoms`, etc.). For a discrete, **unlayered** component,
when the active scheme is `linear`, build that component's bucketed density by
scattering its atoms instead of differencing the step CDF:

```python
# CORRECTED per revision note (4) & (5): use exp_attachment is None for
# "unlayered", read masses off the _DiscreteRV, and pass the dsev scheme.
for sev in self.sevs:
    if (self.dsev_bucket == 'linear'
            and sev.sev_kind in ('dhistogram', 'fixed')
            and sev.exp_attachment is None):          # truly unlayered
        appx = self._rebucket_to_grid(sev.fz.xk, sev.fz.pk,
                                      scheme=self.dsev_bucket)
    else:
        appx = -np.diff(sev.fz.sf(adj_xs))   # current path; == 'nearest' for atoms
    ...
```
> ⚠️ The original guard `sev.detachment == np.inf` is **dead** — discrete
> severities always have a finite `detachment` (= `max(xs)`). Use
> `sev.exp_attachment is None`.

Everything downstream (`normalize`, mixture weighting, the FFT) is unchanged —
`_rebucket_to_grid` returns a length-`n` probability vector exactly like the
`diff` path, and the linear scatter already sums to `Σ pₖ` (≈ 1).

### 3c. The toggle — the one real decision (see §4)

That's the whole change for the headline (unlayered) case: ~1 line on the
Severity, a small branch in `discretize`, plus the toggle. The heavy lifting
(`_rebucket_to_grid`, scheme validation, cache invalidation) is reused verbatim.

## 4. The toggle: dedicated `sev_bucket` (RECOMMENDED) vs reuse `reins_bucket`

Two options:

### Option A — reuse the existing `reins_bucket` (the "iirc reuse it" idea)
Read `self.reins_bucket` in `discretize`.
- **Pro:** no new knob; one word for "how off-grid mass lands."
- **Con 1 — default flip / blast radius.** `reins_bucket` defaults to
  `'linear'`. Reading it for severity discretization would silently switch
  every off-grid `dsev` from today's `nearest` to `linear`, changing the
  `tests/baseline/` density baselines for any non-integer-atom case. Not zero
  blast radius.
- **Con 2 — conceptual coupling.** Reinsurance net/ceded rebucketing and
  severity discretization are different operations; tying them to one switch
  means you can't pick `nearest` severity + `linear` reins (or vice-versa).

### Option B — dedicated `sev_bucket`, default `'nearest'` (RECOMMENDED)
Add a parallel property mirroring `reins_bucket` exactly — same
`{'linear','nearest'}` validation, same setter/cache-invalidation pattern, a
`SEV_BUCKET_DEFAULT = 'nearest'` constant next to `REINS_BUCKET_DEFAULT`, and a
`sev_bucket=None` kwarg on `__init__`/`update`/`update_work`.
- **Zero blast radius by construction:** default `'nearest'` reproduces today's
  behaviour byte-for-byte; `linear` is purely opt-in. Aligns with the project's
  "explicit opt-in over magic" preference.
- **Orthogonal:** severity and reinsurance bucketing chosen independently.
- **Cost:** ~15 lines of boilerplate cloned from the `reins_bucket` property
  (cheap, and the parallel structure is self-documenting).

**Recommendation: Option B.** Keep the `reins_bucket` vocabulary and code shape
(so the two read as siblings) but a separate attribute defaulting to `nearest`.
Surface it in `info()`/`describe` next to `sev_calc` and `reins_bucket`.

## 5. Scope boundary: layered discrete severities (defer to Phase 2)

The clean reuse above assumes the discrete component is **unlayered**
(`attachment == 0`, `detachment == inf`) — exactly the empirical-sample use case
the user cites. A *layered* discrete severity maps each atom through
`min(limit, (x − attachment)₊)` (with conditional renormalisation), which moves
atoms and can collapse several onto the limit. To scatter those linearly you
must first transform the atoms/masses, not read `support_atoms` directly.

**Phase 1 (this plan):** unlayered discrete only; layered discrete components
fall through to the current `diff` path (= `nearest`). Document the limitation.

**Phase 2 (optional follow-up):** apply the layer value-map to
`(support_atoms, support_ps)` —
`vals = np.minimum(limit, np.maximum(atoms - attachment, 0))`, renormalise by
`pattach` when `conditional` — then `_rebucket_to_grid(vals, masses)`. Atoms
landing exactly on the limit/attachment collapse naturally under the scatter.
Mass at `0` (full retention below attachment) needs the same handling the
`SeverityMeta` zero-bucket gives — another reason to keep it out of Phase 1.

## 6. Signed grid note

`_rebucket_to_grid` scatters onto `self.xs` using `(values − self.x_min)/bs`,
so **negative atoms already work** provided the severity is discretized on the
same grid origin. In `discretize` the severity lives on `xs_sev` (physical 0 at
index `i0`), which equals `self.xs` on the default grid but is offset in signed
mode. Confirm `self.x_min` / `self.xs` is the intended scatter target on the
signed grid, or thread the severity grid origin into the call. Low risk, but a
required check before enabling `linear` for signed `dsev [-2 …]`.

## 7. Verification / blast radius

- With `sev_bucket` defaulting to `'nearest'` (Option B), **all existing
  baselines are unchanged byte-for-byte** — confirm via `uv run pytest` and the
  `tests/baseline/` parquet density diffs (must be empty).
- New targeted test: a `dsev` with off-grid atoms and non-integer `bs`
  (e.g. atoms `[0.3, 1.7, 2.4]`, `bs = 0.5`):
  - `sev_bucket='nearest'` → discretized `E[X]` biased by ≤ `bs/2`;
  - `sev_bucket='linear'`  → discretized `E[X] == Σ xₖ pₖ` to FFT precision.
  Assert both, plus total mass `== 1` under each.
- On-grid sanity: integer atoms + `bs = 1` give identical densities under both
  schemes (the `f = 0` degeneracy) — lock this in so the common case is proven
  invariant.
- `info()` / `describe` show the new `sev_bucket` field.

## 8. Recommendation

The mean-preserving option is **easy** — the algorithm, the scheme vocabulary,
the validation, and the cache-invalidation pattern all already exist in the
reinsurance rebucketing code. Implement it as:

1. store `support_ps` on `SeverityDHistogram._build` (1 line);
2. add a dedicated `sev_bucket` property (default `'nearest'`) mirroring
   `reins_bucket`, threaded through `__init__`/`update`/`update_work`;
3. branch in `discretize` to `_rebucket_to_grid(support_atoms, support_ps)` for
   unlayered discrete components when `sev_bucket == 'linear'`;
4. defer layered-discrete and revisit the signed-grid origin as noted.

Default `nearest` keeps blast radius at zero and makes `linear` a clean opt-in
for the empirical-sample / non-integer-`bs` case where it matters.
