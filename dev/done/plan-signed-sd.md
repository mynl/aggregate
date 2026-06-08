# Plan: fix SD/variance derivation for zero-mean (signed) aggregates

## Symptom

```python
from aggregate import build
a = build('agg A2 dfreq [3] dsev [-1 1]')
a.describe        # Sev SD and Agg SD show NaN (rendered ~0)
```

Severity `{-1, +1}` equiprobable: mean 0, var 1, **SD 1**. Aggregate of 3 fixed
claims: mean 0, var 3, **SD √3 ≈ 1.732**. `describe` (correctly on its signed
`SD`-trio path, `_describe_signed`) shows `NaN` for both Sev and Agg SD.

## Root cause

The stored moments are **correct** — `stats_df` has `('sev','ex2')=1.0`,
`('agg','ex2')=3.0`, means 0. Only the **SD/variance derivation** is wrong: SD is
reconstructed as `mean × CV`, and `CV = SD/mean` is `NaN/inf` when the mean is 0,
so `SD = 0 × NaN = NaN`. The second-moment information is destroyed by the
round-trip through CV.

`distributions.py`, four computation sites (all of the form `sd = m * cv`):

| Lines | What | Source in scope |
|---|---|---|
| 3523–3524 | `agg_sd = agg_m*agg_cv`; `agg_var = agg_sd²` | `_mixed = stats_df['mixed']` (has `('agg','ex2')`, `('agg','mean')`) |
| 3529–3530 | `sev_sd = sev_m*sev_cv`; `sev_var = sev_sd²` | `_mixed` (`('sev','ex2')`, `('sev','mean')`) |
| 4247–4248 | `est_sev_sd = est_sev_m*est_sev_cv`; `est_sev_var = …²` | live wrangler `_mw` (4239) + `sev_ex1,sev_ex2` (4240) |
| 4273–4274 | `est_sd = est_m*est_cv`; `est_var = est_sd²` | live wrangler `_mw` (4270) + `agg_ex1,agg_ex2` (4271) |

The irony: `_describe_signed` exists *specifically* to avoid unstable CV at
mean≈0 (its docstring says so) — but the SD it reports is itself built from CV,
inheriting the exact instability it was meant to dodge.

## The clean source already exists

`moments.py` `MomentWrangler` exposes:

- `.central` → `(mean, variance, third_central)` — **variance directly**.
- `.stats` → `Series(ex, var, sd, cv, skew)` with `sd = sqrt(var)` and the
  `cv = nan if mean == 0` / `skew = nan if sd == 0` guards already applied
  (lines 371–385). Mutually consistent by construction.
- `.mcvsk` → `(mean, cv, skew)` — what the code currently uses, **dropping**
  `sd`/`var`, then lossily rebuilding `sd = mean*cv`.

So at the two empirical sites the correct SD/var are one attribute away. At the
two theoretical sites, `var = ex2 − mean²` from the already-stored `_mixed` rows.

## Fix (per site)

Compute variance/SD **directly from moments**, never from CV. Clamp fp dust:
`var = max(var, 0.0)` before `sqrt` (a symmetric ex2−mean² can land slightly
negative).

1. **Theoretical, ~3523–3530.** For agg and sev, replace `sd = m*cv; var = sd*sd`
   with:
   ```python
   self.agg_var = max(float(_mixed[('agg','ex2')]) - self.agg_m**2, 0.0)
   self.agg_sd  = math.sqrt(self.agg_var)
   self.sev_var = max(float(_mixed[('sev','ex2')]) - self.sev_m**2, 0.0)
   self.sev_sd  = math.sqrt(self.sev_var)
   ```
   Leave `agg_cv`/`sev_cv` as read from `_mixed` (they are legitimately `nan` at
   mean 0; nothing should depend on them for SD any more).

2. **Empirical severity, ~4239–4248.** `_mw` is in scope. Source var/sd from the
   wrangler instead of rebuilding:
   ```python
   self.est_sev_m, self.est_sev_cv, self.est_sev_skew = _mw.mcvsk
   self.est_sev_var = max(_mw.central[1], 0.0)
   self.est_sev_sd  = math.sqrt(self.est_sev_var)
   ```
   (Equivalently read the whole row from `_mw.stats`.) Keep the `else` NaN branch
   (no `sev_density`) setting `est_sev_var = est_sev_sd = nan`.

3. **Empirical aggregate, ~4270–4274.** Same pattern with `_mw` from 4270:
   ```python
   self.est_m, self.est_cv, self.est_skew = _mw.mcvsk
   self.est_var = max(_mw.central[1], 0.0)
   self.est_sd  = math.sqrt(self.est_var)
   ```

4. **Affine / P&L path, 4522–4531 (`_apply_agg_affine`).** No change needed: SD
   is invariant under shift+reflect (comment already says so) and now arrives
   correct from step 3; `est_cv` is already guarded (`= est_sd/est_m if est_m
   else inf`). Re-read to confirm it still holds after step 3; the existing
   comment ("describe shows SD when signed") stays accurate.

`import math` if not already imported (else use `np.sqrt`, but guard the scalar
clamp).

## Safety / invariant

For any **positive-mean** object, `ex2 − mean² == (mean·cv)²` exactly (up to fp),
so the values are **unchanged** for all normal aggregates — this only removes a
lossy reconstruction. The behavioural change is confined to mean≈0 signed
objects, where SD goes `NaN → correct`. Non-signed `describe` uses the CV trio
and is untouched. Downstream consumers of `agg_sd`/`agg_var`/`sev_var`
(e.g. lines 4417, 5863) get identical values for non-signed inputs.

Leave the `MomentWrangler` itself alone — its `nan`-at-mean-0 CV is correct.

## Tests

Add to `tests/` (and the matching DecL line to
`src/aggregate/agg/test_decl.agg` under the discrete/signed section, per the
keep-in-sync rule):

- **Zero-mean signed agg** — `agg A2 dfreq [3] dsev [-1 1]`:
  - `a.sev_sd == pytest.approx(1.0, abs=1e-9)`,
    `a.sev_var == approx(1.0)`.
  - `a.agg_sd == approx(sqrt(3))`, `a.agg_var == approx(3.0)`.
  - `a.est_sev_sd`/`a.est_sd` likewise ≈ 1 and √3.
  - `describe` `SD` column for Sev/Agg is finite and matches (both the
    theoretical `SD` and the `Est SD` columns).
- **Regression guard (non-signed unchanged)** — pick an existing positive-mean
  case (e.g. `agg Dice dfreq [3] dsev [1:6]`) and assert `agg_sd`/`agg_var`
  equal the pre-fix values (i.e. `sqrt(ex2 − mean²)` and the old `mean*cv`
  agree). Confirms no drift on the common path.


## Housekeeping (standing rules)

- N-track / signed work → bump `pyproject.toml` `1.0.0a*`, add a `CHANGELOG.md`
  section ("Fixed: SD/variance for zero-mean signed aggregates — derived from
  the second moment instead of `mean × CV`").
- `dev/TODO.md`: note under the N-track (validation/signed) if a tracked item
  covers it; otherwise a one-line Fixed note.
- Move this plan to `dev/done/plan-signed-sd.md` on landing.

## Out of scope

- No change to `MomentWrangler`, the FFT core, or the discretization (moments
  are already correct).
- `_describe_signed`'s `freq_sd = mean*cv` (5624, 5643) is fine for normal
  frequencies (mean > 0); only worth touching if a zero-mean frequency ever
  arises — not part of this fix.
