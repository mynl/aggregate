# Plan: noise-aware validation, denoised `describe`, empirical `ex123` (Aggregate + Portfolio)

## Context

`build('agg Die dfreq [1] dsev [1:6]')` — a symmetric die, analytic skew
exactly 0 — reports `fails sev skew, agg skew`. Confirmed cause: analytic
skew computes as fp dust (`1.7e-14`); the validation guard only skips when
skew is *exactly* `0` (`> 0`), so dust passes; and the check is a *relative*
error `Est/Skew - 1`, meaningless against a ~0 reference (evaluates to
`-0.833`, tripping `100*eps`). The same pattern exists in `Portfolio`.

This batch fixes it and adjacent items, building on human edits to
`moments.py` (moment helpers now share one worker, `_xsden_work`; the
`mw`/`mv` bug there has been fixed). Version bumps to **1.0.0a17**, the
batching bucket for this and subsequent small changes.

Guiding principle: aggregate is essentially exact, so numerical noise is
~1e-12…1e-14. Use a **single definite threshold**, not `np.isclose`'s loose
default `atol=1e-8`. `1e-14` is **too tight** (the die's skew difference is
~1.4e-14); use `1e-12`.

## Changes

### 1. `constants.py` — noise constant
- Add `VALIDATION_NOISE = 1e-12` immediately **below** `VALIDATION_EPS`,
  and to `__all__`. Docstring: absolute floor below which a quantity is
  treated as exact zero / numerical noise; ~3 orders above numpy
  pairwise-sum roundoff (~1e-15), 8 orders below `VALIDATION_EPS`.

### 2. `moments.py` — defective-dist logging, shared helpers, docstrings
- In `_xsden_work`, after `pg` and `xsm` are known, emit `logger.info(...)`
  when definitely defective: `pg > VALIDATION_NOISE`. Message reports prob
  sum, deficit `pg`, implied max `xsm`.
- Add shared helpers (imported by distributions.py and portfolio.py):
  - `_noise_aware_rel_error(est, ref)` → `est/ref - 1` where
    `abs(ref) > VALIDATION_NOISE`, else `est - ref`; vectorized, NaN-safe.
  - `_snap_noise(x)` → `np.where(np.abs(x) <= VALIDATION_NOISE, 0.0, x)`.
- NumPy-style docstrings for `_xsden_work`, `xsden_to_meancv`,
  `xsden_to_meancvskew`, `xsden_to_noncentral`: explain defective handling,
  implied max-loss `xsm + bs`, "noncentral" = raw E[X^k], shared worker.

### 3. `distributions.py` (Aggregate)
- Empirical `ex123` (`update_work`): via `xsden_to_noncentral`, write
  `('sev'|'agg','ex1'|'ex2'|'ex3')` into `empirical`; preserve None branch.
- error column: `_noise_aware_rel_error(empirical, mixed)`.
- `valid()`: re-anchor NOT_UPDATED (agg-mean NaN); drop Err Skew line; keep
  mean/aliasing; CV/skew via `np.isclose(est, theo, rtol=10*eps/100*eps,
  atol=VALIDATION_NOISE)` with `np.isfinite` guard, no `> 0`.
- `describe`: `_snap_noise` on moment columns; `_noise_aware_rel_error` for
  Err E[X]/Err CV(X).

### 4. `portfolio.py` (Portfolio) — analogous
- error column (`:1634`): `_noise_aware_rel_error(empirical, total)`.
- `valid()` (`:1690–1728`): drop Err Skew line; rewrite total-level CV/skew
  via `np.isclose` reading `stats_df['total']`/`['empirical']`; update
  docstring. NOT_UPDATED unchanged.
- `describe` (`:875–919`): `_snap_noise` + `_noise_aware_rel_error` on total
  block.
- ex123: no change — already at `:1618–1626`.

### 5. Version + README
- `pyproject.toml`: `1.0.0a16` → `1.0.0a17`.
- `README.rst`: new `1.0.0a17 (in progress)` section.

## Tests
- `tests/test_moments.py`: die moments; meancv/meancvskew/noncentral
  consistency; defective-dist tail mass + INFO log via caplog; zero-mean
  cv NaN; regression exactness.
- validation tests: die `valid()` ok + describe snapped + ex123 populated +
  error column sane; Portfolio with a symmetric unit; a genuinely skewed
  control still validates with nonzero skew.

## Verification
- `uv run pytest` (`UV_LINK_MODE=copy`) — full suite green.
- Smoke: re-run die; validation passes, describe clean.
- Do NOT build docs.

## Cleanup
- Move this file to `dev/done/` once merged.
