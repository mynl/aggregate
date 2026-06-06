# Sub-project C — Distortion owns its calibration

**Part of the Portfolio refactor.** Master plan: `portfolio-refactor-planning.md`. Prior: Sub-projects A, B must be landed.

## Goal

Move per-distortion-name Newton iterations out of `Portfolio.calibrate_distortion` (currently ~240 LOC of `if name == 'X': ... elif name == 'Y':` by distortion name) into the appropriate `Distortion<Kind>` subclasses in `aggregate/spectral.py`. Each subclass calibrates itself. `Portfolio.calibrate_distortion` shrinks to ~30 lines of dispatch + bookkeeping.

## Pre-conditions

- Sub-projects A, B landed and committed.
- hash: 6de20f2e57e4ec7b2a5a44a6a2b75aa44202265e
- 415/415 pytest green.

## Touches

- `aggregate/spectral.py` (heavy: adds a `calibrate` method to each pricing-distortion subclass)
- `aggregate/portfolio.py` (`calibrate_distortion` reduces dramatically; everything else untouched)
- `tests/` (new unit tests per distortion subclass)
- `README.rst`, `pyproject.toml`

## Why this lives on Distortion

`Portfolio.calibrate_distortion` today is a giant `if name == 'X'` switch over ~240 LOC. Each branch:

1. Defines a local `f(shape) → (residual, derivative)` closure (parametric to the distortion).
2. Runs a Newton iteration to convergence.
3. Returns a calibrated `Distortion` object.

The closures encode distortion-specific math (`f(rho)` for PH, `f(lam)` for Wang, etc.) that obviously belongs on the distortion subclass. Portfolio is just providing the `S` vector, `bs`, and the premium target.

After this sub-project:

- Each `Distortion<Kind>` subclass has a `calibrate(S, bs, premium_target, ess_sup=0.0, **kwargs)` method that mutates `self.shape` (and `self.error`, `self.assets`, `self.premium_target` for audit) and returns `self`.
- `Portfolio.calibrate_distortion(name, ...)` is a thin dispatcher: extract `S` and the target, construct an uncalibrated `Distortion(name=name, ...)`, call its `.calibrate(...)`, return.
- Each distortion is testable in isolation against a synthetic S vector with a known answer.

## Design — the Distortion.calibrate method shape

```python
class DistortionPH(Distortion):
    name = 'ph'

    def calibrate(self, S, bs, premium_target, *, ess_sup=0.0, **kwargs):
        """Newton iteration to find shape parameter rho such that
        integral of g(S) over the bs grid equals premium_target."""
        lS = np.log(S)
        shape = 0.95
        for i in range(50):
            trho = S ** shape
            ex = np.sum(trho) * bs
            ex_prime = np.sum(trho * lS) * bs
            fx = ex - premium_target
            if abs(fx) < 1e-5:
                break
            shape -= fx / ex_prime
        else:
            logger.warning('DistortionPH calibration: %d iters, residual %g',
                           i + 1, fx)
        self.shape = shape
        self.error = fx
        self.assets = kwargs.get('assets')  # informational
        self.premium_target = premium_target
        return self
```

(For PH; analogous for each other pricing distortion. Faithful port of the current branches in `Portfolio.calibrate_distortion`.)

## Distortions to migrate

From the current `Portfolio.calibrate_distortion` switch (portfolio.py:2630-2762):

| name           | subclass        | notes |
|:---------------|:----------------|:------|
| `ph`           | `DistortionPH`  | Newton over `shape=rho`. |
| `wang`         | `DistortionWang`| Newton over `shape=lam`. |
| `ly`           | `DistortionLy`  | Linear yield; uses `r0`. |
| `clin`         | `DistortionClin`| Capped linear; uses `r0`. |
| `roe`/`ccoc`   | `DistortionCCoC`| Closed-form: `r = (premium - el) / (assets - premium)`. Both names map to the same subclass. |
| `lep`          | `DistortionLep` | Layer-equivalent pricing; uses `r0`. |
| `tt`           | `DistortionTt`  | Wang-t with `df` parameter. |
| `cll`          | `DistortionCll` | Capped log-linear; uses `r0`. |
| `dual`         | `DistortionDual`| Dual moment. |
| `tvar`         | `DistortionTVaR`| Newton, `max_iter=200`. |
| `wtdtvar`      | `DistortionWtdTVaR` | Weighted TVaR with `(p0, p1)` from `df` arg. |

Verify the subclass naming against current `aggregate/spectral.py`. The Stage 1c work renamed distortion subclasses to `Distortion<Kind>` (per the Base*-prefix convention in CLAUDE.md). If a subclass doesn't exist yet for one of these names, the calibration logic goes on the base `Distortion` class with a name-dispatch — but check first.

## Tasks (execution order)

1. **Spec the interface.** Decide the signature for `Distortion.calibrate(S, bs, premium_target, **kwargs)` (likely also `ess_sup`, possibly `assets`). Document in the base class.

2. **Migrate one subclass first** (PH is the simplest). Copy the closure body from `Portfolio.calibrate_distortion(name='ph', ...)` into `DistortionPH.calibrate(...)`. Adjust to take `S`, `bs`, `premium_target` as args instead of pulling from `self.bs`, `self.density_df`, etc.

3. **Wire `Portfolio.calibrate_distortion('ph', ...)` to call the new method.** Reduce the PH branch to:
    ```python
    if name == 'ph':
        dist = DistortionPH(shape=0.95, name='ph')
        dist.calibrate(S, bs=self.bs, premium_target=premium_target, ess_sup=ess_sup)
        return dist
    ```

4. **Verify numerical equivalence for PH.** Add a regression test: `port.calibrate_distortion('ph', premium_target=...)` produces the same `shape` value before vs after. Use a captured baseline.

5. **Repeat for each remaining distortion.** One at a time, or in batches. Same pattern: copy the Newton closure into the subclass; reduce the Portfolio branch; verify.

6. **Final Portfolio.calibrate_distortion shape.** Once all subclasses own their math:
    ```python
    def calibrate_distortion(self, name, *, premium_target=0.0, assets=0.0, p=0.0,
                             roe=0.0, kind='lower', r0=0.0, df=None,
                             S_column='S', S_calc='cumsum'):
        # determine assets and premium_target (as today)
        a, prem = self._resolve_a_and_premium(...)  # extract from current method
        # extract S vector at a
        S = self._extract_S(a, S_column, S_calc)
        ess_sup = ...  # tail-detection logic from current method
        # dispatch to the right subclass
        dist = self._make_distortion(name, r0=r0, df=df)  # uncalibrated constructor
        dist.calibrate(S, bs=self.bs, premium_target=prem, ess_sup=ess_sup,
                       assets=a)
        return dist
    ```
   ~30 lines instead of ~240.

7. **Add unit tests in `tests/`.** One per pricing distortion subclass: construct with a known `S` vector + known `premium_target`, assert that `calibrate` recovers the expected `shape`. Synthetic test fixtures, not Portfolio-driven.

8. **Run full pytest.** `uv run pytest`. Should be 415 + new tests. Old tests should pass with identical numerics — `Portfolio.calibrate_distortion(...)` and the new `Distortion<Kind>.calibrate(...)` should produce bit-identical shape values (same Newton iteration, same starting point, same convergence criterion).

9. **README + version bump.** Add the bullet (below).

## Verification

- `uv run pytest` → 415 + 3 PEG regression tests + N new per-distortion calibration tests passed.
- **PEG regression check:** `uv run pytest tests/test_portfolio_peg_regression.py -v` → 3 passed. The `test_calibration_shapes` assertion is the headline check here — moving calibration to the subclasses must reproduce the same Newton convergence to within `rtol=1e-8`.
- Visual `test_suite` green.
- Numerical check: for each of (ph, wang, dual, tvar, ccoc), `port.calibrate_distortion(name, ...)` produces the same `shape` value before vs after. The PEG baseline pins this; new per-subclass synthetic-S tests cover edge cases.
- Code shape: `Portfolio.calibrate_distortion` body ~30 LOC.

## Commit

One commit at sub-project end after human review.

**Commit message draft:**

```
Portfolio refactor C: distortion calibration moves to Distortion subclasses

Portfolio.calibrate_distortion was ~240 LOC of per-name Newton iterations
in a giant if/elif. Each Distortion subclass now knows how to calibrate
itself given S, bs, and a premium target.

- DistortionPH.calibrate, DistortionWang.calibrate, DistortionLy.calibrate,
  DistortionClin.calibrate, DistortionCCoC.calibrate, DistortionLep.calibrate,
  DistortionTt.calibrate, DistortionCll.calibrate, DistortionDual.calibrate,
  DistortionTVaR.calibrate, DistortionWtdTVaR.calibrate.
- Portfolio.calibrate_distortion shrinks to dispatch + state extraction
  (~30 LOC).
- Added per-subclass unit tests in tests/.

Numerical results unchanged.
```

## README bullet draft

```rst
- Distortion calibration responsibility moved from
  ``Portfolio.calibrate_distortion`` (~240 LOC if/elif by name) into the
  ``Distortion`` subclasses themselves. Each pricing-distortion subclass
  (PH, Wang, Dual, TVaR, CCoC, Lep, Ly, Clin, Tt, Cll, WtdTVaR) now
  provides its own ``calibrate(S, bs, premium_target, ...)`` method.
  ``Portfolio.calibrate_distortion`` shrinks to ~30 LOC of dispatch +
  bookkeeping. Each distortion is now testable in isolation.
```

## Post-conditions

Distortion subclasses own their own numerics. `Portfolio.calibrate_distortion` is small and clear. Per-distortion unit tests exist in `tests/`. Sub-project D (distortion pricing redesign) is unblocked.

## Cross-cutting reminders

- `UV_LINK_MODE=copy` is set. Don't override.
- Run `uv run pytest`, not bare `pytest`.
- PowerShell shell: use `rg` directly via Bash; no awk/sed.
- Don't prefix Bash commands with `cd`.
- Update `README.rst` at sub-project close. Bump alpha version.
- Don't build docs in a verification cycle.
- Do NOT commit unless the user explicitly says so.
- Pre-existing visual `test_suite` failure: `Cc.Freq20`. Ignore.
