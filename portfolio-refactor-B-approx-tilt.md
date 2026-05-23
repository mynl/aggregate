# Sub-project B — Drop approximation + tilting paths

**Part of the Portfolio refactor.** Master plan: `portfolio-refactor-planning.md`. Prior: `portfolio-refactor-A-prune-and-move.md` (must be landed before starting B).

## Goal

Remove the auto-fallback-to-method-of-moments approximation path and the FFT tilting path. Both are unused in practice ("never a problem; just use more buckets"). Both reach into `Aggregate.update_work`, so they land together while we have the Aggregate-side context open.

## Pre-conditions

- Sub-project A landed and committed.
- `portfolio.py` at ~3,500 LOC.
- 415/415 pytest green.

## Touches

- `aggregate/portfolio.py` — `Portfolio.update` signature, `Portfolio.__init__` state, `Portfolio.ft`/`Portfolio.ift` wrappers, possibly `info`/`__str__`/`json`.
- `aggregate/distributions.py` — `Aggregate.update_work` (the inner FFT step where the moment-approximation branch lives).
- `aggregate/utilities.py` — module-level `ft`/`ift` if they carry a `tilt=None` parameter.
- `tests/` (pytest must stay green).
- `README.rst`, `pyproject.toml` (bullet + alpha bump).

## Approximation drop

The auto-approximation path is: when a unit's frequency exceeds `approx_freq_ge`, use method-of-moments (slognorm or sgamma) instead of FFT for that unit's update.

**What goes:**

- `approx_freq_ge` and `approx_type` attrs in `Portfolio.__init__`. Plus any `info`/`__str__`/`json` lines that read them.
- `approx_freq_ge`, `approx_type`, `approximation` kwargs on `Portfolio.update`.
- The `'exact' if agg.n < approx_freq_ge else approx_type` ternary in `Portfolio.update` (in the loop building per-unit densities). Pass `'exact'` unconditionally, or rename the param to something cleaner like `kind='exact'` at the call site to `Aggregate.update_work`.
- The slognorm/sgamma branch inside `Aggregate.update_work` keyed on the `approx_type` parameter (the moment-matching shortcut path). The FFT-via-FT-of-discretized-severity path is the only remaining one.

**What stays untouched:**

- `Portfolio.approximate(approx_type='slognorm')` and `Aggregate.approximate(approx_type='slognorm')`. These are user-facing on-demand method-of-moments *fits* (return a `scipy.stats` frozen RV or a DecL agg spec). They take `approx_type` as a parameter and don't read `self.approx_type`. Separate machinery; survives intact.

## Tilting drop

FFT tilting is a numerical trick (Grübel 1999, ASTIN) that exponentially tilts the FFT input to reduce wrap-around aliasing on heavy tails. The user prefers "just use more buckets" so this path is dead weight.

**What goes:**

- `tilt_amount` attr in `Portfolio.__init__`.
- The `if self.tilt_amount != 0: tilt_vector = np.exp(self.tilt_amount * np.arange(N))` block in `Portfolio.update` (around line 1440).
- Propagation of `tilt_vector` as a positional arg from `Portfolio.update` into `Aggregate.update_work`.
- The `tilt=None` parameter on `Portfolio.ft`/`Portfolio.ift` (the per-instance FFT wrappers).
- The `tilt=None` parameter on `aggregate.utilities.ft`/`ift` (the module-level FFT helpers).
- The tilt branch inside `Aggregate.update_work` (and `Severity` if it has one — unlikely but check).

**What stays untouched:**

- The FFT path itself, FFT padding (`padding=1` parameter on `update`), `ft`/`ift` wrapping — all stay. Just the `tilt=` knob disappears.

## Tasks (execution order)

1. **Grep approximation reads.** `rg "approx_freq_ge|approx_type|approximation\\s*=" aggregate/ tests/ -n`. Document every read site. Expect call sites in `Portfolio.update`, `Portfolio.__init__`, `Portfolio.info`, `Portfolio._repr_html_`, `Portfolio.json`, plus a couple of internal helpers and possibly a test or two.

2. **Drop approximation reads in Portfolio.** Remove the attrs from `__init__`. Remove the kwargs from `update`. Remove the ternary in the unit loop. Remove the `info`/`json` lines. If any test sets these kwargs, drop the kwarg from the test call.

3. **Drop approximation branch in Aggregate.update_work.** Find the `if approx_type == 'slognorm': ... elif approx_type == 'sgamma':` block (in `distributions.py`; usually in the inner FFT step where the unit's compound distribution is computed). Drop the branches; keep the FFT path as the only path. Simplify the function signature: remove the `approx_type` parameter if no remaining caller passes it; otherwise leave with a default and ignore.

4. **Confirm `Portfolio.approximate` / `Aggregate.approximate` untouched.** These should still work. Spot-check by calling each one in a Python REPL after the surgery.

5. **Grep tilting reads.** `rg "tilt_amount|tilt_vector|tilt\\s*=" aggregate/ tests/ -n`. Document every read.

6. **Drop tilt construction in Portfolio.update.** Remove the `tilt_amount` attr from `__init__`. Remove the `tilt_vector` build block in `update`. Stop passing `tilt_vector` into `Aggregate.update_work`.

7. **Drop `tilt=` from ft/ift wrappers.** In `Portfolio.ft`/`Portfolio.ift` (the methods around portfolio.py:2323), remove the `tilt=None` param and any references. Same in `aggregate.utilities.ft`/`ift` (the module-level helpers).

8. **Drop tilt branch in Aggregate.update_work.** Find the `if tilt_vector is not None:` block; remove. Keep only the untilted path.

9. **Test pass.** `uv run pytest`. Expected: 415 passed unchanged.

10. **Manual sanity.** Build something with high frequency to confirm FFT path handles it:
    ```python
    from aggregate import build, qd
    a = build('agg Big 1000 claims sev lognorm 10 cv 1 poisson', log2=18, bs=1/16)
    qd(a)
    ```
    Numerical result for `a.agg_m`, `a.agg_cv`, etc. should match a pre-B baseline. (Run this both before and after the diff if you want hard confirmation.)

11. **README + version bump.** Add the bullet (below) to the current alpha-version block. Bump `pyproject.toml`.

## Verification

- `uv run pytest` → 415 + 3 PEG regression tests passed.
- **PEG regression check:** `uv run pytest tests/test_portfolio_peg_regression.py -v` → 3 passed. Numerical baseline reproduced exactly — this is the critical check for B since the change touches the FFT path.
- Visual `test_suite` HTML report green (modulo Cc.Freq20).
- `Portfolio.update` signature: no `approx_freq_ge`, `approx_type`, `approximation`, `tilt_amount` (the last is on `__init__`, not `update`; verify it's gone from `__init__`).
- `Aggregate.update_work` signature shrinks correspondingly.
- `print(port.info)` shows no references to dropped attrs.
- Quick eyeball: `rg "approx_freq_ge|approx_type|tilt_amount|tilt_vector" aggregate/ tests/` returns nothing except possibly `Portfolio.approximate(approx_type=...)` / `Aggregate.approximate(approx_type=...)` (which is fine — those are user-facing on-demand fits).

## Commit

One commit at sub-project end after human review.

**Commit message draft:**

```
Portfolio refactor B: drop approximation and tilting paths

Both were unused in practice ("never a problem; just use more buckets").
Removing them simplifies Portfolio.update and Aggregate.update_work
signatures.

Approximation:
- Remove approx_freq_ge, approx_type, approximation kwargs from
  Portfolio.update.
- Remove approx_freq_ge, approx_type attrs from Portfolio.__init__.
- Drop the 'exact' if agg.n < approx_freq_ge else approx_type ternary;
  always use FFT.
- Drop slognorm/sgamma branch from Aggregate.update_work.
- Portfolio.approximate / Aggregate.approximate (user-facing method-of-
  moments fit) untouched.

Tilting:
- Remove tilt_amount attr from Portfolio.__init__.
- Remove tilt_vector construction in Portfolio.update.
- Remove tilt= parameter from ft/ift wrappers (portfolio.py and
  utilities.py).
- Drop tilt branch from Aggregate.update_work.

415/415 pytest green; numerical results unchanged on test suite.
```

## README bullet draft

```rst
- Dropped the auto-approximation path (slognorm/sgamma fallback when
  frequency was high) from ``Portfolio.update`` and
  ``Aggregate.update_work``; always use the FFT path.
  ``Portfolio.approximate`` / ``Aggregate.approximate`` (on-demand
  method-of-moments fit) untouched.
- Dropped FFT tilting (``tilt_amount`` attr, ``tilt_vector`` construction
  in ``Portfolio.update``, ``tilt=`` parameter on ``ft``/``ift``). Use
  more buckets instead.
```

## Post-conditions

`Portfolio.update` and `Aggregate.update_work` have leaner signatures. The Aggregate-side mechanical touch is done. Sub-project C (Distortion owns calibration) is unblocked.

## Cross-cutting reminders

- `UV_LINK_MODE=copy` is set in `.claude/settings.local.json`. Don't override.
- Run `uv run pytest`, not bare `pytest`.
- PowerShell shell: no awk/sed; use `rg` directly via Bash.
- Don't prefix Bash commands with `cd T:/...` — shell starts at project root.
- Update `README.rst` at sub-project close. Bump alpha version.
- Don't build docs as part of a verification cycle.
- Do NOT commit unless the user explicitly says so.
- Pre-existing visual `test_suite` failure: `Cc.Freq20` ZT-Poisson `brentq` crash. Ignore.
