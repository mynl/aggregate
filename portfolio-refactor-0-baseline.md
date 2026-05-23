# Sub-project 0 — PEG regression baseline (pre-flight)

**Part of the Portfolio refactor.** Master plan: `portfolio-refactor-planning.md`. Runs **before** Sub-project A.

## Goal

Capture a numerical baseline for a representative two-unit Portfolio (`PEG`) on the current code, persist it as JSON, and build a regression test that re-runs the analysis at the end of every subsequent sub-project. Every change to the calibrate/analyze pipeline must reproduce the baseline numbers exactly (within float tolerance).

This is the contract the refactor must not break.

## The Portfolio under test

```python
from aggregate import build

port = build(
    'port PEG '
    'agg A 100 claims 50 xs 0 sev lognorm [30 50 100] cv [0.1 .6 1.8] '
    '       wts [.5 .25 .25] mixed gamma .25 '
    'agg B 150 claims 100 xs 0 sev lognorm [30 50 100] cv [0.1 .6 1.8] '
    '       wts [.5 .25 .25] mixed gamma .35'
)
port.update(log2=16, bs=...)         # bs picked from recommend_bucket; see capture script
port.calibrate_distortions(COCs=[.15], Ps=[.995])  # current API; renamed in Sub-project D
ad = port.analyze_distortions2(.995)               # current API; renamed in Sub-project D
```

PEG exercises:
- Limit-and-attachment severity (50 xs 0 on A, 100 xs 0 on B).
- Three-component severity mixture per unit (lognormal with three CV/mean combinations).
- Gamma frequency mixing with different CVs per unit (.25 vs .35) — the standard "mixed > independent" cross-unit correlation case.
- Five calibrated distortions (`ccoc`, `ph`, `wang`, `dual`, `tvar`).
- A single asset level (`p=0.995`) for pricing readout.

Small enough to run quickly (a few seconds), large enough to exercise the whole pipeline.

## Pre-conditions

- Current branch state: post-Stage 1c++ on Aggregate. `Portfolio` not yet refactored.
- 415/415 pytest green.
- `build('port ...')` works on the current code (no Portfolio-side changes yet).

## Touches

- `tests/peg.py` (new) — helper to construct PEG and (optionally) update + calibrate it. Reused by the capture script and the test.
- `tests/capture_peg_baseline.py` (new) — runs the full sequence, dumps to JSON.
- `tests/data/peg_baseline.json` (new) — the captured baseline.
- `tests/test_portfolio_peg_regression.py` (new) — pytest module that re-runs PEG and compares to baseline.
- `pyproject.toml` — no change.
- `README.rst` — small note that PEG regression test exists.

## What the baseline JSON contains

JSON-stable structure. Keep it small and meaningful — not the full augmented_df (~80 MB across five distortions). Capture only what `analyze_distortions2(.995)` consumers actually read.

```jsonc
{
  "meta": {
    "captured_at": "2026-05-23",
    "aggregate_version": "1.0.0a4",      // pre-A version
    "program": "port PEG agg A ...",
    "log2": 16,
    "bs": 0.5,                            // whatever update picks
    "p_calibration": 0.995,
    "coc_calibration": 0.15
  },

  "portfolio_moments": {
    "agg_m": 12345.6, "agg_cv": 0.42, "agg_skew": 1.23,
    "est_m": 12345.5, "est_cv": 0.42, "est_skew": 1.23  // empirical post-update
  },

  "calibration": {                        // shape per distortion
    "ccoc":  {"shape": 0.13043, "error": 1.2e-9},
    "ph":    {"shape": 0.85,    "error": 5.1e-7},
    "wang":  {"shape": 0.42,    "error": 6.3e-7},
    "dual":  {"shape": 2.31,    "error": 4.7e-7},
    "tvar":  {"shape": 0.88,    "error": 9.2e-7}
  },

  "audit": {                              // analyze_distortion side-info
    "a_cal": 18000.0,                     // q(.995) — should match across distortions
    "p":     0.995,
    "K":     5400.0,                      // capital at a_cal under ccoc
    "LR":    0.69,
    "ROE":   0.15
  },

  "pricing": {                            // pricing DataFrames per distortion
    "ccoc": {
      "A":     {"L": 5000.0, "P": 5500.0, "M": 500.0, "Q": 1500.0, "LR": 0.91, "PQ": 3.67, "COC": 0.33},
      "B":     {"L": 7500.0, "P": 8100.0, "M": 600.0, "Q": 3000.0, "LR": 0.93, "PQ": 2.70, "COC": 0.20},
      "total": {"L": 12500.0,"P": 13600.0,"M": 1100.0,"Q": 4500.0, "LR": 0.92, "PQ": 3.02, "COC": 0.24}
    },
    "ph":    { ... same shape ... },
    "wang":  { ... },
    "dual":  { ... },
    "tvar":  { ... }
  }
}
```

(Values above are illustrative — the capture script writes real numbers.)

**Not captured:** the full augmented_df frames. Too large; redundant with the pricing readouts at `p=0.995`. If a future sub-project introduces drift in augmented_df values not visible at `p=0.995`, that's a signal we need finer-grained tests (add at that point).

## Tasks (execution order)

1. **Create `tests/peg.py`** with a small helper:
    ```python
    def build_peg(*, update=True, calibrate=True, p=0.995, coc=0.15, log2=16, bs=None):
        from aggregate import build
        port = build(
            'port PEG '
            'agg A 100 claims 50 xs 0 sev lognorm [30 50 100] cv [0.1 .6 1.8] '
            '       wts [.5 .25 .25] mixed gamma .25 '
            'agg B 150 claims 100 xs 0 sev lognorm [30 50 100] cv [0.1 .6 1.8] '
            '       wts [.5 .25 .25] mixed gamma .35'
        )
        if update:
            if bs is None:
                bs = port.best_bucket(log2)
            port.update(log2=log2, bs=bs, remove_fuzz=True)
        if calibrate:
            port.calibrate_distortions(COCs=[coc], Ps=[p])  # API renamed in Sub-project D
        return port
    ```
    Sub-project D will need to update the `calibrate_distortions` call (and any `analyze_distortions2` callers downstream); the baseline JSON itself is unchanged.

2. **Create `tests/capture_peg_baseline.py`** — runs PEG, dumps JSON. Pattern follows `tests/capture_sly_snapshot.py` (already in the repo). Key shape:
    ```python
    if __name__ == '__main__':
        port = build_peg(update=True, calibrate=True)
        ad = port.analyze_distortions2(.995)

        baseline = {
            'meta': {...},
            'portfolio_moments': {
                'agg_m': port.agg_m, 'agg_cv': port.agg_cv, 'agg_skew': port.agg_skew,
                'est_m': port.est_m, 'est_cv': port.est_cv, 'est_skew': port.est_skew,
            },
            'calibration': {
                name: {'shape': d.shape, 'error': d.error}
                for name, d in port.dists.items()
            },
            'audit': extract_audit(ad),
            'pricing': extract_pricing(ad),
        }
        with open('tests/data/peg_baseline.json', 'w') as f:
            json.dump(baseline, f, indent=2, default=float)
    ```
   `extract_audit` and `extract_pricing` are small helpers that pull values out of `ad` (which is an `Answer`-bagged dict-of-dataclasses today). They live in the script for now; may move to `tests/peg.py` later.

3. **Run the capture once** to produce `tests/data/peg_baseline.json`. Eyeball the values for sanity (mean roughly matches expected E[A] + E[B], CV plausible, calibrated shapes positive, ROE close to 0.15, etc.).

4. **Commit the baseline JSON** to the repo. This is the contract.

5. **Create `tests/test_portfolio_peg_regression.py`** — pytest module:
    ```python
    import json, pytest, numpy as np
    from tests.peg import build_peg

    BASELINE = json.load(open('tests/data/peg_baseline.json'))

    @pytest.fixture(scope='module')
    def peg():
        port = build_peg(update=True, calibrate=True)
        ad = port.analyze_distortions2(.995)
        return port, ad

    def test_portfolio_moments(peg):
        port, _ = peg
        for k, expected in BASELINE['portfolio_moments'].items():
            actual = getattr(port, k)
            assert np.isclose(actual, expected, rtol=1e-10), f'{k}: {actual} vs {expected}'

    def test_calibration_shapes(peg):
        port, _ = peg
        for name, expected in BASELINE['calibration'].items():
            d = port.dists[name]
            assert np.isclose(d.shape, expected['shape'], rtol=1e-8)
            assert abs(d.error) < 1e-5

    def test_pricing(peg):
        port, ad = peg
        for dname, expected_lines in BASELINE['pricing'].items():
            actual_df = extract_pricing_df(ad, dname)  # tests-local helper
            for line, expected_stats in expected_lines.items():
                for stat, expected_val in expected_stats.items():
                    actual_val = actual_df.loc[line, stat]
                    assert np.isclose(actual_val, expected_val, rtol=1e-8), \
                        f'{dname}.{line}.{stat}: {actual_val} vs {expected_val}'
    ```
   Tolerances: `rtol=1e-10` on moments (exact arithmetic), `rtol=1e-8` on calibrated values (Newton iteration's natural tolerance), `abs(error) < 1e-5` on calibration residual.

6. **Verify the test passes** on current code. `uv run pytest tests/test_portfolio_peg_regression.py -v`. Should be 3 tests passed.

7. **Add the test to the standard pytest run** by virtue of placement (it's in `tests/`).

8. **Commit** the helper module, capture script, baseline JSON, and test file. One commit.

## What each subsequent sub-project does with this

After each sub-project (A through E) completes its work, run:

```
uv run pytest tests/test_portfolio_peg_regression.py -v
```

All three tests must pass.

**Sub-project D changes the API.** Two things to update at that point:

1. In `tests/peg.py`, change `port.calibrate_distortions(COCs=[coc], Ps=[p])` to `port.calibrate_distortions(coc, p)`.
2. In the regression test fixture, change `port.analyze_distortions2(.995)` to `port.analyze_distortions(.995)`.
3. Update the `extract_pricing_df` helper if the result shape changed (`AnalyzeDistortionsResult` dataclass instead of `Answer`).

The **baseline JSON itself is unchanged.** Same numbers, different access pattern.

**Sub-project D also introduces `pricing_at`.** Once it exists, extend the regression with:

```python
def test_pricing_at_matches(peg):
    """pricing_at(p, d) on the cache must produce the same numbers
    as analyze_distortions(p)."""
    port, _ = peg
    for dname, expected_lines in BASELINE['pricing'].items():
        d = port.distortions[dname]
        actual_df = port.pricing_at(0.995, d)
        for line, expected_stats in expected_lines.items():
            for stat, expected_val in expected_stats.items():
                actual_val = actual_df.loc[line, stat]
                assert np.isclose(actual_val, expected_val, rtol=1e-8)

def test_pricing_at_caches(peg):
    """Second call must return the cached frame (same object id)."""
    port, _ = peg
    d = port.distortions['ph']
    frame_a = port.augmented_df(d)
    frame_b = port.augmented_df(d)
    assert frame_a is frame_b

def test_pricing_at_accepts_asset(peg):
    """pricing_at must accept either p (probability) or a (asset level)."""
    port, _ = peg
    d = port.distortions['ph']
    p = 0.995
    a = port.q(p)
    df_p = port.pricing_at(p, d)
    df_a = port.pricing_at(a, d)
    pd.testing.assert_frame_equal(df_p, df_a)
```

These three tests live in the same module; add them inside Sub-project D's work, alongside the `pricing_at` implementation.

## Verification

- `uv run pytest tests/test_portfolio_peg_regression.py -v` → 3 passed (after this sub-project), growing to 6 passed by end of Sub-project D.
- `tests/data/peg_baseline.json` exists and is committed.
- `tests/peg.py` and `tests/capture_peg_baseline.py` exist.
- The standard `uv run pytest` run picks up the new test file automatically and reports 415 + 3 (or 415 + 6 after D) passed.

## Commit

One commit at the end of this sub-project.

**Commit message draft:**

```
Add PEG Portfolio regression baseline

Captures numerical baseline (calibration shapes, portfolio moments,
pricing readouts at p=0.995) for a representative two-unit Portfolio
under five calibrated distortions (ccoc, ph, wang, dual, tvar).
Pinned as tests/data/peg_baseline.json; verified by
tests/test_portfolio_peg_regression.py.

Every subsequent Portfolio refactor sub-project must reproduce these
numbers exactly (rtol=1e-8 on calibrated values, rtol=1e-10 on
moments).
```

## README bullet draft

```rst
- Added a PEG Portfolio regression baseline (tests/data/peg_baseline.json)
  pinning the numerical output of ``calibrate_distortions`` and
  ``analyze_distortions`` on a two-unit Portfolio with limit profile,
  mixed severity, and gamma frequency mixing. Verified by
  ``tests/test_portfolio_peg_regression.py``; every subsequent
  Portfolio refactor commit reproduces these numbers.
```

## Cross-cutting reminders

- `UV_LINK_MODE=copy` is set. Don't override.
- Run `uv run pytest`, not bare `pytest`.
- PowerShell shell: use `rg` directly via Bash; no awk/sed.
- Don't prefix Bash commands with `cd`.
- Update `README.rst` at sub-project close.
- Don't build docs in a verification cycle.
- Do NOT commit unless the user explicitly says so.

## Future enhancements (not part of this sub-project)

- A second baseline at a different `p` value (e.g., `p=0.99`) once `pricing_at` exists — exercises a non-calibration asset level.
- A baseline for a Portfolio built via `create_from_sample` (switcheroo) to cover that path.
- A baseline using a single-unit Portfolio (corner case: `n_units==1`).

Add when the corresponding piece feels under-tested. For now PEG is enough.
