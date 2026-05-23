# Sub-project D — Distortion pricing redesign

**Part of the Portfolio refactor.** Master plan: `portfolio-refactor-planning.md`. Prior: Sub-projects A, B, C must be landed.

## Goal

Six related changes that together produce a clean distortion-pricing pipeline on Portfolio. Order within the sub-project matters because each step feeds the next: **cache lands first → analyze methods collapse onto it → dataclasses wrap the small results → categoricals + renames clean up**.

## Pre-conditions

- Sub-projects A, B, C landed and committed.
- `Distortion` subclasses know how to calibrate themselves (from C).
- 415/415 pytest green.

## Touches

- `aggregate/portfolio.py` (heavy)
- `aggregate/spectral.py` (ordered categorical for distortion names)
- `aggregate/utilities.py` (`Answer` → named dataclasses sweep)
- PMIR call sites (`pmir_package/src/pmir/*.py` and `python/current/*.ipynb`) for the API rename and signature changes
- `tests/`
- `README.rst`, `pyproject.toml`

## Design — the core insight

Once `augmented_df(distortion)` is cache-backed, **the analyze methods collapse to trivial DataFrame manipulation**. The current ~250-LOC `analyze_distortion` body (already pruned by Sub-project A) becomes maybe two dozen lines: warm the cache (one call), extract a row (`pricing_at`), optionally wrap in a dataclass. The bulk of the work moves into the cache itself, where it's done once per (Portfolio, distortion) pair.

First call to `analyze_distortions(p)` builds all augmented_dfs (once each). Subsequent calls at different `p` values are pure row extractions — O(1) per distortion.

The five chained pieces:

1. **Cache** — `_augmented_dfs` dict + `augmented_df(d)` lazy-eval + `pricing_at(p, d)` row extractor.
2. **Analyze collapse** — `analyze_distortion`/`analyze_distortions` reduce to thin wrappers over `pricing_at`.
3. **API simplification** — `calibrate_distortions(coc, p)` and `analyze_distortions(p, distortions=None)` replace the lists-based versions. The current `*2` variants become canonical.
4. **Dataclasses** — `Answer` is bogus; replace with named dataclasses per return type.
5. **Ordered categoricals + renames** — `DISTORTION_ORDER` and `PRICING_STAT_ORDER` baked in; rename `dists`→`distortions`, `dist_ans`→`distortion_df`, `limits`→`_limits`.

## Execution order within Sub-project D

Each step below can be one logical commit (5 commits total during review), or squashed at the end. Recommend: keep separate for review granularity, optionally squash before merge.

### Step D.1 — `augmented_df` lazy-eval cache (planning step 9)

The keystone. Everything else gets simpler once this lands.

**Changes:**

- Add `self._augmented_dfs: dict[str, pd.DataFrame] = {}` to `Portfolio.__init__`. Reset in `update` (since the underlying density changes).
- Move the current `apply_distortion` body — the actual augmented_df construction — into a private `_build_augmented(self, distortion) -> pd.DataFrame`. Pure: no side effects, returns the frame.
- Replace `apply_distortion(d)`'s body with cache-lookup-or-build:
    ```python
    def apply_distortion(self, distortion, *, view='ask', S_calculation='forwards', efficient=True):
        name = distortion.name if isinstance(distortion, Distortion) else str(distortion)
        if name not in self._augmented_dfs:
            self._augmented_dfs[name] = self._build_augmented(distortion, view=view,
                                                              S_calculation=S_calculation, efficient=efficient)
        return self._augmented_dfs[name]
    ```
   (Drop the `df_in=`, `create_augmented=` parameters; those were for the gradient path which is gone.)
- Add a clean accessor `augmented_df(self, distortion)` as a method (not the bare attribute). Internally identical to `apply_distortion(distortion)` — keep both as aliases for now, or pick one.
- Delete `apply_distortions` (plural). Memory math is fine (5 distortions × 2^16 × ~30 cols × 8 bytes ≈ 80 MB on a 32 GB box; user accepts).
- Add `pricing_at(self, p, distortion) -> pd.DataFrame`: warm-cache + extract L/LR/M/P/PQ/Q/ROE for each line at the asset level corresponding to `p`. The row-extraction logic currently lives in `price` (lifted/linear) and `analyze_distortion` — consolidate it here.
- Decide what to do with the current `self.augmented_df` attribute (the "most recently applied" pointer). Two options: (a) drop entirely, (b) keep as a property that returns `self._augmented_dfs[last_applied_name]`. PMIR reads `ans.augmented_dfs[name]` from the `analyze_distortions` return, not `port.augmented_df` directly — so dropping the attr is safe. **Recommended: drop the attr; expose `port.augmented_dfs` as a property returning the dict view.**

**Verification within D.1:**
- `port.apply_distortion(d)` returns the same DataFrame as before; calling it twice returns the cached frame (id is the same).
- `port.augmented_dfs` returns a dict.
- `pricing_at(p, d)` returns a small DataFrame.
- pytest 415 green.

### Step D.2 — Collapse `analyze_distortion(s)` onto the cache (planning steps 6 + 7 combined)

With cache + `pricing_at` in place, the analyze methods reduce dramatically.

**Changes:**

- `analyze_distortion(self, distortion, p)` → ~20 LOC. Returns an `AnalyzeDistortionResult` dataclass (see D.3) containing the per-line pricing DataFrame and a few audit values (`a`, `LR`, `ROE`). Drops `add_comps` and `plot` parameters entirely (they were already no-op stubs after Sub-project A).
- `analyze_distortions(self, p, distortions=None)` → ~15 LOC. For each `d` in `(distortions or self.distortions)`, ensure `augmented_df(d)` is cached, extract `pricing_at(p, d)`, concat keyed by distortion name. Returns an `AnalyzeDistortionsResult` dataclass (D.3). **Replaces both the current list-based `analyze_distortions` AND `analyze_distortions2`** — single-`p`, optional distortion dict.
- `calibrate_distortions(self, coc, p)` → ~20 LOC. Calibrates the standard distortion set (`ccoc`, `ph`, `wang`, `dual`, `tvar`) into `self.distortions` (renamed in D.5). Replaces both the current `calibrate_distortions(LRs=, COCs=, ROEs=, As=, Ps=, ...)` and `calibrate_distortions2(coc, reg_p)`.
- **Update PMIR call sites:**
    - `pmir_package/src/pmir/dmc_p2p.py:69`: `port.analyze_distortions2(p=1, dists=dists)` → `port.analyze_distortions(p=1, distortions=dists)`. (The `dists`→`distortions` rename happens in D.5, but you can do it here in the same commit.)
    - Other PMIR call sites that pass `add_comps=False, efficient=False` to `analyze_distortion` / `analyze_distortions`: drop those kwargs.
    - PMIR call sites that pass `COCs=[coc], Ps=[reg_p]` to `calibrate_distortions`: rewrite as `calibrate_distortions(coc, reg_p)`.

**Verification within D.2:**
- `port.analyze_distortions(p=1)` returns a result dataclass; `.pricing_df` is a frame.
- Numerical values for representative cases unchanged before vs after.
- PMIR call sites work.
- pytest 415 green.

### Step D.3 — `Answer` → named dataclasses (planning step 8)

Define dataclasses for each return type. Define in `aggregate/utilities.py` (or a new `aggregate/results.py` — recommend `results.py` since this is a fresh module that future return types can extend).

**Dataclasses to define:**

- `AnalyzeDistortionResult(distortion, pricing_df, audit_df)` — single distortion.
- `AnalyzeDistortionsResult(distortions, pricing_df, augmented_dfs)` — multi-distortion. `augmented_dfs` is the cache dict (or a snapshot); `pricing_df` is the concat.
- `PricingResult(df, price, price_dict, a_reg, reg_p)` — for `price()`. Replaces the current `namedtuple('PricingResult', ...)` defined inline in `price()`. Promote to the dataclass module.
- `PricingBoundsResult(bounds, allocs, stats, comp, allocs_slow, p_star)` — for `pricing_bounds()`.
- `BlendResult(...)` if `calibrate_blends` (now in extensions/portfolio_pir) needs it. May not — `calibrate_blends` returns a dict today.

**Sweep:**

- `rg "Answer\\(" aggregate/ tests/ -n` — find every instantiation.
- Replace `Answer(field1=..., field2=...)` with the appropriate `<X>Result(field1=..., field2=...)`. Same field names, just typed.
- Drop the `Answer` class from `aggregate/utilities.py` (after the sweep is complete; verify no remaining references).
- PMIR call sites that read `ans.augmented_dfs[name]` continue to work — the field name is preserved on `AnalyzeDistortionsResult`.

**Verification within D.3:**
- `rg "Answer\\(" aggregate/` returns nothing.
- pytest 415 green.
- PMIR call sites work.

### Step D.4 — Ordered categoricals (planning step 10)

Bake "the right order" into the data with `pd.CategoricalDtype(..., ordered=True)`.

**In `aggregate/spectral.py`:**

```python
DISTORTION_ORDER = [
    'ccoc', 'ph', 'wang', 'dual', 'tvar', 'wtdtvar',
    'lep', 'ly', 'clin', 'tt', 'cll', 'bitvar', 'blend',
]
DISTORTION_DTYPE = pd.CategoricalDtype(categories=DISTORTION_ORDER, ordered=True)
```

**In `aggregate/portfolio.py`:**

```python
PRICING_STAT_ORDER = ['L', 'LR', 'M', 'P', 'PQ', 'Q', 'ROE']
# L=loss, LR=loss ratio, M=margin, P=premium, PQ=P:Q (leverage),
# Q=capital/equity, ROE=return on equity
PRICING_STAT_DTYPE = pd.CategoricalDtype(categories=PRICING_STAT_ORDER, ordered=True)
```

**Apply at frame-construction:**

- `distortion_df.index` (the renamed `dist_ans` — see D.5) cast to `DISTORTION_DTYPE`.
- `pricing_at` output columns (or MultiIndex level) cast to `PRICING_STAT_DTYPE`.
- `analyze_distortion(s)` result frames (the `pricing_df` in the dataclasses) cast appropriately.

**Note on Q vs equity vs capital:** the terms drift in actuarial literature. Sticking with `Q` throughout (per PIR convention) is the lesser evil.

**Verification within D.4:**
- `port.distortion_df.index.dtype` is the ordered categorical.
- Sorting `port.distortion_df` by index produces the canonical order.
- pytest 415 green.

### Step D.5 — Renames (planning step 11)

Mechanical sweep, one commit.

| current                | new                | scope of touch |
|:-----------------------|:-------------------|:---------------|
| `self.dists`           | `self.distortions` | aggregate/, tests/, pmir_package/, current/*.ipynb |
| `self.dist_ans`        | `self.distortion_df` (merge with the current `distortion_df` property — column trim per master plan) | aggregate/, tests/ |
| `self.limits()` method | `self._limits()`   | aggregate/portfolio.py only (internal) |

**Tasks:**

1. `rg "\\.dists\\b" aggregate/ tests/` — find every read. Replace with `.distortions`.
2. `rg "\\.dist_ans\\b" aggregate/ tests/` — find every read. Replace with `.distortion_df`.
3. Merge the existing `distortion_df` property (a tidy view of `dist_ans`) into the renamed `dist_ans` itself: build the frame directly with the trimmed columns the property used to produce. Drop the now-redundant property.
4. `rg "self\\.limits\\(" aggregate/portfolio.py` — replace with `self._limits(`.
5. **Update PMIR.** Sweep `pmir_package/src/pmir/*.py` and `python/current/*.ipynb` for `\\.dists\\b` reads.

**Verification within D.5:**
- `rg "\\.dists\\b|\\.dist_ans\\b" aggregate/ tests/ pmir_package/` returns nothing.
- pytest 415 green.
- PMIR smoke test on a representative notebook.

## Overall verification (end of Sub-project D)

- `uv run pytest` → 415 + 6 PEG regression tests passed (the regression test gains three new assertions in this sub-project; see below).
- **PEG regression check:** `uv run pytest tests/test_portfolio_peg_regression.py -v` → 6 passed. This sub-project changes the API; the test fixture is updated to the new signatures but the baseline JSON is unchanged.
- Visual `test_suite` HTML report green.
- PMIR notebooks remain runnable: pick one (e.g., `pricing-from-the-insureds-perspective.ipynb` or `UD-NA-example.ipynb`), run the cells touching Portfolio, confirm numerical results match a pre-D capture.
- `Portfolio.augmented_dfs` returns a dict; building once + reading twice doesn't recompute (verified by the new `test_pricing_at_caches` assertion).
- `Portfolio.analyze_distortions(p=1)` returns an `AnalyzeDistortionsResult` with the expected fields.

### PEG regression test changes in this sub-project

Update `tests/peg.py` and `tests/test_portfolio_peg_regression.py` alongside the API change. **The baseline JSON itself does not change** — same numbers, new access pattern.

**In `tests/peg.py`** — flip the calibration call:

```python
# was:
port.calibrate_distortions(COCs=[coc], Ps=[p])
# becomes:
port.calibrate_distortions(coc, p)
```

**In the test fixture** — flip the analyze call and the dist-dict reference:

```python
# was:
ad = port.analyze_distortions2(.995)
# becomes:
ad = port.analyze_distortions(.995)         # singular API now takes single p

# and where the test extracts per-distortion pricing, the dict lookup
# follows the rename:
# was: port.dists[name]
# becomes: port.distortions[name]
```

**Add three new tests** that exercise `pricing_at` (this is the "appropriate helpers" the user mentioned — `pricing_at` is the helper, and the tests below pin its contract):

```python
def test_pricing_at_matches_baseline(peg):
    """pricing_at(p, d) reproduces the baseline pricing readout
    that analyze_distortions also produces."""
    port, _ = peg
    for dname, expected_lines in BASELINE['pricing'].items():
        d = port.distortions[dname]
        actual_df = port.pricing_at(0.995, d)
        for line, expected_stats in expected_lines.items():
            for stat, expected_val in expected_stats.items():
                actual_val = actual_df.loc[line, stat]
                assert np.isclose(actual_val, expected_val, rtol=1e-8), \
                    f'{dname}.{line}.{stat}: {actual_val} vs {expected_val}'

def test_pricing_at_uses_cache(peg):
    """The augmented_df cache is real: a second call returns the same
    object, not a recomputed frame."""
    port, _ = peg
    d = port.distortions['ph']
    frame_a = port.augmented_df(d)
    frame_b = port.augmented_df(d)
    assert frame_a is frame_b, 'augmented_df should return cached frame on second call'

def test_pricing_at_accepts_p_or_a(peg):
    """pricing_at takes either a probability (<=1) or an asset level (>1).
    Both forms produce the same row."""
    port, _ = peg
    d = port.distortions['ph']
    p = 0.995
    a = port.q(p)
    df_p = port.pricing_at(p, d)
    df_a = port.pricing_at(a, d)
    pd.testing.assert_frame_equal(df_p, df_a, rtol=1e-12)
```

These three pin the three things the cache-and-extractor pipeline must guarantee: (1) numerically equivalent to the analyze path, (2) actually caches, (3) accepts both p and a inputs uniformly.

## Commit

Five logical commits during review (one per step D.1 through D.5), or one squashed commit at the end. User preference. Recommend keeping them separate during review for diff clarity; can squash at merge.

**Commit message draft (squashed version):**

```
Portfolio refactor D: distortion pricing redesign

Six related changes that together produce a clean distortion-pricing
pipeline:

1. augmented_df becomes a lazy-eval method backed by an _augmented_dfs
   cache keyed by distortion name. apply_distortion routes through the
   cache; apply_distortions (plural) gone. New pricing_at(p, distortion)
   extractor replaces the row-extraction logic scattered across price
   and analyze_distortion.

2. analyze_distortion(s) collapse to pricing_at + concat over the
   distortion dict. The previous ~250-LOC body becomes ~20 LOC.

3. calibrate_distortions / analyze_distortions adopt single-coc / single-p
   signatures (formerly the *2 variants). List-based versions removed.
   PMIR call sites updated.

4. Answer class replaced with named dataclasses per return type:
   AnalyzeDistortionResult, AnalyzeDistortionsResult, PricingResult,
   PricingBoundsResult.

5. Ordered categoricals added: DISTORTION_ORDER in spectral.py,
   PRICING_STAT_ORDER in portfolio.py. Indices/columns of distortion_df,
   pricing_at results, analyze_distortions results sort correctly
   without manual reordering.

6. Renames: dists→distortions, dist_ans→distortion_df (merged with the
   existing property), limits→_limits.

415/415 pytest green; PMIR smoke-tested; numerical results unchanged.
```

## README bullet draft

```rst
- ``Portfolio`` distortion pricing pipeline rewritten:

    - ``augmented_df(distortion)`` is now a lazy-eval method backed by an
      internal cache keyed by distortion name. ``apply_distortion`` warms
      the cache; ``apply_distortions`` (plural) removed. New
      ``pricing_at(p, distortion)`` extractor pulls the L/LR/M/P/PQ/Q/ROE
      row from the cached frame.
    - ``analyze_distortion`` and ``analyze_distortions`` collapse to
      ``pricing_at`` + concat over the distortion dict (~20 LOC each;
      previously ~250 LOC).
    - ``calibrate_distortions(coc, p)`` and
      ``analyze_distortions(p, distortions=None)`` replace the prior
      LRs/COCs/ROEs/As/Ps list-based signatures.
    - Generic ``Answer`` return type replaced with named dataclasses
      (``AnalyzeDistortionResult``, ``AnalyzeDistortionsResult``,
      ``PricingResult``, ``PricingBoundsResult``).
    - ``DISTORTION_ORDER`` and ``PRICING_STAT_ORDER`` ordered categoricals
      ensure consistent index/column ordering in pricing exhibits.
    - Renamed ``dists`` → ``distortions``, ``dist_ans`` →
      ``distortion_df``, ``limits`` → ``_limits``.
```

## Post-conditions

Portfolio's distortion-pricing surface is clean: one cache, one extractor, one dataclass family, one canonical signature pattern, one canonical ordering convention. PMIR happy. Sub-project E (stats consolidation) is unblocked.

## Cross-cutting reminders

- `UV_LINK_MODE=copy` is set. Don't override.
- Run `uv run pytest`, not bare `pytest`.
- PowerShell shell: use `rg` directly via Bash; no awk/sed.
- Don't prefix Bash commands with `cd`.
- Update `README.rst` at sub-project close. Bump alpha version.
- Don't build docs in a verification cycle.
- Do NOT commit unless the user explicitly says so.
- Pre-existing visual `test_suite` failure: `Cc.Freq20`. Ignore.
