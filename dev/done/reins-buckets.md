# reins-buckets — selectable net/ceded rebucketing + layer-order validation

**Target version: 1.0.0a18** (execute first — quick win, isolated to the reins
*calc* path; reporting rationalization is the separate `reins-reporting.md`).

## Context

The reinsurance subsystem is in the same pre-rationalization state the
`Aggregate`/`Portfolio` stats reporting was in before the report refactor. This
document covers the first of two problems: **how net/ceded distributions get
rebucketed onto the model grid**.

Today `Aggregate._apply_reins_work` (`src/aggregate/distributions.py:3391`)
rebuckets via `groupby(loss_net).p_subject.sum()` → `interp1d` on the resulting
CDF → resample on `self.xs` → `np.diff`, plus two `len(...)==1` special cases.
This is effectively a *third*, undocumented scheme (CDF linear interpolation)
that is neither "nearest bucket" nor a mass-preserving linear spread, and it
does not cleanly preserve the first moment. It is also not selectable.

Separately, `make_ceder_netter` (`src/aggregate/utilities.py:224`) assumes reins
layers are entered ascending by attachment and non-overlapping (it tracks a
running `base`/`h`) but performs **no validation** — out-of-order input silently
produces wrong cessions.

## Goal

1. Make rebucketing **explicit and selectable** via a `reins_bucket ∈
   {'linear','nearest'}` switch (default `'linear'`).
2. Add **hard-error validation** of multi-layer ordering.

Both are quick, well-isolated changes to the reins calculation path.

## Design

### 1. `reins_bucket` switch

- New module constant `REINS_BUCKET_DEFAULT = 'linear'` in
  `src/aggregate/constants.py` (add to `__all__`), alongside `VALIDATION_NOISE`.
- `Aggregate.__init__`: set `self.reins_bucket = REINS_BUCKET_DEFAULT`. Accept an
  optional `reins_bucket=None` kwarg that falls back to the constant (lets
  `build` thread it through if desired; not a DecL concept).
- Property + validating setter on `Aggregate`, mirroring
  `Portfolio.allocation_method` (`src/aggregate/portfolio.py:869`): rejects
  anything not in `{'linear','nearest'}` with a clear `ValueError`; on change,
  clears cached reins frames (`self._reinsurance_df = None`,
  `self._reinsurance_audit_df = None`, `self._reinsurance_report_df = None`).
  Docstring notes reinsurance is baked in at `update`, so a re-`update()` is
  required for a post-build change to take effect.
- Accept `reins_bucket=` as an `update` / `update_work` kwarg for convenience
  (mirrors how `sev_calc` is threaded), defaulting to the current attribute.

### 2. Rewrite the rebucket core in `_apply_reins_work`

Replace the groupby / `interp1d` / `diff` / `len==1` block (≈ lines 3414–3440)
with a vectorized scatter onto the uniform grid. The grid is
`self.xs == bs * arange`, `xs[0] == 0`, so the grid index of a value `v` is
`v / bs`.

```python
v = netter(xs)            # (and ceder(xs)); off-grid target values
# clip targets into [0, xs[-1]] so overflow piles at the top bucket
if self.reins_bucket == 'nearest':
    idx = np.clip(np.round(v / bs).astype(int), 0, N - 1)
    p_net = np.zeros(N)
    np.add.at(p_net, idx, p_subject)
else:  # 'linear' — mass split preserves E[X] exactly
    k = np.clip(np.floor(v / bs).astype(int), 0, N - 1)
    f = np.clip(v / bs - k, 0.0, 1.0)
    kp1 = np.clip(k + 1, 0, N - 1)
    p_net = np.zeros(N)
    np.add.at(p_net, k,   p_subject * (1 - f))
    np.add.at(p_net, kp1, p_subject * f)
```

- Apply the same to the ceded target. **Mass identity:** `linear` splits each
  point's probability into two parts that sum to it, and the weighted bucket
  positions average back to `v`, so the first moment is exact; `nearest` keeps
  full mass at one bucket (≤ `bs/2` positional bias). Both preserve `Σp == 1`.
- Keep the assembled `reins_df` column names **identical** (`loss`, `p_subject`,
  `F_subject`, `loss_net`, `loss_ceded`, `F_net`, `F_ceded`, `p_net`, `p_ceded`)
  so `reinsurance_df` / audit / plot consumers are untouched.
  `F_net = p_net.cumsum()`, `F_ceded = p_ceded.cumsum()`.
- The degenerate "everything ceded → net is 0" case is handled naturally by the
  scatter (all mass lands in bucket 0); **drop** the two `len(sn|sc)==1`
  branches.
- Document the top-of-grid overflow behavior in the Notes section: values at or
  beyond `xs[-1]` pile into the last bucket — the same failure mode as an
  aggregate deficit, surfaced the same way.

### 3. Multi-layer ordering: hard error

Add a small validator `_validate_reins_layers(reins_list)` in
`src/aggregate/utilities.py`, called at the **top of `make_ceder_netter`** — the
single choke point used by every reins path (`_apply_reins_work`, the audit
work, etc.). Rules over `[(share, limit, attach), ...]`:

- attachments must be **non-decreasing**;
- layers must **not overlap**: `attach_{i+1} >= attach_i + limit_i - tol`
  (gaps are allowed — represent them with `0 po L xs A`; `share == 0` entries
  still must respect ordering).

Raise `ValueError` with an explicit, actionable message: enter layers bottom-up
(lowest attachment first); use `0 po L xs A` to represent a gap. Single-layer
calls (`[(s, y, a)]`, used by the per-layer audit path) are trivially valid.

## Files

- `src/aggregate/constants.py` — add `REINS_BUCKET_DEFAULT`.
- `src/aggregate/distributions.py` — `__init__` attribute + kwarg, property +
  validating setter, rewrite `_apply_reins_work`, thread `reins_bucket` through
  `update_work`.
- `src/aggregate/utilities.py` — `_validate_reins_layers`, called in
  `make_ceder_netter`.

## Verification

- New focused test `tests/test_reins_buckets.py`: on an excess layer with an
  **off-grid** attachment, assert
  - `linear` preserves ceded + net mean to `VALIDATION_NOISE` (mass-split
    identity), `nearest` mean within `bs/2`;
  - mass sums to 1 for both methods;
  - `p_net + p_ceded == p_subject` (net + ceded reconstructs subject).
- Validator tests: out-of-order and overlapping layer lists raise `ValueError`;
  a gap expressed via `0 po L xs A` is accepted.
- If any test uses DecL, append the program(s) to
  `src/aggregate/agg/test_decl.agg` under the matching section.
- `uv run pytest` green (set `UV_LINK_MODE=copy`).
- Smoke: build one of the example ports with each `reins_bucket` setting and
  eyeball `reinsurance_df` means.

## Close-out

- Bump `pyproject.toml` to `1.0.0a18`.
- README.rst bullets: the switch + default, the rebucket rewrite (linear vs
  nearest, first-moment guarantee), and the layer-ordering hard error.
