# Plan: consolidate fuzz removal into one vectorized utility

## Goal

Replace the scattered, hand-rolled fuzz-removal idioms with a single vectorized
utility. Two flavours are duplicated across the code:

- an **elementwise `DataFrame.map` lambda** (`Portfolio.remove_fuzz` and a verbatim
  copy inside `Aggregate.density_df`) — slow, one Python call per cell;
- a handful of `np.where(np.abs(x) < eps, 0.0, x)` array copies feeding
  `xsden_to_mwrangler`, with inconsistent thresholds.

Unify on one helper using the fast form:

```python
df.mask(df.abs() < eps, 0.0)        # DataFrame (float cols)
np.where(np.abs(a) < eps, 0.0, a)   # ndarray
```

**This is primarily a consolidation change** (one helper, no scattered copies),
with a modest performance win at the two `DataFrame.map` sites. It must not move
any number except the one deliberate `1e-16 → eps` fix at the MMSE site — see
below.

## The new utility (utilities.py)

A single transformer dispatching on input type, default threshold = machine eps,
explicit `eps=` for callers that need a looser tolerance:

```python
def remove_fuzz(data, eps=None):
    """Zero entries with ``|x| < eps`` (machine-epsilon FFT/round-off noise).

    Vectorized. Accepts an ndarray or a DataFrame; for a DataFrame only the
    float64 columns are touched and a new frame is returned (callers that need
    in-place semantics assign the result back -- see the Portfolio note). Two-
    sided: large negatives are preserved (unlike a one-sided ``x < eps`` clip).

    Parameters
    ----------
    data : numpy.ndarray | pandas.DataFrame
    eps : float, optional
        Threshold; defaults to ``np.finfo(float).eps`` (~2.22e-16).

    Returns
    -------
    Same type as ``data`` (a copy; the input is never mutated).
    """
```

- ndarray branch: `np.where(np.abs(data) < eps, 0.0, data)` (returns a copy).
- DataFrame branch: select `float64` columns, `cols.mask(cols.abs() < eps, 0.0)`,
  write back into a copy of the frame, return it. Non-float columns untouched.
- `eps=None → np.finfo(float).eps`.

Optionally promote the default to a named constant `FUZZ_EPS = np.finfo(float).eps`
in `constants.py` (consistent with the tight-noise-threshold convention). Low
priority; the `eps=None` default is sufficient.

## Call-site conversions

| # | Site | Current | After |
|---|---|---|---|
| 1 | `portfolio.py` `Portfolio.remove_fuzz` (~754) | `.map(lambda…)` over float cols, **in place** | `remove_fuzz(df)` written back **in place** — see the in-place note below; keep the `self._remove_fuzz or force` gate, the `eps==0 → finfo.eps` default, signature, and logging |
| 2 | `distributions.py` `Aggregate.density_df` (~2096) | inline `.map(lambda…)`, unconditional | `self._density_df = remove_fuzz(self._density_df)` (stays unconditional) |
| 3 | `distributions.py` est moments (~4277) | `np.where(abs<eps,0,…)` on copy | `remove_fuzz(self.agg_density)` |
| 4 | `distributions.py` `_moments` helper (~4370) | `np.where(abs<eps,0,arr)` | `remove_fuzz(arr)` |
| 5 | `portfolio.py` total-stats moments (~579) | `np.where(np.abs(_p)<eps,0.0,_p)` → mwrangler | `remove_fuzz(_p)` — **same idiom as #3, was missing from the original inventory** |
| 6 | `portfolio.py` `update` empirical moments (~2003) | `np.where(np.abs(_p)<eps,0.0,_p)` → mwrangler | `remove_fuzz(_p)` — **same idiom as #3, was missing from the original inventory** |
| 7 | `distributions.py` moment fit / MMSE (~6507) | `np.where(abs<1e-16,0,p)` | `remove_fuzz(p)` — **threshold `1e-16 → eps` (the one intended numeric change)** |
| 8 | `ft.py` `recentering_convolution` (~312) | `df.loc[df.a.abs()<2*eps,'a']=0`, single col | `remove_fuzz(df, eps=2*np.finfo(float).eps)` — looser tolerance preserved via the arg |

Sites 5 and 6 are **character-for-character identical** to site 3 (the same
`np.where(np.abs(_p) < eps, 0.0, _p)` feeding `xsden_to_mwrangler`). They were
absent from the first draft; leaving them un-converted would defeat the
consolidation goal, so they are now in scope.

### The Portfolio in-place write-back (site 1) — correctness-critical

`Portfolio.density_df` is a **plain attribute**, and the current
`remove_fuzz` mutates it **in place** (`df[cols] = …`). Callers such as
`self.remove_fuzz(log='update')` (portfolio.py ~1964) rely on that mutation
hitting `self.density_df`. The new helper **returns a copy**, so a naive
`df = remove_fuzz(df)` would rebind the local and silently discard the de-fuzz —
with **zero test failures**, because fuzz is sub-eps and nothing asserts on it.

The conversion must therefore preserve in-place semantics explicitly:

```python
if self._remove_fuzz or force:
    logger.debug(...)
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = remove_fuzz(df)[float_cols]   # in-place write-back
```

(The `df is None` branch already binds `df = self.density_df` at the top, so the
column assignment writes through to `self.density_df`. Do **not** rebind `df`.)

### Carve-outs (do NOT route through the utility)

- `distributions.py` `Frequency.pmf` (~4922): `dist[dist < eps] = 0`. This is
  **one-sided** — it zeroes 0..eps *and every negative value*, not just
  sub-eps. The two-sided utility would leave large negatives in place. There are
  no legitimate negatives in a frequency pmf, so the existing clip is correct as
  written. **Leave it unchanged.** Add a one-line comment noting it is
  intentionally one-sided and therefore not the shared `remove_fuzz`.
- `distributions.py` reins occ-plot helper (~2294/2296): `s[np.abs(s)<1e-15]=0`
  then `np.where(np.abs(s_values)<1e-15, 0, …)` then `==0 → np.nan`. This is a
  **plot-cosmetic** clip at the looser `1e-15`, on a cumulative survival curve,
  with a follow-on `0 → nan` step so empty buckets drop out of the line. Both
  the threshold and the extra step differ from the shared util. **Leave it
  unchanged**; add a one-line comment marking it a deliberate plot-only carve-out
  (mirrors the `Frequency.pmf` note).

## Behavioural-equivalence notes

- `mask(cond, 0.0)` replaces where `cond` is True with `0.0` — identical to
  `0 if abs(x) < eps else x`. No numeric change at the two DataFrame sites
  (#1, #2).
- The array sites (#3–#6) already used `np.where(abs<eps,0,·)` at exactly `eps`;
  routing them through the same expression is a no-op.
- The **only** intended numeric change is #7 (MMSE), whose threshold tightens
  from `1e-16` to `~2.22e-16` (eps). That is what freeze/check must surface (if
  it surfaces anywhere).
- ft (#8) keeps `2*eps` exactly via the explicit arg → no change there.
- Float-cols-only selection preserved on the DataFrame path, so integer/object
  columns (e.g. index helpers) are untouched as before.

## Verification

1. **`uv run pytest`** — must pass.
2. **Portfolio in-place assertion** (guards the silent-no-op risk on site 1):
   build a Portfolio with `remove_fuzz=True`, inject a known sub-eps value into a
   float column of `port.density_df`, call `port.remove_fuzz(force=True)`, and
   assert the cell is now `0.0` **on the same object** (`port.density_df`, not a
   return value). This must fail if the write-back is wrong.
3. **freeze/check harness** (the behavioural net):
   ```
   python scripts/freeze_knowledge.py freeze --root <durable>   # BEFORE
   # ... do the refactor ...
   python scripts/freeze_knowledge.py check <durable>/<date>    # AFTER
   ```
   Expect **100% all-match**. The MMSE moment-fit path (#7) is not exercised by
   the frozen `describe`/`density_df` surface, so even the `1e-16→eps` change
   should not register; if any object diffs, stop and confirm it traces to that
   site and is an acceptable last-bit shift, not a regression.
4. **Grep guards** — after the change:
   - `rg "\.map\(lambda" src/aggregate/portfolio.py src/aggregate/distributions.py`
     for the fuzz idiom should return nothing;
   - `rg "1e-16" src/aggregate/distributions.py` should no longer show the
     moment-fit line (other `1e-16` uses — integration splits, plot ylim — remain);
   - `rg "np.where\(np.abs\(_p\)" src/aggregate/portfolio.py` should return
     nothing (sites 5, 6 converted).
5. Quick smoke: build a Portfolio and an Aggregate, confirm `remove_fuzz` /
   `density_df` still zero sub-eps cells and the result equals the old lambda
   form on a sample frame.

## Housekeeping (standing rules)

- Bump `pyproject.toml` `1.0.0a*`.
- `CHANGELOG.md`: new section — "Consolidated fuzz removal into a single
  vectorized `utilities.remove_fuzz` (replaces per-cell `DataFrame.map` lambda
  and four `np.where` copies; faster at the two DataFrame sites). Standardized
  threshold on machine epsilon; the moment-fit path's stray `1e-16` is now `eps`.
  `ft.recentering_convolution` retains its `2*eps` tolerance via an explicit
  argument. Plot-cosmetic `1e-15` clip and the one-sided `Frequency.pmf` clip
  left as deliberate carve-outs."
- `dev/TODO.md`: one-line under the appropriate track, marked done with the version.
- Move this plan to `dev/done/plan-remove-fuzz.md` on landing.

## Out of scope

- `Frequency.pmf` one-sided clip and the reins occ-plot `1e-15` cosmetic clip
  (intentional carve-outs, above).
- The `remove_fuzz=True` *flags* threaded through `update()` / `build()` callers
  (`portfolio.py`, `underwriter.py`, `pedagogy.py`) — those are gating switches,
  not implementations; they stay.
- The `spectral.py` warning string (mentions `remove_fuzz=True` in advice).
