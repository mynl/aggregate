# Plan: signed severity bounded window, kill the `int(inf)` overflow and the half applied layer

> **Status: EXECUTED at `1.0.0a309`, 2026-08-21.** A focused robustness fix in the bucket window sizer plus an honest metadata fix for a layer clause on a signed severity. Touches core grid sizing (`_bucket_window.py`, `_aggregate.py`, `_severity.py`), so the regression bar is "ordinary aggregates byte for byte unchanged". See the execution notes at the end for what landed and the three places the execution went beyond the plan's letter.
>
> **Rewritten 2026-08-21.** The original draft predates `[Reflected-Loss-Severity]` at `1.0.0a230`, and its line references, its reading of the build dispatch, and its central open decision are all superseded. The diagnostic work behind the rewrite also produced two general validation findings, which are now their own plan, `dev/done/plan-validation-punchup.md`. This plan is the bug fix only.

## Reproduction

```python
build('agg NT 50 claims 25000 xs 0 ssev -lognorm 200 cv 10 + 180 mixed ig .4', bs=5)
# OverflowError: cannot convert float infinity to integer
#   _bucket_window.py:667  need = int(np.ceil(np.log2(max(span / bs + 1.0, 1.0)))) if span > 0 else 0
#   reached from _bucket_window.py:760 (the bounded_small row) via :691 (_row)
```

Confirmed 2026-08-21 at `1.0.0a308`. The severity's own state, read off the built object:

```
signed True   limit 25000.0   attachment 0.0   detachment 25000.0
bounded True  tail_class bounded   fz.support() (-inf, 180.0)
_bounded_severity_window(1 - 1e-12) -> (-inf, 4980600.05)
```

## What actually happens

Three defects, in increasing depth. The first two are exactly as originally diagnosed. The third has narrowed since the draft.

### The sizer overflows on a non finite window

`_bucket_window.py:667`, inside `_size`, computes `need = int(np.ceil(np.log2(max(span / bs + 1.0, 1.0))))` with `span = x_hi - x0` and `x0 = floor(x_lo / bs) * bs`. A non finite `x_lo` gives `x0 = -inf`, `span = +inf`, and `int(inf)` raises. Two aggravating details:

* The user pinned `bs=5`, so `need` is discarded three lines later (`if bs_in > 0: l2 = log2`). On the reported path the crash is in arithmetic whose result is never read.
* The existing `if span > 0 else 0` guard does not catch it, because `inf > 0` is `True`. Even where `need` **is** used, a non finite span should never reach `int()`.

### The bounded window claims a window it does not have

`Aggregate._bounded_severity_window`, `_aggregate.py:6074`. The gate at `:6092` is `all(_tail._severity_bounded(s) for s in self.sevs)`, which is `True` here because the recorded `limit = 25000` makes `tail_class == BOUNDED`. The lower edge at `:6097` is `float(s.fz.support()[0]) if signed else 0.0`, and the reflect patch gives `fz.support() = (180 - inf, 180 - 0) = (-inf, 180)`. The return at `:6117` is therefore `(-inf, 4.98e6)`, a bounded window with an unbounded edge, which is what `_size` chokes on.

The same hazard for the **upper** edge is already documented and patched on the splice path (`_severity.py:1537`, "`fz.support()` reporting the underlying (0, inf) ... `_bounded_severity_window` reads an infinite upper edge"). This is its lower edge, signed severity twin, and it is unguarded.

### The layer is recorded but never applied on a signed severity

**This is the part that moved.** The original draft read the build dispatch as branching on `sev_reflect` first, so that a reflected base took the no clamp path under either `sev` or `ssev`. That is no longer true. `dev/done/plan-reflected-loss-severity.md` landed at `1.0.0a230` and the dispatch now branches on `signed` (`_severity.py:1263` to `:1281`):

```python
self._apply_lb_ub()
if not self.signed:
    self._validate_moments()
if self.sev_reflect:
    self._apply_reflect()
    if not self.signed:
        self._warn_reflected_clamp()
if self.signed:
    self._apply_signed()               # identity layering, no clamp
else:
    self._compute_attachment_probs()
    self._apply_layer_attachment()     # the clamp lives here
```

So `sev -lognorm 200 cv 10 + 180` is now an ordinary loss: it clamps at zero and says so through `ReflectedSeverityClampWarning`. What survives is the narrower case, a layer clause on a genuinely signed severity (`ssev`, or a `dsev` carrying negative atoms). `_apply_signed` skips the clamp by design, but `__init__` has already recorded `self.limit = 25000`, so `bounded`, `tail_class` and `detachment` all report a bound the law does not have. The layer is **half applied**: the bookkeeping is set, the transform is skipped, and nothing tells the user.

The math is worth stating, because it explains why the reported case is not merely unsupported but contradictory. A layer `y xs a` is `min(y, max(X - a, 0))`. On `X = 180 - lognormal`, `25000 xs 0` is `min(25000, max(X, 0)) = [0, 180]`, which annihilates the negative half, that is, the entire reason for writing `ssev`. A **negative** attachment (`25000 xs -1000`, keeping the signed region down to `-1000`) is the meaningful general case, and implementing it is the deferred feature, not this bug fix.

For completeness, fixing the window alone does not rescue the reported program. The raw `180 - lognorm(cv 10)` has sigma about 2.15, so over 50 claims it reaches roughly 14.5 million buckets and builds defective (pmf deficit 0.977, negative reach clipped). That is honest, and it is the subject of `dev/done/plan-validation-punchup.md`, not of this plan.

## The fix

### `[Non-Finite-Window-Guard]`

`_bucket_window.py`, `_size`, around `:659` to `:682`. Two independent changes, both wanted:

* Skip the `need` computation entirely when `bs_in > 0`. It is unused on that path (`l2 = log2`), so the reported crash disappears with no behavior change anywhere.
* Treat a non finite `span`, `bs`, or ratio as "needs more than the cap" rather than calling `int()` on it, so a non finite window arriving from any other method falls through to the coarsen branch instead of raising. Belt and suspenders for the unpinned path.

### `[Bounded-Window-Finite-Edge]`

`_aggregate.py`, `_bounded_severity_window`, after `s_max, s_min = max(s_his), min(s_los)` at `:6100`. A severity whose computed `s_min` or `s_max` is not finite is not boundedly windowable: return `None` so the caller falls back to the moment window. The documented contract already tolerates `None` ("the caller selects this method only when it is at least as tight as the moment window"), and this mirrors the intent of the existing upper edge guard on the splice path.

### `[Signed-Layer-Clause-Ignored]`

`_severity.py`, in or immediately after `_apply_signed`. A layer clause on a signed severity is a clause the declaration cannot use, which is a shape the house already has a canonical answer for: `IgnoredDecLClauseWarning`, the warning a pure `agg` raises for the ceded premium, reinstatement and variable rating clauses it has no premium context to activate. Follow that precedent. Clear `limit`, `attachment` and `detachment` back to their unlayered values so `bounded` and `tail_class` tell the truth, and warn once naming the dropped clause and pointing at the two real alternatives: drop the layer, or use plain `sev` if the clamp is what was wanted.

With the metadata honest, `[Bounded-Window-Finite-Edge]` never fires on this program, because `_severity_bounded` is `False`, the bounded row is not built, and the moment window is used. The guard stays anyway, because it is the correct behavior for any other route to a non finite edge.

**One confirm before executing.** Warning and dropping is the recommendation, on the `IgnoredDecLClauseWarning` precedent. The alternative is a clean build error ("layering a signed severity is not supported; drop the layer, or use `sev` to clamp"), which is more forceful but invents a second convention for a situation the house already has one for. Implementing the clamp properly, `min(y, max(X - a, 0))` on a signed base with an arbitrary and possibly negative attachment, is a real feature and stays deferred as the successor to `dev/plan-negative-x-agg.md` section 6.

## Considered and rejected here

* **Re-classing `bounded` and `tail_class` globally** so a signed severity with non finite reflected support is never `BOUNDED`. Correct in principle, too broad for a bug fix: `tail_class` feeds `tail.py`, plotting and the descriptions. `[Signed-Layer-Clause-Ignored]` gets the metadata right at the source, which is where the lie is introduced.
* **Escalating the 97.7% deficit case to a hard refusal.** The `DefectiveDistributionWarning` and the negative reach clip already fire. Whether a severe deficit should ever raise is a policy question, and it belongs with the validation work, not here.

## Order of work

1. `[Signed-Layer-Clause-Ignored]`, the root cause: the metadata stops lying and the bounded row is never built for this program.
2. `[Bounded-Window-Finite-Edge]`, the correctness fix at the right layer.
3. `[Non-Finite-Window-Guard]`, defense in depth.
4. Full fast suite, with attention to the bucket sizing and signed / negative x suites.

## Tests

Extend `tests/test_bucket_sizing.py` and `tests/test_negative_x.py`:

* **No crash.** The reproduction builds without `OverflowError`, asserting the specific failure is gone.
* **`[Signed-Layer-Clause-Ignored]`.** A layer clause on an `ssev` warns `IgnoredDecLClauseWarning`, and the built severity reports `limit == inf`, `bounded is False`, and a `tail_class` other than `BOUNDED`. The same clause on a non signed severity is unchanged and still clamps.
* **`[Bounded-Window-Finite-Edge]`.** `_bounded_severity_window` returns `None` for a signed severity with a non finite `fz.support()` edge, and still returns a finite window for a genuine non signed bounded layer.
* **`[Non-Finite-Window-Guard]`.** A synthetic `_size(x_lo=-inf, x_hi=finite, bs_in > 0)` returns `l2 == log2` and raises nothing; the same call with `bs_in == 0` coarsens instead of raising.
* **No regression.** A normal bounded layer (`1000 xs 0 sev ...`, not signed) still selects `bounded_small` with the same `_bs_window_df` rows.

## Housekeeping

Plan based code change, so bump `1.0.0a*` in `pyproject.toml` and add the `CHANGELOG.md` section. Tick `[Signed-Bounded-Window]` in `dev/TODO.md`, and drop its stale "needs the author's D1 pick" note, since that decision moved with the dispatch. Move this plan to `dev/done/` at close.

## Execution notes, `1.0.0a309`, 2026-08-21

Executed in the plan's order. The author confirmed the open question the same day: **warn and drop**, on the `IgnoredDecLClauseWarning` precedent, not a hard build error.

### What landed

**`[Signed-Layer-Clause-Ignored]`**, `_severity.py`. New `Severity._drop_layer_clause`, called from `__init__` immediately before `_apply_signed`. It resets `limit`, `attachment`, `exp_attachment` and `detachment` to their unlayered values and warns once through `IgnoredDecLClauseWarning`, naming the dropped clause and the two alternatives. `_apply_signed`'s docstring cross references it.

**`[Bounded-Window-Finite-Edge]`**, `_aggregate.py`. `_bounded_severity_window` returns `None` when `s_max` or `s_min` is not finite, with the reason in the docstring's Returns section.

**`[Non-Finite-Window-Guard]`**, `_bucket_window.py`. `_size` no longer computes `need` when `bs` is pinned, and the remaining computation moved into a module level `_need_log2`, which returns `inf` for a non-finite span, step or ratio.

### Three places the execution went beyond the plan's letter

1. **The unlayered `limit` is not always `inf`.** A histogram `_build` sets `limit = min(exp_limit, xs.max())`, so with no clause it reports the atom support max, and a signed `dsev [-2 5]` reset to `inf` would replace one lie with another. `_drop_layer_clause` restores `fz.support()[1]` for a histogram kind and `inf` otherwise. Detecting the clause needs the limit **as declared**, which `self.limit` can no longer supply after that truncation, so `__init__` records a new private `_exp_limit` next to `exp_attachment`.

2. **`Aggregate._severity_high_estimate` also capped a signed reach by the dropped clause** (`min(hi, self.limit.max())`), and on a signed book that estimate sizes the grid. The author ruled it in with the main confirm. `Aggregate.limit` itself is untouched: it records the layer as declared, which is right, and the guard is `not self._signed_severity()`. Only the buggy combination is affected, since an unlayered signed aggregate already carries `limit = inf`.

3. **A non-finite window is rejected, not coarsened.** The plan asked for the non-finite case to "fall through to the coarsen branch instead of raising", but coarsening calls `round_bucket(span / N0)`, and `round_bucket` refuses `inf` by design, so that path trades one raise for another. Instead `_row` records a method whose window has an infinite edge as `applies=False` with a `non-finite window (rejected)` note, the `exact_discrete` unreachable-support precedent, and selection falls through to another method. `_size` keeps a matching non-raising answer (the cap grid) for a direct call, and the `bounded_small` selection test now also reads `applies`, which no row reaches today but which is what the flag means.

### Verification

Full gate, `pytest -m 'slow or not slow'`. Twenty two new cases: five in `tests/test_negative_x.py` for the dropped clause (`ssev`, auto-signed `dsev`, the non-signed contrast that still clamps, the message content, and the silent unlayered case) and seven plus a nine-way parametrization in `tests/test_bucket_sizing.py` for the reproduction, the two window guards, `_need_log2`, the pinned-`bs` path and the no-regression `bounded_small` selection. Two DecL programs mirrored into `decl-testers.agg` section P&L (`PnL.SLay`, `PnL.DLay`), both round-tripping.

Five failures in `tests/test_agg_libraries.py` and `tests/test_library_entries.py` are **pre-existing and unrelated**: the author's uncommitted `src/aggregate/agg/library.agg` edits (`ISOMixedExponential` renamed to `CommAutoMixedExponential`, new `LayerPicksMED`) have not reached the baselines or the canonical layout yet.

The reproduction now builds and reports itself defective, exactly as the plan predicted, which is `dev/done/plan-validation-punchup.md`'s subject rather than this plan's.
