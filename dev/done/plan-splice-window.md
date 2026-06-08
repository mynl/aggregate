# Plan — honest `support()` for spliced severities (fixes `_bs_window` crash)

> **Status: ready to execute.** Small, localized fix + one regression case.
> Bumps `1.0.0a*` (code change). Author (prod-claude) executes and commits.

## The bug

Building an aggregate whose severity splices an **unbounded** base family crashes
in window sizing:

```python
build('agg Wind dfreq [0 1] [.3 .7] sev lognorm 40 cv 0.65 splice [1 100]')
# ValueError: Inadmissible value passed to round_bucket, inf
#   (or OverflowError: cannot convert float infinity to integer)
```

Traceback: `update` → `_bs_window` → `_bounded_severity_window` → `_size` →
`round_bucket(inf)`.

## Root cause

`splice [1 100]` records the cap in `sev_lb`/`sev_ub` and applies it by wrapping
`fz.cdf/sf/isf/ppf/pdf` in `_apply_lb_ub` (distributions.py:7843). The severity
therefore correctly reports `bounded == True` (a finite `sev_ub` is one of the
things `tail.classify_severity` treats as bounding).

But `_apply_lb_ub` **does not patch `fz.support()`** — it still returns the
underlying lognorm's `(0, inf)`. The window sizer `_bounded_severity_window`
(distributions.py:6130) is invoked precisely *because* the severity is bounded,
then looks for the finite upper edge only in `s.limit` (the policy layer limit =
`inf` here) and `s.fz.support()[1]` (= `inf`):

```python
hi = s.limit if np.isfinite(s.limit) else float(s.fz.support()[1])   # → inf
```

so `a_hi = n_hi · inf = inf` → `round_bucket(inf)` / `np.log2(inf)→int` blows up.
Pinning `bs`/`log2` does **not** help: `_bs_window` always builds the
`bounded_small` row when the severity is bounded.

**Why untested:** every splice case in the corpus splices a `uniform` base
(`10 * uniform + 5 splice [...]`), which is *already* a bounded scipy family, so
`fz.support()` is finite on its own and the missing `sev_ub` is harmless. The one
lognorm splice (`T.Splice01`) is a bare `sev`, never built into an aggregate, so
`_bs_window` never runs on it. Splicing an *unbounded* family into an aggregate is
the untested path.

## The fix

Make `fz.support()` honest about the splice — the spliced distribution's support
genuinely *is* `[sev_lb, sev_ub]`. This is the same instance-method monkeypatch
the **reflect-shift** path already uses (`_apply_reflect_shift`,
distributions.py:7948: `self.fz.support = lambda ...`), so it follows an
established pattern in the class.

In `_apply_lb_ub` (distributions.py:7843), after the existing five method swaps
(`cdf`, `sf`, `isf`, `ppf`, `pdf`), add:

```python
# Keep support() honest: the spliced distribution lives on [lb, ub].
# (Mirrors the reflect-shift support patch; the method-swap pattern above.)
self.fz.support = lambda _lo=self.sev_lb, _hi=self.sev_ub: (_lo, _hi)
```

Notes:
- Capture `sev_lb`/`sev_ub` as default args (value-bound, not late-bound), matching
  the reflect-shift lambda; call convention is no-arg (`fz.support()`), as scipy
  frozen `support()` takes none.
- The method only runs past its trivial-bounds early-return
  (`if self.sev_lb == 0 and self.sev_ub == np.inf: return`), so unspliced
  severities are untouched.
- **Composition with reflect-shift is correct.** Splice runs first, so when
  `_apply_reflect_shift` reads `Z.support()` (line 7942) it already gets
  `(lb, ub)` and reflects it to `(shift - ub, shift - lo)`.

**No change to `_bounded_severity_window` is needed** — with honest support,
`s.fz.support()[1]` returns `sev_ub` (finite) for a spliced unbounded family, and
the existing `hi = s.limit if finite else support()[1]` does the right thing.

## Blast radius (audited)

Only four sites read `.support()` in the library:

| Site | Effect |
|---|---|
| `_bounded_severity_window` (6147–48) | **fixed** — finite `hi` for spliced unbounded family |
| `__repr__`/display (7814) | signed severities show the correct tighter range (display only) |
| `_apply_reflect_shift` (7942) | composes correctly (see above) |

**Moments do not move.** `_numerical_moms` integrates in isf/quantile space
(distributions.py:7463–76, bounds from `fz.sf`), and analytic moments use `_munp`
— neither reads `fz.support()`. The splice golden moment pins
(`test_splice_suite.py`: `m1/m2/m3`, survival probabilities) are window-invariant.

**Expected baseline movement: none captured.** The window *does* tighten on the
existing uniform-splice aggregate cases (e.g. `[8 12]` splice of `10*uniform+5`:
window top `15 → 12`, a strict accuracy gain — no grid wasted over zero-mass
`[12,15]`), but:
- `test_splice_suite.py` pins moments (window-invariant) — unaffected;
- the meta baseline corpus (`tests/baseline/corpus.py`) has **no** splice cases;
- the `freeze_knowledge` default set is `test_suite.agg` only (splice lives in
  `test_suite2.agg`/`ln.agg`) — not in the freeze set;
- `capture_severity_golden.py` pins severity probes, not aggregate density.

So no current snapshot is expected to move. **Confirm with a full `uv run pytest`**
before committing. If anything *does* move, it will be a density grid on a
splice-agg case and the move is a correctness improvement — recapture that
snapshot deliberately, in this commit, with a one-line note.

## Regression test

Add the missing path — a spliced **unbounded** family built into an aggregate — to
`src/aggregate/agg/test_decl.agg` section **T** (keep `test_decl.agg` in sync with
new pytest DecL):

```
agg T.SpliceUnbounded 5 claims sev lognorm 40 cv .65 splice [1 100] poisson   note{splice on unbounded base; window must be finite}
```

And a focused assertion in `tests/test_splice_suite.py` (or a small new test):

```python
def test_splice_unbounded_base_builds_finite_window():
    """A splice of an unbounded family builds (regression: _bs_window inf crash)."""
    from aggregate import build
    import numpy as np
    a = build('agg SpliceUB 5 claims sev lognorm 40 cv .65 splice [1 100] poisson')
    assert np.isfinite(a.bs) and a.bs > 0
    assert a.sevs[0].fz.support() == (1.0, 100.0)   # honest support
    # all aggregate mass sits inside [0, n_hi * 100]; mean is finite & positive
    assert 0 < a.agg_m < np.inf
```

## Verification

1. The original failing program builds:
   `build('agg Wind dfreq [0 1] [.3 .7] sev lognorm 40 cv 0.65 splice [1 100]')`.
2. `uv run pytest` is green (watch for any unexpected snapshot move per above).
3. `a.sevs[0].fz.support()` reports `(1.0, 100.0)` for the spliced severity.

## Housekeeping

- Bump `1.0.0a*` in `pyproject.toml` (next: `a48`).
- Add a `CHANGELOG.md` `## 1.0.0a48` section: *"Fixed: spliced unbounded severities
  crashed window sizing — `fz.support()` now honestly reports `[sev_lb, sev_ub]`
  (mirrors the reflect-shift support patch). Window of existing uniform-splice
  aggregates tightens slightly (strict accuracy gain)."*
- Mark the item in `dev/TODO.md` if tracked; move this plan to `dev/done/`.
