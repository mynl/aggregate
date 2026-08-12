# [Allocation-Default-Linear] Honor `allocation_method` across the pentagon surface

Drafted 2026-08-12 from an app-side report (Pricing pane, Calibrate at p < 1 on
an unbounded book: `analyze_distortions: skipping ccoc` warning, no ccoc row in
the Allocate subtab, while Calibrate and Evaluate both carry ccoc). Status:
**EXECUTED at 1.0.0a265**, 2026-08-12. Execution notes and the divergences from
the plan as drafted are in section 10.

## 1. The ruling being enforced (and its receipt)

Linear is the default natural allocation method. That ruling is not new: it
landed at **1.0.0a17** ("Portfolio pricing & allocation: pentagon, linear
default, ROE fix"), which flipped `Portfolio.price` to `allocation='linear'`
and created `Portfolio.allocation_method` (default `'linear'`, validating
setter that clears the augmented cache) as **the source of truth**, with
`price(allocation=...)` a one-off override. The docs agree:
`docs/2_aggregate_overview/pipeline-portfolio.rst` (the paragraph at line 181)
calls the move from lifted to linear "the single most important conceptual
evolution in this pipeline".

The bug: the pentagon-era Portfolio surface built during the refactor
hardcodes `allocation='lifted'` as a keyword default and never consults
`allocation_method`. `price` honors the member; `apply_distortion`,
`pricing_at`, `pentagon_at`, `analyze_distortion`, `analyze_distortions`, and
therefore `CalibrationResult.pricing_df` and the `pricing.allocate` exhibit,
do not. The `allocation_method` docstring says it "drives `price` (and
downstream readouts)"; the downstream readouts were never wired.

Author ruling 2026-08-12: linear is the default everywhere, with **no
per-distortion special casing** (no linear-for-ccoc-only fallback). Lifted
stays available by explicit request only. The app passes no allocation
argument anywhere, so after this change the app always gets linear, which is
the intended state.

## 2. Why this fixes the reported symptom

The chain today: app Calibrate press calls `calibrate_distortions(coc, p)`
(succeeds at p < 1; the a260 guard refuses only the exact spelling `p = 1`),
then the `pricing.allocate` exhibit touches `CalibrationResult.pricing_df`
(`results.py:300`), which calls `analyze_distortions(distortions=..., p=p)`.
The sweep loop at `_portfolio.py:4043` skips any `has_mass` distortion on an
unbounded book, anchor unseen, because the sweep prices through
`pricing_at`'s hardcoded lifted default.

The skip is **correct under lifted at any anchor**: the lifted tail share
beta integrates the distorted law beyond the anchor all the way to the
essential sup, where the ccoc mass `d = r/(1+r)` lives. Numerically the mass
collapses onto the FFT truncation row (`_portfolio_common.py:209`), and at a
p = 0.99 anchor with r = 10% roughly ninety percent of `gS(a)` is that mass,
so the split reads the truncation row rather than the book (the numerics-3 G6
"different bounded problem"). A finite anchor trumps unboundedness for every
total-level number, which is what a260 and a261 encoded, but not for beta.

The linear allocation has no such problem: alpha is the objective tail share,
`build_augmented` builds the linear frame under mass plus unbounded with the
beta columns blanked, and `unit_capital_at` documents the alpha-based margin
as stable (`_portfolio_common.py:215`, `:312`). With linear as the resolved
default, ccoc flows through the sweep like any other family and the warning
never fires from the app.

## 3. Scope: the call sites (audited 2026-08-12, LIB at a263)

| Site | Today | Change |
|---|---|---|
| `Portfolio.apply_distortion` (`_portfolio.py:3313`) | `allocation='lifted'` | `allocation=None`, resolve `None` to `self.allocation_method` here (the cache gatekeeper); cache key uses the resolved value |
| `Portfolio.pricing_at` (`_portfolio.py:3394`) | `allocation='lifted'` | `allocation=None`, pass through to `apply_distortion` |
| `Portfolio.pentagon_at` (`_portfolio.py:3470`) | `allocation='lifted'` | same as `pricing_at` |
| `Portfolio._build_augmented` (`_portfolio.py:3538`) | `allocation='lifted'` | `allocation=None`, resolve to `self.allocation_method` (belt and braces; `apply_distortion` always passes the resolved value) |
| `build_augmented` free function (`_portfolio_common.py:90`) | `allocation='lifted'` | `allocation=None`, resolve to `port.allocation_method`; the lifted mass guard at `:138` is unchanged and now reachable only by explicit request |
| `Portfolio.analyze_distortion` (`_portfolio.py:3901`) | no parameter, inherits lifted via `pricing_at` at `:3937` | add `allocation=None`, pass through |
| `Portfolio.analyze_distortions` (`_portfolio.py:3959`) | no parameter; skip guard at `:4043` tests only `has_mass and not bounded`; `pricing_at` call at `:4054` on the lifted default; snapshot filter `alloc_ == 'lifted'` at `:4084` | add `allocation=None`; resolve once at the top; **skip and warn only when the resolved allocation is `'lifted'`**; pass the resolved value to `pricing_at`; snapshot filter keyed on the resolved value; docstring Notes paragraph rewritten |
| `Portfolio.price` (`_portfolio.py:3602`) | already resolves `None` to `allocation_method` | no behavior change; align the resolution idiom only if it falls out naturally |
| `Portfolio.augmented_df` accessor (`_portfolio.py:3374`) | bare `apply_distortion` call | no edit; now returns the linear frame by default, which is the point |
| `allocation_diagnostics` (`_portfolio.py:3557`, `_portfolio_common.py:335`, `pedagogy.py:1761`) | `surface='lifted'` | **unchanged**: a surface-explicit diagnostic about layer curves, `surface` is a deliberate choice, not a default trap |
| `Aggregate.apply_distortion` (`_aggregate.py:4315`) | hard raise on mass plus unbounded, no allocation concept (one distribution, nothing to allocate) | **unchanged** |
| `CalibrationResult.pricing_df` (`results.py:300`) | no allocation plumbing | no signature change; inherits the new default, which is the fix for the exhibit |

Warning text when the resolved allocation is `'lifted'` (explicit request),
suggested wording, keep it wire-readable:

```
analyze_distortions: skipping {name}: mass distortion on an unbounded
portfolio, and the lifted split reads the truncation row rather than the
book (numerics-3 G6). Re-run with allocation='linear' (the default), or
certify `bounded = True` if the support is in fact bounded.
```

The "nothing to price" ValueError at `:4062` stays; it is now reachable only
under an explicit lifted request with an all-mass set.

## 4. What deliberately does not change

- The lifted refusal in `build_augmented` (`_portfolio_common.py:138`) and the
  `Aggregate.apply_distortion` raise: both correct, both stay.
- `guard_unbounded_anchor` (a260): p = 1 on an unbounded risk is still refused.
- All total-level numbers: the two allocations differ only in the tail share
  used to split the last layer, so `exag_total`, `price_pentagon`, `evaluate`,
  `calibrate_distortions`, and `reins_price_df` (which prices each view's pmf
  through `Distortion.price`, never through the augmented frame) are identical
  before and after.
- Bounded books keep taking lifted plus ccoc **when asked for lifted
  explicitly**. Their default per-unit splits flip from lifted to linear along
  with everything else; that is the a17 intent, but it is a visible number
  change on any default-path readout, which is why section 6 regenerates
  nothing silently.

## 5. Tests

Existing pins to update, each keeping its lifted coverage by passing
`allocation='lifted'` explicitly:

1. `tests/test_numerics3_distortion.py::test_mass_guard_in_builder`
   (`:320`). The bare `p.apply_distortion(CCOC)` raise at `:327` now needs
   `allocation='lifted'` (the comment "lifted default" comes out). The
   `analyze_distortions` skip pin at `:336` likewise gets
   `allocation='lifted'`. Add the default-path assertions alongside: bare
   `apply_distortion(CCOC)` builds (beta columns blanked, `exag_*` finite),
   and `analyze_distortions(p=0.99, distortions={'ccoc': ..., 'dual': ...})`
   includes **both** families with no warning.
2. `tests/test_portfolio_peg_regression.py` and
   `tests/capture_peg_baseline.py`. The 120-cell regression lock was captured
   on the lifted path (the fixture comment at `:53` already calls itself "the
   pre-change lifted regression lock"). Keep it byte-identical: pass
   `allocation='lifted'` explicitly in the fixture (`:59`) and in the capture
   script (`:73`), update the comments, and keep the no-mass filter there
   (lifted still refuses ccoc). Do **not** regenerate `peg_baseline.json`.
3. `tests/test_pricing_results.py` (`:106`, `:196`): re-run; where a default
   sweep on an unbounded book now includes ccoc, adjust the shape pins to
   match and treat any other movement as a finding.
4. `tests/baseline/*` (corpus, capture, manifest): untouched; every priced
   case there passes its method explicitly.

New coverage, named for the symptom:

5. A regression test (suggested home: `tests/test_pricing_results.py`): build
   an unbounded portfolio, `calibrate_distortions(coc=0.10, p=0.99)`, assert
   `result.pricing_df` carries ccoc rows with finite per-unit cells, that no
   UserWarning was emitted, and that the ccoc `total` column equals the
   `calibration_df` octet (the round trip the app draws).
6. In `tests/test_exhibits.py` beside the `pricing.allocate` cases (`:738`):
   the exhibit built on that result serves ccoc in every stat slice.
7. An `allocation_method` honor test: set `port.allocation_method = 'lifted'`
   on the unbounded book and assert the default sweep skips ccoc with the
   warning (the member drives the default, both directions).

Gate: full suite (`uv run pytest -m 'slow or not slow'`) at the bump, plus the
numerics gate (`-W error::RuntimeWarning`) since this touches the distortion
numerics path.

## 6. Docs and docstrings

- `analyze_distortions` docstring Notes (`_portfolio.py:4002` region): the
  skip paragraph now describes explicit-lifted behavior; the default prices
  mass distortions on the linear split.
- `apply_distortion` / `pricing_at` / `pentagon_at` / `analyze_distortion`
  Parameters sections: `allocation : {'linear', 'lifted', None}` with `None`
  reading `allocation_method`, matching `price`'s wording at `:3626`.
- `allocation_method` docstring (`:1210`): now true as written; add
  "and the pentagon readouts" if the author wants the list explicit.
- `docs/2_aggregate_overview/pipeline-portfolio.rst`: line 42 (the
  `price / pricing_at` row) and the linear/lifted section at 125 to 137 state
  the default; the refusal paragraph at 192 gains "by explicit request".
- `docs/2_aggregate_overview/features.rst:1380` region: one sentence noting
  the pentagon surface honors `allocation_method` as of this version.
- `docs/flow/_use-pricing-augmented.md:11`: the options line.
- Docs are edited in lockstep, rebuild left to the author per house rule.

## 7. Housekeeping

One coherent version-bump commit: code, tests, docstrings, `.rst` edits,
`pyproject.toml` bump to the next `1.0.0a` version (a264 at drafting),
`CHANGELOG.md` section, `dev/TODO.md` tick if listed, this plan moved to
`dev/done/`. Commit subject:

```
[Allocation-Default-Linear] aNNN: the pentagon surface honors allocation_method, and ccoc allocates on an unbounded book at a finite anchor
```

## 8. Questions for the author before execution

1. **PEG lock**: section 5 keeps `peg_baseline.json` frozen by pinning the
   fixture to explicit lifted. The alternative, regenerating the baseline
   under the linear default, retires the lifted lock entirely. Recommended:
   keep the lock, it is the only end-to-end lifted regression left.
2. **`price_stand_alone`** (`_portfolio.py:3782`) reads `pricing_at` totals
   only, and totals are allocation-independent, so it is untouched here. It
   also stops raising on ccoc plus unbounded as a side effect (the default
   frame is now buildable). Confirm that is welcome rather than scoped out.

## 9. Downstream ripple (API repo, after the LIB sync, not this plan)

Recorded here so it is not lost; execution belongs to the API side under its
own version bump once `uv sync --extra dev` picks up the new LIB.

- `tests/test_pricing_exhibits.py::test_a_skipped_distortion_is_a_warning_and_not_a_failure`
  (`:240`) pins today's skip. It re-pins to the new truth: ccoc present in the
  allocate blocks, no ccoc warning on the response.
- `src/aggregate_api/pricing.py` `run_calibration` docstring Notes (`:211` to
  `:214`) describes the skip as expected; rewrite.
- `dev/plan-pricing-exhibits.md` line 98 ("Distortions skipped by the
  library...") gets a follow-up note; the plan is canonical in the API repo
  and symlinked into LIB `dev/`.
- Version skew reminder: after the LIB bump, the API needs a re-sync or
  `/v1/meta` reports the stale `aggregate_version`.

## 10. Execution notes (2026-08-12, 1.0.0a265)

Executed as written for sections 1 to 4 and 6. Nine things came out
differently, recorded here rather than back-edited into the plan above.

1. **Version.** a264 was taken by `[Bounds-Envelope-Jump]` between drafting
   and execution, so this landed at **a265**.
2. **Section 8 question 1, the PEG lock: answered as recommended.**
   `peg_baseline.json` is frozen and the fixture asks for
   `allocation='lifted'` by name, as does `capture_peg_baseline.py`, so the
   captured cells stay reachable. It is the only end-to-end lifted regression
   left.
3. **Section 8 question 2 had a false premise, so there was nothing to
   confirm.** `price_stand_alone` does **not** stop raising on ccoc plus an
   unbounded book. Its total leg reads `pricing_at`, which now builds, but it
   prices each unit stand alone first through that unit's own
   `Aggregate.price`, and `Aggregate.apply_distortion` raises on mass plus
   unbounded unconditionally (one distribution, nothing to allocate, section
   3's own last row). Untouched, and still raising.
4. **The test ripple was wider than section 5 listed.** Three more sites in
   `tests/test_numerics3_distortion.py`: `test_precapture_lifted_surfaces_survive`
   (its `pricing_at` and `pentagon_at` legs read the pre-change lifted capture
   through the default), `test_cache_keys_coexist` (the sentinel and explicit
   linear are now one key, so the fourth frame asks for lifted), and
   `test_pricing_at_matches_price_and_pentagon` (parametrized over both
   surfaces rather than comparing a default readout against an explicit
   lifted price).
5. **Section 5 item 4 was wrong about `tests/baseline`.** Only its `price()`
   legs pass a method explicitly; the `augmented__*` and `pricing_at__*` legs
   inherited the default in both `tests/baseline/capture.py` and
   `tests/test_baseline.py`, including the `pytest.raises` guard for mass on
   unbounded. Both now say `allocation='lifted'`; no parquet was regenerated.
6. **`tests/data/exhibit_snapshots.json` was regenerated**, which the plan did
   not anticipate. The `EX.Port` fixture is bounded, so it always allocated
   ccoc; what moved is the per-unit capital split under the new default. The
   diff is confined to the two `pricing.allocate/*/CalibrationPortfolio` keys:
   `L`, `M` and `P` per unit are unchanged at that anchor, `Q`, `a`, `PQ` and
   `ROE` move.
7. **A new DecL fixture, `EX.UnbPort`**, for the exhibit-level test in
   section 5 item 6: every book in the exhibit fixtures is discrete and so
   bounded, where lifted could always allocate a mass family. Mirrored in
   section EX of `decl-testers.agg` per the house rule.
8. **One more docstring than section 6 listed**: `_reinsurance.py`'s
   `reins_price_df` Notes told the reader that the same fact "makes
   `Portfolio.analyze_distortions` skip them on an unbounded book", which is
   now true only of an explicit lifted request.
9. **`dev/FEATURES.csv` was not regenerated.** The author's refresh of it is
   in flight and uncommitted; regenerating would have clobbered it. The
   change adds an `allocation` keyword to existing public methods and no new
   capability, so nothing is owed beyond the refresh already tracked in
   `dev/TODO.md`. Separately, `dev/check_features_rst.py` fails at block 80
   (`book_pnl.economic_df`); verified against `HEAD` as pre-existing and
   unrelated.
