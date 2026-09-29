# Plan: bivariate hints tuples and the dfreq-outer window defect

Date: 2026-09-29. Status: **approved**. Two independent fixes, two version
bumps (next free: 1.0.0a354, 1.0.0a355).

Decisions settled by the author 2026-09-29:

* **Wide over clipping**: scaled empirical outcomes use `ceil`, not `rint`;
  a slightly wide measured window is the safe direction.
* **Guard is in**: the analytic mean guard in `_measure_marginal_window`
  ships, tolerance `max(0.01 * |mean|, 0.01 * sd)`.
* **Two bumps, two commits**: a354 and a355 land separately so the history
  stays bisectable.
* **CantorArt joins the library**: a `bv CantorArt` entry is added to
  `src/aggregate/agg/library.agg` beside the other bivariates, rolled into
  the a355 commit (it only builds correctly once the dfreq-outer window
  fix lands), not a commit of its own.

## Background (standalone; no conversation context needed)

A `bv` (bivariate) statement couples two component aggregates with a shared
outer frequency and builds their joint density by 2-D FFT
(`src/aggregate/bivariate.py`, class `BivariateAggregate`). Axis sizing is
measured, not guessed: each component's standalone loss marginal is realized
once on a fine 1-D grid and the axis window is read off that pmf
(`_size_axes` / `_measure_marginal_window` / `_standalone_marginal`).

`BivariateAggregate.update` already accepts per-axis sizing pairs, and they
work when passed as `build()` keywords: `build(prog, bs=(3, 1))` and
`build(prog, log2=(9, 12))` flow verbatim through the bvagg branch of
`Underwriter.build_many` (`underwriter.py:2665`) into `update`
(`bivariate.py:1840`). A scalar `log2` is the total 2-D cell budget, a pair
pins the per-axis lengths; a scalar `bs` applies to both axes, a pair pins
each axis; a `0` entry in a pair leaves that axis auto.

Two defects, both verified by direct experiment on 1.0.0a353:

**Defect 1 (hints).** The DecL trailer route cannot express the pairs.
`hints{bs=(3, 1)}` crashes with
`ValueError: could not convert string to float: '(3, 1)'` and
`hints{log2=(9, 12)}` crashes at `int(hints['log2'])`. Cause:
`_coerce_hint_value` (`underwriter.py:252`) only tries int, float, an `a/b`
fraction, and bool, so a parenthesized pair survives as a raw string that
blows up downstream, violating the `_parse_hints` contract that a malformed
hint degrades to a warning, never a crash. The grammar itself is fine: the
`HINTS.3` token is `hints\{[^}]*\}` (`decl.lark:840`), so parens and commas
already reach the underwriter, and `_parse_hints` splits the body on `;`,
which a tuple does not contain.

**Defect 2 (dfreq outer window).** Any `bv` whose shared outer frequency is
empirical (`dfreq`) mis-measures both axis windows, and the resulting joint
can be quietly catastrophic. The reproducing case:

```decl
bv CantorArt
  dfreq [1]
  agg A dfreq [5] sev cantor
  agg B dfreq [5] sev cantor
```

builds a joint holding **8.6% of its mass**, marginal mean 0.12 against a
true 2.5 (a `DefectiveDistributionWarning`, deficit 0.707, fires once during
update, but the object builds and plots). Root cause, in
`_standalone_marginal` (`bivariate.py:1577`): the measurement marginal is
built as `Aggregate(unit sev_* keys, exp_en = outer_en * unit_n,
**self._freq_kwargs)`, relying on the thinning identity so the unit's own
per-event count folds into `exp_en`. That identity holds for
Poisson/mixed/negbin outers (the docstring says so). For `dfreq [1]` the
outer `_freq_kwargs` carry `freq_name='empirical', freq_a=[1]`, which pins
the count at exactly 1 and **silently discards `exp_en=5`**: the measured
marginal is one per-claim cantor draw (support [0, 1], mean 0.5) instead of
the unit aggregate (5 draws, support [0, 5], mean 2.5). `_size_axes` then
windows each axis to about [0, 2] and the real unit update clips 70% of the
pmf per axis. Verified: standalone-marginal mean is 0.5 for `dfreq [1]`
outer and 1.0 for `dfreq [2]` outer (true 5.0); the Poisson-outer control
(`... 1 claim ... poisson`) measures 2.5 and sizes correctly to [0, 16].

The netceded routes are not involved: `_netceded_sizing_kwargs`
(`bivariate.py:1899`) already accepts a `(log2_x, log2_y)` pair and already
rejects a `bs` pair with a clear error (one common `bs` couples the axes).

## Workstream [Hints-Tuple-Values] (bump 1.0.0a354)

Goal: `hints{log2=(9,12); bs=(3,1);}` on a `bv` statement behaves exactly
like `build(..., log2=(9, 12), bs=(3, 1))`; a pair hint on a non-bivariate
object fails with one clear sentence, not a downstream float() traceback.

Changes, all in `src/aggregate/underwriter.py`:

1. **`_coerce_hint_value` learns pairs.** Recognize `(a, b)`: strip outer
   parens, split on the comma, coerce each element by the existing scalar
   ladder (int, float, `a/b` fraction), return a 2-tuple. `0` elements are
   legal (that axis stays auto, matching the `update` contract). Anything
   that is not exactly two coercible numeric elements falls through to the
   current raw-string behavior, preserving warn-and-degrade. Implement with
   a module-level `_HINT_PAIR_RE` next to the existing fraction regex; no
   new public names.
2. **`_resolve_hints` passes a pair through.** At `underwriter.py:383` the
   unconditional `int(hints['log2'])` becomes: pass a tuple through
   untouched, `int()` only a scalar. `bs` already passes verbatim.
3. **Clear rejection on non-bivariate objects.** In `build_many`, the `agg`
   branch (`underwriter.py:~2680`) and `port` branch (`~2700`) consume
   `log2`/`bs` scalars (`_bs_window`, `Portfolio.update`); a pair reaching
   them today would die obscurely. After `_resolve_hints`, raise
   `ValueError(f"per-axis (x, y) log2/bs sizing applies to bivariate "
   f"objects only; {name} is a {kind}")` when either is a tuple. Guard the
   netceded DecL-prefix inner-aggregate sizing the same way
   (`_update_netceded`, `bivariate.py:1958`, where `int(log2)` on a tuple
   is a bare TypeError today).
4. **Docstrings.** `_coerce_hint_value`, `_parse_hints`, and the
   `BivariateAggregate.update` parameter docs mention the hints spelling.
   Grep `docs/` for the hints{} reference page and add the pair form in
   lockstep; docs rebuild stays with the author per house rules.

Tests (locate the existing hints tests with `rg -l "_parse_hints|hints\{"
tests/` and extend in place):

* Unit: `_coerce_hint_value` on `'(9, 12)'` gives `(9, 12)`; `'(3,1)'`
  gives `(3, 1)`; `'(1/8, 1/2)'` gives `(0.125, 0.5)`; `'(0, 12)'` keeps
  the 0; `'(1, 2, 3)'` and `'(a, b)'` degrade to the raw string.
* End-to-end: the goal program with `hints{log2=(9,12); bs=(3,1);}` builds
  with per-axis `(bs, nout) == ((3.0, 512), (1.0, 4096))`; pick component
  scales so the pinned grids cover the measured windows (no clipped
  warning).
* Error: the same hint on a plain `agg` raises the bivariate-only message.
* Corpus: add one `bv ... hints{log2=(9,12); bs=(3,1);}` line to
  `aggregate/agg/test_suite.agg`, regenerate
  `tests/data/expected_specs.json` via
  `uv run python tests/capture_spec_snapshot.py`, and read the diff: only
  the new line may appear.

## Workstream [Bv-Dfreq-Outer-Window] (bump 1.0.0a355)

Goal: the CantorArt program (and any empirical-outer `bv`) sizes its axes
off the true marginal; a future mis-measurement of any cause warns loudly
instead of shipping a defective joint.

Changes, all in `src/aggregate/bivariate.py`:

1. **Scale the empirical outer's outcomes.** In `_standalone_marginal`
   (`bivariate.py:1577`), when `self._freq_kwargs.get('freq_name') ==
   'empirical'`, copy the freq kwargs and replace the outcome vector:
   `freq_a -> np.ceil(freq_a * float(self.units[i].n))` element-wise
   (`ceil` keeps outcomes integral and biases the measured window wide, the
   safe direction; a 0 outcome stays 0). Weights `freq_b` and
   `freq_zm`/`freq_p0` pass through unchanged; `exp_en` stays as set (the
   empirical family ignores it, now consistently). This is **exact** when
   the unit's per-event count is degenerate (`dfreq [n]`, the copula-art
   case) and mean-exact, variance-understated otherwise; say so in the
   docstring Notes and drop the "expected count en * (per-event trigger
   mean)" sentence's implicit claim that the construction is uniform across
   families.
2. **Analytic mean guard.** In `_measure_marginal_window`
   (`bivariate.py:1585`), after the pmf is realized, compare the measured
   mean `(xs * density).sum()` against the analytic marginal mean and sd
   from the existing `_marginal_moments(i)` (`bivariate.py:1543`, the outer
   compound of the per-event severity, valid for every frequency family
   including empirical). If `|measured - analytic_mean| >
   max(0.01 * |analytic_mean|, 0.01 * analytic_sd)`, `warn_once` a
   `DefectiveDistributionWarning`: the axis window was measured off a
   marginal that disagrees with theory, so the grid may clip; name the
   axis, both means, and the object. This catches any frequency family the
   thinning identity misses in future, not just the empirical case being
   fixed. Verify at implementation that `_marginal_moments` returns 2.5 for
   the CantorArt axis (it uses `freq_moms`, which handles empirical); if it
   does not, that is a finding to fix first, since it is also the
   validation target.

3. **Library entry.** Add to `src/aggregate/agg/library.agg`, in the
   "Bivariate and dependent risks" section beside the other copula
   bivariates, styled to match its neighbors (multi-line layout, `note{}`
   describing the object, `tags{topic:bivariate, ...}`):

   ```decl
   bv CantorArt
     dfreq [2]
     agg A dfreq [2] sev cantor
     agg B dfreq [2] sev cantor
   ```

   The marginal is 4 cantor draws (mean 2, support [0, 4]); confirm at
   implementation that the entry builds clean under the fix (full mass, no
   warning) and that whatever suite exercises library.agg entries picks it
   up.

Tests, in `tests/test_bivariate.py` (module is `slow`-marked; keep grids
small so the cases stay cheap):

* CantorArt builds with `density.sum() >= 0.999` and `marginal(0)` mean
  within 1% of 2.5, no `DefectiveDistributionWarning`.
* `dfreq [2]` outer variant: the standalone measurement marginal has mean
  10.0 (2 events of 5 claims each).
* Poisson-outer control unchanged: axis window still covers [0, ~16],
  marginal mean 2.5.
* Guard unit test: monkeypatch `_marginal_moments` to return a mismatching
  mean and assert the warning fires from `_measure_marginal_window`.

## Acceptance checks

1. `build('bv ... hints{log2=(9,12); bs=(3,1);}')` realizes per-axis
   `bs == (3.0, 1.0)` and axis lengths `(512, 4096)`.
2. The same pair via `build()` keywords is unchanged (regression).
3. A pair hint on an `agg`/`port` raises the one-sentence bivariate-only
   error.
4. CantorArt: total joint mass ~1, both marginal means ~2.5, no warning.
5. Tier 2 (`uv run pytest`) green after each workstream; tier 3
   (`uv run pytest -m 'slow or not slow'`) at the a355 bump since
   `bivariate.py` and the slow bivariate suites are touched.
6. Spec snapshot diff contains only the deliberately added corpus line.
7. `build('CantorArt')` (name lookup resolves library bivariates, verified
   with `build('BivariateNormal')` on a353) loads the new library entry
   and builds with full mass and no warning.

## Version and commit plan

Two bumps, one commit each, house format:

```
[Hints-Tuple-Values] a354: hints{} accepts per-axis (x, y) log2/bs pairs
[Bv-Dfreq-Outer-Window] a355: empirical outer frequency sizes bv axes off the true marginal
```

Each commit carries its code, `pyproject.toml` bump, one-paragraph
`CHANGELOG.md` entry, and any `dev/TODO.md` touch; this plan moves to
`dev/done/` when the author says it is done.

## Names introduced (for review, per house naming rule)

* `_HINT_PAIR_RE` (module constant, `underwriter.py`): private, no
  collisions (`rg` clean).
* No new public methods, attributes, or kwargs; both `log2` and `bs`
  keep their existing meanings, extended to pairs already documented on
  `BivariateAggregate.update`.
