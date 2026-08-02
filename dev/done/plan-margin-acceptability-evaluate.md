# [Margin-Acceptability-Evaluate]

**Executed at `1.0.0a187`.** Three things differed from the plan as written;
see *What differed* at the foot.

## Context

`PnL.evaluate()` (landed a105, `src/aggregate/_pnl.py:1938`) reports the
Cherny and Madan breakeven acceptability panel by solving
`rho_g(obligation) = E[consideration]`. That statement is correct only because
distortion risk measures are translation-equivariant: when the consideration is
a fixed number `P`, `rho_g(P - L) = 0` and `rho_g(L) = P` say the same thing.

The moment the consideration is random (swing rating, slide, profit commission,
reinstatement premium, corridor) translation equivariance no longer applies.
`rho_g` is comonotone-additive, not additive, and the margin `M = P - L` is
generally not monotone in `L`, so `rho_g(P) - rho_g(L)` is not `rho_g(P - L)`.
The right statement is `rho_g(margin) = 0`, which requires distorting the
**pushforward distribution of the margin itself**.

Three things fall out once the calibration reads a margin rather than a loss:

1. The single-obligation-leg restriction disappears. A ledger's margin is one
   random variable however many legs feed it, so expense ledgers become
   evaluable.
2. `evaluate` applies to **every margin row of the tower**, not just the grand
   result. You read the acceptability of the gross deal, of each reinsurance
   layer as a standalone position, and of the running net after each purchase,
   so you watch the deal improve as you buy cover.
3. The same worker serves `Aggregate.evaluate(P)` and
   `Portfolio.evaluate(P, unit=...)`, because a constant premium is just the
   degenerate margin `P - X`.

The blocker is that a margin pushforward generally lands on an **irregular**
support, and every `Distortion.calibrate` body integrates with the scalar
lattice sum `np.sum(v) * bs`.

## Design decisions (settled)

- **Grid-agnostic quadrature, not rebucketing.** Rebucketing a margin onto a
  regular lattice is mean-preserving, so it looks right and is not. The layer
  integral gets a width vector.
- **The bifurcation lives in one helper**, `Distortion._quad(v, dx)`. A scalar
  `dx` keeps the exact `np.sum(v) * dx` summation order, so the byte-for-byte
  legacy guarantee on `calibrate_distortions` survives untouched. A vector `dx`
  uses `np.sum(v * dx)`, which for a purely atomic support is **exact**, not an
  approximation: the survival function is a step function and the layer integral
  is a finite sum of rectangles.
- **Tidy output.** `dev/reporting-guidelines.md:55` rule 2 says "Columns are
  pure: one unit per column". A wide frame with a `param` column per family
  would put a PH exponent, a Wang lambda, a Dual `b` and a TVaR `p` under one
  heading, four different units. So `evaluate` returns the long form indexed
  `(Step, distortion)`; the wide presentation view is `.unstack('distortion')`,
  one call away.
- **`E_consideration` is deleted**, property and info row both.
- **Degenerate positions return `NaN` with a warning**, never `0` or `inf`.

## The math, and the exact regression anchor

Write `M` for the margin (payoff orientation) and `D = -M` for its loss
orientation. `_canonical_loss_frame` already reverses a payoff and shifts by
`c = max(0, -min(support))`, giving `Z = D + c >= 0`. Then

```
rho_g(M) = 0  <=>  rho_g(D) = 0  <=>  rho_g(Z) = c
```

by translation equivariance. **The calibration target is the canonical shift
constant**, with no premium term anywhere.

Specialize to a constant premium: `M = P - L`, so `min(D) = min(L) - P` and
`c = P - min(L)`, giving `Z = L - min(L)` and target `P - min(L)`, hence
`rho_g(L) = P`. That is exactly what a105 computes. **Every constant-premium
`evaluate` must return bit-comparable numbers**, which is the regression anchor
for the whole change.

## Phase 1: grid-agnostic quadrature (`src/aggregate/spectral.py`)

Add to `Distortion`:

```python
@staticmethod
def _quad(v, dx):
    """Layer-integral quadrature. Scalar ``dx`` keeps the exact legacy
    summation order; a per-node width vector is the irregular case."""
    return np.sum(v) * dx if np.isscalar(dx) else np.sum(v * dx)
```

Rename the second parameter `bs` to `dx` on the base `calibrate` (line 1976),
on `calibrate_set` (line 2054), and on all ten overrides, documenting `dx` as
either a scalar bucket size or the width vector `x[i+1] - x[i]`. The parameter
is positional-second, so positional callers are unaffected; the keyword `bs=`
callers are `_pricing.py:459`, `_pricing.py:493`, `_pnl.py:2001` and five sites
in `tests/test_distortion_calibrate.py`.

Swap the eight iterative bodies to the helper. Representative, `PHDistortion`
(line 2316):

```python
ex       = self._quad(trho, dx)          # was np.sum(trho) * bs
ex_prime = self._quad(trho * lS, dx)     # was np.sum(trho * lS) * bs
```

The same two-line pattern applies to `WangDistortion` (2411),
`DualDistortion` (2487), `TVaRDistortion` (2612), `CLLDistortion` (3624),
`CLinDistortion` (3741), `LEPDistortion` (3870), `LYDistortion` (3994). The
`+ mass` terms in the last three are `ess_sup * r0` style point-mass
corrections, independent of `dx`; leave them alone. `CCoCDistortion.calibrate`
(2226) is closed-form and ignores the argument.

## Phase 2: GD-native calibration frame (`src/aggregate/_pricing.py`)

The existing helpers assume a contiguous `bs`-lattice `pd.Series`. Give each a
`GridDistribution` form, reusing `GridDistribution` rather than reimplementing
(it already owns `lev`, `mean`, `q`, `cdf`, `sf` and a lazy cumulative):

- `_canonical_gd(gd)` returns `(x, p, c, reverse)` on a 0-based non-negative
  axis. Faithful port of `_canonical_loss_frame` (line 311): reverse when
  `not gd.is_loss_value`, trim `VALIDATION_NOISE` dust, shift by `c`. The
  lattice snap `np.round((x + c) / bs) * bs` runs **only** when `gd.bs` is not
  `None`. `_canonical_loss_frame(obj)` becomes a thin wrapper that builds a GD
  from `obj.density_df['p_total']` and delegates, so the `Aggregate` and
  `Portfolio` path is unchanged.
- `_calibration_survival_gd(x, p, assets)` returns `(S, dx, ess_sup)`. Faithful
  port of `_calibration_survival` (line 277): `S = 1 - cumsum` truncated at the
  first zero, `ess_sup` recorded at the truncation point, the
  `S > 0 and weakly decreasing` assertion kept. `dx` is the scalar `gd.bs` when
  regular, otherwise `np.diff(x)` aligned to the truncated `S` (truncation drops
  the tail, so `x[k+1]` always exists).
- Delete `_LossFrameShim` (`_pnl.py:2238`). It exists only because
  `_canonical_loss_frame` wanted something with a `density_df`.

## Phase 3: the evaluation worker (`src/aggregate/_pricing.py`)

Move `_EVAL_FAMILIES` here from `_pnl.py:136` so all three public faces share it,
next to `DEFAULT_CALIBRATION_DISTORTIONS`.

```python
def evaluate_margin(gd, *, step='', names=None):
    """Cherny and Madan breakeven acceptability panel for a margin distribution."""
```

1. **Degenerate guards, in this order.** If the support carries no mass below 0,
   the position is acceptable at every stress and the index is unbounded: warn
   `DegenerateEvaluationWarning('M >= 0 a.s. (arbitrage)')`, return the NaN
   panel. Otherwise if `gd.mean <= 0` the position is acceptable at no stress:
   warn with `'E[M] <= 0'`, return the NaN panel. Both return `NaN` rather than
   `1` / `0` because the limiting value varies by family and carries no
   information.
2. `x, p, c, _ = _canonical_gd(gd)`, `S, dx, ess_sup = _calibration_survival_gd(...)`.
3. `Distortion.calibrate_set(S=S, dx=dx, premium_target=c, ess_sup=ess_sup,
   assets=ess_sup or a_full, el=..., names=names)`. `el` comes from the
   canonical GD's `lev`, needed only if a caller asks for `ccoc`.
4. Return the tidy block: one row per family, columns `param_name` / `param` /
   `gini_p` / `error` / `status`.

Column notes. `gini_p = 2*integral(g) - 1` stays the family-agnostic index.
`area` is dropped, being exactly `(gini_p + 1) / 2`. `status` is `'ok'` or the
degenerate reason, so a `NaN` explains itself where it is read. `EX` is **not**
duplicated here: `stats_df` already owns per-row moments.

## Phase 4: `PnL.evaluate` over every margin row (`src/aggregate/_pnl.py`)

Replace the body of `evaluate` (line 1938). Iterate the ledger's margin rows in
plan order from `self._by_kind`, namely the kinds `group_result`,
`running_net`, `tier_result`, `grand_result` and `total_impact` (the
`_VIEW_DEFAULTS['margin']` set, `_pnl.py:749`). Each row is an `_EvaluatedLeg`
whose `.gd` is already the signed payoff-oriented margin distribution
(`_pnl.py:466`), exact-irregular at `bs = 0` and a regular rebucket otherwise.
Call the Phase 3 worker per row and concatenate.

- Result: `DataFrame` with `MultiIndex` rows `(Step, distortion)`, `Step`
  carrying the ledger row labels. Wide view via `.unstack('distortion')`.
- Delete the single-obligation-leg and `_source_bs` guards outright.
- **Aggregate the degenerate warnings**: one warning per `evaluate()` call
  naming the affected steps, not one per row. `total_impact` (the reinsurance
  program as a standalone position) will normally read `E[M] <= 0`, correctly,
  since you pay for cover; it stays in the frame as an explained `NaN`.
- The **massive route works unchanged**: sweep-backed rows are constructed with
  an explicit `gd=` from the band sweep (`_EvaluatedLeg.__init__`, `_pnl.py:440`)
  and derived rows come back from the sweep as their own signed-sum functions
  (`_EvaluatedGroup`, `_pnl.py:577`), so every margin row has a regular-`bs` GD.

Aside, not in scope: `PnL.density_df` (line 1883) omits `running_net` rows while
`evaluate` will report them. Worth reconciling later; `evaluate` reads
`_by_kind` directly so nothing here depends on it.

## Phase 5: the two new faces

- `Aggregate.evaluate(P=None, *, names=None)` in `src/aggregate/_aggregate.py`,
  beside `calibrate_distortions` (line 6018). Builds the margin GD `P - X` from
  `density_df['p_total']` with `is_loss_value=False` and hands it to the worker.
  `P` defaults to `self.exp_premium` (`_aggregate.py:1818`) and raises when that
  is unset.
- `Portfolio.evaluate(P=None, *, unit='total', names=None)` in
  `src/aggregate/_portfolio.py`, beside `calibrate_distortions` (line 2767).
  Same construction off `p_total` or the unit's marginal column; `P` defaults to
  `self.exp_premium` (`_portfolio.py:251`) and raises when unset. `unit` accepts
  a name or a list of names, giving the per-unit acceptability profile the
  deferred portfolio plan asks for (`dev/deferred/plan-pnl-portfolio-DEFERRED.md:52`),
  one `Step` row per unit.

Both return the same tidy frame, with `Step` the object or unit name, so the
three faces are shape-compatible and concatenate.

Name vet per `CLAUDE.md`: `evaluate` is currently unattached on `Aggregate`,
`Portfolio`, `GridDistribution` and `Distortion`. The module-level
`evaluate_pgf_polynomial` in `_aggregate_compute.py` is a different namespace.
No shadowing.

## Phase 6: retire `E_consideration`

In `src` it has exactly two consumers: `evaluate` (line 1998, removed by
Phase 3) and one `info` block row (line 2128). It is **not** used by `ratio_df`,
which accumulates `P` per block from `LEG_KINDS` buckets; the claim in
`CHANGELOG.md:58` and `dev/done/plan-pnl-ratio-frame.md:50` that it is "the `P`
denominator `ratio_df` needs" is stale. Both are history and stay as written;
the new CHANGELOG section records the correction.

Delete the property (`_pnl.py:1359`) and the `E[consideration]` info row
(`_pnl.py:2128`). Downstream edits: `tests/test_create_pnl.py:261`,
`tests/test_renewal_agg.py:154`, `docs/new_material/info-strings.rst:270`, and
`tests/test_fcc_surface.py` where it asserts on the info block.

## Phase 7: the warning class

`DegenerateEvaluationWarning(UserWarning)` in `src/aggregate/constants.py`,
following `CoarseJointGridWarning` (line 266) exactly: class with a docstring
explaining when it fires, plus the name in `constants.__all__` (line 33).

## Verification

Run tier 1 (`pytest -n0 --dist no --testmon-forceselect`) through the edit loop,
tier 2 (`uv run pytest`) before declaring done, tier 3
(`uv run pytest -m 'slow or not slow'`) once at the bump.

New and changed cases:

- **Legacy invariance.** `tests/test_distortion_calibrate.py` passes with values
  unchanged after the `_quad` swap, confirming the scalar path is bit-identical.
- **The regression anchor.** The two existing constant-premium cases
  (`tests/test_pnl.py:338` and `:352`) return the same parameters through the
  margin route, per the specialization above. Adjust only for the reshaped
  frame, not for the numbers.
- **Irregular exactness.** Build a small `dsev` P&L whose margin lands off any
  lattice. After calibration, independently recompute the distorted mean of the
  margin at the solved parameter and assert it is 0 to ~1e-10. This is the test
  that a rebucketing implementation would fail while still matching the mean.
- **Variable premium.** A swing-rated or profit-commission P&L where the a105
  answer and the margin answer genuinely differ. Assert the new parameter
  satisfies `rho_g(M) = 0` and the old one does not.
- **Tower monotonicity.** On a program bought at a reasonable price, `gini_p`
  rises down the `net through ...` rows. This is the headline behavior.
- **Degenerate ends.** `E[M] <= 0` and `M >= 0` a.s. each warn once and return
  `NaN` with the right `status`.
- **Massive route.** `evaluate` runs on a `MassiveBivariateDistribution`-backed
  P&L.
- **Cross-face agreement.** `Aggregate.evaluate(P)` matches the grand-result row
  of the equivalent `build('pnl ...')`, and `Portfolio.evaluate(P)` matches its
  total.
- Add any new DecL programs used by tests to `src/aggregate/agg/decl-testers.agg`
  under the matching section.

Interactive check, the story the change is for:

```python
from aggregate import build, qd
p = build('pnl ...')            # gross plus two reinsurance layers
qd(p.evaluate().unstack('distortion'))
```

## Release hygiene

One version bump, one commit, subject
`[Margin-Acceptability-Evaluate] a1NN: evaluate solves rho(margin)=0 over every
margin row; E_consideration retired`. The commit carries the code, the
`pyproject.toml` bump, the `CHANGELOG.md` section (which must state that
variable-premium answers change, that this is a correction rather than an
enhancement, and that the `ratio_df` claim about `E_consideration` was stale),
the `dev/FEATURES.csv` regen via `dev/regen_features.py`, and `dev/TODO.md`.

## Out of scope

`[Named-Cherny-Madan-Families]` (#23, `dev/TODO.md:422`), which would add
MINMAXVAR / MAXVAR / MAXMINVAR so the panel can report the named indices. It
needs none of this and stays pending, per your call.

## What differed

**1. The dust trim is uniform, not gated on grid regularity.** The plan had
`_canonical_grid` skip the `VALIDATION_NOISE` trim on an irregular grid, on the
argument that every atom of an exact pushforward is a genuine outcome. That was
wrong twice over. It moved `ph` by 6e-5 against the a105 constant-premium
anchor (the margin frame integrated a far tail the obligation frame trimmed),
and worse, on a stitched occurrence peel it set `c` from a bucket holding
~1e-18 of probability at the top of a padded grid: targets of 262,096 against a
grid top of 262,144, and "questionable convergence" from every TVaR solve. The
trim is about mass being numerically negligible, which is equally true on
either grid, and `c` **is** the calibration target, so a dust bucket poisons
the whole solve. With the trim uniform, the constant-premium anchor holds on
all four families to displayed precision, which is why the regression test
asserts the a105 values directly.

**2. A stitched `total impact` row has no distribution.** It is a `_DeltaRow`:
a delta of two statistics whose sides ride different marginals, with only a
quantile shim for a `gd`. Not a degenerate margin, a row that is not a random
variable at all. It gets `no_distribution_panel` and its own `status` rather
than an exception, so the row keeps its place in the tower and says why it is
`NaN`. This also replaced the old
`test_stitched_peel_refuses_evaluate_and_composition`: the rest of a stitched
peel now evaluates fine.

**3. `evaluate_margin` validates orientation instead of assuming it.** The
solve reads "more is better", so a loss-valued `GridDistribution` raises rather
than silently returning an inverted answer. The array core
`_evaluate_margin_arrays` is shared with `evaluate_constant_premium`, which is
how the `Aggregate` / `Portfolio` faces reach the same code path.

### Open, deliberately

`PnL.density_df` omits the `running_net` rows that `evaluate` reports. Tracked
as `[PnL-Density-DF-Running-Nets]` in `dev/TODO.md`; nothing depends on it,
since `evaluate` reads `_by_kind` directly.
