# plan-a362: PnL punch-ups, the Palm ladder and the exhibit pass-throughs

**Status:** drafted 2026-09-29; the author's rulings on all five open
questions recorded 2026-09-29 (see the Rulings section at the end). Ready to
execute.
**Repo:** `V:\worktrees\aggregate_REFACTOR`, the `aggregate` library, six
bumps `1.0.0a362` through `1.0.0a367`, phases in order below. One small
`aggregate-api` commit at the end (nav comment and hint residue; version
mechanics per that repo's rules, checked at execution). The same session
executes both sides.

## Goal

Two workstreams, one plan, per the author's request of 2026-09-29.

**A. Waterfall, ledger, and capital.** The scenario (`κ`) ladder becomes
computable on builds where it is absent today (multi-layer occurrence peels
and other stitched builds), via a new 1-D conditional-mean identity (the
"Palm" route, background below), filling the whole `economic_df` ladder and,
as a consequence, the waterfall's `M01 diversified` column. Nothing 2-D is
removed: the `occ_bivariate` per-atom route stays primary wherever it serves
today, and the Palm route fills gaps only. The ledger additionally learns to
show **both** ladders: a new `economic_marginal_df` carries the marginal
(`P`) ladder always, beside `economic_df`'s scenario (`κ`) ladder, so the
either/or switch becomes additive ("ledger-sa and ledger-div", the author's
ruling on question 1). `M01 standalone` on a ceded step changes meaning: from
blank (the a361 state) to the step's own result at the **right-tail**
percentile, the writer's 1-in-100, positive almost always. The evaluation
frame's ratio columns rename to `MSD`, `SA CoC`, and `Div CoC`; the a361
`Cost of relief` column is removed and its content folds back into `Div CoC`.
Builds with loss-sensitive features (swing, slide, profit commission,
corridor, retro, reinstatements) keep the marginal ladder and serve `NaN` in
the waterfall's diversified cells, as do the frequency families outside the
supported set.

**B. Exhibit pass-throughs.** Four PnL exhibit-surface fixes, all following
the pattern `PnL.stats_df` already uses (delegate to the live wrapped engine
on `PnL.engine`): the Validation tab shows the engine's validation frame
instead of an empty ledger audit; the Tail tab lights up (the frame already
exists as `PnL.tail_df`); and the Re tab's Plot, Summary, Stats, and Density
leaves light up by delegating the `reins` exhibit and the `reins` chart to
the engine.

## Background A: the Palm identity, and why the gap exists

The full mathematical write-up, with derivations, is in the working note
`C:\tmp\explain.qmd` (rendered `explain.html`), authored 2026-09-29. The plan
restates what the implementation needs.

For a compound sum sharing one event process, with per-claim functions `c`
(the target, e.g. a layer's cession) and `n` (the conditioning subject, e.g.
the occurrence-retained claim), the conditional mean satisfies, for any
frequency with an evaluable PGF derivative:

```
E[S_c | S_n = s] * f_{S_n}(s) = integral c(x) w(s - n(x)) dF(x),
FT(w) = P_N'(phi_n),  phi_n = FT of the n-image severity density
```

The right side is one 1-D convolution of `w` against the image measure `nu`
obtained by scattering `c(x_a) * p_a` into the bucket at `n(x_a)`. Cost: one
shared inverse FFT for `w` plus one FFT per target. Only the PGF derivative
is needed, never the frequency MGF (author's question, 2026-09-29). Sanity
anchor: with `c = n = id` the identity is the integral form of the Panjer
recursion, so `kappa(s) = s`, the closing-row identity
`test_waterfall_closing_row_standalone_equals_diversified` already pins.

An aggregate cover written on the occurrence-retained aggregate (any
deterministic `a(S_r)`, linear or not) does not escalate the computation:
conditioning on the net `T(S_r) = S_r - a(S_r)` is a coarsening of
conditioning on `S_r`, handled by a level-set average over the retained grid
(no FFT). The identity's premise fails exactly for loss-sensitive and
path-dependent features (reinstatement premiums, swing, slide, pc, corridor,
retro, limited reinstatements), where the net is a joint function of two
dependent aggregates; those builds keep the marginal ladder, per the author's
ruling.

Why the gap exists today: the module comment at `_pnl_builders.py:801-818`
records that splitting `m` occurrence layers per-atom "would need an
(m+1)-axis joint", so multi-layer occurrence peels route through the stitched
seam with marginal dispersion only, and both the ledger's ladder and the
waterfall's diversified column stay marginal or blank (the `Peel` test
fixture). The `(m+1)`-axis claim is true of the joint distribution and moot
for the ladder, which wants only conditional means.

## Background: the ledger's ladder semantics (verified 2026-09-29)

`PnL.economic_df` (`_pnl.py:1826-1897`) serves every ledger row with
`EX / SD / CV / Skew` plus the nine-point `PERCENTILE_LADDER`
(`_pnl.py:151`: 0.01 through 0.99). The ladder is either/or: scenario (`κ`)
headers via `_scenario_ladder` (`_pnl.py:1764`) when the ledger shares one
source (`self._probs is not None`), each cell `E[row | grand result at its
q-quantile]`, footing down the sheet; plain marginal `P` headers otherwise
(`[Decision-Kappa-Shared-Source-Rule]`, `[Decision-Ladder-Column-Names]`,
headers built by `_pct_label` / `_kappa_label`, `_pnl.py:706-731`). The
massive one-sweep route (`stack_marginal_pnls`) also keeps marginal ladders
and is out of scope here. The waterfall reads the ledger's `κ01` cell per
step (`_waterfall_frames`), so filling the ledger fills the waterfall.

## Background B: the wiring facts (verified 2026-09-29)

- `PnL.engine` (`_pnl.py:922`, set by `_adopt_engine` at `:2696-2720`) holds
  the **live wrapped Aggregate** for every DecL-built P&L. The "opaque
  reference, no retained engine dependency" language applies to `PnL.source`
  (`:1319-1328`), not to `.engine`. `PnL.stats_df` (`:1796-1822`) already
  delegates: `engine = self.engine; return pd.DataFrame() if engine is None
  else engine.stats_df`. Charts already reach through too:
  `charts\_emit_structure.py:112-120` (`_engine(obj)`).
- The `validation` exhibit for PnL (`exhibits\__init__.py:198-201`) serves
  `PnL.validation_df` (`_pnl.py:2345-2373`), a per-leg **rebucketing audit**
  that is empty for every DecL build because no builder passes `bs=` to a
  `Leg`. Its empty-for-exact-legs behavior is pinned by
  `tests\test_create_pnl.py:472-486` and must not change meaning.
- The `tail` exhibit is registered for `[Aggregate, Portfolio]` only
  (`exhibits\__init__.py:213-217`); `PnL.tail_df` already exists
  (`_pnl.py:1552-1569`) over the margin via the shared
  `return_period_frame`, payoff-oriented (low tail adverse, caveat documented
  at `_pnl.py:1538-1548`).
- The `reins` exhibit (`_core.py:977`, frames builder `_reins_frames` at
  `_core.py:1097-1113`) serves two blocks off `obj.reins_stats_df` and
  `obj.reins_summary_df`; registered for Aggregate and Portfolio; availability
  predicate `_perspectives_reins` (`_core.py:359-362`) checks
  `obj.occ_reins / obj.agg_reins`, which PnL does not have. The insurer
  override for Aggregate (`exhibits\_aggregate.py:160-215`) additionally reads
  `reins_layer_terms` and `reins_layer_moments`.
- The `reins` chart is registered for Aggregate only, by explicit note
  "**Aggregate only, at 1.0** (author's decision)" (`charts\_emit_reins.py:21-24`).
  The author's request of 2026-09-29 (item B3) supersedes that note for PnL.
- The SPA (`V:\dev\aggregate-api\web\src\nav.js`) drives tabs generically off
  exhibit and chart availability: `Overview->Tail` = exhibit `tail`,
  `Overview->Validation` = exhibit `validation`, `Re->Plot` = chart `reins`,
  `Re->Summary/Stats/Density` = exhibit `reins`. Lighting the tabs is LIB
  work; the API side owes only stale comments and `why` hints (nav.js
  lines 100-104 and 107-180, including the recorded 2026-09-25 ruling that a
  reinsured P&L lights Diagram only, now superseded).

## Frequency and severity facts the kernel builds on (verified 2026-09-29)

- The PGF is `Frequency.freq_pgf(n, z)` (`_frequency.py`), elementwise in
  `z`; the compound kernel is `freq_sev_convolution`
  (`_aggregate_compute.py:23`, core at `:97-101`). **Every PGF evaluation
  must pass `Aggregate.base_mean`** (`_aggregate.py:3503-3519`), not `n`.
- No PGF derivative exists anywhere in the codebase. Families and their
  `freq_pgf` bodies are enumerated at `_frequency.py:775-1380`; the G-mixed
  Poisson base `_FrequencyMixedPoisson` (`:1162`) writes each composed PGF
  longhand with **no materialized mixing MGF**.
- Zero modification monkey-patches `freq_pgf` on the instance
  (`_install_zm_wrappers`, `_frequency.py:406-437`): `G_M = (1-c) + c G`, so
  the derivative wrapper is `c * G'` and any new derivative method must
  receive the same treatment or silently disagree.
- Per-claim maps: `make_ceder_netter(reins_list)` (`_reinsurance.py:84`,
  vectorized piecewise-linear), `agg.occ_ceder / occ_netter`, severity grids
  `agg.xs_sev` and views `agg.sev_density_gross / _net / _ceded`,
  mean-preserving scatter `Aggregate._rebucket_to_grid` (`_aggregate.py:4953`).
  `_sev_transform_marginal(agg, f)` (`_pnl_builders.py:898`) is the existing
  one-FFT idiom the kernel extends.
- Variable-feature detection: spec-parse time in `underwriter.py:1607-1745`;
  the classification lands on `inner._pnl_recipe['kind']`: `'gcn'` / `'plain'`
  are fixed programs (Palm-eligible), `'var'` / `'reins'` are loss-sensitive
  (marginal ladder, `NaN` diversified). Runtime discriminators on the engine:
  `getattr(agg, 'variable_terms', None)` and
  `getattr(agg, 'reinstatement_terms', None)`.
- The Portfolio independence trick (`_portfolio_density.py:198-223`) is the
  structural template for the Fourier-domain conditional mean; the Palm kernel
  is the shared-event analogue with `P_N'` in place of the leave-one-out FT.

## Names introduced (vetted 2026-09-29; re-run `rg` at execution)

- `Frequency.freq_pgf_prime(n, z)`: the PGF derivative, elementwise, same
  contract as `freq_pgf`. No existing `*_prime`, `dpgf`, or derivative name
  collides. Families outside the supported set raise `NotImplementedError`
  (callers catch; the ladder stays marginal). The frequency MGF is never
  needed.
- `palm_conditional_mean(...)` in `_aggregate_compute.py`, beside
  `freq_sev_convolution`: the kernel, taking the n-image severity density,
  the target image measure, `freq_pgf_prime`, and the base mean; returns the
  kappa vector on the aggregate grid.
- `PnL._palm_ladder`: private storage for the build-time Palm scenario
  ladder, `{ledger row label: [value per PERCENTILE_LADDER point]}`;
  `None`/absent when not computed.
- `PnL.economic_marginal_df`: the ledger sheet with the marginal (`P`)
  ladder, always computable. `rg` confirms only `economic_df` and
  `economic_ratios_df` exist in the `economic_*` family today.
- `PnL.engine_validation_df`, `PnL.reins_stats_df`, `PnL.reins_summary_df`:
  delegating properties over `self.engine`, empty-frame/`None` safe when the
  engine is absent. `rg` confirms no `reins_*` or `engine_*_df` name exists
  on PnL today.
- Evaluation columns `MSD`, `SA CoC`, `Div CoC`, exactly as typed (author's
  ruling on question 2): no collisions in `formats-raw.yaml` or the exhibits
  vocabulary.

## Phase [Palm-Kernel], bump 1.0.0a362

`freq_pgf_prime(n, z)` on `Frequency`, two routes, scoped per the author's
ruling (Poisson and the common mixed Poissons suffice; no universal
coverage):

1. **The `(a, b, 0)` identity.** The Panjer recursion `k p_k = (a k + b)
   p_{k-1}` sums to `P_N'(z) = (a + b) P_N(z) / (1 - a z)`, algebraic in the
   already-implemented `freq_pgf` plus the stored `panjer_ab` constants (set
   in each family's `freq_moms`; verify coverage at execution). Poisson
   (`a = 0`, `b = lambda`, so `P' = lambda P`, the size-bias fact), binomial,
   negbin, and geometric come for free.
2. **Closed-form overrides**: fixed (`n z^(n-1)`), bernoulli, empirical and
   renewal (`evaluate_pgf_polynomial(atoms - 1, weights * atoms, z)`), and
   the common mixed Poissons via `P_N'(z) = n M_G'(n(z-1))`: gamma
   (`M_G'(s) = (1 - theta s)^(-a-1)` up to constants), delaporte and ig
   (product/chain-rule forms of the longhand PGFs at `_frequency.py:1223`,
   `:1249`).

Everything else (sig, beta, sichel, neymana, pascal) raises
`NotImplementedError` and the ladder stays marginal. A recorded future
option, deliberately not built now: the universal count-pmf fallback
(`P_N'(z) = sum k p_k z^(k-1)` off the existing `freq_pmf` FFT inversion,
evaluated with `evaluate_pgf_polynomial`) would cover every family; note it
in the docstring and move on. The zero-modification wrapper installs the
matching `c * G'` form in `_install_zm_wrappers`, applied ON TOP of
whichever route computed the base derivative; the layering order (route
first, zm wrapper second) gets its own test, since the `(a, b, 0)` identity
holds for the base PGF, not the zero-modified one.

`palm_conditional_mean` in `_aggregate_compute.py`, plus a thin
Aggregate-level convenience that assembles the pieces for an occurrence view
(severity image via `_rebucket_to_grid`, image measure scatter, base mean).

Tests (`tests/test_palm.py`, new):

- Brute-force check on a tiny discrete compound (enumerate outcomes, compare
  conditional means exactly).
- The Panjer self-check: `c = n = id` gives `kappa(s) = s` on the grid.
- Cross-check against the 2-D route: on the `Tower` exhibit fixture, Palm
  kappa at the net 1-in-100 agrees with the `occ_bivariate`-derived
  diversified cell within a tolerance set at implementation (the 2-D joint
  runs on a budget grid; expect roughly percent-level agreement, and record
  the observed gap in the test comment).
- Route agreement on negbin, which both the `(a, b, 0)` identity and the
  gamma-mixed closed form reach (same law, two parameterizations,
  `_frequency.py:878` vs `:1199`).
- Zero-modified Poisson derivative agrees with a numerical derivative of the
  wrapped `freq_pgf` (the route-then-wrapper layering test).
- `NotImplementedError` path for an unsupported family (e.g. sichel).

No exhibit change in this phase; no snapshot movement.

## Phase [Palm-Ledger], bump 1.0.0a363

The Palm route fills the **whole scenario ladder** on eligible stitched
builds, not just the waterfall cells: the builders compute, at build time
with the engine in hand, `E[row | net result at its q-quantile]` for every
ledger row at every `PERCENTILE_LADDER` point, attach it as `_palm_ladder`,
and `economic_df` serves it under `κ` headers where today it falls back to
marginal `P` headers. The waterfall's `M01 diversified` column then follows
with **no waterfall change at all**, since `_waterfall_frames` already reads
the ledger's `κ01` cell.

Eligibility, all three required, else the ladder stays marginal exactly as
today: `_pnl_recipe['kind']` in `('gcn', 'plain')`; no `variable_terms` and
no `reinstatement_terms`; a frequency whose `freq_pgf_prime` is implemented
(catch `NotImplementedError`). All-or-nothing per build: no half-filled
ladders.

Row coverage, by row type (this is why full coverage is feasible):

- premium / commission / expense-share legs: constants, conditional mean is
  the constant;
- loss and recovery legs: per-event functions (identity; single-layer ceders
  from `make_ceder_netter([(s, y, a)])`; loss-basis LAE is `rate * l`, a
  per-event scaling), via the kernel;
- aggregate-tier legs: deterministic transforms of the retained aggregate,
  via the level-set transport through `T(u) = u - a(u)` (no FFT);
- group / tier totals, results, running nets, total impact: sums and
  differences of the above (linearity), so every column foots by
  construction;
- the grand-result row: its own quantiles, the footing anchor.

The 2-D per-atom route remains primary wherever it runs today: **no existing
ladder or diversified number moves**, and the plan explicitly does not
attempt the deeper refactor (replacing the `is2d` recovery leg with a 1-D
kappa leg to avoid `occ_bivariate` entirely); that is noted as a future item
only.

Tests:

- The `Peel` fixture's `economic_df` carries `κ` headers and every column
  foots; its waterfall diversified column populates and foots to the closing
  row (the headline acceptance).
- `test_waterfall_blanks_diversified_without_shared_atoms` reworks: `Peel` no
  longer blanks, so the blank case moves to a new swing-rated fixture
  program (author's ruling on question 5), asserted to keep the marginal
  ladder, blank diversified, and the caption rider.
- Tower / capstone-shaped builds byte-identical (the 2-D route still serves
  them).
- Snapshot regen: the diff carries the `Peel` economic ladder switching
  `P` to `κ`, its waterfall cells populating, its caption losing the "blank
  here" rider, the new swing fixture's blocks, and nothing else.

## Phase [Ledger-Both-Ladders], bump 1.0.0a364

The "ledger-sa and ledger-div" ruling (question 1). New property
`PnL.economic_marginal_df`: the same sheet as `economic_df` (same row index,
same `EX / SD / CV / Skew`), with the ladder always **marginal** under `P`
headers, computed from each row's own GridDistribution (the quantiles the
docstring today says are "one line away via `density_df[row].q(p)`", now on
a frame). Always available; does not foot, by nature, and its caption says
so.

The `economic` exhibit serves two blocks when the ladders differ:
`('economic_df', ...)` with the scenario ladder and
`('economic_marginal_df', ...)` with the marginal one, captions
distinguishing "the state the book is in" from "each row on its own". When
`economic_df` is itself marginal (an ineligible build), the second block is
suppressed rather than served twice. The insurer perspective's narrowed
economic view (`a305`) is inspected at execution; the default is to leave
insurer untouched and let RAW carry both, noting the option.

Tests: block presence per fixture kind (Tower and Peel serve both blocks;
the swing fixture serves one); the marginal frame's grand-result row equals
`economic_df`'s (both are that row's own quantiles); `economic_ratios_df`
untouched; snapshot regen (new block bodies); `FEATURES.csv` gains the row.

## Phase [Writer-Standalone-CoC], bump 1.0.0a365

One presentational phase, because the three changes share the same frames,
captions, tests, and snapshot.

**Standalone.** In `_waterfall_frames`: on a risk-bearing step
`standalone = gd.q(p)` as today; on a ceded step `standalone = gd.q(1 - p)`,
the step's own result at the right-tail percentile, the writer's 1-in-100,
positive almost always. Tier-subtotal steps containing any cover (e.g. the
all-occ subtotal) follow the ceded reading, per the author's ruling on
question 4, consistent with a361's `_step_is_ceded` router, which is
retained. This supersedes the a361 blank.

**Ratio columns.** `evaluation_df` columns become
`['Premium spent', 'Margin spent', 'CR', 'MSD', 'SA CoC', 'Div CoC']`:

- `MSD` is the former `M / SD`, unchanged arithmetic, renamed (the author's
  broking-era name, margin to standard deviation).
- `SA CoC` is `_capital_ratio(margin, standalone)` on every row. On a ceded
  row both inputs flip sign (margin < 0, standalone > 0), so the quotient is
  positive: the cost of the layer per unit of the writer's standalone
  capital.
- `Div CoC` is `_capital_ratio(margin, divers)` on every row; the a361
  ceded/risk column split reverts and `Cost of relief` is removed. The sign
  convention carries the reading: risk rows have margin > 0 and M01 < 0
  (a return on capital); ceded rows have margin < 0 and M01 > 0 (a cost of
  relief). The captions state this; the a361 `VALIDATION_NOISE` guard in
  `_capital_ratio` is unchanged.

**Formats** (`formats-raw.yaml`): remove `'M / M01 standalone'`,
`'M / M01 diversified'`, `'Cost of relief'`; add `MSD: '.3f'` (the `PQ` /
`VaR/Mean` precedent for a multiple read to three places), `'SA CoC': ratio`,
`'Div CoC': ratio` (`.1%`). `PENDING_VOCABULARY` in `tests/test_exhibits.py`
drops `'M / SD'` (no longer served; the ratchet requires the removal).

**Captions.** Both waterfall captions rewrite: the walk caption explains the
two-sided standalone convention (own left tail on risk rows, own right tail,
the writer's adverse state, on ceded rows) and that the sign flip is the
marker; the evaluation caption is the rubric the author asked for, spelling
out MSD, SA CoC, and Div CoC, the sign expectation (margin > 0 and M01 < 0 on
gross and net, the reverse on ceded rows), and that the reinsurance test is a
ceded row's CoC against the return the risk rows earn. Docstrings for
`walk_df` / `evaluation_df` follow; `FEATURES.csv` rows 24 and 25 re-edit.

**Tests.** The a361 waterfall tests rework once more:
`test_waterfall_capital_ratio_definition` asserts the single-column identity
on all rows; `test_waterfall_ceded_step_blanks_standalone_and_reads_relief`
becomes the sign-convention test (ceded rows: margin < 0, both M01 > 0, both
CoC finite and positive; risk rows the mirror); the closing-row identity test
survives unchanged. Snapshot regen; the diff carries the renamed columns, the
repopulated ceded standalone cells, and the two captions.

Acceptance sketch on the author's capstone program (exact values fixed at
implementation): Div CoC column reads 0.2332 / 1.3955 / 0.1541 / -0.2136 top
to bottom (the a361 acceptance numbers, now in one column); MSD gross 0.451;
SA CoC on the XOL and QS are new positive numbers from the right-tail
standalone, recorded in the test when first computed.

## Phase [PnL-Overview-Punchups], bump 1.0.0a366

Items B1 and B2, one small bump.

**Validation.** New property `PnL.engine_validation_df` (name confirmed,
question 3) delegating exactly as `stats_df` does (`pd.DataFrame()` when
`engine is None`). The PnL `validation` exhibit registration is replaced by a
custom frames function serving `('engine_validation_df', ...)` as the first
block, and the existing ledger audit `('validation_df', ...)` as a second
block only when non-empty. The existing `PnL.validation_df` property and its
pinning test are untouched. The RAW invariant (block name is an attribute on
the served object, `test_exhibits.py:157-180`) is satisfied by the new
property name.

**Tail.** `register_simple_exhibit('tail', 'Return periods', 'tail_df',
[PnL], caption=...)` with a PnL-specific caption noting the payoff
orientation (the adverse tail is the low one; the frame is over the closing
margin). `PnL.tail_df` already exists and needs no change.

Tests: `EXPECTED_EXHIBITS` for `'PnL'`, `'Tower'`, `'Peel'` gain `'tail'`
(order per `available_exhibits`); a content test that the served validation
block equals `pn.engine.validation_df` (mirroring the stats test at
`test_exhibits.py:543-556`); snapshot regen (new `tail` and `validation`
bodies for the PnL fixtures).

## Phase [PnL-Reins-Passthrough], bump 1.0.0a367

Items B3 and B4, one bump.

- `PnL.reins_stats_df` and `PnL.reins_summary_df`: delegating properties over
  `self.engine` (`None`-safe). If the insurer override's extra blocks
  (`reins_layer_terms`, `reins_layer_moments`, `exhibits\_aggregate.py:160-215`)
  are wanted for PnL too, add the same thin delegations; decide at execution
  by what the insurer frames function actually reads.
- Availability: extend `_has_reinsurance` (`exhibits\_core.py:348-356`) with a
  PnL branch that looks through `obj.engine`, mirroring the charts'
  `_engine()` pattern.
- Register the `reins` exhibit for PnL, RAW via the shared `_reins_frames`
  (which now works because the delegating properties exist on the PnL), and
  the insurer perspective likewise.
- Register the `reins` chart for PnL in `charts\_emit_reins.py` via the
  `_engine()` pattern from `_emit_structure.py:112-120`, predicate: engine
  present with an occurrence program. Update the "Aggregate only, at 1.0"
  comment at `_emit_reins.py:21-24` to record the 2026-09-29 supersession for
  PnL (the a244 portfolio-level objection does not apply: an xpnl wraps a
  single Aggregate).

Tests: `EXPECTED_EXHIBITS` for `'Tower'` and `'Peel'` gain `'reins'` (the
plain `'PnL'` fixture has no cession and must NOT gain it, which is itself an
assertion); a chart-availability test that `available_charts(tower)` includes
`'reins'` and the plain PnL's does not; served frames equal the engine's;
snapshot regen.

## Phase [API-Residue], aggregate-api commit (no LIB bump)

In `V:\dev\aggregate-api`: update `web/src/nav.js` residue only, the tabs
light themselves from the capability payload. The `tail` leaf's `why` line
("needs a full loss distribution, so an aggregate or a portfolio") extends to
the P&L; the Re group's comment block recording the 2026-09-25 "Diagram and
nothing else" ruling is updated to record the 2026-09-29 supersession; the
Plot leaf's comment about `chart_reins` being Aggregate-only follows. Version
and CHANGELOG mechanics per that repo's own rules, checked at execution;
committed by the same session, separately from the LIB bumps.

## Acceptance checks

1. `uv run --no-sync pytest tests/test_exhibits.py tests/test_exhibit_formats.py
   tests/test_palm.py tests/test_create_pnl.py` green at each phase; the full
   fast suite once per bump; tier 3 at each commit boundary per the
   version-bump skill.
2. The `Peel` fixture's `economic_df` carries a footing `κ` ladder and its
   waterfall serves a populated, footing `M01 diversified` column (phase
   [Palm-Ledger]).
3. Palm and the 2-D route agree on the `Tower` fixture within the recorded
   tolerance (phase [Palm-Kernel]).
4. Tower and Peel serve both economic blocks, scenario and marginal; the
   swing fixture serves one (phase [Ledger-Both-Ladders]).
5. The capstone program serves the Div CoC column 0.2332 / 1.3955 / 0.1541 /
   -0.2136 and an MSD gross of 0.451 (phase [Writer-Standalone-CoC]).
6. In the SPA, a reinsured xpnl session lights Overview->Validation (with
   engine content), Overview->Tail, Re->Plot, Re->Summary/Stats/Density
   (phases a366 to a367; verified by exhibit and chart availability tests in
   LIB, eyeballed once in the app).
7. Every snapshot diff read deliberately; each phase's diff carries only that
   phase's cells, columns, and captions.

## Bookkeeping

One bump, one commit, per phase, house format, labels as above:

```
[Palm-Kernel] a362: freq_pgf_prime and the 1-D conditional mean kernel
[Palm-Ledger] a363: scenario ladder on eligible stitched ledgers via Palm
[Ledger-Both-Ladders] a364: economic_marginal_df serves the P ladder beside kappa
[Writer-Standalone-CoC] a365: writer-side standalone, MSD / SA CoC / Div CoC
[PnL-Overview-Punchups] a366: engine validation pass-through and the tail exhibit
[PnL-Reins-Passthrough] a367: reins exhibit and chart serve through the engine
```

Each carries its code, tests, `pyproject.toml`, `CHANGELOG.md` section,
snapshot regen where applicable, and `dev/FEATURES.csv` where the member
surface moves (`freq_pgf_prime`, `economic_marginal_df`, the three new PnL
properties, the renamed evaluation columns). The plan moves to `dev/done/`
with the final LIB bump. The a363 CHANGELOG entry states that stitched
ledgers' ladder headers switch from `P` to `κ` where eligible; the a365
entry states plainly that `evaluation_df` renames three columns and drops
one, and that `M01 standalone` on ceded rows changes from blank to a
positive right-tail number, all breaking for a caller indexing the frames;
it supersedes part of a361's presentation.

Not pushed. The author pushes.

## Execution log (Claude, 2026-09-29)

Divergences and observations recorded at the moment they were made, per the
execute-plan hygiene. Each phase's gates and results are in the run summary
and the CHANGELOG.

- **[Palm-Kernel] (a362).** The Aggregate-level convenience is named
  `Aggregate.palm_kappa(target, conditioning)` (the plan left it unnamed);
  it returns `(kappa, conditioning_density)` on `self.xs` and raises
  `NotImplementedError` on signed or windowed grids. The Tower cross-check's
  observed Palm vs 2-D gap is 1.4% relative (Palm 521.8 vs joint 514.6),
  recorded in the test with a 2% tolerance. The logarithmic family is
  explicitly excluded from the `(a, b, 0)` route (it stores `panjer_ab` but
  is `(a, b, 1)`), with its own raise test.
- **[Palm-Ledger] (a363), ruling 5 superseded.** The plan's premise that a
  swing-rated walk is blank-diversified is contradicted by the code: a swing
  `xpnl` walk is per-atom over a shared source and carries a populated `κ`
  ladder, and a swing-decorated peel is refused by the builder. Asked
  2026-09-29; the author ruled: the ineligible fixture is the Peel program
  under a **logarithmic** frequency (`PeelMarginal`, exercising the
  `freq_pgf_prime` eligibility gate), **plus** a pin test that the swing
  walk keeps its populated `κ` ladder
  (`test_swing_walk_keeps_its_kappa_ladder`).
- **[Palm-Ledger] transport subtlety.** Under aggregate covers the stage-2
  level-set transport uses the same linear rebucket scatter that built the
  served final-net density, so every column foots by construction; the
  grand-result cell is then the level-set average of the true pre-scatter
  net values, agreeing with its own marginal quantile to within one bucket
  of scatter. With no aggregate covers (the `Peel` fixture) the read is
  direct and the closing-row identity is exact. Documented in
  `_palm_scenario_ladder`'s Notes.
- **[Palm-Ledger] extra test.** Cross-route agreement added beyond the plan:
  the peel's `All occurrence` tier-subtotal diversified cell agrees with the
  lumped tier walk's 2-D-joint cell to 0.11% observed
  (`test_palm_tier_subtotal_agrees_with_the_lumped_tier_walk`, 2%
  tolerance).

- **[Ledger-Both-Ladders] (a364).** As the plan's default ruled: the insurer
  economic view is untouched (it reads the first RAW block only), RAW
  carries both sheets. The RAW `economic` passthrough became a custom
  frames function in `exhibits/_pnl.py` because the manifest foot would
  otherwise re-register over it; the single-block (marginal) caption now
  states the regime instead of claiming kappa columns it does not serve.
- **[Writer-Standalone-CoC] (a365).** The capstone acceptance numbers came
  out exactly as the plan projected (Div CoC 0.2332 / 1.3955 / 0.1541 /
  -0.2136, MSD gross 0.451); the new SA CoC ceded cells are XOL 0.3750
  (1800 / 4800) and QS 0.1541 (776.25 / 5036.25, agreeing with Div CoC
  because a quota share of the whole book is comonotone with it), recorded
  in `test_waterfall_capstone_acceptance`.
- **[PnL-Overview-Punchups] (a366).** The engine validation frame on ceded
  books serves `Ceded EX / CV / Sk` columns, added to the vocabulary gate's
  pending set beside the existing Gross / Net / Subject triplets. The
  `PeelMarginal` fixture (created a363) gains the tail exhibit alongside
  the plan's named fixtures, being a P&L like the rest.
- **[PnL-Reins-Passthrough] (a367).** The insurer override's extra blocks
  needed **no** extra delegations: `_reins_insurer_aggregate` reads only
  the served frames, so it is registered for PnL by stacking the decorator
  (the execution-time decision the plan asked for). `PeelMarginal` gains
  the reins exhibit too (its engine carries the same occurrence program).
  The `_has_occurrence` chart predicate was generalized through the
  charts' `_engine` helper rather than a parallel PnL branch.
- **[Bivariate-Gate-Flake] observed at the a367 gate.**
  `test_netceded_refuses_a_pin_it_cannot_honor` failed twice under the full
  parallel tier-3 run and passed in isolation and on the rerun (5426
  green); it builds a `2^14 x 2^14` joint, the memory-pressure class the
  testing notes already document for the bivariate suites, and touches
  nothing this plan changed.

## Rulings (author, 2026-09-29)

1. **[palm-scope, now ledger-both-ladders]** Confirmed that `economic` is
   the ledger with the P-or-`κ` ladder switch. Ruling: show **both** ladders
   ("ledger-sa and ledger-div"): the Palm route fills the full scenario
   ladder on eligible builds ([Palm-Ledger]) and a new
   `economic_marginal_df` carries the marginal ladder beside it
   ([Ledger-Both-Ladders]).
2. **[column-spellings]** `MSD`, `SA CoC`, `Div CoC`, exactly as typed.
3. **[validation-block-name]** `engine_validation_df` confirmed.
4. **[subtotal-standalone]** Confirmed: a tier subtotal containing any cover
   (e.g. the all-occ subtotal) takes the ceded, right-tail standalone
   reading.
5. **[blank-fixture]** Confirmed: add a swing-rated fixture program as the
   ineligible (blank-diversified, marginal-ladder) test case.
6. **[frequency-scope]** Poisson and the common mixed Poissons suffice
   (gamma, delaporte, ig, plus what the `(a, b, 0)` identity gives free);
   the universal count-pmf fallback is recorded as a future option, not
   built.
