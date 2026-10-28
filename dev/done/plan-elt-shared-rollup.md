---
bibliography: D:/Projects/Biblio/uber-library.bib
csl: D:/Projects/Biblio/journal-of-risk-and-uncertainty.csl
---

# Plan: roll up each shared ELT event into one row

`[ELT-Shared-Event-Rollup]`, the shared-event half of `[DecL-Loss-Tables]`.
Drafted 2026-10-28 against aggregate 1.18.2 and eltq 2.14.0 (Rust backend).
**Status: ruled 2026-10-28 (rulings 23 to 27), executed as 1.19.0 on
2026-10-28; see the execution log.** One minor bump. It replaces the pending step
`[ELT-Shared-Event-Uncertainty]` of `dev/done/plan-elt.md` and
`dev/done/plan-elt-uncertainty.md`. The joint distribution of two units
(`[ELT-Joint-Units]`) is pended; what was learned about it is in
`dev/note-joint-elt.md`.

## Reading this plan

An ELT6 row is `[id rate mean max sdi sdc]`: event ID, annual rate
$\lambda$, mean loss $\mu$, maximum (exposed) loss $M$, and the independent
and correlated standard deviations $\sigma_I$, $\sigma_C$. A row with
secondary uncertainty is a beta on $[0, M]$ with mean $\mu$ and standard
deviation $\sigma=\sigma_I+\sigma_C$ (the ELT convention), the two-point law
on $\{0, M\}$ when that variance is inadmissible
($\sigma^2\ge\mu(M-\mu)$, eltq's `variance_policy='bernoulli'`), or an atom
at $\mu$ when both SDs are zero. The annual loss is compound Poisson with
rate $\Lambda=\sum_e\lambda_e$ and severity the rate-weighted mixture of the
event-total laws (`[ELT-Poisson-Mixture]` in `dev/done/plan-elt.md`).

A **shared event** is one event ID with rows for two or more units, written
`agg X elt unit A [...] elt unit B [...]`. The event total is
$S=\sum_uX_u$. The parent plans model $S$ with Burke's coupling
[@Burke2016], the one-factor Gaussian copula of eltq's spec, section 4A
(`V:/dev/eltq/secondary-uncertainty-mathematical-specification.md`). That
route needs a windowed multi-unit quantization in eltq
(`[Eltq-Windowed-Shared-Events]`), which was measured on 2026-10-27 at one
to three orders of magnitude over budget, so today a shared ELT6 builds with
exact model moments and refuses `update`.

The rulings of `dev/done/plan-elt.md` (1 to 17) and
`dev/done/plan-elt-uncertainty.md` (18 to 22) bind except where the rulings
below supersede them.

## Goal

A shared ELT6 updates like any other ELT6. Each shared event is rolled up
into one row before the model is built, and the 1.16.0 one-row path (closed
form model moments, the `[ELT-Grid]` tail check, eltq quantization at
`update`) runs on the rolled rows unchanged. Tables without a shared event
are bitwise unchanged. The Burke route stays in the code, off the default
path.

## Rulings

Author, 2026-10-28:

23. **[ELT-Shared-Event-Rollup]** The default model of a shared event is the
    roll-up: the event's rows become one row with
    $\mu_R=\sum_u\mu_u$, $M_R=\sum_uM_u$, $\sigma_{C,R}=\sum_u\sigma_{C,u}$,
    $\sigma_{I,R}=(\sum_u\sigma_{I,u}^2)^{1/2}$ and the event's common rate,
    and the rolled row is classified and quantized as a one-row event. This
    supersedes `[ELT-Shared-Events]` of `dev/done/plan-elt.md` for the
    univariate total.
24. **[ELT-Rollup-Name]** The rule is called the roll-up, in code, messages,
    `info`, documentation and the CHANGELOG. It is never called Burke's:
    Burke's coupling is the Gaussian copula, and eltq's name for the roll-up
    of an event total is `coupling='homer'` [@Homer2017a, section 3.2.2], the
    same rule as @MitchellWallace2017 [section 1.10.5.10].
25. **[ELT-Burke-Total]** Pended. The Burke total (exact moments of each
    event total under Burke's coupling, from
    `eltq.multiunit_moments(coupling='burke')`, with no grid) is too complex
    to pursue now. Its branch of `_elt_model` stays in the code, reachable
    only through the private constant `SHARED_EVENT_RULE`, and
    `dev/TODO.md` records it as pended. The coupling itself lives on in
    eltq as `coupling='burke'`.
26. **[Rollup-Stated-SDs]** The roll-up uses the stated SDs, including those
    of an inadmissible unit row ($\sigma^2\ge\mu(M-\mu)$, a two-point law in
    the unit model). Rows are not shrunk to their admissible bound first; that
    rule has no published source. The audit counts such rows.
27. **[ELT-No-Moment-Matching]** The event's beta is not fitted to the exact
    moments of the Burke-coupled total. That alternative was considered
    (the beta on $[0, M_R]$ with $\mathsf PS$ and $\mathsf PS^2$ of the
    coupled model, eltq spec section 4B) and declined: it needs per-event
    moments from eltq and departs from the published convention.

## What the roll-up is, and is not

**Exact for the linear model.** Under Burke's linear model
$X_u=\mu_u+\sigma_{C,u}Z_c+\sigma_{I,u}Z_u$ the total is
$S=\mu_R+\sigma_{C,R}Z_c+\sigma_{I,R}Z'$ with $Z'$ standard normal, so the
component rule of ruling 23 is exact for that model. The rolled row is then
fitted with the ELT convention $\sigma_R=\sigma_{I,R}+\sigma_{C,R}$.

**Approximate for the coupled model.** Against the beta-marginal Burke model
the roll-up is exact in the mean and in the support $[0, M_R]$, and
approximate in the variance and tail of $S$, in no fixed direction, for three
reasons given in the eltq spec, section 4A: the convention
$\sigma_I+\sigma_C$ against $(\sigma_I^2+\sigma_C^2)^{1/2}$ (inflates), copula
attenuation of the loss correlation below the score correlation $w_uw_k$
(deflates), and the beta family not being closed under addition (shape).
It is not the beta fitted to the moments of the coupled total (ruling 27).

**Feasibility.** If every unit row of an event is an admissible beta, so is
the rolled row:
$$
\sigma_R=\Bigl(\sum_u\sigma_{I,u}^2\Bigr)^{1/2}+\sum_u\sigma_{C,u}
\le\sum_u\sigma_u
<\sum_u\bigl(\mu_u(M_u-\mu_u)\bigr)^{1/2}
\le\bigl(\mu_R(M_R-\mu_R)\bigr)^{1/2},
$$
the last step by Cauchy-Schwarz. The roll-up never turns admissible rows
into a two-point row. The reverse can happen: inadmissible unit rows (two-point
laws in the unit model) can roll into an admissible beta, because the
roll-up uses the stated SDs. That is the published convention (ruling 26);
the audit counts such rows (`[Rollup-Reports]`).

**Worked numbers**, all at rate $0.1$, two units A and B, computed
2026-10-28 with eltq 2.14.0:

| Case | A row | B row | Rolled row | Rolled severity moments | Burke severity moments |
|---|---|---|---|---|---|
| clean | `30 300 20 10` | `20 200 15 10` | `50 500 25 20`, beta(1.01111, 9.1) | 50, 4525, 562511.47 | 50, 4341.85, 510574.37 |
| `LT.Comonotone` | `50 100 0 60` | `20 100 0 60` | `70 200 0 120`, two-point | 70, 14000, 2800000 | 70, 11000, 1900000 |
| `LT.Independent` | `50 100 60 0` | `20 100 60 0` | `70 200 84.853 0`, beta(0.0923611, 0.171528) | 70, 12100, 2236644.17 | 70, 9000, 1300000 |

Rows are `mean max sdi sdc`. In the clean case the roll-up SD is 45 against
Burke's 42.92 (4.9% high) and the third moment is 10% high. The two corpus
cases have inadmissible unit rows, so their Burke model is two-point units,
and the roll-up differs by much more; they are kept as regression cases, not
as illustrations. eltq's `multiunit_moments(coupling='homer')` returns the
rolled moments to within one unit in the last place, which makes it an
independent oracle.

## Current behavior, 1.18.2

- `_elt_model` (`src/aggregate/_loss_table.py:891`) sorts positive-rate rows
  stably by event (:932), sets `shared = bool((np.bincount(event_code) > 1).any())`
  (:946), and calls `eltq.multiunit_moments(..., coupling='burke')` when
  shared (:947), else `eltq.moments_arrays(...)` (:955).
- `EltModel` (:114) holds the input rows (`event_ids`, `event_code`,
  `unit_code`, `units`, `rate`, `mean`, `max`, `sdi`, `sdc`), `shared`,
  `moments`, `digest` (SHA-256 of the rows, :972) and the eltq version and
  backend. `columns()` (:183) returns the five columns as eltq keywords and
  feeds every one-row eltq call.
- Validation in `_lower_elt` (:744) already refuses two rows for one unit in
  one event (:802) and different rates within an event (:811, tolerance
  `RATE_RTOL = 1e-12`); the canonical rate is the event's first row.
- `SeverityElt.check_grid_supported` (`src/aggregate/_severity.py:3517`)
  raises `NotImplementedError` for a shared model. It runs from
  `Aggregate.update` (`src/aggregate/_aggregate.py:4524`), from
  `SeverityElt.tail_measures` (:3568) and from `SeverityElt.quantize`
  (:3706), which covers a `Portfolio` unit.
- `_elt_severity` (`src/aggregate/_bucket_window.py:805`) skips a shared
  severity, so the `[ELT-Grid]` tail check never runs on one.
- `describe` (`_loss_table.py:335`) gives `info` the text
  `secondary uncertainty, shared events (model moments only)`; the
  validation frame adds a `moment integration` row for a shared table
  (:1498); `_grid_rows` (:1241) documents that a shared table never reaches
  a grid.
- ELT3, ELT4 and `no2u` tables are deterministic: shared events already sum
  their means into one atom per event. They are not affected by this plan.

## Design

### [Rollup-Model] where the roll-up happens

In `_elt_model`, after the stable sort and before the eltq call. The input
rows stay on `EltModel` unchanged: they identify the table, feed the digest
and `shared_example`, and stay available to the pended `[ELT-Joint-Units]`
(`dev/note-joint-elt.md`), which passes unit rows to eltq.

- **Rule constant.** A module constant `SHARED_EVENT_RULE = 'rollup'`, with
  `'burke'` the kept alternative (ruling 25). It is read at call time so a
  test can monkeypatch it. It is not public: no DecL word, no `build`
  keyword, no config setting.
- **Rolled columns.** When the table is shared and the rule is `'rollup'`,
  with `starts` the first row of each event and `count` its number of rows:
  `rate[starts]`, `np.add.reduceat` of `mean`, `max` and `sdc`, and
  `sdi` as `np.sqrt(np.add.reduceat(sdi**2, starts))` where `count > 1` and
  `sdi[starts]` where `count == 1`. A one-row event passes through untouched,
  so no arithmetic touches it, and event IDs are `event_ids[starts]`.
  Overflow of `sdi**2` falls under the existing moment overflow refusal
  (:964), which asks the user to rescale.
- **Moments.** `eltq.moments_arrays` on the rolled columns, with the rolled
  event IDs, exactly as the one-row branch calls it today.
- **New `EltModel` fields.** `shared_rule` (`None` for a table without a
  shared event, else `'rollup'` or `'burke'`) and `rolled_columns` (the five
  rolled columns as read-only arrays, `None` unless the rule is
  `'rollup'`). `columns()` returns `rolled_columns` when present, else the
  input columns; every one-row eltq call (moments, tail measures,
  quantization) therefore runs on one row per event.
- **Digest.** Add `shared_rule` to the digest only when it is not `None`, so
  the digest of a table without shared events is unchanged and a change of
  rule changes the digest of a shared one.
- `meta['secondary']` gains `'shared_rule'` and `'shared_events'` (events
  with two or more rows) and `'shared_rows'` (the rows they hold).

### [Rollup-Grid] update reaches the grid

- `SeverityElt` gains the property `grid_supported`, false only when
  `shared_rule == 'burke'`. `check_grid_supported` raises only then. Its
  message is rewritten: the Burke total route has no grid, and the default
  roll-up has one.
- `_elt_severity` tests `s.grid_supported` instead of `not s.shared`, so the
  tail check runs on rolled tables.
- `SeverityElt.shared` keeps its meaning (the table has a shared event) for
  reports.
- `tail_measures` and `quantize` need no change beyond their docstrings:
  they already take `columns()`.

### [Rollup-Reports] info, audit and validation

These rows land in the three `loss_table_*_df` frames as they stand; the
next plan, `[ELT-Reporting-Punchup]`, moves them into the narrative pair.

- `info`: `secondary uncertainty, {k:,d} shared events rolled up` under the
  roll-up; the Burke text stays for the Burke rule.
- `loss_table_summary_df` gains one row, present only for a shared table:
  `('table', 'shared events')`, value
  `{k:,d} events from {r:,d} rows, rolled up; {j:,d} of the rows have
  inadmissible variance`, where $j$ counts input rows of shared events with
  positive SD and $\sigma^2\ge\mu(M-\mu)$.
- eltq's row classification (`beta_rows`, `two_point_rows`, `atom_rows`)
  counts model rows, which are rolled rows; the summary row above holds the
  input count.
- `loss_table_validation_df`: the `moment integration` row only under the
  Burke rule (the condition changes from `shared` to
  `shared_rule == 'burke'`).
- `loss_table_stats_df` needs no change: its `table` column is the model
  moments, now the rolled ones.

## Steps

### [ELT-Shared-Event-Rollup], one minor bump

1. `[Rollup-Model]`, `[Rollup-Grid]` and `[Rollup-Reports]` as above.
2. Tests (below).
3. Documentation: rewrite the **Shared events.** paragraph of
   `docs/4_dec_Language_Reference.rst` (line 178) as below, and the `loss
   table` row of `docs/2_aggregate_overview/info-strings.rst` (lines 27 to
   32).
4. Corpus: in `src/aggregate/agg/decl-testers.agg`, the comment at line 1472
   (a shared event no longer refuses `update`) and the notes of
   `LT.Comonotone` and `LT.Independent` (lines 1480 and 1481), which state
   Burke moments; replace them with the rolled moments of the table above.
   Add `LT.Rollup`, the clean case.
5. Release: `__version__`, `CHANGELOG.md`, `dev/TODO.md`, `dev/FEATURES.csv`
   via `dev/regen_features.py`; move this plan to `dev/done/`.

Proposed documentation paragraph:

> **Shared events.** When one event has rows for two or more units, the rows
> are rolled up into one row before modelling, the convention of Homer and Li
> and of Mitchell-Wallace et al.: means, maximums and correlated SDs add, and
> independent SDs add in quadrature, :math:`\sigma_{I,R} = (\sum_u
> \sigma_{I,u}^2)^{1/2}`. The rolled row is then classified and quantized
> like any one-row event, so the table updates normally and ``info`` shows
> ``shared events rolled up``. The roll-up is exact in the mean and
> approximate in the variance and tail of an event total. Units whose events
> are all distinct (separate wind and quake catalogs, say) are not shared.

No grammar change, so no spec snapshot or grammar reference regeneration.

## Files

| Area | Files |
|---|---|
| Model | `src/aggregate/_loss_table.py` (`SHARED_EVENT_RULE`, `_elt_model`, `EltModel`, `describe`, the summary and validation frame builders, `_grid_rows` docstring) |
| Severity and grid | `src/aggregate/_severity.py` (`SeverityElt.grid_supported`, `check_grid_supported`, docstrings of the class, `tail_measures`, `quantize`), `src/aggregate/_bucket_window.py` (`_elt_severity`), `src/aggregate/_aggregate.py` (the comment at the refusal call, :4524) |
| Tests | `tests/test_elt_uncertainty.py` |
| Documentation | `docs/4_dec_Language_Reference.rst`, `docs/2_aggregate_overview/info-strings.rst` |
| Corpus | `src/aggregate/agg/decl-testers.agg` |
| Release | `src/aggregate/__init__.py`, `CHANGELOG.md`, `dev/TODO.md`, `dev/FEATURES.csv` |

`dev/TODO.md`, entry `[DecL-Loss-Tables]` (lines 29 to 37): shared events
are rolled up as of this bump; `[ELT-Shared-Event-Uncertainty]` is
superseded; `[ELT-Burke-Total]` is pended, its branch kept behind
`SHARED_EVENT_RULE`; open are `[ELT-Grid-Retune]` and
`[ELT-Reporting-Punchup]` (`dev/plan-elt-reporting-punchup.md`); pended is
`[ELT-Joint-Units]` (`dev/note-joint-elt.md`).

## Names checked

On 2026-10-28, `rg` over `src/aggregate` found no use of `shared_rule`,
`SHARED_EVENT_RULE`, `rolled_columns`, `grid_supported` (as a whole word;
`check_grid_supported` is the existing method), `shared_events` or
`shared_rows`. `SeverityElt` has no attribute of those names to shadow.
Recheck at execution.

## Acceptance and tests

In `tests/test_elt_uncertainty.py`:

- **Rolled oracle.** The clean case builds and updates; severity moments
  equal $(50, 4525, 562511.4678899)$ to `rtol=1e-12`, and equal
  `eltq.multiunit_moments(coupling='homer')` on the input rows to the same
  tolerance; annual mean $5$ on the grid within `validation_eps`; `info`
  contains `1 shared events rolled up` (plural at every count, as the
  existing `{events} events` text is).
- **Corpus regressions.** `LT.Comonotone` gives two-point moments
  $(70, 14000, 2800000)$ and `LT.Independent` gives
  $(70, 12100, 2236644.1717791)$, both updating.
- **Burke kept.** With `SHARED_EVENT_RULE` monkeypatched to `'burke'`,
  today's `test_shared_event_coupling`, `test_shared_refusal_names_the_event`
  and `test_shared_refusal_as_portfolio_unit` pass unchanged (their message
  match is updated for the new wording).
- **Portfolio.** A `port` with a shared ELT6 unit updates (the flipped
  portfolio refusal test).
- **Three units and atoms.** An event with three units, one of them an atom
  (`sdi = sdc = 0`), alongside one-row events: the rolled row is checked by
  hand and the one-row events pass through bitwise.
- **Unchanged tables.** For `LT.Beta`, `LT.Disjoint` and `LT.Large1`, the
  digest, model moments and grid equal their 1.18.2 values (the existing
  tests, unchanged, plus a digest equality).
- **Feasibility.** Seeded random admissible unit rows (two to five units per
  event) always roll into an admissible row; a constructed pair of
  inadmissible rows rolls into an admissible beta and is counted in the
  summary row.
- **Tail check runs.** Two units with `[1 .1 10000 1e9 30000 0]` each roll
  to $M_R=2\times10^9$ and update with the tail passing far below $M_R$, as
  `LT.Large1` does.
- **Order.** Swapping the order of the `elt` clauses gives the same model
  moments to `rtol=1e-15` (summation order can move the last bit).

## Release gates

Edit loop: `tests/test_elt_uncertainty.py`, `tests/test_elt.py`,
`tests/test_loss_table_reports.py`. At the bump: the full fast suite, then
`uv run pytest -m 'slow or not slow' -W error::RuntimeWarning` (a numerics
change), with `UV_PROJECT_ENVIRONMENT=.venv`. Docs are not built.

## Deferred

- `[ELT-Burke-Total]` (ruling 25) and `[ELT-Coupling-Selection]` (a public
  choice of rule).
- `[ELT-Reporting-Punchup]` (`dev/plan-elt-reporting-punchup.md`), the next
  plan.
- `[ELT-Joint-Units]`, pended; see `dev/note-joint-elt.md`.

## Execution log

**1.19.0, 2026-10-28**, eltq 2.14.0 (Rust backend). Reviewed against 1.18.2: every cited path, line and symbol matched; the names `shared_rule`, `SHARED_EVENT_RULE`, `rolled_columns`, `grid_supported`, `shared_events`, `shared_rows`, `shared_audit` and `shared_inadmissible_rows` were rechecked with `rg` and collide with nothing. `coupling='homer'` in `multiunit_moments` dates from eltq 2.13.0, so the oracle test runs at `ELTQ_MIN_VERSION`.

**Measured.** Clean case (`LT.Rollup`): severity moments $(50, 4525, 562511.4678899084)$, equal to `multiunit_moments(coupling='homer')` to the last bit; automatic grid `bs` 1/32, `log2` 16, annual mean 4.999999886 (relative error $2.3\times10^{-8}$), tail check passed. `LT.Comonotone` $(70, 14000, 2800000)$, one two-point row, `bs` 1/128; `LT.Independent` $(70, 12100, 2236644.1717791413)$, one beta row, `bs` 1/128; both update with the tail check passing. Digests of `LT.Beta`, `LT.Disjoint` and `LT.Large1` equal their 1.18.2 values (captured before the edit, pinned in `DIGESTS_1_18_2`).

**Divergences.**

1. `[Rollup-Quadrature]` The plan specified `np.sqrt(np.add.reduceat(sdi**2, starts))` where `count > 1` and `sdi[starts]` where `count == 1`, and said an overflow of `sdi**2` falls under the existing moment overflow refusal. It does not: an overflowing square gives an infinite rolled SD, which eltq classifies or drops, and neither reaches the overflow refusal (which tests the moments, and a two-point law on a finite $M$ has finite moments). The code uses `np.hypot.reduceat(sdi, starts)`, the same quantity, which cannot overflow where the squares would and emits no `RuntimeWarning`. `reduceat` returns a one-element segment untouched (checked bitwise on 1,004 values including subnormals and zero), so the `count` branch is unnecessary and a one-row event still passes through with no arithmetic.
2. `[Rollup-Audit-Names]` The summary row needs the count $j$ of inadmissible input rows, which the plan did not place. `EltModel.shared_audit()` returns the three counts (events, rows, inadmissible rows), and `meta['secondary']` carries them as `shared_events`, `shared_rows` and `shared_inadmissible_rows`. An overflowing $\sigma^2$ counts as inadmissible under `np.errstate(over='ignore')`.
3. `[Rollup-Meta-Always]` The four new `meta['secondary']` keys are present on every ELT6 table (`None`, 0, 0, 0 without a shared event), so consumers need no `.get`. The `EltModel` digest of an unshared table is unchanged, but `Aggregate._spec_hash`, the display-only `id` row of `info`, serializes `_loss_table_meta` and so moves for every ELT6 aggregate.
4. `[Rollup-Rule-Guard]` A tuple `SHARED_EVENT_RULES = ('rollup', 'burke')` beside the constant, and `_elt_model` raises `ValueError` on any other value, so a mistyped monkeypatch cannot fall through to the one-row path silently. Tested in `test_burke_rule_moves_the_digest`, which also checks that the rule moves a shared table's digest (a test the plan did not list).
5. `[Rollup-Summary-Burke]` Under the Burke rule the summary row says `under Burke's coupling` in place of `rolled up`; the plan gave only the roll-up wording.
6. `[Rollup-Corpus]` Besides `LT.Rollup`, the corpus gains `LT.RollupThree` and `LT.RollupLarge`, the three-unit and large-exposure cases, under the project rule that a DecL program used in a test is mirrored in `decl-testers.agg`. The variants built only inside tests (`LT.RollupHand`, `LT.RollupSwap`, `LT.RollupThreeSwap`, `LT.RollupPort`, `LT.RollupRandom`) are not mirrored, as `LT.SharedPort` and `LT.ZeroMean` were not.
7. `[Rollup-Docs-Verb]` In the `loss table` row of `info-strings.rst`, "an ELT reads" became "an ELT shows" (house rule on verbs of a person).

**Gates.** Edit loop: `tests/test_elt_uncertainty.py` (43 passed, under `-W error::RuntimeWarning`), `tests/test_elt.py`, `tests/test_loss_table_reports.py` and `tests/test_loss_table_files.py` (81 passed), the corpus consumers (776 passed). Fast suite: 6,670 passed. `uv run pytest -m 'slow or not slow' -W error::RuntimeWarning` with `UV_PROJECT_ENVIRONMENT=.venv`: 6,877 passed, which covers tier 3. `dev/regen_features.py`: no drift, `dev/FEATURES.csv` unchanged. Docs not built.

**Finding, not acted on.** `LT.Comonotone` on its automatic grid warns on the annual CV and skewness estimates ($-1.6\times10^{-4}$, $-1.3\times10^{-3}$ against `validation_eps` $10^{-4}$). That is the grid chosen for any one-row two-point severity: the rolled row equals `[1 .1 70 200 0 120]`, which gives the same numbers, and the one-row `LT.TwoPoint` warns the same way at 1.18.2. It belongs to `[ELT-Grid-Retune]`.

## References

::: {#refs}
:::
