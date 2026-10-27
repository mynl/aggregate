---
bibliography: D:/Projects/Biblio/uber-library.bib
csl: D:/Projects/Biblio/journal-of-risk-and-uncertainty.csl
---

# Plan: secondary uncertainty for ELT aggregates, one-row events first

`[ELT-Uncertainty]`, the open half of `[DecL-Loss-Tables]`. Drafted
2026-10-27 against aggregate 1.14.0 and the local eltq 2.14.0. **Status:
ruled 2026-10-27 (rulings 18 to 22); executing from 2026-10-27, see the
execution log.** It plans the two open steps of `dev/done/plan-elt.md`,
`[ELT-Single-Event-Uncertainty]` in full and `[ELT-Shared-Event-Uncertainty]`
only as far as parsing, theoretical moments and a refusal at `update`.

## Reading this plan

`dev/done/plan-elt.md` (the parent plan) holds the mathematics and the
author's rulings 1 to 17; they bind here except where rulings 18 and 21
below supersede them. The sections cited by label are:

- `[ELT-Poisson-Mixture]`: one Poisson count per physical event, severity the
  rate-weighted mixture of event totals, annual cumulants
  $\kappa_r=C_r=\sum_e\lambda_e\mathsf PX_e^r$.
- `[ELT-Row-Laws]`: the beta row law $X=M\operatorname{Beta}(\alpha,\beta)$,
  the Bernoulli (two-point) replacement for inadmissible variance, and the
  atom cases, all as eltq classifies them.
- `[ELT-Shared-Events]` and `[ELT-Moment-Pass]`: Burke's coupling and the
  shared-event moment recursion.
- `[ELT-Support-In-Sizing]`: true support $B$ versus the numerical window,
  the tail measures $R(t)$ and $D_r(t,c)$, and the two large-exposure
  regression rows.
- `[ELT-Severity-Protocol]`, `[ELT-Source-Grid]`, `[Loss-Table-Audit]`,
  `[ELT-Probability-Oracles]` and `[ELT-Grid-And-Window-Tests]`.

The eltq half is `V:/dev/eltq/dev/done/plan-model-moments-and-windowed-views.md`
(the eltq plan). It delivered the grid-free moment and tail functions this
plan calls, and stopped at `[Eltq-Windowed-Shared-Events]` for a decision:
its measurement on 2026-10-27 put the windowed multi-unit route one to three
orders of magnitude over budget. Shared events therefore cannot be quantized
on a window yet, which is why this plan stops short of them.

## Goal

An `agg ... elt ...` declaration whose sources are ELT6 (`[id rate mean max
sdi sdc]`) without `no2u` builds an ordinary `Aggregate` with secondary
uncertainty, provided every event has one row. Its theoretical moments come
from eltq before any grid exists; `update` sizes the grid with the ordinary
routines, checks the severity tail at the chosen edge, and quantizes the
event severity with eltq on that grid. An ELT6 with a shared event (one event
ID carrying rows for two or more units) parses and builds with its
theoretical moments, then raises `NotImplementedError` at `update`.

## Rulings

Rulings 1 to 17 of the parent plan bind. Author, 2026-10-27:

18. **[ELT-No-Grid-Budget]** `elt_max_log2` is dropped. There is no ELT grid
    setting, no `ELT_MAX_LOG2` binding and no ELT cap on `log2`. An ELT sizes
    like any aggregate: the build default, `hints{log2=...}` or
    `build(..., log2=...)`. This supersedes ruling 6 apart from its first
    sentence (larger grids stay possible, requested the ordinary way) and
    ruling 14's assignment of the budget to this step, and with them the
    parent's `[Loss-Table-Large-Grids]` budget text.
19. **[ELT-Shared-Events-Pending]** This plan completes ELT6 for one-row
    events. A shared event parses, builds with theoretical statistics, and
    raises `NotImplementedError` on `update`. The multi-unit grid route waits
    for the eltq windowing decision.
20. **[ELT-Shared-Event-Line]** The refusal is triggered by a *shared
    event*, one event ID with rows for two or more units, not by a second
    unit. Units with disjoint event IDs (the parent's `MultipleCatalogs`
    example, wind and quake) are one-row events, quantized by the same eltq
    call at no extra cost, and they work. Shared events are the problem.
21. **[ELT-Pinned-Grid-Warns]** A pinned `bs` that fails the tail check is
    honored with a warning and a `warning` validation row, as a pinned `bs`
    is elsewhere (`snap_bs_to_reference`). This supersedes the parent's
    "a pinned grid that fails the specified tail contract raises". The tail
    targets and the automatic `bs` choice of `[ELT-Grid]` are provisional:
    the author will try them before they are finalized, so the execution
    log records the measured edges and any retuning.
22. **[ELT-Generic-Handling]** The `elt` clause is another way to state
    frequency and severity: it specifies the Lévy measure
    $\nu=\sum_e\lambda_e\,\mathsf P(X_e\in\cdot)$, whose total mass
    $\Lambda$ is the Poisson rate and whose normalization $\nu/\Lambda$ is
    the severity. Everything after the severity is the generic `Aggregate`.
    No ELT hints or settings: `elt_nodes_per_decade`, `elt_nodes_per_sigma`,
    `elt_z_nodes` and `elt_moment_rtol` are not added, and eltq's defaults
    apply.

## Current behavior, 1.14.0

- `_loss_table._lower_elt` validates every source, projects `no2u` ELT6 to
  ELT4, and then raises `NotImplementedError` for any remaining ELT6
  (`src/aggregate/_loss_table.py`, the `if 'ELT6' in effective:` block).
  Mixing an active ELT6 with ELT3/4 raises `LossTableError` first; that stays.
- Deterministic ELTs lower to `sev_name='dhistogram'` atoms with Poisson
  frequency `exp_en=Λ`; YLTs lower to `dfreq [1] dsev`. Neither imports eltq.
- `_loss_table_meta` carries the audit; `summary_frame`, `stats_frame` and
  `validation_frame` build the three `loss_table_*_df` frames from it, and
  `describe` gives the `info` row, which shows `no secondary uncertainty`.
- eltq is not installed in `.venv`, and nothing in aggregate imports it.
  `pyproject.toml` has no `elt` extra. PyPI holds eltq 1.12.0 only; the
  local `V:/dev/eltq` is 2.14.0, a maturin (Rust) package.
- `Aggregate.update` calls `_bs_window` (`_bucket_window.bs_window`: the
  `moment`, `exact_discrete`, `bounded_small` and `windowed` candidates, then
  `snap_bs_to_reference` as a final query), then `update_work`, whose
  `discretize` calls `_aggregate_compute.discretize_severities`. That kernel
  differences each severity's `sf` on the bucket edges
  `xs - bs/2, xs[-1] + bs/2`, drops mass above the top edge, and normalizes.

## eltq surface used

All four functions exist in the local eltq 2.14.0. **The column orders
differ between functions**; the adapter passes every column by keyword.

| function (since) | used for | columns, in eltq's order | returns |
|---|---|---|---|
| `moments_arrays` (2.11.0) | theoretical moments, one-row events | `rate, max_loss, mean_loss, sd_indep, sd_corr` | `ModelMoments` |
| `multiunit_moments` (2.13.0) | theoretical moments, shared events | `event_id, unit_id, rate, mean_loss, sd_indep, sd_corr, max_loss` | `ModelMoments` |
| `tail_moments_arrays` (2.12.0) | tail check at candidate edges | as `moments_arrays`, plus `edge`, `top_node` | `TailMoments` |
| `quantize_arrays` | the severity on the grid | as `moments_arrays`, plus `bucket_size`, `grid_length` | `raw, secondary, Meta` |

`ModelMoments` gives `total_rate` ($\Lambda$), `raw_moments` ($C_1,C_2,C_3$),
`severity_moments` ($C_r/\Lambda$), `rate_at_zero`, `max_support` ($B$),
row counts by effective law (`atom_rows`, `beta_rows`, `two_point_rows`,
`dropped_rows`), `moment_overflow`, `diagnostics`, and for shared events
`method`, `achieved_error`, `converged` and the event counts. `TailMoments`
gives, per edge $t$ and node $c$, `rate_above` ($R$), `excess_above`
($D_1,D_2,D_3$) and the final-half-bucket atoms (`top_half_rate`,
`top_half_excess`). `quantize_arrays` returns rate-weighted arrays on nodes
$ih$, $i=0,\dots,n-1$; `secondary` is the law with secondary uncertainty,
`raw` the mean-only law (discarded at once), and `Meta` the overflow
(`rate_above_grid`, `mean_above_grid`), `secondary_mass`,
`secondary_mean_error` and `interpolation_error`. The bucket rule is the
nearest node, top edge $(n-1/2)h$ strict, the same rounding as aggregate's
`sev_calc='discrete'`.

Not used: `multiunit_tail_moments` (its union bound is 85 to 240 times its
lower bound on the eltq benchmark, too loose to size a window) and
`multiunit_views` (no window).

Calls use eltq's defaults where the parent's rulings agree with them:
`variance_policy='bernoulli'`, `coupling='burke'`, `method='survival'`,
`overflow='top_bucket'`, `normalize=False`, `mean_is_loss_rate=False`.
Pass them explicitly anyway, so a later change of eltq default cannot
silently change aggregate's model.

## Design

### [ELT-Model] the prepared model

Add a frozen dataclass `EltModel` to `_loss_table.py`, built by
`_lower_elt` when the effective form is ELT6. It holds:

- the positive-rate rows only, as read-only float64 arrays `rate, mean, max,
  sdi, sdc`, with the event identifiers, integer event and unit codes, and
  the unit names. Rows are stably sorted by event code so each event's rows
  are contiguous, as `multiunit_moments` requires; the canonical event rate
  (the first row's, validated as now) is written into every row of the event;
- `shared`: whether any event has two or more rows;
- `moments`: the `ModelMoments` from `moments_arrays` (one-row) or
  `multiunit_moments` (shared, `coupling='burke'`, `rtol=1e-6`);
- `digest`: SHA-256 of the canonical array bytes and identifiers, plus
  `eltq_version` and `eltq_backend` (`eltq.__version__`, `eltq.BACKEND`).

`__repr__` and `__str__` are compact and content-addressed, for example
`EltModel(rows=3, events=3, units=1, sha256=1a2b3c4d)`, because
`Aggregate._spec_hash` serializes the spec with `default=str` and `info`,
errors and repr must stay bounded for a million-row table.

`_lower_elt` then returns `sev_name=<EltModel>`, `freq_name='poisson'`,
`exp_en=Λ` (from `moments.total_rate`, which equals the existing sum over
positive-rate events), and an audit record extended with the model's facts
(below). It raises `LossTableError` if `dropped_rows` is nonzero (the input
was prevalidated, so eltq dropping a row is a bug to surface with examples
from `diagnostics`, never a smaller portfolio), and if `moment_overflow` is
set for any order. A shared model whose `converged` is false builds, and the
validation frame reports it as a `warning`.

Passing the model as `sev_name` follows the `meta` and `copy` precedents,
where `sev_name` is an object, and needs no new constructor argument on
`Aggregate` or `Severity`. `Aggregate(**a.spec)` and `copy` carry the model
by reference (its arrays are read-only), so a rebuilt object neither rereads
files nor recomputes moments. A fresh build from the recipe rereads and
recomputes, as now.

### [ELT-Severity] `SeverityElt`

Add `SeverityElt(Severity)` in `_severity.py` with `sev_kind = 'elt'`, and a
`_classify_sev` branch returning `'elt'` for an `EltModel` (checked by type,
imported lazily as the `Aggregate` and `Portfolio` checks are). It is
registered like the other kinds and dispatched by `Severity.__new__`.

Its contract:

- `_build` sets `fz` to a small private object, `_EltLaw`, whose
  `support()` returns $(0,B)$ with $B$ = `moments.max_support`, and whose
  `cdf`, `sf`, `pdf`, `ppf`, `isf` and `stats` exist and raise
  `NotImplementedError` with a message listing the alternatives: the discretized severity
  after `update` (`sev_density_df`), or `tail_measures`. They must exist:
  the base constructor accesses them as attributes (`_severity.py:2118-2128`),
  and `Underwriter.build` swallows an `AttributeError` from `update`
  (`underwriter.py:2997`, it logs and returns an object never updated), so
  a missing method would fail silently.
- `limit` and `detachment` stay infinite, `attachment` 0, `exp_attachment`
  None, not signed, not reflected, `support_atoms` None (no lattice is
  claimed for a beta mixture). $B$ reaches every consumer through
  `fz.support()`: `tail._severity_bounded` (its finite-support fallback),
  the tail classification and `severity_support`, `_bounded_severity_window`
  (it uses `fz.support()[1]` when `limit` is infinite), `tail_description`
  in `info`, and `output_support_max` when the aggregate is reused as `sev
  agg.NAME`. Setting `limit = B`, as a histogram does, would make
  `detachment` finite and send `moms` and `_compute_attachment_probs` into
  `fz.sf` and `fz.isf`.
- `moms()` is overridden to return `moments.severity_moments`, so
  `Aggregate.__init__` (which calls `sevs[r].moms()`) forms `stats_df` from
  model moments.
- `_compute_attachment_probs` is overridden to set `pattach`,
  `moment_pattach` and `pdetach` to 0 with no call: the base version calls
  `fz.sf(0)` on every non-histogram severity without a layer clause.
- `quantize(xs, bs)` returns the severity pmf on the grid `xs`
  (`[ELT-Quantize]`); for a shared model it raises (`[ELT-Shared-Refusal]`).
- `tail_measures(edge, node)` returns eltq's `TailMoments` for a one-row
  model.

The base `Severity.__init__` stays in charge: it sets about 35 attributes,
calls `rv_continuous.__init__`, runs `_build`, and then the splice, layer,
reflection and validation helpers. Overriding `_build` and the two methods
above is how every existing subclass works (`SeverityCantor`,
`SeverityDHistogram`, `SeverityMeta`, `SeverityCopy` override `_build`
only). The other helpers do nothing for an ELT: splice, reflection, signed
handling and `_validate_moments` run only when their inputs are set, and an
ELT body cannot set them. No temporary law is fabricated (parent,
`[ELT-Severity-Protocol]`).

### [ELT-Severity-Protocol] consumers of the severity

From a read-only audit of `src/aggregate` on 2026-10-27; "default" means it
runs during `build`.

| Phase | Consumer | What it uses | ELT outcome |
|---|---|---|---|
| construction | `Aggregate.__init__`, `_aggregate.py:3182` | `moms()` | model moments |
| construction | base `Severity.__init__` | `fz` method attributes; `fz.sf(0)` in `_compute_attachment_probs` | attributes exist; override sets zeros |
| sizing, default | `bs_window` via `_tail_rows`, `_bounded_severity_window`, `tail._severity_bounded` | `signed`, `support_atoms`, `limit`, `fz.support()` | bounded on $[0,B]$, no atoms |
| sizing, default | `severity_feasibility` via `feasible_bucket` | returns `None` unless `sev_kind == 'scipy'` | no reading for an ELT |
| sizing, rare | `_aggregate.py:7241` far-tail clip, `fz.sf` in a `try` | thick right tail only | never reached: bounded |
| update, default | `discretize_severities`, `_aggregate_compute.py:409` | `sev.sf(adj_xs)` | replaced by the `quantize` branch |
| update | reinsurance, `one_claim`, empirical `stats_df`, `validation_df` | grid and `stats_df` only | unchanged |
| reporting | `info`, `_repr_html_`, `qd`, tail reports | `tail_description`, `fz.support()` | bounded at $B$ |
| explicit only | `severity_error_analysis`, the `sev` property, `oep`, `charts/_emit_approximation.py`, `ft.py` | `sf`, `pdf`, `isf` | `NotImplementedError` with guidance |
| portfolio | `Portfolio.update`: `best_window` runs each unit's `_bs_window`, then `update_work` on the portfolio grid; the signed or windowed route calls `agg.update` with `bs` pinned | as sizing and update | see `[ELT-Grid]` |

Because `feasible_bucket` handles only scipy severities, a coarse bucket chosen
for reach is not flagged by the feasibility report. For an ELT the
validation rows comparing the discretized moments with the model's carry
that signal instead (`[ELT-Reports]`).

### [ELT-Grid] sizing with a tail check

The grid is chosen by the existing `_bs_window` from the model's moments and
support. Two things are added: a **tail check**, a function of the model and
a grid, and one **choice**, made by the sizer only when `bs` is automatic.

Let the severity grid have $n$ nodes at spacing $h$, top node $c=(n-1)h$ and
top edge $t=(n-1/2)h$. The severity grid always starts at zero for an
unsigned aggregate (`xs_sev = arange(n) * bs`, `_aggregate.py:4631`), even
when the output window has a positive origin. The check passes when

$$
R(t)\le 10^{-w},\qquad D_r(t,c)\le\frac{\varepsilon}{10}\,C_r,\quad r=1,2,3,
$$

with $w$ = `discretization.window_nines` (12) and $\varepsilon$ =
`validation.eps` (`1e-4`), the parent's proposal, provisional under
ruling 21. $R$ bounds the
probability of an overflowing occurrence in a year by $1-e^{-R}\le R$, the
same tail allowance as the annual window; $D_r$ is what `top_bucket` takes
from the $r$-th annual cumulant. A zero $C_r$ (an all-zero table) passes
trivially. The atoms in the final half bucket, $(c,t]$, stay on the grid at
$c$; they are reported, not tested, as ordinary rounding.

**The choice, automatic `bs` only.** `_bucket_window.elt_tail_check(agg,
bs, log2, pinned)` runs at the end of `bs_window`, after
`snap_bs_to_reference` and in its style (a final query on the winning
candidate), when the aggregate has a one-row `SeverityElt` and `bs` was not
pinned. If the winning grid fails the check, query one ladder of edges in a
single vectorized call, $t_k=t\cdot2^{k/4}$ from the current edge up to $B$
with $B$ itself last, each with node $c_k=t_k(n-1)/(n-1/2)$. $R$ and $D_r$
are nonincreasing along the ladder and zero at $B$, so a passing edge
exists. Take the smallest, set $h^\ast$ = `round_bucket`$(t_k/(n-1/2))$, and
rerun the sizer with $h^\ast$ (as `snap_bs_to_reference` does); confirm at
the final grid, stepping to the next ladder edge if rounding moved the node
so the check fails. This buys reach with `bs`, as the sizer already does for
any severity (author ruling 2026-08-21, recorded at `severity_feasibility`):
the bucket only grows, so the annual window only widens. An explicit `log2`
is the bucket count the sizer fills; the choice then gives the smallest
`bs` whose edge passes at that count. The outcome (edge, measures, ladder
edge, reason) is recorded in `_bs_window_df`'s note on the winning row.
A `Portfolio` runs each unit's `_bs_window` in `best_window`, so the choice
informs the unit's candidate; the portfolio's own selection among units is
unchanged.

**The measurement, every quantization.** Whoever chose the grid (the sizer,
a pin from the caller or `hints{bs=...}`, a `Portfolio`, a `sharpen` probe),
`SeverityElt.quantize` evaluates the check at the grid it is given, with one
`tail_moments_arrays` call at one edge, and stores the result with the
quantization metadata. If it fails, it warns once per aggregate
(`warn_once`, so `sharpen`'s probes do not repeat it), with a message giving the measured
$R$ and $D_r/C_r$, the edge that would pass, and the `bs` at the current
`log2` that reaches it. A pinned grid is honored, as a pinned `bs` is
elsewhere. The report rows are computed from this measurement, so they describe the grid
actually used.

### [ELT-Quantize] the severity on the grid

`discretize_severities` gains one branch ahead of the existing ones:

```python
if sev.sev_kind == 'elt':
    appx = sev.quantize(xs, bs)
```

`SeverityElt.quantize` checks `xs[0] == 0` and spacing `bs` (an internal
error otherwise; see `[ELT-Grid]` for why it always holds), calls
`quantize_arrays` with `bucket_size=bs, grid_length=len(xs)` and the options
above, drops `raw`, checks that the returned rate mass equals $\Lambda$
within `MASS_TOLERANCE` times $\Lambda$ (under `top_bucket` the rate is
exact), runs the tail measurement of `[ELT-Grid]`, and returns
`secondary / Λ`. The kernel's `normalize` step then divides by a sum of one
up to rounding. eltq's `Meta` and the tail measurement are kept on the
severity as `last_quantization` (scalars only, no arrays) for the reports.
Every new grid quantizes again from the model, so a later wider or finer
update recovers the tail, and no array is cached across updates.

Everything downstream is unchanged: occurrence reinsurance acts on the
discretized severity, then the Poisson transform, then aggregate reinsurance
and validation. The severity and annual grids share `bs`, so the parent's
padded shorter severity window is not needed; `len(xs)` is the grid.

### [ELT-Shared-Refusal] shared events build and then stop

A shared model builds: `stats_df`, `info`, the three report frames, `spec`,
copying and `update=False` all work. `Aggregate.update` checks
`SeverityElt.shared` before sizing and raises `NotImplementedError`, with a
message giving an example event and its units, the label
`[ELT-Shared-Event-Uncertainty]`, `build(..., update=False)` for the
theoretical statistics, and `no2u` on every clause for the mean-only model.
`SeverityElt.quantize` raises the same error, which covers any other path
into `discretize` (a `Portfolio` unit, `sharpen`, `severity_error_analysis`).
The check runs before `_bs_window`, so no tail query or allocation precedes
it. `build(...)` updates by default, so a plain build of a shared ELT6
raises; that is the intended state until the multi-unit route lands.

Between the two bumps of this plan a one-row model also raises at `update`,
with the label `[ELT-Single-Event-Uncertainty]`; the second bump removes
that refusal.

### [ELT-Reports] audit, stats and validation

`describe` (the `info` row) shows `secondary uncertainty` for an ELT6, and
`secondary uncertainty, shared events (model moments only)` for a shared
one. The frames gain rows; no new frame or suffix:

- `loss_table_summary_df`, part `table`: `secondary uncertainty` (beta,
  two-point and atom row counts), `largest event support` ($B$),
  `model moments` (`closed form`, or `factor integration` with its achieved
  error), `eltq` (version and backend). Part `grid`, after `update`:
  `severity edge`, `tail rate above edge`.
- `loss_table_stats_df`: for an ELT6, column `table` holds eltq's model
  moments (severity from `severity_moments`, annual from `raw_moments`),
  `model` the aggregate's `stats_df` value computed through `SeverityElt.moms`,
  and `estimate` the discretized density's, as for the deterministic forms.
  `table` against `model` then checks the plumbing; `estimate` against
  `table` measures quantization, overflow and FFT error together.
- `loss_table_validation_df`: `row classification` (two-point replacements
  are a `pass` with their count, since they are the ruled policy),
  `moment integration` (shared only: `pass` when converged, else
  `warning`), and after `update` `tail rate above edge`, `tail mean share`,
  `tail second moment share`, `tail third moment share` against the targets
  of `[ELT-Grid]`, `quantized mass against event rate`, and eltq's
  `secondary mean error` against $\varepsilon$, all read from
  `last_quantization`, so they describe the grid actually used. Before
  `update` these are `not computed`, never `pass`. The existing estimate
  rows (discretized severity and annual mean, CV and skewness against the
  model) are the body-resolution signal for an ELT, which the feasibility
  reading does not cover (`[ELT-Severity-Protocol]`).

### [ELT-Dependency-Local] building against the local eltq

eltq is imported lazily, by one helper in `_loss_table.py`, at the first
ELT6 lowering. A missing eltq, or one older than 2.13.0 (the first with
`multiunit_moments`), raises `ImportError` with a message giving the minimum version, the
install command, and `no2u` as the mean-only alternative. Parsing, ELT3/4,
all-`no2u` ELT6 and YLTs never import it.

Until eltq 2.13 or later is on PyPI, aggregate does not declare it:
declaring a version PyPI cannot supply would break `uv lock`. It is
installed by hand into the development environment, like the optional
compiled engine:

```powershell
$env:UV_PROJECT_ENVIRONMENT = '.venv'
uv pip install V:/dev/eltq
```

This builds the wheel with maturin and the Rust toolchain. `uv sync
--all-extras` removes an undeclared package, so sync with `--inexact`, as
`CLAUDE.md` already records for `aggregate-ntt`. The ELT6 tests skip when eltq
is absent (`pytest.importorskip('eltq', minversion='2.13')`), so CI and a
clean checkout stay green. When eltq is released, a follow-up adds the
optional `elt` extra (`eltq>=` the released version) to `pyproject.toml`
and to the `all` extra, and drops the skip; that is a TODO entry, not part
of this plan.

## Steps

Two minor bumps, one commit each, under the repo's release rules: version in
`src/aggregate/__init__.py`, one-paragraph CHANGELOG entry, `dev/TODO.md`,
`dev/FEATURES.csv` via `dev/regen_features.py` where a row changes. This
plan moves to `dev/done/` in the commit that lands its first step. Versions
are assigned at execution.

### [ELT-Uncertainty-Moments]

`EltModel`, the lazy eltq import, ELT6 lowering through `moments_arrays` or
`multiunit_moments`, `SeverityElt` with its model moments and refusals,
`describe`, the summary and validation rows that need no grid, and the
refusal at `update` for every active ELT6 (`[ELT-Shared-Refusal]`). After
this bump an ELT6 builds with `update=False` and reports its theoretical
statistics; `build` with its default update raises.

Acceptance (`tests/test_elt_uncertainty.py`, skipped without eltq):

- One row, rate 0.1, mean 50, max 100, SDs 10 and 10: beta $(2.625,2.625)$;
  annual mean 5, variance 290, $C_3=18{,}500$, in `stats_df` and in the
  stats frame.
- The same row with SDs summing to 60: two-point on $\{0,100\}$, variance
  500, `rate_at_zero` 0.05, counted under `row classification`.
- The same row with SDs 0 and 0 in an ELT6: an atom, variance 250, equal to
  the `no2u` build's moments.
- A positive-rate zero-mean event keeps its rate in $\Lambda$; an all-zero
  ELT6 builds with zero moments and a truthful audit.
- Two units with disjoint event IDs have the moments of the same rows in
  one unit, and `shared` is false.
- Shared, two units at rate 0.1, both two-point with $M=100$: means 50 and
  20 with `sdi=0, sdc=60` (Burke loading 1, comonotone) give event moments
  $\mathsf PS=70$, $\mathsf PS^2=11{,}000$, $\mathsf PS^3=1{,}900{,}000$;
  with `sdi=60, sdc=0` (loading 0, independent) $\mathsf PS^2=9{,}000$ and
  $\mathsf PS^3=1{,}300{,}000$. Both are exact closed forms, so the
  coupling's direction is tested.
- `update=False` builds; `update` raises `NotImplementedError` with the
  labels above; `Aggregate(**a.spec)` keeps the model without reading a file
  (the source file deleted after the first build); the spec hash is stable
  across two builds of the same table and changes when a value changes.
- Without eltq (`sys.modules['eltq'] = None` in a monkeypatched test, so it
  runs everywhere): ELT6 raises `ImportError` with the install guidance;
  ELT3/4, all-`no2u` ELT6 and YLT still build.
- Existing `tests/test_elt.py` cases that expected the construction-time
  `NotImplementedError` are updated to the new behavior.

Every moment above, and the edges and tail measures quoted in the next
step's acceptance, were reproduced on 2026-10-27 with the local eltq 2.14.0
(Rust backend) in eltq's own environment: `moments_arrays`,
`multiunit_moments` (converged; `factor_integration` for the comonotone
pair, `closed_form` for the independent one) and `tail_moments_arrays`.

### [ELT-Single-Event-Uncertainty]

`elt_tail_check` in `_bucket_window`, the `quantize` branch in
`discretize_severities`, `SeverityElt.quantize` and `tail_measures`,
`last_quantization`, the grid rows of the frames, and removal of the one-row
refusal. Documentation: the "Event and Year Loss Tables" section of
`docs/4_dec_Language_Reference.rst` gains secondary uncertainty, the shared
refusal and the eltq install note (docs pending a rebuild by the author).

Acceptance, added to the same test file:

- **Oracles on a fine grid.** The beta, two-point and atom rows above, at a
  pinned grid fine enough to separate quantization from FFT error: the
  discretized severity and annual mean, CV and skewness within
  $\varepsilon$ of the model; the quantized mass equal to $\Lambda$ before
  division.
- **`no2u` parity.** An ELT6 of zero-SD rows with integer means, at `bs=1`,
  has the same severity and annual densities as its `no2u` build.
- **Moments fixed across grids.** Two updates at different `bs` leave
  `stats_df`'s theoretical columns unchanged and move only the estimates.
- **No quantization before `update`.** A spy on `quantize_arrays` records no
  call under `update=False` and exactly one per update without a ladder
  retry.
- **Automatic sizing, no hints.** The parent's two large-exposure rows
  (rate 0.1; $\mu=20{,}000$, $\sigma=60{,}000$, $M=10^9$; and $\mu=100$,
  $\sigma=10{,}000$, $M=10^9$) build at the default `log2` with a grid top
  far below $M$ (under $M/10$), and pass the tail check. On the $2^{16}$
  ladder of `[ELT-Grid]` the passing edges are about 4.31 million and 15.6
  million, 0.43% and 1.6% of $M$ (eltq 2.14.0, 2026-10-27); each test
  asserts its edge within one ladder rung of these.
- **The 40-SD counterexample.** An edge near $\mu+40\sigma$ fails both
  rows: the first on $R$ ($6.9\times10^{-10}$) and the third moment share
  ($1.7\times10^{-5}$), the second on every moment, 38% of $C_1$ in excess
  of the node. The automatic sizer chooses a larger edge; a pinned `bs`
  placing the edge there warns and reports `warning` rows.
- **A remote two-point row.** Rate $5\times10^{-7}$, mean 10, $M=10^7$,
  `sdi=0, sdc=20000` (two-point, atom at $M$ with rate $5\times10^{-13}$)
  beside the beta row: at any edge below $M$, $R$ passes its $10^{-12}$
  target and $D_1$ is a $10^{-6}$ share, yet the row carries 15% of $C_2$
  and almost all of $C_3$. The automatic edge reaches $M$; an edge exactly
  at $M$ keeps the atom on the grid (strict inequality); a pinned edge below
  $M$ warns on the second and third moment shares.
- **Reinsurance.** `occurrence net of 30 xs 20` and `aggregate net of 50 xs
  10` on the one-row beta ELT equal the same treaties on a `dsev` built from
  its quantized severity on the same grid.
- **Composition.** A `Portfolio` with an ELT6 unit, a `pnl` with an ELT6
  engine, and `agg.NAME` reuse all update; `sharpen()` runs and leaves the
  theoretical moments unchanged.
- **Shared refusal unchanged.** The shared fixtures of the first bump still
  raise at `update`, including as a `Portfolio` unit.
- **Disjoint units.** Wind and quake ELT6 sources with distinct IDs update
  and match the single-unit build of the same rows.

Measure once and record in the execution log, not as a test: a synthetic
250,000-row one-row ELT6 at `log2` 16, 20 and 22, wall time and peak memory
of `build`, against a direct `quantize_arrays` call at the same grid. The
budget is set from that baseline.

### [ELT-Shared-Event-Uncertainty], pending

Not planned here. It waits on the eltq decision recorded at the end of the
eltq plan's execution log (`[Eltq-Windowed-Shared-Events]`, the
measurement), and then needs: a windowed `multiunit_views` with overflow
diagnostics, a shared-event tail check (the union bound is too loose; the
lattice bracket or the windowed grid's own overflow), the dispatch in
`SeverityElt.quantize`, and the parent's shared-event acceptance cases. The
refusal of `[ELT-Shared-Refusal]` is the seam it replaces.

## Files

| Area | Files |
|---|---|
| Model and lowering | `src/aggregate/_loss_table.py` (`EltModel`, eltq import helper, `_lower_elt`, `describe`, the three frame builders) |
| Severity | `src/aggregate/_severity.py` (`SeverityElt`, `_EltLaw`, `_classify_sev`) |
| Grid and update | `src/aggregate/_bucket_window.py` (`elt_tail_check`, its call in `bs_window`), `src/aggregate/_aggregate_compute.py` (the `discretize_severities` branch), `src/aggregate/_aggregate.py` (the shared refusal in `update`) |
| Tests | new `tests/test_elt_uncertainty.py`; `tests/test_elt.py` and `tests/test_loss_table_reports.py` where they assert the old refusal |
| Documentation | `docs/4_dec_Language_Reference.rst`, the `info` catalogue in `dev/info-strings.rst` |
| Release | `src/aggregate/__init__.py`, `CHANGELOG.md`, `dev/TODO.md`, `dev/FEATURES.csv` via `dev/regen_features.py` |

`dev/TODO.md`: rewrite the `[DecL-Loss-Tables]` entry to drop
`elt_max_log2`, point at this plan, and add `[ELT-Eltq-Extra]` (declare the
`elt` extra when eltq is released).

`CLAUDE.md` (the project file at the repository root, line 17, the
paragraph headed "`uv sync` removes the optional compiled engine") gains a
sentence at the end of that paragraph. The file is the author's to edit, so
the executor hands this text over rather than writing it:

> eltq is discovered the same way until it is on PyPI: the secondary
> uncertainty path of `agg ... elt ...` imports it, it is installed by hand
> with `uv pip install V:/dev/eltq` (a maturin build), and the same
> `--inexact` sync keeps it.

No grammar change, so no spec snapshot or grammar reference regeneration.

## Names checked

On 2026-10-27, `rg` over `src/aggregate` found no definition or use of
`EltModel`, `_EltLaw`, `SeverityElt`, `elt_tail_check`,
`quantize`, `tail_measures`, `last_quantization` or `sev_kind = 'elt'`.
`Severity` has no `quantize` or `shared` attribute to shadow. Recheck at
execution. `elt_max_log2` and `ELT_MAX_LOG2` are not introduced (ruling 18).

## Release gates

Edit loop: `tests/test_elt_uncertainty.py`, `tests/test_elt.py`,
`tests/test_loss_table_reports.py`. Each bump: the full fast suite, then
`uv run pytest -m 'slow or not slow' -W error::RuntimeWarning` (a numerics
change), with `UV_PROJECT_ENVIRONMENT=.venv`, restoring any prior value.
Docs are not built.

## Deferred

- **[ELT-Shared-Event-Uncertainty]**, above.
- **[ELT-Eltq-Extra]**: the `elt` extra once eltq is on PyPI.
- The parent's remaining deferred extensions (`[ELT-Coupling-Selection]`,
  `[ELT-Joint-Units]`, `[ELT-Unit-Allocation]`, `[ELT-Unit-Terms]`,
  `[ELT-Frozen-Inputs]`, `[ELT-General-Continuous-Queries]`) are unchanged.

## Execution log

Record actual versions, measured tolerances, the performance baseline and
divergences here as the plan executes.

### Review, 2026-10-27

Every cited path, symbol and line was checked against 1.14.0 and holds, with
three exceptions recorded below as divergences: the `info` catalogue lives at
`docs/2_aggregate_overview/info-strings.rst`, not `dev/info-strings.rst`; the
control list `V:/worktrees/dev-files.md` no longer exists, so there is no row
to update; and the `[ELT-Grid]` choice cannot rerun `_size` with $h^\ast$ as
the snap does (divergence 6). The names of "Names checked" were rechecked with
`rg` over `src/aggregate` and none collides. eltq 2.14.0 was installed into
`.venv` with `uv pip install V:/dev/eltq` (a maturin build, 7 s). The
`CLAUDE.md` sentence on eltq was already in the author's working tree, so no
hand-over was needed.

### [ELT-Uncertainty-Moments], 1.15.0

Landed as planned: `EltModel`, `import_eltq`, the ELT6 lowering through
`moments_arrays` or `multiunit_moments`, `SeverityElt` and `_EltLaw`, the
`describe`, summary and validation rows that need no grid, and the refusal at
`update`. Every moment of the acceptance list reproduced exactly (the
comonotone pair by factor integration to $10^{-10}$, the rest closed form).

Divergences:

1. **Attachment probabilities.** The plan sets `pattach`, `moment_pattach` and
   `pdetach` to 0. The override sets `pattach = moment_pattach = 1` and
   `pdetach = 0`, the values the base gives any severity with no layer clause
   (an `elt` body admits none), so a reader dividing by `pattach` never sees a
   zero. Nothing outside `_severity.py` reads `pattach`.
2. **The quantize seam lands in this bump.** The `discretize_severities`
   branch and a `SeverityElt.quantize` that raises the refusal are added now,
   not in `[ELT-Single-Event-Uncertainty]`, so a `Portfolio` unit (whose
   update goes through `update_work`, not `Aggregate.update`) refuses with the
   plan's message rather than with `_EltLaw`'s pointwise refusal.
3. **One new public name.** `SeverityElt.check_grid_supported(owner)` raises
   the refusal; `Aggregate.update` and `quantize` both call it. Also public:
   the `model` and `shared` properties the plan implies.
4. **Validation rows.** For an ELT6 the `severity mass` row (lowered weight
   against the event rate) is replaced by `row classification`, since nothing
   is lowered to weights before a grid; the next bump adds `quantized mass
   against event rate`. The `table against model` rows read `eltq model
   moments against stats_df`. The summary keeps `largest event exposure` (the
   sum of unit maximums) beside the new `largest event support` ($B$).
5. **A test the plan did not list.** `tests/test_loss_table_files.py`
   `test_no2u_file` also asserted the construction refusal; that half became
   `test_elt6_file_with_secondary_uncertainty`.

## References

::: {#refs}
:::
