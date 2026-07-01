# TODO

> The v1.0 backlog. **Active sequence first** (the ordered P&L → reporting path),
> then the remaining backlog grouped by theme, then post-v1.0 ideas. Snappy
> entries only — details live in the plan files (`dev/`, `dev/done/`) and the git
> log; **what's landed is in `CHANGELOG.md`** and removed from here.
>
> **Labels, not codes.** Every item has a descriptive `[Bracket-Label]`; the old
> single-letter track codes (`N6`, `H4`, `T2`, …) are retired (CLAUDE.md, Naming
> conventions — 2026-06-30). Phase: `alpha` = must finish before cutting
> `1.0.0b1`; `beta` = fine just after the alpha→beta cut. GitHub issue numbers
> kept in parentheses as the stable external cross-reference.
>
> **Last updated: 2026-06-30** — Retired the cryptic track-code system for
> descriptive labels and added the active P&L / reporting sequence on top. Pruned
> the shipped god-module, P&L-expenses, variable-rating, reinstatement, and
> bivariate-leg-kernel work (now in `CHANGELOG.md`, through `1.0.0a121`).

---

## Active sequence — making P&L a first-class citizen (in order)

> The current focus, in dependency order. Item 1 *defines* the target; items 2–4
> build to it. The three "sequenced around" items are slotted by the author:
> `[Display-Mode]` waits on item 1, `[Signed-Bounded-Window]` is anytime,
> `[Doc-Fix]` is last.

1. **[PnL-API]** ✅ **DONE (`1.0.0a122`, `dev/done/plan-pnl-api.md`).**
   `create_pnl(source, *, consideration, obligation, role, …)` + `create_pnl_tower`
   / `PnLTower`: the **domain-agnostic** P&L constructor over the pushforward
   engine (labels-as-data; consideration/obligation/result → exact GDs + the four
   FCC reports). `PnL` reshaped into an engine-free value object.
   The a121 leg wrappers (`legs.py` / `_insurance_view.py`) and
   `gcn_assemble_column` / `_gcn_*` are removed; the pushforward primitives stay.
   `gcn_df` adopted the new-canonical stats×waterfall schema (byte-identical bar
   relaxed by the author). DecL unchanged.
1b. **[PnL-Exhibits]** ✅ **DONE (`1.0.0a123`, `dev/done/plan-pnl-exhibits.md`).**
   Makes the `PnL` value object **generic and self-describing** and advances
   `[Reporting-Guidelines]` for the P&L surface: `build('pnl …')` **always
   returns a `PnL`** (the domain-specific `PnLTower` / `ReinstatementAnalysis` /
   `VariableRatingAnalysis` return types are gone — attached as `pnl.tower` /
   `pnl.analysis`). Fixed-shape `summary_df` (`% Consid`→`Scaled`, `P01`→`P1`,
   constant legs exact via prob-renormalization + dust-snap); `stats_df` a
   **property** with a construction-committed scale; new `margin_df` (the cession
   waterfall), `stochastic_engine`, `scale`, `tower`, `analysis`; `loss`-basis
   expense stochastic in the plain path. DecL unchanged.
2. **[Reporting-Guidelines]** `alpha` — *define what "first-class citizen" means*
   for a reporting object, against the `[PnL-API]` shapes: a report's **rows are
   fixed** (it does not morph as the object gains properties), columns are
   **pure** (one unit per column — currency and ratios never mixed), headings are
   **presentation-ready**, and the `summary` (what *is* this?) vs `validation`
   (= the old `summary`: is it calculating right?) vs `reins_<flavor>` (only when
   reinsurance present) split is settled. Plan: `dev/reporting-guidelines.md`.
   (The first fix it named, the `summary_df`→`gcn_df` morph, landed in
   `1.0.0a121`.) Depends on `[PnL-API]`.
3. **[Portfolio-of-PnL]** `alpha` — `PortPnL`: portfolios of P&L positions.
   Constant-consideration v1 (`make_pnl(port_total, Σ Cᵢ)` + a stacked per-unit
   signed summary; obligation combine is the existing Portfolio FFT); loss-
   sensitive net-then-combine deferred. A book is a `create_pnl_tower` (or a
   `create_pnl` over a portfolio total), so it is **nearly free once `[PnL-API]`
   lands**. Plan: `dev/plan-pnl-portfolio.md`. Depends on `[PnL-API]`.
4. **[Plotting-Punchups]** `alpha` — plotting polish; lead item: a `pnl` aggregate
   plots the aggregate **only** (no loss-convention severity overlaid on a payoff
   — wrong-sign distraction for the UW/finance audience). Plan:
   `dev/plan-plotting-punchups.md`. Pairs with `[PnL-API]` plotting; do
   alongside / after. (Related, separate: `[Plot-Severity-Outside-Window]` below.)

**Sequenced around the above:**

- **[Display-Mode]** `beta` — a presentation-only `user` / `dev` repr toggle (a
  `ReprMixin`) that chooses *which* view an object renders without changing any
  computed value. **Deferred until after `[Reporting-Guidelines]`** — what it
  toggles between depends on the report shapes decided there. Plan:
  `dev/plan-display-mode.md`.
- **[Signed-Bounded-Window]** `alpha` — robustness fix: kill the `int(inf)`
  overflow in the bucket/window sizer and guard the silently-ignored layer on a
  signed severity. Independent — **do whenever** (regression bar: ordinary
  aggregates byte-for-byte unchanged). Plan:
  `dev/plan-signed-bounded-window-overflow.md`.
- **[Doc-Fix]** `beta` — clear the executed-cell `*Error`s in the docs build.
  **Last — after the code has settled** (it churns with every API change). Plan:
  `dev/doc-fix.md`.

---

## Backlog — numerics & pricing core

- **[Validation-Calc-Review]** `alpha` (#49) — audit the validation algorithm vs
  the published *Aggregate* paper and make the docs match the actual algo. The
  "all switches → config" sub-goal is done (`eps`/`noise`, `aliasing_ratio`,
  `exeqa_noise_floor`, `deficit_materiality`); remaining: fix the false-positive
  *agg-mean-error ≫ sev-error / aliasing* failure (try larger `bs`; revisit the
  too-tight tolerance, now an `aliasing_ratio` config edit).
- **[Input-Guards]** `alpha` (ported from README) — three correctness/guard items:
  zero `lb` not consistent with attachment equals zero; flag **fixed** frequency
  with a non-integer expected value; flag **mixing** with an inconsistent
  frequency distribution.

## Backlog — bugs & investigations

- **[ZT-ZM-Frequency-Fix]** `alpha` — zero-truncated / zero-modified frequency is
  broken (`poisson zt` → NaN solver for every parameterization; `zm` builds but
  the semantics are wrong — it inverts a post-modification mean). Redesign: the
  user inputs the **un-truncated/un-modified base mean** and we apply the shift
  **forward** (no solver); ship documented shift helpers (both directions). Two
  examples are commented out in `examples.agg` until then. Pairs with `[Doc-Gaps]`.
- **[bs_describe-Wart]** `beta` — investigate the `bs_describe` / `bs_explain`
  module workers (the `color=` workers behind `bs_description` /
  `bs_explanation`): purpose, the local-`line` accumulator, and whether they earn
  their place / want reshaping. **Reconcile with `dev/done/plan-consistent-naming.md`
  §3** (the deferred rename of the verb workers that shadow the noun properties by
  one letter, e.g. → `_format_bs_grid`). Do the two together. *Standing reminder —
  surface periodically until scoped.*

## Backlog — hygiene (module organization)

- **[Docstring-Sweep-NumPy]** `alpha` (#18) — Sphinx `:param:` → NumPy style in
  `iman_conover.py` / `moments.py` (and pockets elsewhere); public surface first.
  Feeds `[API-Docstring-Coverage]`.
- **[Pedagogy-Migrations]** `beta` (#19) — move figure generators out of `ft.py` /
  `tweedie.py` into `pedagogy.py` so those stay API-focused. Feeds
  `[Pedagogy-Docs-Punchup]`.

> The god-module refactor (the `GridDistribution` value type, the `plots/`
> subsystem + matplotlib defer, the `distributions.py` / `portfolio.py` kind/
> subsystem splits, and the shared-concern modules `_validation` /
> `_bucket_window` / `_pricing` / `_reinsurance`) is **complete** — shipped
> `1.0.0a90`–`a95`, see `CHANGELOG.md`. Post-beta / conditional leftovers: the
> plots visual refresh, `ReinsuranceProgram` composition, the sample-subsystem
> (correlation / switcheroo) review.

## Backlog — tests

- **[Rationalize-Tests]** `alpha` (#51) — needed vs no-longer-needed; untangle and
  re-wire how the suite *consumes* the single test library (the `conftest`
  parametrization of every `test_suite.agg` line, the SLY snapshot regression)
  without losing coverage.
- **[Switcheroo-Sample-Regression]** `beta` (#12) — a `Port.Sample` regression
  case guarding the kappa-replacement path.

## Backlog — docs & packaging

> Most of these have **no code dependency** — ready whenever the docs cycle opens.

- **[README-Stable-Body]** `alpha` (#39, #13, #14) — rewrite the README body for
  the stable-v1.0 audience (what / who / install / one-liner DecL); the
  `README.md` + `CHANGELOG.md` split is done, only the body remains.
- **[v1-Journey-Philosophy]** `alpha` (#15) — the v1.0 intro / "Journey" page +
  statements of philosophy (user manages logging / warnings / matplotlib; the
  distribution **is** `pᵢ` at `xᵢ`, no jump detection; `qd` is the doc-only
  fixed-font exception); cover the v1.0 shifts (linear allocation default, bounded
  detection, forwards-`S`, pentagon columns, `DefectiveDistributionWarning`).
- **[Grammar-Reference-From-Lark]** `alpha` (#17) — regenerate
  `docs/4_agg_language_reference/` from `decl.lark` / `grammar(add_to_doc=True)`
  (it still describes the SLY-era grammar). No code dependency.
- **[Tail-Descriptor-Docs-Tests]** `alpha` (#41, #42) — bounded / log-concave /
  super-exp / exp / sub-exp descriptors for freq **and** sev, plus the
  bounded/unbounded indicator; with tests.
- **[Doc-Gaps]** `alpha` (#58–#62) — custom errors; syntax checker / better error
  reporting; stale "site" database refs; ZT/ZM zero-truncation/modification;
  splice examples. No code dependency.
- **[API-Docstring-Coverage]** `alpha` — every public function/class carries a
  NumPy-style docstring that renders in the API reference. The doc side of
  `[Docstring-Sweep-NumPy]`.
- **[Reinsurance-Case-Study-Docs]** `beta` (#16) — rebuild `bahnemann` /
  `enterprise risk` / `other_misc` per-layer exhibits from `reins_stats_df`,
  verify vs published (numerics now stable).
- **[Pedagogy-Docs-Punchup]** `beta` (#40) — punch up `pedagogy` and integrate
  with docs; possible minor renames. Needs `[Pedagogy-Migrations]`.
- **[Reinsurance-Structure-Diagrams]** `beta` (#45) — under-specified; confirm
  source / scope (PMIR code?).
- **[Cheat-Sheet-Tweaks]** `beta` — at the alpha→beta cut, re-run `introspect` per
  class, reconcile any renames/removals, and apply pending wording/layout tweaks
  (incl. whether to densify DecL pages 2–3). Held until first beta.
- **[Plot-Severity-Outside-Window]** `beta` (#8) — plot severity when its grid
  doesn't overlap the aggregate window (inset, broken axis, or separate figure;
  `info` already warns). Approach undecided.

## Backlog — pre-beta scaffold retirement (do at the `1.0.0b1` cut)

> The `.agg` libraries were split (a71) into a **shipped** set (`examples`,
> `actuarial-severity-curves`, `decl-testers`, `cookbook`) and a temporary
> SLY-parity scaffold (`_test_suite.agg`, `_test_suite2.agg`). The scaffold has
> done its job (proving the Lark parser matches the retired SLY parser); the
> surviving net is `tests/test_agg_libraries.py`.

- **[Scaffold-Retirement]** `alpha` (at b1) — delete `_test_suite.agg` /
  `_test_suite2.agg` and retire their dependents: the SLY snapshot
  (`tests/data/expected_specs.json` + `capture_sly_snapshot.py`),
  `test_decl_parser.py`, `test_splice_suite.py`, the `conftest`
  `test_suite_lines` / `underwriter` fixtures, `config.py` `TEST_SUITE_FILENAME` +
  `Underwriter.test_suite_file` + `interpret_file`'s default, the
  `freeze_knowledge.py` / `bucket_baseline.py` `DEFAULT_DATABASES`, the default
  toml `_test_suite` line, and the docs "Test Suite Programs" `literalinclude`.
  Confirm `test_agg_libraries.py` covers the shipped libraries (optionally extend
  to a build smoke test); decide whether `decl-testers.agg` needs its own
  permanent parse harness.

---

## Related plans (shipped → context)

- **Numerics program** (`dev/done/plan-numerics-0-meta.md` + `-1`…`-4`) —
  complete; the apply-distortion calcs shipped a55–a57, windowed combine via
  `dev/done/plan-mv.md` (a72/a76).
- **Bivariate firm-up** (`dev/done/plan-mv.md`) — shipped a70–a80.
- **`prob_loss_assets` / `pla`** (`dev/done/plan-pla.md`) — free capital anchor
  over `{p, L, a}` + `price_pentagon_ex`, shipped a97.
- **First-class P&L (Stages A–E)** (`dev/done/plan-pnl.md`) — the `PnL` veneer the
  `[PnL-First-Class]` work builds on.
- **P&L expenses / variable rating / reinstatements / bivariate leg kernel** —
  shipped a114–a121 (`dev/done/plan-pnl-expenses-ceded-premium.md`,
  `plan-reinstatements.md`, `plan-variable-rating.md`, `plan-bivariate-legs.md`).
  The leg kernel (`aggregate.legs`) + insurance View (`aggregate._insurance_view`)
  are the foundation `[Portfolio-of-PnL]` composes over.
- `dev/done/` — the remaining shipped plans (config, pentagon, database loading,
  bucket-window, allocation/pricing bounds, decl-unparser, bibliography, …).

## Post-v1.0 ideas

- **[Multi-Resolution-Portfolio-Combine]** (#20) — compute each unit on its own
  `bs`, decimate onto the shared grid before the Fourier product (the real fix for
  the coarse shared-`bs` deficit). Deficit accepted / surfaced for now.
- **[Premium-Loss-Algebra-DecL]** (#21, v2.0) — constant aggregates and full
  aggregate arithmetic (`agg.A - agg.B`, `agg.A + c`); `pnl` covers the common
  case for v1.0.
- **[Named-Cherny-Madan-Families]** (#23) — add the MINMAXVAR / MAXVAR / MAXMINVAR
  distortion kinds so `PnL.evaluate` can surface the *named* indices (`dual`
  already *is* MINVAR). `@Cherny2009a`.
- **[Rate-Based-Reins-Clauses]** — extend reinsurance clauses to accept e.g.
  `net of 50% of 500 xs 500 at .3 rol or 3000 ceded or .25 ros` (rate on subject =
  quota share).
- **[Reinstatement-Event-Date-Terms]** — reinstatement terms depending on event
  date (pro-rata as to time); from `dev/done/pre-plan-reinstatements.md`.
- *(rejected: DecL colorization — aesthetic-only and structurally weak; see
  `dev/done/plans-considered-and-rejected.md`.)*
