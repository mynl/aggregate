# Aggregate Refactor Plan

Standing reference for the multi-stage rewrite of the heavy compute modules. Each stage is picked up in its own session; this file is the durable source of truth across sessions. Update it as decisions land.

## Status

- **Done:** parser (SLY → Lark), spectral
- **In flight (separate branch):** underwriter
- **Pending, in order:** distributions → portfolio → utilities → bounds
- **Side task (any time):** migrate `frequency_examples` to a Quarto page, add zero-truncated distribution options while doing it.

Scope reference: ~14,100 LOC across the four pending files (distributions 4,082; portfolio 5,296; utilities 3,789; bounds 943).

---

## Cross-cutting principles

These apply to every stage and were settled in the planning conversation.

### 1. Plotting policy
- Each of `Distortion`, `Aggregate`, `Portfolio` keeps a *single* `.plot()` method, broadly as-is.
- `Portfolio.twelve_plot` stays.
- `bounds.py` keeps its visualizations — they are its product.
- **Every other method** that currently mixes plotting with computation is rewritten *without* plotting. Plotting is added back ad hoc by the caller when needed.
- Rationale: most embedded plotting was put in before the API stabilized and is now mess, not feature.

### 2. Registry/dispatch pattern for typed factories
The pattern already used in `spectral.py` for `Distortion` is the model. First application: refactor `Frequency` away from its giant if/elif `__init__`.

### 3. Public-surface discipline
`bounds.py` is polymorphic over Portfolio/Aggregate's public API — that is the constraint that lets it come last. During the dist and port refactors, **explicitly define** the public surface and stop the leaks. Currently-leaked Aggregate internals that Portfolio reads: `ftagg_density`, `report_ser`, `update_work`, `rescale`, `add_exa`.

### 4. Stage order: dist → port → util → bounds
No utilities pre-pass. Utilities cleanup happens after its main callers (dist, port) have settled — that's when the right shape becomes obvious.

---

## Stage 1 — `distributions.py`

**Goal.** Untangle the god object. Define the public surface that Portfolio and Bounds will rely on.

**Current shape.** Three classes:
- `Frequency` (~640 LOC) — factory disguised as a base class. `__init__` is an if/elif chain over ~18 frequency types; each branch builds closures `_freq_moms(n)` and `pgf(n, z)` stored on `self`.
- `Aggregate` (~2,600 LOC) — orchestrator. `__init__` ~390 lines; `update_work` ~180 lines. Mixes spec storage, exposure broadcasting, severity discretization, FFT convolution, two flavors of reinsurance (occ + agg), validation, statistics, density-DataFrame construction, plotting. Core FFT is only ~5 lines (`ft → freq_pgf → ift`) but buried at distributions.py:1750–1774.
- `Severity` (~760 LOC) — wraps `scipy.stats.rv_continuous`. Multiple input modes (scipy-named / histogram / fixed / meta) handled by `__init__` branches. Layer/attachment logic tangled across `_pdf`/`_cdf`/`_sf`/`_isf`/`_ppf` overrides. `moms()` is ~290 lines of numerical integration with nested `safe_integrate` and many heuristics — fragility hot-spot.

**In scope.**
- (a) **Define the target public API of `Aggregate` first.** Decide what Portfolio is allowed to touch. Plug the leaks listed under cross-cutting principle 3.
- (b) **Frequency → registry pattern** (spectral.py style). Closures become named, testable functions. Add zero-truncated frequency options (currently missing).
- (c) **Disentangle `Aggregate`** into convolution / reinsurance / statistics / plotting layers. The 5-line FFT core should be visible, not buried. Approximation modes and reinsurance branches sit alongside it, not inside.
- (d) **Severity**: separate layer/attachment logic from scipy-wrapping. Isolate `moms()` so it can later be rewritten without disturbing the wrapper.
- Apply the plotting policy: keep `.plot()`, strip plotting from everything else.

**Out of scope.**
- Rewriting `moms()`'s numerical algorithm (isolate it, leave the algorithm for a separate pass).
- Anything in Portfolio.

**Pain markers in current file.** Multiple TODOs including `# WHOA! WTF` at distributions.py:1716, `# TODO sort out` (reinsurance) at ~821, `# TODO check this code still works!` at ~3420.

**Coupling.** ~15 symbols imported from utilities. Lazy-imports `Portfolio` inside `Severity.__init__` (Severity-as-meta-distribution). Star-imports constants.

**Tests to keep green.**
- `uv run pytest` — parametrized over every line of `aggregate/agg/test_suite.agg`.
- `tests/data/expected_specs.json` snapshot — spec-shape canary.
- `uv run python -m aggregate.extensions.test_suite` — HTML report at stage boundary.

---

## Stage 2 — `portfolio.py`

**Goal.** Refactor against the new Aggregate. Aggressively tighten scope around two canonical use cases.

**Canonical use cases driving scope:**
1. Create `Portfolio` → use its `density_df` directly.
2. Create `Portfolio` → Iman-Conover sample.
3. Create `Portfolio` → calibrate distortions → apply / analyze distortions.

Anything that doesn't serve one of these is a candidate for deletion or migration.

**Current shape.** Single `Portfolio` class, ~5,296 LOC, ~75 public methods. Sprawl: collection management, FFT portfolio aggregation, capital allocation, distortion pricing, risk measures, sensitivity, plotting, reporting, persistence, specialty workflows (`uat`, `bodoff`, EVE accounting).

**In scope — deletions and migrations.**
- **Delete unused functions.** Audit; several are never called.
- **Delete the gradient family.** ~196 LOC in `gradient()` plus dependent audit code — not part of the supported use cases.
- **Delete legacy non-spectral pricing methods.** Added for book illustrations, no longer needed.
- **Move `calibrate_distortion(s)` into `Distortion`** (in spectral.py). 15 per-family closures (~238 LOC in `calibrate_distortion`) belong with the distortion families they parameterize. Pending decision, do it here.

**In scope — restructuring.**
- **Mainstream the "switcheroo."** Currently `swap_density_df`, flagged EXPERIMENTAL. It's important — promote to first-class, reengineer accordingly.
- **Constructor consolidation.** Five input modes (dict, Aggregate list, tuples, strings, DataFrame) is too many. Sort them out into a clean surface.
- **Apply plotting policy.** Methods that currently return mixed analysis+matplotlib lose the matplotlib half. Single `.plot()` and `twelve_plot` survive; everything else loses its plot code.

**Deferred (not this stage).**
- The massive in-Portfolio DataFrames are their own problem. Note it; don't try to solve it inside this refactor.
- Finer-grained extraction of plotting/reporting/EVE/sensitivity into separate modules. Decide once dust settles.

**Coupling.** Currently reaches into Aggregate internals (see cross-cutting principle 3). Stage 1 should have already defined the public surface — Portfolio should now use *only* that surface. If a piece of Portfolio can't be expressed against the public surface, that's a signal the public surface needs another method, not that Portfolio should poke internals.

**Pain markers.** 13 TODO/FIXME comments in the file, including `# TODO What is this crap?` at portfolio.py:2185 / :2353, `# TODO EVIL! this reorders lines and messes up …` at :4026. Side-by-side old/new variants exist (`stand_alone_pricing` vs `stand_alone_pricing_work`, `calibrate_distortions` vs `calibrate_distortions2`) — pick one, delete the other.

---

## Stage 3 — `utilities.py`

**Goal.** Slim down to a coherent numerical core. Remove infrastructure that the slimmer downstream modules no longer need.

**Current shape.** 3,789 LOC, 53 functions + 7 classes, organized thematically but interleaved. Asymmetric coupling: distributions imports ~15 symbols, portfolio ~12, bounds 1 (lazy), underwriter ~5, spectral 1, parser 1 (lazy). Utilities depends on none of them — clean direction.

**In scope.**
- **`MomentAggregator` / `MomentWrangler`: rework.** Useful and used but klutzy. **Add fourth moments** while doing this.
- **Delete the AxisManager / FigureManager cluster entirely.** Predates knowing matplotlib properly. As graphics thin out across earlier stages the need dissipates; what remains is easy with raw matplotlib at the call site. No replacement module — just gone.
- **Delete the DataFrame-formatting helpers** (`style_df`, `sEngFormatter`, `GreatFormatter`, `easy_formatter`, `get_fmts`, similar). Not needed once display surface shrinks; pandas plus a couple of inline format calls handles what's left. Review `qd` / `qdp` / `pprint_ex` individually; keep only what's actively useful.
- **`Answer` class → namedtuple / dataclass.** Standard library does this cleanly.
- **Migrate single-caller functions to their caller** where appropriate.
- **Iman-Conover and friends:** decide whether they stay shared or move next to Portfolio. By this stage the answer should be obvious from how Stage 2 ended up.

**Goal end state.** Coherent slim "numerical core" module: FFT, moments (incl. 4th), fitting, the few genuinely-shared algorithms. Viz/formatting infrastructure is removed, not relocated.

**Numba.** One `@njit` function (`make_comonotonic_allocations`), isolated, fallback works. Non-issue.

---

## Stage 4 — `bounds.py`

**Goal.** Reengineer the user-facing surface. Smoke-test the new public APIs of dist and port.

**Current shape.** 943 LOC, one main `Bounds` class plus two standalone helpers. Polymorphic over inputs (Portfolio / Aggregate / Series / density DataFrame). Uses only public methods on its inputs — already well-insulated.

**In scope.**
- **UI / call-pattern reengineering.** Gets the job done but the entry-point surface is disastrously complicated. Serious cleanup of how callers interact with `Bounds`.
- **Local cleanups.** `cloud_view` (~142 LOC) splits computation from matplotlib. Resolve the commented-out `FigureManager` import (bounds.py:13 — note: FigureManager itself is being deleted in Stage 3, so this just goes away).
- **Smoke test.** If the dist/port refactors preserved sane public APIs, Bounds should still work with minimal changes. If it can't, that's evidence those APIs need rework — escalate.

---

## Side task — `frequency_examples` → Quarto

`utilities.py` contains `frequency_examples` (~320 LOC) with no callers. Keep the content — it reads like a blog post and has educational value — but migrate it to a Quarto page with executable Python code that runs to verify itself. Add the zero-truncated distribution options here (or coordinate with Stage 1 where they're added to Frequency). Not part of any module-refactor stage; slot whenever convenient.

---

## Verification (every stage)

- `uv run pytest` after each substantive change. `test_suite.agg` parametrization catches semantic drift fast.
- `uv run python -m aggregate.extensions.test_suite` for the HTML report at stage boundaries.
- `tests/data/expected_specs.json` SLY snapshot stays the spec-shape canary.
- Per stage: identify one or two notebooks / case studies that exercise the touched module end-to-end and run them.

Per CLAUDE.md: `UV_LINK_MODE=copy` is required for `uv` on this path.

---

## Working notes / parking lot

For decisions made mid-stage that affect later stages. Empty at start.
