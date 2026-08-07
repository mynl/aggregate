# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`aggregate` is a Python actuarial library for building fast, accurate compound (aggregate) probability distributions — the sum of a random number of random variables. It targets insurance pricing, capital allocation, reinsurance analysis, and portfolio risk management. The core insight is delivering simulation-like accuracy at parametric distribution speed via FFT-based convolution.

Published at https://aggregate.readthedocs.io/ and https://github.com/mynl/aggregate. Author: Stephen J. Mildenhall.

## Commands

Use `uv` for all environment and dependency management.

**Set `UV_LINK_MODE=copy` whenever invoking `uv`.** The repo lives on a path where uv's default hardlink mode falls back with a warning; copy mode is the supported choice here. The repo's `.claude/settings.local.json` sets this automatically for the Claude harness; in a regular shell run `export UV_LINK_MODE=copy` (POSIX) or `$env:UV_LINK_MODE = "copy"` (PowerShell).

**Sync environment — use `--all-extras` for a dev checkout:**
```
uv sync --all-extras
```
The project's optional dependencies live in five extras — `dev` (docs build,
pytest, ruff), `notebook` (JupyterLab, widgets, jupytext), `numba` (compiled
TVaR/biTVaR paths), `massive` (`zarr`, disk-backed bivariate), and `viz`
(`holoviews`/`datashader`/`bokeh`, bivariate exploration). `--all-extras`
installs the lot in one shot.

**Do NOT `uv sync --extra dev` (or any single extra) in this checkout.** `uv sync`
is an *exact* sync: selecting a subset of extras makes uv **prune** the packages
belonging to the *unselected* extras. `uv sync --extra dev` deletes the
`massive`/`viz`/`numba` packages (`zarr`, `holoviews`, `datashader`, `numba`) and
their transitive deps — which breaks the bivariate suites (and once broke `zarr`'s
pytest plugin mid-prune). Always sync `--all-extras`; add one tool ad hoc with
`uv pip install <pkg>`.

After syncing, JupyterLab is available: `uv run jupyter lab`.

**There are TWO virtualenvs here, and `uv run` may pick the wrong one.**
`.venv` is the development environment — the one to work in. `.doc-venv` is for
**building the docs only**. An ambient `UV_PROJECT_ENVIRONMENT=.doc-venv` is set
in this shell, so a bare `uv run …` silently resolves to `.doc-venv`. Both are
editable installs onto the same `src/`, so test *results* agree — but packages
installed into one are invisible to the other, which is exactly how a tool can
appear installed and still be missing. When it matters, be explicit:

```
.venv/Scripts/python.exe -m pytest          # unambiguous (Windows)
UV_PROJECT_ENVIRONMENT=.venv uv run pytest  # or force uv to the dev env
UV_PROJECT_ENVIRONMENT=.venv uv pip install <pkg>
```

**Run anything in the managed environment:**
```
uv run python ...
uv run jupyter notebook
```

**Run the pytest suite** (primary test mechanism):
```
uv run pytest
```
Tests live in `tests/`. Every line of `aggregate/agg/test_suite.agg` is exercised as its own parametrized test case (one assert-parses test and one shape-regression test against a captured SLY snapshot).

**Build documentation:**
```
cd docs && uv run make.bat html          # Windows
cd docs && uv run make html              # Unix
```

**Do NOT build docs as part of a verification cycle.** The doc tree is large (500+ rendered pages) and the build is slow. When editing `.rst` files during a refactor:
- Keep the `.rst` edits in lockstep with code changes (grep for stale `:meth:` / `:class:` references against deleted/renamed symbols).
- Note in the PR / commit that docs are pending a rebuild.
- The author runs the build manually outside the iteration loop.

**Quick interactive smoke test** (Python/Jupyter):
```python
from aggregate import build, qd
a = build('agg Dice dfreq [3] dsev [1:6]')
qd(a)
```

## Architecture

### Core class hierarchy

```
Frequency          ← base class for frequency distributions
  └── Aggregate    ← compound distribution (frequency × severity)
        └── Portfolio  ← collection of Aggregate units
```

`Severity` is a standalone wrapper around `scipy.stats` continuous RVs and discrete empirical distributions, with support for layers, limits, and spliced forms. `Aggregate` combines a `Frequency` and one or more `Severity` objects via FFT convolution.

### Key modules

| Module | Role |
|---|---|
| `distributions.py` | `Frequency`, `Aggregate`, `Severity` — the computational core |
| `portfolio.py` | `Portfolio` — multi-unit analysis, diversification, capital allocation |
| `spectral.py` | `Distortion` — risk measures (TVaR, Wang, PH, biTVaR, etc.) |
| `underwriter.py` | `Underwriter` — knowledge base, persistence, top-level `build()` entry point |
| `parser.py` | `UnderwritingLexer` / `UnderwritingParser` — DecL lexer/parser using Lark (Earley + dynamic lexer); grammar in `decl.lark` |
| `decl.lark` | The DecL grammar — single source of truth for the language |
| `utilities.py` | FFT helpers, quantile/TVaR functions, plotting, moment utilities |
| `bounds.py` | `Bounds` — pricing bounds (IME 2022 methodology) |
| `constants.py` | Global constants and validation flag definitions |
| `pedagogy.py` | Figure/exhibit generators cited in docs, papers, and blogs (not core API) |
| `pentagon.py` | `Pentagon` — algebra over the (L, P, M, a, Q, lr, pq, coc) accounting identities |
| `ft.py` | `FourierTools` — direct chf inversion (post-extensions consolidation) |
| `tweedie.py` | `Tweedie` — frozen scipy-like distribution; ``tweedie_convert``/``_density`` |

The `extensions/` package was removed at 1.0.0a12. Its useful contents either promoted (pentagon, ft, tweedie), absorbed into `pedagogy.py` (figures, bodoff, plot helpers), or migrated to the PMIR companion package (PIR case-study machinery). No top-level re-exports for `Bounds`, `Tweedie`, `FourierTools`, `Pentagon`, or anything in `pedagogy` — submodule access only.

### DecL — the domain-specific language

Users declare aggregates in a concise DSL. Example:
```
agg MyBook
    100 claims                     # expected claim count
    sev lognorm 100 cv 2           # lognormal severity, mean 100, CV 2
    occurrence net of 50 xs 0      # per-occurrence reinsurance
```

`build('...')` is the primary public API — it parses DecL and returns an `Aggregate` or `Portfolio`. The full grammar is in `aggregate/decl.lark`; reference examples are in `aggregate/agg/test_suite.agg`.

### Computation pattern

1. Severity distribution → discretized PMF on a fixed grid (`bs` = bucket size, `log2` = log₂ of grid points)
2. Frequency distribution → PGF (probability generating function)
3. FFT of severity PMF → raised to PGF power → inverse FFT → aggregate PMF
4. All risk measures (quantiles, TVaR, distortion premiums, allocations) computed from this PMF

Aliasing/moment-matching validation is controlled by flags in `constants.py` and reported via `explain_validation()`.

### numba

`utilities.py` has optional numba-compiled paths for TVaR and biTVaR inner loops. Numba is not required (pure-numpy fallbacks exist); it was removed as a hard dependency in 0.27. It is a declared opt-in extra — `pip install aggregate[numba]` — and is included when you sync a dev checkout with `uv sync --all-extras`.

## Naming conventions

**Subclasses use the `Base<Kind>` prefix form**, not the `<Kind>Base` suffix form. So `FrequencyPoisson`, `FrequencyNegbin`, `SeverityLognorm`, `DistortionPH`, `DistortionTVaR` — not `PoissonFrequency` / `PHDistortion`. Rationale: subclasses sort with their base class alphabetically in the file, in autocomplete, in stack traces, and in docs. Apply this convention to any new class taxonomy introduced during the refactor (Frequency in Stage 1b, Severity in Stage 1d, future Distortion cleanup).

**Mixins use the `<Role>Mixin` suffix form.** The `Base<Kind>` rule above is for
sibling *taxonomies*; a mixin is a different idiom, so the `…Mixin` suffix is the
clear signal. There are two: `LabeledMixin` (`_labeled.py`) — the shared label
surface (`label` / `label_map` / `labels` / `renamer` / `use_labels`) mixed into
`Aggregate`, `Portfolio`, `PnL`, `Severity`, `Distortion`, and `Copula`, which
share no common base — and `HelpMixin` (`_help.py`), the one `help(regex, ...)`
implementation that replaced nine copy-pasted copies across the first-class
classes. Mixins define **no `__init__`** (they stay transparent to the host's
`super()` chain); each `LabeledMixin` host calls an explicit
`self._init_labels(...)` when ready. See `dev/done/plan-labels.md`
(`[DecL-Labels-Everywhere]`) and `dev/done/plan-label-canonical.md`, which
collapsed the old `display_label` / `display_name` twins onto the single
resolved `label` property.

**No cryptic codes — label everything with a descriptive bracketed name.** Every
task, plan, workstream, TODO item, and phase carries a self-describing bracketed
label like `[Reporting-Guidelines]`, `[PnL-First-Class]`, `[Signed-Bounded-Window]`
— **never** a terse mnemonic (`N6`, `H4`, `T2`, `P3`, bare `[A]`/`[B]`). The
author cannot hold short codes in their head across sessions; a spelled-out label
reads on sight, and the brackets mark it as a label. The author may still *type* a
short code at you in conversation (trusting your recall) — answer in full labels
anyway, and write only full labels into any artifact. When relabeling a legacy
item, keep any external cross-reference (a GitHub issue `#49`) in parentheses but
lead with the descriptive label. This applies to `dev/TODO.md`, plan files, commit
subjects, and run summaries alike.

**Vet every new public method / attribute / kwarg name against the existing surface before adding it — at planning time, not after.** A new instance attribute set in `__init__` *silently shadows* a method of the same name (Python resolves the instance dict first), so `self.approximate = ...` turned the `Aggregate.approximate()` method into an uncallable string and went unnoticed through a whole plan (`dev/done/plan-approximate.md`). Before introducing a name: `rg` for it across `src/aggregate` (method defs, properties, `self.<name> =` assignments, DecL keywords / spec keys), confirm it does not collide with an existing callable or attribute on the *same* class (or its parents), and prefer a distinct noun for a stored value vs. a verb for an action (e.g. attribute `approximation` holding the kind, method `approximate()` doing the fit). Call out the chosen names explicitly in the plan so they can be reviewed.

## Documentation and docstrings

All new functions and any modified existing functions must include a docstring. The project uses NumPy-style docstrings (Parameters / Returns / Notes sections). For non-trivial mathematical logic, the Notes section should explain the algorithm or formula: this is an actuarial library where the "why" is often as important as the "what". Inline comments are appropriate for non-obvious numpy/FFT operations.

**No dashes as punctuation. Ever.** Not the em dash `—`, not the ASCII
double `--`, not a spaced hyphen ` - `. The dash-as-aside is the single
loudest AI tell in prose, and the author does not write that way. It applies
everywhere text is authored: docstrings, comments, `note{}` / `doc{{{}}}`
bodies, `.rst` and `.qmd` pages, `CHANGELOG.md`, `dev/` plans, commit
subjects, and replies in the terminal.

Rewrite instead of substituting. A dash is almost always doing a job that a
comma, a colon, parentheses, or a full stop does better:

- aside or gloss, use commas or parentheses: ~~`the count is an output -- not an input`~~ to `the count is an output, not an input`
- explanation or expansion, use a colon: ~~`one frame -- identity first`~~ to `one frame: identity first`
- a second thought, use a second sentence: ~~`it works -- but watch the tail`~~ to `it works. Watch the tail.`

Still fine, because these are not punctuation: hyphenated compounds
(`loss-ratio`, `first-class`, `zero-truncated`), negative numbers, ranges
written with `to`, command-line flags (`--all-extras`), and `--` inside code
or DecL.

## Citations and bibliography (standing order)

All authored documents — Quarto `.qmd` pages and any future artifact that supports citations — reference the author's master BibTeX library. This applies to every future doc request without being re-asked.

- **Bibliography:** `C:/s/TELOS/Biblio/uber-library.bib` (~7,100 entries), maintained by the author with `archivum` (`C:/s/TELOS/Python/archivum_project`). **Read-only from this project — never edit it.** If a needed reference is missing, list it in the run summary for the author to add via archivum, then cite once the key exists.
- **Keys** follow `AuthorYYYY[a-z]` (e.g. `Mildenhall2022a`). Always `rg` the bib file for the exact key — never guess or fabricate one.
- **Quarto YAML** on every page:
  ```yaml
  bibliography: C:/s/TELOS/Biblio/uber-library.bib
  csl: C:/s/TELOS/Biblio/journal-of-risk-and-uncertainty.csl
  ```
- Cite inline with `@Key` / `[@Key; @Key2]`; pages that cite end with a `## References` heading over an empty `::: {#refs}` div.
- House anchors: `Mildenhall2022` (Similar Risks Have Similar Prices, IME), `Mildenhall2022a` (*Pricing Insurance Risk*, with Major), `Major2026` (*Introduction to Capital Modeling and Portfolio Management*, CAS), `Grubel1999`/`Grubel2000` (FFT compound distributions), `Klugman2012` (*Loss Models*), `Heckman1983`, `Panjer1981`, `Wang1995`/`Wang1996` (distortions).

## Testing

The pytest suite at `tests/` is the primary test mechanism — run with `uv run pytest`. Each line of `aggregate/agg/test_suite.agg` (categories A–O: frequencies, severities, reinsurance, distortions, case studies, papers) becomes two parametrized cases:

- `test_line_parses` — the line parses to a valid `(kind, name, spec)` shape.
- `test_spec_matches_snapshot` — the spec matches `tests/data/expected_specs.json`, a snapshot captured from the legacy SLY parser before the Lark migration. This catches semantic drift in the grammar/transformer.

The snapshot can be regenerated with `uv run python tests/capture_sly_snapshot.py` IF the SLY parser is restored from git history; otherwise treat it as a frozen reference.

Validation failures surface as warnings via `explain_validation()`; numerical issues (aliasing, CV mismatch, skewness) set flags in `constants.py`.

### Running the suite efficiently (standard operating procedure)

**The suite is not bloated — measure before trimming it.** As of 1.0.0a161:
**2,562 fast cases in ~105 s, i.e. ~41 ms per test** on FFT/numpy-bound work.
The three parametrized corpus files (`test_decl_parser` 326 cases,
`test_decl_unparser` 444, `test_grammar_sync` 169) are 37% of the case count but
only ~29% of wall clock. If the suite *feels* slow, the cause is almost always
running the whole thing inside the edit loop. Re-measure with
`uv run pytest -m 'slow or not slow' --durations=40` before proposing a cull.

Three tiers. Use the first one that covers the change:

- **1. Edit loop — `pytest -n0 --dist no --testmon-forceselect`.**
  `pytest-testmon` records which tests execute which source lines and reruns
  **only those your edit touched**. Measured on this repo: a 3-file scope went
  **10.9 s → 0.16 s** when nothing changed, and a real edit to
  `src/aggregate/recipe.py` selected **6 of 55** in 0.71 s. The first run builds
  the map (one full run); every run after is near-instant.

  Every flag in that command is load-bearing — this is not `--testmon` alone:
  - **`--testmon-forceselect`, not `--testmon`.** `addopts` carries
    `-m 'not slow'`, and testmon *silently* downgrades to
    `--testmon-noselect` (reorder, deselect nothing) whenever `-m` / `-k` /
    `--lf` / `::test_name` is in play. Plain `--testmon` therefore looks like
    it works — it writes `.testmondata` and prints no warning — while running
    every test. `--testmon-forceselect` intersects the impact set with the
    selectors, which is what you actually want.
  - **`-n0 --dist no`.** testmon traces coverage in-process; xdist breaks it.
    Both flags are needed to override `-n auto --dist loadgroup` in `addopts`
    (`-p no:xdist` does *not* work — it makes those `addopts` unparseable).
    Losing parallelism costs nothing when the point is running 6 tests.
  - A **comment-only edit correctly selects nothing** — testmon hashes
    executable blocks, not file mtimes. That is right, not a failure.
  - Database is `.testmondata` (gitignored); delete it to force a rebuild.
  - No-setup fallbacks: `pytest tests/test_pnl.py` (one file, or
    `::test_name`), `-k "pnl and not engine"`, `--lf` (rerun last failures;
    `--ff` failures-first, `-x` stop at first).
- **2. Pre-commit — `uv run pytest`.** The full fast suite, parallel via
  `-n auto`, `slow` deselected. This is the gate for "am I done", **not** an
  edit-loop tool. Running it eight times in one session is the mistake this
  section exists to prevent.
- **3. Version bump — `uv run pytest -m 'slow or not slow'`.** Everything,
  ~6 minutes. Once, at the commit boundary.

**Do NOT rely on eyeballing blast radius as the *only* check.** Use tier 1 to
iterate; always finish with tier 2 before declaring a change done, and tier 3
before a bump.
- **Numerics gate — `uv run pytest -m 'slow or not slow' -W error::RuntimeWarning`.**
  The library emits no `RuntimeWarning` of its own as of `1.0.0a220`
  (`[RuntimeWarning-Census]`), so a new one is a real finding: either
  arithmetic whose result is discarded, which wants an `np.errstate` guard and
  a `Notes` paragraph saying why, or a `NaN` that reaches an answer, which
  wants a fix. Deliberately **not** in `addopts`: a third-party release could
  break the everyday loop, and one bivariate case flakes under it
  (`[Bivariate-Gate-Flake]`). Run it at a numerics-touching bump.
- **`slow` marker — fast-by-default.** `addopts` carries `-m 'not slow'`, so the
  everyday `uv run pytest` skips the quarantined heavy cases. The three
  bleeding-edge bivariate suites (`test_bivariate.py`, `test_massive_bivariate.py`,
  `test_reins_bivariate.py`) are tagged `slow` at module level via
  `pytestmark = pytest.mark.slow` — they hold the two multi-minute monsters that
  set the whole suite's wall-clock floor. **Run everything (the gate / CI) with
  `uv run pytest -m 'slow or not slow'`;** run just the heavy suites with
  `-m slow`. To quarantine a new expensive case, tag it `@pytest.mark.slow` (or
  add `pytestmark` for a whole module); find candidates with
  `uv run pytest -m 'slow or not slow' --durations=20`.
- **`xdist_group` — memory, not time.** `addopts` carries `--dist loadgroup`;
  ungrouped tests are distributed as before, but every test sharing an
  `xdist_group` name goes to **one** worker and so runs sequentially against its
  group-mates. `test_bivariate.py` uses it: each case allocates a 2-D FFT grid,
  and several running concurrently once exhausted memory (a numpy allocation
  failure in a test whose own grid was 64x64 — the pressure was its neighbours).
  Reach for this when tests are individually fine but collectively too large;
  reach for `slow` when a test is simply long. Costs ~10% on the gate.
  `--strict-markers` is on,
  so every marker must be registered in `[tool.pytest.ini_options].markers`.
- **Sync with `uv sync --all-extras`, never a single `--extra`.** `uv sync` is an
  *exact* sync: it prunes anything outside the selected extras. The optional deps
  are split across five extras (`dev`/`notebook`/`numba`/`massive`/`viz`), so
  `uv sync --extra dev` *deletes* the `massive`/`viz`/`numba` packages (`zarr`,
  `holoviews`, `datashader`, `numba`) and breaks the bivariate suites — this once
  broke `zarr`'s pytest plugin by deleting `donfig`/`google-crc32c` mid-prune.
  `--all-extras` selects them all, so nothing gets pruned. Add a one-off tool with
  `uv pip install <pkg>`; `uv sync --inexact` is a fallback that keeps extraneous
  packages if you ever need it.

## Release & housekeeping workflow

These are standing rules — follow them without being re-asked:

- **Every plan-based code change bumps the version.** Any code change executed
  from a plan (a `dev/plan-*.md`, or any multi-step feature/refactor) bumps the
  `1.0.0a*` version in `pyproject.toml`. (Pure tidying — file moves, comment or
  doc-only edits with no behaviour change — does not.)
- **Keep `CHANGELOG.md` current.** Each version bump adds a `## <version>`
  section to `CHANGELOG.md` (the running release-notes draft) describing what
  landed and any breaking changes — add it at the close of the iteration, don't
  defer. `README.md` is the stable-audience front page (purpose, install,
  getting started, links) and points at `CHANGELOG.md`; touch it only when that
  front-page material itself changes.
- **Keep `dev/TODO.md` current.** When a tracked item lands, mark it done (and
  note the version / `dev/done/plan-*.md`); when scope shifts, edit the entry.
  Move a completed plan from `dev/` to `dev/done/`.
- **Claude commits version bumps; the author commits everything else.** Every
  version bump is committed by Claude as its own commit, at the moment it lands.
  A multi-step implementation that bumps three times leaves three commits, so
  the history stays **bisectable** — one bump, one commit, never batched, never
  deferred to the end. Anything that does *not* bump the version (pure tidying,
  file moves, doc-only edits, work in progress) is left uncommitted for the
  author to handle.
- **Version-bump commit messages are ONE LINE.** House format, unchanged from
  existing history:

  ```
  [Descriptive-Label] a149: terse summary of what landed
  ```

  Subject only — **no body, no trailers, no `Co-Authored-By`**. `CHANGELOG.md`
  *is* the full commit message; the one-liner is the index into it. If the
  summary will not fit on one line, either the commit is too big or the
  CHANGELOG entry is not doing its job.
- **A version-bump commit is one coherent unit.** It carries the code change,
  the `pyproject.toml` bump, the `CHANGELOG.md` section, and whatever release
  hygiene the change implies — `dev/TODO.md`, `dev/FEATURES.csv` (via
  `dev/regen_features.py`), grammar-reference regen, moving a finished plan to
  `dev/done/`. Never split those across commits, and never let a version land in
  a different commit from its CHANGELOG section — that is what breaks bisect.
- **Never push; never bypass.** Pushing to GitHub is the author's, on explicit
  request only. No `--no-verify`, no `--amend` of anything already pushed —
  prefer a new commit to rewriting one.
- **Check git state before asserting it — never from memory.** The author also
  commits frequently without announcing it, so any claim about what is or isn't
  committed written from session memory is routinely *wrong* by the time it is
  read. Run `git status` / `git log` first; mention uncommitted work only when
  the tree really is dirty (e.g. an *expected* commit is missing). When in
  doubt, stay silent about commits rather than nag.

## TODO

The full pending list — pre-ship work and post-v1.0 ideas — lives in
**`dev/TODO.md`**; what's landed is in `CHANGELOG.md` and the git log. Check
there before proposing structural changes so you don't reinvent something
already scoped (or already deferred for a reason).

- **PIR case-study reproduction.** The `CaseStudy` machinery (formerly `extensions/case_studies.py`, `portfolio_pir.py`, `risk_progression.py`, and the `cnc`/`discrete`/`hs`/`tame` runner scripts) was deleted at 1.0.0a12. **PMIR is a separate forward-looking project and does NOT reproduce PIR exhibits** — do not point users at it for that purpose. The only path to reproducing the published PIR exhibits is `pip install aggregate==0.30.1` in an isolated environment.
