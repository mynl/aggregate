# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`aggregate` is a Python actuarial library for building fast, accurate compound (aggregate) probability distributions — the sum of a random number of random variables. It targets insurance pricing, capital allocation, reinsurance analysis, and portfolio risk management. The core insight is delivering simulation-like accuracy at parametric distribution speed via FFT-based convolution.

Published at https://aggregate.readthedocs.io/ and https://github.com/mynl/aggregate. Author: Stephen J. Mildenhall.

## Commands

Use `uv` for all environment and dependency management.

**Set `UV_LINK_MODE=copy` whenever invoking `uv`.** The repo lives on a path where uv's default hardlink mode falls back with a warning; copy mode is the supported choice here. The repo's `.claude/settings.local.json` sets this automatically for the Claude harness; in a regular shell run `export UV_LINK_MODE=copy` (POSIX) or `$env:UV_LINK_MODE = "copy"` (PowerShell).

**Sync environment:**
```
uv sync
```

**Install with dev extras** (docs build, pytest):
```
uv sync --extra dev
```

**Install with notebook extras** (JupyterLab, widgets, jupytext — for interactive testing/hacking):
```
uv sync --extra notebook
uv run jupyter lab
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

**Generate the visual report** (HTML with plots, optional):
```
uv run python -m aggregate.extensions.test_suite
```
This builds every object in `test_suite.agg`, plots it, and writes an HTML page. Useful for spot-checking the full DecL → build → statistics pipeline.

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

`aggregate/extensions/` contains optional modules (case studies, figures, the visual test-suite reporter, Fourier tools, Tweedie distributions). These are not imported by the core package and are slated for removal at 1.0; anything important moves into the core then.

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

`utilities.py` has optional numba-compiled paths for TVaR and biTVaR inner loops. Numba is not required (pure-numpy fallbacks exist); it was removed as a hard dependency in 0.27.

## Naming conventions

**Subclasses use the `Base<Kind>` prefix form**, not the `<Kind>Base` suffix form. So `FrequencyPoisson`, `FrequencyNegbin`, `SeverityLognorm`, `DistortionPH`, `DistortionTVaR` — not `PoissonFrequency` / `PHDistortion`. Rationale: subclasses sort with their base class alphabetically in the file, in autocomplete, in stack traces, and in docs. Apply this convention to any new class taxonomy introduced during the refactor (Frequency in Stage 1b, Severity in Stage 1d, future Distortion cleanup).

## Documentation and docstrings

All new functions and any modified existing functions must include a docstring. The project uses NumPy-style docstrings (Parameters / Returns / Notes sections). For non-trivial mathematical logic, the Notes section should explain the algorithm or formula — this is an actuarial library where the "why" is often as important as the "what". Inline comments are appropriate for non-obvious numpy/FFT operations.

## Testing

The pytest suite at `tests/` is the primary test mechanism — run with `uv run pytest`. Each line of `aggregate/agg/test_suite.agg` (categories A–O: frequencies, severities, reinsurance, distortions, case studies, papers) becomes two parametrized cases:

- `test_line_parses` — the line parses to a valid `(kind, name, spec)` shape.
- `test_spec_matches_snapshot` — the spec matches `tests/data/expected_specs.json`, a snapshot captured from the legacy SLY parser before the Lark migration. This catches semantic drift in the grammar/transformer.

The snapshot can be regenerated with `uv run python tests/capture_sly_snapshot.py` IF the SLY parser is restored from git history; otherwise treat it as a frozen reference.

`aggregate/extensions/test_suite.py` is a visual reporter — it builds each object, plots it, writes HTML. It complements pytest but is not an assertion suite.

Validation failures surface as warnings via `explain_validation()`; numerical issues (aliasing, CV mismatch, skewness) set flags in `constants.py`.

## TODO

- **Docs reference SLY-era grammar.** `docs/4_agg_language_reference/` describes the grammar in SLY's `@_` form and may mention shift/reduce conflicts. After the Lark migration both descriptions are stale; the grammar reference should `include` `aggregate/decl.lark` (or call `aggregate.parser.grammar(add_to_doc=True)` which writes `docs/4_agg_language_reference/ref_include.rst`).

- **Docstring style sweep.** `iman_conover.py` and `moments.py` use Sphinx `:param x:` style with many empty parameter slots; the rest of the codebase uses NumPy style (per CLAUDE.md "Documentation and docstrings"). Convert in one sweep across all modules — public surface first, private helpers second. ~25 docstrings in moments + iman_conover alone, plus pockets in other modules. Do as a single dedicated pass, not piecemeal.

- **`xsden_to_meancv` vs `xsden_to_meancvskew` tail-mass inconsistency** (`moments.py`). `xsden_to_meancvskew` adds one bucket-width to the tail-mass point (`xsm = xsm + bs` at line 455), `xsden_to_meancv` does not. With `xs[0] = 0` (the typical FFT use), `bs = xs[1]` and the difference is one bucket on the mean correction — small for FFT use, but the two should agree. Decide which convention is correct (likely the meancvskew "place at xs[-1] + bs" form, putting the tail mass at the *next* bucket) and apply to both.

- **`pedagogy.py` / `figures.py` consolidation.** Several modules carry paper/blog figure-generation helpers that have no internal consumers: `ft.py`'s `poisson_example`, `fft_wrapping_illustration`, `recentering_convolution`, `recentering_convolution_example`; the `tweedie_illustration` in `tweedie.py`; likely others scattered across `extensions/`. Pull these together into a single `aggregate.pedagogy` (or `aggregate.figures`) module — keeps `ft.py` / `tweedie.py` focused on the API, gives readers one place to find figure-generators when reproducing paper results. For now the helpers stay where they are.
