 [![Latest Version](https://img.shields.io/github/commit-activity/m/mynl/aggregate)](https://github.com/mynl/aggregate) [![Documentation Status](https://readthedocs.org/projects/aggregate/badge/?version=latest)](https://aggregate.readthedocs.io/en/latest/) [![Latest version](https://img.shields.io/pypi/v/aggregate.svg?label=pypi)](https://pypi.org/project/aggregate)
 ![Supported Python versions](https://img.shields.io/pypi/pyversions/aggregate.svg) [![Downloads](https://img.shields.io/pypi/dm/aggregate.svg)](https://pepy.tech/project/aggregate) [![Github stars](https://img.shields.io/github/stars/mynl/aggregate.svg)](https://github.com/mynl/aggregate/stargazers) [![Github forks](https://img.shields.io/github/forks/mynl/aggregate.svg)](https://github.com/mynl/aggregate/network/members)
 [![License](https://img.shields.io/pypi/l/aggregate.svg)](https://github.com/mynl/aggregate/blob/master/LICENSE) [![Binary packages](https://repology.org/badge/tiny-repos/python:aggregate.svg)](https://repology.org/metapackage/python:aggregate/versions) [![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10557199.svg)](https://zenodo.org/records/10557199)

------------------------------------------------------------------------

# aggregate: working with actuarial compound distributions

## Purpose

`aggregate` builds approximations to compound (aggregate) probability distributions quickly and accurately.
It can be used to solve insurance, risk management, and actuarial problems using realistic models that reflect
underlying frequency and severity. It delivers the speed and accuracy of parametric distributions to situations
that usually require simulation, making it as easy to work with an aggregate (compound) probability distribution
as the lognormal. `aggregate` includes an expressive language called DecL to describe aggregate distributions
and is implemented in Python under an open source BSD-license.

## Aggregate White Paper

[Aggregate: fast, accurate, and flexible approximation of compound probability distributions](https://www.cambridge.org/core/journals/annals-of-actuarial-science/article/aggregate-fast-accurate-and-flexible-approximation-of-compound-probability-distributions/1BF9A534D944D983B1D780C60885F065) describes the `Aggregate` class within `aggregate`. This paper has been published in the peer reviewed journal [Annals of Actuarial Science](https://www.cambridge.org/core/journals/annals-of-actuarial-science)'s Actuarial Software series.
The paper describes the purpose, implementation, and use `Aggregate`, showing how it can be used to create and manipulate compound frequency-severity distributions.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for the full version history.

## Documentation

<https://aggregate.readthedocs.io/>

## Where to get it

<https://github.com/mynl/aggregate>

## Installation

`aggregate` requires Python 3.12 or later. The strongly recommended way to
install and manage it is with [uv](https://docs.astral.sh/uv/), Astral's fast
Python package and project manager — follow the
[uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)
to get it.

Once uv is installed, add `aggregate` to a uv-managed project:

```bash
uv init myproject      # or cd into an existing uv project
cd myproject
uv add aggregate       # resolves, locks, and installs into .venv
```

`uv add` records the dependency in your `pyproject.toml` and syncs the project
environment; from then on `uv sync` recreates that exact, locked environment on
any machine. To also pull the optional documentation and test tooling, request
the `dev` extra:

```bash
uv add "aggregate[dev]"
```

Run anything inside the managed environment with `uv run`, for example
`uv run python` or `uv run jupyter lab`. All the code examples have been tested
in such an environment and the documentation builds in it.

If you already have an environment and simply want the package dropped into it,
install it directly — with uv:

```bash
uv pip install aggregate
```

or with plain pip:

```bash
pip install aggregate
```

## Getting started

To get started, import `build`. It provides easy access to all functionality.

Here is a model of the sum of three dice rolls. The DataFrame `describe` compares exact mean, CV and skewness with the `aggregate` computation for the frequency, severity, and aggregate components. Common statistical functions like the cdf and quantile function are built-in. The whole probability distribution is available in `a.density_df`.

    from aggregate import build, qd
    a = build('agg Dice dfreq [3] dsev [1:6]')
    qd(a)

\>\>\>        EX Est EX     Err EX      CV  Est CV Sk Est Sk
\>\>\> X
\>\>\> Freq    3                         0
\>\>\> Sev   3.5    3.5          0 0.48795 0.48795  0      0
\>\>\> Agg  10.5   10.5 2.2204e-16 0.28172 0.28172  0      0
\>\>\> log2 = 5, bandwidth = 1, validation: not unreasonable.

    print(f'\nProbability sum < 12 = {a.cdf(12):.3f}\nMedian = {a.q(0.5):.0f}')

\>\>\> Probability sum \< 12 = 0.741
\>\>\> Median = 10

`aggregate` can use any `scipy.stats` continuous random variable as a severity, and
supports all common frequency distributions. Here is a compound-Poisson with lognormal
severity, mean 50 and cv 2.

    a = build('agg Example 10 claims sev lognorm 50 cv 2 poisson')
    qd(a)

\>\>\>       EX Est EX     Err EX      CV  Est CV      Sk Est Sk
\>\>\> X
\>\>\> Freq  10                   0.31623         0.31623
\>\>\> Sev   50     50 8.8689e-06       2       2      14 13.981
\>\>\> Agg  500    500 8.8689e-06 0.70711 0.70711  3.5355 3.5312
\>\>\> log2 = 16, bandwidth = 2, validation: not unreasonable.

    # cdf and quantiles


\>\>\> Pr(X\<=500)=0.612
\>\>\> 0.99 quantile=1736.0

See the documentation for more examples.

## Dependencies

See requirements.txt.

## Install from source

    git clone --no-single-branch --depth 50 https://github.com/mynl/aggregate.git .

    python -mvirtualenv ./venv
    # activate the virtual environment (Windows, YRMV)
    venv\Scripts\activate.bat

    # install the package
    pip install aggregate[dev]

## Running the tests

All commands assume `UV_LINK_MODE=copy` is set (the Claude harness
sets it automatically via `.claude/settings.local.json`; in a regular
shell run `$env:UV_LINK_MODE = "copy"` on PowerShell or
`export UV_LINK_MODE=copy` on POSIX).

The pytest suite lives in `tests/`. Each line of
`src/aggregate/agg/test_suite.agg` (categories A–O) becomes two
parametrized cases: `test_line_parses` and
`test_spec_matches_snapshot` (against
`tests/data/expected_specs.json`). Splice cases come from
`test_suite2.agg`.

Whole suite:

    uv run pytest                          # full suite
    uv run pytest -v                       # verbose
    uv run pytest -x                       # stop at first failure
    uv run pytest --lf                     # re-run last failed
    uv run pytest --ff                     # last failed first, then rest

Single file or single test:

    uv run pytest tests/test_decl_parser.py
    uv run pytest tests/test_decl_parser.py::test_line_parses
    uv run pytest "tests/test_decl_parser.py::test_line_parses[A.01]"

Filter by name pattern:

    uv run pytest -k "splice"
    uv run pytest -k "parses and not snapshot"

Output / debugging:

    uv run pytest -s                       # don't capture stdout
    uv run pytest --tb=short               # shorter tracebacks
    uv run pytest -l                       # show local vars on failure
    uv run pytest --pdb                    # drop into pdb on first failure

The individual test modules:

    uv run pytest tests/test_decl_parser.py             # DecL parser, parametrized
    uv run pytest tests/test_distortion_calibrate.py    # Distortion calibration
    uv run pytest tests/test_portfolio_peg_regression.py # Portfolio vs peg_baseline.json
    uv run pytest tests/test_severity_layer_golden.py   # Severity layer golden
    uv run pytest tests/test_splice_suite.py            # Spliced severity
    uv run pytest tests/test_underwriter.py             # Underwriter build/persist

Regenerating golden / baseline files (scripts, not pytest cases —
only when intentionally updating snapshots):

    uv run python tests/capture_sly_snapshot.py        # expected_specs.json
    uv run python tests/capture_peg_baseline.py        # peg_baseline.json
    uv run python tests/capture_severity_golden.py     # severity_layer_golden.json

Visual / non-pytest check (HTML report with plots, builds every
`test_suite.agg` line):

    uv run python -m aggregate.extensions.test_suite

Tight inner loop: `uv run pytest -x --tb=short` — stop fast, readable
failure. Single area being changed: `uv run pytest tests/test_<file>.py -v`.
Pre-commit: plain `uv run pytest`.

## License

BSD 3 licence.

## Help and contributions

Limited help available. Email me at <help@aggregate.capital>.

All contributions, bug reports, bug fixes, documentation improvements,
enhancements and ideas are welcome. Create a pull request on github and/or
email me.

Social media: <https://www.reddit.com/r/AggregateDistribution/>.
