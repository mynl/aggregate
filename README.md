 [![Commit activity](https://img.shields.io/github/commit-activity/m/mynl/aggregate)](https://github.com/mynl/aggregate) [![Documentation Status](https://readthedocs.org/projects/aggregate/badge/?version=latest)](https://aggregate.readthedocs.io/en/latest/) [![Latest version](https://img.shields.io/pypi/v/aggregate.svg?label=pypi)](https://pypi.org/project/aggregate)
 ![Supported Python versions](https://img.shields.io/pypi/pyversions/aggregate.svg) [![Downloads](https://img.shields.io/pypi/dm/aggregate.svg)](https://pepy.tech/project/aggregate) [![Github stars](https://img.shields.io/github/stars/mynl/aggregate.svg)](https://github.com/mynl/aggregate/stargazers) [![Github forks](https://img.shields.io/github/forks/mynl/aggregate.svg)](https://github.com/mynl/aggregate/network/members)
 [![License](https://img.shields.io/pypi/l/aggregate.svg)](https://github.com/mynl/aggregate/blob/master/LICENSE) [![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10557199.svg)](https://zenodo.org/records/10557199)

------------------------------------------------------------------------

# aggregate: working with actuarial compound distributions

## Purpose
`aggregate` builds approximations to compound (aggregate) probability distributions quickly and accurately.
It can be used to solve insurance, risk management, and actuarial problems using realistic models that reflect
underlying frequency and severity. It delivers the speed and accuracy of parametric distributions to situations
that usually require simulation, making it as easy to work with an aggregate (compound) probability distribution
as the lognormal. `aggregate` includes an expressive language called DecL to describe aggregate distributions
and is implemented in Python under an open source BSD-license.

## Version 1.0
Version 1.0 represents a substantial extension over the prior 0.30.1 release. It was written in collaboration with Claude Code. Version 1.0 adds:

* The ability to model positive and negative amounts, opening the way for a specific PnL profit-and-loss class.
* Automated generation of incremental gross–ceded–net views across multi-layer occurrence and aggregate programs.
* The FFT calculations are now managed in a window that need not include the origin, providing more efficient discretization.
* Bivariate distributions, including bivariate severity, clash, ceded–net, gross-cede, and gross–net models.
* All standard reinsurance variable features: swings, slides, profit commissions, reinstatements, and loss corridors, as well as loss-sensitive retro rating used in large accounts.

## Aggregate White Paper

[Aggregate: fast, accurate, and flexible approximation of compound probability distributions](https://www.cambridge.org/core/journals/annals-of-actuarial-science/article/aggregate-fast-accurate-and-flexible-approximation-of-compound-probability-distributions/1BF9A534D944D983B1D780C60885F065) describes the `Aggregate` class within `aggregate`. This paper has been published in the peer reviewed journal [Annals of Actuarial Science](https://www.cambridge.org/core/journals/annals-of-actuarial-science)'s Actuarial Software series.
The paper describes the purpose, implementation, and use `Aggregate`, showing how it can be used to create and manipulate compound frequency-severity distributions.

## Changelog

See [CHANGELOG.md](https://github.com/mynl/aggregate/blob/master/CHANGELOG.md) for the full version history.

## API stability

Almost everything is **stable**. `Aggregate`, `Portfolio`, `PnL`, `Severity`, `Frequency`, `Distortion`, `BivariateAggregate`, `Underwriter`, `build`, `qd` and the DecL grammar carry the usual promise: from 1.0 onward a documented name keeps its meaning, and a breaking change waits for a major release after a deprecation period.

Two modules are **provisional**, in the sense of [PEP 411](https://peps.python.org/pep-0411/): `aggregate.charts` and `aggregate.exhibits`. They are not part of the 1.0 API contract and may change in a minor release with no deprecation period. They are additive side projects to the release, they import from the core and the core does not import them, so nothing in them can reach the stable surface. They are public on purpose: use them and report what does not fit, which is how a provisional module graduates to stable.

The full statement, including exactly what "provisional" covers in each, is in the [API Stability](https://aggregate.readthedocs.io/en/latest/3_reference/3_x_API_Stability.html) page of the documentation.

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
any machine. Optional extras add capabilities:

```bash
uv add "aggregate[numba]"     # numba-compiled TVaR paths
uv add "aggregate[viz]"       # interactive bivariate exploration: holoviews, datashader, bokeh
uv add "aggregate[massive]"   # disk-backed bivariate grids via zarr
uv add "aggregate[notebook]"  # JupyterLab, widgets
uv add "aggregate[dev]"       # documentation build and test tooling
uv add "aggregate[all]"       # all of the above except dev
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

To get started, import `build`. It provides easy access to all functionality. The function `qd` is a quick display helper, printing germane information.

Here is a model of the sum of three dice rolls. Running `qd(a)` prints the mean, SD, CV, skewness, and 1st, 50th and 99th percentiles for the frequency, severity, and aggregate components. Common statistical functions like the cdf and quantile function are built-in. The whole probability distribution is available in `a.density_df`.

    from aggregate import build, qd
    a = build('agg Dice dfreq [3] dsev [1:6]')
    qd(a)

    >>> Aggregate object: Dice. Frequency distribution empirical. Severity
        dhistogram, [1, 6], bounded; atoms [1 2 3 4 5 6]. Updated with bucket
        size 1 and log2 = 5. Validation: not unreasonable.

    >>>       Mean     SD      CV Skew P01 Median P99
    >>> X
    >>> Freq     3      0       0
    >>> Sev    3.5 1.7078 0.48795    0   1      3   6
    >>> Agg   10.5  2.958 0.28172    0   4     10  17

    print(f'\nProbability sum < 12 = {a.cdf(12):.3f}\nMedian = {a.q(0.5):.0f}')

    >>> Probability sum < 12 = 0.741
    >>> Median = 10

`aggregate` can use any `scipy.stats` continuous random variable as a severity, and
supports all common frequency distributions. Here is a compound-Poisson with lognormal
severity, mean 50 and cv 2.

    a = build('agg Example 10 claims sev lognorm 50 cv 2 poisson')
    qd(a)

    >>> Aggregate object: Example. Frequency distribution poisson. Severity lognorm,
        [0, inf), subexponential right tail. Updated with bucket size 2 and log2 =
        16. Validation: not unreasonable.

    >>>       Mean     SD      CV    Skew P01 Median  P99
    >>> X
    >>> Freq    10 3.1623 0.31623 0.31623
    >>> Sev     50    100       2  13.981   2     22  428
    >>> Agg    500 353.56 0.70711  3.5312  68    422 1736

See the documentation for more examples.

## In JupyterLab: the `%%agg` cell magic

In a notebook the DecL does not have to live inside a Python string. Load the magic once per kernel:

    %load_ext aggregate.magics

and write the program as the cell:

    %%agg
    agg Dice dfreq [3] dsev [1:6]

which is exactly `a = build('agg Dice dfreq [3] dsev [1:6]')` followed by `qd(a)`, with two conveniences: the program is not in quotes, so your editor still highlights it as DecL, and the object is bound to its declared name as well as to `a`, giving both `a` and `Dice`.

Full descriptions are in the [documentation](https://aggregate.readthedocs.io/en/latest/).

## Dependencies

See pyproject.toml.


## Running the tests

The pytest suite lives in `tests/`:

    uv run pytest                             # fast suite (multi-minute cases deselected)
    uv run pytest -m "slow or not slow"       # everything, including the slow bivariate cases
    uv run pytest tests/test_decl_parser.py   # one file; add -k "pattern" to filter by name

## License

BSD 3 license.

## Contributions

All contributions, bug reports, bug fixes, documentation improvements,
enhancements and ideas are welcome. Create a pull request on github and/or
email me.

Social media: <https://www.reddit.com/r/AggregateDistribution/>.

Blog: <https://blog.mynl.com>
