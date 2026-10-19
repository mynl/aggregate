"""Library figures fit a document's page: :func:`aggregate.plots.fit_page`.

``dev/plan-library-figures.md`` (``[Library-Figures-Fit-Page]``). With no page
set nothing moves, which the pixel baselines in ``test_chartdoc_render.py``
prove. These tests check the rule itself and the figures it reaches. The
programs mirror lines of ``src/aggregate/agg/decl-testers.agg``.
"""
from __future__ import annotations

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from aggregate import build  # noqa: E402
from aggregate.plots import fit_page, make_grid, make_mosaic  # noqa: E402
from aggregate.plots._style import _fit_to_page  # noqa: E402

_AGG = 'agg P.Tail.LN 5 claims sev lognorm 100 cv 3 poisson'
_PORT = 'port U.TestSym agg A dfreq [1] dsev [1:6] agg B dfreq [1] dsev [1:6]'


@pytest.fixture(autouse=True)
def _no_page_after():
    """Every test leaves no page set and no figure open."""
    yield
    fit_page()
    plt.close('all')


@pytest.fixture(scope='module')
def agg():
    return build(_AGG)


@pytest.fixture(scope='module')
def port():
    return build(_PORT)


def _size(fig):
    return tuple(fig.get_size_inches())


def test_no_page_returns_the_size_unchanged():
    size = (7.0, 2.45)
    assert _fit_to_page(size) is size


@pytest.mark.parametrize('size, page, fitted', [
    # The plan's house page: 6.3in wide, capped at 8in.
    ((3.5, 2.45), (6.3, 8.0), (3.5, 2.45)),
    ((7.0, 2.45), (6.3, 8.0), (6.3, 2.205)),
    ((10.5, 2.45), (6.3, 8.0), (6.3, 2.0)),
    # The house slide: 5.3in wide, capped at 2.4in.
    ((3.5, 2.45), (5.3, 2.4), (3.5, 2.4)),
    ((7.0, 2.45), (5.3, 2.4), (5.3, 2.0)),
    # A figure already lower than the 2in floor keeps its own height.
    ((10.0, 1.5), (5.0, None), (5.0, 1.5)),
    # A cap alone only caps.
    ((7.0, 4.9), (None, 3.0), (7.0, 3.0)),
    # The cap wins over the floor.
    ((10.5, 2.45), (6.3, 1.8), (6.3, 1.8)),
])
def test_rule(size, page, fitted):
    fit_page(*page)
    assert _fit_to_page(size) == pytest.approx(fitted)


@pytest.mark.parametrize('bad', [0, -1.0, float('nan')])
def test_page_must_be_positive(bad):
    with pytest.raises(ValueError, match='positive'):
        fit_page(bad)
    with pytest.raises(ValueError, match='positive'):
        fit_page(6.3, bad)


def test_house_page_sizes(agg):
    fit_page(6.3, 8.0)
    assert _size(agg.plot()) == pytest.approx((6.3, 2.205))
    fig, _ = make_grid(1, 3)
    assert _size(fig) == pytest.approx((6.3, 2.0))
    fig, _ = make_grid(1, 1)
    assert _size(fig) == pytest.approx((3.5, 2.45))


def test_mosaic_given_size_is_fitted():
    fit_page(6.3, 8.0)
    fig, _ = make_mosaic('AB\nCD', figsize=(12, 9))
    assert _size(fig) == pytest.approx((6.3, 9 * 6.3 / 12))


def test_slide_page_bounds_every_figure(agg, port):
    fit_page(5.3, 2.4)
    figures = [agg.plot(), port.plot(), agg.sevs[0].plot(), make_grid(1, 3)[0],
               make_mosaic('ABC')[0]]
    for fig in figures:
        w, h = _size(fig)
        assert w <= 5.3 + 1e-9 and h <= 2.4 + 1e-9


def test_clearing_the_page_restores_sizes(agg):
    fit_page(5.3, 2.4)
    fit_page()
    assert _size(agg.plot()) == pytest.approx((7.0, 2.45))


def test_scatter_honors_figsize(port):
    axs = port.scatter(figsize=(4, 4))
    assert _size(axs.flat[0].figure) == pytest.approx((4.0, 4.0))
