"""Return-period x-axis for the Lee/quantile panel (``quantile_x='return'``).

Smoke tests across the classes that own a Lee panel -- ``Aggregate``,
``Severity`` and the occurrence-reinsurance plot -- plus a unit test of the
transform itself. Each checks that ``quantile_x='return'`` runs, makes the Lee
axis log, and labels it "Return period", with no regression to the linear
default.
"""

import matplotlib
matplotlib.use('Agg')  # headless

import numpy as np
import pytest

from aggregate import build
from aggregate.plots._quantile import plot_quantile, MAX_RETURN_PERIOD


# ----------------------------------------------------------------------
# Layer-1 worker: the transform itself
# ----------------------------------------------------------------------
def test_quantile_transform_loss_branch():
    """Loss value: large p -> large T via T = 1/(1-p); p=1 endpoint dropped."""
    fig, ax = matplotlib.pyplot.subplots()
    p = np.array([0.0, 0.5, 0.9, 1.0])
    loss = np.array([0.0, 10.0, 90.0, 100.0])
    (line,) = plot_quantile(ax, p, loss, quantile_x='return', is_loss_value=True)
    xs, ys = line.get_data()
    # the p == 1 atom is dropped (T would be infinite)
    assert len(xs) == 3
    np.testing.assert_allclose(xs, [1.0, 2.0, 10.0])      # 1/(1-p)
    np.testing.assert_allclose(ys, [0.0, 10.0, 90.0])
    assert ax.get_xscale() == 'log'
    assert ax.get_xlabel() == 'Return period'


def test_quantile_transform_payoff_branch():
    """Payoff value: small p -> large T via T = 1/p; p=0 endpoint dropped."""
    fig, ax = matplotlib.pyplot.subplots()
    p = np.array([0.0, 0.1, 0.5, 1.0])
    loss = np.array([-100.0, -50.0, 0.0, 100.0])
    (line,) = plot_quantile(ax, p, loss, quantile_x='return', is_loss_value=False)
    xs, ys = line.get_data()
    assert len(xs) == 3                                   # p == 0 dropped
    np.testing.assert_allclose(xs, [10.0, 2.0, 1.0])      # 1/p
    np.testing.assert_allclose(ys, [-50.0, 0.0, 100.0])
    assert ax.get_xscale() == 'log'


def test_quantile_caps_return_period():
    """Points beyond MAX_RETURN_PERIOD are dropped (T stays finite)."""
    fig, ax = matplotlib.pyplot.subplots()
    # p = 1 - 1e-12 -> T = 1e12, well past the 1e9 cap -> dropped.
    p = np.array([0.5, 1.0 - 1e-6, 1.0 - 1e-12])
    loss = np.array([10.0, 50.0, 9999.0])
    (line,) = plot_quantile(ax, p, loss, quantile_x='return', is_loss_value=True)
    xs, ys = line.get_data()
    assert xs.max() <= MAX_RETURN_PERIOD
    assert 9999.0 not in ys                               # the off-cap point is gone


def test_quantile_return_bounds_y_to_cap():
    """The cap bounds the outcome (y) axis to the deepest plotted point."""
    fig, ax = matplotlib.pyplot.subplots()
    ax.set_ylim(0, 1)                                     # a stale linear y-limit
    p = np.array([0.0, 0.9, 1.0 - 1e-12])
    loss = np.array([0.0, 90.0, 1e9])                     # huge off-cap outcome
    plot_quantile(ax, p, loss, quantile_x='return', is_loss_value=True)
    # y rescaled to the kept data (top ~90), not the stale [0,1] nor the 1e9 atom
    lo, hi = ax.get_ylim()
    assert hi < 1e6


def test_quantile_linear_default_unchanged():
    """The linear default plots x = p, no log scale, no relabel by the worker."""
    fig, ax = matplotlib.pyplot.subplots()
    p = np.array([0.0, 0.5, 1.0])
    loss = np.array([0.0, 10.0, 20.0])
    (line,) = plot_quantile(ax, p, loss)                  # default quantile_x='linear'
    xs, _ = line.get_data()
    np.testing.assert_allclose(xs, p)
    assert ax.get_xscale() == 'linear'


def test_quantile_bad_arg_raises():
    fig, ax = matplotlib.pyplot.subplots()
    with pytest.raises(ValueError, match="quantile_x"):
        plot_quantile(ax, [0.5], [1.0], quantile_x='nope')


# ----------------------------------------------------------------------
# Per-class smoke tests
# ----------------------------------------------------------------------
def _lee_axis():
    """The Lee/quantile panel of the current figure: the Axes titled 'Lee'."""
    fig = matplotlib.pyplot.gcf()
    for ax in fig.axes:
        if 'Lee' in ax.get_title():
            return ax
    raise AssertionError('no Lee panel found')


def test_aggregate_continuous_return():
    a = build('agg RetCont 100 claims sev gamma 100 cv 1 poisson')
    a.plot(quantile_x='return')
    ax = _lee_axis()
    assert ax.get_xscale() == 'log'
    assert ax.get_xlabel() == 'Return period'


def test_aggregate_discrete_return():
    a = build('agg RetDice dfreq [3] dsev [1:6]')
    a.plot(quantile_x='return')
    ax = _lee_axis()
    assert ax.get_xscale() == 'log'


def test_aggregate_linear_default_no_regression():
    a = build('agg RetLin 100 claims sev gamma 100 cv 1 poisson')
    a.plot()
    ax = _lee_axis()
    assert ax.get_xscale() == 'linear'
    assert ax.get_xlabel() == 'Non-exceeding probability p'


def test_severity_return():
    a = build('sev RetSev lognorm 100 cv 2')
    a.plot(quantile_x='return')
    ax = _lee_axis()
    assert ax.get_xscale() == 'log'
    assert ax.get_xlabel() == 'Return period'


def test_reins_occ_return():
    a = build('agg RetReins 100 claims sev gamma 100 cv 1 '
              'occurrence net of 50 xs 50 poisson')
    a.reins_occ_plot(quantile_x='return')
    # the reins plot titles its quantile panel 'Aggregate', not 'Lee'
    ax = [x for x in a.figure.axes if x.get_title() == 'Aggregate'][0]
    assert ax.get_xscale() == 'log'
    assert ax.get_xlabel() == 'Return period'


def test_lee_kwargs_ride_through_compositor():
    """A Lee-worker option neither stub nor compositor names (``max_return_period``)
    still reaches the worker through ``**kwargs`` -- the point of (a)."""
    a = build('agg RetKw 100 claims sev gamma 100 cv 1 poisson')
    a.plot(quantile_x='return', max_return_period=1e3)
    ax = _lee_axis()
    # x capped near the lowered 1e3, not the 1e9 default
    assert ax.get_xlim()[1] < 1e5
