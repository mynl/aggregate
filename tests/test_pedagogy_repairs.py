"""Tests for the three ``pedagogy.py`` repairs (``dev/done/plan-pre-release-housekeeping.md``).

Covers:

- ``similar_risks_example`` runs on today's ``Bounds.plot_envelope`` and
  ``Portfolio.plot`` and returns its frame.
- ``plot_similar_risks_graphs`` with ``p_reg < 1`` prices the new portfolio
  with assets limited at ``q(p_reg)``; ``p_reg = 1`` is unchanged.
- ``ClassicalPremium.calibrate`` runs, with the Fischer power set by the
  keyword ``fischer_power``.

The DecL programs are mirrored in ``src/aggregate/agg/decl-testers.agg``
(section PED).
"""

from __future__ import annotations

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from aggregate import Bounds, build
from aggregate import pedagogy as pg

UNIF_PROGRAM = 'port PED.Unif agg ONE 1 claim sev 1 * beta 1 1 fixed'
BETA_PROGRAM = ('port PED.Beta agg TWO 1 claim sev 1 * beta [200 300 400 500 600 7] '
                '[600 500 400 300 200 1] wts=6 fixed')
CLASSICAL_PROGRAM = ('port PED.Classical agg Limit1 0.1 claims 1000000 xs 0 '
                     'sev lognorm 50000 cv 10 poisson')

# t1 range of the BETA portfolio over the UNIF brackets, unbounded
T1_MIN, T1_MAX = 0.635566, 0.650011


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


@pytest.fixture(scope='module')
def similar_risks():
    """The base and new portfolios, calibrated as in ``similar_risks_example``."""
    p_base = build(UNIF_PROGRAM, update=False)
    p_base.update(11, 1 / 1024, remove_fuzz=True)
    prem = p_base.tvar(0.2)
    a = 1
    d = (prem - p_base.ex) / (a - p_base.ex)
    roe = d / (1 - d)
    p_base.calibrate_distortions(roe, a=a)
    bounds = Bounds(p_base, premium=prem, a=a)
    p_new = build(BETA_PROGRAM, update=False)
    p_new.update(11, 1 / 1024, remove_fuzz=True)
    return dict(bounds=bounds, port=p_base, pnew=p_new, roe=roe, prem=prem)


@pytest.fixture(scope='module')
def classical():
    return build(CLASSICAL_PROGRAM)


def test_similar_risks_example_runs():
    df = pg.similar_risks_example()
    assert df.shape == (11913, 7)
    assert df.t1.min() == pytest.approx(T1_MIN, abs=1e-6)
    assert df.t1.max() == pytest.approx(T1_MAX, abs=1e-6)
    assert len(plt.get_fignums()) == 3


def test_similar_risks_graphs_unbounded_unchanged(similar_risks):
    df = pg.plot_similar_risks_graphs(None, p_reg=1, **similar_risks)
    assert df.t1.min() == pytest.approx(T1_MIN, abs=1e-6)
    assert df.t1.max() == pytest.approx(T1_MAX, abs=1e-6)


def test_similar_risks_graphs_regulatory_bound(similar_risks):
    """Limiting assets at q(0.9) lowers every bracket price of the new risk."""
    pnew = similar_risks['pnew']
    assert pnew.q(0.9) == pytest.approx(0.876953125)
    df = pg.plot_similar_risks_graphs(None, p_reg=0.9, **similar_risks)
    assert df.t1.max() == pytest.approx(0.628715, abs=1e-6)
    assert df.t1.max() < T1_MAX


def test_bounded_tvar_matches_direct_sum(similar_risks):
    """``Bounds._tvar_x_a`` equals TVaR of min(X, a) summed over the pmf."""
    pnew, prem = similar_risks['pnew'], similar_risks['prem']
    a = pnew.q(0.9)
    got = float(Bounds(pnew, premium=prem, a=a)._tvar_x_a(0.5))
    y = np.minimum(pnew.density_df.loss.to_numpy(), a)
    q = pnew.density_df.p_total.to_numpy()
    F = np.cumsum(q)
    p = 0.5
    i = np.searchsorted(F, p, side='right')
    direct = ((F[i] - p) * y[i] + np.sum(q[i + 1:] * y[i + 1:])) / (1 - p)
    assert got == pytest.approx(0.73640109, abs=1e-8)
    assert got == pytest.approx(direct, rel=1e-12)


def test_classical_premium_calibrates_every_method(classical):
    """Each of the nine methods reproduces the target premium. The tolerance
    is relative because ``newton``'s is absolute and the Exponential and
    Esscher parameters are of order 1e-6."""
    cp = pg.ClassicalPremium({'gross': classical}, 8000)
    params = cp.calibrate('gross', 'Limit1', 8000)
    assert set(params) == set(pg.ClassicalPremium.__classical_methods__)
    assert len(params) == 9
    for method, price in cp.prices('gross', 'Limit1', params).items():
        assert float(price) == pytest.approx(8000, rel=1e-3), method


def test_fischer_power_one_is_dutch(classical):
    cp = pg.ClassicalPremium({'gross': classical}, 8000, fischer_power=1)
    fischer = cp.price(0.5, 'gross', 'Limit1', 'Fischer')
    dutch = cp.price(0.5, 'gross', 'Limit1', 'Dutch')
    assert float(fischer) == pytest.approx(float(dutch), rel=1e-12)
    assert pg.ClassicalPremium({}, 1).fischer_power == 2
