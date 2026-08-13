"""Tests for the ``picks`` severity adjustment [Picks-Robustness].

Feasible picks are hit exactly on the grid; the debug audit path tolerates
``quad``'s absolute error estimate growing with the integration range (it
checks relative error, 1.0.0a272); infeasible picks warn instead of failing
silently. Layer conventions: the attachment list holds the layer tops and
the bottom layer runs from zero to the first entry.
"""

import logging

import numpy as np
import pytest

from aggregate import build

TOPS = [125, 250, 500, 1000, 2000, 5000, 10000]
PICKS = [100, 74, 96, 97, 75, 51, 16]


def layer_losses(a, tops):
    """Layer losses from the discrete severity, the picks convention:
    left sums of the grid survival function times the bucket size."""
    p = np.asarray(a.sev_density)
    fill = max(0., 1. - p.sum())
    S = np.append(p[1:], fill)[::-1].cumsum()[::-1]
    xs = np.asarray(a.xs)
    bs = xs[1] - xs[0]
    levs = np.array([S[xs < t].sum() * bs for t in tops])
    return np.diff(np.append(0., levs))


@pytest.fixture(scope='module')
def gross():
    return build('agg PicksGross dfreq [1] sev lognorm 500 cv 2',
                 bs=1 / 2, log2=16)


def test_picks_hit_targets():
    a = build('agg PicksHit dfreq [1] sev lognorm 500 cv 2 '
              f'picks {TOPS} {PICKS}', bs=1 / 2, log2=16)
    achieved = layer_losses(a, TOPS)
    assert np.allclose(achieved, PICKS, atol=1e-8)
    assert np.asarray(a.sev_density).min() >= 0
    assert np.isclose(np.asarray(a.sev_density).sum(), 1.0)


def test_picks_debug_audit_relative_quad_error(gross):
    # before a272 this raised AssertionError: quad's absolute error estimate
    # integrating the lognormal sf to 10,000 exceeds a fixed 1e-6 even
    # though the relative error is tiny
    p = gross.picks(TOPS, PICKS, debug=True)
    assert p.exact is not None
    # rows 1..7 are the tower; row 8 is above the tower, then the sum row
    achieved = p.audit['adj'].iloc[:len(TOPS)].values
    assert np.allclose(achieved, PICKS, atol=1e-6)


def test_picks_infeasible_warns(caplog):
    # layer 6 pick of 5 sits below the full-limit losses implied from above
    # (about 17 here), forcing a negative adjustment weight
    bad = [100, 74, 96, 97, 75, 5, 16]
    with caplog.at_level(logging.WARNING, logger='aggregate'):
        a = build('agg PicksBad dfreq [1] sev lognorm 500 cv 2 '
                  f'picks {TOPS} {bad}', bs=1 / 2, log2=16)
    messages = ' '.join(r.message for r in caplog.records)
    assert 'Infeasible picks' in messages
    assert 'negative probabilities' in messages
    # the density is still returned, negative mass and all
    assert np.asarray(a.sev_density).min() < 0
