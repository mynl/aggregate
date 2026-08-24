"""Tests for the ``picks`` severity adjustment [Picks-Robustness].

Feasible picks are hit exactly on the grid; the debug audit path tolerates
``quad``'s absolute error estimate growing with the integration range (it
checks relative error, 1.0.0a272); infeasible picks warn instead of failing
silently. Layer conventions: the attachment list holds the layer tops and
the bottom layer runs from zero to the first entry.

Attachments must land on the realized grid and inside the window
(``[Picks-Off-Grid-Error]``, 1.0.0a319). Off grid is refused rather than
snapped, because a layer boundary inside a bucket cannot divide that bucket's
mass between the layer below and the layer above. Before a319 the miss
surfaced as a raw pandas ``KeyError`` from the survival lookup. Every case
below that exercises the guard **pins the bucket**, so a change in the auto
sizer can never silently stop testing what it is here to test.
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


# ---------------------------------------------------------------------------
# The grid check [Picks-Off-Grid-Error]
# ---------------------------------------------------------------------------
#: The motivating program from dev/done/plan-picks-grid-and-reference-trailers.md.
#: At bs=8 the attachments 100 and 500 fall mid bucket (100/8 = 12.5); 200 does
#: not. bs=4 divides all three.
OFF_GRID_TOPS = [100, 200, 500]
OFF_GRID_PICKS = [45, 20, 25]
OFF_GRID_PROGRAM = ('agg PicksOffGrid 1 claim sev lognorm 100 cv 2 '
                    f'picks {OFF_GRID_TOPS} {OFF_GRID_PICKS} fixed')


def test_off_grid_attachment_raises_naming_the_bucket():
    """The offenders, the realized grid and a bucket that would work."""
    with pytest.raises(ValueError) as exc:
        build(OFF_GRID_PROGRAM, bs=8, log2=16)
    message = str(exc.value)
    assert '100' in message
    assert '500' in message
    assert 'bs=8' in message
    assert 'bs=4' in message


def test_the_attachment_on_the_grid_is_not_named():
    """200 is a multiple of 8, so only the two genuine misses are reported."""
    with pytest.raises(ValueError) as exc:
        build(OFF_GRID_PROGRAM, bs=8, log2=16)
    offenders = str(exc.value).split('do not lie on the grid')[0]
    assert '200' not in offenders


def test_the_suggested_bucket_builds_and_hits_the_bottom_pick():
    """bs=4 is what the message offers, so it has to work."""
    a = build(OFF_GRID_PROGRAM, bs=4, log2=18)
    achieved = layer_losses(a, OFF_GRID_TOPS)
    assert np.allclose(achieved, OFF_GRID_PICKS, atol=1e-6)


def test_a_non_dyadic_pinned_grid_still_builds():
    """bs=0.1 divides the attachments without being a power of two.

    The old code read the survival function by exact float label, which works
    on 0.1 only by rounding luck. The positional index does not depend on luck.
    """
    a = build(OFF_GRID_PROGRAM, bs=0.1, log2=16)
    achieved = layer_losses(a, OFF_GRID_TOPS)
    assert np.allclose(achieved, OFF_GRID_PICKS, atol=1e-6)


def test_an_attachment_above_the_window_raises(gross):
    """Beyond the top of the grid is refused, and the top is named."""
    with pytest.raises(ValueError, match='above the top of the window'):
        gross.picks([125, 250, 1e9], [100, 74, 96])


def test_an_infinite_attachment_raises(gross):
    """``inf`` took the same raw KeyError route before a319."""
    with pytest.raises(ValueError, match='above the top of the window'):
        gross.picks([125, 250, np.inf], [100, 74, 96])
