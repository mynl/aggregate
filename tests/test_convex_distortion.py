"""
Direct tests of ``aggregate.spectral.convex_distortion``, checked by hand.

The knots are the vertices of the upper hull, and the weights follow from the
slopes: ``w_i = (k_{i-1} - k_i) s_i`` with ``k_K = 0`` above the last knot.

On ``(0, 0), (1/2, 0.8), (1, 1)`` the slopes are 1.6 and 0.4, so the weight at
``s = 1/2`` is ``(1.6 - 0.4) / 2 = 0.6`` and at ``s = 1`` it is 0.4, and
``g(1/4) = 0.6 * (1/2) + 0.4 * (1/4) = 0.4``.
"""

import numpy as np
import pandas as pd
import pytest

from aggregate.spectral import (bagged_distortion, convex_distortion,
                                convex_example)


TOL = 1e-12


def test_the_toy_has_no_mass_and_interpolates():
    d = convex_distortion([0, .5, 1], [0, .8, 1])
    np.testing.assert_allclose(d.ps, [0, .5], atol=TOL)
    np.testing.assert_allclose(d.wts, [.4, .6], atol=TOL)
    assert not d.has_mass and d.mass == 0
    assert float(d.g(.25)) == pytest.approx(.4, abs=TOL)
    assert float(d.g(.5)) == pytest.approx(.8, abs=TOL)
    assert float(d.g(1e-9)) < 1e-8


def test_a_point_below_the_envelope_is_not_a_knot():
    """The lower side of the hull used to enter the knot set and move the curve."""
    d = convex_distortion([0, .3, .5, 1], [0, .1, .8, 1])
    np.testing.assert_allclose(d.ps, [0, .5], atol=TOL)
    assert float(d.g(.5)) == pytest.approx(.8, abs=TOL)
    assert float(d.g(.3)) == pytest.approx(.48, abs=TOL)


def test_the_corners_are_added():
    d = convex_distortion([.5], [.8])
    assert float(d.g(0)) == 0
    assert float(d.g(.5)) == pytest.approx(.8, abs=TOL)
    assert float(d.g(1)) == pytest.approx(1, abs=TOL)


def test_collinear_points_are_the_identity():
    """Qhull refuses a flat input; the envelope is simply the diagonal."""
    d = convex_distortion([0, .25, .5, 1], [0, .25, .5, 1])
    np.testing.assert_allclose(d.ps, [0], atol=TOL)
    np.testing.assert_allclose(d.wts, [1], atol=TOL)
    s = np.linspace(0, 1, 11)
    np.testing.assert_allclose(d.g(s), s, atol=TOL)


def test_the_envelope_reaches_one_before_s_is_one():
    d = convex_distortion([0, .5, 1], [0, 1, 1])
    assert float(d.g(.25)) == pytest.approx(.5, abs=TOL)
    assert float(d.g(.75)) == pytest.approx(1, abs=TOL)


def test_a_repeated_s_takes_its_largest_value():
    d = convex_distortion([.5, .5], [.2, .8])
    assert float(d.g(.5)) == pytest.approx(.8, abs=TOL)


def test_a_supplied_height_at_zero_is_a_mass():
    """A positive ``g(0+)`` is a real mass at ``p = 1``, the only way one arises."""
    d = convex_distortion([0, .5, 1], [.1, .8, 1])
    assert d.has_mass and d.mass == pytest.approx(.1, abs=TOL)
    # .1 + .7 * s / .5 on the first segment
    assert float(d.g(.25)) == pytest.approx(.45, abs=TOL)
    assert float(d.g(.5)) == pytest.approx(.8, abs=TOL)
    assert float(d.g(1)) == pytest.approx(1, abs=TOL)


@pytest.mark.parametrize('s, gs', [([.5], [1.2]), ([.5], [-.2]),
                                   ([1.5], [.5]), ([-.5], [.5])])
def test_a_point_outside_the_unit_square_is_refused(s, gs):
    with pytest.raises(ValueError, match='unit square'):
        convex_distortion(s, gs)


def test_a_mismatch_in_length_is_refused():
    with pytest.raises(ValueError, match='same length'):
        convex_distortion([0, 1], [0])


@pytest.mark.parametrize('source', ['bond', 'cat'])
def test_the_examples_have_no_mass(source):
    d = convex_example(source)
    assert not d.has_mass
    assert float(d.g(1e-12)) < 1e-9
    assert float(d.g(1)) == pytest.approx(1, abs=TOL)
    assert np.all(np.diff(d.g(np.linspace(0, 1, 1001))) >= -TOL)


def test_a_bagged_distortion_is_a_distortion_with_no_mass():
    data = pd.DataFrame({'EL': [.01, .02, .05, .1, .2, .4],
                         'Spread': [.05, .08, .15, .25, .4, .6]})
    d = bagged_distortion(data, .75, 8)
    assert not d.has_mass
    assert float(d.g(0)) == 0
    assert float(d.g(1)) == pytest.approx(1, abs=TOL)
    assert float(d.g(1e-9)) < 1e-6
    assert np.all(np.diff(d.g(np.linspace(0, 1, 1001))) >= -TOL)
