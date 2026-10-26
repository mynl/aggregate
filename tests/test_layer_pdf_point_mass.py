"""The layered pdf marks the point mass at the limit (``[Deductible-Forms-Follow-Ups]``).

A layer ``y xs a`` pays ``min(max(X - a, 0), y)``, so its payment lives on
``[0, y]`` with a point mass ``S(a + y)`` at ``y``. The pdf marks a point mass
with ``inf``, as :func:`aggregate._severity.make_layer_retention_pdf` does.
``make_layer_attachment_pdf`` tested ``x >= limit`` before ``x == limit``, so
the conditional form never returned the marker, and the unconditional form put
it at the detachment ``a + y`` (a ground-up loss) with a nonzero density
between the limit and the detachment.
"""
from __future__ import annotations

import numpy as np
import pytest

from aggregate import Severity, build

LOGNORM = dict(sev_name='lognorm', sev_mean=100, sev_cv=2)


@pytest.mark.parametrize('conditional', [True, False],
                         ids=['conditional', 'unconditional'])
def test_point_mass_is_at_the_limit(conditional):
    sev = Severity(exp_attachment=50, exp_limit=1000, sev_conditional=conditional,
                   **LOGNORM)
    assert sev.fz.sf(1050) > 0
    assert np.isinf(sev.pdf(1000.0))
    # nothing between the limit and the detachment, nor beyond it
    np.testing.assert_array_equal(sev.pdf(np.array([1000.5, 1020, 1050, 1e4])), 0)
    # the body inside the layer is the shifted ground-up density
    x = np.array([10.0, 500.0, 999.0])
    scale = sev.fz.sf(50) if conditional else 1.0
    np.testing.assert_allclose(sev.pdf(x), sev.fz.pdf(x + 50) / scale, rtol=1e-12)


def test_unconditional_keeps_the_mass_at_zero():
    sev = Severity(exp_attachment=50, exp_limit=1000, sev_conditional=False,
                   **LOGNORM)
    assert np.isinf(sev.pdf(0.0))
    assert sev.pdf(-1.0) == 0


def test_no_marker_when_the_layer_cannot_be_exhausted():
    """``pdetach = 0``: a uniform on ``[0, 100]`` never reaches ``1000``."""
    sev = Severity(sev_name='uniform', sev_scale=100, exp_attachment=0,
                   exp_limit=1000)
    assert sev.fz.sf(1000) == 0
    assert sev.pdf(1000.0) == 0


def test_ordinary_marker_matches_the_retention_wrapper():
    """An ordinary layer and a franchise at the same limit mark the same mass."""
    ordinary = Severity(exp_attachment=50, exp_limit=1000, **LOGNORM)
    franchise = Severity(exp_attachment=50, exp_limit=1000,
                         exp_retention_end=50, **LOGNORM)
    assert np.isinf(ordinary.pdf(1000.0)) and np.isinf(franchise.pdf(1000.0))


@pytest.mark.parametrize('layers', ['5 xs 1', '5 xs 1 franchise'],
                         ids=['ordinary', 'franchise'])
def test_error_analysis_is_finite_when_a_grid_point_is_the_limit(layers):
    """With ``bs = 2`` the grid ``xs - bs / 2`` has a point at the limit 5.

    The quadratic term reads the point mass marker as 0, so one ``inf`` no
    longer turns the gradient, and the whole term, into ``nan``.
    """
    a = build(f'agg T 10 claims {layers} sev lognorm 2 cv 1 poisson',
              update=False)
    a.update(bs=2, log2=10)
    xs = np.hstack((a.xs - a.bs / 2, a.xs[-1] + a.bs / 2))
    assert np.isinf(a.sevs[0].pdf(xs)).any()
    assert np.isfinite(a.severity_error_analysis()['h2_adj']).all()


def test_error_analysis_quadratic_term_reads_the_density_part():
    """``h2_adj`` is about ``bs^2 / 24 (f(y-) - f(0+))`` on the conditional layer."""
    a = build('agg T 10 claims 1000 xs 50 sev lognorm 100 cv 2 poisson',
              update=False)
    a.update(bs=1 / 4, log2=16)
    sev = a.sevs[0]
    expected = a.bs ** 2 / 24 * (sev.pdf(1000 - 1e-9) - sev.pdf(1e-9))
    got = a.severity_error_analysis().loc[0, 'h2_adj']
    assert got == pytest.approx(expected, rel=0.02)
