"""Discrete severity: honest ``_DiscreteRV`` and exact moments (1.0.0a26).

``SeverityDHistogram`` / ``SeverityFixed`` now back ``self.fz`` with
``_DiscreteRV`` -- an honest frozen discrete RV (exact right-continuous step
``cdf``/``sf``, ``pdf = 0``, exact ``ppf``/``isf``/``support``) -- instead of
the old ``rv_histogram`` epsilon-jump hack. Every discrete moment (unlimited,
limited, or layered) is now an exact finite sum over the atoms via
``_DiscreteRV.layer_moments``, replacing the numerical isf-integration that
returned trailing-9s artifacts (e.g. ``3.4999999995`` for a mean of ``3.5``).

The DecL programs are mirrored in ``src/aggregate/agg/test_decl.agg`` under the
DISC section.
"""

import logging

import numpy as np
import pytest

from aggregate import build
from aggregate.distributions import Severity, _DiscreteRV

logging.disable(logging.CRITICAL)

X = np.array([1.0, 2.0, 10.0])
P = np.array([0.5, 0.25, 0.25])


def _dsev(**kw):
    return Severity(sev_name="dhistogram", sev_xs=X, sev_ps=P, **kw)


def _exact(xk, pk, n):
    return float(np.sum(np.asarray(xk, float) ** n * np.asarray(pk, float)))


# ----------------------------------------------------------------------
# _DiscreteRV unit behaviour
# ----------------------------------------------------------------------
def test_discreterv_step_cdf_right_continuous():
    rv = _DiscreteRV(X, P)
    # right-continuous: the atom is included at its own value, excluded just below
    assert rv.cdf(1.0) == pytest.approx(0.5)
    assert rv.cdf(0.999) == pytest.approx(0.0)
    assert rv.cdf(2.0) == pytest.approx(0.75)
    assert rv.cdf(9.999) == pytest.approx(0.75)
    assert rv.cdf(10.0) == pytest.approx(1.0)
    # sf is the exact complement
    assert rv.sf(2.0) == pytest.approx(0.25)


def test_discreterv_pdf_is_zero():
    rv = _DiscreteRV(X, P)
    assert rv.pdf(2.0) == 0.0
    assert np.all(rv.pdf(np.array([1.0, 5.0, 10.0])) == 0.0)


def test_discreterv_ppf_isf_return_exact_atoms():
    rv = _DiscreteRV(X, P)
    # smallest atom with cdf >= q
    assert rv.ppf(0.01) == 1.0
    assert rv.ppf(0.5) == 1.0          # cdf(1)=0.5 >= 0.5
    assert rv.ppf(0.5001) == 2.0
    assert rv.ppf(0.76) == 10.0
    # isf(0.25) = ppf(0.75); cdf(2)=0.75 >= 0.75 so the atom is 2.0
    assert rv.isf(0.25) == 2.0
    assert rv.isf(0.24) == 10.0        # ppf(0.76)
    # no eps artifact: exact, not 0.99999...
    assert rv.ppf(0.99) == 10.0


def test_discreterv_support_and_arrays():
    rv = _DiscreteRV(X, P)
    assert rv.support() == (1.0, 10.0)
    # array in -> array out, scalar in -> numpy scalar
    out = rv.cdf(np.array([0.0, 1.0, 10.0]))
    assert isinstance(out, np.ndarray) and out.tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert np.ndim(rv.cdf(1.0)) == 0


def test_discreterv_handles_negative_and_unsorted():
    rv = _DiscreteRV([5.0, -2.0], [0.5, 0.5])
    assert rv.support() == (-2.0, 5.0)
    assert rv.cdf(-3.0) == 0.0
    assert rv.cdf(-2.0) == pytest.approx(0.5)
    assert rv.cdf(5.0) == pytest.approx(1.0)


def test_layer_moments_zero_denom_returns_zeros():
    rv = _DiscreteRV(X, P)
    # layer entirely above support -> P(X>a)=0 -> exact 0, not 0/0=nan
    assert rv.layer_moments(100.0, 50.0, 0.0) == (0.0, 0.0, 0.0)


# ----------------------------------------------------------------------
# Exact moments through Severity.moms() -- the headline fix
# ----------------------------------------------------------------------
def test_unlimited_discrete_moments_exact():
    """Previously fell to numerical integration and returned 3.4999999995."""
    s = _dsev()
    assert isinstance(s.fz, _DiscreteRV)
    m1, m2, m3 = s.moms()
    assert (m1, m2, m3) == (_exact(X, P, 1), _exact(X, P, 2), _exact(X, P, 3))
    assert m1 == 3.5  # bit-exact, no trailing-9s


def test_layered_discrete_moments_exact():
    s = _dsev(exp_attachment=0, exp_limit=5)
    y = np.clip(X - 0.0, 0.0, 5.0)
    assert s.moms() == pytest.approx(
        (float(np.sum(y * P)), float(np.sum(y**2 * P)), float(np.sum(y**3 * P))),
        abs=0, rel=0)


def test_conditional_layer_with_attachment_exact():
    # 8 xs 2 conditional: y = clip(X-2, 0, 8) = [0,0,8]; pattach = P(X>2) = 0.25
    s = _dsev(exp_attachment=2, exp_limit=8)
    pa = 0.25
    y = np.clip(X - 2.0, 0.0, 8.0)
    assert s.moms() == pytest.approx(
        (float(np.sum(y * P)) / pa,
         float(np.sum(y**2 * P)) / pa,
         float(np.sum(y**3 * P)) / pa))


def test_fixed_severity_exact():
    s = Severity(sev_name="fixed", sev_xs=np.array([50.0]))
    assert isinstance(s.fz, _DiscreteRV)
    assert s.moms() == (50.0, 2500.0, 125000.0)
    assert float(s.fz.ppf(0.01)) == 50.0
    assert s.fz.support() == (50.0, 50.0)


def test_signed_discrete_moments_exact():
    xs = np.array([-2.0, 5.0])
    ps = np.array([0.5, 0.5])
    s = Severity(sev_name="dhistogram", sev_xs=xs, sev_ps=ps)
    assert s.signed is True
    assert s.moms() == (_exact(xs, ps, 1), _exact(xs, ps, 2), _exact(xs, ps, 3))


# ----------------------------------------------------------------------
# End-to-end: an aggregate built on a discrete severity validates clean
# ----------------------------------------------------------------------
def test_dice_aggregate_exact_severity_moments():
    a = build("agg DISC.Dice dfreq [1] dsev [1:6]")
    # the per-claim severity mean is exactly 3.5 (a fair die)
    assert a.sevs[0].moms()[0] == 3.5
    # one claim => aggregate mean equals severity mean, exact
    assert a.agg_m == pytest.approx(3.5, abs=1e-9)


def test_layered_dsev_aggregate_builds():
    a = build("agg DISC.Layer 3 claims 5 xs 3 dsev [1 4 8] [.5 .3 .2] fixed")
    assert a is not None and a.agg_m > 0
