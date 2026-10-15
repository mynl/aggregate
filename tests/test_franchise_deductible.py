"""Franchise deductible and the retention family it belongs to (#9).

``[Franchise-Deductible]``. A layer ``y xs d1`` has ground-up top ``top = d1 +
y``. The retention is ``d1`` up to ``d1``, ``d1 (d2 - x) / (d2 - d1)`` on
``(d1, d2]`` and zero above ``d2``; ``d2 = inf`` is the ordinary deductible and
``d2 = d1`` the franchise, which pays ``min(X, top)`` once ``X > d1``. The
transform is built with ``d2`` as its parameter, so the disappearing deductible
(``d1 < d2``) is tested here at the engine level although DecL exposes only
``franchise``.

Everything is checked against closed forms or direct integration, never
against a captured snapshot.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from aggregate import Severity, build
from aggregate import _severity as S
from aggregate.decl_writer import spec_to_decl

LOGNORM = dict(sev_name='lognorm', sev_mean=100, sev_cv=2)


def _brute_moments(sev, conditional):
    """``E[g(X)^m]`` by direct integration of the payment function."""
    fz, d1, top = sev.fz, sev.attachment, sev.detachment

    def g(x):
        return float(S._retention_payment(x, d1, sev.retention_end, top))

    out = []
    for m in (1, 2, 3):
        body = quad(lambda x: g(x) ** m * fz.pdf(x), d1, top, limit=500)[0]
        val = body + g(top) ** m * fz.sf(top)
        out.append(val / (fz.sf(d1) if conditional else 1.0))
    return out


# ---------------------------------------------------------------------------
# Engine: moments on every path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('conditional', [True, False])
@pytest.mark.parametrize('kw', [
    LOGNORM,                                                # closed form
    dict(sev_name='gamma', sev_mean=100, sev_cv=0.8),       # closed form
    dict(sev_name='weibull_min', sev_a=0.8, sev_scale=80),  # numerical
], ids=['lognorm', 'gamma', 'weibull'])
@pytest.mark.parametrize('d1, d2, y', [(50, 50, 1000), (25, 50, 1000)],
                         ids=['franchise', 'disappearing'])
def test_moments_match_direct_integration(kw, conditional, d1, d2, y):
    """The partial expectation form agrees with integrating ``g(X)^m``."""
    sev = Severity(exp_attachment=d1, exp_limit=y, exp_retention_end=d2,
                   sev_conditional=conditional, **kw)
    np.testing.assert_allclose(sev.moms(), _brute_moments(sev, conditional),
                               rtol=1e-9)


@pytest.mark.parametrize('conditional', [True, False])
def test_franchise_mean_is_ordinary_mean_plus_attachment(conditional):
    """Each claim above ``d1`` recovers the ``d1`` it would otherwise keep."""
    f = Severity(exp_attachment=50, exp_limit=1000, exp_retention_end=50,
                 sev_conditional=conditional, **LOGNORM)
    o = Severity(exp_attachment=50, exp_limit=1000,
                 sev_conditional=conditional, **LOGNORM)
    recovered = 50 * (1.0 if conditional else o.fz.sf(50))
    assert f.moms()[0] == pytest.approx(o.moms()[0] + recovered, rel=1e-12)


def test_discrete_franchise_is_exact():
    """Atoms above the attachment pay in full, capped at the top ``3 + 10``."""
    sev = Severity('dhistogram', exp_attachment=3, exp_limit=10,
                   exp_retention_end=3, sev_xs=[1, 3, 4, 8, 20],
                   sev_ps=[.2] * 5)
    pays = np.array([4.0, 8.0, 13.0])
    expected = [float(np.mean(pays ** m)) for m in (1, 2, 3)]
    np.testing.assert_allclose(sev.moms(), expected, rtol=1e-14)


# ---------------------------------------------------------------------------
# Engine: distribution functions
# ---------------------------------------------------------------------------

def test_franchise_gap_carries_no_probability():
    """No payment lies strictly between zero and the attachment."""
    cond = Severity(exp_attachment=50, exp_limit=1000, exp_retention_end=50,
                    **LOGNORM)
    unc = Severity(exp_attachment=50, exp_limit=1000, exp_retention_end=50,
                   sev_conditional=False, **LOGNORM)
    gap = np.array([0.0, 1.0, 25.0, 49.99, 50.0])
    np.testing.assert_allclose(cond.cdf(gap), 0.0, atol=1e-15)
    np.testing.assert_allclose(unc.cdf(gap), unc.fz.cdf(50), rtol=1e-14)
    assert float(cond.cdf(1050.0)) == 1.0


@pytest.mark.parametrize('d1, d2', [(50, 50), (25, 50)],
                         ids=['franchise', 'disappearing'])
@pytest.mark.parametrize('conditional', [True, False])
def test_cdf_sf_and_quantiles_are_consistent(d1, d2, conditional):
    """``cdf + sf = 1``, ``ppf`` is monotone and agrees with ``isf``."""
    sev = Severity(exp_attachment=d1, exp_limit=1000, exp_retention_end=d2,
                   sev_conditional=conditional, **LOGNORM)
    t = np.array([0.0, 10, 30, 60, 200, 800, 1049])
    np.testing.assert_allclose(sev.cdf(t) + sev.sf(t), 1.0, atol=1e-14)
    q = np.linspace(0.001, 0.999, 400)
    assert np.all(np.diff(sev.ppf(q)) >= 0)
    np.testing.assert_allclose(sev.ppf(q), sev.isf(1 - q), rtol=1e-9)


def test_cdf_matches_ground_up_formula():
    """``(F(d1 + t/k) - F(d1)) / S(d1)`` on the ramp, ``(F(t) - F(d1)) / S(d1)`` above."""
    d1, d2 = 25.0, 50.0
    sev = Severity(exp_attachment=d1, exp_limit=1000, exp_retention_end=d2,
                   **LOGNORM)
    fz, k = sev.fz, d2 / (d2 - d1)
    for t in (10.0, 40.0):
        assert float(sev.cdf(t)) == pytest.approx(
            (fz.cdf(d1 + t / k) - fz.cdf(d1)) / fz.sf(d1), rel=1e-12)
    for t in (60.0, 500.0):
        assert float(sev.cdf(t)) == pytest.approx(
            (fz.cdf(t) - fz.cdf(d1)) / fz.sf(d1), rel=1e-12)


def test_retention_end_must_lie_in_the_layer():
    with pytest.raises(ValueError, match='between the attachment'):
        Severity(exp_attachment=50, exp_limit=100, exp_retention_end=500,
                 **LOGNORM)
    with pytest.raises(ValueError, match='needs a layer clause'):
        Severity(exp_retention_end=50, **LOGNORM)


# ---------------------------------------------------------------------------
# DecL
# ---------------------------------------------------------------------------

def _parse(program):
    return build.parser.parse(build.lexer.tokenize(program))


@pytest.mark.parametrize('program', [
    'agg F 100 claims 1000 xs 50 franchise sev lognorm 100 cv 2 poisson',
    'agg F 100 claims [500 1000] xs [25 50] franchise sev lognorm 100 cv 2 poisson',
    'agg F 100 claims 1000 xs 50 franchise as "Fr" sev lognorm 100 cv 2 poisson',
], ids=['scalar', 'vector', 'label'])
def test_franchise_parses_and_round_trips(program):
    """``exp_retention_end`` copies the attachment and the unparser restores it."""
    kind, name, spec = _parse(program)
    np.testing.assert_array_equal(spec['exp_retention_end'],
                                  spec['exp_attachment'])
    kind2, name2, spec2 = _parse(spec_to_decl(spec, kind=kind, name=name))
    for key in ('exp_attachment', 'exp_limit', 'exp_retention_end',
                'label_map'):
        assert str(spec2.get(key)) == str(spec.get(key)), key


def test_ordinary_layer_has_no_retention_end():
    _, _, spec = _parse('agg O 100 claims 1000 xs 50 sev lognorm 100 cv 2 poisson')
    assert spec.get('exp_retention_end') is None


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def test_aggregate_franchise_adds_attachment_per_claim():
    """Agg mean moves by exactly ``d1`` times the layer claim count."""
    f = build('agg GS.F 100 claims 1000 xs 50 franchise sev lognorm 100 cv 2 poisson')
    o = build('agg GS.O 100 claims 1000 xs 50 sev lognorm 100 cv 2 poisson')
    assert f.est_m - o.est_m == pytest.approx(50 * f.n, rel=1e-9)
    assert f.valid.passes


def test_aggregate_vector_franchise():
    a = build('agg GS.FV 100 claims [500 1000] xs [25 50] franchise '
              'sev lognorm 100 cv 2 poisson')
    assert [s.retention_end for s in a.sevs] == [25.0, 50.0]
    assert a.valid.passes


def test_aggregate_discrete_franchise_lands_on_the_atoms():
    a = build('agg GS.FD dfreq [1] 10 xs 3 franchise dsev [1 3 4 8 20]')
    support = a.xs[a.agg_density > 1e-12]
    np.testing.assert_allclose(support, [4.0, 8.0, 13.0])
    assert a.est_m == pytest.approx(25 / 3, rel=1e-12)


# ---------------------------------------------------------------------------
# The largest payment is the ground-up top, not the limit
# ---------------------------------------------------------------------------

def test_payment_max_is_the_ground_up_top():
    """``1000 xs 50 franchise`` pays up to 1050; an ordinary layer up to 1000."""
    f = Severity(exp_attachment=50, exp_limit=1000, exp_retention_end=50,
                 **LOGNORM)
    o = Severity(exp_attachment=50, exp_limit=1000, **LOGNORM)
    assert f.payment_max == 1050
    assert o.payment_max == 1000


def test_narrow_high_franchise_layer_fits_the_grid():
    """A grid sized from the limit alone would drop most of the mass here.

    ``100 xs 1000 franchise`` pays between 1000 and 1100, eleven times the
    limit; before the bounded window read ``payment_max`` it lost 46% of the
    probability off the top of the grid and reported an aggregate mean of 3009
    against 10904.
    """
    a = build('agg GS.Narrow 10 claims 100 xs 1000 franchise '
              'sev lognorm 500 cv 1 poisson')
    theory = float(a.stats_df.loc[('agg', 'mean'), 'mixed'])
    assert a.est_m == pytest.approx(theory, rel=1e-6)
    assert a.valid.passes


def test_tail_report_reads_the_ground_up_top():
    from aggregate import tail as _tail
    sev = Severity(exp_attachment=1000, exp_limit=100, exp_retention_end=1000,
                   **LOGNORM)
    assert _tail.severity_support(sev)[1] == 1100
