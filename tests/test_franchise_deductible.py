"""Franchise deductible and the retention family it belongs to (#9).

``[Franchise-Deductible]``. A layer ``y xs d1`` retains ``d1`` up to ``d1``,
``d1 (d2 - x) / (d2 - d1)`` on ``(d1, d2]`` and zero above ``d2``, and pays
``min(x - r(x), y)``: deductible first, limit second, so ``y`` is the largest
payment in every form. ``d2 = inf`` is the ordinary deductible and ``d2 = d1``
the franchise, which pays ``min(X, y)`` once ``X > d1``. The
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
    fz, d1, det, y = sev.fz, sev.attachment, sev.detachment, sev.limit

    def g(x):
        return float(S._retention_payment(x, d1, sev.retention_end, y))

    out = []
    for m in (1, 2, 3):
        body = (quad(lambda x: g(x) ** m * fz.pdf(x), d1, det, limit=500)[0]
                if det > d1 else 0.0)
        val = body + y ** m * fz.sf(det)
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
@pytest.mark.parametrize('d1, d2, y', [
    (50, 50, 1000), (50, 50, 30), (25, 50, 1000), (25, 50, 40)],
    ids=['franchise', 'franchise-low-limit', 'disappearing',
         'disappearing-limit-on-ramp'])
def test_moments_match_direct_integration(kw, conditional, d1, d2, y):
    """The partial expectation form agrees with integrating ``g(X)^m``."""
    sev = Severity(exp_attachment=d1, exp_limit=y, exp_retention_end=d2,
                   sev_conditional=conditional, **kw)
    np.testing.assert_allclose(sev.moms(), _brute_moments(sev, conditional),
                               rtol=1e-9)


def test_franchise_is_attachment_plus_ordinary_layer():
    """For ``y > d1``, ``y xs d1 franchise`` pays ``d1`` plus ``(y - d1) xs d1``.

    Conditional on ``X > d1`` the franchise payment ``min(X, y)`` is
    ``d1 + min(X - d1, y - d1)``, so every moment follows from the ordinary
    layer shifted by ``d1``.
    """
    d1, y = 50.0, 1000.0
    f = Severity(exp_attachment=d1, exp_limit=y, exp_retention_end=d1,
                 **LOGNORM)
    o = Severity(exp_attachment=d1, exp_limit=y - d1, **LOGNORM)
    o1, o2, o3 = o.moms()
    shifted = (d1 + o1, d1 ** 2 + 2 * d1 * o1 + o2,
               d1 ** 3 + 3 * d1 ** 2 * o1 + 3 * d1 * o2 + o3)
    np.testing.assert_allclose(f.moms(), shifted, rtol=1e-12)


@pytest.mark.parametrize('conditional', [True, False])
def test_franchise_with_limit_below_attachment_pays_the_limit(conditional):
    """For ``y <= d1`` every loss above ``d1`` pays exactly ``y``."""
    f = Severity(exp_attachment=50, exp_limit=30, exp_retention_end=50,
                 sev_conditional=conditional, **LOGNORM)
    p = 1.0 if conditional else f.fz.sf(50)
    np.testing.assert_allclose(f.moms(), [30 * p, 900 * p, 27000 * p],
                               rtol=1e-12)
    assert f.detachment == 50


@pytest.mark.parametrize('d1, d2, y, expected', [
    (50, np.inf, 1000, 1050),     # ordinary: d1 + y
    (50, 50, 1000, 1000),         # franchise: max(d1, y)
    (50, 50, 30, 50),
    (25, 50, 40, 45),             # disappearing, y <= d2: d1 + y / k, k = 2
    (25, 50, 1000, 1000),         # disappearing, y > d2: y
])
def test_detachment_is_where_the_payment_reaches_the_limit(d1, d2, y, expected):
    sev = Severity(exp_attachment=d1, exp_limit=y, exp_retention_end=d2,
                   **LOGNORM)
    assert sev.detachment == pytest.approx(expected, rel=1e-14)
    assert float(S._retention_payment(sev.detachment + 1e-9, d1, d2, y)
                 if np.isfinite(d2) else y) == pytest.approx(y)


def test_discrete_franchise_is_exact():
    """Atoms above the attachment pay in full, capped at the limit 10."""
    sev = Severity('dhistogram', exp_attachment=3, exp_limit=10,
                   exp_retention_end=3, sev_xs=[1, 3, 4, 8, 20],
                   sev_ps=[.2] * 5)
    pays = np.array([4.0, 8.0, 10.0])
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
    assert float(cond.cdf(1000.0)) == 1.0
    assert float(cond.cdf(999.0)) < 1.0


@pytest.mark.parametrize('d1, d2', [(50, 50), (25, 50)],
                         ids=['franchise', 'disappearing'])
@pytest.mark.parametrize('conditional', [True, False])
def test_cdf_sf_and_quantiles_are_consistent(d1, d2, conditional):
    """``cdf + sf = 1``, ``ppf`` is monotone and agrees with ``isf``."""
    sev = Severity(exp_attachment=d1, exp_limit=1000, exp_retention_end=d2,
                   sev_conditional=conditional, **LOGNORM)
    t = np.array([0.0, 10, 30, 60, 200, 800, 999])
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


def test_retention_end_must_not_precede_the_attachment():
    with pytest.raises(ValueError, match='at least the attachment'):
        Severity(exp_attachment=50, exp_limit=100, exp_retention_end=20,
                 **LOGNORM)
    # beyond the old d1 + y bound is fine: the limit caps the payment
    sev = Severity(exp_attachment=50, exp_limit=100, exp_retention_end=500,
                   **LOGNORM)
    assert sev.detachment == pytest.approx(50 + 100 * 450 / 500)
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

def test_aggregate_franchise_is_attachment_plus_narrower_layer():
    """``1000 xs 50 franchise`` is ``50`` per claim plus ``950 xs 50``."""
    f = build('agg GS.F 100 claims 1000 xs 50 franchise sev lognorm 100 cv 2 poisson')
    o = build('agg GS.O 100 claims 950 xs 50 sev lognorm 100 cv 2 poisson')
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
    np.testing.assert_allclose(support, [4.0, 8.0, 10.0])
    assert a.est_m == pytest.approx(22 / 3, rel=1e-12)


# ---------------------------------------------------------------------------
# The limit is the largest payment in every form
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('d2', [50, 75], ids=['franchise', 'disappearing'])
def test_no_payment_exceeds_the_limit(d2):
    sev = Severity(exp_attachment=50, exp_limit=1000, exp_retention_end=d2,
                   **LOGNORM)
    assert float(sev.ppf(1 - 1e-12)) == 1000
    assert float(sev.sf(1000.0)) == 0.0
    assert float(sev.sf(999.0)) == pytest.approx(
        sev.fz.sf(999.0) / sev.fz.sf(50), rel=1e-12)


def test_high_franchise_with_narrow_limit_is_a_point_mass():
    """``100 xs 1000 franchise`` pays exactly 100 on every claim above 1000."""
    a = build('agg GS.Narrow 10 claims 100 xs 1000 franchise '
              'sev lognorm 500 cv 1 poisson')
    assert a.sevs[0].moms() == pytest.approx((100.0, 1e4, 1e6), rel=1e-12)
    assert a.est_m == pytest.approx(1000.0, rel=1e-9)
    assert a.valid.passes


def test_tail_report_reads_the_limit():
    from aggregate import tail as _tail
    sev = Severity(exp_attachment=1000, exp_limit=100, exp_retention_end=1000,
                   **LOGNORM)
    assert _tail.severity_support(sev)[1] == 100


# ---------------------------------------------------------------------------
# The qd one-liner names the deductible form
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('layers, phrase', [
    ('1000 xs 50', '1,000 xs 50'),
    ('1000 xs 50 franchise', '1,000 xs 50 franchise'),
    ('1000 xs 50 to 80 disappearing', '1,000 xs 50 to 80 disappearing'),
    ('inf xs 50 franchise', 'inf xs 50 franchise'),
], ids=['ordinary', 'franchise', 'disappearing', 'unlimited-franchise'])
def test_layer_phrase_names_the_deductible_form(layers, phrase, capsys):
    """``[Deductible-Forms-Follow-Ups]``: the phrase shows the form in DecL syntax.

    The phrase feeds ``tail_description`` (the severity in the ``qd``
    one-liner and the ``info`` card) and ``tail_explanation``.
    """
    from aggregate import qd
    a = build(f'agg Q.Phrase 10 claims {layers} sev lognorm 100 cv 2 poisson')
    sev = a.sevs[0]
    assert sev.tail_description.endswith(f'; {phrase}')
    assert f'Declared support: {phrase}.' in sev.tail_explanation
    qd(a)
    # qd wraps at 72 characters, so compare with the line breaks folded
    one_liner = ' '.join(capsys.readouterr().out.split())
    assert f'; {phrase}. Updated with bucket size' in one_liner
