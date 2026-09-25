"""End-to-end tests for the ``cantor`` DecL severity.

Covers :class:`aggregate._severity.SeverityCantor` and everything downstream of
it: the three declaration routes, the arithmetic form, the two bucket families,
a real compound, a layered case, and the two shipped library entries. The
distribution itself is tested in ``tests/test_cantor.py``.

Every program here mirrors a line of ``src/aggregate/agg/decl-testers.agg``
section ``CAN.``, which is where the round-trip and colorization coverage
lives.

The bucket story is the point of most of this. A Cantor severity is the
documented exception to the house rule that ``bs`` should be a binary fraction.
Its level-``m`` cylinders have width ``1/q**m`` with ``q = 2/(1-c)``, so with
``bs = scale / q**m`` each cylinder straddles the bucket pair ``(2j, 2j+1)``
and its mass ``2 ** -m`` lands **exactly half in each**, which is both an exact
reproduction of the level-``m`` law and a mean-preserving one. See
``dev/done/plan-cantor.md`` and its execution notes, divergence D2.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from aggregate import Severity, build
from aggregate._severity import SeverityCantor
from aggregate.cantor import cantor_bs, cantor_pmf

#: Relative tolerance on a discretized moment against its analytic value.
GRID_TOLERANCE = 1e-6


def _severity_pmf(agg):
    """The discretized severity of a single-component aggregate."""
    return np.asarray(agg.discretize('discrete', 'survival', True)[0])


# ---------------------------------------------------------------------------
# Construction and classification
# ---------------------------------------------------------------------------

def test_the_registry_dispatches_cantor_to_its_own_subclass():
    """``Severity('cantor')`` is a ``SeverityCantor``, not a scipy lookup."""
    sev = Severity('cantor')
    assert isinstance(sev, SeverityCantor)
    assert sev.sev_kind == 'cantor'
    assert sev.sev_a == pytest.approx(1 / 3)
    assert sev.sev_scale == 1.0
    assert sev.moms() == pytest.approx((0.5, 0.375, 0.3125), rel=1e-13)


def test_the_three_declaration_routes_resolve_the_shape():
    """Explicit shape, cv, and the bare default, in that precedence."""
    assert build('agg C 1 claim sev cantor 0.5 fixed').sevs[0].sev_a == 0.5
    assert build('agg C 1 claim sev cantor fixed').sevs[0].sev_a == pytest.approx(1 / 3)
    # cv route: cv**2 = (1-a)/(1+a) inverts analytically.
    sev = build('agg C 1 claim sev cantor 10 cv 0.8 fixed').sevs[0]
    a = (1 - 0.8 ** 2) / (1 + 0.8 ** 2)
    assert sev.sev_a == pytest.approx(1 - 2 * a, rel=1e-13)
    assert sev.sev_scale == pytest.approx(20.0, rel=1e-13)
    assert sev.moms()[0] == pytest.approx(10.0, rel=1e-13)


def test_a_declared_mean_sets_the_scale_to_twice_it():
    """The base law has mean 1/2 on ``[0, 1]``, so scale is twice the mean."""
    sev = Severity('cantor', sev_mean=250)
    assert sev.sev_scale == 500.0
    assert sev.moms()[0] == pytest.approx(250.0, rel=1e-13)


def test_the_shape_out_of_range_message_points_at_the_scaling_form():
    """``sev cantor 10`` is the natural mistake; the message says the fix."""
    with pytest.raises(ValueError, match=r'10 \* cantor'):
        Severity('cantor', sev_a=10)
    with pytest.raises(ValueError, match='0 <= c < 1'):
        Severity('cantor', sev_a=-0.5)


def test_the_cv_out_of_range_message_gives_the_attainable_interval():
    """The cv route spans ``[1/sqrt(3), 1)`` and nothing outside it."""
    with pytest.raises(ValueError, match='attainable cv'):
        Severity('cantor', sev_cv=0.4)
    with pytest.raises(ValueError, match='attainable cv'):
        Severity('cantor', sev_cv=1.5)
    # The endpoint itself is attainable: it is the uniform law.
    sev = Severity('cantor', sev_cv=1 / np.sqrt(3))
    assert sev.sev_a == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Moments through the pipeline
# ---------------------------------------------------------------------------

def test_a_binary_bucket_reproduces_the_analytic_moments():
    """``bs`` a binary fraction is right, merely blurry at the finest scales.

    The law is atomless and its distribution function continuous, so cdf
    differencing converges whatever the base. What a binary bucket gives up is
    alignment with the cylinders, not correctness.
    """
    a = build('agg CAN.Binary 1 claim sev cantor fixed hints{bs=1/65536; log2=17}')
    assert a.bs == 2.0 ** -16
    assert a.est_m == pytest.approx(0.5, rel=GRID_TOLERANCE)
    assert a.est_sd ** 2 == pytest.approx(0.125, rel=1e-4)
    assert a.validation_description == 'not unreasonable'


@pytest.mark.parametrize('program, c, m, q, name', [
    ('agg CAN.Ternary 1 claim sev cantor fixed hints{bs=1/6561; log2=14}',
     1 / 3, 8, 3, 'ternary'),
    ('agg CAN.Half 1 claim sev cantor 0.5 fixed hints{bs=1/4096; log2=13}',
     0.5, 6, 4, 'base four'),
])
def test_the_natural_bucket_reproduces_the_exact_level_m_masses(program, c, m, q, name):
    """Each level-m cylinder's mass lands exactly on one bucket pair.

    With ``bs = 1 / q ** m`` the cylinder starting at atom ``xs[j]`` occupies
    bucket indices ``2j`` and ``2j+1``, split exactly in half by the
    half-bucket centering. The check is therefore on the pair, and it holds
    bit for bit, with no mass anywhere else on the grid.
    """
    a = build(program)
    assert a.bs == cantor_bs(m, c)
    xs, ps = cantor_pmf(m, q)
    d = _severity_pmf(a)
    idx = np.round(xs / a.bs).astype(int)
    paired = d[idx] + d[idx + 1]
    assert np.array_equal(paired, ps), name
    assert d[idx] == pytest.approx(d[idx + 1], abs=0.0)
    assert d.sum() == pytest.approx(paired.sum(), abs=0.0)
    assert a.est_m == pytest.approx(0.5, rel=1e-13)


def test_the_scaled_and_shifted_form():
    """``sev 3 * cantor 0.5 + 5`` is the law of ``3X + 5``."""
    a = build('agg CAN.Shifted 1 claim sev 3 * cantor 0.5 + 5 fixed '
              'hints{bs=3/4096; log2=14}')
    sev = a.sevs[0]
    assert sev.sev_a == 0.5
    assert sev.sev_scale == 3.0
    assert sev.sev_loc == 5.0
    kept = (1 - 0.5) / 2
    variance = 9 * (1 - kept) / (4 * (1 + kept))
    m1, m2, _ = sev.moms()
    assert m1 == pytest.approx(6.5, rel=1e-13)
    assert m2 - m1 ** 2 == pytest.approx(variance, rel=1e-12)
    assert a.est_m == pytest.approx(6.5, rel=1e-4)


def test_a_poisson_compound_builds_and_validates():
    """A real frequency smooths the staircase away entirely."""
    a = build('agg CAN.Poisson 10 claims sev cantor poisson')
    assert a.est_m == pytest.approx(5.0, rel=1e-6)
    # Poisson compound: variance is n * E[X**2].
    assert a.est_sd ** 2 == pytest.approx(10 * 0.375, rel=1e-4)
    assert a.validation_description == 'not unreasonable'


def test_a_layered_severity_has_finite_warning_free_moments():
    """A layer clause routes the moments through quadrature on ``ppf``.

    There is no density to integrate, so the closed-form and histogram paths
    are both unavailable and ``_numerical_moms`` integrates the quantile
    function over the layer in probability space. The integrand is monotone
    and bounded, so it converges without complaint.

    The expected value is exact, by hand from the self-similar structure.
    ``F(0.2) = 1/4``, so ``P(X > 0.2) = 3/4``; ``F`` is flat at ``1/2`` across
    ``[1/3, 2/3]``; and the left third is a scaled copy of the whole, which
    gives ``int_{0.2}^{0.5} S = 61/360``. The conditional layer mean is that
    over ``3/4``.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        a = build('agg CAN.Layer 1 claim 0.3 xs 0.2 sev cantor fixed '
                  'hints{bs=1/59049; log2=15}')
        moments = a.sevs[0].moms()
    assert np.all(np.isfinite(moments))
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert moments[0] == pytest.approx((61 / 360) / 0.75, rel=1e-9)
    assert a.est_m == pytest.approx((61 / 360) / 0.75, rel=1e-4)
    assert a.validation_description == 'not unreasonable'


def test_occurrence_reinsurance_over_a_cantor_severity_builds():
    """The cession rides the usual machinery; the gross severity is unlayered."""
    a = build('agg CAN.Layered 10 claims sev cantor '
              'occurrence net of 0.3 xs 0.2 poisson')
    assert a.sevs[0].moms() == pytest.approx((0.5, 0.375, 0.3125), rel=1e-13)
    assert a.est_m == pytest.approx(3.3055343628, rel=1e-6)
    assert np.isfinite(a.est_sd)


# ---------------------------------------------------------------------------
# The bucket-size helpers and the shipped library entries
# ---------------------------------------------------------------------------

def test_natural_bs_delegates_with_the_severity_filled_in():
    """The bound method carries this severity's own shape and scale."""
    sev = build('agg C 1 claim sev 3 * cantor + 5 fixed').sevs[0]
    assert sev.natural_bs(6) == cantor_bs(6, 1 / 3, scale=3.0)
    assert sev.natural_bs(6) == pytest.approx(3 / 3 ** 6, rel=1e-15)
    # A shape with no integer reciprocal has no lattice and says so.
    odd = build('agg C 1 claim sev cantor 0.4 fixed').sevs[0]
    with pytest.raises(ValueError, match='not an integer'):
        odd.natural_bs(6)


@pytest.mark.parametrize('name, c, m, q, hinted_bs', [
    ('CantorMiddleThirds', 1 / 3, 10, 3, 1 / 59049),
    ('CantorMiddleHalf', 0.5, 6, 4, 1 / 4096),
])
def test_the_library_entries_carry_their_own_natural_bucket(name, c, m, q, hinted_bs):
    """They build from the knowledge base with no explicit ``bs``.

    The hint is the whole point of the entries: the recipe is written where it
    is tripped over rather than where it has to be recalled.
    """
    a = build(name)
    assert a.bs == hinted_bs == cantor_bs(m, c)
    assert a.sevs[0].natural_bs(m) == hinted_bs
    xs, ps = cantor_pmf(m, q)
    d = _severity_pmf(a)
    idx = np.round(xs / a.bs).astype(int)
    assert np.array_equal(d[idx] + d[idx + 1], ps)
    assert a.est_m == pytest.approx(0.5, rel=1e-13)
    assert a.validation_description == 'not unreasonable'


def test_the_corpus_lines_build():
    """The four ``_test_suite.agg`` severity lines are real programs."""
    for program in (
            'agg D.Sev18 1 claim sev cantor fixed',
            'agg D.Sev19 1 claim sev cantor 0.5 fixed',
            'agg D.Sev20 1 claim sev 3 * cantor 0.5 + 5 fixed',
            'agg D.Sev21 1 claim sev cantor 10 cv 0.8 fixed'):
        a = build(program)
        assert isinstance(a.sevs[0], SeverityCantor)
        assert np.isfinite(a.est_m)
