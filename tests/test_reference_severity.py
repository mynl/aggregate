"""[SeverityMeta-Afresh]: the programmatic reference severity (1.0.0a290).

``Severity(some_aggregate)`` / ``Aggregate.as_severity()`` /
``Portfolio.as_severity()`` build a :class:`SeverityMeta`, which is now a
:class:`SeverityDHistogram` over the source's output pmf rather than the
pre-1.0 ``rv_histogram`` hybrid: exact discrete moments, ``support_atoms``, an
honest ``_DiscreteRV``, no in-place update of the source.

The DecL half of the feature (``sev agg.NAME``) is tested in
``test_agg_as_severity.py``; both routes share
``aggregate._severity._dhistogram_from_object``, so the exactness claims here
carry over.

See ``dev/plan-agg-port-as-sev.md`` phase A.
"""

import logging
import warnings

import numpy as np
import pytest

from aggregate import build
from aggregate.distributions import Severity

logging.disable(logging.CRITICAL)


def _inner(name='RS.Inner'):
    """The plan's headline inner: a per-policy split-limit aggregate."""
    return build(f'agg {name} 1.5 claims 100 xs 0 sev gamma 50 cv 2 poisson zt !')


def _exact_moment(xs, ps, n):
    return float(np.sum(np.asarray(xs, float) ** n * np.asarray(ps, float)))


# ----------------------------------------------------------------------
# The author's 2026-08-16 repro, and its Aggregate twin
# ----------------------------------------------------------------------
def test_portfolio_as_severity_no_longer_raises():
    # Pre-a290 this raised ``IndexError: string index out of range`` inside
    # scipy, before any severity logic ran: an object-valued ``sev_name`` was
    # handed ``name=''`` and scipy indexes ``name[0]``.
    p = build('port RS.pSL agg RS.SL 1.5 claims 100 xs 0 '
              'sev gamma 50 cv 2 poisson zt !')
    s = p.as_severity()
    assert s.sev_kind == 'meta'
    assert s.support_atoms is not None and len(s.support_atoms) > 1
    assert s.sev1 == pytest.approx(p.est_m, rel=1e-10)


def test_aggregate_as_severity_twin():
    a = _inner()
    s = a.as_severity()
    assert s.sev_kind == 'meta'
    assert s.sev1 == pytest.approx(a.est_m, rel=1e-10)


def test_as_severity_passes_the_layer_through():
    a = _inner()
    s = a.as_severity(limit=300, attachment=0, conditional=False)
    assert s.limit == 300
    assert s.attachment == 0
    assert s.conditional is False


# ----------------------------------------------------------------------
# It is a fully formed dsev: exact moments off the atoms
# ----------------------------------------------------------------------
def test_meta_moments_are_the_exact_atom_sums():
    a = _inner()
    s = Severity(a)
    xs, ps = a.xs, a.agg_density
    ps = ps / ps.sum()
    for n, got in ((1, s.sev1), (2, s.sev2), (3, s.sev3)):
        assert got == pytest.approx(_exact_moment(xs, ps, n), rel=1e-12)


def test_meta_carries_support_atoms_and_a_discrete_rv():
    a = _inner()
    s = Severity(a)
    # zero-probability buckets are dropped, so the atom count is below the
    # source's 2**log2 grid
    assert len(s.support_atoms) < 2 ** a.log2
    # honest step cdf, not a continuous interpolation
    assert s.fz.pdf(s.support_atoms[5]) == 0.0
    # the step lands exactly on the atom and nowhere below it
    lo, nxt = s.support_atoms[0], s.support_atoms[1]
    assert s.fz.cdf(lo) == pytest.approx(float(s.sev_ps[0]), rel=1e-12)
    assert s.fz.cdf((lo + nxt) / 2) == pytest.approx(float(s.sev_ps[0]), rel=1e-12)


def test_meta_layer_moments_are_exact():
    a = _inner()
    xs, ps = a.xs, a.agg_density
    ps = ps / ps.sum()
    s = Severity(a, exp_attachment=0, exp_limit=100, sev_conditional=False)
    y = np.clip(xs - 0.0, 0.0, 100.0)
    assert s.moms()[0] == pytest.approx(float(np.sum(y * ps)), rel=1e-12)
    assert s.moms()[1] == pytest.approx(float(np.sum(y * y * ps)), rel=1e-12)


# ----------------------------------------------------------------------
# The nullary query: the source answers from its current state, or refuses
# ----------------------------------------------------------------------
def test_never_updated_source_raises():
    a = build('agg RS.Bare 1.5 claims 100 xs 0 sev gamma 50 cv 2 poisson',
              update=False)
    with pytest.raises(ValueError, match='before it has computed itself'):
        Severity(a)
    with pytest.raises(ValueError, match='update the aggregate'):
        a.as_severity()


def test_never_updated_portfolio_raises():
    p = build('port RS.Bare2 agg RS.U 1.5 claims 100 xs 0 '
              'sev gamma 50 cv 2 poisson', update=False)
    with pytest.raises(ValueError, match='update the portfolio'):
        p.as_severity()


def test_source_is_not_updated_in_place():
    a = _inner()
    log2, bs = a.log2, a.bs
    Severity(a)
    assert (a.log2, a.bs) == (log2, bs)


def test_conflicting_sev_a_sev_b_raise():
    a = _inner()
    with pytest.raises(ValueError, match='no longer re-grids the source'):
        Severity(a, sev_a=a.log2 + 2)
    with pytest.raises(ValueError, match='no longer re-grids the source'):
        Severity(a, sev_b=a.bs * 4)
    # restating the current grid is accepted: that is the old call shape
    s = Severity(a, sev_a=a.log2, sev_b=a.bs)
    assert s.sev1 == pytest.approx(a.est_m, rel=1e-10)


def test_wrong_source_type_raises():
    with pytest.raises(ValueError, match='only'):
        from aggregate._severity import _dhistogram_from_object
        _dhistogram_from_object(object())


# ----------------------------------------------------------------------
# Signed sources: ssev keeps the negative atoms, sev clamps and warns
# ----------------------------------------------------------------------
def _signed_source():
    return build('agg RS.Signed dfreq [1] ssev 100 - lognorm 50 cv .5')


def test_signed_source_under_ssev_keeps_negative_atoms():
    a = _signed_source()
    assert a.xs.min() < 0
    s = Severity(a, sev_signed=True)
    assert s.support_atoms.min() < 0
    assert s.signed is True


def test_signed_source_under_sev_clamps_and_warns():
    a = _signed_source()
    with pytest.warns(UserWarning, match='clamped onto the zero atom'):
        s = Severity(a)
    assert s.support_atoms.min() >= 0
    # the clamped mass lands on the zero atom, so the total is preserved
    assert float(np.sum(s.sev_ps)) == pytest.approx(1.0, rel=1e-12)
    # and the mean rises: negative outcomes were moved up to 0
    assert s.sev1 > a.est_m


def test_nonnegative_source_under_ssev_is_silent():
    a = _inner()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        Severity(a, sev_signed=True)
    assert not [w for w in record if 'clamped' in str(w.message)]
