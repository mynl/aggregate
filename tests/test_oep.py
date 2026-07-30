"""[OEP-Curve] regressions: ``oep`` and the exact severity inverse.

``oep(agg, p)`` answers "there is a probability ``p`` that one or more
occurrences in a year exceed ``x``, what is ``x``?" for a Poisson aggregate. It
is built on :meth:`Aggregate.sev`, the exact continuous severity, whose ``ppf``
and ``isf`` landed alongside it; the pre-existing :meth:`Aggregate.q_sev` can
only return a point on the ``bs`` lattice.

These pin the contract both must deliver: the loss solves
``p = 1 - exp(-lam * Pr(L > x))`` exactly, it reproduces the published values in
*Return Period Confusions Clarified* and the formula in the catastrophe modeling
user guide, it is genuinely off the lattice, the mixture inverse actually
inverts the mixture, and the Poisson precondition and the ``1 - exp(-lam)``
ceiling are enforced rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.stats as ss
from scipy.optimize import brentq

from aggregate import build
from aggregate.utilities import oep

# ``ss.lognorm(1, scale=1000)``, the severity used throughout the blog post,
# expressed in the (mean, cv) parameterization DecL wants. Full precision: the
# published losses are quoted to three decimals and a truncated cv misses them.
BLOG_MEAN = 1648.7212707001281      # 1000 * exp(1/2)
BLOG_CV = 1.3108324944320862        # sqrt(exp(1) - 1)
BLOG_LAM = 2.0
BLOG_FZ = ss.lognorm(1, scale=1000)

# Table 8 of the post, the EQ column, at lam = 2.
BLOG_TABLE_8 = {
    0.001: 26853.227,
    0.002734122436058: 20000.000,
    0.01: 13119.408,
    0.02: 10201.775,
    0.05: 7021.788,
    0.1: 5050.076,
    0.5: 1483.772,
}


def _blog_agg(update=False):
    """The post's compound Poisson: lam = 2, ``lognorm(1, scale=1000)``.

    Built lazily by default. ``oep`` reads only the input severity and the
    claim count, so running the FFT would cost time and raise unrelated
    thin-tail-clipping warnings. Pass ``update=True`` for the tests that
    compare against the grid.
    """
    return build(f'agg OEP:Blog {BLOG_LAM:g} claims '
                 f'sev lognorm {BLOG_MEAN!r} cv {BLOG_CV!r} poisson',
                 update=update)


def _mixed_agg(update=False):
    """A four-component severity mixture, so the inverse needs a root solve."""
    return build('agg OEP:Mix [1 2] claims sev lognorm [1000 5000] cv [1 2] '
                 'wts [.6 .4] poisson', log2=16, bs=32, update=update)


# ---------------------------------------------------------------------------
# the published reference values
# ---------------------------------------------------------------------------

def test_matches_blog_table_8():
    """Reproduces the EQ column of the post's Table 8."""
    a = _blog_agg()
    ps = list(BLOG_TABLE_8)
    got = oep(a, ps).loss.values
    want = np.array([BLOG_TABLE_8[p] for p in ps])
    # the published values are rounded to three decimals
    assert np.allclose(got, want, atol=5e-4), f'{got} != {want}'


def test_matches_scipy_isf_directly():
    """``loss`` is exactly ``fz.isf(-log1p(-p) / lam)``, no tolerance needed."""
    a = _blog_agg()
    ps = np.array(list(BLOG_TABLE_8))
    got = oep(a, ps).loss.values
    want = BLOG_FZ.isf(-np.log1p(-ps) / BLOG_LAM)
    assert np.array_equal(got, want), f'{got} != {want}'


@pytest.mark.parametrize('n', [10, 100, 1000])
def test_matches_cat_guide_formula(n):
    """Pins ``OEP(n) = q(1 + log(1 - 1/n) / lam)`` from ``2_x_cat.rst``."""
    a = _blog_agg()
    got = float(oep(a, 1 / n).loss.iloc[0])
    want = float(BLOG_FZ.ppf(1 + np.log(1 - 1 / n) / BLOG_LAM))
    assert got == pytest.approx(want, rel=1e-11)


# ---------------------------------------------------------------------------
# the internal identities
# ---------------------------------------------------------------------------

def test_round_trips_to_the_requested_probability():
    """For a continuous severity the achieved ``oep`` is the requested ``p``."""
    a = _blog_agg()
    df = oep(a, list(BLOG_TABLE_8))
    assert np.allclose(df.oep.values, df.index.values, rtol=1e-12, atol=1e-15)


def test_column_identities():
    """The two return periods and the severity probabilities are consistent."""
    a = _blog_agg()
    df = oep(a, [0.001, 0.01, 0.1, 0.5])
    assert np.allclose(df.F_sev, 1.0 - df.S_sev)
    assert np.allclose(df.S_sev, a.sev.sf(df.loss.values))
    assert np.allclose(df.occurrence_return_period, 1.0 / (BLOG_LAM * df.S_sev))
    assert np.allclose(df.annual_return_period, 1.0 / df.oep)
    # counting occurrences is always more frequent than counting the years
    # holding them, so the occurrence return period is the shorter one
    assert (df.occurrence_return_period < df.annual_return_period).all()
    assert (df.annual_return_period >= 1.0).all()


def test_occurrence_return_period_can_be_sub_annual():
    """The occurrence clock is not bounded below by a year; the annual one is."""
    a = _blog_agg()
    df = oep(a, 0.8)
    assert df.occurrence_return_period.iloc[0] < 1.0
    assert df.annual_return_period.iloc[0] > 1.0


# ---------------------------------------------------------------------------
# exact, not bucketed: the reason this function exists
# ---------------------------------------------------------------------------

def test_loss_is_off_the_lattice():
    """``oep`` beats the ``q_sev`` recipe, which can only return a grid point."""
    a = _blog_agg(update=True)
    p = 0.001
    exact = float(oep(a, p).loss.iloc[0])
    snapped = float(a.q_sev(1 + np.log1p(-p) / BLOG_LAM))
    assert exact % a.bs != 0.0, 'exact answer landed on the lattice by accident'
    assert snapped % a.bs == 0.0
    assert exact != snapped
    assert abs(exact - snapped) < a.bs


def test_sev_inverse_beats_q_sev():
    """``sev.ppf`` is the exact quantile; ``q_sev`` snaps to the bucket."""
    a = build('agg OEP:Snap 2 claims sev lognorm 1000 cv 1.31 poisson')
    assert float(a.sev.ppf(0.99)) == pytest.approx(6207.853451972701, rel=1e-12)
    assert float(a.q_sev(0.99)) == 6208.0


# ---------------------------------------------------------------------------
# Aggregate.sev inverses
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('agg_fn', [_blog_agg, _mixed_agg])
def test_sev_ppf_isf_round_trip(agg_fn):
    """``ppf`` inverts ``cdf`` and ``isf`` inverts ``sf``, mixture included."""
    a = agg_fn()
    for q in (0.001, 0.01, 0.25, 0.5, 0.9, 0.99):
        assert float(a.sev.cdf(a.sev.ppf(q))) == pytest.approx(q, rel=1e-10)
        assert float(a.sev.sf(a.sev.isf(q))) == pytest.approx(q, rel=1e-10)
    for x in (500.0, 1234.5, 20000.0):
        assert float(a.sev.isf(a.sev.sf(x))) == pytest.approx(x, rel=1e-9)
        assert float(a.sev.ppf(a.sev.cdf(x))) == pytest.approx(x, rel=1e-9)


def test_sev_has_five_functions():
    """The namedtuple carries the inverses alongside the forward functions."""
    a = _blog_agg()
    assert a.sev._fields == ('cdf', 'sf', 'pdf', 'ppf', 'isf')


def test_sev_inverse_vectorizes():
    """Array in, array out; a scalar stays 0-d, per the house ``_unwrap`` rule."""
    a = _mixed_agg()
    qs = [0.5, 0.1, 0.01, 0.001]
    got = a.sev.isf(qs)
    assert got.shape == (4,)
    assert np.allclose(got, [float(a.sev.isf(q)) for q in qs])
    assert np.asarray(a.sev.isf(0.01)).shape == ()


def test_mixture_inverse_matches_direct_root_solve():
    """The bracketed solve agrees with a naive brentq on the weighted sf."""
    a = _mixed_agg()
    assert len(a.sevs) > 1, 'fixture must be a genuine mixture'
    for s in (0.5, 0.05, 0.01, 0.001):
        lo = min(float(sv.isf(s)) for sv in a.sevs)
        hi = max(float(sv.isf(s)) for sv in a.sevs)
        want = brentq(lambda x: float(a.sev.sf(x)) - s, lo, hi)
        assert float(a.sev.isf(s)) == pytest.approx(want, rel=1e-10)


def test_mixture_inverse_is_bracketed_by_its_components():
    """The mixture quantile lies between the extreme component quantiles."""
    a = _mixed_agg()
    for s in (0.5, 0.01, 0.001):
        cands = [float(sv.isf(s)) for sv in a.sevs]
        assert min(cands) <= float(a.sev.isf(s)) <= max(cands)


def test_oep_on_a_mixture():
    """The whole curve round trips when the severity is a mixture."""
    a = _mixed_agg()
    df = oep(a, [0.001, 0.01, 0.1])
    assert np.allclose(df.oep.values, df.index.values, rtol=1e-10)
    assert (df.loss.values > 0).all()
    assert (np.diff(df.loss.values) < 0).all(), 'higher p must mean lower loss'


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------

def test_ceiling_is_enforced():
    """``p`` at or above ``1 - exp(-lam)`` has no solution."""
    a = _blog_agg()
    ceiling = -np.expm1(-BLOG_LAM)
    assert ceiling == pytest.approx(0.8646647167633873)
    for p in (ceiling, ceiling + 1e-9, 0.95):
        with pytest.raises(ValueError, match='ceiling'):
            oep(a, p)
    # just below it still resolves
    assert float(oep(a, ceiling - 1e-9).loss.iloc[0]) > 0


def test_ceiling_moves_with_freq():
    """A larger lambda lifts the ceiling, so a rejected ``p`` becomes legal."""
    a = _blog_agg()
    with pytest.raises(ValueError, match='ceiling'):
        oep(a, 0.9)
    assert float(oep(a, 0.9, freq=10).loss.iloc[0]) > 0


@pytest.mark.parametrize('program', [
    'agg OEP:Negbin 2 claims sev lognorm 1000 cv 2 negbin 2',
    'agg OEP:Fixed 2 claims sev lognorm 1000 cv 2 fixed',
    'agg OEP:Mixed 2 claims sev lognorm 1000 cv 2 mixed gamma 0.3',
])
def test_non_poisson_frequency_raises(program):
    """The thinning derivation is Poisson only."""
    with pytest.raises(ValueError, match='Poisson'):
        oep(build(program, update=False), 0.01)


def test_zero_modified_poisson_raises():
    """Zero modification breaks ``P(N = 0) = exp(-lam)``."""
    a = build('agg OEP:ZM 2 claims sev lognorm 1000 cv 2 poisson zm 0.5',
              update=False)
    with pytest.raises(ValueError, match='zero modified'):
        oep(a, 0.01)


@pytest.mark.parametrize('p', [0.0, 1.0, -0.1, 1.5, [0.01, 0.0], [0.01, 2.0]])
def test_p_out_of_range_raises(p):
    """``p`` must be in the open unit interval, scalar or inside an iterable."""
    with pytest.raises(ValueError, match=r'\(0, 1\)'):
        oep(_blog_agg(), p)


def test_non_aggregate_raises():
    with pytest.raises(TypeError, match='Aggregate'):
        oep('not an aggregate', 0.01)


@pytest.mark.parametrize('bad', [-1.0, -1e-9, np.nan, np.inf])
def test_bad_freq_raises(bad):
    with pytest.raises(ValueError, match='freq'):
        oep(_blog_agg(), 0.01, freq=bad)


# ---------------------------------------------------------------------------
# freq override
# ---------------------------------------------------------------------------

def test_freq_overrides_the_claim_count():
    """``freq`` replaces ``agg.n`` and nothing else."""
    a = _blog_agg()
    p = 0.01
    got = float(oep(a, p, freq=5).loss.iloc[0])
    want = float(BLOG_FZ.isf(-np.log1p(-p) / 5.0))
    assert got == pytest.approx(want, rel=1e-12)
    # a higher event rate makes any given annual probability a smaller loss
    assert got > float(oep(a, p).loss.iloc[0])


def test_freq_zero_uses_agg_n():
    """``freq=0`` is the sentinel for 'use the aggregate's own count'."""
    a = _blog_agg()
    assert oep(a, 0.01).equals(oep(a, 0.01, freq=0))
    assert oep(a, 0.01).equals(oep(a, 0.01, freq=BLOG_LAM))


# ---------------------------------------------------------------------------
# shape, ordering, and the atom case
# ---------------------------------------------------------------------------

def test_scalar_p_gives_one_row():
    df = oep(_blog_agg(), 0.01)
    assert df.shape == (1, 6)
    assert df.index.name == 'p'
    assert list(df.columns) == ['loss', 'S_sev', 'F_sev', 'oep',
                                'occurrence_return_period', 'annual_return_period']


def test_input_order_is_preserved():
    """The frame is not sorted behind the caller's back."""
    ps = [0.05, 0.001, 0.01]
    assert list(oep(_blog_agg(), ps).index) == ps


def test_limited_severity_reports_the_atom_honestly():
    """Above the limit nothing is exceeded, and the frame says so."""
    a = build('agg OEP:Limit 2 claims 5000 xs 0 sev lognorm 1000 cv 2 poisson',
              update=False)
    df = oep(a, [0.02, 0.01, 0.0001])
    assert (df.loss == 5000.0).all(), 'quantile should pin to the limit'
    assert (df.S_sev == 0.0).all()
    # the achieved probability is capped below what was asked for, and the
    # return periods go infinite rather than quietly repeating the same loss
    assert (df.oep == 0.0).all()
    assert (df.oep < df.index.values).all()
    assert np.isinf(df.occurrence_return_period).all()
    assert np.isinf(df.annual_return_period).all()


def test_works_without_update():
    """``oep`` reads the input severity, never ``density_df``."""
    for program in ('agg OEP:NoUpd 2 claims sev lognorm 1000 cv 2 poisson',
                    'agg OEP:NoUpdMix [1 2] claims sev lognorm [1000 5000] '
                    'cv [1 2] wts [.6 .4] poisson'):
        lazy = build(program, update=False)
        eager = build(program)
        assert float(oep(lazy, 0.01).loss.iloc[0]) == pytest.approx(
            float(oep(eager, 0.01).loss.iloc[0]), rel=1e-12)
