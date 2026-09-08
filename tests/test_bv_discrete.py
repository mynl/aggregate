"""Tests for the discrete bivariate severity (``dbvsev``) + discrete-frequency
``bv`` forms (``mode='discrete'``).

A ``dbvsev`` lattice supplies the joint per-claim severity matrix ``S`` directly
(rows = X outcomes, columns = Y outcomes), the 2-D analogue of ``dsev``. The
shared outer frequency is an ordinary ``dfreq`` empirical count or an
exposure-count + named distribution; the joint runs through the same 2-D compound
FFT as the copula path, with ``S`` given instead of built from a copula.

Covers (dev/done/plan-bv-discrete.md):

- the four forms build (dfreq + dbvsev headline, count + dbvsev + freq, dfreq +
  agg/agg copula, sparse / uniform / range variants);
- an independence ``S = outer(p_x, p_y)`` reproduces the ``copula independent``
  joint of the same per-event marginals;
- marginals reproduce the standalone discrete compounds **exactly**;
- a non-separable ``S`` gives the hand-computable per-claim joint moments;
- dense == sparse, and the uniform default == the explicit uniform matrix;
- shape / normalisation guards, and the rejected ``pnl`` axis.

The DecL programs are mirrored in ``src/aggregate/agg/decl-testers.agg`` under the
``DBVSEV`` block of the ``MV`` section.
"""

import numpy as np
import pytest

from aggregate import build
from aggregate.bivariate import BivariateAggregate, _lattice_bs


# ----------------------------------------------------------------------
# build + mass conservation
# ----------------------------------------------------------------------

DENSE = ('bv DBV dfreq [1 2 3] [.5 .3 .2] '
         'dbvsev [0 1 2] [0 5 10] [[.4 .1 .0] [.1 .1 .1] [.0 .05 .15]]')


def test_dbvsev_builds_discrete_mode():
    mv = build(DENSE)
    assert isinstance(mv, BivariateAggregate)
    assert mv.mode == 'discrete'
    assert mv.copula is None
    assert np.isclose(mv.density.sum(), 1.0, atol=1e-9)
    assert mv.deficit == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize('prog', [
    'bv F3 5 claims dbvsev [10 20] [1 2 3] [[.4 .1 .0] [.2 .1 .2]] mixed gamma 0.1',
    'bv F3p 5 claims dbvsev [10 20] [1 2 3] [[.4 .1 .0] [.2 .1 .2]]',
    'bv F4 dfreq [1 2] [.7 .3] dbvsev [[10 1 .4] [10 2 .1] [20 1 .2] [20 2 .1] [20 3 .2]]',
    'bv FU 5 claims dbvsev [0 1 2] [0 5 10]',
    'bv FR 5 claims dbvsev [0:2] [0:2]',
])
def test_dbvsev_forms_build_and_conserve_mass(prog):
    mv = build(prog)
    assert mv.mode == 'discrete'
    assert np.isclose(mv.density.sum(), 1.0, atol=1e-9)


def test_dbvsev_form2_dfreq_agg_copula_builds():
    # form 2: a discrete shared frequency (dfreq) with two agg components coupled
    # by a copula -- ordinary copula mode, dfreq replacing the exposure head.
    mv = build('bv F2 dfreq [1 2] [.7 .3] '
               'agg A dfreq [0 1] [.5 .5] sev lognorm 50 cv 1 '
               'agg B dfreq [0 1] [.5 .5] sev gamma 50 cv 1.0 copula gumbel 0.4')
    assert mv.mode == 'copula'
    assert mv.freq_name == 'empirical'
    assert np.isclose(mv.density.sum(), 1.0, atol=1e-6)


# ----------------------------------------------------------------------
# independence: S = outer(p_x, p_y) == copula independent of the same marginals
# ----------------------------------------------------------------------

def _matrix_decl(S):
    return '[' + ' '.join(
        '[' + ' '.join(str(v) for v in row) + ']' for row in S) + ']'


def test_dbvsev_independence_matches_copula_independent():
    px = np.array([.5, .3, .2])
    py = np.array([.25, .25, .5])
    S = np.outer(px, py)
    disc = build(f'bv I 5 claims dbvsev [0 1 2] [0 1 2] {_matrix_decl(S)}')
    # the copula-independent build with the SAME per-event severities (each a
    # 1-claim dsev whose density is p_x / p_y directly).
    indep = build('bv J 5 claims '
                  'agg A dfreq [1] dsev [0 1 2] [.5 .3 .2] '
                  'agg B dfreq [1] dsev [0 1 2] [.25 .25 .5] copula independent')
    assert np.isclose(disc.corr, indep.corr, atol=1e-9)
    assert np.allclose(disc.moments(2).to_numpy(),
                       indep.moments(2).to_numpy(), rtol=1e-9, atol=1e-9)
    # independence at the per-claim severity level: Sev corr ~ 0
    assert abs(disc.dependency_df.loc['Sev', 'corr']) < 1e-9


# ----------------------------------------------------------------------
# marginal reproduction is EXACT (the unit marginals ARE the row/col sums of S)
# ----------------------------------------------------------------------

def test_dbvsev_marginal_reproduction_exact():
    mv = build(DENSE)
    # theory (stats_df) == realized marginal moments to FFT tolerance: the
    # unit marginals ARE the row / col sums of the given severity matrix
    stats = mv.stats_df
    for i, unit in enumerate(mv.unit_names):
        gd = mv.marginal(i)
        me = float(gd.x @ gd.p)
        assert me == pytest.approx(
            float(stats.loc[('agg', 'mean'), unit]), rel=1e-6)
    # the validation frame's per-component Agg errors are ~0
    vdf = mv.validation_df
    for unit in mv.unit_names:
        assert abs(vdf.loc[(unit, 'Agg'), 'Err EX']) < 1e-6
        assert abs(vdf.loc[(unit, 'Agg'), 'Err CV']) < 1e-6


def test_dbvsev_marginal_equals_standalone_compound():
    # marginal X of the headline equals the standalone dfreq-compound of the
    # per-event severity g0 = S.sum(axis=1) = [.5, .3, .2] on [0, 1, 2].
    mv = build(DENSE)
    m0 = mv.marginals[0]
    mean0 = float((m0 * mv.axis_xs[0]).sum())
    std = build('agg S0 dfreq [1 2 3] [.5 .3 .2] dsev [0 1 2] [.5 .3 .2]',
                bs=mv.bs[0], log2=int(np.log2(len(mv.axis_xs[0]))))
    assert np.isclose(mean0, std.est_m, rtol=1e-6)


# ----------------------------------------------------------------------
# dependence: a non-separable S gives the hand-computable per-claim moments
# ----------------------------------------------------------------------

def test_dbvsev_dependence_hand_computed_sev_moments():
    # perfect comonotone 2x2 on {0,1}x{0,1}: X == Y always.
    # E[X]=E[Y]=.5, E[XY]=.5, cov=.25, var=.25, corr=1.
    mv = build('bv C 5 claims dbvsev [0 1] [0 1] [[.5 .0] [.0 .5]]')
    dep = mv.dependency_df
    assert np.isclose(dep.loc['Sev', 'cov'], 0.25)
    assert np.isclose(dep.loc['Sev', 'corr'], 1.0)
    assert np.isnan(dep.loc['Sev', 'tau'])      # no copula in discrete mode


# ----------------------------------------------------------------------
# dense == sparse == uniform-default
# ----------------------------------------------------------------------

def test_dbvsev_dense_equals_sparse():
    dense = build('bv DN 5 claims dbvsev [10 20] [1 2 3] [[.4 .1 .0] [.2 .1 .2]]')
    sparse = build('bv SP 5 claims '
                   'dbvsev [[10 1 .4] [10 2 .1] [20 1 .2] [20 2 .1] [20 3 .2]]')
    assert np.allclose(dense._S, sparse._S)
    assert dense.density.shape == sparse.density.shape
    assert np.allclose(dense.density, sparse.density)


def test_dbvsev_uniform_default_equals_explicit():
    u1 = build('bv U1 5 claims dbvsev [0 1 2] [0 5 10]')
    u2 = build('bv U2 5 claims dbvsev [0 1 2] [0 5 10] '
               '[[1/9 1/9 1/9] [1/9 1/9 1/9] [1/9 1/9 1/9]]')
    assert np.allclose(u1._S_given, u2._S_given)
    assert np.allclose(u1.density, u2.density)


def test_dbvsev_range_form_lattice():
    mv = build('bv R 5 claims dbvsev [0:2] [0:2]')
    assert mv._dbv_xs.tolist() == [0.0, 1.0, 2.0]
    assert mv._dbv_ys.tolist() == [0.0, 1.0, 2.0]
    assert np.isclose(mv.density.sum(), 1.0, atol=1e-9)


# ----------------------------------------------------------------------
# guards
# ----------------------------------------------------------------------

def test_dbvsev_shape_mismatch_raises():
    with pytest.raises(ValueError, match='does not match the lattice'):
        build('bv Bad 5 claims dbvsev [0 1] [0 1 2] [[.5 .5] [.5 .5]]')


def test_dbvsev_negative_entry_raises():
    with pytest.raises(ValueError, match='negative'):
        build('bv Neg 5 claims dbvsev [0 1] [0 1] [[.6 .6] [-.1 -.1]]')


def test_dbvsev_unnormalised_renormalises(caplog):
    # probabilities summing to != 1 warn and renormalise; mass still conserved.
    mv = build('bv UN 5 claims dbvsev [0 1] [0 1] [[.2 .1] [.1 .0]]')   # sum .4
    assert np.isclose(mv._S_given.sum(), 1.0)
    assert np.isclose(mv.density.sum(), 1.0, atol=1e-9)


# ----------------------------------------------------------------------
# lattice bucket helper
# ----------------------------------------------------------------------

@pytest.mark.parametrize('xs,expected', [
    ([0, 1, 2], 1.0),
    ([0, 5, 10], 5.0),
    ([10, 20, 35], 5.0),
    ([0.5, 1.5], 0.5),
    ([0], 1.0),
])
def test_lattice_bs(xs, expected):
    assert np.isclose(_lattice_bs(xs), expected)


# ----------------------------------------------------------------------
# [Bivariate-Punchup] independent pair: accessors match hand arithmetic
# ----------------------------------------------------------------------

# One claim always; S = outer([.5 .3 .2], [.6 .3 .1]) on the shared lattice
# [0 1 2] x [0 1 2], so the joint aggregate IS the severity matrix and every
# accessor has a closed form. Mirrored in decl-testers.agg (DBVSEV block).
INDEP = ('bv DBVIndep dfreq [1] dbvsev [0 1 2] [0 1 2] '
         '[[.30 .15 .05] [.18 .09 .03] [.12 .06 .02]]')


def test_dbvsev_independent_pair_hand_arithmetic():
    mv = build(INDEP)
    px = np.array([.5, .3, .2])
    py = np.array([.6, .3, .1])
    # marginals (the dyadic grid pads with zeros past the support)
    np.testing.assert_allclose(mv.marginal(0).p[:3], px, atol=1e-12)
    np.testing.assert_allclose(mv.marginal(1).p[:3], py, atol=1e-12)
    assert mv.marginal(0).p[3:].sum() == 0
    # independence: the conditional given X = 1 is the Y marginal
    np.testing.assert_allclose(mv.conditional('x', 1).p[:3], py, atol=1e-12)
    # the total is the convolution
    t = mv.total
    np.testing.assert_allclose(t.p[:5], np.convolve(px, py), atol=1e-12)
    assert t.p[5:].sum() == 0
    np.testing.assert_allclose(t.x[:5], np.arange(5.0))
    # law of X given X + Y = 2: [.05 .09 .12] / .26, and its report=1 mirror
    d0 = mv.conditional('x+y', 2)
    np.testing.assert_allclose(d0.p[:3], np.array([.05, .09, .12]) / .26,
                               atol=1e-12)
    d1 = mv.conditional('x+y', 2, report=1)
    np.testing.assert_allclose(d1.p[:3], np.array([.12, .09, .05]) / .26,
                               atol=1e-12)


# ----------------------------------------------------------------------
# signed lattice: negative outcomes survive the scatter (regression, a345)
# ----------------------------------------------------------------------

# Uniform 7x7 on a signed X lattice. Through a344 the scatter used raw
# (negative, numpy-wrapped) indices while _lay_signed_2d wrapped the first
# i0 rows again, so every X atom landed i0 buckets low and the negative
# atoms fell off the output window (mass dropped to 4/7 on this example).
SIGNED = ('bv DBVSigned dfreq [1] dbvsev [-3:3] [0:6] '
          '[' + ' '.join(['[' + ' '.join(['0.02040816326530612'] * 7) + ']'] * 7) + ']')


def test_dbvsev_signed_x_lattice():
    mv = build(SIGNED)
    assert np.isclose(mv.density.sum(), 1.0, atol=1e-9)
    mx = mv.density.sum(axis=1)
    my = mv.density.sum(axis=0)
    assert float(mv.axis_xs[0] @ mx) == pytest.approx(0.0, abs=1e-9)
    assert float(mv.axis_xs[1] @ my) == pytest.approx(3.0, rel=1e-9)
    # one fixed claim: the joint IS the severity matrix, so every lattice
    # atom carries exactly 1/49 at its own (x, y) location
    for x in range(-3, 4):
        ixx = int(np.searchsorted(mv.axis_xs[0], x))
        assert mx[ixx] == pytest.approx(1.0 / 7.0, rel=1e-9), f'X atom {x}'


def test_dbvsev_signed_both_axes_random_freq():
    mat = '[' + ' '.join(['[' + ' '.join(['0.02040816326530612'] * 7) + ']'] * 7) + ']'
    mv = build(f'bv DBVSigned2 dfreq [0 1 2] [.25 .5 .25] dbvsev [-3:3] [-3:3] {mat}')
    assert np.isclose(mv.density.sum(), 1.0, atol=1e-9)
    assert float(mv.axis_xs[0] @ mv.density.sum(axis=1)) == pytest.approx(0.0, abs=1e-9)
    assert float(mv.axis_xs[1] @ mv.density.sum(axis=0)) == pytest.approx(0.0, abs=1e-9)
