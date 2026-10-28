"""Dyadic lattice buckets for discrete severities ([Fractional-Lattice-Bucket]).

When every severity atom is a whole multiple of a common dyadic step ``h``, the
sizer picks the largest nice divisor of ``h`` whose window fits the ``log2``
cap, else ``h``, else no lattice. Fractional and signed atoms get an exact,
small grid as integer atoms always did, and a fixed and a random count on the
same atoms get the same bucket. Decimal atoms and thirds have no dyadic lattice
and size exactly as before.

The programs are mirrored in ``decl-testers.agg`` (section FLB). See
``dev/done/plan-fractional-lattice-bucket.md``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from aggregate import build
from aggregate._bucket_window import _bucket_is_exact
from aggregate.utilities import (LATTICE_DEPTH, LATTICE_RTOL,
                                 _dyadic_lattice_step, _lattice_candidates)


# ---------------------------------------------------------------- the routine

@pytest.mark.parametrize('atoms, h', [
    ([0.25, 0.5, 1.75], 0.25),
    ([-1.75, 0.25, 0.5], 0.25),
    ([0.75, 1.5, 2.25], 0.75),
    ([1.5, 3, 4.5], 1.5),
    ([3, 6, 9], 3.0),
    ([0, 5, 10], 5.0),
    ([10, 20, 35], 5.0),
    ([1000, 2000, 5000], 1000.0),
    # scale free: the search starts at max|x|, not at 1
    ([200000, 400000, 600000, 800000, 1000000], 200000.0),
    ([0, 2.0 ** 20], 2.0 ** 20),
    ([0, 0], 1.0),
])
def test_dyadic_step(atoms, h):
    step, residual = _dyadic_lattice_step(atoms)
    assert step == h
    assert residual == 0.0


@pytest.mark.parametrize('atoms', [
    [0.1, 0.2], [0.1, 0.2, 0.3], [0.1, 0.2, 0.4], [1 / 3, 2 / 3],
    [0, 0.1, 0.3000001], [0, 1, 100000], [], [1.0, np.inf]])
def test_no_dyadic_step(atoms):
    """Decimals, thirds, and steps deeper than LATTICE_DEPTH have no lattice."""
    assert _dyadic_lattice_step(atoms) is None


def test_noise_lands_on_the_lattice():
    """A few units in the last place is noise, not a finer lattice."""
    atoms = [0.25000000000000006, 0.5000000000000001, 1.7499999999999998]
    step, residual = _dyadic_lattice_step(atoms)
    assert step == 0.25
    assert 0 < residual <= LATTICE_RTOL * 1.75


def test_depth_is_relative_to_the_largest_atom():
    big = 2.0 ** 30                       # max|x| exactly, so the floor is sharp
    assert _dyadic_lattice_step([big, big - 2.0 ** (30 - LATTICE_DEPTH)])[0] \
        == 2.0 ** (30 - LATTICE_DEPTH)
    assert _dyadic_lattice_step([big, big - 2.0 ** (29 - LATTICE_DEPTH)]) is None
    assert _dyadic_lattice_step([big, big - 2.0 ** (29 - LATTICE_DEPTH)],
                                depth=LATTICE_DEPTH + 1) is not None


@pytest.mark.parametrize('h, candidates', [
    (0.25, [0.25]), (0.75, [0.25, 0.75]), (1.5, [0.5, 1.5]),
    (1, [1.0]), (3, [1.0, 3.0]), (6, [2.0, 6.0]), (12, [4.0, 12.0]),
    (15, [5.0, 15.0]), (300, [100.0, 300.0]), (1000, [1000.0]),
    (40, [40.0]), (1.5e6, [5e5, 1.5e6])])
def test_lattice_candidates(h, candidates):
    assert _lattice_candidates(h) == candidates


# ---------------------------------------------------------- the worked examples

WORKED = [
    # atoms, count, frequency, log2 cap or None, rule bs, rule log2
    ('0.25 0.5 1.75', '3 claims', 'fixed', None, 0.25, 5),
    ('-1.75 0.25 0.5', '3 claims', 'fixed', None, 0.25, 5),
    ('0.75 1.5 2.25', '3 claims', 'fixed', None, 0.25, 5),
    ('0.75 1.5 2.25', '3 claims', 'fixed', 4, 0.75, 4),
    ('1.5 3 4.5', '3 claims', 'fixed', None, 0.5, 5),
    ('3 6 9', '3 claims', 'fixed', None, 1.0, 5),
    ('3 6 9', '3 claims', 'poisson', None, 1.0, 8),
    ('3 6 9', '1000000 claims', 'poisson', None, 3.0, 15),
    ('5 10 15', '3 claims', 'fixed', None, 5.0, 4),
    ('1000 2000 5000', '3 claims', 'fixed', None, 1000.0, 4),
    ('1000 2000 5000', '3 claims', 'poisson', None, 1000.0, 7),
]


@pytest.mark.parametrize('atoms, count, freq, cap, bs, log2', WORKED)
def test_worked_examples(atoms, count, freq, cap, bs, log2):
    kw = {} if cap is None else dict(log2=cap)
    a = build(f'agg FLB.T {count} dsev [{atoms}] [.25 .5 .25] {freq}', **kw)
    assert (a.bs, a.log2) == (bs, log2)
    # every atom on a grid point, after the origin and any pad balancing
    xs = np.array([float(v) for v in atoms.split()])
    k = (xs - a.x_min) / a.bs
    assert np.array_equal(k, np.round(k))


def test_fixed_and_random_counts_agree():
    """The hard-coded bs = 1 of the exact-discrete row is gone."""
    f = build('agg FLB.Fixed 3 claims dsev [5 10 15] fixed')
    p = build('agg FLB.Poi 3 claims dsev [5 10 15] poisson')
    assert f.bs == p.bs == 5


DECIMAL = [
    # 1.23.0 grids, measured 2026-10-28: no lattice, so unchanged
    ('0.1 0.2', '[.5 .5]', 'fixed', 2.0 ** -16),
    ('0.1 0.2 0.3', '[.25 .5 .25]', 'fixed', 2.0 ** -16),
    ('0.1 0.2 0.4', '[.25 .5 .25]', 'fixed', 2.0 ** -15),
    ('1/3 2/3', '[.5 .5]', 'fixed', 2.0 ** -14),
    ('0.1 0.2 0.3', '[.25 .5 .25]', 'poisson', 2.0 ** -13),
    ('1/3 2/3', '[.5 .5]', 'poisson', 2.0 ** -12),
]


@pytest.mark.parametrize('atoms, ps, freq, bs', DECIMAL)
def test_decimal_atoms_size_as_before(atoms, ps, freq, bs):
    a = build(f'agg FLB.Dec 3 claims dsev [{atoms}] {ps} {freq}')
    assert a._severity_lattice() is None
    assert 'exact_discrete' not in a.bs_window_df.index
    assert (a.bs, a.log2) == (bs, 16)


# ------------------------------------------------------------- noise, pins, notes

NOISY = '0.25000000000000006 0.5000000000000001 1.7499999999999998'


@pytest.mark.parametrize('scheme', ['linear', 'nearest'])
def test_noisy_atoms_land_whole(scheme):
    clean = build('agg FLB.Clean 1 claim dsev [0.25 0.5 1.75] [.25 .5 .25] fixed')
    a = build(f'agg FLB.Noisy 1 claim dsev [{NOISY}] [.25 .5 .25] fixed',
              dsev_bucket=scheme)
    assert (a.bs, a.log2) == (clean.bs, clean.log2) == (0.25, 3)
    p = np.asarray(a.sev_density)
    assert np.count_nonzero(p) == 3
    assert np.allclose(p[np.nonzero(p)], [.25, .5, .25], rtol=0, atol=1e-15)
    note = a.bs_window_df.loc['exact_discrete', 'note']
    assert 'lattice 1/4 (residual 2.2e-16), bs 1/4 on it' in note


def test_nice_divisor_is_noted():
    a = build('agg FLB.Nice 3 claims dsev [0.75 1.5 2.25] [.25 .5 .25] fixed')
    df = a.bs_window_df
    assert df.loc['exact_discrete', 'coverage'] == 'exact'
    assert 'lattice 0.75 (residual 0), bs 1/4 on it' in df.loc['exact_discrete', 'note']
    assert 'exact discrete (lattice) support' in a.bs_explanation


def test_pinned_bs_off_the_lattice_is_noted_not_warned():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        a = build('agg FLB.Pin 3 claims dsev [0.75 1.5 2.25] [.25 .5 .25] fixed',
                  bs=0.5)
    assert a.bs == 0.5
    assert 'pinned bs 1/2 misses the lattice 0.75' in \
        a.bs_window_df.loc['exact_discrete', 'note']


def test_a_lattice_too_fine_for_the_cap_is_no_lattice():
    """No candidate fits log2 4, so the resolution estimate runs."""
    a = build('agg FLB.Fine 3 claims dsev [0.25 0.5 1.75] [.25 .5 .25] fixed',
              log2=2)
    row = a.bs_window_df.loc['exact_discrete']
    assert row['coverage'] == 'support exact, bs coarsened'
    assert 'too fine for the log2 cap' in row['note']


# ------------------------------------------------------- sharpen and Portfolios

def test_portfolio_lattice_is_the_pooled_step():
    p = build('port FLB.Port '
              'agg FLB.P1 2 claims dsev [-1.75 0.25 0.5] [.25 .5 .25] fixed '
              'agg FLB.P2 2 claims dsev [0.75 1.5] fixed')
    assert _bucket_is_exact(p, 0.25) == (True, 0.25)
    assert not _bucket_is_exact(p, 0.5)[0]


def test_sharpen_does_not_call_the_nice_bucket_too_fine():
    """bs 1 on a lattice of 3 is the sizer's choice, not a wasteful one."""
    a = build('agg FLB.Three 3 claims dsev [3 6 9] [.25 .5 .25] poisson')
    assert a.bs == 1
    a.sharpen(good_enough=0, execute=False)
    assert 'finer than it needs to be' not in a.sharpen_explanation


# ------------------------------------- [Portfolio-Lattice-Bucket], step 2

def test_portfolio_takes_the_pooled_step_not_the_formula():
    """Poisson units on steps 3 and 5: pooled step 1 (bs 4 through 1.24.0)."""
    p = build('port FLB.Steps agg FLB.S3 3 claims dsev [3 6 9] poisson '
              'agg FLB.S5 3 claims dsev [5 10 15] poisson')
    assert p.bs == 1
    assert p._bs_lattice == dict(h=1.0, residual=0.0, bs=1.0)
    assert 'lattice of 1' in p.bs_explanation


SIGNED_PORT = ('port FLB.SignedPort '
               'agg FLB.SP1 3 claims dsev [-1.75 0.25 0.5] [.25 .5 .25] poisson '
               'agg FLB.SP2 2 claims dsev [-0.5 0.75] poisson '
               'agg FLB.SP3 1 claim dsev [0.25 1.5] fixed')


def test_signed_portfolio_lands_every_atom_and_shrinks_log2():
    p = build(SIGNED_PORT)
    assert p.bs == 0.25
    assert p.log2 < 16                     # stayed at the cap through 1.24.0
    x0 = float(p.density_df.index[0])
    atoms = np.array([-1.75, 0.25, 0.5, -0.5, 0.75, 1.5])
    k = (atoms - x0) / p.bs
    assert np.array_equal(k, np.round(k))
    s = p.density_df.p_total
    assert s.sum() == pytest.approx(1, abs=1e-12)
    assert float((s * s.index).sum()) == pytest.approx(p.est_m, rel=1e-12)
    assert 'lattice 1/4' in p.bs_window_df.loc['used', 'note']


@pytest.mark.slow                          # about 8 s: 200 units
def test_many_signed_units_keep_their_mass():
    """The realized origin is the sum of the unit origins, far below the window.

    200 units on ``[-1.75 .25 .5]`` realize an origin of -350 against a moment
    window of about ``[-136, 45]``. Sizing ``log2`` from the window (10) gives
    a grid on ``[-350, -94)``, below most of the mass, so the need is measured
    from the realized origin (11). It takes many units: with few, skewness
    widens the moment window past the summed support. At 1.23.0, 1000 such
    units held 17.7% of their mass ([Signed-Portfolio-Origin]).
    """
    from aggregate import Aggregate, Portfolio
    # constructed, not built: the Portfolio update sizes and runs every unit
    aggs = [Aggregate(name=f'FLB.M{i}', exp_en=1, freq_name='fixed',
                      sev_name='dhistogram', sev_xs=[-1.75, 0.25, 0.5],
                      sev_ps=[.25, .5, .25]) for i in range(200)]
    p = Portfolio('FLB.Many', aggs)
    p.update(log2=16, bs=0)
    assert p.bs == 0.25
    assert float(p.density_df.index[0]) == -350.0
    assert p.log2 == 11
    s = p.density_df.p_total
    assert s.sum() == pytest.approx(1, abs=1e-12)
    assert float((s * s.index).sum()) == pytest.approx(-0.1875 * 200, rel=1e-12)


def test_a_continuous_unit_means_no_portfolio_lattice():
    p = build('port FLB.Mixed agg FLB.MD 3 claims dsev [3 6 9] poisson '
              'agg FLB.MC 3 claims sev gamma 10 cv 1 poisson')
    assert p._bs_lattice is None


def test_a_pinned_portfolio_bs_is_left_alone():
    p = build(SIGNED_PORT, bs=1 / 16)
    assert p.bs == 1 / 16 and p.log2 == 16
    assert p._bs_lattice is None
