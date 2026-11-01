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
from aggregate.constants import LatticeGridWarning, Validation, reset_warn_once
from aggregate.utilities import (LATTICE_DEPTH, LATTICE_RTOL,
                                 _dyadic_lattice_step, _lattice_candidates,
                                 silence_warnings)


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


# --------------------------------------------- [Lattice-Views]: every view on the grid

def _law(ob):
    """The nonzero outcomes of ``p_total``."""
    p = ob.density_df.p_total
    return p[p > 1e-12]


def test_net_of_an_occurrence_layer_is_on_the_grid():
    """Net is 250, 500 or 1500. On the gross step of 1000 it was 0, 1000 or 2000."""
    a = build('agg LV.Net 1 claims dsev [1000 2000 3000] '
              'occurrence net of 1500 xs 250 fixed')
    # the step of the views is 250; the nice divisor of 250 is 50
    assert a._severity_lattice() == 250
    assert a.bs == 50
    law = _law(a)
    assert list(law.index) == [250, 500, 1500]
    assert np.allclose(law, 1 / 3)


def test_ceded_to_an_occurrence_layer_is_on_the_grid():
    a = build('agg LV.Ceded 1 claims dsev [1000 2000 3000] '
              'occurrence ceded to 1500 xs 250 fixed')
    law = _law(a)
    assert list(law.index) == [750, 1500]
    assert np.allclose(law, [1 / 3, 2 / 3])


def test_a_layered_dsev_is_on_the_step_of_its_payments():
    """``Frees2018a.Revised`` sat at bs 6 with payments 2, 8 and 20 ([Layered-Dsev-Lattice])."""
    a = build('agg LV.Layer 1 claim 1500 xs 250 dsev [1000 2000 3000] fixed')
    assert a._severity_lattice() == 750
    law = _law(a)
    assert list(law.index) == [750, 1500]
    assert np.allclose(law, [1 / 3, 2 / 3])


@pytest.mark.parametrize('layer, kept', [
    ('', [1, 4, 8]),
    ('5 xs 3', [1, 5]),
    ('2 xs 3', [1, 2]),
])
def test_conditional_layer_atoms(layer, kept):
    a = build(f'agg LV.Atoms 1 claim {layer} dsev [1 4 8] fixed')
    assert sorted(a.sevs[0]._layered_atoms()) == kept


def test_unconditional_layer_keeps_the_zero_atom():
    a = build('agg LV.Bang 1 claim 5 xs 3 dsev [1 4 8] ! fixed')
    assert sorted(a.sevs[0]._layered_atoms()) == [0, 1, 5]


def test_a_franchise_layer_pays_the_loss():
    a = build('agg LV.Franchise 1 claim 100 xs 3 franchise dsev [1 4 8] fixed')
    assert sorted(a.sevs[0]._layered_atoms()) == [4, 8]


def test_the_views_pool_each_layer_and_the_net():
    a = build('agg LV.Pool 1 claim dsev [1000 2000] occurrence net of 50% po 500 xs 250 fixed')
    pool = set(a._severity_views())
    # gross, the 50% layer, and the net of it
    assert {1000, 2000, 250, 750, 1750} <= pool


def test_a_cover_attachment_widens_the_lattice_not_the_search():
    """The attachment and limit enter reduced modulo the largest gross atom."""
    a = build('agg LV.Cover 3 claims dsev [8 10] fixed aggregate net of 3 xs 1000000')
    assert a._severity_lattice() == 1
    assert max(a._severity_views()) == 10


def test_a_share_off_the_dyadic_lattice_leaves_no_lattice():
    a = build('agg LV.Share 3 claims dsev [7 11 13] occurrence net of 30% po 5 xs 3 fixed')
    assert a._severity_lattice() is None
    assert 'exact_discrete' not in a.bs_window_df.index


def test_the_exact_row_holds_the_gross_when_it_fits():
    """Sized on its net support, 56, the gross up to 80 wrapped at log2 6, and the verdict said so."""
    a = build('agg LV.Fix dfreq [4] dsev [1:20] occurrence net of 6 xs 6')
    row = a.bs_window_df.loc['exact_discrete']
    assert row['x_max'] == 80 and row['log2'] == 7
    assert 'subject fails' not in a.validation_description


def test_the_exact_row_falls_back_to_the_view_when_the_gross_cannot_fit():
    """Gross needs log2 15 at bs 1, net (9 or 10 a claim) needs 7, and a claim of 2000 needs 11."""
    a = build('agg LV.Tight dfreq [10] dsev [1999 2000] occurrence net of 1990 xs 5', log2=12)
    row = a.bs_window_df.loc['exact_discrete']
    assert row['x_max'] == 2000                      # the gross hull, above the net support
    assert (a.bs, a.log2) == (1, 11)
    law = _law(a)
    assert law.index.min() == 90 and law.index.max() == 100


def test_a_portfolio_pools_its_units_views():
    p = build('port LV.Port agg LV.U1 1 claim dsev [1000 2000] occurrence net of 1000 xs 250 fixed '
              'agg LV.U2 1 claim dsev [1000 2000] fixed')
    assert _bucket_is_exact(p, 250) == (True, 250.0)


# ------------------- [Lattice-Infeasible-Warning]: a lattice the cap cannot hold

AWK = 'agg LW.AWK dfreq [1:100] dsev [1000 1001]'


def _lattice_warnings(build_call):
    """The object a call builds, and the ``LatticeGridWarning`` texts it raised."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        ob = build_call()
    return ob, [str(w.message) for w in rec if issubclass(w.category, LatticeGridWarning)]


def test_infeasible_builds_warns_and_says_so_in_the_verdict():
    a, texts = _lattice_warnings(lambda: build(AWK))
    assert (a.bs, a.log2) == (2, 16)
    assert a._lattice_feasibility == dict(h=1.0, bs=2.0, log2=16, log2_required=17)
    assert len(texts) == 1
    assert ('LW.AWK: the severity atoms are on the lattice of step 1 and the grid '
            'needs log2 >= 17 at bs 1; log2 16 was given, so bs 2 is used and the '
            'atoms do not sit on the grid. Pass build(..., log2=17) or add '
            'hints{log2=17} to the declaration.') in texts[0]
    assert a.validation_description.startswith(
        'grid infeasible for this lattice (needs log2 17 at bs 1); fails')
    assert a.valid & Validation.INFEASIBLE


@pytest.mark.parametrize('kw, program', [
    (dict(log2=17), AWK),
    (dict(), AWK + ' hints{log2=17}'),
])
def test_the_log2_the_warning_gives_builds_exactly(kw, program):
    a, texts = _lattice_warnings(lambda: build(program, **kw))
    assert (a.bs, a.log2) == (1, 17)
    assert not texts
    assert a._lattice_feasibility is None
    assert a.validation_description == 'not unreasonable'


def test_the_reading_is_in_every_report():
    a, _ = _lattice_warnings(lambda: build(AWK))
    assert 'lattice 1 needs log2 17, log2 16 given, bs 2 off it' in a.bs_description
    assert 'an exact grid exists, and it needs log2 17 at bs 1' in a.bs_explanation
    assert 'an exact grid exists' in a.validation_explanation
    assert 'too fine for the log2 cap (needs log2 17), bs 2 off it' in \
        a.bs_window_df.loc['exact_discrete', 'note']


def test_the_warning_is_once_per_name_step_and_log2():
    _, first = _lattice_warnings(lambda: build(AWK))
    _, again = _lattice_warnings(lambda: build(AWK))
    _, other = _lattice_warnings(lambda: build(AWK, log2=15))
    assert (len(first), len(again), len(other)) == (1, 0, 1)
    reset_warn_once()
    assert len(_lattice_warnings(lambda: build(AWK))[1]) == 1


def test_the_warning_can_be_silenced():
    with warnings.catch_warnings():
        warnings.simplefilter('error', LatticeGridWarning)
        silence_warnings(LatticeGridWarning)
        a = build(AWK)
    assert a.valid & Validation.INFEASIBLE


def test_a_feasible_lattice_book_has_no_reading():
    a, texts = _lattice_warnings(lambda: build('agg LW.Dice dfreq [3] dsev [1:6]'))
    assert a._lattice_feasibility is None and not texts
    assert not a.valid & Validation.INFEASIBLE


def test_a_pinned_bucket_is_the_users_opt_out():
    a, texts = _lattice_warnings(lambda: build(AWK, bs=2))
    assert a._lattice_feasibility is None and not texts
    assert 'grid infeasible' not in a.validation_description


def test_an_update_to_a_pinned_bucket_clears_the_reading():
    a, _ = _lattice_warnings(lambda: build(AWK))
    a.update(log2=17, bs=1)
    assert a._lattice_feasibility is None
    assert a.validation_description == 'not unreasonable'


def test_a_continuous_severity_has_no_reading():
    a, texts = _lattice_warnings(lambda: build('agg LW.Gamma 10 claims sev gamma 10 cv 1 poisson'))
    assert a._lattice_feasibility is None and not texts


def test_a_reference_severity_has_no_reading():
    """Its atoms are a materialized law, and the snap keeps the grids commensurable."""
    build('agg LW.Inner dfreq [1:10] dsev [1:10] hints{log2=10; bs=1}')
    a, texts = _lattice_warnings(lambda: build('agg LW.Outer 100000 claims sev agg.LW.Inner poisson',
                                               log2=10))
    assert a.bs > 1
    assert a._lattice_feasibility is None and not texts


def test_the_depth_follows_a_cap_above_16():
    """LATTICE_DEPTH alone refuses the step, and the cap can hold it."""
    assert _dyadic_lattice_step([1, 200000]) is None
    a = build('agg LW.Wide 1 claims dsev [1 200000] fixed', log2=18)
    assert (a.bs, a.log2) == (1, 18)
    assert a.validation_description == 'not unreasonable'


def test_the_depth_does_not_fall_below_16_at_a_low_cap():
    """At depth 10 this lattice is invisible: no exact grid and no warning."""
    assert _dyadic_lattice_step([10002, 10007], depth=10) is None
    a, texts = _lattice_warnings(lambda: build('agg LW.Low dfreq [1:100] dsev [10002 10007]',
                                               log2=10))
    assert a._lattice_feasibility['h'] == 1
    assert len(texts) == 1


def test_a_low_cap_names_the_window_when_that_needs_less():
    """From 0 it needs log2 20. The window needs 14, for the claim of 10007."""
    a, texts = _lattice_warnings(lambda: build('agg LW.Near 100 claims dsev [10002 10007] fixed',
                                               log2=9))
    assert a._lattice_feasibility['log2_required'] == 14
    assert 'log2=14' in texts[0]
    assert build('agg LW.Near 100 claims dsev [10002 10007] fixed', log2=14).bs == 1


def test_a_one_unit_portfolio_warns_once_as_itself():
    p, texts = _lattice_warnings(lambda: build(f'port LW.P1 {AWK}'))
    assert (p.bs, p.log2) == (2, 16)
    assert len(texts) == 1 and texts[0].startswith('LW.P1: ')
    assert p._lattice_feasibility['log2_required'] == 17
    assert p.validation_description.startswith('grid infeasible for this lattice')
    assert p.agg_list[0]._lattice_feasibility is None


def test_a_portfolio_of_several_units_warns_as_itself():
    program = ('port LW.P2 agg LW.A 1 claims 1500 xs 0 dsev [500 1000 2000] fixed '
               'agg LW.B dfreq [1:100] dsev [1000 1001]')
    p, texts = _lattice_warnings(lambda: build(program))
    assert len(texts) == 1 and texts[0].startswith('LW.P2: ')
    assert p._lattice_feasibility['log2_required'] == 17
    assert 'needs log2 17' in p.bs_window_df.loc['used', 'note']
    assert 'lattice 1 needs log2 17' in p.bs_description
    assert 'an exact grid exists' in p.bs_explanation
    assert all(a._lattice_feasibility is None for a in p.agg_list)
    exact, again = _lattice_warnings(lambda: build(program, log2=17))
    assert exact.bs == 1 and not again
    assert 'grid infeasible' not in exact.validation_description


def test_a_pinned_portfolio_has_no_reading():
    p, texts = _lattice_warnings(lambda: build(f'port LW.P3 {AWK}', bs=2))
    assert p._lattice_feasibility is None and not texts


# ---------------------------- the named programs of [Correctness-Oracle], section LO

@pytest.mark.parametrize('program, bs, log2, x0, outcomes, verdict', [
    ('agg LO.T 100 claims dsev [1000 1001] fixed', 1, 10, 99640, None, 'not unreasonable'),
    ('port LO.P agg LO.A dfreq [1] dsev [3]', 1, 2, 0, [3], 'not unreasonable'),
    ('agg LO.AWK dfreq [1:100] dsev [1000 1001]', 2, 16, 0, None,
     'grid infeasible for this lattice (needs log2 17 at bs 1); fails sev cv, agg skew'),
    ('agg LO.Net 1 claims dsev [1000 2000 3000] occurrence net of 1500 xs 250 fixed',
     50, 6, 0, [250, 500, 1500], 'reinsurance; subject not unreasonable'),
])
def test_the_programs_that_started_the_lattice_study(program, bs, log2, x0, outcomes, verdict):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ob = build(program)
    assert (ob.bs, ob.log2, float(ob.density_df.index[0])) == (bs, log2, x0)
    assert ob.validation_description == verdict
    if outcomes is not None:
        assert list(_law(ob).index) == outcomes


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

def test_portfolio_takes_the_pooled_step():
    """Poisson units on steps 3 and 5: pooled step 1 (bs 4 at 1.23.0)."""
    p = build('port FLB.Steps agg FLB.S3 3 claims dsev [3 6 9] poisson '
              'agg FLB.S5 3 claims dsev [5 10 15] poisson')
    assert p.bs == 1
    assert p._bs_lattice == dict(h=1.0, residual=0.0, bs=1.0)
    assert 'lattice of 1' in p.bs_explanation


def test_portfolio_units_whose_buckets_do_not_divide():
    """Steps 2 and 5: the finest unit bucket, 2, misses the 5 lattice.

    At 1.24.0 the resolution formula took ``bs`` 2 and put the atoms at 5 and
    15 between grid points; the pooled step is 1.
    """
    p = build('port FLB.Steps25 agg FLB.S2 3 claims dsev [2 4 6] poisson '
              'agg FLB.S5b 3 claims dsev [5 10 15] poisson')
    assert p.bs == 1
    x0 = float(p.density_df.index[0])
    k = (np.array([2, 4, 6, 5, 10, 15.0]) - x0) / p.bs
    assert np.array_equal(k, np.round(k))


SIGNED_PORT = ('port FLB.SignedPort '
               'agg FLB.SP1 3 claims dsev [-1.75 0.25 0.5] [.25 .5 .25] poisson '
               'agg FLB.SP2 2 claims dsev [-0.5 0.75] poisson '
               'agg FLB.SP3 1 claim dsev [0.25 1.5] fixed')


def test_signed_portfolio_lands_every_atom_and_shrinks_log2():
    p = build(SIGNED_PORT)
    assert p.bs == 0.25
    assert p.log2 < 16                     # 0.25 at the cap 16 at 1.24.0
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


# ------------------------------------- [Bivariate-Dyadic-Lattice], step 3

def _bv(axis):
    return build(f'bivariate FLB.BV dfreq [1 2] [.5 .5] dbvsev {axis} [0 1 2]')


def _margins_match(b):
    S = b._S.toarray() if hasattr(b._S, 'toarray') else np.asarray(b._S)
    g0, g1 = b._gs
    return (np.allclose(S.sum(axis=1), g0, rtol=0, atol=1e-14)
            and np.allclose(S.sum(axis=0), g1, rtol=0, atol=1e-14))


@pytest.mark.parametrize('axis, bs', [('[0 1 2]', 1.0), ('[0 0.25 0.75]', 0.25)])
def test_lattice_axes_keep_their_bucket(axis, bs):
    b = _bv(axis)
    assert b.bs == [bs, 1.0]
    assert b._on_lattice == [True, True]
    assert _margins_match(b)


@pytest.mark.parametrize('axis', ['[0 0.1 0.2]', '[0.1 0.2 0.3]', '[-0.3 0 0.2]'])
def test_decimal_axis_is_free_dyadic_and_scattered(axis):
    """Through 1.25.0 a float gcd gave bs 0.1 or 0.09999999999999998."""
    b = _bv(axis)
    num, den = b.bs[0].as_integer_ratio()
    assert num == 1 and den & (den - 1) == 0          # a power of two
    assert b._on_lattice == [False, True]
    assert _margins_match(b)
    assert float(np.asarray(b.density).sum()) == pytest.approx(1, abs=1e-12)


def test_noisy_decimal_axis_builds():
    """A float gcd of about 1e-7 never finished building through 1.25.0."""
    b = _bv('[0 0.1 0.3000001]')
    assert len(b.axis_xs[0]) <= 2 ** 16
    assert _margins_match(b)
