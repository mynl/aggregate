"""Tests for ``Aggregate.create_frequency`` / ``Portfolio.create_frequency``.

These materialize the claim-count distribution as a first-class ``Aggregate``
(or ``Portfolio`` of counts) via the ``dsev [1]`` point-mass trick. The
invariant under test: the built count distribution reproduces the parent's
*frequency* moments exactly, regardless of severity, layers, or reinsurance on
the parent.
"""

import numpy as np
import pytest

from aggregate import build, Aggregate


def _freq_moments(agg):
    """Return (mean, cv, skew) of the parent's frequency distribution."""
    f = agg.stats_df['mixed']['freq']
    return float(f['ex1']), float(f['cv']), float(f['skew'])


def _count_moments(freq_agg):
    """Return (mean, cv, skew) of a materialized count distribution."""
    return float(freq_agg.agg_m), float(freq_agg.agg_cv), float(freq_agg.agg_skew)


@pytest.mark.parametrize('program', [
    'agg P 100 claims sev lognorm 50 cv 1 poisson',
    'agg G 100 claims sev lognorm 50 cv 2 mixed gamma 0.5',
    'agg N 100 claims sev lognorm 50 cv 1 negbin 2',
    'agg ZM 100 claims sev lognorm 50 cv 1 poisson zm 0.3',
])
def test_count_matches_parent_frequency(program):
    """The count distribution reproduces the parent frequency moments."""
    a = build(program)
    fa = a.create_frequency()
    assert isinstance(fa, Aggregate)
    assert fa.name == f'{a.name}.freq'
    np.testing.assert_allclose(_count_moments(fa), _freq_moments(a), rtol=1e-3)


def test_layers_and_reinsurance_are_stripped():
    """Layers and occ/agg reinsurance must not reach the count distribution.

    A point-mass severity through ``occurrence net of 50 xs 0`` would net every
    unit to 0; the count must instead equal the plain poisson count.
    """
    plain = build('agg P 100 claims sev lognorm 50 cv 1 poisson').create_frequency()
    fancy = build(
        'agg R 100 claims 50 xs 0 sev lognorm 50 cv 1 '
        'occurrence net of 30 xs 10 poisson aggregate ceded to 20 xs 0'
    ).create_frequency()
    np.testing.assert_allclose(_count_moments(fancy), _count_moments(plain), rtol=1e-3)
    # the rendered program carries neither a layer nor a reins clause
    assert 'xs' not in fancy.program
    assert 'net of' not in fancy.program and 'ceded to' not in fancy.program


def test_exposure_derived_count_collapses_to_n():
    """``500 loss`` with a mean-50 severity is 10 claims, not 500."""
    a = build('agg L 500 loss sev lognorm 50 cv 1 poisson')
    fa = a.create_frequency()
    assert fa.agg_m == pytest.approx(a.n, rel=1e-6)
    assert fa.agg_m == pytest.approx(10.0, rel=1e-6)


def test_empirical_dfreq_is_reproduced():
    """A ``dfreq`` frequency is itself the count distribution."""
    a = build('agg D dfreq [1 2 3] [.5 .3 .2] dsev [10 20]')
    fa = a.create_frequency()
    assert 'dfreq' in fa.program
    np.testing.assert_allclose(_count_moments(fa), _freq_moments(a), rtol=1e-6)


def test_portfolio_total_is_total_count():
    """The portfolio total mean equals the sum of unit expected counts."""
    port = build(
        'port Book '
        'agg A 100 claims sev lognorm 50 cv 1 poisson '
        'agg B 50 claims sev lognorm 30 cv 2 mixed gamma 0.4'
    )
    pf = port.create_frequency()
    assert pf.name == 'Book.freq'
    assert pf.unit_names == ['A.freq', 'B.freq']
    total_n = sum(a.n for a in port.agg_list)
    assert pf.agg_m == pytest.approx(total_n, rel=1e-3)
    per_unit = {u.name: u.agg_m for u in pf.agg_list}
    assert per_unit['A.freq'] == pytest.approx(100.0, rel=1e-3)
    assert per_unit['B.freq'] == pytest.approx(50.0, rel=1e-3)


def test_programmatic_object_without_program_raises():
    """An object built without a DecL program has nothing to re-parse."""
    a = Aggregate(name='X', exp_en=10, sev_name='lognorm', sev_mean=50,
                  sev_cv=1, freq_name='poisson')
    with pytest.raises(ValueError, match='requires a DecL program'):
        a.create_frequency()
