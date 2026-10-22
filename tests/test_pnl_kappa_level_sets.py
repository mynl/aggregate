"""κ columns condition on the whole level set ([Kappa-Level-Sets]).

``dev/done/plan-kappa-level-sets.md``. A ``κq`` cell of
:attr:`aggregate.PnL.economic_df` is ``E[row | result == x_q]``. The same
result reached by different arithmetic (the many ``(l, r)`` cells of a 2-D
occurrence joint with one ``l - r``, a corridor or retro leg) lands a few ulps
apart, so exact float equality caught an arbitrary subset of the level set.
Footing cannot detect that, since linearity holds on any subset of atoms, so
these tests measure the slice itself: its mass against the level set's, and
each cell against the brute-force tolerant conditional mean.
"""
from __future__ import annotations

import numpy as np
import pytest

from aggregate import PnL, build
from aggregate._pnl import PERCENTILE_LADDER, _atom_groups
from aggregate.config import get_settings
from tests.test_pnl_peel import _assert_column_foots, _kappa_columns

RTOL = get_settings().validation.level_set_rtol
#: The brute force's own tolerance, fixed rather than read from the settings,
#: so the comparison is not with the implementation's own choice.
BRUTE_RTOL = 1e-9

# Four atoms; atoms 1 and 2 both reach 0.3, by 0.1 + 0.2 and by 0.3 + 0.0,
# which differ in the last bit (0.30000000000000004 against 0.3).
SOURCE = (np.array([0.0, 1.0, 2.0, 3.0]), np.full(4, 0.25))
_A = np.array([0.0, 0.1, 0.3, 1.0])
_B = np.array([0.0, 0.2, 0.0, 0.0])

# The occurrence walk of the evidence table, on its default joint.
WALK = ('xpnl KLS.Walk 2000 premium less agg KLS.WalkE 2000 premium at 70% lr '
        '1000 xs 0 sev 120 * lomax 2.5 occurrence net of 750 xs 250 '
        'deposit 361.83 as Layer mixed gamma .185')


def _leg(table):
    """A leg looking its value up by atom index on the four-atom source."""
    return lambda x: table[x.astype(int)]


def _split_pnl(obligation=None):
    return PnL(name='KLS.Split', source=SOURCE, role='sell',
               consideration={'A': _leg(_A), 'B': _leg(_B)},
               obligation=obligation or {'Loss': 0.0})


@pytest.fixture(scope='module')
def walk():
    return build(WALK)


def _tolerant_slices(pnl, anchor):
    """Brute force: for each ladder point, the atoms within tol of ``x_q``."""
    v, gd = anchor.values, anchor.gd
    scale = max([1.0] + [float(np.abs(r.values).max())
                         for r in pnl._rows.values()])
    return [np.abs(v - float(gd.q(q))) <= BRUTE_RTOL * scale
            for q in PERCENTILE_LADDER]


def test_a_split_outcome_is_one_atom_and_one_slice():
    pnl = _split_pnl()
    result = pnl._grand_result
    assert result.values[1] != result.values[2], 'the fixture must split'
    gd = result.gd
    assert len(gd.x) == 3, 'atoms 1 and 2 are one outcome'
    assert gd.p[1] == pytest.approx(0.5)
    kappa = pnl.economic_df.loc[:, 'κ50']
    # both atoms of the level set, not whichever one carries the float x_q
    assert kappa[('Consideration', 'A')] == pytest.approx(0.2)
    assert kappa[('Consideration', 'B')] == pytest.approx(0.1)


def test_atom_groups_keeps_a_lone_value_exactly():
    values = np.array([0.3, 0.1 + 0.2, 1.0, 2.0])
    x, p, lo, hi = _atom_groups(values, np.full(4, 0.25), 1e-9)
    assert x[1:].tolist() == [1.0, 2.0]
    assert lo[0] == 0.3 and hi[0] == 0.1 + 0.2
    assert lo[0] <= x[0] <= hi[0]
    x0, p0, _, _ = _atom_groups(values, np.full(4, 0.25), 0.0)
    assert len(x0) == 4, 'zero tolerance groups by exact value'


def test_atom_groups_gives_a_massless_group_its_smallest_value():
    values = np.array([1.0, 1.0 + 1e-15, 5.0])
    x, p, _, _ = _atom_groups(values, np.array([0.0, 0.0, 1.0]), 1e-9)
    assert x.tolist() == [1.0, 5.0] and p.tolist() == [0.0, 1.0]


def test_the_walk_slices_the_whole_level_set(walk):
    """Each slice holds the level set's mass and each cell its mean."""
    probs = walk._probs
    econ = walk.economic_df[_kappa_columns(walk)]
    for j, near in enumerate(_tolerant_slices(walk, walk._grand_result)):
        gd = walk._grand_result.gd
        mask = walk._grand_result.level_set(float(gd.q(PERCENTILE_LADDER[j])))
        assert probs[mask].sum() == pytest.approx(probs[near].sum(), rel=1e-12)
        pw = probs[near]
        for i, row in enumerate(walk._rows.values()):
            brute = (row.values[near] * pw).sum() / pw.sum()
            assert econ.iloc[i, j] == pytest.approx(brute, rel=1e-9, abs=1e-9)


def test_the_walk_splits_its_level_sets(walk):
    """The fixture exercises the fix: some ladder point's level set spans
    several floats, so exact equality would have caught part of it."""
    v = walk._grand_result.values
    spans = [len(np.unique(v[near]))
             for near in _tolerant_slices(walk, walk._grand_result)]
    assert max(spans) > 1


def test_the_gross_basis_slices_the_whole_level_set(walk):
    """The waterfall's gross anchor goes through the same ladder."""
    gross = walk._group_result_rows[0]
    ladder = walk._kappa_ladder(gross)
    probs = walk._probs
    for j, near in enumerate(_tolerant_slices(walk, gross)):
        pw = probs[near]
        for label, row in walk._rows.items():
            brute = (row.values[near] * pw).sum() / pw.sum()
            assert ladder[label][j] == pytest.approx(brute, rel=1e-9, abs=1e-9)


def test_every_exact_distribution_is_separated(walk):
    """No two atoms of a row's exact GD lie within tol, and the mass is 1."""
    for row in walk._rows.values():
        if row.bs:
            continue
        gd = row.gd
        assert gd.p.sum() == pytest.approx(1.0, abs=1e-12)
        assert np.all(np.diff(gd.x) > walk._level_tol), row.label


def test_the_tolerance_scales_with_the_ledger(walk):
    scale = max(float(np.abs(r.values).max()) for r in walk._rows.values())
    assert walk._level_tol == pytest.approx(RTOL * scale)


def test_prob_eq_0_counts_a_break_even_split_by_rounding():
    """Atom 1 nets 0.1 + 0.2 - 0.3 = 5.6e-17, atom 2 nets 0.3 - 0.3 = 0."""
    pnl = _split_pnl(obligation={'Loss': _leg(np.array([1.0, 0.3, 0.3, 0.0]))})
    v = pnl._grand_result.values
    assert v[1] != 0.0 and v[2] == 0.0, 'the fixture must split'
    assert pnl.prob_eq_0 == pytest.approx(0.5)


@pytest.mark.parametrize('column', ['EX', 'κ01', 'κ50', 'κ99'])
def test_the_walk_still_foots(walk, column):
    _assert_column_foots(walk, column)
