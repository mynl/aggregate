"""Discrete programs checked against an independent oracle ([Correctness-Oracle]).

Every program of :mod:`tests.lattice_oracle` is built and its whole distribution
compared with the oracle's, which shares no code with the sizer. The class a
program should be in comes from the specification
(``dev/note-lattice-grid-sizing.md``), not from the library:

- **Feasible**: an exact grid fits the ``log2`` cap, so the law is exact (total
  variation below 1e-4) and no ``LatticeGridWarning`` is raised. A reinsured book
  may warn with an exact output: a cover that always cedes the same amount has a
  point mass for its law, which a coarse bucket can hold while the gross atoms it
  is computed from are off the grid, and the warning is right about those.
- **Infeasible**: no exact grid fits. The verdict and the warning say so, and the
  ``log2`` the oracle gives builds the exact law.
- **No lattice**: a view has no dyadic step, which is outside the specification.

A seeded subset of about 120 programs runs in the fast suite. The full 858, and
the 718 single-unit programs built alone and as a one-unit Portfolio, are slow.
See ``dev/done/plan-lattice-grid-sizing.md``.
"""
from __future__ import annotations

import random
import warnings

import numpy as np
import pytest

from aggregate import build
from tests.lattice_oracle import (SEED, TV_TOLERANCE, compare, decl, exact, feasible,
                                  in_ntt_scope, law_of, make_sets, reference_law,
                                  total_variation)

SETS = make_sets()

#: Feasible by the specification and still wrong: they need a windowed grid and
#: carry reinsurance, and the library does not window a reinsured book. All are
#: flagged. Named one by one, so any new wrong answer fails rather than joining
#: them. These are ``[Windowed-Reinsured-Book]`` in ``dev/TODO.md``.
WINDOWED_REINSURED = {
    'agg X 100 claims dsev [1000 1001] [0.5714285714285714 0.42857142857142855] '
    'occurrence net of 25% po 1500 xs 200 fixed',
    'agg X 30 claims dsev [10002 10007] [0.5 0.5] occurrence ceded to 50% po 10000 xs 1000 fixed',
    'agg X 100 claims dsev [10002 10007] [0.6666666666666666 0.3333333333333333] '
    'occurrence ceded to 50% po 20014 xs 0 fixed',
    'agg X 1 claims dsev [10002 10007] [0.8 0.2] poisson aggregate ceded to 50000 xs 0',
    'agg X 10 claims dsev [10002 10007] [0.5714285714285714 0.42857142857142855] fixed '
    'aggregate ceded to 150000 xs 20000',
    'port P agg U0 1 claims dsev [1000 1001] poisson agg U1 10 claims dsev [10002 10007] '
    '[0.5714285714285714 0.42857142857142855] fixed aggregate ceded to 150000 xs 20000',
    'port P agg U0 30 claims dsev [1000 2000 5000] fixed aggregate net of 100000 xs 50000 '
    'agg U1 1000 claims dsev [2 4 6] fixed',
}


def _labeled(rows, key):
    return [pytest.param(key, specs, id=f'{key} {i}') for i, specs in enumerate(rows)]


def _every_program():
    return [p for key, rows in SETS.items() for p in _labeled(rows, key)]


def _subset(per_set=15):
    rng = random.Random(SEED)
    out = []
    for key, rows in SETS.items():
        picks = sorted(rng.sample(range(len(rows)), per_set))
        out += [pytest.param(key, rows[i], id=f'{key} {i}') for i in picks]
    return out


def _assert_in_its_class(key, specs):
    port = key[0] in 'GH'
    r = compare(specs, port=port)
    if r['h'] is None:
        return
    context = f"{r['prog']}: tv {r['tv']:.3g} at bs {r['bs']:g}, log2 {r['log2']}; {r['verdict']}"
    if r['feasible']:
        if r['prog'] in WINDOWED_REINSURED:
            assert 'fails' in r['verdict'] or 'infeasible' in r['verdict'], context
        else:
            assert not r['wrong'], context
            reinsured = any(s.get('occ') or s.get('agg') for s in specs)
            assert reinsured or not r['warned'], context
    elif r['wrong']:
        assert 'grid infeasible for this lattice' in r['verdict'], context
        assert r['warned'], context
        again = compare(specs, port=port, log2=int(min(r['need0'], r['needw'])))
        assert not again['wrong'], f"{context}; at the oracle's log2: {again['tv']:.3g}"
        assert not again['warned'], context


@pytest.mark.parametrize('key, specs', _subset())
def test_a_program_is_in_the_class_the_specification_gives_it(key, specs):
    _assert_in_its_class(key, specs)


@pytest.mark.slow
@pytest.mark.parametrize('key, specs', _every_program())
def test_every_program_is_in_the_class_the_specification_gives_it(key, specs):
    _assert_in_its_class(key, specs)


@pytest.mark.slow
@pytest.mark.parametrize('key, specs', [
    p for key, rows in SETS.items() if key[0] in 'ABCDEF' for p in _labeled(rows, key)])
def test_a_unit_alone_and_as_a_one_unit_portfolio_are_the_same(key, specs):
    """A one-unit Portfolio adds nothing to the mathematics: same grid, same law."""
    unit = decl(specs[0], 'U0')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = build(unit)
        p = build(f'port P {unit}')
    assert (p.bs, p.log2) == (a.bs, a.log2), unit
    assert float(p.density_df.index[0]) == float(a.density_df.index[0]), unit
    assert total_variation(law_of(a), law_of(p)) < 1e-9, unit


# ------------------------------------------------------------- the oracle itself

@pytest.mark.parametrize('key, specs', [
    p for p in _subset() if p.values[0][0] in 'AB' and in_ntt_scope(p.values[1][0])])
def test_the_enumeration_agrees_with_nttagg(key, specs):
    """``nttagg`` is the exact in-library engine and shares no code with the enumeration."""
    spec = dict(specs[0])
    ntt = reference_law(specs)
    if spec.get('probs'):
        # the oracle has to see the fractions nttagg was given
        from fractions import Fraction
        spec['probs'] = [float(Fraction(p).limit_denominator(100)) for p in spec['probs']]
    assert total_variation(ntt, exact(spec)) < 1e-12


@pytest.mark.parametrize('spec, h, from0, windowed', [
    (dict(freq=('fixed', 100), atoms=[1000, 1001]), 1, 17, 10),
    (dict(freq=('dfreq', list(range(1, 101)), None), atoms=[1000, 1001]), 1, 17, 17),
    (dict(freq=('dfreq', [1], None), atoms=[3]), 3, 1, 1),
    (dict(freq=('fixed', 1), atoms=[1000, 2000, 3000], occ=[(1.0, 1500.0, 250.0)], occ_kind='net'),
     250, 4, 4),
    (dict(freq=('fixed', 1), atoms=[-5, -4]), 1, 3, 3),
])
def test_the_needs_of_the_named_programs(spec, h, from0, windowed):
    assert feasible([spec], exact(spec))[:3] == (h, from0, windowed)


def test_the_needs_hold_a_claim_as_well_as_the_law():
    """The law of ``dsev [1000 1001]`` over 100 claims is at most 100 wide; a claim of 1001 needs 10 bits."""
    spec = dict(freq=('fixed', 100), atoms=[1000, 1001])
    law = exact(spec)
    assert max(law) - min(law) <= 100
    assert feasible([spec], law)[2] == 10


def test_a_share_off_the_dyadic_lattice_has_no_step():
    spec = dict(freq=('fixed', 3), atoms=[7, 11, 13], occ=[(0.3, 5.0, 3.0)], occ_kind='net')
    assert feasible([spec], exact(spec))[0] is None


def test_the_tolerance_is_far_below_a_real_failure():
    """The failures this exists for are at 0.5 and above."""
    assert TV_TOLERANCE <= 1e-4
    a = build('agg X 100 claims dsev [1000 1001] fixed', bs=2, log2=16)
    law = exact(dict(freq=('fixed', 100), atoms=[1000, 1001]))
    assert total_variation(law, law_of(a)) > 0.3
    assert np.isclose(sum(law.values()), 1.0)
