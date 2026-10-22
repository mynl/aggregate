"""Disappearing deductible: ``y xs d1 to d2 disappearing`` (#9).

``[Disappearing-Deductible]``. The retention falls linearly from ``d1`` at a
loss of ``d1`` to zero at a loss of ``d2``, and the layer pays
``min(X - r(X), y)``: on ``(d1, d2]`` that is ``k (X - d1)`` with
``k = d2 / (d2 - d1)``, above ``d2`` the whole loss, each capped at ``y``. The
engine is the retention family built for the franchise
(``tests/test_franchise_deductible.py`` checks its moments and distribution
functions at ``d1 < d2``); this file covers the DecL surface and the aggregate.

Everything is checked against closed forms, never a captured snapshot.
"""
from __future__ import annotations

import numpy as np
import pytest

from aggregate import build
from aggregate.decl_writer import spec_to_decl


def _parse(program):
    return build.parser.parse(build.lexer.tokenize(program))


# ---------------------------------------------------------------------------
# DecL
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('program, expected', [
    ('agg D 100 claims inf xs 25 to 50 disappearing sev lognorm 100 cv 2 poisson',
     50.0),
    ('agg D 100 claims [500 1000] xs [25 50] to [40 80] disappearing '
     'sev lognorm 100 cv 2 poisson', [40.0, 80.0]),
    ('agg D 100 claims [500 1000] xs [25 50] to 80 disappearing '
     'sev lognorm 100 cv 2 poisson', 80.0),
    ('agg D 100 claims [500 1000] xs 50 to [60 70] disappearing '
     'sev lognorm 100 cv 2 poisson', [60.0, 70.0]),
    ('agg D 100 claims 1000 xs 25 to 50 disappearing as "DD" '
     'sev lognorm 100 cv 2 poisson', 50.0),
], ids=['scalar', 'vector', 'scalar-end-vector-layers',
        'vector-end-scalar-attachment', 'label'])
def test_disappearing_parses_and_round_trips(program, expected):
    """``exp_retention_end`` holds ``d2`` and the unparser restores it."""
    kind, name, spec = _parse(program)
    np.testing.assert_array_equal(spec['exp_retention_end'], expected)
    text = spec_to_decl(spec, kind=kind, name=name)
    assert 'disappearing' in text
    kind2, name2, spec2 = _parse(text)
    for key in ('exp_attachment', 'exp_limit', 'exp_retention_end',
                'label_map'):
        assert str(spec2.get(key)) == str(spec.get(key)), key


def test_retention_end_at_the_attachment_is_the_franchise():
    """``to d1 disappearing`` is the same spec as ``franchise``."""
    _, _, d = _parse('agg D 10 claims 1000 xs 50 to 50 disappearing '
                     'sev lognorm 100 cv 2 poisson')
    _, _, f = _parse('agg D 10 claims 1000 xs 50 franchise '
                     'sev lognorm 100 cv 2 poisson')
    assert d['exp_retention_end'] == f['exp_retention_end'] == 50
    assert 'franchise' in spec_to_decl(d, kind='agg', name='D')


@pytest.mark.parametrize('layers', ['[500 1000] xs [25 50]', '[500 1000] xs 25'],
                         ids=['vector-attachment', 'scalar-attachment'])
def test_vector_retention_end_must_conform(layers):
    """The count to match is the broadcast layer count, not the attachments."""
    with pytest.raises(ValueError, match='has 3 entries but there are 2 layers'):
        build(f'agg D 100 claims {layers} to [40 80 90] disappearing '
              'sev lognorm 100 cv 2 poisson')


def test_retention_end_below_the_attachment_is_refused():
    with pytest.raises(ValueError, match='at least the attachment'):
        build('agg D 100 claims 1000 xs 50 to 25 disappearing '
              'sev lognorm 100 cv 2 poisson')


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def test_discrete_disappearing_is_exact():
    """``100 xs 20 to 40`` on atoms 10, 30, 60, 100: ``k = 2``.

    The loss 10 is below the attachment. 30 is on the ramp and pays
    ``2 (30 - 20) = 20``; 60 is above ``d2`` and pays in full; 100 pays the
    limit, 100. Conditional on a loss above 20 the mean is 60.
    """
    a = build('agg D.Exact dfreq [1] 100 xs 20 to 40 disappearing '
              'dsev [10 30 60 100]')
    support = a.xs[a.agg_density > 1e-12]
    np.testing.assert_allclose(support, [20.0, 60.0, 100.0])
    assert a.est_m == pytest.approx(60.0, rel=1e-12)
    assert a.sevs[0].moms()[1] == pytest.approx((400 + 3600 + 1e4) / 3,
                                                rel=1e-12)


def test_disappearing_lies_between_ordinary_and_franchise():
    """A shorter ramp gives back more of the retention: means are ordered."""
    body = '100 claims 1000 xs 50 {} sev lognorm 100 cv 2 poisson'
    means = [build(f'agg D.M{i} ' + body.format(m)).est_m for i, m in enumerate(
        ['', 'to 400 disappearing', 'to 100 disappearing', 'franchise'])]
    assert means == sorted(means)
    assert len(set(np.round(means, 6))) == 4


def test_aggregate_matches_its_theoretical_moments():
    """Within the bucket error an ordinary layer has here (5e-6 at ``bs = 1``)."""
    a = build('agg D.Theory 100 claims [500 1000] xs [25 50] to [40 80] '
              'disappearing sev lognorm 100 cv 2 poisson')
    theory = float(a.stats_df.loc[('agg', 'mean'), 'mixed'])
    assert a.est_m == pytest.approx(theory, rel=1e-5)
    assert a.valid.passes


def test_vector_retention_end_broadcasts_over_a_scalar_attachment():
    """``[500 1000] xs 50 to [60 70]`` is two layers sharing one attachment.

    The engine broadcasts the attachment and the retention end to the layers,
    so each severity carries its own ``d2`` and has the moments of the same
    layer built alone.
    """
    a = build('agg D.Broadcast 100 claims [500 1000] xs 50 to [60 70] '
              'disappearing sev lognorm 100 cv 2 poisson')
    assert [s.retention_end for s in a.sevs] == [60.0, 70.0]
    for sev, (y, d2) in zip(a.sevs, [(500, 60), (1000, 70)]):
        alone = build(f'agg D.Alone{y} 100 claims {y} xs 50 to {d2} '
                      'disappearing sev lognorm 100 cv 2 poisson').sevs[0]
        np.testing.assert_allclose(sev.moms(), alone.moms(), rtol=1e-12)
    theory = float(a.stats_df.loc[('agg', 'mean'), 'mixed'])
    assert a.est_m == pytest.approx(theory, rel=1e-5)
