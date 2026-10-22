"""A ``fixed`` frequency refuses a claim count that is not a whole number.

``[Input-Guards]``. The generating function of a fixed count ``n`` is ``z^n``;
for a fractional ``n`` the FFT result is not a distribution (it carries negative
probabilities and misses the mean), so the build raises instead. The count can
be stated directly, or derived from an expected loss or from premium times loss
ratio, and the message names whichever route produced it. A ``dfreq`` support
with a fractional outcome has the same defect through the other door, a
fractional power in ``sum p_k z^k``, and is refused when the frequency is built,
as is a negative outcome, which validation would otherwise clamp onto 0.
"""
from __future__ import annotations

import numpy as np
import pytest

from aggregate import build


@pytest.mark.parametrize('program, count, route', [
    ('agg GF.Claims 2.5 claims dsev [1 2] fixed', '2.5',
     'the exposure clause states 2.5 claims'),
    ('agg GF.Loss 7 loss dsev [1 2] fixed', '4.66667',
     'derived from the expected loss: 7 / mean severity 1.5'),
    ('agg GF.Prem 10 premium at 0.7 lr dsev [1 2] fixed', '4.66667',
     'derived from the premium: premium 10 x loss ratio 0.7'),
], ids=['claims', 'loss', 'premium'])
def test_fractional_fixed_count_is_refused(program, count, route):
    with pytest.raises(ValueError) as info:
        build(program)
    msg = str(info.value)
    assert 'whole number of claims' in msg
    assert f'claim count is {count}' in msg
    assert route in msg
    assert 'dfreq' in msg and 'poisson' in msg


def test_suggested_dfreq_has_the_stated_mean():
    """The two point ``dfreq`` named in the message builds at the same mean."""
    with pytest.raises(ValueError, match=r'dfreq \[4 5\] \[0\.333333 0\.666667\]'):
        build('agg GF.Loss 7 loss dsev [1 2] fixed')
    a = build('agg GF.Suggested dfreq [4 5] [1/3 2/3] dsev [1 2]')
    assert a.est_m == pytest.approx(7.0, rel=1e-12)
    assert np.all(a.agg_density >= -1e-12)


@pytest.mark.parametrize('program, n', [
    ('agg GF.Whole 3 claims dsev [1 2] fixed', 3),
    ('agg GF.WholeLoss 6 loss dsev [1 2] fixed', 4),
    ('agg GF.WholePrem 10 premium at 0.6 lr dsev [1 2] fixed', 4),
    ('agg GF.Ulps 3.0000000001 claims dsev [1 2] fixed', 3),
], ids=['claims', 'loss', 'premium', 'within-tolerance'])
def test_whole_fixed_count_builds(program, n):
    """A count within ``1e-9`` of a whole number builds; its residue is round off."""
    a = build(program)
    assert a.n == pytest.approx(n, rel=1e-9)
    assert np.all(a.agg_density >= -1e-10)
    assert a.est_m == pytest.approx(1.5 * a.n, rel=1e-9)


def test_other_frequencies_accept_a_fractional_count():
    """Only ``fixed`` is guarded: an expected count may be fractional."""
    a = build('agg GF.Poisson 2.5 claims dsev [1 2] poisson')
    assert a.n == pytest.approx(2.5)


@pytest.mark.parametrize('program, shown', [
    ('agg DF.One dfreq [2.5] dsev [1 2]', '2.5'),
    ('agg DF.Mixed dfreq [1 2.5 4] [.2 .3 .5] dsev [1]', '2.5'),
    ('agg DF.Two dfreq [0.5 1.5] dsev [1]', '0.5, 1.5'),
], ids=['single', 'one-of-three', 'two'])
def test_fractional_dfreq_outcome_is_refused(program, shown):
    with pytest.raises(ValueError) as info:
        build(program)
    msg = str(info.value)
    assert 'must be non-negative whole numbers' in msg
    assert 'fractional count' in msg
    assert f'support includes {shown}.' in msg
    assert 'poisson' in msg


@pytest.mark.parametrize('program, shown', [
    ('agg N.One dfreq [-1 2] dsev [1 2]', '-1'),
    ('agg N.Two dfreq [-2 -1 3] [.2 .3 .5] dsev [1]', '-2, -1'),
], ids=['one', 'two'])
def test_negative_dfreq_outcome_is_refused(program, shown):
    """A negative outcome used to be clamped onto 0, moving the mean silently."""
    with pytest.raises(ValueError) as info:
        build(program)
    msg = str(info.value)
    assert 'must be non-negative whole numbers' in msg
    assert f'support includes {shown}.' in msg
    assert 'negative count is not a number of claims' in msg
    assert 'fractional count' not in msg


def test_negative_and_fractional_dfreq_outcomes_give_both_reasons():
    with pytest.raises(ValueError) as info:
        build('agg N.Both dfreq [-1 1.5] dsev [1]')
    msg = str(info.value)
    assert 'support includes -1, 1.5.' in msg
    assert 'fractional count' in msg and 'negative count' in msg


def test_fractional_dfreq_refused_in_the_constructor():
    """The guard sits on the empirical frequency, so the Python API is covered too."""
    from aggregate._frequency import Frequency
    with pytest.raises(ValueError, match='support includes 2.5'):
        Frequency('empirical', [1, 2.5], [0.5, 0.5], False, 0)


@pytest.mark.parametrize('program', [
    'agg DF.Whole dfreq [1 2 3] dsev [1 2]',
    'agg DF.Ulps dfreq [3.0000000001] dsev [1 2]',
    'agg DF.Zero dfreq [0 1 2] dsev [1 2]',
], ids=['whole', 'within-tolerance', 'zero'])
def test_whole_dfreq_outcomes_build(program):
    a = build(program)
    assert np.all(a.agg_density >= -1e-10)
    assert a.agg_density.sum() == pytest.approx(1.0, abs=1e-10)
