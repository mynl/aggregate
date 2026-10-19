"""A ``fixed`` frequency refuses a claim count that is not a whole number.

``[Input-Guards]``. The generating function of a fixed count ``n`` is ``z^n``;
for a fractional ``n`` the FFT result is not a distribution (it carries negative
probabilities and misses the mean), so the build raises instead. The count can
be stated directly, or derived from an expected loss or from premium times loss
ratio, and the message names whichever route produced it.
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
