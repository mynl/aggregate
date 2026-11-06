"""``calibrate_distortions(..., extended=True)``: the standard set with ``tsallis`` (2.1.0a5).

The cases follow test 12 of ``dev/done/plan-tsallis-distortion-and-gini.md``
for ``[Extended-Calibration-Set]``. Test 14, the envelope chart, is in
``tests/test_chart_bounds.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from aggregate import build
from aggregate._pricing import (DEFAULT_CALIBRATION_DISTORTIONS,
                                EXTENDED_CALIBRATION_DISTORTIONS)
from aggregate.exhibits import build_exhibit

_AGG = 'agg CX.A 50 claims sev lognorm 40 cv 1.8 poisson'
_PORT = ('port CX.P agg CX.P1 50 claims sev lognorm 50 cv 1.5 poisson '
         'agg CX.P2 30 claims sev lognorm 40 cv 1.2 poisson')
_CHF = 'chfagg CX.C 25 * gamma 4'


@pytest.fixture(scope='module', params=['Aggregate', 'Portfolio', 'AggregateCHF'])
def obj(request):
    program = {'Aggregate': _AGG, 'Portfolio': _PORT, 'AggregateCHF': _CHF}[request.param]
    kw = {'log2': 12, 'bs': 1 / 4} if request.param == 'AggregateCHF' else {}
    return build(program, **kw)


def test_extended_calibrates_six_families_in_order(obj):
    result = obj.calibrate_distortions(0.1, p=0.99, extended=True)
    assert tuple(result.distortion_df.index) == EXTENDED_CALIBRATION_DISTORTIONS
    assert tuple(obj.distortions) == EXTENDED_CALIBRATION_DISTORTIONS
    assert (result.distortion_df['error'].abs() < 1e-5).all()


def test_the_default_is_the_five_and_unchanged(obj):
    default = obj.calibrate_distortions(0.1, p=0.99).distortion_df
    named = obj.calibrate_distortions(0.1, p=0.99,
                                      names=DEFAULT_CALIBRATION_DISTORTIONS).distortion_df
    extended = obj.calibrate_distortions(0.1, p=0.99, extended=True).distortion_df
    assert tuple(default.index) == DEFAULT_CALIBRATION_DISTORTIONS
    np.testing.assert_array_equal(default['param'].to_numpy(), named['param'].to_numpy())
    # the five standard rows of the extended set are the default rows
    np.testing.assert_array_equal(
        extended.loc[list(DEFAULT_CALIBRATION_DISTORTIONS), 'param'].to_numpy(),
        default['param'].to_numpy())


def test_names_with_extended_raises(obj):
    with pytest.raises(ValueError, match='names= or extended=True, not both'):
        obj.calibrate_distortions(0.1, p=0.99, names=['ph'], extended=True)


def test_the_calibrate_exhibit_takes_a_sixth_family():
    result = build(_AGG).calibrate_distortions(0.1, p=0.99, extended=True)
    for view in ('raw', 'insurer'):
        text = str(build_exhibit(result, 'pricing.calibrate', view).ir_blocks[0].model_dump())
        assert 'tsallis' in text.lower(), view
