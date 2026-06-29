"""Tests for the insurance View vocabulary and composition (``_insurance_view``)."""

import numpy as np
import pytest

from aggregate.legs import Leg, LegSet, GraphSource
from aggregate._insurance_view import (
    InsuranceView, perspective_of, category_of, kind_of, label_of)


# ----------------------------------------------------------------------
# the label vocabulary: name -> (perspective, category) -> kind
# ----------------------------------------------------------------------
@pytest.mark.parametrize('name, persp, cat, knd', [
    ('gross_loss', 'gross', 'loss', 'obligation'),
    ('ceded_premium', 'ceded', 'premium', 'consideration'),
    ('net_uw', 'net', 'underwriting', 'margin'),
    ('gross_expense', 'gross', 'expense', 'obligation'),
    ('commission', 'ceded', 'expense', 'obligation'),
    ('reinstatement_premium', 'ceded', 'premium', 'consideration'),
    ('unlimited_ceded_loss', 'ceded', 'loss', 'obligation'),
    ('total_ceded_premium', 'total_ceded', 'premium', 'consideration'),
    ('ceded_agg_loss', 'ceded_agg', 'loss', 'obligation'),
    ('net_agg_uw', 'net_agg', 'underwriting', 'margin'),
])
def test_label_vocabulary(name, persp, cat, knd):
    assert perspective_of(name) == persp
    assert category_of(name) == cat
    assert kind_of(name) == knd
    assert label_of(name) == (persp, cat)


def test_unknown_leg_name_raises():
    with pytest.raises(ValueError):
        perspective_of('mystery')
    with pytest.raises(ValueError):
        category_of('mystery')


# ----------------------------------------------------------------------
# InsuranceView: cached kernel evaluation + exact point masses
# ----------------------------------------------------------------------
def _source():
    x = np.arange(0.0, 10.0001, 1.0)
    p = np.ones_like(x) / len(x)
    return GraphSource(x, p, kappa=lambda v: 0.5 * np.asarray(v, dtype=float))


def test_view_evaluates_legs_and_keeps_point_masses_exact():
    legs = LegSet([
        Leg('gross_loss', lambda x, y: x),
        Leg('ceded_loss', lambda x, y: y),
        Leg('net_loss', lambda x, y: x - y),
    ])
    view = InsuranceView(legs, _source(), point_masses={'gross_premium': 1234.0})
    # every leg plus the point mass appears once
    assert set(view.distributions) == {'gross_loss', 'ceded_loss', 'net_loss',
                                       'gross_premium'}
    # the point mass is an exact one-atom distribution (not rebucketed)
    gp = view.distributions['gross_premium']
    assert gp.x.tolist() == [1234.0] and gp.p.tolist() == [1.0]
    assert view.exact('gross_premium') == (1234.0, 0.0, 0.0)
    # means add through the kernel
    gm, _, _ = view.exact('gross_loss')
    cm, _, _ = view.exact('ceded_loss')
    nm, _, _ = view.exact('net_loss')
    assert nm == pytest.approx(gm - cm, rel=1e-12)


def test_view_caches_one_evaluation():
    view = InsuranceView(LegSet([Leg('gross_loss', lambda x, y: x)]), _source())
    assert view.distributions is view.distributions
    assert view.stats_df is view.stats_df
