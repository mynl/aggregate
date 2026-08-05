"""[Chart-Conversions] distortion g(s): the emitter's document contract.

Structural and determinism checks on ``charts.chart_distortion``; the
pixel acceptance gate against the compositor lives in
``test_chartdoc_render.py``.
"""

import pytest

from aggregate import Distortion
from aggregate.charts import (
    available_charts, canonical_json, chart_distortion,
)


@pytest.fixture(scope='module')
def dist():
    return Distortion('ph', 0.7)


def test_document_shape(dist):
    doc = chart_distortion(dist)
    assert doc.name == 'distortion'
    assert doc.title == dist.label
    panel = doc.panels[0]
    assert panel.kind == 'xy' and panel.aspect == 'equal'
    roles = [s.role for s in doc.series]
    assert roles == ['distortion', 'distortion', 'identity']
    # both axes live on the probability unit square
    for a in doc.axes:
        assert a.unit == 'probability'
        assert a.suggested_range == (0.0, 1.0)
    # the g curve reads the knot-spliced density_df grid verbatim
    g = doc.series[0]
    assert len(g.x) == len(dist.density_df)
    assert g.y[0] == pytest.approx(float(dist.density_df['g'].iloc[0]))


def test_dual_optional(dist):
    doc = chart_distortion(dist, dual=False)
    roles = [s.role for s in doc.series]
    assert roles == ['distortion', 'identity']


def test_deterministic(dist):
    assert canonical_json(chart_distortion(dist)) == \
        canonical_json(chart_distortion(dist))


def test_capability(dist):
    assert available_charts(dist) == ['distortion']
