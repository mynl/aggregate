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
from aggregate.constants import DISTORTION_DUAL_LABEL, DISTORTION_DUAL_TEX


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


def test_both_axes_offer_their_reflection(dist):
    """Reflecting the unit square in both axes gives the dual.

    The labels are the literal coordinates rather than the dual's name:
    every series on the panel reflects, the identity diagonal included,
    and only the g curve reflects *to* the dual.
    """
    axes = {a.id: a for a in chart_distortion(dist).axes}
    assert axes['s_complement'].complement_of == 's'
    assert axes['g_complement'].complement_of == 'g'
    assert axes['s_complement'].label == '1 - s'
    assert axes['g_complement'].label == '1 - g(s)'


def test_dual_optional(dist):
    from aggregate.charts import human_strings
    doc = chart_distortion(dist, dual=False)
    roles = [s.role for s in doc.series]
    assert roles == ['distortion', 'identity']
    # The map stays total, and with the one string that has a typeset form
    # gone every entry is now a plain word mapping to itself.
    assert set(doc.tex) == set(human_strings(doc))
    assert all(k == v for k, v in doc.tex.items())


def test_dual_name_is_plain_text_with_a_tex_companion(dist):
    """The schema's naming rule, on the one series that needs it.

    ECharts has no TeX, so the name itself must read anywhere; the typeset
    form travels separately for renderers that can use it.
    """
    doc = chart_distortion(dist)
    dual = doc.series[1]
    assert dual.name == DISTORTION_DUAL_LABEL == 'ǧ(s)'
    assert '$' not in dual.name and '\\' not in dual.name
    assert doc.tex[dual.name] == DISTORTION_DUAL_TEX
    # the plain name is what a TeX-less consumer sees in the wire form
    assert 'ǧ(s)' in canonical_json(doc).decode('utf-8')


def test_deterministic(dist):
    assert canonical_json(chart_distortion(dist)) == \
        canonical_json(chart_distortion(dist))


def test_capability(dist):
    assert available_charts(dist) == ['distortion']
