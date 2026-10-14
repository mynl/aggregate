"""The ``exact_tail`` chart: an nttagg's exact answer as base-10 logarithms.

[NTT-Exact-Tail-Chart] (1.5.0), dev/done/plan-1.4.0-nttagg-arithmetic-exact-chart.md
step 2. The chart exists for the deep tail, so the headline test is one number
the float FFT answer cannot represent against a body of order 1.
"""

import json
import math
from fractions import Fraction

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from aggregate import build  # noqa: E402
from aggregate.charts import (available_charts, build_chart_doc,  # noqa: E402
                              canonical_dict, doc_hash, load_chart_doc,
                              primary_chart)

#: Wilson2016a.Wilson from agg/reproductions.agg: two claims, severity mass
#: 1e-20 at 3, so the exact mass at 6 is 1e-40 and at 5 is 2e-25.
WILSON = ('nttagg Wilson dfreq [2] dsev [1 2 3] '
          '[99998999999999999999/100000000000000000000 1/100000 '
          '1/100000000000000000000]')

#: Zero mass at losses 5 to 9 and 13 to 19: support 0, 1, 2, 3, 4, 10, 11,
#: 12, 20.
GAPPED = 'nttagg Gapped dfreq [0 1 2] [1/4 1/2 1/4] dsev [1 2 10] [3/10 1/2 1/5]'


@pytest.fixture(scope='module')
def wilson():
    return build(WILSON)


@pytest.fixture(scope='module')
def gapped():
    return build(GAPPED)


def _series(doc, panel_id, name):
    """The one series called ``name`` on panel ``panel_id``."""
    found, = [s for s in doc.series
              if s.panel_id == panel_id and s.name == name]
    return found


def _points(series):
    """``{loss: value}`` over a series, whatever form its x travels in."""
    return dict(zip(series.x_values, series.y_values))


def test_registry_availability_and_primary(wilson):
    assert 'exact_tail' in available_charts(wilson)
    assert primary_chart(wilson) == 'agg'
    plain = build('agg Plain dfreq [2] dsev [1 2 3]')
    assert 'exact_tail' not in available_charts(plain)


def test_unavailable_before_update():
    a = build('nttagg NotYet dfreq [1:3] dsev [1:3]', update=False)
    assert 'exact_tail' not in available_charts(a)
    with pytest.raises(ValueError, match='availability predicate'):
        build_chart_doc(a, 'exact_tail')


def test_document_shape(wilson):
    doc = build_chart_doc(wilson, 'exact_tail')
    assert [p.id for p in doc.panels] == ['mass', 'survival']
    axes = {a.id: a for a in doc.axes}
    assert axes['log10_mass'].unit == 'log10_density'
    assert axes['log10_survival'].unit == 'log10_probability'
    assert axes['log10_mass'].scales == ('linear',)
    # the window is the whole support, not a quantile crop
    assert axes['outcome'].suggested_range == (0.0, 6.0)
    assert axes['outcome'].full_range == (0.0, float(wilson.xs[-1]))
    assert axes['log10_mass'].suggested_range == (-40.0, 0.0)
    assert doc.marks == ()
    assert {(s.panel_id, s.name) for s in doc.series} == {
        ('mass', 'Exact'), ('mass', 'Float FFT'),
        ('survival', 'Exact'), ('survival', 'Float FFT')}
    # nothing non-finite is ever sent
    assert all(math.isfinite(v) for s in doc.series for v in s.y_values)


def test_deep_value_the_float_cannot_hold(wilson):
    """The whole point in one number: 1e-40 against a body of order 1."""
    doc = build_chart_doc(wilson, 'exact_tail')
    mass = _points(_series(doc, 'mass', 'Exact'))
    assert mass[6.0] == pytest.approx(-40.0, abs=1e-10)
    assert mass[5.0] == pytest.approx(math.log10(2e-25), abs=1e-10)
    survival = _points(_series(doc, 'survival', 'Exact'))
    assert survival[5.0] == pytest.approx(-40.0, abs=1e-10)
    assert not any(x >= 6.0 for x in survival)


def test_interior_zeros_are_omitted(gapped):
    doc = build_chart_doc(gapped, 'exact_tail')
    exact = _series(doc, 'mass', 'Exact')
    assert set(_points(exact)) == {0.0, 1.0, 2.0, 3.0, 4.0, 10.0, 11.0,
                                   12.0, 20.0}
    # holes, so the coordinates travel explicitly rather than as a lattice
    assert exact.x_lattice is None and exact.x is not None
    # the survival is a gap-free prefix up to the top of the support
    survival = _series(doc, 'survival', 'Exact')
    assert survival.x_lattice == (0.0, 1.0, 20)


def test_survival_is_exact():
    """Every point is log10 of the exact suffix sum of p_exact."""
    a = build('nttagg Dice dfreq [1:6] dsev [1:6]')
    doc = build_chart_doc(a, 'exact_tail')
    exact = list(a.p_exact)
    survival = _series(doc, 'survival', 'Exact')
    values = list(survival.y_values)
    for x, value in zip(survival.x_values, values):
        tail = sum(exact[int(x) + 1:], Fraction(0))
        assert value == pytest.approx(
            math.log10(tail.numerator) - math.log10(tail.denominator),
            abs=1e-12)
    assert all(b <= a for a, b in zip(values, values[1:]))


def test_wire_round_trip_and_render(wilson, gapped):
    for obj in (wilson, gapped):
        doc = build_chart_doc(obj, 'exact_tail')
        back = load_chart_doc(json.loads(json.dumps(canonical_dict(doc))))
        assert doc_hash(back) == doc.hash
        from aggregate.plots import plot_chartdoc
        fig = plot_chartdoc(doc)
        plt.close(fig)


def test_mass_draws_as_markers_only(gapped):
    """No line may bridge the omitted exact zeros."""
    from aggregate.plots import plot_chartdoc
    fig = plot_chartdoc(build_chart_doc(gapped, 'exact_tail'))
    try:
        mass_ax = fig.axes[0]
        assert mass_ax.lines
        for line in mass_ax.lines:
            assert line.get_linestyle() == 'None'
            assert line.get_marker() not in (None, 'None', '')
        assert not mass_ax.collections      # no stems
    finally:
        plt.close(fig)
