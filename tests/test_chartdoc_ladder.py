"""The renderer's atomic ladder: three drawings of one truth.

A discretized distribution **is** the distribution here, not an
approximation to a continuous ideal, so the question is never whether the
data is stepped but whether the steps can be seen. That is a property of
the figure, so the document says only what the law is
(``ChartSeries.support``) and the renderer picks the drawing:

    few atoms       stems with markers, each atom drawn
    steps visible   steps, read as a bar at each bucket
    sub-pixel       a plain line, indistinguishable from steps anyway

with cumulative functions skipping the first rung, because F and S take a
value at every x rather than only at the atoms.

No matplotlib version pin here: these assert draw styles, not pixels.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')

from aggregate.charts import (  # noqa: E402
    ChartAxis, ChartDoc, ChartSeries, Panel,
)
from aggregate.plots._chartdoc import (  # noqa: E402
    LOLLIPOP_ATOMS, STEP_PIXELS,
)


# Atom counts either side of each rung, for the house panel width. The
# stem rung ends at LOLLIPOP_ATOMS and the step rung ends where an atom is
# thinner than STEP_PIXELS, which for a single house-width panel is a few
# hundred; these two sit safely inside their bands rather than on an edge.
STEP_BAND = 80
CROWDED = 50_000


def doc_with(n, unit, support='atomic', span=None):
    """A one-panel document with ``n`` evenly spaced points."""
    span = n if span is None else span
    x = tuple(float(v) for v in np.linspace(0.0, span, n))
    y = tuple(float(v) for v in np.linspace(1.0, 0.1, n))
    return ChartDoc(
        name='t',
        axes=(ChartAxis(id='loss', label='Loss', unit='currency',
                        suggested_range=(0.0, span)),
              ChartAxis(id='y', label='y', unit=unit)),
        panels=(Panel(id='p', kind='xy', x_axis='loss', y_axis='y'),),
        series=(ChartSeries(name='s', role='density', panel_id='p',
                            x=x, y=y, support=support),))


def drawn(doc):
    """``(line draw styles, marker set, number of stem collections)``."""
    from aggregate.plots import plot_chartdoc, plt
    fig = plot_chartdoc(doc)
    ax = fig.axes[0]
    styles = [ln.get_drawstyle() for ln in ax.get_lines()]
    markers = {ln.get_marker() for ln in ax.get_lines()}
    stems = len(ax.collections)
    plt.close('all')
    return styles, markers, stems


def test_few_atoms_draw_as_stems():
    styles, markers, stems = drawn(doc_with(LOLLIPOP_ATOMS - 10, 'density'))
    assert stems == 1                    # the vertical to each atom
    assert markers == {'o'}
    assert styles == ['default']         # the markers' own line, not a curve


def test_many_atoms_draw_as_steps():
    """Past the stem threshold but with room per atom: the bar reading."""
    styles, markers, stems = drawn(doc_with(STEP_BAND,'density'))
    assert styles == ['steps-mid']
    assert stems == 0 and markers == {'None'}


def test_sub_pixel_atoms_draw_as_a_line():
    """No lie is told: at this density steps and a line are one picture."""
    styles, _, stems = drawn(doc_with(CROWDED,'density'))
    assert styles == ['default']
    assert stems == 0


def test_cumulative_steps_right_continuously_and_never_stems():
    """F and S have a value at every x, not only at the atoms."""
    styles, markers, stems = drawn(doc_with(LOLLIPOP_ATOMS - 10, 'probability'))
    assert styles == ['steps-post']
    assert stems == 0 and markers == {'None'}


def test_cumulative_also_falls_back_to_a_line_when_crowded():
    styles, _, _ = drawn(doc_with(CROWDED,'probability'))
    assert styles == ['default']


def test_a_level_steps_but_never_stems():
    """A kappa is a conditional mean per bucket: stepped, but a stem would
    say it is a mass, which it is not."""
    styles, markers, stems = drawn(doc_with(STEP_BAND,'currency'))
    assert styles == ['steps-mid']
    assert stems == 0 and markers == {'None'}


def test_continuous_support_never_steps():
    """Samples of a function that lives between them draw as a line at any
    density, however few points there are."""
    for n in (5, 200, 50_000):
        styles, markers, stems = drawn(doc_with(n, 'density',
                                                support='continuous'))
        assert styles == ['default']
        assert stems == 0 and markers == {'None'}


def test_the_ladder_reads_the_axis_not_the_series_role():
    """A reinsurance series is called 'gross' in both panels, and only the
    axis knows one carries mass and the other accumulated probability."""
    doc = doc_with(STEP_BAND,'probability')
    role_says_density = doc.series[0].role == 'density'
    styles, _, _ = drawn(doc)
    assert role_says_density and styles == ['steps-post']


def test_threshold_is_pixels_per_atom_not_the_grid_size():
    """A cropped window is judged on what it shows: the same 50,000-point
    grid steps once the window holds few enough of them."""
    wide = doc_with(CROWDED, 'density', span=float(CROWDED))
    assert drawn(wide)[0] == ['default']
    narrow = ChartDoc(
        name='t',
        axes=(ChartAxis(id='loss', label='Loss', unit='currency',
                        suggested_range=(0.0, float(STEP_BAND))),
              ChartAxis(id='y', label='y', unit='density')),
        panels=wide.panels, series=wide.series)
    assert drawn(narrow)[0] == ['steps-mid']


def test_real_documents_declare_what_they_are():
    from aggregate import build
    from aggregate.charts import chart_distortion, chart_reins, chart_severity
    from aggregate import Distortion
    # a distortion is a function of s, defined everywhere
    assert {s.support for s in chart_distortion(Distortion('ph', 0.7)).series} \
        == {'continuous'}
    # a frozen severity has no bs and no discrete density
    assert {s.support for s in
            chart_severity(build('sev LD.LN lognorm 100 cv 2')).series} \
        == {'continuous'}
    # a discrete one is atoms, and an aggregate's grid always is
    assert {s.support for s in chart_severity(
        build('sev LD.D dhistogram xps [1 10 40] [.5 .3 .2]')).series} \
        == {'atomic'}
    agg = build('agg LD.Re dfreq [1 2] dsev [1:40] occurrence net of 10 xs 10')
    assert {s.support for s in chart_reins(agg).series} == {'atomic'}


def test_thresholds_are_renderer_constants():
    """They describe the figure, so they have no business in a document."""
    from aggregate.charts import ir
    assert LOLLIPOP_ATOMS > 0 and STEP_PIXELS > 0
    assert not [n for n in dir(ir) if 'LOLLIPOP' in n or 'PIXEL' in n]
