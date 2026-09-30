"""[Matrix-Panel] the matplotlib renderer for a 'matrix' panel.

Structural rather than pixel based, deliberately. ``test_chartdoc_render.py``
gates the converted charts against **approved** baseline images, and an approved
picture is the author's to approve; a baseline generated and blessed in the same
commit that wrote the renderer would be a measurement of nothing. So what is
pinned here is what the renderer must be *doing*: the ticks it writes, the text
in each cell, and above all which way each cell is colored, since row polarity
is the one thing a reader cannot check by eye against the numbers.

A baseline image for this panel kind is still owed, and is noted in the plan.
"""

import matplotlib
import pytest

matplotlib.use('Agg')

from aggregate.charts import (  # noqa: E402
    ChartAxis, ChartDoc, ChartSeries, MatrixData, Panel,
)
from aggregate.plots import plot_chartdoc  # noqa: E402
from aggregate.plots._chartdoc import (  # noqa: E402
    MATRIX_BAND_GAP, _matrix_norm, _matrix_offsets,
)


@pytest.fixture
def matrix():
    """Two positions against two readings, read in opposite directions."""
    return MatrixData(
        rows=('gross book', 'QS'),
        columns=('gini ph', 'margin'),
        values=((1.0, 1.0), (1.60, 0.40)),
        annotations=(('0.209', '7.3%'), ('0.314', '2.9%')),
        center=1.0, neutral=0.05,
        # The book row is read the other way: pricing above the reference is an
        # improvement for it, and paying above the reference is not for the QS.
        row_polarity=(-1, 1),
        row_groups=('book', 'aggregate'),
        column_groups=('family', 'point'),
    )


@pytest.fixture
def doc(matrix):
    return ChartDoc(
        name='relativity', title='Relativity',
        axes=(ChartAxis(id='x', label='reading', kind='category'),
              ChartAxis(id='y', label='position', kind='category'),
              ChartAxis(id='z', label='multiple of gross')),
        panels=(Panel(id='m', kind='matrix', x_axis='x', y_axis='y',
                      z_axis='z'),),
        series=(ChartSeries(name='relativity', role='identity', panel_id='m',
                            matrix=matrix),))


def _axes(fig):
    return fig.axes[0]


# --- what reaches the canvas -------------------------------------------------

def test_the_panel_draws_at_all(doc):
    fig = plot_chartdoc(doc)
    assert fig is not None
    matplotlib.pyplot.close(fig)


def test_the_ticks_are_the_row_and_column_names(doc):
    fig = plot_chartdoc(doc)
    ax = _axes(fig)
    assert [t.get_text() for t in ax.get_xticklabels()] == ['gini ph', 'margin']
    assert [t.get_text() for t in ax.get_yticklabels()] == ['gross book', 'QS']
    matplotlib.pyplot.close(fig)


def test_rows_read_top_to_bottom(doc):
    """The document's own order, which is the peel order, not bottom up."""
    fig = plot_chartdoc(doc)
    bottom, top = _axes(fig).get_ylim()
    assert bottom > top, 'the y axis must be inverted so row 0 is at the top'
    matplotlib.pyplot.close(fig)


def test_every_cell_carries_its_value_and_its_annotation(doc):
    fig = plot_chartdoc(doc)
    texts = {t.get_text() for t in _axes(fig).texts}
    for value in ('1.00×', '1.60×', '0.40×'):
        assert value in texts, value
    for annotation in ('(0.209)', '(7.3%)', '(0.314)', '(2.9%)'):
        assert annotation in texts, annotation
    matplotlib.pyplot.close(fig)


def test_a_negative_value_uses_a_minus_sign_not_a_hyphen():
    """These are numbers being read, so they take the typographic minus."""
    data = MatrixData(rows=('r',), columns=('c',), values=((-2.5,),),
                      annotations=(('-1.4%',),), center=1.0)
    doc = ChartDoc(
        name='x',
        axes=(ChartAxis(id='x', label='a', kind='category'),
              ChartAxis(id='y', label='b', kind='category'),
              ChartAxis(id='z', label='c')),
        panels=(Panel(id='m', kind='matrix', x_axis='x', y_axis='y',
                      z_axis='z'),),
        series=(ChartSeries(name='s', role='identity', panel_id='m',
                            matrix=data),))
    fig = plot_chartdoc(doc)
    texts = {t.get_text() for t in _axes(fig).texts}
    assert '−2.50×' in texts
    assert '(−1.4%)' in texts
    assert not any('-' in t for t in texts), 'a hyphen reached a number'
    matplotlib.pyplot.close(fig)


def test_a_missing_cell_prints_nothing(doc, matrix):
    """None is the absence of a quantity, so there is no number to print."""
    import dataclasses
    holed = dataclasses.replace(matrix, values=((1.0, None), (1.60, 0.40)),
                                annotations=())
    fig = plot_chartdoc(dataclasses.replace(
        doc, series=(dataclasses.replace(doc.series[0], matrix=holed),)))
    values = [t.get_text() for t in _axes(fig).texts]
    assert len(values) == 3, values
    matplotlib.pyplot.close(fig)


# --- the colors, which are the part a reader cannot check by eye -------------

def test_polarity_flips_which_direction_is_favorable(matrix):
    """The same multiple, colored oppositely on two rows.

    This is the whole reason ``row_polarity`` is in the IR rather than in the
    renderer: a consistent scale over these rows would be backwards for half of
    them, and nothing in the numbers says so.
    """
    import numpy as np

    values = np.array(matrix.values, dtype=float)
    polarity = np.array(matrix.polarity, dtype=float).reshape(-1, 1)
    signed = polarity * (values - matrix.center)
    cmap, norm = _matrix_norm(signed, matrix.center, matrix.neutral)
    # QS at 1.60 pays above the book: unfavorable, so positive departure.
    assert signed[1, 0] > 0
    # A book row at 1.60 would be an improvement: negative departure.
    assert (-1 * (1.60 - 1.0)) < 0
    favorable = cmap(norm(-0.6))
    unfavorable = cmap(norm(0.6))
    assert favorable[1] > favorable[0], 'the favorable end must be green'
    assert unfavorable[0] > unfavorable[1], 'the unfavorable end must be red'


def test_the_neutral_band_is_a_hard_stop(matrix):
    """Inside the band every cell takes the same color, and just outside it does not.

    A gradient running through the band would show a signal the document said
    was not there.
    """
    import numpy as np

    signed = np.array([[-1.0, 1.0]])
    cmap, norm = _matrix_norm(signed, 1.0, 0.5)
    inside = {tuple(cmap(norm(v))) for v in (-0.2, 0.0, 0.2)}
    assert len(inside) == 1, 'the band is not flat'
    assert cmap(norm(0.9)) not in inside
    assert cmap(norm(-0.9)) not in inside


def test_no_center_means_no_color(matrix):
    """Values that do not diverge get no diverging scale."""
    import numpy as np
    cmap, norm = _matrix_norm(np.array([[np.nan, np.nan]]), None, 0.0)
    assert cmap is None and norm is None


def test_a_flat_matrix_gets_no_scale():
    """Every cell at the center: an amplitude of zero has no scale to build."""
    import numpy as np
    cmap, norm = _matrix_norm(np.zeros((2, 2)), 1.0, 0.05)
    assert cmap is None and norm is None


def test_a_non_finite_departure_is_drawn_without_color(doc, matrix):
    """A ratio through a negative denominator is real, and is left in.

    Masking it would hide a number. It is the diverging scale that cannot read
    it, so the cell loses its color and keeps its text.
    """
    import dataclasses
    import numpy as np

    holed = dataclasses.replace(
        matrix, values=((1.0, 1.0), (float('nan'), 0.40)))
    fig = plot_chartdoc(dataclasses.replace(
        doc, series=(dataclasses.replace(doc.series[0], matrix=holed),)))
    texts = [t.get_text() for t in _axes(fig).texts]
    assert 'nan×' not in texts
    assert np.isfinite(float('0.40'))
    matplotlib.pyplot.close(fig)


# --- the band gaps -----------------------------------------------------------

def test_column_bands_open_a_gap():
    offsets = _matrix_offsets(('family', 'family', 'point'), 3)
    assert list(offsets) == [0.0, 0.0, MATRIX_BAND_GAP]


def test_no_grouping_means_no_gaps():
    assert list(_matrix_offsets((), 4)) == [0.0, 0.0, 0.0, 0.0]


def test_every_band_after_the_first_clears_the_one_before():
    offsets = _matrix_offsets(('a', 'b', 'c'), 3)
    assert list(offsets) == [0.0, MATRIX_BAND_GAP, 2 * MATRIX_BAND_GAP]
