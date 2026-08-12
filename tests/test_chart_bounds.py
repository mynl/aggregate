"""[Chart-Bounds] the envelope of admissible prices, in two panels.

Every distortion pricing the book to its premium lies between two curves,
and that band is the subject: it says what prices are possible before
anyone argues about which is right. The compositor drew three panels and
split the five calibrated distortions across the last two, which was an
accident of the order they were added; the question is how the five
compare, so they share one band now.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')

from aggregate import build  # noqa: E402
from aggregate.bounds import Bounds  # noqa: E402
from aggregate.charts import (  # noqa: E402
    available_charts, build_chart_doc, chart_envelope, human_strings,
    primary_chart,
)
from aggregate.charts._emit_bounds import CALIBRATED  # noqa: E402

_PORT = ('port CB.P agg A 50 claims sev lognorm 50 cv 1.5 poisson '
         'agg B 30 claims sev lognorm 40 cv 1.2 poisson')


@pytest.fixture(scope='module')
def bounds():
    port = build(_PORT)
    prem = float(port.tvar(0.5))
    coc = (prem - port.actual_m) / (port.q(1) - prem)
    # a=, not p=1: the Bounds methodology takes the top of the realized grid
    # as the asset level deliberately ([Unbounded-Anchor-Guard], a260)
    port.calibrate_distortions(coc, a=port.q(1))
    return Bounds(port, premium=prem)


@pytest.fixture(scope='module')
def bare():
    """Bounds on an object with nothing calibrated: one panel, not two."""
    agg = build('agg CB.A 100 claims sev lognorm 50 cv 2 poisson')
    return Bounds(agg, premium=float(agg.tvar(0.5)))


def series_on(doc, panel):
    return [s for s in doc.series if s.panel_id == panel]


def test_two_squares(bounds):
    doc = chart_envelope(bounds)
    assert [p.id for p in doc.panels] == ['cloud', 'calibrated']
    assert all(p.aspect == 'equal' for p in doc.panels)
    assert {a.suggested_range for a in doc.axes} == {(0.0, 1.0)}


def test_the_unit_square_offers_no_other_reading(bounds):
    """It is the window and the whole extent, so neither button exists."""
    for axis in chart_envelope(bounds).axes:
        assert axis.scales == ('linear',)
        assert axis.full_range is None


def test_the_envelope_is_one_band_not_two_curves(bounds):
    """The series *is* the region between the extremes."""
    band = series_on(chart_envelope(bounds), 'cloud')[0]
    assert band.name == 'Envelope'
    assert band.y2 is not None and len(band.y2) == len(band.y)
    lo = np.array(band.y)
    hi = np.array(band.y2)
    assert (hi >= lo).all()
    np.testing.assert_allclose(lo, bounds.cloud_df.min(axis=1).to_numpy())
    np.testing.assert_allclose(hi, bounds.cloud_df.max(axis=1).to_numpy())


def test_all_five_calibrated_distortions_share_one_band(bounds):
    """Where the compositor split them across two panels."""
    names = [s.name for s in series_on(chart_envelope(bounds), 'calibrated')]
    for label in CALIBRATED.values():
        assert label in names
    assert 'Envelope' in names and 'Avg extreme' in names


def test_a_cloud_curve_carries_its_weight(bounds):
    """The number is a fact a reader asks about, so it travels as data."""
    doc = chart_envelope(bounds, n_resamples=12)
    cloud = [s for s in series_on(doc, 'cloud') if s.value is not None]
    assert len(cloud) == 12
    assert all(0.0 <= s.value <= 1.0 for s in cloud)
    assert doc.meta['value_label']           # and the document names it
    # zero draws the band alone, which is what the compositor defaulted to
    assert not [s for s in series_on(chart_envelope(bounds), 'cloud')
                if s.value is not None]


def test_a_panel_with_nothing_to_say_is_omitted(bare):
    """A Bounds on an uncalibrated object still has an envelope to draw."""
    doc = chart_envelope(bare)
    assert [p.id for p in doc.panels] == ['cloud']
    assert doc.meta['calibrated'] == ()


def test_tex_is_total(bounds):
    doc = chart_envelope(bounds)
    assert not set(human_strings(doc)) - set(doc.tex)


def test_capability(bounds):
    assert available_charts(bounds) == ['envelope']
    assert primary_chart(bounds) == 'envelope'
    assert build_chart_doc(bounds, 'envelope').hash


def test_it_renders(bounds):
    from aggregate.plots import plt
    fig = bounds.plot_envelope(n_resamples=8)
    assert len(fig.axes) == 3            # two panels and the weight colorbar
    cloud = fig.axes[0]
    assert cloud.get_aspect() == 1.0
    assert cloud.collections            # the band filled
    plt.close('all')
