"""[Chart-Kappa-Band] the conditional cession, as a curve with a band.

1.0.0a280, phase four of ``dev/notes-net-natural-allocation.md``. A mean is
the wrong summary for the question a cedent asks about an occurrence program:
the gross outcome does not determine the cession, so "having come in at 500,
how much am I actually ceding" has a spread that the kappa curve averages
away. Two panels: the curves with their bands over the identity, and the same
reading as a share against the most the program could possibly cede.
"""

import json

import numpy as np
import pytest

from aggregate import build
from aggregate.charts import (
    CHARTS, available_charts, build_chart_doc, canonical_dict, doc_hash,
    load_chart_doc,
)
from aggregate.charts._emit_bivariate import (
    KAPPA_CDF_RANGE, KAPPA_LEVELS, _kappa_ceiling,
)

# A limited severity, deliberately: an unlimited lognormal needs a gross axis
# wide enough that nothing else in the picture is visible.
LAYER = ('agg CK.L 10 claims 500 xs 0 sev lognorm 50 cv 1.5 '
         'occurrence net of 50 xs 50 poisson')
TOWER = ('agg CK.T 10 claims 500 xs 0 sev lognorm 50 cv 1.5 '
         'occurrence net of 50 xs 50 and 100 xs 100 poisson')
COPULA = ('bivariate CK.MV 5 claims '
          'agg A dfreq [0 1] [.3 .7] sev lognorm 40 cv 1.2 '
          'agg B dfreq [0 1] [.5 .5] sev lognorm 60 cv 1.5 '
          'copula gumbel 0.4 poisson')


@pytest.fixture(scope='module')
def joint():
    return build(LAYER).occ_bivariate(views=('gross', 'ceded'))


@pytest.fixture(scope='module')
def doc(joint):
    return build_chart_doc(joint, 'kappa')


def series_of(doc, panel):
    return [s for s in doc.series if s.panel_id == panel]


def named(doc, panel, name):
    """One series by panel and exact name.

    Panel scoped and exact on purpose: the two bands of one view share a name
    across panels (the IR's way of saying they are the same entity seen
    twice), and 'gross' is both the identity series and a substring of every
    conditional series' name.
    """
    return next(s for s in series_of(doc, panel) if s.name == name)


# --- availability -----------------------------------------------------------

def test_registered_and_available(joint):
    assert 'kappa' in CHARTS
    assert 'kappa' in available_charts(joint)


def test_not_available_without_a_gross_axis():
    """The curves condition on the gross outcome; a (net, ceded) joint has none."""
    nc = build(LAYER).occ_bivariate()          # the default (net, ceded)
    assert 'kappa' not in available_charts(nc)
    with pytest.raises(ValueError, match='not available'):
        build_chart_doc(nc, 'kappa')


def test_not_available_on_a_copula_joint():
    """Conditioning on a dependent total is the deferred question, not this one."""
    assert 'kappa' not in available_charts(build(COPULA).bivariate)


def test_kappa_is_not_the_primary_picture(joint):
    """A joint's own picture is its density surface; this is a reading of it."""
    from aggregate.charts import primary_chart

    assert primary_chart(joint) == 'joint_surface'


# --- structure --------------------------------------------------------------

def test_two_panels_over_one_gross_axis(doc):
    assert [p.id for p in doc.panels] == ['cession', 'share']
    assert {p.x_axis for p in doc.panels} == {'outcome'}
    # both axes of the left panel are losses, so the identity reads as 45
    # degrees and a stretched box would misstate it
    assert doc.panels[0].aspect == 'equal'


def test_bands_are_one_series_with_two_edges(doc):
    """A band *is* the region between its edges, not two curves to associate."""
    bands = [s for s in doc.series if s.y2 is not None]
    assert len(bands) == 3           # ceded and net on cession, ceded on share
    for band in bands:
        assert len(band.y2) == len(band.y_values)
        assert 'percentile' in band.name
        assert 'confidence' not in band.name.lower()


def test_meta_says_what_the_band_is(doc):
    """Nothing here is an estimate with sampling error: the joint is the law."""
    assert doc.meta['band'] == 'percentile'
    # lists, because meta travels as JSON and must come back hash for hash
    assert doc.meta['levels'] == list(KAPPA_LEVELS)
    assert doc.meta['cdf_range'] == list(KAPPA_CDF_RANGE)


def test_share_axis_is_a_ratio(doc):
    share = next(a for a in doc.axes if a.id == 'share')
    assert share.unit == 'ratio'
    assert share.suggested_range == (0.0, 1.0)


def test_the_grid_rides_as_a_lattice(doc):
    """The window is a contiguous slice of a bucket grid, so it ships as three
    numbers rather than as thousands."""
    for s in doc.series:
        assert s.x_lattice is not None
        assert s.x is None


# --- the arithmetic ---------------------------------------------------------

def test_the_two_curves_sum_to_the_identity(doc):
    """The conservation statement the left panel exists to make.

    ``kappa_N = g - kappa_C`` pointwise, because the third view is read off
    the index rather than measured a second time.
    """
    g = np.asarray(named(doc, 'cession', 'gross').y_values, dtype=float)
    ceded = np.asarray(
        named(doc, 'cession', 'E[ceded | gross]').y_values, dtype=float)
    net = np.asarray(
        named(doc, 'cession', 'E[net | gross]').y_values, dtype=float)
    assert (ceded + net) == pytest.approx(g, abs=1e-9)
    assert np.asarray(
        named(doc, 'cession', 'gross').x_values) == pytest.approx(g)


def test_the_net_band_is_the_reflected_ceded_band(doc):
    """Two shaded regions of equal width, one at zero and one at the diagonal."""
    g = np.asarray(named(doc, 'cession', 'gross').y_values, dtype=float)
    ceded = [s for s in series_of(doc, 'cession')
             if s.y2 is not None and s.role == 'ceded'][0]
    net = [s for s in series_of(doc, 'cession')
           if s.y2 is not None and s.role == 'net'][0]
    assert np.asarray(net.y_values) == pytest.approx(
        g - np.asarray(ceded.y2), abs=1e-9)
    assert np.asarray(net.y2) == pytest.approx(
        g - np.asarray(ceded.y_values), abs=1e-9)


def test_the_band_has_real_width(doc):
    """The point of the picture: a gross outcome does not fix the cession."""
    ceded = [s for s in series_of(doc, 'cession') if s.y2 is not None][0]
    width = np.asarray(ceded.y2) - np.asarray(ceded.y_values)
    assert (width >= -1e-12).all()
    assert width.max() > 50.0            # the layer limit, comfortably cleared


def test_share_is_the_value_divided_by_the_outcome(doc):
    """Quantiles commute with ``c -> c / g`` at fixed ``g``: no second pass."""
    g = np.asarray(named(doc, 'cession', 'gross').y_values, dtype=float)
    value = np.asarray(
        named(doc, 'cession', 'E[ceded | gross]').y_values, dtype=float)
    share = np.asarray(next(s for s in series_of(doc, 'share')
                            if s.y2 is None and s.role == 'ceded').y_values)
    live = g > 0
    assert share[live] == pytest.approx(value[live] / g[live])
    assert ((share[live] >= -1e-12) & (share[live] <= 1.0 + 1e-12)).all()


# --- the deterministic ceiling ----------------------------------------------

def test_ceiling_is_a_comb_with_the_right_teeth():
    """One layer ``limit xs attach``: teeth every ``attach + limit``.

    The largest cession with a gross total of ``g`` splits it into claims of
    exactly ``attach + limit``, each ceding the full limit, so the share peaks
    at ``limit / (attach + limit)`` at each tooth.
    """
    agg = build(LAYER)
    g = np.arange(0.0, 1000.0, 1.0)
    top = _kappa_ceiling(agg, g)
    assert top is not None
    with np.errstate(invalid='ignore', divide='ignore'):
        share = np.where(g > 0, top / g, np.nan)
    assert np.nanmax(share) == pytest.approx(0.5)          # 50 / (50 + 50)
    # a tooth every 100: k whole claims cede exactly 50k
    for k in (1, 3, 7):
        assert top[int(100 * k)] == pytest.approx(50.0 * k)
    # and nothing at all below the attachment
    assert (top[:51] == 0.0).all()


def test_no_ceiling_for_a_tower():
    """A tower has no simple envelope, so the panel has one fewer curve."""
    assert _kappa_ceiling(build(TOWER), np.arange(0.0, 100.0)) is None
    doc = build(TOWER).occ_bivariate(views=('gross', 'ceded'))
    got = build_chart_doc(doc, 'kappa')
    assert not [s for s in got.series if s.role == 'ceiling']


def test_the_realized_band_sits_under_the_ceiling(doc):
    """How much of the available cession the program actually delivers.

    Getting several claims to land exactly at the top of the layer is a lot to
    ask, so even the good tail of the conditional law falls short of the comb.
    """
    top = np.asarray(
        named(doc, 'share', 'most the program could cede').y_values,
        dtype=float)
    band = next(s for s in series_of(doc, 'share') if s.y2 is not None)
    hi = np.asarray(band.y2, dtype=float)
    ok = np.isfinite(top) & np.isfinite(hi)
    assert (hi[ok] <= top[ok] + 1e-9).all()
    assert np.nanmax(hi) < np.nanmax(top)


def test_ceiling_can_be_turned_off(joint):
    from aggregate.charts import chart_kappa

    assert not [s for s in chart_kappa(joint, ceiling=False).series
                if s.role == 'ceiling']


# --- the wire ---------------------------------------------------------------

def test_round_trips_hash_for_hash(doc):
    back = load_chart_doc(json.loads(json.dumps(canonical_dict(doc))))
    assert doc_hash(back) == doc.hash
    assert back == doc


def test_levels_must_be_a_pair(joint):
    from aggregate.charts import chart_kappa

    with pytest.raises(ValueError, match='two edges'):
        chart_kappa(joint, levels=(0.1, 0.5, 0.9))


def test_renders(joint):
    """The generic renderer draws it: no per-chart matplotlib code exists."""
    import matplotlib
    matplotlib.use('Agg')
    from aggregate.plots import plot_chartdoc

    fig = plot_chartdoc(build_chart_doc(joint, 'kappa'))
    assert len(fig.axes) >= 2


@pytest.mark.slow
def test_a_disk_backed_joint_is_accepted(tmp_path):
    """Surviving the massive route is the point of a row-wise band."""
    pytest.importorskip('zarr')
    agg = build(LAYER)
    disk = agg.occ_bivariate(views=('gross', 'ceded'),
                             store_dir=str(tmp_path / 'ck'))
    assert 'kappa' in available_charts(disk)
    assert 'joint_surface' not in available_charts(disk)   # needs the array
    on_disk = build_chart_doc(disk, 'kappa')
    in_core = build_chart_doc(agg.occ_bivariate(views=('gross', 'ceded')),
                              'kappa')
    assert [s.name for s in on_disk.series] == [s.name for s in in_core.series]
