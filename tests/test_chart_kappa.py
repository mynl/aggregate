"""[Chart-Kappa-Band] the conditional cession, as a curve with a band.

1.0.0a280, phase four of ``dev/done/notes-net-natural-allocation.md``. A mean is
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


# --- the other two surfaces ([Kappa-Chart-Surfaces], 1.0.0a285) -------------

PORT = ('port CK.P agg A 50 claims sev lognorm 50 cv 1.5 poisson '
        'agg B 30 claims sev lognorm 40 cv 1.2 poisson')


@pytest.fixture(scope='module')
def port():
    return build(PORT)


def test_one_name_across_three_sources(port, joint):
    """The question is the same one: what does each part contribute, given the
    whole. Three shapes answer it, so one name serves them."""
    assert 'kappa' in available_charts(port)
    assert 'kappa' in available_charts(joint)
    assert 'kappa' in available_charts(build(LAYER))


def test_a_plain_aggregate_has_no_parts_to_condition_on():
    """No cession, no split, nothing to draw."""
    plain = build('agg CK.Plain 10 claims sev lognorm 50 cv 1.5 poisson')
    assert 'kappa' not in available_charts(plain)


def test_the_book_panel_is_the_overview_panel_served_alone(port):
    """Built by the same function, so the two documents cannot drift."""
    alone = build_chart_doc(port, 'kappa')
    overview = build_chart_doc(port, 'port')
    assert [p.id for p in alone.panels] == ['kappa']
    assert alone.panels[0].aspect == 'equal'
    mine = [s for s in alone.series if s.panel_id == 'kappa']
    theirs = [s for s in overview.series if s.panel_id == 'kappa']
    assert [s.name for s in mine] == [s.name for s in theirs]
    for a, b in zip(mine, theirs):
        assert a.y_values == b.y_values
        assert a.x_lattice == b.x_lattice
        assert a.role == b.role


def test_the_book_panel_carries_no_band(port):
    """Unit kappas come off the independence trick, not off a stored joint.

    A conditional band there would be new machinery with no session behind
    it, so the honest picture is the mean curves.
    """
    doc = build_chart_doc(port, 'kappa')
    assert not [s for s in doc.series if s.y2 is not None]


def test_the_book_curves_sum_to_the_diagonal(port):
    """The identity the panel is read against.

    To kappa's own error rather than to the bit: the curves must sum to the
    total by construction, so the residual of that identity **is** the error,
    and the floor the panel drops points at is set from it (parts per
    thousand at the floor, parts per hundred past it).
    """
    doc = build_chart_doc(port, 'kappa')
    units = [np.asarray(s.y_values, dtype=float)
             for s in doc.series if s.role == 'unit']
    total = np.asarray(
        next(s for s in doc.series if s.role == 'total').y_values, dtype=float)
    residual = np.abs(sum(units) - total) / np.where(total > 0, total, np.nan)
    assert np.nanmax(residual) < 1e-3


def test_the_book_keeps_its_own_primary_picture(port):
    """Kappa is a reading of a book, not the book's own picture."""
    from aggregate.charts import primary_chart

    assert primary_chart(port) == 'port'


def test_the_delegate_reads_the_held_joint(joint):
    """Decision 6 end to end: drawing after an allocation costs a lookup.

    Under the Palm default no joint is built at all, so the memo reading is
    exercised where the joint is actually wanted: the ``bands`` overlay.
    """
    agg = build(LAYER)
    build_chart_doc(agg, 'kappa', bands=True)
    first = agg.occ_joint(views=('gross', 'ceded'))
    build_chart_doc(agg, 'kappa', bands=True)
    assert agg.occ_joint(views=('gross', 'ceded')) is first


# --- the Palm route ([Palm-Kappa-Chart], 1.0.0a368) -------------------------

# A frequency outside the pgf-derivative set: the fallback's load-bearing case.
FALLBACK = ('agg CK.F 10 claims 500 xs 0 sev lognorm 50 cv 1.5 '
            'occurrence net of 50 xs 50 logarithmic')


@pytest.fixture(scope='module')
def palm_doc():
    return build_chart_doc(build(TOWER), 'kappa')


def test_palm_is_the_default_route(palm_doc):
    """Mean curves off the 1-D identity: no band, and the meta says why."""
    assert palm_doc.meta['route'] == 'palm'
    assert palm_doc.meta['band'] == 'none'
    assert not [s for s in palm_doc.series if s.y2 is not None]
    assert [p.id for p in palm_doc.panels] == ['cession', 'share']
    assert {p.x_axis for p in palm_doc.panels} == {'outcome'}
    assert palm_doc.panels[0].aspect == 'equal'


def test_palm_serves_one_curve_per_layer(palm_doc):
    """The picture the 2-D route cannot draw: each layer's own cession."""
    layers = [s for s in series_of(palm_doc, 'cession')
              if s.role == 'ceded' and s.name != 'E[ceded | gross]']
    assert [s.name for s in layers] == ['occ 50 xs 50', 'occ 100 xs 100']


def test_palm_layers_foot_to_the_total_and_the_identity(palm_doc):
    """Linearity of the conditional mean, cell by cell.

    Per-layer ceders sum identically to the program ceder (the validator
    rejects overlapping cessions), so the layer curves sum to the total; and
    ``total + net = g`` is the same identity footing as the band chart.
    """
    g = np.asarray(named(palm_doc, 'cession', 'gross').y_values, dtype=float)
    total = np.asarray(
        named(palm_doc, 'cession', 'E[ceded | gross]').y_values, dtype=float)
    net = np.asarray(
        named(palm_doc, 'cession', 'E[net | gross]').y_values, dtype=float)
    layers = sum(np.asarray(s.y_values, dtype=float)
                 for s in series_of(palm_doc, 'cession')
                 if s.role == 'ceded' and s.name != 'E[ceded | gross]')
    assert layers == pytest.approx(total, abs=1e-9)
    assert (total + net) == pytest.approx(g, abs=1e-9)


def test_palm_total_agrees_with_the_2d_route():
    """The two computations of one conditional mean meet in the middle.

    The Palm curve lives on the fine model grid, the 2-D ``exeqa`` curve on
    the joint's budget grid; interpolating the fine curve at the coarse
    grid's points, the gap is the joint's budget-grid coarseness. Observed
    at implementation (2026-09-29): max 0.20% relative, median 0.017%, over
    the 353 coarse cells clearing the 5% floor; the 5% tolerance absorbs
    the interpolation, not the identity.
    """
    agg = build(TOWER)
    palm = build_chart_doc(agg, 'kappa')
    band = build_chart_doc(agg.occ_joint(views=('gross', 'ceded')), 'kappa')
    fine = named(palm, 'cession', 'E[ceded | gross]')
    coarse = named(band, 'cession', 'E[ceded | gross]')
    g_fine = np.asarray(fine.x_values, dtype=float)
    y_fine = np.asarray(fine.y_values, dtype=float)
    g_coarse = np.asarray(coarse.x_values, dtype=float)
    y_coarse = np.asarray(coarse.y_values, dtype=float)
    inside = (g_coarse >= g_fine[0]) & (g_coarse <= g_fine[-1])
    at = np.interp(g_coarse[inside], g_fine, y_fine)
    ok = y_coarse[inside] > 0.05 * np.nanmax(y_coarse)
    rel = np.abs(at[ok] - y_coarse[inside][ok]) / y_coarse[inside][ok]
    assert np.nanmax(rel) < 0.05


def test_bands_opt_back_in():
    """``bands=True``: the joint's band series ride on the Palm chart."""
    agg = build(TOWER)
    doc = build_chart_doc(agg, 'kappa', bands=True)
    assert doc.meta['route'] == 'palm'
    assert doc.meta['band'] == 'percentile'
    pure = build_chart_doc(agg.occ_joint(views=('gross', 'ceded')), 'kappa')
    for role in ('ceded', 'net'):
        mine = next(s for s in series_of(doc, 'cession')
                    if s.y2 is not None and s.role == role)
        theirs = next(s for s in series_of(pure, 'cession')
                      if s.y2 is not None and s.role == role)
        assert mine.name == theirs.name
        assert mine.y_values == theirs.y_values
        assert mine.y2 == theirs.y2
        assert mine.x_lattice == theirs.x_lattice


def test_fallback_for_a_frequency_without_a_pgf_derivative():
    """Chart availability moves for no object: logarithmic still serves.

    The served document is the band chart off the implied joint, unchanged
    apart from the ``route`` meta key that says which way it came.
    """
    agg = build(FALLBACK)
    assert 'kappa' in available_charts(agg)
    doc = build_chart_doc(agg, 'kappa')
    assert doc.meta['route'] == '2d-fallback'
    assert doc.meta['band'] == 'percentile'
    direct = build_chart_doc(agg.occ_joint(views=('gross', 'ceded')), 'kappa')
    assert doc.series == direct.series
    assert doc.axes == direct.axes
    assert doc.panels == direct.panels


def test_single_layer_serves_the_total_only():
    """One layer's curve is the total: serve one line, not two identical."""
    doc = build_chart_doc(build(LAYER), 'kappa')
    assert doc.meta['route'] == 'palm'
    cession = [s.name for s in series_of(doc, 'cession')]
    assert cession == ['E[ceded | gross]', 'E[net | gross]', 'gross']
    assert [s.name for s in series_of(doc, 'share')] == [
        'ceded share', 'most the program could cede']


def test_palm_doc_round_trips_hash_for_hash(palm_doc):
    back = load_chart_doc(json.loads(json.dumps(canonical_dict(palm_doc))))
    assert doc_hash(back) == palm_doc.hash
    assert back == palm_doc


def test_palm_doc_renders(palm_doc):
    """The generic renderer draws it: no per-chart matplotlib code exists."""
    import matplotlib
    matplotlib.use('Agg')
    from aggregate.plots import plot_chartdoc

    fig = plot_chartdoc(palm_doc)
    assert len(fig.axes) >= 2


# --- the P&L surface ([PnL-Kappa-Chart], 1.0.0a369) -------------------------

TOWER_PNL = ('xpnl CK.PT 1000 prem less agg CK.PTE 1000 prem at 70% lr '
             'sev lognorm 100 cv 2 '
             'occurrence ceded to 500 xs 500 deposit 100 poisson')
PEEL_PNL = ('xpnl CK.PP 1000 premium less agg CK.PPE 1000 premium at 70% lr '
            'sev lognorm 100 cv 2 '
            'occurrence net of 100 xs 100 deposit 60 and 300 xs 200 '
            'deposit 40 poisson peel top-down')
PLAIN_PNL = ('pnl CK.PB 1000 premium less agg CK.PBL 100 claims '
             'sev lognorm 5 cv 2 poisson')


def test_a_reinsured_pnl_lights_kappa():
    """The engine look-through: a cession lights the tab, no cession stays dark."""
    assert 'kappa' in available_charts(build(TOWER_PNL))
    assert 'kappa' in available_charts(build(PEEL_PNL))
    assert 'kappa' not in available_charts(build(PLAIN_PNL))


def test_the_pnl_doc_is_the_engines_own():
    """A thin delegate, the a367 ``chart_reins`` pattern: nothing to drift."""
    pn = build(TOWER_PNL)
    assert (build_chart_doc(pn, 'kappa').hash
            == build_chart_doc(pn.engine, 'kappa').hash)


def test_the_peel_serves_one_curve_per_layer():
    """The multi-layer picture the 2-D route cannot produce, on a P&L."""
    doc = build_chart_doc(build(PEEL_PNL), 'kappa')
    assert doc.meta['route'] == 'palm'
    layers = [s.name for s in doc.series
              if s.panel_id == 'cession' and s.role == 'ceded'
              and s.name != 'E[ceded | gross]']
    assert layers == ['occ 100 xs 100', 'occ 300 xs 200']


def test_the_pnl_doc_round_trips_and_renders():
    doc = build_chart_doc(build(TOWER_PNL), 'kappa')
    back = load_chart_doc(json.loads(json.dumps(canonical_dict(doc))))
    assert doc_hash(back) == doc.hash
    import matplotlib
    matplotlib.use('Agg')
    from aggregate.plots import plot_chartdoc

    assert len(plot_chartdoc(doc).axes) >= 2


def test_supports_pgf_prime_truth_table():
    """The capability the route switch keys on, family by family."""
    from aggregate.distributions import Frequency

    supported = {
        'poisson': (0, 0), 'binomial': (0.4, 0), 'negbin': (2.25, 0),
        'geometric': (0, 0), 'fixed': (0, 0), 'bernoulli': (0, 0),
        'empirical': (np.array([0, 1, 3]), np.array([0.4, 0.4, 0.2])),
        'gamma': (0.5, 0), 'delaporte': (0.5, 0.3), 'ig': (0.6, 0),
    }
    unsupported = {
        'logarithmic': (0, 0), 'sig': (0.5, 0.5), 'beta': (2.0, 3.0),
        'sichel': (0.5, 0.5), 'neymana': (2.0, 0), 'pascal': (1.5, 2.0),
    }
    for name, (a, b) in supported.items():
        assert Frequency(name, a, b, False, np.nan).supports_pgf_prime, name
    for name, (a, b) in unsupported.items():
        assert not Frequency(name, a, b, False, np.nan).supports_pgf_prime, \
            name
