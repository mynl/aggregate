"""First-class display / exhibit surface of :class:`ReinstatementAnalysis`.

Covers the ``[trimmings]`` workstream of dev/plan-reinstatements.md: ``info``,
``_repr_html_``, ``plot`` (the four-panel mosaic), ``density_df``,
``validation_explanation``, the reused ``bs_*`` joint-grid sizing audit, and
``qd`` support. These are display surfaces, so the tests assert structure /
non-emptiness and that nothing raises, not pixel content.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

from aggregate import build, qd
from aggregate.reinstatement import ReinstatementTerms, ReinstatementAnalysis


def _analysis(rates=(0.0, 0.5, 1.0), deposit=10.0):
    a = build('agg Cat 3 claims sev lognorm 60 cv 2.5 '
              'occurrence net of 100 xs 100 poisson')
    return a.reinstatement_analysis(
        gross_premium=1000.0,
        terms=ReinstatementTerms(100.0, rates, deposit))


# ----------------------------------------------------------------------
# info / narratives
# ----------------------------------------------------------------------
def test_info_is_structured_text():
    an = _analysis()
    s = an.info()
    assert isinstance(s, str)
    for label in ('gross premium', 'rate on line', 'recovery capacity Y',
                  'E[net UW]', 'validation'):
        assert label in s


def test_validation_explanation_reports_pass():
    an = _analysis()
    txt = an.validation_explanation
    assert 'PASS' in txt and 'tolerance' in txt
    assert an._validation_passes() is True


def test_repr_html_has_tables():
    an = _analysis()
    html = an._repr_html_()
    assert '<table' in html and 'Reinstatement analysis' in html


# ----------------------------------------------------------------------
# density_df (per leg)
# ----------------------------------------------------------------------
def test_density_df_default_and_named_leg():
    an = _analysis()
    df = an.density_df()                      # default net_uw
    assert list(df.columns) == ['p', 'F', 'S']
    assert df['p'].sum() == pytest.approx(1.0, abs=1e-6)
    assert df.index.name == 'net_uw'
    # a different leg
    cp = an.density_df('ceded_premium')
    assert cp.index.name == 'ceded_premium'
    # F is a nondecreasing cdf ending at ~1
    assert np.all(np.diff(cp['F'].to_numpy()) >= -1e-12)
    assert cp['F'].iloc[-1] == pytest.approx(1.0, abs=1e-6)


def test_density_df_unknown_leg_raises():
    an = _analysis()
    with pytest.raises(ValueError, match='unknown leg'):
        an.density_df('not_a_leg')


# ----------------------------------------------------------------------
# bs_* sizing audit (reused from the BivariateAggregate holder)
# ----------------------------------------------------------------------
def test_bs_audit_reused_from_holder():
    an = _analysis()
    assert an.joint_aggregate is not None
    assert 'grid' in an.bs_description
    bw = an.bs_window_df
    assert bw is not None and len(bw) == 2          # one row per axis
    assert 'joint' in an.bs_explanation


def test_bs_window_df_none_without_holder():
    # a directly-constructed analysis (no holder) degrades gracefully
    a = build('agg Cat 3 claims sev lognorm 60 cv 2.5 '
              'occurrence net of 100 xs 100 poisson')
    joint = a.occ_bivariate(views=('gross', 'ceded')).bivariate
    an = ReinstatementAnalysis(joint, ReinstatementTerms(100.0, (1.0,), 10.0),
                               1000.0)
    assert an.bs_window_df is None
    assert 'joint grid' in an.bs_description


# ----------------------------------------------------------------------
# plot mosaic
# ----------------------------------------------------------------------
def test_plot_has_four_panels():
    an = _analysis()
    fig = an.plot()
    assert an.figure is fig
    titles = {ax.get_title() for ax in fig.axes if ax.get_title()}
    assert {'Joint (L, R) log density', 'Reinstatement economics',
            'Underwriting result: gross vs net', 'Cession impact'} <= titles


def test_pnl_reinstatement_analysis_plots_mosaic():
    # a DecL reinstatement pnl returns a PnL whose attached .analysis plots the
    # stochastic-ceded mosaic (p.plot() itself is the generic net-result plot).
    p = build('pnl Cat 10000 premium less agg Cat_e 10000 prem at 85% lr sev lognorm 50 cv 3 '
              'occurrence net of 100 xs 100 rol 18% '
              'reinstatements [0 1] poisson')
    fig = p.analysis.plot()
    titles = {ax.get_title() for ax in fig.axes if ax.get_title()}
    assert 'Cession impact' in titles


# ----------------------------------------------------------------------
# qd support
# ----------------------------------------------------------------------
def test_qd_runs(capsys):
    an = _analysis()
    qd(an)
    out = capsys.readouterr().out
    assert 'Reinstatement analysis' in out
    assert 'Underwriting' in out          # summary_df row
    assert 'return_period' in out         # tail_df index
