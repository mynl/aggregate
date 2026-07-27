"""First-class-citizen surface consistency ([FCC-Surface-Sweep], a149).

``dev/FEATURES.csv`` is the capability / consistency matrix over the nine
first-class classes; this module is its executable half. It checks the
*shared* surfaces actually exist and agree across classes:

* every FCC class answers to ``info`` (terse, fixed-layout, ``info_row``-shaped);
* every DecL-creatable class answers to ``pprogram`` / ``pprogram_html``;
* the moment families are ``actual_*`` (theory) and ``est_*`` (realised grid),
  with no leftover ``agg_*`` / bare ``mean`` on the renamed classes;
* ``prob_eq_0`` is the sign-neutral break-even atom on Aggregate / Portfolio /
  PnL;
* ``tail_df`` is a **property** everywhere, with ``tail_periods_df(periods=)``
  the parametrized worker on Aggregate / Portfolio;
* the ``*_description`` / ``*_explanation`` narrative pairs are complete.
"""
import numpy as np
import pandas as pd
import pytest

from aggregate import build
from aggregate.bounds import Bounds
from aggregate.constants import INFO_LABEL_WIDTH


# ---------------------------------------------------------------------------
# fixtures -- one live object per first-class class
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def agg():
    return build('agg FCC.Agg 100 claims sev lognorm 100 cv 2 '
                 'occurrence net of 50 xs 50 poisson')


@pytest.fixture(scope='module')
def port():
    return build('port FCC.Port '
                 'agg A 100 claims sev lognorm 100 cv 2 '
                 'occurrence net of 500 xs 500 poisson '
                 'agg B 50 claims sev gamma 50 cv 1 poisson')


@pytest.fixture(scope='module')
def pnl():
    return build('pnl FCC.PnL 1000 premium less agg FCC.PnL_e '
                 '1000 prem at 70% lr sev lognorm 100 cv 2 poisson')


@pytest.fixture(scope='module')
def biv():
    return build('bivariate FCC.Biv 25 claims '
                 'agg Wind dfreq [0 1] [.3 .7] sev lognorm 40 cv 1.2 '
                 'agg Flood dfreq [0 1] [.5 .5] sev lognorm 60 cv 1.5 '
                 'copula gumbel 0.4 poisson')


# ---------------------------------------------------------------------------
# info -- every first-class class carries one, fixed-layout
# ---------------------------------------------------------------------------

def _labels(info):
    """Leading label of each info line (text before the value column)."""
    return [line[:INFO_LABEL_WIDTH].rstrip() for line in info.split('\n')]


def test_info_on_every_first_class_class(agg, port, pnl, biv):
    objs = {
        'Aggregate': agg,
        'Portfolio': port,
        'BivariateAggregate': biv,
        'PnL': pnl,
        'Severity': agg.sevs[0],
        'Frequency': agg.frequency,
        'Bounds': Bounds(agg, premium=agg.actual_m * 1.1),
        'AllocationBounds': port.allocation_bounds(p=0.99),
        'PricingBounds': port.pricing_bounds(port.agg_list[0], p=0.99),
    }
    for name, obj in objs.items():
        info = obj.info
        assert isinstance(info, str) and info, f'{name}.info is empty'
        labels = _labels(info)
        assert labels[0].endswith('object') or 'object' in labels[0], \
            f'{name}.info does not open with an identity row: {labels[0]!r}'
        # fixed-layout: every row is label-padded, no colons, no blank lines
        assert all(line for line in info.split('\n')), f'{name}.info has a blank line'
        assert ':' not in ''.join(labels), f'{name}.info labels carry a colon'


def test_severity_and_frequency_info_are_stable_across_instances():
    """Two objects of one class emit the same rows in the same order."""
    a = build('agg FCC.S1 5 claims sev lognorm 50 cv 1 poisson', update=False)
    b = build('agg FCC.S2 1 claim 100 xs 50 sev gamma 20 cv .5 binomial 0.5',
              update=False)
    assert _labels(a.sevs[0].info) == _labels(b.sevs[0].info)
    assert _labels(a.frequency.info) == _labels(b.frequency.info)


# ---------------------------------------------------------------------------
# pprogram -- every DecL-creatable class round-trips its declaration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('fixture', ['agg', 'port', 'pnl', 'biv'])
def test_pprogram_and_html_on_decl_creatable(fixture, request):
    obj = request.getfixturevalue(fixture)
    pp = obj.pprogram
    assert isinstance(pp, str) and pp.strip(), f'{fixture}.pprogram is empty'
    assert obj.name.split('.')[-1] in pp
    html = obj.pprogram_html
    assert isinstance(html, str) and '<' in html


def test_severity_pprogram_round_trips_a_standalone_declaration():
    """a150: ``sev`` is DecL-creatable, so it renders a canonical program too."""
    sev = build('sev FCC.Sev lognorm 100 cv 2')
    assert sev.program, 'build should stamp the source text on a standalone sev'
    assert 'FCC.Sev' in sev.pprogram
    assert '<' in sev.pprogram_html


def test_inline_severity_has_no_program_of_its_own(agg):
    """An inline ``sev`` clause belongs to its Aggregate, which owns the text."""
    assert agg.sevs[0].pprogram == ''


def test_pnl_program_falls_back_to_the_engine(pnl):
    """The declaration is stamped on the P&L; the engine carries the same one."""
    assert pnl.program.startswith('pnl ')
    assert pnl.engine is not None
    # clearing the stamp falls through to the engine, not to ''
    stamped, pnl.program = pnl.program, ''
    try:
        assert pnl.program == pnl.engine.program
    finally:
        pnl.program = stamped


# ---------------------------------------------------------------------------
# moments -- actual_* (theory) and est_* (realised grid)
# ---------------------------------------------------------------------------

def test_actual_and_est_families_on_aggregate_and_portfolio(agg, port):
    for obj in (agg, port):
        assert obj.actual_sd == pytest.approx(obj.actual_m * obj.actual_cv)
        assert obj.actual_var == pytest.approx(obj.actual_sd ** 2)
        assert not hasattr(obj, 'agg_m')
    # on a clean (gross) book the realised grid reproduces the theory; on a
    # reinsured one ``actual_*`` stays gross while ``est_*`` is the net output
    clean = build('agg FCC.Clean1 100 claims sev lognorm 100 cv 2 poisson')
    assert clean.est_m == pytest.approx(clean.actual_m, rel=1e-3)
    assert agg.est_m < agg.actual_m           # net of 50 xs 50


def test_severity_actual_moments_match_scipy(agg):
    sv = agg.sevs[0]
    assert sv.actual_m == pytest.approx(sv.mean())
    assert sv.actual_sd == pytest.approx(sv.std())
    assert sv.actual_var == pytest.approx(sv.actual_sd ** 2)
    assert sv.actual_cv == pytest.approx(sv.actual_sd / sv.actual_m)


def test_severity_actual_moments_are_exact_on_a_discrete_law():
    a = build('agg FCC.Dice 1 claim dsev [1:6] fixed', update=False)
    sv = a.sevs[0]
    assert sv.actual_m == pytest.approx(3.5, abs=1e-12)
    assert sv.actual_var == pytest.approx(35 / 12, abs=1e-12)
    assert sv.actual_skew == pytest.approx(0.0, abs=1e-12)


def test_pnl_moments_are_est_not_actual(pnl):
    """A P&L is evaluated over the source's *realised* grid, so ``est_*``."""
    assert pnl.est_sd > 0
    assert pnl.est_cv == pytest.approx(pnl.est_sd / pnl.est_m)
    for gone in ('mean', 'sd', 'cv', 'skew'):
        assert not hasattr(pnl, gone), f'PnL still exposes {gone}'


# ---------------------------------------------------------------------------
# prob_eq_0 -- the sign-neutral break-even / no-loss atom
# ---------------------------------------------------------------------------

def test_prob_eq_0_is_exact_on_a_discrete_aggregate():
    """A discrete book puts genuine mass at exactly zero: ``P(N = 0)``."""
    a = build('agg FCC.Z dfreq [0 1 2] [.4 .35 .25] dsev [10 20]')
    assert a.prob_eq_0 == pytest.approx(0.4, abs=1e-12)


def test_prob_eq_0_brackets_p_no_claims_on_a_continuous_severity():
    """The zero bucket carries ``P(N = 0)`` *plus* losses under one bucket."""
    a = build('agg FCC.ZC 1.5 claims sev lognorm 100 cv 2 poisson')
    assert a.prob_eq_0 == pytest.approx(a.pmf(0.0))
    assert np.exp(-1.5) <= a.prob_eq_0 < np.exp(-1.5) + 0.05


def test_prob_eq_0_is_none_before_update():
    a = build('agg FCC.NU 5 claims sev lognorm 50 cv 1 poisson', update=False)
    assert a.prob_eq_0 is None


def test_portfolio_prob_eq_0_multiplies_under_independence():
    p = build('port FCC.Ind '
              'agg U 1.5 claims sev lognorm 100 cv 2 poisson '
              'agg V 0.8 claims sev lognorm 50 cv 1 poisson')
    expected = p.agg_list[0].prob_eq_0 * p.agg_list[1].prob_eq_0
    assert p.prob_eq_0 == pytest.approx(expected, rel=1e-6)


def test_pnl_prob_eq_0_is_the_break_even_atom():
    p = build('pnl FCC.BE 100 prem less agg FCC.BE_e dfreq [1] '
              'dsev [0 100 250] [.2 .5 .3]')
    assert p.prob_eq_0 == pytest.approx(0.5, abs=1e-12)


# ---------------------------------------------------------------------------
# tail_df is a property; tail_periods_df is the worker
# ---------------------------------------------------------------------------

def test_tail_df_is_a_property_on_every_class_that_has_it(agg, port, biv):
    for obj in (agg, port, biv):
        assert isinstance(type(obj).tail_df, property), \
            f'{type(obj).__name__}.tail_df is not a property'
        assert isinstance(obj.tail_df, pd.DataFrame)


def test_tail_periods_df_takes_a_custom_ladder(agg, port):
    for obj in (agg, port):
        df = obj.tail_periods_df([3, 7])
        assert list(df.index.get_level_values(-1).unique()) == [3, 7]
        # the property is the default ladder, not the custom one
        assert len(obj.tail_df) > len(df)


def test_tail_df_is_none_before_update():
    a = build('agg FCC.NT 5 claims sev lognorm 50 cv 1 poisson', update=False)
    assert a.tail_df is None
    assert a.tail_periods_df([10]) is None


# ---------------------------------------------------------------------------
# narrative pairs -- description (short) + explanation (long)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('fixture,pairs', [
    ('agg', ['tail', 'bs']),
    ('port', ['tail', 'bs']),
    ('biv', ['tail', 'bs']),
])
def test_description_explanation_pairs_are_complete(fixture, pairs, request):
    obj = request.getfixturevalue(fixture)
    for stem in pairs:
        short = getattr(obj, f'{stem}_description')
        long = getattr(obj, f'{stem}_explanation')
        assert isinstance(short, str) and short
        assert isinstance(long, str) and long
        assert len(long) > len(short), f'{fixture}.{stem}_explanation is not longer'


def test_severity_and_frequency_gained_tail_explanation(agg):
    for obj in (agg.sevs[0], agg.frequency):
        assert isinstance(obj.tail_description, str) and obj.tail_description
        assert len(obj.tail_explanation) > len(obj.tail_description)
    # support_description folded into the tail_* pair and retired
    assert not hasattr(agg.sevs[0], 'support_description')


def test_severity_tail_description_reveals_the_declared_layer():
    a = build('agg FCC.L 1 claim 500 xs 250 sev lognorm 100 cv 2 fixed',
              update=False)
    assert '500 xs 250' in a.sevs[0].tail_description
    d = build('agg FCC.D 1 claim dsev [1:6] fixed', update=False)
    assert 'atoms' in d.sevs[0].tail_description


def test_bivariate_validation_explanation(biv):
    txt = biv.validation_explanation
    assert 'Tail deficit' in txt
    for unit in biv.unit_names:
        assert unit in txt


def test_frequency_name_and_explanation(agg):
    f = agg.frequency
    assert f.name == f.freq_name == 'poisson'
    assert 'Poisson-dispersed' in f.tail_explanation


# ---------------------------------------------------------------------------
# Portfolio reinsurance look-throughs
# ---------------------------------------------------------------------------

def test_portfolio_reins_lookthrough_names_every_unit(port):
    kinds, desc = port.reins_kinds, port.reins_description
    for a in port.agg_list:
        assert f'Unit {a.name}' in kinds
        assert f'Unit {a.name}' in desc
    assert 'occurrence only' in kinds
    assert 'no reinsurance' in desc


def test_portfolio_reins_lookthrough_collapses_on_a_clean_book():
    p = build('port FCC.Clean '
              'agg C1 5 claims sev lognorm 50 cv 1 poisson '
              'agg C2 3 claims sev gamma 20 cv .5 poisson', update=False)
    assert p.reins_kinds == 'None'
    assert p.reins_description == 'No reinsurance'


# ---------------------------------------------------------------------------
# the one-line text intro
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('fixture', ['agg', 'port'])
def test_text_info_blob_is_one_line_and_states_validation(fixture, request):
    obj = request.getfixturevalue(fixture)
    blob = obj._text_info_blob()
    assert '\n' not in blob
    assert blob.endswith(f'Validation: {obj.validation_explanation}.')


# ---------------------------------------------------------------------------
# help -- the universal discovery front door (a150 [FCC-Help-Mixin])
# ---------------------------------------------------------------------------

def test_help_on_every_column_of_the_matrix(agg, port, pnl, biv):
    """``.help`` reaches every class with a column in ``dev/FEATURES.csv``.

    Before a150 it was nine copy-pasted methods and ``Frequency`` /
    ``GridDistribution`` had none; ``HelpMixin`` closed both holes.
    """
    from aggregate.bounds import AllocationBounds, PricingBounds
    from aggregate.spectral import Distortion
    from aggregate._grid_distribution import GridDistribution
    from aggregate._frequency import Frequency

    hosts = [agg, port, pnl, biv, agg.sevs[0], agg.frequency,
             agg._grid_distribution(), Distortion('ph', 0.5),
             Bounds(agg, premium=agg.actual_m * 1.1)]
    for obj in hosts:
        assert callable(getattr(obj, 'help', None)), \
            f'{type(obj).__name__} has no help()'
    # the classes too, so a future refactor cannot lose the mixin silently
    for cls in (GridDistribution, Frequency, Distortion,
                AllocationBounds, PricingBounds):
        assert 'help' in dir(cls), f'{cls.__name__} lost HelpMixin'


def test_help_runs_and_prints_matching_names(agg, capsys):
    agg.frequency.help('freq_p')
    out = capsys.readouterr().out
    assert 'freq_pgf' in out and 'freq_p0' in out


# ---------------------------------------------------------------------------
# tvar -- the flagship risk measure, on every class that carries a quantile
# ---------------------------------------------------------------------------

def test_pnl_tvar_delegates_to_the_result_grid(pnl):
    """a150: PnL delegated q/var/cdf/sf but not tvar -- the gap is closed."""
    for p in (0.5, 0.9, 0.99):
        assert pnl.tvar(p) == pnl.result.tvar(p)
