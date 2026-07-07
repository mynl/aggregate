"""[One-Classifier-Fix] acceptance: the reinsurance composition matrix.

The pnl/xpnl assembly classifies the occurrence tier {none | gc |
reinstatements} and the aggregate tier {none | gc | feature} independently
and dispatches on the pair (``dev/PLAN-A.md``). These tests pin the two
previously silently-wrong cells -- a feature on the aggregate cover composed
with an inuring occurrence program, guaranteed-cost
([Var-Feature-Composed-With-Occ-Program]) or reinstated
([Reinstatements-Dropped-By-Feature-Branch]) -- plus the routing of every
cell. Companion coverage: zero-premium cessions in
``test_pnl_ceded_premium.py``, the one-step plain walk in
``test_pnl_engine_source.py``.
"""
import warnings

import numpy as np
import pytest

from aggregate import Underwriter
from aggregate._pnl import PnL

warnings.filterwarnings('ignore', message='.*heavy right tail.*')

# the author's CatBook acceptance program (dev/PLAN-A.md), tempered tail for
# test speed (pareto 1.6 vs 1.2 -- same structure, smaller grid)
_ENGINE = ('agg CB{tag} as "Gross Loss" 12500 premium as "Plan Premium" '
           'at 80% lr 5000 xs 0 as "Basic Limits" '
           'sev 200 * pareto 1.6 - 200 '
           'occurrence net of 4750 xs 250 rate 55% {occ_extra}'
           'as "Cat Program" poisson {agg_clause}')
_SWING = ('aggregate net of 1000 xs 3100 '
          'swing basic 100 lcm 1.5 min 200 max 500 as "Swing Program"')
_P_G, _PC_OCC = 12500.0, 0.55 * 12500.0


@pytest.fixture(scope='module')
def uw():
    u = Underwriter(databases=None, update=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        u(_ENGINE.format(tag='GF', occ_extra='cede 10% ',
                         agg_clause=_SWING))
        u(_ENGINE.format(tag='RN', occ_extra='no reinstatements ',
                         agg_clause=''))
        u(_ENGINE.format(tag='RF', occ_extra='no reinstatements ',
                         agg_clause=_SWING))
    return u


def _build(uw, face, ref):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return uw(f'{face} W{face}{ref} as "W" inherit premium '
                  f'less agg.{ref} less 1000 fixed expenses as "GE"')


def _g(t):
    return np.minimum(np.maximum(np.asarray(t, dtype=float) - 3100.0, 0.0),
                      1000.0)


def _phi(a):
    return np.clip(100.0 + 1.5 * np.asarray(a, dtype=float), 200.0, 500.0)


# ----------------------------------------------------------------------
# (GC occ, feature agg): the composed 1-D cell
# ----------------------------------------------------------------------
def test_gc_feat_consolidated_books_occ_constants(uw):
    """The consolidated net premium carries the inuring occ program's
    constants ``- pc_occ + c_occ`` and the feature map, all exact against a
    hand pushforward of the net-of-occ marginal."""
    p = _build(uw, 'pnl', 'CBGF')
    c_occ = 0.10 * _PC_OCC
    rd = p._source.reins_density_df
    xs = rd['loss'].to_numpy()
    pn = rd['p_agg_net_occ'].to_numpy()
    pn = pn / pn.sum()
    hand_np = _P_G - _PC_OCC + c_occ - float((_phi(_g(xs)) * pn).sum())
    booked = p.stats_df.xs('net premium', level='Line').iloc[0]
    assert booked['EX'] == pytest.approx(hand_np, abs=1e-6)
    assert booked['SD'] > 0                    # the swing premium is live
    hand_nl = -float(((xs - _g(xs)) * pn).sum())
    assert p.stats_df.xs('Gross Loss (net)', level='Line').iloc[0]['EX'] \
        == pytest.approx(hand_nl, abs=1e-6)
    assert p.economics['pc_occ'] == pytest.approx(_PC_OCC)
    assert p.economics['c_occ'] == pytest.approx(c_occ)
    # consolidated rides the one net-occ source -> scenario ladder
    assert 'κ01' in p.stats_df.columns


def test_gc_feat_walk_has_occ_step_and_true_gross(uw):
    """The composed walk: Gross -> ceded occ -> feature cover -> All, a
    per-atom tower over the occurrence (gross, ceded) joint: the gross step
    books the TRUE gross loss (the mislabeled-net defect), the occ step is
    present with its economics, the feature rides the net-of-occ subject.
    Rows agree with the engine's exact 1-D marginals to joint-grid accuracy;
    the ladder is the footing scenario (κ) pass."""
    x = _build(uw, 'xpnl', 'CBGF')
    s = x.stats_df
    steps = list(dict.fromkeys(s.index.get_level_values('Step')))
    assert steps == ['Gross', 'Cat Program', 'Swing Program', 'All']
    # the source is the occurrence joint; the engine ref carries the exact
    # 1-D marginals ([Engine-Reference-On-PnL])
    rd = x.engine.reins_density_df
    xs = rd['loss'].to_numpy()

    def m(col, f=lambda t: t):
        p_ = rd[col].to_numpy()
        return float((f(xs) * (p_ / p_.sum())).sum())

    assert s.loc[('Gross', 'Obligation', 'Gross Loss'), 'EX'] \
        == pytest.approx(-m('p_agg_gross'), rel=1e-3)
    assert s.loc[('Cat Program', 'Obligation', 'Cat Program recovery'),
                 'EX'] == pytest.approx(m('p_agg_ceded_occ'), rel=1e-3)
    assert s.loc[('Cat Program', 'Consideration', 'Cat Program premium'),
                 'EX'] == pytest.approx(-_PC_OCC)
    assert s.loc[('Swing Program', 'Consideration', 'Swing Program premium'),
                 'EX'] == pytest.approx(-m('p_agg_net_occ',
                                           lambda t: _phi(_g(t))), rel=2e-2)
    assert s.loc[('Swing Program', 'Obligation', 'Swing Program recovery'),
                 'EX'] == pytest.approx(m('p_agg_net_occ', _g), rel=2e-2)
    # the EX column foots exactly (per-atom partial sums)...
    legs = [i for i in s.index if i[2] not in ('Total', 'Net', 'Impact')]
    assert s['EX'].loc[legs].sum() == pytest.approx(
        s.loc[('All', 'Margin', 'Total'), 'EX'], abs=1e-9)
    # ...and the consolidated pnl (exact net-occ marginal) agrees with the
    # joint-riding walk to joint-grid accuracy
    p = _build(uw, 'pnl', 'CBGF')
    assert p.mean == pytest.approx(
        s.loc[('All', 'Margin', 'Total'), 'EX'], rel=5e-3)
    # one shared joint -> the scenario ladder
    assert 'κ01' in s.columns and 'P01' not in s.columns


# ----------------------------------------------------------------------
# (reinstatements occ, feature agg): the joint-riding feature cell
# ----------------------------------------------------------------------
def test_reinst_feat_routes_through_joint_and_keeps_cap(uw):
    """The reinstatements clause survives an aggregate feature
    ([Reinstatements-Dropped-By-Feature-Branch]): the walk routes through
    the (L, R) joint, the occ recovery keeps its annual cap, every shared
    row equals the feature-less build, and the swing tier books its collared
    stochastic premium off the same joint."""
    a = _build(uw, 'xpnl', 'CBRN')       # occ 'no reinstatements', no agg
    b = _build(uw, 'xpnl', 'CBRF')       # same + swing agg cover
    assert type(b.engine.reinstatement_terms).__name__ == 'ReinstatementTerms'
    assert type(b._source).__name__ == 'BivariateDistribution'
    sa, sb = a.stats_df, b.stats_df
    # every row of the feature-less walk reappears identically
    for idx in sa.index:
        if idx[0] == 'All' or idx[1:] == ('Margin', 'Net'):
            continue                      # grand rows shift with the cover
        assert sb.loc[idx, 'EX'] == pytest.approx(sa.loc[idx, 'EX'],
                                                  abs=1e-9), idx
    # the annual cap holds: recovery = E[A(R)] = E[min(R, 4750)], NOT E[R].
    # Both are exact moments off the SAME joint the ledger rides.
    bv = b._source
    A = b.engine.reinstatement_terms.recovery
    rec = sb.loc[('Cat Program', 'Obligation', 'Cat Program recovery'), 'EX']
    e_capped = bv.transformed_moments(lambda l, r: A(r))['mean']
    e_unlimited = bv.transformed_moments(lambda l, r: r)['mean']
    assert rec == pytest.approx(e_capped, abs=1e-9)
    assert e_capped < e_unlimited
    # the swing tier: collared stochastic premium, exact off the joint
    prem = sb.loc[('Swing Program', 'Consideration', 'Swing Program premium')]
    assert 200.0 <= -prem['EX'] <= 500.0 and prem['SD'] > 0
    mm = bv.transformed_moments(
        lambda l, r: _phi(_g(np.maximum(l - A(r), 0.0))))
    assert -prem['EX'] == pytest.approx(mm['mean'], abs=1e-9)
    # pnl face = the consolidated net view over the SAME joint
    # ([Decision-PnL-Is-Consolidated]); agrees with the walk exactly
    pb = _build(uw, 'pnl', 'CBRF')
    assert list(pb.stats_df.index.names) == ['View', 'Line']
    assert pb.mean == pytest.approx(
        sb.loc[('All', 'Margin', 'Total'), 'EX'], abs=1e-12)
    # the swing premium rides the net-premium leg (stochastic)
    assert pb.stats_df.xs('net premium', level='Line').iloc[0]['SD'] > 0


def test_reinst_feat_consolidated_foots(uw):
    """The consolidated pnl with the feature folded in foots: the leg means
    sum to the grand net result (the additive uw identity the old analysis
    audit checked, now read off the ledger)."""
    b = _build(uw, 'pnl', 'CBRF')
    s = b.stats_df
    legs = [i for i in s.index if i[1] != 'Total']   # exclude subtotal rows
    assert s.loc[('Margin', 'Total'), 'EX'] == pytest.approx(
        s.loc[legs, 'EX'].sum(), abs=1e-6)
    assert b.mean == pytest.approx(s.loc[('Margin', 'Total'), 'EX'], abs=1e-9)


def test_engine_reference_on_every_face(uw):
    """[Engine-Reference-On-PnL]: the wrapped Aggregate stays reachable via
    ``pnl.engine`` on every DecL-assembled face; the ``source`` is the
    simplest sufficient object (GD / joint), never the engine itself for
    the plain face."""
    for face, ref in (('pnl', 'CBGF'), ('xpnl', 'CBGF'), ('pnl', 'CBRF')):
        p = _build(uw, face, ref)
        assert type(p.engine).__name__ == 'Aggregate', (face, ref)
    u = Underwriter(databases=None, update=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plain = u('pnl EP 1000 premium less agg EP_e 100 claims '
                  'sev lognorm 100 cv 2 poisson')
    assert type(plain.engine).__name__ == 'Aggregate'
    assert type(plain._source).__name__ == 'GridDistribution'


def test_coarse_joint_grid_warning():
    """[Reinst-Joint-Grid-Adequacy]: a narrow aggregate layer after a
    reinstated occ program fires CoarseJointGridWarning; the roomy program
    stays silent."""
    from aggregate.constants import CoarseJointGridWarning
    u = Underwriter(databases=None, update=True)
    narrow = ('pnl NG inherit premium less agg NG_e 12500 premium at 80% lr '
              '5000 xs 0 sev 200 * pareto 1.6 - 200 '
              'occurrence net of 4750 xs 250 rate 55% no reinstatements '
              'poisson aggregate net of 100 xs 3100 deposit 20')
    with pytest.warns(CoarseJointGridWarning, match='aggregate-cover layer'):
        with warnings.catch_warnings():
            warnings.simplefilter('always')
            u(narrow)
    roomy = narrow.replace('pnl NG', 'pnl RG').replace(
        'agg NG_e', 'agg RG_e').replace('100 xs 3100', '2000 xs 3100')
    with warnings.catch_warnings():
        warnings.simplefilter('error', CoarseJointGridWarning)
        u(roomy)


# ----------------------------------------------------------------------
# routing: every matrix cell lands on the right face
# ----------------------------------------------------------------------
def test_matrix_routing(uw):
    """One assert per cell: each matrix cell lands on the right face shape."""
    u = Underwriter(databases=None, update=True)
    base = '5000 prem at 70% lr sev lognorm 100 cv 2 '
    occ = {'none': '', 'gc': 'occurrence net of 400 xs 100 rate 10% ',
           'reinst': ('occurrence net of 400 xs 100 rate 10% '
                      'reinstatements [1] ')}
    agg = {'none': '', 'gc': 'aggregate net of 500 xs 2000 deposit 100',
           'feat': ('aggregate net of 500 xs 2000 swing basic 100 lcm 1 '
                    'min 100 max 400')}
    expect = {
        # (occ, agg, face) -> stats_df index names (no analysis object exists
        # post-[Decommission-Analysis-Classes])
        ('none', 'none', 'pnl'): ['View', 'Line'],
        ('none', 'none', 'xpnl'): ['Step', 'View', 'Line'],
        ('gc', 'feat', 'pnl'): ['View', 'Line'],
        ('gc', 'feat', 'xpnl'): ['Step', 'View', 'Line'],
        ('reinst', 'feat', 'pnl'): ['View', 'Line'],
        ('reinst', 'feat', 'xpnl'): ['Step', 'View', 'Line'],
    }
    for (o, ag, face), names in expect.items():
        prog = (f'{face} M{o}{ag}{face} 5000 premium less agg E{o}{ag}{face} '
                + base + occ[o] + 'poisson ' + agg[ag])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p = u(prog)
        assert isinstance(p, PnL)
        assert list(p.stats_df.index.names) == names, (o, ag, face)
        assert not hasattr(p, 'analysis'), (o, ag, face)
