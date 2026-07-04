"""The domain-agnostic P&L kernel: :class:`Leg` / :class:`Group` / :class:`PnL`.

A P&L is a probability space plus named accounting functionals
(``dev/plan-yapnl.md``): a source, an ordered list of signed groups, and
everything else derived. These tests exercise the cross-domain core -- a raw
``(values, probs)`` slot, a :class:`GridDistribution` slot, and a coupled
bivariate joint -- plus the ledger algebra, role orientation, the
[One-2D-Source] rule, per-leg ``bs`` rebucketing with its ``validation_df``
audit, ``scaled_stats_df``, ``+`` composition, and
:func:`stack_marginal_pnls`. The insurance builders are covered by their own
suites.
"""
import numpy as np
import pytest

from aggregate import Leg, Group, PnL
from aggregate._grid_distribution import GridDistribution
from aggregate._pnl import stack_marginal_pnls
from aggregate.bivariate import BivariateDistribution

TOL = 1e-12

# a tiny discrete loss for the 1-D cases
_VALS = np.array([0.0, 10.0, 20.0, 30.0])
_PROBS = np.array([0.4, 0.3, 0.2, 0.1])


def _simple(role='sell', **kwargs):
    return PnL(name='t', source=(_VALS, _PROBS), role=role,
               consideration=15.0, obligation=lambda x: x, **kwargs)


# ----------------------------------------------------------------------
# Source slots
# ----------------------------------------------------------------------
def test_raw_pair_and_gd_slots_agree():
    """The same ``(values, probs)`` via a raw pair and a GD slot coincide."""
    a = _simple()
    gd = GridDistribution(_VALS, _PROBS)
    b = PnL(name='t', source=gd, role='sell', consideration=15.0,
            obligation=lambda x: x)
    np.testing.assert_allclose(a.summary_df.to_numpy(), b.summary_df.to_numpy())


def test_exact_moments_no_rebucketing():
    """A leg is an exact group-by: moments equal the closed-form values.

    Ledger rows are **signed**: the sold obligation books at ``-E[X]`` and the
    EX column adds down the sheet to the result.
    """
    s = _simple().summary_df
    # E[loss] = 10, booked -10 (sold obligation); Var = 100 either way
    assert s.loc['obligation', 'EX'] == pytest.approx(-10.0, abs=TOL)
    assert s.loc['obligation', 'SD'] == pytest.approx(10.0, abs=1e-9)
    # result = 15 - loss: mean 5, SD unchanged (a constant shift)
    assert s.loc['result', 'EX'] == pytest.approx(5.0, abs=TOL)
    assert s.loc['result', 'SD'] == pytest.approx(10.0, abs=1e-9)
    # the EX column foots: consideration + obligation = result
    assert s.loc['result', 'EX'] == pytest.approx(
        s.loc['consideration', 'EX'] + s.loc['obligation', 'EX'], abs=TOL)


# ----------------------------------------------------------------------
# Role orientation ([Signed-Exhibits-Role-Orientation])
# ----------------------------------------------------------------------
def test_buy_role_flips_every_row_and_percentiles():
    """The same position built sell vs buy flips every row sign and its
    percentile ladder consistently: ``Pq(-X) = -P(1-q)(X)``."""
    sell = PnL(name='s', source=(_VALS, _PROBS), role='sell',
               consideration=4.0, obligation=lambda x: np.minimum(x, 20) * 0.5)
    buy = PnL(name='b', source=(_VALS, _PROBS), role='buy',
              consideration=4.0, obligation=lambda x: np.minimum(x, 20) * 0.5)
    assert sell.mean == pytest.approx(-buy.mean, abs=TOL)
    ss, bs = sell.stats_df, buy.stats_df
    for row in ss.index:
        assert ss.loc[row, 'EX'] == pytest.approx(-bs.loc[row, 'EX'], abs=TOL)
        assert ss.loc[row, 'SD'] == pytest.approx(bs.loc[row, 'SD'], abs=TOL)
        # quantile reflection: q_sell(p) = -q_buy(1-p) (up to atom convention)
        assert ss.loc[row, 'P1'] <= -bs.loc[row, 'P99'] + 1e-9
        assert ss.loc[row, 'P99'] >= -bs.loc[row, 'P1'] - 1e-9


# ----------------------------------------------------------------------
# Ledger template rows
# ----------------------------------------------------------------------
def test_multileg_totals_rows():
    """A multi-leg obligation surfaces a 'total obligation' row; a single
    does not."""
    two = PnL(name='t', source=(_VALS, _PROBS), role='sell',
              consideration={'premium': 15.0},
              obligation={'loss': lambda x: x, 'expense': 3.0})
    assert 'total obligation' in two.summary_df.index
    # signed: -(10 + 3)
    assert two.summary_df.loc['total obligation', 'EX'] == pytest.approx(-13.0)
    one = _simple()
    assert 'total obligation' not in one.summary_df.index
    assert 'total consideration' not in one.summary_df.index


def test_leg_shorthand_and_leg_objects_agree():
    """``{label: func}`` is exactly ``[Leg(label, func), ...]``."""
    a = PnL(name='a', source=(_VALS, _PROBS), role='sell',
            consideration={'premium': 15.0}, obligation={'loss': lambda x: x})
    b = PnL(name='b', source=(_VALS, _PROBS), role='sell',
            consideration=[Leg('premium', 15.0)],
            obligation=[Leg('loss', lambda x: x)])
    assert list(a.summary_df.index) == list(b.summary_df.index)
    np.testing.assert_allclose(a.summary_df.to_numpy(), b.summary_df.to_numpy())


def test_duplicate_row_labels_raise():
    with pytest.raises(ValueError, match='duplicate ledger row'):
        PnL(name='d', source=(_VALS, _PROBS), role='sell',
            consideration={'x': 1.0}, obligation={'x': 2.0})


# ----------------------------------------------------------------------
# Ledger algebra (multi-group)
# ----------------------------------------------------------------------
def _two_group():
    return PnL(name='m', source=(_VALS, _PROBS), groups=[
        Group('base', 'sell', {'premium': 15.0}, {'loss': lambda x: x}),
        Group('cover', 'buy', {'ceded premium': 2.0},
              {'recovery': lambda x: np.maximum(x - 20, 0)}),
    ])


def test_group_results_are_step_deltas_and_grand_result_sums():
    """Group result rows = step deltas of the running net; grand result = sum
    of the group results; means add but SDs do not (covariance per atom)."""
    t = _two_group()
    s = t.summary_df
    assert s.loc['result', 'EX'] == pytest.approx(
        s.loc['base result', 'EX'] + s.loc['cover result', 'EX'], abs=1e-9)
    assert s.loc['net through cover', 'EX'] == pytest.approx(
        s.loc['result', 'EX'], abs=TOL)
    # SDs do not add (the cover trims the tail, so net SD < base SD)
    assert s.loc['result', 'SD'] < s.loc['base result', 'SD']
    # total impact = grand result - first group's result
    assert s.loc['total impact', 'EX'] == pytest.approx(
        s.loc['result', 'EX'] - s.loc['base result', 'EX'], abs=TOL)
    # grand totals foot to the result
    assert s.loc['result', 'EX'] == pytest.approx(
        s.loc['total consideration', 'EX'] + s.loc['total obligation', 'EX'],
        abs=1e-9)


def test_plus_composition_concatenates_ledgers():
    """``pnl_a + pnl_b`` (same source) concatenates the group ledgers."""
    a = PnL(name='a', source=(_VALS, _PROBS), groups=[
        Group('base', 'sell', {'premium': 15.0}, {'loss': lambda x: x})])
    b = PnL(name='b', source=(_VALS, _PROBS), groups=[
        Group('cover', 'buy', {'ceded premium': 2.0},
              {'recovery': lambda x: np.maximum(x - 20, 0)})])
    combined = a + b
    ref = _two_group()
    assert list(combined.summary_df.index) == list(ref.summary_df.index)
    np.testing.assert_allclose(combined.summary_df.to_numpy(),
                               ref.summary_df.to_numpy())


def test_plus_requires_same_source():
    a = _simple()
    other = PnL(name='o', source=(_VALS * 2, _PROBS), role='sell',
                consideration=1.0, obligation=lambda x: x)
    with pytest.raises(ValueError, match='same source'):
        a + other


# ----------------------------------------------------------------------
# stats_df (View, Line) MultiIndex
# ----------------------------------------------------------------------
def test_stats_df_view_line_multiindex():
    """stats_df rows carry a (View, Line) MultiIndex: legs under their side,
    total rows -> (View, 'Total'), the result -> ('Margin', 'Total'). The
    flat ledger labels stay the canonical keys on the other exhibits."""
    p = PnL(name='p', source=(_VALS, _PROBS), role='sell',
            consideration={'premium': 15.0},
            obligation={'loss': lambda x: x, 'expense': 3.0})
    s = p.stats_df
    assert list(s.index.names) == ['View', 'Line']
    assert list(s.index) == [
        ('Consideration', 'premium'),
        ('Obligation', 'loss'), ('Obligation', 'expense'),
        ('Obligation', 'Total'),
        ('Margin', 'Total')]
    assert p.scaled_stats_df.index.equals(s.index)
    # canonical flat keys untouched
    assert list(p.summary_df.index) == [
        'premium', 'loss', 'expense', 'total obligation', 'result']


def test_stats_df_multiindex_multigroup():
    """Multi-group sheet: per-group totals qualify by group label, group
    results sit under Margin keyed by the group label, running nets and the
    grand rows close under ('Margin', 'Total') / ('Margin', 'Total impact')."""
    t = PnL(name='m', source=(_VALS, _PROBS), groups=[
        Group('base', 'sell', {'premium': 15.0, 'fee': 1.0},
              {'loss': lambda x: x}),
        Group('cover', 'buy', {'ceded premium': 2.0},
              {'recovery': lambda x: np.maximum(x - 20, 0)}),
    ])
    assert list(t.stats_df.index) == [
        ('Consideration', 'premium'), ('Consideration', 'fee'),
        ('Consideration', 'base total'),
        ('Obligation', 'loss'),
        ('Margin', 'base'),
        ('Consideration', 'ceded premium'),
        ('Obligation', 'recovery'),
        ('Margin', 'cover'), ('Margin', 'Net through cover'),
        ('Consideration', 'Total'), ('Obligation', 'Total'),
        ('Margin', 'Total'), ('Margin', 'Total impact')]


# ----------------------------------------------------------------------
# scale / scaled_stats_df
# ----------------------------------------------------------------------
def test_scaled_stats_df_is_stats_of_x_over_scale():
    """scaled_stats_df = the stats of ``X / scale`` exactly: EX / SD /
    percentiles divide by the committed scale; CV / Skew pass through."""
    p = PnL(name='p', source=(_VALS, _PROBS), role='sell',
            consideration={'premium': 20.0}, obligation={'loss': lambda x: x},
            scale=10.0)
    assert p.scale == (10.0, 'scale')
    s, sc = p.stats_df, p.scaled_stats_df
    for row in s.index:
        for col in s.columns:
            if col in ('CV', 'Skew'):
                np.testing.assert_equal(sc.loc[row, col], s.loc[row, col])
            else:
                assert sc.loc[row, col] == pytest.approx(
                    s.loc[row, col] / 10.0, abs=TOL, nan_ok=True)


def test_default_scale_is_expected_total_consideration():
    p = PnL(name='p', source=(_VALS, _PROBS), role='sell',
            consideration={'premium': 20.0}, obligation={'loss': lambda x: x})
    assert p.scale == (20.0, 'consideration')
    # summary Scaled column divides EX by the committed scale
    assert p.summary_df.loc['premium', 'Scaled'] == pytest.approx(1.0)


def test_scale_by_consideration_leg_name():
    p = PnL(name='p', source=(_VALS, _PROBS), role='sell',
            consideration={'premium': 20.0, 'fee': 5.0},
            obligation={'loss': lambda x: x}, scale='fee')
    assert p.scale == (5.0, 'fee')


# ----------------------------------------------------------------------
# density_df
# ----------------------------------------------------------------------
def test_density_df_is_ordered_dict_of_gds():
    """density_df is {row: GridDistribution} -- legs, group results
    (multi-group), grand result."""
    p = PnL(name='p', source=(_VALS, _PROBS), role='sell',
            consideration={'premium': 15.0},
            obligation={'loss': lambda x: x, 'expense': 3.0},
            result_name='margin')
    dd = p.density_df
    assert list(dd) == ['premium', 'loss', 'expense', 'margin']
    assert all(isinstance(v, GridDistribution) for v in dd.values())
    assert dd['loss'].to_series().sum() == pytest.approx(1.0)
    t = _two_group()
    assert list(t.density_df) == ['premium', 'loss', 'ceded premium',
                                  'recovery', 'base result', 'cover result',
                                  'result']


# ----------------------------------------------------------------------
# per-leg bs: exact by default, rebucket + audit when requested
# ----------------------------------------------------------------------
def test_bs_leg_rebuckets_and_feeds_validation_df():
    exact = PnL(name='e', source=(_VALS, _PROBS), role='sell',
                consideration=15.0, obligation=[Leg('loss', lambda x: x)])
    assert len(exact.validation_df) == 0            # bs=0 legs absent
    bucketed = PnL(name='b', source=(_VALS, _PROBS), role='sell',
                   consideration=15.0,
                   obligation=[Leg('loss', lambda x: x, bs=10.0)])
    v = bucketed.validation_df
    assert list(v.index) == ['loss']
    assert list(v.columns) == ['EX', 'Est', 'abs_err', 'rel_err']
    # the linear scheme preserves the mean
    assert v.loc['loss', 'abs_err'] == pytest.approx(0.0, abs=1e-12)
    # the leg GD sits on the regular bs grid; the exact EX is off the atoms
    assert bucketed.stats_df.loc[('Obligation', 'loss'), 'EX'] == \
        pytest.approx(-10.0)


def test_derived_rows_stay_exact_alongside_bs_legs():
    """The result row is per-atom exact even when a leg is rebucketed."""
    p = PnL(name='p', source=(_VALS, _PROBS), role='sell',
            consideration=15.0,
            obligation=[Leg('loss', lambda x: x, bs=10.0)])
    assert p.mean == pytest.approx(5.0, abs=TOL)
    assert p.sd == pytest.approx(10.0, abs=1e-9)


# ----------------------------------------------------------------------
# [One-2D-Source]
# ----------------------------------------------------------------------
def _independent_joint(y, py, p, pp):
    """A product joint of two 1-D laws -> a BivariateDistribution."""
    dens = np.outer(py, pp)
    return BivariateDistribution(dens, y, p, bs_ceded=1.0, bs_net=1.0)


def test_is2d_over_1d_source_errors():
    with pytest.raises(ValueError, match='One-2D-Source'):
        PnL(name='x', source=(_VALS, _PROBS), role='sell',
            consideration=1.0,
            obligation=[Leg('bad', lambda l, r: l + r, is2d=True)])


def test_1d_legs_over_2d_source_read_axis_0():
    """A 2-D source with 1-D legs is fine -- they read axis 0."""
    y = np.array([0.0, 10.0])
    z = np.array([0.0, 100.0])
    joint = _independent_joint(y, np.array([0.5, 0.5]), z,
                               np.array([0.5, 0.5]))
    p = PnL(name='p', source=joint, role='sell', consideration=6.0,
            obligation=lambda l: l)                  # axis-0 only
    assert p.summary_df.loc['obligation', 'EX'] == pytest.approx(-5.0)


def test_crop_revenue_negative_dependence_lowers_risk():
    """The natural hedge -- a negatively-coupled (Y, P) joint cuts put risk.

    Crop revenue insurance pays ``max(guarantee - Y*P, 0)``. With yield Y and
    price P negatively dependent (a short crop is dear), revenue ``Y*P`` is
    less volatile than under independence, so the shortfall put has a
    *thinner* tail. A marginal-only view (independence) overstates the risk.
    """
    y = np.array([80.0, 100.0, 120.0])          # yield
    p = np.array([8.0, 10.0, 12.0])             # price
    guarantee = 1000.0
    put = Leg('shortfall put',
              lambda yy, pp: np.maximum(guarantee - yy * pp, 0.0), is2d=True)

    # negative dependence: high yield <-> low price on the anti-diagonal
    neg = np.zeros((3, 3))
    neg[0, 2] = neg[1, 1] = neg[2, 0] = 1 / 3   # (80,12),(100,10),(120,8)
    coupled = BivariateDistribution(neg, y, p, bs_ceded=1.0, bs_net=1.0)
    indep = _independent_joint(y, np.full(3, 1 / 3), p, np.full(3, 1 / 3))

    pc = PnL(name='c', source=coupled, role='sell', consideration=60.0,
             obligation=[put])
    pi = PnL(name='i', source=indep, role='sell', consideration=60.0,
             obligation=[put])
    assert pc.summary_df.loc['shortfall put', 'SD'] < \
        pi.summary_df.loc['shortfall put', 'SD']


def test_count_axis_fixed_plus_variable_cost():
    """A bivariate (N, X) with f(N, X) = STARTUP*N + FUEL*X mixes the axes."""
    n = np.array([0.0, 1.0, 2.0])               # event count N
    x = np.array([0.0, 100.0, 200.0])           # amount used X
    dens = np.full((3, 3), 1 / 9)
    biv = BivariateDistribution(dens, n, x, bs_ceded=1.0, bs_net=1.0)
    plant = PnL(
        name='Plant', source=biv, role='sell', consideration=400.0,
        obligation=[Leg('cost', lambda nn, xx: 50.0 * nn + 1.5 * xx,
                        is2d=True)],
        result_name='margin')
    # E[cost] = 50*E[N] + 1.5*E[X] = 50*1 + 1.5*100 = 200, booked -200
    assert plant.summary_df.loc['cost', 'EX'] == pytest.approx(-200.0)
    assert plant.mean == pytest.approx(400.0 - 200.0)


# ----------------------------------------------------------------------
# stack_marginal_pnls (the no-joint assembler)
# ----------------------------------------------------------------------
def test_stack_marginal_pnls_rows_and_impacts():
    """Perspectives stack as rows x stats; impact rows are per-stat deltas."""
    base = PnL(name='base', source=(_VALS, _PROBS), role='sell',
               consideration=15.0, obligation=lambda x: x,
               result_name='base')
    trimmed = PnL(name='trimmed', source=(_VALS, _PROBS), role='sell',
                  consideration=13.0, obligation=lambda x: np.minimum(x, 20),
                  result_name='trimmed')
    df = stack_marginal_pnls([('base', base), ('trimmed', trimmed)],
                             impacts=[('impact', 'trimmed', 'base')])
    assert list(df.index) == ['base', 'trimmed', 'impact']
    assert 'EX' in df.columns and 'P99' in df.columns
    assert df.loc['impact', 'EX'] == pytest.approx(
        df.loc['trimmed', 'EX'] - df.loc['base', 'EX'], abs=1e-9)
