"""The domain-agnostic P&L kernel: :class:`Leg` / :class:`Group` / :class:`PnL`.

A P&L is a probability space plus named accounting functionals
(``dev/plan-yapnl.md``): a source, an ordered list of signed groups, and
everything else derived. These tests exercise the cross-domain core -- a raw
``(values, probs)`` slot, a :class:`GridDistribution` slot, and a coupled
bivariate joint -- plus the ledger algebra, role orientation, the
[One-2D-Source] rule, per-leg ``bs`` rebucketing with its ``validation_df``
audit, ``economic_ratios_df`` / ``legs_df``, ``+`` composition, and
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
    np.testing.assert_allclose(a.economic_df.to_numpy(), b.economic_df.to_numpy())
    np.testing.assert_allclose(a.summary_df.to_numpy(), b.summary_df.to_numpy())


def test_exact_moments_no_rebucketing():
    """A leg is an exact group-by: moments equal the closed-form values.

    Ledger rows are **signed**: the sold obligation books at ``-E[X]`` and the
    EX column adds down the sheet to the result.
    """
    s = _simple().economic_df
    # E[loss] = 10, booked -10 (sold obligation); Var = 100 either way
    assert s.loc[('Obligation', 'obligation'), 'EX'] == pytest.approx(-10.0, abs=TOL)
    assert s.loc[('Obligation', 'obligation'), 'SD'] == pytest.approx(10.0, abs=1e-9)
    # result = 15 - loss: mean 5, SD unchanged (a constant shift)
    assert s.loc[('Margin', 'Total'), 'EX'] == pytest.approx(5.0, abs=TOL)
    assert s.loc[('Margin', 'Total'), 'SD'] == pytest.approx(10.0, abs=1e-9)
    # the EX column foots: consideration + obligation = result
    assert s.loc[('Margin', 'Total'), 'EX'] == pytest.approx(
        s.loc[('Consideration', 'consideration'), 'EX']
        + s.loc[('Obligation', 'obligation'), 'EX'], abs=TOL)


# ----------------------------------------------------------------------
# Role orientation ([Signed-Exhibits-Role-Orientation])
# ----------------------------------------------------------------------
def test_buy_role_flips_every_row_and_percentiles():
    """The same position built sell vs buy flips every row sign; the scenario
    ladder reflects exactly: the bought cells are the negated,
    ``p <-> 1-p``-reflected cells of the sold ones.

    Probabilities are chosen so no ladder point sits on a CDF jump, making
    the quantile reflection ``q_buy(1-p) = -q_sell(p)`` exact -- and with it
    the conditioning slices coincide, so the kappa cells reflect exactly.
    """
    probs = np.array([0.37, 0.28, 0.22, 0.13])       # cum: .37 .65 .87 1.0
    sell = PnL(name='s', source=(_VALS, probs), role='sell',
               consideration=4.0, obligation=lambda x: np.minimum(x, 20) * 0.5)
    buy = PnL(name='b', source=(_VALS, probs), role='buy',
              consideration=4.0, obligation=lambda x: np.minimum(x, 20) * 0.5)
    assert sell.est_m == pytest.approx(-buy.est_m, abs=TOL)
    ss, bs = sell.economic_df, buy.economic_df
    pcols = [c for c in ss.columns if c.startswith('κ')]
    assert len(pcols) == 9                   # the scenario ladder is present
    for row in ss.index:
        assert ss.loc[row, 'EX'] == pytest.approx(-bs.loc[row, 'EX'], abs=TOL)
        assert ss.loc[row, 'SD'] == pytest.approx(bs.loc[row, 'SD'], abs=TOL)
        # exact scenario reflection: cell_buy(κq) = -cell_sell(κ(1-q))
        for c, rc in zip(pcols, reversed(pcols)):
            assert bs.loc[row, c] == pytest.approx(-ss.loc[row, rc], abs=TOL)


# ----------------------------------------------------------------------
# Ledger template rows
# ----------------------------------------------------------------------
def test_multileg_totals_rows():
    """A multi-leg obligation surfaces an ('Obligation', 'Total') stats row;
    a single does not. The card is fixed-shape either way."""
    two = PnL(name='t', source=(_VALS, _PROBS), role='sell',
              consideration={'premium': 15.0},
              obligation={'loss': lambda x: x, 'expense': 3.0})
    # signed: -(10 + 3)
    assert two.economic_df.loc[('Obligation', 'Total'), 'EX'] == \
        pytest.approx(-13.0)
    one = _simple()
    assert ('Obligation', 'Total') not in one.economic_df.index
    assert ('Consideration', 'Total') not in one.economic_df.index
    # the card never varies with the leg count
    assert list(two.summary_df.index) == list(one.summary_df.index) == \
        ['Consideration', 'Obligation', 'Margin']
    assert two.summary_df.loc['Obligation', 'EX'] == pytest.approx(-13.0)


def test_leg_shorthand_and_leg_objects_agree():
    """``{label: func}`` is exactly ``[Leg(label, func), ...]``."""
    a = PnL(name='a', source=(_VALS, _PROBS), role='sell',
            consideration={'premium': 15.0}, obligation={'loss': lambda x: x})
    b = PnL(name='b', source=(_VALS, _PROBS), role='sell',
            consideration=[Leg('premium', 15.0)],
            obligation=[Leg('loss', lambda x: x)])
    assert list(a.economic_df.index) == list(b.economic_df.index)
    np.testing.assert_allclose(a.economic_df.to_numpy(), b.economic_df.to_numpy())


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
    s = t.economic_df
    assert s.loc[('All', 'Margin', 'Net'), 'EX'] == pytest.approx(
        s.loc[('base', 'Margin', 'Gross'), 'EX']
        + s.loc[('cover', 'Margin', 'Total'), 'EX'], abs=1e-9)
    assert s.loc[('cover', 'Margin', 'Net'), 'EX'] == pytest.approx(
        s.loc[('All', 'Margin', 'Net'), 'EX'], abs=TOL)
    # SDs do not add (the cover trims the tail, so net SD < base SD)
    assert s.loc[('All', 'Margin', 'Net'), 'SD'] < \
        s.loc[('base', 'Margin', 'Gross'), 'SD']
    # total impact = grand result - first group's result
    assert s.loc[('All', 'Margin', 'Impact'), 'EX'] == pytest.approx(
        s.loc[('All', 'Margin', 'Net'), 'EX']
        - s.loc[('base', 'Margin', 'Gross'), 'EX'], abs=TOL)
    # grand totals foot to the result
    assert s.loc[('All', 'Margin', 'Net'), 'EX'] == pytest.approx(
        s.loc[('All', 'Consideration', 'Net'), 'EX']
        + s.loc[('All', 'Obligation', 'Net'), 'EX'], abs=1e-9)


def test_plus_composition_concatenates_ledgers():
    """``pnl_a + pnl_b`` (same source) concatenates the group ledgers."""
    a = PnL(name='a', source=(_VALS, _PROBS), groups=[
        Group('base', 'sell', {'premium': 15.0}, {'loss': lambda x: x})])
    b = PnL(name='b', source=(_VALS, _PROBS), groups=[
        Group('cover', 'buy', {'ceded premium': 2.0},
              {'recovery': lambda x: np.maximum(x - 20, 0)})])
    combined = a + b
    ref = _two_group()
    assert list(combined.economic_df.index) == list(ref.economic_df.index)
    np.testing.assert_allclose(combined.economic_df.to_numpy(),
                               ref.economic_df.to_numpy())


def test_plus_requires_same_source():
    a = _simple()
    other = PnL(name='o', source=(_VALS * 2, _PROBS), role='sell',
                consideration=1.0, obligation=lambda x: x)
    with pytest.raises(ValueError, match='same source'):
        a + other


# ----------------------------------------------------------------------
# economic_df MultiIndex: (Side, Label) single-group, (Step, Side, Label) tower
# ----------------------------------------------------------------------
def test_stats_df_side_label_multiindex():
    """economic_df rows carry a (Side, Label) MultiIndex: legs under their side,
    total rows -> (Side, 'Total'), the result -> ('Margin', 'Total'). The
    flat ledger labels stay the canonical keys on the other exhibits."""
    p = PnL(name='p', source=(_VALS, _PROBS), role='sell',
            consideration={'premium': 15.0},
            obligation={'loss': lambda x: x, 'expense': 3.0})
    s = p.economic_df
    assert list(s.index.names) == ['Side', 'Label']
    assert list(s.index) == [
        ('Consideration', 'premium'),
        ('Obligation', 'loss'), ('Obligation', 'expense'),
        ('Obligation', 'Total'),
        ('Margin', 'Total')]
    # canonical flat keys untouched on the non-presentation exhibits
    assert list(p.density_df) == ['premium', 'loss', 'expense', 'result']


def test_stats_df_multiindex_multigroup():
    """Multi-group sheet: three-level (Step, Side, Label) -- the step level is
    the group label, in ledger order.

    Because this ledger buys something, the three margins that would all read
    ``Total`` are told apart ([Ledger-Side-Label-Levels]): the ``sell`` group's
    own result is ``Direct``, the cession's stays ``Total``, and the grand
    block reads ``Net`` on all three of its rows. The impact closes at
    ('All', 'Margin', 'Impact')."""
    t = PnL(name='m', source=(_VALS, _PROBS), groups=[
        Group('base', 'sell', {'premium': 15.0, 'fee': 1.0},
              {'loss': lambda x: x}),
        Group('cover', 'buy', {'ceded premium': 2.0},
              {'recovery': lambda x: np.maximum(x - 20, 0)}),
    ])
    s = t.economic_df
    assert list(s.index.names) == ['Step', 'Side', 'Label']
    assert list(s.index) == [
        ('base', 'Consideration', 'premium'),
        ('base', 'Consideration', 'fee'),
        ('base', 'Consideration', 'Total'),
        ('base', 'Obligation', 'loss'),
        ('base', 'Margin', 'Gross'),
        ('cover', 'Consideration', 'ceded premium'),
        ('cover', 'Obligation', 'recovery'),
        ('cover', 'Margin', 'Total'), ('cover', 'Margin', 'Net'),
        ('All', 'Consideration', 'Net'), ('All', 'Obligation', 'Net'),
        ('All', 'Margin', 'Net'), ('All', 'Margin', 'Impact')]
    # single-group frames stay two-level
    assert list(_simple().economic_df.index.names) == ['Side', 'Label']
    assert ('All', 'Margin', 'Total') not in s.index


def test_direct_block_and_net_need_something_bought():
    """The named direct block and ``Net`` appear only if a buy group is present.

    Two sold books merged into one ledger have nothing to be direct *of*, so
    every group result and the grand block keep the neutral ``Total``. Same for
    a plain single-group ``pnl``, whose one ``sell`` group holds legs that are
    already net: naming that margin for the subject business would be a lie,
    since it is the net.

    The per-step ``net through <g>`` row is a different thing and keeps its
    ``Net`` label either way: it is a running total, not half of the
    direct-versus-net contrast.
    """
    both_sold = PnL(name='s', source=(_VALS, _PROBS), groups=[
        Group('book A', 'sell', {'A premium': 15.0}, {'A loss': lambda x: x},
              margin_label='Book A'),
        Group('book B', 'sell', {'B premium': 4.0},
              {'B loss': lambda x: 0.2 * x}),
    ])
    s = both_sold.economic_df
    # margin_label is declared and still ignored: no purchase, no direct block
    assert set(s.xs('Margin', level='Side').index.get_level_values('Label')) \
        == {'Total', 'Net', 'Impact'}
    for side in ('Consideration', 'Obligation', 'Margin'):
        assert ('All', side, 'Total') in s.index
    assert ('book B', 'Margin', 'Net') in s.index      # still a running net
    # ... and the single-group case
    assert ('Margin', 'Total') in _simple().economic_df.index


def test_margin_label_names_the_direct_block():
    """With a purchase in the ledger, the sold group's margin takes its name.

    The subject business names its own margin row the way it already names its
    loss leg ([First-Step-Label]); ``None`` falls back to ``'Gross'``.
    """
    named = PnL(name='n', source=(_VALS, _PROBS), groups=[
        Group('base', 'sell', {'premium': 15.0}, {'loss': lambda x: x},
              margin_label='Motor'),
        Group('cover', 'buy', {'ceded premium': 2.0},
              {'recovery': lambda x: np.maximum(x - 20, 0)}),
    ])
    assert ('base', 'Margin', 'Motor') in named.economic_df.index
    assert ('base', 'Margin', 'Gross') in _two_group().economic_df.index


# ----------------------------------------------------------------------
# economic_ratios_df / legs_df ([PnL-Ratio-Frame]) -- the retired scale / Scaled /
# scaled_stats_df surface is replaced by these two raw-materials frames
# ----------------------------------------------------------------------
def _classified_two_group():
    """A sell book with one cession, every leg carrying its ``kind``."""
    return PnL(name='m', source=(_VALS, _PROBS), groups=[
        Group('base', 'sell', [Leg('premium', 40.0, kind='premium')],
              [Leg('loss', lambda x: x, kind='loss'),
               Leg('LAE', 1.0, kind='expense')]),
        Group('cover', 'buy', [Leg('ceded premium', 6.0, kind='premium')],
              [Leg('recovery', lambda x: np.maximum(x - 20, 0),
                   kind='recovery')]),
    ])


def test_the_retired_scale_surface_is_gone():
    p = _simple()
    for name in ('scale', 'scaled_stats_df', '_resolve_scale',
                 'E_consideration'):
        assert not hasattr(p, name), name
    assert 'Scaled' not in p.summary_df.columns
    # E_consideration went with it: one committed ledger-wide premium number
    # was the same idea as Scaled. economic_ratios_df reads P per block from the leg
    # kinds, and evaluate() targets the canonical shift, not a premium.
    assert p.economic_ratios_df['P'].iloc[-1] == pytest.approx(15.0)


def test_ratio_df_amounts_foot_to_the_margin():
    """``M == P - L - E - C`` holds by construction, on every block."""
    p = PnL(name='p', source=(_VALS, _PROBS), role='sell',
            consideration=[Leg('premium', 20.0, kind='premium')],
            obligation=[Leg('loss', lambda x: x, kind='loss'),
                        Leg('expense', 2.0, kind='expense')])
    r = p.economic_ratios_df
    assert list(r.index) == ['p']
    row = r.loc['p']
    assert row['P'] == pytest.approx(20.0)
    assert row['E'] == pytest.approx(2.0)
    assert row['M'] == pytest.approx(
        row['P'] - row['L'] - row['E'] - row['C'], abs=TOL)
    # 1 - CR == M / P, the other reading of the same identity
    assert 1.0 - row['CR'] == pytest.approx(row['M'] / row['P'], abs=TOL)


def test_ratio_df_unclassified_obligation_folds_into_loss():
    """No ``kind`` declared: the residual rule puts obligation in ``L``.

    A ledger that declares no expense legs reports ``E == 0`` rather than
    guessing which of its legs was meant to be an expense.
    """
    # the dict shorthand declares no kinds, so both legs are unclassified
    p = PnL(name='u', source=(_VALS, _PROBS), role='sell',
            consideration={'premium': 20.0},
            obligation={'loss': lambda x: x, 'expense': 2.0})
    row = p.economic_ratios_df.loc['u']
    assert row['E'] == 0.0 and row['C'] == 0.0
    obl = -p.economic_df.loc[('Obligation', 'Total'), 'EX']
    assert row['L'] == pytest.approx(obl, abs=TOL)   # loss AND expense
    assert row['M'] == pytest.approx(row['P'] - row['L'], abs=TOL)


def test_ratio_df_ratios_are_re_derived_not_averaged():
    """The pricing_df rule: amounts add, ratios come off the summed amounts."""
    r = _classified_two_group().economic_ratios_df
    assert list(r.index) == ['base', 'cover', 'All']
    for col in ('P', 'L', 'E', 'C', 'M'):
        assert r.loc['All', col] == pytest.approx(
            r.loc['base', col] + r.loc['cover', col], abs=TOL)
    assert r.loc['All', 'LR'] == pytest.approx(
        r.loc['All', 'L'] / r.loc['All', 'P'], abs=TOL)
    # ...and is NOT the average of the block loss ratios
    assert r.loc['All', 'LR'] != pytest.approx(
        (r.loc['base', 'LR'] + r.loc['cover', 'LR']) / 2, abs=1e-6)


def test_ratio_df_cession_ratios_read_positive():
    """Signing amounts in the gross direction keeps a ceded LR conventional.

    A cession's premium and recovery are both negative contributions, so their
    ratio comes out positive: "this cover paid back a multiple of its premium".
    """
    r = _classified_two_group().economic_ratios_df
    assert r.loc['cover', 'P'] < 0 and r.loc['cover', 'L'] < 0
    assert r.loc['cover', 'LR'] > 0
    assert r.loc['base', 'P_share'] == pytest.approx(1.0)
    assert r.loc['base', 'M_share'] == pytest.approx(1.0)


def test_ratio_df_mean_of_ratio_equals_ratio_of_means_when_premium_is_fixed():
    """``E_LR`` and ``LR`` coincide exactly for a deterministic premium."""
    p = PnL(name='p', source=(_VALS, _PROBS), role='sell',
            consideration=[Leg('premium', 20.0, kind='premium')],
            obligation=[Leg('loss', lambda x: x, kind='loss')])
    row = p.economic_ratios_df.loc['p']
    assert row['E_LR'] == pytest.approx(row['LR'], abs=TOL)


def _stitched(premium_sd):
    """A one-group gd-backed ledger whose premium sd the caller chooses."""
    from aggregate._grid_distribution import GridDistribution

    def gd(vals, probs):
        return GridDistribution(np.asarray(vals, dtype=float),
                                np.asarray(probs, dtype=float), bs=None,
                                is_loss_value=False)

    groups = [Group('base', 'sell', [Leg('premium', 15.0, kind='premium')],
                    [Leg('loss', lambda x: x, kind='loss')])]
    prem = (gd([10.0, 20.0], [0.5, 0.5]) if premium_sd
            else gd([15.0], [1.0]))
    entries = {'premium': (prem, 15.0, premium_sd),
               'loss': (gd([-10.0, 0.0], [0.5, 0.5]), -5.0, 5.0),
               'result': (gd([5.0, 15.0], [0.5, 0.5]), 10.0, 5.0)}
    return PnL(name='s', source=None, groups=groups, result_name='result',
               stitched_rows=entries)


def test_mean_of_ratio_survives_a_missing_joint_when_premium_is_fixed():
    """A constant denominator factors out, so no joint is needed.

    The gate is whether the **denominator** is random, not whether a joint
    happens to exist: ``E[L / P] == E[L] / P`` exactly for constant ``P``.
    """
    row = _stitched(premium_sd=0.0).economic_ratios_df.loc['base']
    assert row['LR'] == pytest.approx(5.0 / 15.0, abs=TOL)
    assert row['E_LR'] == row['LR']            # exactly, not approximately
    assert row['E_ER'] == row['ER'] and row['E_CR'] == row['CR']


def test_mean_of_ratio_is_nan_when_premium_is_random_and_the_joint_is_gone():
    """Random denominator and no atoms to average over: genuinely unknowable.

    Never a silent fallback to the ratio of the means, which is a different
    number precisely when the denominator is random.
    """
    row = _stitched(premium_sd=5.0).economic_ratios_df.loc['base']
    assert row['LR'] == pytest.approx(5.0 / 15.0, abs=TOL)
    assert np.isnan(row['E_LR'])
    assert np.isnan(row['E_ER']) and np.isnan(row['E_CR'])


def test_legs_df_itemizes_declared_legs_only():
    df = _classified_two_group().legs_df
    assert list(df.columns) == ['Step', 'Side', 'Label', 'kind', 'EX', 'SD']
    assert len(df) == 5                    # declared legs only, no derived rows
    assert list(df['kind']) == ['premium', 'loss', 'expense', 'premium',
                                'recovery']
    assert list(df['Step']) == ['base'] * 3 + ['cover'] * 2
    # EX is the signed booked mean, matching economic_df
    assert df.loc[0, 'EX'] == pytest.approx(40.0)
    assert df.loc[3, 'EX'] == pytest.approx(-6.0)


def test_leg_kind_validates_its_value_set():
    with pytest.raises(ValueError, match='Leg kind must be one of'):
        Leg('x', 1.0, kind='banana')
    assert Leg('x', 1.0).kind is None
    assert "kind='loss'" in repr(Leg('x', 1.0, kind='loss'))


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
    assert bucketed.economic_df.loc[('Obligation', 'loss'), 'EX'] == \
        pytest.approx(-10.0)


def test_derived_rows_stay_exact_alongside_bs_legs():
    """The result row is per-atom exact even when a leg is rebucketed."""
    p = PnL(name='p', source=(_VALS, _PROBS), role='sell',
            consideration=15.0,
            obligation=[Leg('loss', lambda x: x, bs=10.0)])
    assert p.est_m == pytest.approx(5.0, abs=TOL)
    assert p.est_sd == pytest.approx(10.0, abs=1e-9)


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
    assert p.economic_df.loc[('Obligation', 'obligation'), 'EX'] == \
        pytest.approx(-5.0)


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
    assert pc.economic_df.loc[('Obligation', 'shortfall put'), 'SD'] < \
        pi.economic_df.loc[('Obligation', 'shortfall put'), 'SD']


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
    assert plant.economic_df.loc[('Obligation', 'cost'), 'EX'] == \
        pytest.approx(-200.0)
    assert plant.est_m == pytest.approx(400.0 - 200.0)


# ----------------------------------------------------------------------
# [Kappa-Scenario-Percentiles]: economic_df ladder columns are states
# ----------------------------------------------------------------------
# scenario (kappa) ladder headers -- the κ marks conditioning on the sheet
# ([Decision-Ladder-Column-Names]); marginal ladders keep plain P01... headers
_KCOLS = ['κ01', 'κ05', 'κ10', 'κ25', 'κ50', 'κ75', 'κ90', 'κ95', 'κ99']


def _assert_columns_foot(pnl):
    """Every scenario column foots: the declared legs sum to the grand
    result's cell, which is the grand result's own marginal quantile."""
    s = pnl.economic_df
    multi = 'Step' in s.index.names
    legs = [lbl for g in pnl.groups
            for leg in (g.consideration + g.obligation)
            for lbl in [leg.label]]
    result_key = (('All', 'Margin', 'Net') if multi
                  else ('Margin', 'Total'))
    for c, q in zip(_KCOLS, (.01, .05, .10, .25, .50, .75, .90, .95, .99)):
        leg_sum = sum(float(s.xs(lbl, level='Label')[c].iloc[0])
                      for lbl in legs)
        cell = float(s.loc[result_key, c])
        assert leg_sum == pytest.approx(cell, abs=1e-9), c
        # the grand-result cell is its own marginal quantile, automatically
        assert cell == pytest.approx(float(pnl.result.q(q)), abs=TOL), c


def test_scenario_columns_foot_1d_both_roles():
    _assert_columns_foot(_simple('sell'))
    _assert_columns_foot(_simple('buy'))
    _assert_columns_foot(_two_group())


def test_scenario_columns_foot_2d_joint():
    y = np.array([0.0, 10.0, 20.0])
    z = np.array([0.0, 4.0, 8.0])
    joint = _independent_joint(y, np.array([0.5, 0.3, 0.2]), z,
                               np.array([0.4, 0.4, 0.2]))
    p = PnL(name='j', source=joint, role='sell', consideration=18.0,
            obligation=[Leg('own', lambda l: l),
                        Leg('assumed', lambda l, r: r, is2d=True)])
    _assert_columns_foot(p)


def test_scenario_direction_retro_premium_high_in_bad_columns():
    """The motivating retro bug, pinned: a loss-sensitive premium shows
    *high* in the bad (small p) columns -- the column is one state, so an
    impossible premium-low-next-to-loss-high pairing cannot appear."""
    p = PnL(name='r', source=(_VALS, _PROBS), role='sell',
            consideration={'retro premium': lambda x: 10.0 + 0.5 * x},
            obligation={'loss': lambda x: x})
    s = p.economic_df
    prem = s.loc[('Consideration', 'retro premium')]
    # result = 10 - 0.5x is decreasing in x: bad state (P1) = big loss = big premium
    assert prem['κ01'] > prem['κ50'] > prem['κ99']
    # and the loss row is high (very negative) in the same bad columns
    loss = s.loc[('Obligation', 'loss')]
    assert loss['κ01'] < loss['κ99']


def test_scenario_monotone_1d_cells_evaluate_at_state():
    """Monotone result over distinct atoms: each cell is pointwise 'evaluate
    every leg at the state'."""
    p = _simple('sell')                      # result = 15 - x, monotone
    s = p.economic_df
    for c, q in zip(_KCOLS, (.01, .05, .10, .25, .50, .75, .90, .95, .99)):
        x_q = float(p.result.q(q))
        x_atom = 15.0 - x_q                  # invert the state
        assert s.loc[('Obligation', 'obligation'), c] == \
            pytest.approx(-x_atom, abs=TOL), c


def test_scenario_switcheroo_level_set_means():
    """Non-monotone result (a hump): the cell is the exact probability-
    weighted mean over the level set, and the column still foots."""
    p = PnL(name='h', source=(_VALS, _PROBS), role='sell',
            consideration=15.0, obligation=lambda x: np.abs(x - 15.0))
    # obligation values 15,5,5,15 -> result values 0,10,10,0: two level sets
    s = p.economic_df
    # result = 0 pools atoms {0, 30} (prob .4/.1): E[x|slice] = 6, loss cell -15
    # result = 10 pools atoms {10, 20} (prob .3/.2): loss cell -5
    lo = float(p.result.q(0.01))            # 0, the bad-side level set
    assert lo == pytest.approx(0.0, abs=TOL)
    assert s.loc[('Obligation', 'obligation'), 'κ01'] == \
        pytest.approx(-15.0, abs=TOL)
    assert s.loc[('Obligation', 'obligation'), 'κ99'] == \
        pytest.approx(-5.0, abs=TOL)
    _assert_columns_foot(p)


def test_scenario_constant_result_cells_equal_ex():
    """A fully hedged (constant) result conditions on everything: every
    scenario cell equals its row's EX."""
    p = PnL(name='c', source=(_VALS, _PROBS), role='sell',
            consideration={'swap': lambda x: x + 5.0},
            obligation={'loss': lambda x: x})
    s = p.economic_df
    for key in s.index:
        for c in _KCOLS:
            assert s.loc[key, c] == pytest.approx(s.loc[key, 'EX'], abs=TOL)


def test_scenario_moment_columns_stay_marginal():
    """EX / SD / CV / Skew are row properties -- unchanged by the scenario
    ladder; per-row marginal quantiles remain one line away via density_df."""
    p = _simple('sell')
    s = p.economic_df
    assert s.loc[('Obligation', 'obligation'), 'SD'] == \
        pytest.approx(10.0, abs=1e-9)
    gd = p.density_df['obligation']
    assert float(gd.q(0.5)) == pytest.approx(-10.0, abs=TOL)


# ----------------------------------------------------------------------
# [Summary-Fixed-Card]: summary_df is the fixed headline card
# ----------------------------------------------------------------------
_CARD_COLS = ['EX', 'SD', 'CV', 'Skew', 'P01', 'Median', 'P99']


def test_card_single_group_fixed_three_rows():
    """Single P&L: a flat three-row card off the grand references, whatever
    the leg count; percentiles are each row's own marginal quantiles."""
    p = PnL(name='p', source=(_VALS, _PROBS), role='sell',
            consideration={'premium': 15.0},
            obligation={'loss': lambda x: x, 'expense': 3.0})
    df = p.summary_df
    assert list(df.index) == ['Consideration', 'Obligation', 'Margin']
    assert df.index.name == 'Side'
    assert list(df.columns) == _CARD_COLS
    assert df.loc['Consideration', 'EX'] == pytest.approx(15.0)
    assert df.loc['Obligation', 'EX'] == pytest.approx(-13.0)
    assert df.loc['Margin', 'EX'] == pytest.approx(2.0)
    # marginal percentiles: the Margin row is the result's own quantile
    assert df.loc['Margin', 'P01'] == pytest.approx(float(p.result.q(0.01)))
    assert df.loc['Margin', 'Median'] == pytest.approx(float(p.result.q(0.5)))


def test_card_tower_blocks():
    """Tower: one (Step, View) block per step + a closing Total block; Net
    omitted on the first step; rows scale with steps, never legs."""
    t = _two_group()
    df = t.summary_df
    assert list(df.index) == [
        ('base', 'Consideration'), ('base', 'Obligation'), ('base', 'Margin'),
        ('cover', 'Consideration'), ('cover', 'Obligation'),
        ('cover', 'Margin'), ('cover', 'Net'),
        ('All', 'Consideration'), ('All', 'Obligation'),
        ('All', 'Margin'), ('All', 'Impact')]
    assert list(df.index.names) == ['Step', 'Side']
    # per-step Margin = the step delta; Total block foots on EX
    s = t.economic_df
    assert df.loc[('base', 'Margin'), 'EX'] == pytest.approx(
        s.loc[('base', 'Margin', 'Gross'), 'EX'])
    assert df.loc[('All', 'Margin'), 'EX'] == pytest.approx(
        df.loc[('All', 'Consideration'), 'EX']
        + df.loc[('All', 'Obligation'), 'EX'], abs=1e-9)
    assert df.loc[('All', 'Impact'), 'EX'] == pytest.approx(
        df.loc[('All', 'Margin'), 'EX'] - df.loc[('base', 'Margin'), 'EX'],
        abs=1e-9)


def test_ratio_df_reads_as_a_combined_ratio_decomposition():
    """The combined ratio now lives in ``economic_ratios_df``, its own table.

    The card is currency only ([Reporting-Guidelines] rule 2: one unit per
    column), so the decomposition the retired ``Scaled`` column carried is read
    off the ratio frame instead.
    """
    p = PnL(name='p', source=(_VALS, _PROBS), role='sell',
            consideration=[Leg('premium', 20.0, kind='premium')],
            obligation=[Leg('loss', lambda x: x, kind='loss')])
    row = p.economic_ratios_df.loc['p']
    assert row['LR'] == pytest.approx(0.5)         # E[loss] 10 / premium 20
    assert row['CR'] == pytest.approx(0.5)         # no expense declared
    assert row['M'] / row['P'] == pytest.approx(0.5)
    assert 'Scaled' not in p.summary_df.columns


def test_card_marginal_vs_stats_scenario_differ_on_dependent_legs():
    """Pins the two percentile meanings apart: the card posts marginal
    quantiles (range), the stats sheet posts scenario conditional means
    (alignment). On a joint where the obligation total is not comonotone
    with the result the cells differ."""
    l = np.array([0.0, 10.0])
    r = np.array([0.0, 4.0])
    joint = _independent_joint(l, np.array([0.5, 0.5]), r,
                               np.array([0.5, 0.5]))
    p = PnL(name='d', source=joint, role='sell',
            consideration={'fee': lambda x: 2.0 * x},       # rises with l
            obligation=[Leg('own', lambda x: x),
                        Leg('assumed', lambda x, y: y, is2d=True)])
    # result = 2l - l - r = l - r: the obligation total -(l+r) is NOT
    # comonotone with it. Card P99 (best state for the result) vs the
    # obligation's own marginal 99th differ.
    card = p.summary_df.loc['Obligation', 'P99']            # marginal: 0
    scen = p.economic_df.loc[('Obligation', 'Total'), 'κ99']   # state: -10
    assert card == pytest.approx(0.0, abs=TOL)
    assert scen == pytest.approx(-10.0, abs=TOL)
    assert card != scen


def test_card_empty_side_posts_zero_row():
    """A group with no consideration (a pure cost position) still gets its
    fixed card row -- a constant zero."""
    p = PnL(name='z', source=(_VALS, _PROBS), role='sell',
            obligation={'loss': lambda x: x})
    df = p.summary_df
    assert df.loc['Consideration', 'EX'] == 0.0
    assert df.loc['Consideration', 'SD'] == 0.0
    assert df.loc['Consideration', 'P01'] == 0.0


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


# ----------------------------------------------------------------------
# the stitched kernel mode (gd-backed rows, no shared atoms) -- kept as the
# no-joint assembly seam (e.g. a future massive-source xpnl) after the DecL
# walks moved to per-atom construction (a141); covered here directly
# ----------------------------------------------------------------------
def test_stitched_rows_kernel_mode_direct():
    from aggregate._grid_distribution import GridDistribution
    from aggregate._pnl import Group, Leg

    def gd(vals, probs):
        return GridDistribution(np.asarray(vals, dtype=float),
                                np.asarray(probs, dtype=float), bs=None,
                                is_loss_value=False)

    groups = [Group('base', 'sell', [Leg('premium', 15.0)],
                    [Leg('loss', lambda x: x)]),
              Group('cover', 'buy', [Leg('cover premium', 3.0)],
                    [Leg('cover recovery', lambda x: x)])]
    entries = {
        'premium': (gd([15.0], [1.0]), 15.0, 0.0),
        'loss': (gd([-30.0, -10.0, 0.0], [0.25, 0.5, 0.25]), -12.5, 10.9),
        'base result': (gd([-15.0, 5.0, 15.0], [0.25, 0.5, 0.25]), 2.5, 10.9),
        'cover premium': (gd([-3.0], [1.0]), -3.0, 0.0),
        'cover recovery': (gd([0.0, 5.0], [0.5, 0.5]), 2.5, 2.5),
        'cover result': (gd([-3.0, 2.0], [0.5, 0.5]), -0.5, 2.5),
        'net through cover': (gd([-10.0, 4.0, 12.0], [0.25, 0.5, 0.25]),
                              2.0, 8.0),
        'total consideration': (gd([12.0], [1.0]), 12.0, 0.0),
        'total obligation': (gd([-25.0, -5.0, 0.0], [0.25, 0.5, 0.25]),
                             -10.0, 9.0),
        'result': (gd([-10.0, 4.0, 12.0], [0.25, 0.5, 0.25]), 2.0, 8.0),
        'total impact': ('delta', 'result', 'base result'),
    }
    p = PnL(name='stitched', source=None, groups=groups,
            result_name='result', stitched_rows=entries)
    assert p._stitched
    s = p.economic_df
    # marginal ladder (plain P headers) -- no shared atoms, no kappa
    assert 'P01' in s.columns and '\u03ba01' not in s.columns
    # supplied exact means surface on EX; the impact row is a per-stat delta
    assert s.loc[('All', 'Margin', 'Net'), 'EX'] == pytest.approx(2.0)
    assert s.loc[('All', 'Margin', 'Impact'), 'EX'] == pytest.approx(
        2.0 - 2.5)
    # no shared atoms: + composition refuses
    with pytest.raises(ValueError, match='stitched'):
        p + p
