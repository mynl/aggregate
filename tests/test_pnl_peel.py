"""Layer-peeled ``xpnl``: one group per reinsurance layer.

Covers ``[Layer-Peeling-Shorthand]`` (``dev/done/plan-layer-peeling.md``) and
the ``[Walk-Step-Default-Labels]`` half that landed with it: ``xpnl ... peel
top-down`` / ``peel bottom-up`` replaces the tier walk's lumped occurrence and
aggregate groups with one group per layer, so each layer reports its own ceded
premium, ceding commission and marginal impact.

The suite pins the two routes and the boundary between them:

- **per-atom** (at most one occurrence layer peeled) keeps the scenario
  (``kappa``) ladder and every column foots, so a one-layer-per-tier program
  must reproduce the tier walk exactly. That is the degenerate anchor.
- **stitched** (two or more occurrence layers) supplies each row as its own
  exact marginal: the ``EX`` column still foots exactly by linearity and agrees
  with the engine's own ``reins_density_df`` figures, but the ladder carries
  plain ``P`` headers and ``evaluate`` / ``+`` are unavailable.

The DecL programs here are mirrored in ``src/aggregate/agg/decl-testers.agg``
section AI (this module is the canonical source).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from aggregate import build
from aggregate.config import get_settings
from aggregate.constants import DegenerateEvaluationWarning
from aggregate.decl_writer import spec_to_decl
from aggregate.underwriter import Underwriter

VALIDATION_NOISE = get_settings().validation.noise

#: Footing tolerance *within* a step. Those rows are affine shifts of one
#: marginal, so the column foots to numerical noise, not to a modelling
#: tolerance.
FOOTS = 1e-9

#: Footing tolerance *across* steps on the stitched route. Each step rides its
#: own independently computed marginal, so a cross-step sum is exact only up to
#: the rebucketing's first-moment accumulation. Relative, because the absolute
#: drift scales with the amounts: measured ~1.7e-10 relative on the two-tier
#: program below.
FOOTS_ACROSS = 1e-9

#: One occurrence tower, two priced layers. The peel target.
OCC2 = (
    'xpnl PeelOcc 1000 premium less agg PeelOcc_e 1000 premium at 70% lr '
    'sev lognorm 100 cv 2 '
    'occurrence net of 100 xs 100 deposit 60 and 300 xs 200 deposit 40 '
    'poisson'
)

#: One aggregate tower, two priced layers, no occurrence program: the tier that
#: peels per-atom.
AGG2 = (
    'xpnl PeelAgg 1000 premium less agg PeelAgg_e 1000 premium at 70% lr '
    'sev lognorm 100 cv 2 poisson '
    'aggregate net of 200 xs 600 deposit 50 and 400 xs 800 deposit 30'
)

#: One layer per tier: peeling must degenerate to the tier walk.
ONE_EACH = (
    'xpnl PeelOne 1000 premium less agg PeelOne_e 1000 premium at 70% lr '
    'sev lognorm 100 cv 2 '
    'occurrence net of 500 xs 500 deposit 100 poisson '
    'aggregate net of 200 xs 400 deposit 40'
)

#: Two layers in each tier, so both tiers earn a subtotal block.
TWO_EACH = (
    'xpnl PeelBoth 1000 premium less agg PeelBoth_e 1000 premium at 70% lr '
    'sev lognorm 100 cv 2 '
    'occurrence net of 100 xs 100 deposit 60 cede 0.2 and '
    '300 xs 200 deposit 40 poisson '
    'aggregate net of 100 xs 300 deposit 25 and 200 xs 400 deposit 15'
)


def _steps(pnl):
    """The ordered ``Step`` index keys of a walk's ``economic_df``."""
    return list(dict.fromkeys(pnl.economic_df.index.get_level_values(0)))


def _kappa_columns(pnl):
    return [c for c in pnl.economic_df.columns if c.startswith('κ')]


def _side_total(frame, step, side, column):
    """One side's total for a step: its ``Total`` line, else its single leg.

    A side gains an explicit ``Total`` line only when it carries more than one
    leg, so summing the block blindly would double count it. The ``All`` block
    reads ``Net`` rather than ``Total`` and carries one row per side, so the
    sum branch covers it ([Ledger-Side-Label-Levels]).
    """
    block = frame.loc[(step, side), column]
    return block['Total'] if 'Total' in block.index else block.sum()


def _step_result(frame, step, column):
    """A step's **own** result cell, whatever its ``Label`` reads.

    The label varies with ledger position ([Ledger-Side-Label-Levels],
    [First-Step-Label]): the direct step is named for the subject business
    (``'Gross'`` unlabelled), a cession reads ``Total``, the grand block
    ``Net``. In every case the step's own result is the first ``Margin`` row of
    the block, in plan order, ahead of the running net or the impact that may
    follow it.
    """
    return frame.loc[(step, 'Margin'), column].iloc[0]


def _assert_column_foots(pnl, column):
    """``margin == consideration + obligation`` at every step and at ``All``."""
    s = pnl.economic_df
    for step in _steps(pnl):
        margin = _step_result(s, step, column)
        parts = (_side_total(s, step, 'Consideration', column)
                 + _side_total(s, step, 'Obligation', column))
        assert margin == pytest.approx(parts, abs=FOOTS), \
            f'step {step!r} column {column!r} does not foot'


# ----------------------------------------------------------------------
# The degenerate anchor: one layer per tier IS the tier walk
# ----------------------------------------------------------------------
@pytest.mark.parametrize('direction', ['top-down', 'bottom-up'])
def test_one_layer_per_tier_reproduces_the_tier_walk(direction):
    """Peeling a program whose tiers hold one layer each changes nothing.

    With one layer per tier there is nothing to split, so the peel takes the
    per-atom route over the same shared source as the tier walk and must land
    on the identical sheet.
    """
    walk = build(ONE_EACH)
    peeled = build(f'{ONE_EACH} peel {direction}')
    assert not peeled._stitched
    assert _steps(peeled) == _steps(walk)
    assert peeled.economic_df.equals(walk.economic_df)


def test_one_layer_per_tier_keeps_the_kappa_ladder():
    peeled = build(f'{ONE_EACH} peel top-down')
    assert _kappa_columns(peeled), 'per-atom peel must keep the kappa ladder'
    assert 'P01' not in peeled.economic_df.columns


# ----------------------------------------------------------------------
# Step order
# ----------------------------------------------------------------------
def test_top_down_introduces_the_highest_layer_first():
    p = build(f'{OCC2} peel top-down')
    assert _steps(p) == ['Gross', 'occ 300 xs 200', 'occ 100 xs 100',
                         'All occurrence', 'All']


def test_bottom_up_introduces_the_lowest_layer_first():
    p = build(f'{OCC2} peel bottom-up')
    assert _steps(p) == ['Gross', 'occ 100 xs 100', 'occ 300 xs 200',
                         'All occurrence', 'All']


def test_occurrence_tier_precedes_the_aggregate_tier():
    """Peeling preserves the tier walk's gross -> occ -> agg step order."""
    prog = (
        'xpnl PeelMix 1000 premium less agg PeelMix_e 1000 premium at 70% lr '
        'sev lognorm 100 cv 2 '
        'occurrence net of 100 xs 100 deposit 60 and 300 xs 200 deposit 40 '
        'poisson aggregate net of 150 xs 300 deposit 25 peel top-down'
    )
    # the occurrence tier peels into two steps so it earns a subtotal; the
    # single-layer aggregate tier is already its own subtotal
    assert _steps(build(prog)) == [
        'Gross', 'occ 300 xs 200', 'occ 100 xs 100', 'All occurrence',
        'agg 150 xs 300', 'All']


# ----------------------------------------------------------------------
# The per-atom route (aggregate layers only)
# ----------------------------------------------------------------------
def test_aggregate_only_peel_is_per_atom():
    p = build(f'{AGG2} peel top-down')
    assert not p._stitched
    assert _steps(p) == ['Gross', 'agg 400 xs 800', 'agg 200 xs 600',
                         'All aggregate', 'All']
    assert _kappa_columns(p)


def test_aggregate_only_peel_foots_in_every_column():
    """The per-atom promise: legs -> totals -> result add down every column."""
    p = build(f'{AGG2} peel top-down')
    for column in ['EX'] + _kappa_columns(p):
        _assert_column_foots(p, column)


def test_aggregate_layers_sum_to_the_lumped_tier_recovery():
    """Aggregate layers are disjoint on one subject, so their ceders sum.

    The peeled per-layer recoveries must add to the engine's own total
    aggregate cession.
    """
    p = build(f'{AGG2} peel top-down')
    rd = p.engine.reins_density_df
    xs = rd['loss'].to_numpy(dtype=float)
    engine_ceded = float((xs * rd['p_agg_ceded'].to_numpy()).sum())
    s = p.economic_df
    peeled = sum(
        s.loc[(step, 'Obligation', f'{step} recovery'), 'EX']
        for step in _steps(p) if step.startswith('agg '))
    assert peeled == pytest.approx(engine_ceded, abs=1e-8)


# ----------------------------------------------------------------------
# The stitched route (two or more occurrence layers)
# ----------------------------------------------------------------------
def test_occurrence_peel_takes_the_marginal_route():
    p = build(f'{OCC2} peel top-down')
    assert p._stitched
    assert not _kappa_columns(p), \
        'no shared source, so no kappa ladder ([Decision-Kappa-Shared-Source-Rule])'
    assert 'P01' in p.economic_df.columns


@pytest.mark.parametrize('direction', ['top-down', 'bottom-up'])
def test_occurrence_peel_ex_column_foots_exactly(direction):
    """The headline guarantee of the marginal route: EX foots by linearity."""
    _assert_column_foots(build(f'{OCC2} peel {direction}'), 'EX')


@pytest.mark.parametrize('direction', ['top-down', 'bottom-up'])
def test_occurrence_layers_sum_to_the_engine_cession(direction):
    """Peeled occurrence recoveries add to the engine's exact ceded aggregate."""
    p = build(f'{OCC2} peel {direction}')
    rd = p.engine.reins_density_df
    xs = rd['loss'].to_numpy(dtype=float)
    engine_ceded = float((xs * rd['p_agg_ceded_occ'].to_numpy()).sum())
    s = p.economic_df
    peeled = sum(
        s.loc[(step, 'Obligation', f'{step} recovery'), 'EX']
        for step in _steps(p) if step.startswith('occ '))
    assert peeled == pytest.approx(engine_ceded, abs=1e-8)


def test_peel_total_agrees_with_the_consolidated_pnl():
    """Peeling changes the presentation, not the answer.

    The closing margin must match the consolidated ``pnl``, which reads the
    engine's own deepest net marginal.
    """
    peeled = build(f'{OCC2} peel top-down')
    consolidated = build(OCC2.replace('xpnl', 'pnl', 1))
    assert peeled.economic_df.loc[('All', 'Margin', 'Net'), 'EX'] == \
        pytest.approx(consolidated.est_m, abs=1e-8)


@pytest.mark.parametrize('direction', ['top-down', 'bottom-up'])
def test_peel_order_does_not_change_the_total(direction):
    """The order layers are introduced in cannot move the closing margin."""
    total = build(f'{OCC2} peel {direction}').economic_df.loc[
        ('All', 'Margin', 'Net'), 'EX']
    other = 'bottom-up' if direction == 'top-down' else 'top-down'
    assert total == pytest.approx(
        build(f'{OCC2} peel {other}').economic_df.loc[
            ('All', 'Margin', 'Net'), 'EX'], abs=FOOTS)


def test_aggregate_tier_rides_the_occurrence_net_subject():
    """A peeled aggregate layer pushes forward the net-of-occurrence aggregate."""
    prog = (
        'xpnl PeelMix 1000 premium less agg PeelMix_e 1000 premium at 70% lr '
        'sev lognorm 100 cv 2 '
        'occurrence net of 100 xs 100 deposit 60 and 300 xs 200 deposit 40 '
        'poisson aggregate net of 150 xs 300 deposit 25 peel top-down'
    )
    p = build(prog)
    rd = p.engine.reins_density_df
    xs = rd['loss'].to_numpy(dtype=float)
    engine_agg_ceded = float((xs * rd['p_agg_ceded'].to_numpy()).sum())
    got = p.economic_df.loc[
        ('agg 150 xs 300', 'Obligation', 'agg 150 xs 300 recovery'), 'EX']
    assert got == pytest.approx(engine_agg_ceded, abs=1e-8)
    _assert_column_foots(p, 'EX')


def test_stitched_peel_evaluates_its_nets_and_flags_underpriced_layers():
    """A stitched peel evaluates: each row's own gd is all ``evaluate`` needs.

    The layers here are deliberate bargains, deposit 60 against an expected
    recovery of 123.76 and deposit 40 against 123.89, so read from the seller's
    side they carry ``E[M] <= 0``: a position priced below its own expected loss
    survives no stress at all. ``total impact`` is not evaluated at all, being a
    difference between positions rather than one. Composition still needs shared
    atoms and refuses.
    """
    p = build(f'{OCC2} peel top-down')
    layers = ['occ 100 xs 100 result', 'occ 300 xs 200 result']
    with pytest.warns(DegenerateEvaluationWarning, match=r'E\[M\]'):
        ev = p.evaluate()
    assert (ev.loc[layers, 'role'] == 'buy').all()
    assert ev.loc[layers, 'status'].str.startswith('E[M]').all()
    assert 'total impact' not in ev.index.get_level_values('Step')
    # the book written, then the rows that net the cover against it
    assert (ev.loc['Gross result', 'role'] == 'sell').all()
    nets = ['net through occ 300 xs 200', 'net through occ 100 xs 100',
            'margin']
    assert (ev.loc[nets, 'role'] == 'net').all()
    assert (ev.loc[nets + ['Gross result'], 'status'] == 'ok').all()
    # buying cover this cheap improves the deal: gini_p rises down the nets
    gini = ev.unstack('distortion')['gini_p']
    for fam in ('ph', 'wang', 'dual', 'tvar'):
        assert (gini.loc['Gross result', fam]
                < gini.loc['net through occ 300 xs 200', fam]
                < gini.loc['net through occ 100 xs 100', fam])
    with pytest.raises(ValueError, match='stitched'):
        p + p


# ----------------------------------------------------------------------
# Per-layer economics
# ----------------------------------------------------------------------
def test_each_layer_books_its_own_ceded_premium():
    p = build(f'{OCC2} peel top-down')
    s = p.economic_df
    assert s.loc[('occ 100 xs 100', 'Consideration',
                  'occ 100 xs 100 premium'), 'EX'] == pytest.approx(-60.0)
    assert s.loc[('occ 300 xs 200', 'Consideration',
                  'occ 300 xs 200 premium'), 'EX'] == pytest.approx(-40.0)
    # and the grand consideration is the gross premium less both cessions
    assert s.loc[('All', 'Consideration', 'Net'), 'EX'] == \
        pytest.approx(1000.0 - 60.0 - 40.0, abs=FOOTS)


def test_per_layer_commission_books_as_its_own_leg():
    """``cede`` on one layer books that layer's commission, not the tier's."""
    prog = (
        'xpnl PeelCede 1000 premium less agg PeelCede_e 1000 premium at 70% lr '
        'sev lognorm 100 cv 2 '
        'occurrence net of 100 xs 100 deposit 60 cede 0.2 and '
        '300 xs 200 deposit 40 poisson peel bottom-up'
    )
    p = build(prog)
    s = p.economic_df
    assert s.loc[('occ 100 xs 100', 'Obligation',
                  'occ 100 xs 100 commission'), 'EX'] == pytest.approx(12.0)
    # the unceded layer has no commission leg at all
    assert ('occ 300 xs 200', 'Obligation', 'occ 300 xs 200 commission') \
        not in s.index
    _assert_column_foots(p, 'EX')


def test_resolve_reins_economics_reports_per_layer_figures():
    """The economics dict keeps the per-layer figures beside the totals."""
    p = build(f'{OCC2} peel top-down')
    econ = p.economics
    assert econ['pc_occ_by_layer'] == [60.0, 40.0]
    assert econ['pc_occ'] == pytest.approx(100.0)
    assert econ['c_occ_by_layer'] == [0.0, 0.0]


# ----------------------------------------------------------------------
# Step labels ([Walk-Step-Default-Labels])
# ----------------------------------------------------------------------
def test_declared_layer_label_names_its_step():
    prog = (
        'xpnl PeelLab 1000 premium less agg PeelLab_e 1000 premium at 70% lr '
        'sev lognorm 100 cv 2 '
        'occurrence net of 100 xs 100 deposit 60 as "Working layer" and '
        '300 xs 200 deposit 40 poisson peel bottom-up'
    )
    assert _steps(build(prog)) == [
        'Gross', 'Working layer', 'occ 300 xs 200', 'All occurrence', 'All']


def test_partial_share_layer_descriptor():
    """An undeclared partial-share layer names itself in DecL, share included."""
    prog = (
        'xpnl PeelPart 1000 premium less agg PeelPart_e 1000 premium at 70% lr '
        'sev lognorm 100 cv 2 poisson '
        'aggregate net of 50% so 200 xs 600 deposit 20 and '
        '400 xs 800 deposit 15 peel top-down'
    )
    assert _steps(build(prog)) == [
        'Gross', 'agg 400 xs 800', 'agg 50% so 200 xs 600',
        'All aggregate', 'All']


def test_single_layer_tier_walk_step_takes_the_descriptor():
    """The tier walk names an undeclared one-layer tier by its layer."""
    walk = build(
        'xpnl TierOne 1000 premium less agg TierOne_e 1000 premium at 70% lr '
        'sev lognorm 100 cv 2 occurrence net of 500 xs 500 deposit 100 poisson')
    assert _steps(walk) == ['Gross', 'occ 500 xs 500', 'All']


def test_multi_layer_tier_walk_step_stays_generic():
    """A lumped multi-layer tier has no single descriptor, so it stays generic."""
    assert _steps(build(OCC2)) == ['Gross', 'ceded occ', 'All']


# ----------------------------------------------------------------------
# Zero-share gap fillers
# ----------------------------------------------------------------------
def test_zero_share_filler_gets_no_step():
    """A gap filler is structural, not a cession: no group, no row."""
    prog = (
        'xpnl PeelGap 1000 premium less agg PeelGap_e 1000 premium at 70% lr '
        'sev lognorm 100 cv 2 '
        'occurrence net of 100 xs 0 deposit 30 and 0 po 100 xs 100 and '
        '200 xs 200 deposit 20 poisson peel bottom-up'
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p = build(prog)
    assert _steps(p) == ['Gross', 'occ 100 xs 0', 'occ 200 xs 200',
                         'All occurrence', 'All']
    _assert_column_foots(p, 'EX')


def test_all_zero_share_has_nothing_to_peel():
    prog = (
        'xpnl PeelNil 1000 premium less agg PeelNil_e 1000 premium at 70% lr '
        'sev lognorm 100 cv 2 occurrence net of 0 po 100 xs 100 poisson '
        'peel top-down'
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(ValueError, match='nothing to peel'):
            build(prog)


# ----------------------------------------------------------------------
# Rejections: peel is a guaranteed-cost xpnl clause
# ----------------------------------------------------------------------
def test_bad_direction_lists_the_allowed_values():
    with pytest.raises(ValueError, match='top-down, bottom-up'):
        build(f'{OCC2} peel sideways')


def test_peel_on_a_consolidated_pnl_points_at_xpnl():
    with pytest.raises(ValueError, match="'peel' is an 'xpnl' clause"):
        build(f"{OCC2.replace('xpnl', 'pnl', 1)} peel top-down")


def test_peel_without_reinsurance_says_so():
    prog = ('xpnl PeelBare 1000 premium less agg PeelBare_e 700 loss '
            'sev lognorm 100 cv 2 poisson peel top-down')
    with pytest.raises(ValueError, match='carries no reinsurance'):
        build(prog)


def test_peel_on_a_reinstated_program_is_refused():
    prog = (
        'xpnl PeelRe 1000 premium less agg PeelRe_e 1000 premium at 70% lr '
        'sev lognorm 100 cv 2 '
        'occurrence net of 500 xs 500 deposit 100 reinstatements [1] '
        'poisson peel top-down'
    )
    with pytest.raises(ValueError, match='single occurrence layer'):
        build(prog)


def test_peel_on_a_variable_rated_layer_is_refused():
    prog = (
        'xpnl PeelVar 10000 premium less agg PeelVar_e 10000 prem at 85% lr '
        'sev lognorm 50 cv 3 poisson '
        'aggregate net of 5000 xs 4000 deposit 1500 '
        'slide 45% at 60% and 25% at 70% and 19% at 80% peel top-down'
    )
    with pytest.raises(ValueError, match='single aggregate layer'):
        build(prog)


def test_peel_over_a_portfolio_engine_is_refused():
    with pytest.raises(ValueError, match='no layer structure'):
        build('xpnl PeelPort 1000 premium less port.TwoLineBook peel top-down')


# ----------------------------------------------------------------------
# Introspection and round-trip
# ----------------------------------------------------------------------
def test_construction_names_the_route_and_the_caveat():
    stitched = build(f'{OCC2} peel top-down')
    assert 'peel (top-down)' in stitched.construction_description
    assert 'marginal' in stitched.construction_description
    assert 'EX foots exactly' in stitched.construction_description
    explanation = stitched.construction_explanation
    assert '[Layer-Peeling-Shorthand]' in explanation
    assert 'random claim count decouples them' in explanation

    per_atom = build(f'{AGG2} peel bottom-up')
    assert 'per-atom' in per_atom.construction_description
    assert 'every column foots' in per_atom.construction_explanation


@pytest.mark.parametrize('program', [
    f'{OCC2} peel top-down',
    f'{AGG2} peel bottom-up',
    f'{ONE_EACH} peel top-down',
])
def test_peel_round_trips_through_the_writer(program):
    uw = Underwriter()
    kind, name, spec = uw.parser.parse(program)
    rendered = spec_to_decl(spec, kind, name).replace('\n', ' ')
    assert 'peel' in rendered
    kind2, name2, spec2 = uw.parser.parse(rendered)
    assert (kind, name) == (kind2, name2)
    assert spec == spec2


def test_absent_peel_adds_no_spec_key():
    """The default renders nothing, so unpeeled programs are untouched."""
    uw = Underwriter()
    _kind, _name, spec = uw.parser.parse(OCC2)
    assert 'peel' not in spec
    assert 'peel' not in spec_to_decl(spec, 'xpnl', 'PeelOcc')


# ----------------------------------------------------------------------
# Tier subtotals ([Tier-Subtotal-Rows])
#
# A tier that peels into two or more steps earns its own three-row block after
# the last step it spans, so a peeled tower still shows the whole occurrence
# and whole aggregate program. A tier that peels into one step already *is* its
# own subtotal, so it gets none.
# ----------------------------------------------------------------------
def _side(frame, step, view, column='EX'):
    return _side_total(frame, step, view, column)


def test_two_layer_tier_earns_a_subtotal_block():
    p = build(f'{OCC2} peel top-down')
    s = p.economic_df
    assert 'All occurrence' in _steps(p)
    for view in ('Consideration', 'Obligation', 'Margin'):
        assert ('All occurrence', view, 'Total') in s.index


def test_single_layer_tier_earns_no_subtotal():
    """One layer in a tier: its own group rows already are the subtotal."""
    assert 'All occurrence' not in _steps(build(f'{ONE_EACH} peel top-down'))
    assert 'All aggregate' not in _steps(build(f'{ONE_EACH} peel top-down'))


def test_both_tiers_get_their_own_subtotal():
    p = build(f'{TWO_EACH} peel top-down')
    assert _steps(p) == [
        'Gross', 'occ 300 xs 200', 'occ 100 xs 100', 'All occurrence',
        'agg 200 xs 400', 'agg 100 xs 300', 'All aggregate', 'All']


@pytest.mark.parametrize('direction', ['top-down', 'bottom-up'])
def test_tier_subtotal_matches_the_lumped_tier_walk(direction):
    """The cross-path anchor: the subtotal IS the tier walk's step.

    The peel reaches the tier total by summing its own per-layer rows; the tier
    walk reaches it as one lumped group off the occurrence joint. Two
    independent code paths, one answer. The walk rides the coarser 2-D joint,
    so it is the looser of the two.
    """
    walk = build(OCC2).economic_df
    peeled = build(f'{OCC2} peel {direction}').economic_df
    for view in ('Consideration', 'Obligation', 'Margin'):
        assert _side(peeled, 'All occurrence', view) == pytest.approx(
            _side(walk, 'ceded occ', view), abs=1e-6)


def test_tier_subtotal_sums_its_own_layers():
    """Consideration and obligation add over exactly the tier's own steps."""
    p = build(f'{TWO_EACH} peel top-down')
    s = p.economic_df
    for tier, prefix in (('All occurrence', 'occ '), ('All aggregate', 'agg ')):
        layers = [st for st in _steps(p) if st.startswith(prefix)]
        assert len(layers) == 2
        for view in ('Consideration', 'Obligation', 'Margin'):
            assert _side(s, tier, view) == pytest.approx(
                sum(_side(s, st, view) for st in layers), abs=FOOTS)


def test_tier_subtotals_chain_into_the_running_net():
    """Gross margin plus the tier subtotals equals the closing margin.

    A cross-step sum on the stitched route, so the tolerance is relative: the
    four rows ride four independently computed marginals and linearity holds to
    the rebucketing's first-moment accuracy.
    """
    p = build(f'{TWO_EACH} peel top-down')
    s = p.economic_df
    total = (s.loc[('Gross', 'Margin', 'Gross'), 'EX']
             + s.loc[('All occurrence', 'Margin', 'Total'), 'EX']
             + s.loc[('All aggregate', 'Margin', 'Total'), 'EX'])
    assert total == pytest.approx(
        s.loc[('All', 'Margin', 'Net'), 'EX'], rel=FOOTS_ACROSS)


def test_tier_subtotal_foots_in_every_column_per_atom():
    """On the per-atom route the subtotal keeps the kappa ladder and foots."""
    p = build(f'{AGG2} peel top-down')
    assert 'All aggregate' in _steps(p)
    for column in ['EX'] + _kappa_columns(p):
        _assert_column_foots(p, column)


def test_tier_subtotal_ex_foots_on_the_stitched_route():
    p = build(f'{TWO_EACH} peel bottom-up')
    assert p._stitched
    _assert_column_foots(p, 'EX')


def test_summary_df_gains_the_tier_block():
    card = build(f'{TWO_EACH} peel top-down').summary_df
    for tier in ('All occurrence', 'All aggregate'):
        for view in ('Consideration', 'Obligation', 'Margin'):
            assert (tier, view) in card.index
        # a tier block carries no Net row: the running net through the tier is
        # already the last layer's Net
        assert (tier, 'Net') not in card.index


def test_density_df_carries_the_tier_results():
    dd = build(f'{TWO_EACH} peel top-down').density_df
    assert 'All occurrence result' in dd
    assert 'All aggregate result' in dd
    assert np.isfinite(dd['All occurrence result'].x).all()


def test_tier_spans_shift_on_composition():
    """``+`` renumbers groups, so the right-hand spans must shift with them."""
    from aggregate._grid_distribution import GridDistribution
    from aggregate._pnl import Group, Leg, PnL

    src = GridDistribution(np.array([0.0, 10.0]), np.array([0.5, 0.5]),
                           bs=None, is_loss_value=False)

    def two_group(tag):
        groups = [Group(f'{tag}a', 'sell', [Leg(f'{tag} p1', 4.0)],
                        [Leg(f'{tag} o1', lambda x: x)]),
                  Group(f'{tag}b', 'sell', [Leg(f'{tag} p2', 6.0)],
                        [Leg(f'{tag} o2', lambda x: x)])]
        return PnL(name=tag, source=src, groups=groups, result_name='margin',
                   tier_spans=((f'All {tag}', 0, 2),))

    left, right = two_group('L'), two_group('R')
    both = left + right
    assert both._tier_spans == (('All L', 0, 2), ('All R', 2, 4))
    s = both.economic_df
    # each span still totals its OWN two groups, not the other pair's
    for tag, steps in (('L', ['La', 'Lb']), ('R', ['Ra', 'Rb'])):
        assert _side(s, f'All {tag}', 'Margin') == pytest.approx(
            sum(s.loc[(st, 'Margin', 'Total'), 'EX'] for st in steps),
            abs=FOOTS)


def test_unknown_row_kind_raises(monkeypatch):
    """The kernel no longer books an unhandled kind as the total impact."""
    from aggregate import _pnl as pnl_mod

    real = pnl_mod._ledger_plan

    def bogus(groups, result_name, tier_spans=()):
        return real(groups, result_name, tier_spans) + [
            ('mystery row', 'no_such_kind', None)]

    monkeypatch.setattr(pnl_mod, '_ledger_plan', bogus)
    with pytest.raises(ValueError, match='unknown ledger row kind'):
        build(f'{AGG2} peel top-down')


# ----------------------------------------------------------------------
# economic_ratios_df / legs_df over a peeled ledger ([PnL-Ratio-Frame])
# ----------------------------------------------------------------------
def test_ratio_df_has_a_row_per_block_including_the_tier_subtotals():
    p = build(f'{TWO_EACH} peel top-down')
    assert list(p.economic_ratios_df.index) == _steps(p)


def test_ratio_df_amounts_add_across_the_peeled_blocks():
    """Layers add into their tier, tiers into ``All``; ratios re-derived."""
    p = build(f'{TWO_EACH} peel top-down')
    r = p.economic_ratios_df
    for tier, prefix in (('All occurrence', 'occ '), ('All aggregate', 'agg ')):
        layers = [s for s in r.index if s.startswith(prefix)]
        for col in ('P', 'L', 'E', 'C', 'M'):
            assert r.loc[tier, col] == pytest.approx(
                sum(r.loc[s, col] for s in layers), abs=FOOTS)
        # the tier LR comes off the tier's own amounts
        assert r.loc[tier, 'LR'] == pytest.approx(
            r.loc[tier, 'L'] / r.loc[tier, 'P'], abs=FOOTS)


def test_ratio_df_margin_identity_holds_on_every_peeled_block():
    p = build(f'{TWO_EACH} peel top-down')
    r = p.economic_ratios_df
    for step in r.index:
        row = r.loc[step]
        assert row['M'] == pytest.approx(
            row['P'] - row['L'] - row['E'] - row['C'], rel=FOOTS_ACROSS,
            abs=FOOTS)


def test_ratio_df_gross_shares_are_one_and_cessions_are_negative():
    r = build(f'{TWO_EACH} peel top-down').economic_ratios_df
    assert r.loc['Gross', 'P_share'] == pytest.approx(1.0)
    assert r.loc['Gross', 'M_share'] == pytest.approx(1.0)
    assert r.loc['All occurrence', 'P_share'] < 0     # premium paid away
    assert r.loc['All occurrence', 'LR'] > 0           # ...but the LR reads +


def test_ratio_df_e_columns_survive_the_stitched_peel():
    """No joint, but every peel premium is a constant, so the ratio is exact.

    A peeled ledger is guaranteed-cost by construction (``peel`` is refused on
    the variable-rating and reinstatement recipes) and every consideration row
    is a resolved ``deposit`` / ``rol`` / ``rate``, so the denominator never
    varies and ``E[L / P] == E[L] / P`` identically.
    """
    p = build(f'{OCC2} peel top-down')
    assert p._stitched
    r = p.economic_ratios_df
    assert r[['E_LR', 'E_ER', 'E_CR']].notna().all().all()
    for plain, mean_of in (('LR', 'E_LR'), ('ER', 'E_ER'), ('CR', 'E_CR')):
        assert (r[plain] == r[mean_of]).all(), plain


def test_ratio_df_ex_columns_are_live_on_the_per_atom_peel():
    p = build(f'{AGG2} peel top-down')
    assert not p._stitched
    r = p.economic_ratios_df
    assert r[['E_LR', 'E_CR']].notna().all().all()
    # every premium here is deterministic, so the two readings coincide
    for step in r.index:
        assert r.loc[step, 'E_LR'] == pytest.approx(
            r.loc[step, 'LR'], abs=1e-9)


def test_legs_df_classifies_every_peeled_leg():
    """Every library-built leg carries a kind; none is left unclassified."""
    p = build(f'{TWO_EACH} peel top-down')
    df = p.legs_df
    assert df['kind'].notna().all()
    assert set(df['kind']) == {'premium', 'loss', 'recovery', 'commission'}
    # itemized, so derived rows are absent: one row per declared leg
    assert len(df) == sum(len(g.consideration) + len(g.obligation)
                          for g in p.groups)


def test_expense_legs_are_classified_as_expense():
    prog = (
        'xpnl PeelExp 1000 premium less agg PeelExp_e 1000 premium at 70% lr '
        'sev lognorm 100 cv 2 '
        'occurrence net of 100 xs 100 deposit 60 and 300 xs 200 deposit 40 '
        'poisson less 5% loss expense as LAE peel top-down'
    )
    p = build(prog)
    df = p.legs_df
    assert df.loc[df['Label'] == 'LAE', 'kind'].iloc[0] == 'expense'
    # ...so the ratio frame can split the gross block's LR from its ER
    gross = p.economic_ratios_df.loc['Gross']
    assert gross['E'] > 0
    assert gross['CR'] == pytest.approx(gross['LR'] + gross['ER'], abs=FOOTS)


def test_every_peeled_row_carries_a_distribution():
    """The stitched route must supply a gd for every ledger row it declares."""
    p = build(f'{OCC2} peel top-down')
    dd = p.density_df
    # one entry per declared leg plus the derived rows the exhibits read
    assert {'Premium', 'Loss', 'margin'} <= set(dd)
    for step in _steps(p):
        if step.startswith('occ '):
            assert f'{step} premium' in dd and f'{step} recovery' in dd
    for label, gd in dd.items():
        assert np.isfinite(gd.x).all(), label
        assert gd.p.sum() == pytest.approx(1.0, abs=1e-6), label
