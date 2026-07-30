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
from aggregate.decl_writer import spec_to_decl
from aggregate.underwriter import Underwriter

VALIDATION_NOISE = get_settings().validation.noise

#: Footing tolerance. The ledger rows are partial sums of the same exact
#: marginals, so a column that foots does so to numerical noise, not to a
#: modelling tolerance.
FOOTS = 1e-9

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


def _steps(pnl):
    """The ordered ``Step`` index keys of a walk's ``stats_df``."""
    return list(dict.fromkeys(pnl.stats_df.index.get_level_values(0)))


def _kappa_columns(pnl):
    return [c for c in pnl.stats_df.columns if c.startswith('κ')]


def _side_total(frame, step, view, column):
    """One side's total for a step: its ``Total`` line, else its single leg.

    A side gains an explicit ``Total`` line only when it carries more than one
    leg, so summing the block blindly would double count it.
    """
    block = frame.loc[(step, view), column]
    return block['Total'] if 'Total' in block.index else block.sum()


def _assert_column_foots(pnl, column):
    """``margin == consideration + obligation`` at every step and at ``All``."""
    s = pnl.stats_df
    for step in _steps(pnl):
        margin = s.loc[(step, 'Margin', 'Total'), column]
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
    assert peeled.stats_df.equals(walk.stats_df)


def test_one_layer_per_tier_keeps_the_kappa_ladder():
    peeled = build(f'{ONE_EACH} peel top-down')
    assert _kappa_columns(peeled), 'per-atom peel must keep the kappa ladder'
    assert 'P01' not in peeled.stats_df.columns


# ----------------------------------------------------------------------
# Step order
# ----------------------------------------------------------------------
def test_top_down_introduces_the_highest_layer_first():
    p = build(f'{OCC2} peel top-down')
    assert _steps(p) == ['Gross', 'occ 300 xs 200', 'occ 100 xs 100', 'All']


def test_bottom_up_introduces_the_lowest_layer_first():
    p = build(f'{OCC2} peel bottom-up')
    assert _steps(p) == ['Gross', 'occ 100 xs 100', 'occ 300 xs 200', 'All']


def test_occurrence_tier_precedes_the_aggregate_tier():
    """Peeling preserves the tier walk's gross -> occ -> agg step order."""
    prog = (
        'xpnl PeelMix 1000 premium less agg PeelMix_e 1000 premium at 70% lr '
        'sev lognorm 100 cv 2 '
        'occurrence net of 100 xs 100 deposit 60 and 300 xs 200 deposit 40 '
        'poisson aggregate net of 150 xs 300 deposit 25 peel top-down'
    )
    assert _steps(build(prog)) == [
        'Gross', 'occ 300 xs 200', 'occ 100 xs 100', 'agg 150 xs 300', 'All']


# ----------------------------------------------------------------------
# The per-atom route (aggregate layers only)
# ----------------------------------------------------------------------
def test_aggregate_only_peel_is_per_atom():
    p = build(f'{AGG2} peel top-down')
    assert not p._stitched
    assert _steps(p) == ['Gross', 'agg 400 xs 800', 'agg 200 xs 600', 'All']
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
    s = p.stats_df
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
    assert 'P01' in p.stats_df.columns


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
    s = p.stats_df
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
    assert peeled.stats_df.loc[('All', 'Margin', 'Total'), 'EX'] == \
        pytest.approx(consolidated.est_m, abs=1e-8)


@pytest.mark.parametrize('direction', ['top-down', 'bottom-up'])
def test_peel_order_does_not_change_the_total(direction):
    """The order layers are introduced in cannot move the closing margin."""
    total = build(f'{OCC2} peel {direction}').stats_df.loc[
        ('All', 'Margin', 'Total'), 'EX']
    other = 'bottom-up' if direction == 'top-down' else 'top-down'
    assert total == pytest.approx(
        build(f'{OCC2} peel {other}').stats_df.loc[
            ('All', 'Margin', 'Total'), 'EX'], abs=FOOTS)


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
    got = p.stats_df.loc[
        ('agg 150 xs 300', 'Obligation', 'agg 150 xs 300 recovery'), 'EX']
    assert got == pytest.approx(engine_agg_ceded, abs=1e-8)
    _assert_column_foots(p, 'EX')


def test_stitched_peel_refuses_evaluate_and_composition():
    """No shared atoms: the two atom-only operations must say so."""
    p = build(f'{OCC2} peel top-down')
    with pytest.raises(NotImplementedError):
        p.evaluate(['margin'])
    with pytest.raises(ValueError, match='stitched'):
        p + p


# ----------------------------------------------------------------------
# Per-layer economics
# ----------------------------------------------------------------------
def test_each_layer_books_its_own_ceded_premium():
    p = build(f'{OCC2} peel top-down')
    s = p.stats_df
    assert s.loc[('occ 100 xs 100', 'Consideration',
                  'occ 100 xs 100 premium'), 'EX'] == pytest.approx(-60.0)
    assert s.loc[('occ 300 xs 200', 'Consideration',
                  'occ 300 xs 200 premium'), 'EX'] == pytest.approx(-40.0)
    # and the grand consideration is the gross premium less both cessions
    assert s.loc[('All', 'Consideration', 'Total'), 'EX'] == \
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
    s = p.stats_df
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
        'Gross', 'Working layer', 'occ 300 xs 200', 'All']


def test_partial_share_layer_descriptor():
    """An undeclared partial-share layer names itself in DecL, share included."""
    prog = (
        'xpnl PeelPart 1000 premium less agg PeelPart_e 1000 premium at 70% lr '
        'sev lognorm 100 cv 2 poisson '
        'aggregate net of 50% so 200 xs 600 deposit 20 and '
        '400 xs 800 deposit 15 peel top-down'
    )
    assert _steps(build(prog)) == [
        'Gross', 'agg 400 xs 800', 'agg 50% so 200 xs 600', 'All']


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
    assert _steps(p) == ['Gross', 'occ 100 xs 0', 'occ 200 xs 200', 'All']
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


def test_every_peeled_row_carries_a_distribution():
    """The stitched route must supply a gd for every ledger row it declares."""
    p = build(f'{OCC2} peel top-down')
    dd = p.density_df
    # one entry per declared leg plus the derived rows the exhibits read
    assert {'premium', 'Loss', 'margin'} <= set(dd)
    for step in _steps(p):
        if step.startswith('occ '):
            assert f'{step} premium' in dd and f'{step} recovery' in dd
    for label, gd in dd.items():
        assert np.isfinite(gd.x).all(), label
        assert gd.p.sum() == pytest.approx(1.0, abs=1e-6), label
