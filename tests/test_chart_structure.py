"""[Structure-Emitter] a reinsurance program as a tower of layers.

The broker-slide diagram: a gross slab, then one tower of rectangles per
cession stage, each layer's band its attachment to its exhaustion point and
its width its share. Available before ``update()``, because a program's
shape is declared rather than computed.

Three enrichment tiers and one ``annotate`` tuple: geometry on an
un-updated object, risk statistics on a built one, economics on a
:class:`~aggregate.PnL` only, because a plain ``agg`` strips its ceded
premium clauses for want of a premium context to resolve them against.
"""

import json

import numpy as np
import pytest

from aggregate import build
from aggregate.constants import IgnoredDecLClauseWarning
from aggregate.charts import (
    TowerBlock, available_charts, build_chart_doc, canonical_json,
    chart_structure, human_strings, load_chart_doc, primary_chart,
)
from aggregate.charts._emit_structure import (
    ANNOTATE_FIELDS, DEFAULT_ANNOTATE, _money, _pct, _terms,
)

# Geometry only: two stages, a partial placement, named layers.
_DECLARED = ('agg CS.Declared 10 claims 100 xs 0 sev lognorm 20 cv 1.5 '
             'occurrence net of 50% po 30 xs 40 as ClashLayer poisson '
             'aggregate net of 100 xs 200 as AggCover')
# The two ``p_agg_subject`` cases: ``net of`` and ``ceded to`` occurrence
# output under an aggregate cover. Both are needed, because either
# underlying column coincides with the subject for exactly one of them.
_NET_OF = ('agg CS.NetOf 5 claims 100 xs 0 sev lognorm 10 cv .75 '
           'occurrence net of 15 xs 5 poisson aggregate net of 20 xs 0')
_CEDED_TO = ('agg CS.CededTo 5 claims 100 xs 0 sev lognorm 10 cv .75 '
             'occurrence ceded to 15 xs 5 poisson aggregate net of 20 xs 0')
# An aggregate-stage quota share, which is unlimited from zero.
_QUOTA = ('agg CS.Quota 10 claims 100 xs 0 sev lognorm 20 cv 1.5 poisson '
          'aggregate net of 25% po inf xs 0')
# An explicit zero-share gap between two covered bands, and an unlimited top.
_GAPPED = ('agg CS.Gapped 10 claims 100 xs 0 sev lognorm 20 cv 1.5 '
           'occurrence net of 20 xs 20 and 0% po 20 xs 40 and inf xs 60 '
           'poisson')
# Priced on both stages, with partial placements and ceding commission.
_PRICED = ('xpnl CS.Priced 33333.33 premium as GWP less agg CS.Priced_e '
           'as Subject 2 claims sev 20000 * uniform '
           'occurrence net of 50% po 5000 xs 10000 rate 0.3 cede 0.3 '
           'as "Occ1" and 50% po 5000 xs 15000 rate 0.2 cede 0.3 as "Occ2" '
           'fixed aggregate net of 2500 xs 23500 rol 0.25 as "Agg1" '
           'and 2000 xs 26000 rol 0.15 as "Agg2" peel top-down')
_UNPRICED_PNL = ('pnl CS.Unpriced 5000 prem less agg CS.Unpriced_e 100 claims '
                 'sev lognorm 50 cv 1.5 poisson aggregate net of 2000 xs 3000')
_NO_REINS = 'agg CS.Gross 10 claims 100 xs 0 sev lognorm 20 cv 1.5 poisson'


@pytest.fixture(scope='module')
def declared():
    return build(_DECLARED, update=False)


@pytest.fixture(scope='module')
def net_of():
    return build(_NET_OF)


@pytest.fixture(scope='module')
def priced():
    return build(_PRICED)


def blocks_on(doc, panel):
    return [b for b in doc.blocks if b.panel_id == panel]


def roles_on(doc, panel):
    return [b.role for b in blocks_on(doc, panel)]


def lines_of(block):
    """The annotation stack as one string, for a substring assertion."""
    return ' | '.join(block.label_lines)


def layers_on(doc, panel):
    return [b for b in blocks_on(doc, panel) if b.role == 'layer']


# ------------------------------------------------------------- capability

def test_available_exactly_when_something_cedes(declared):
    assert 'structure' in available_charts(declared)
    assert 'structure' not in available_charts(build(_NO_REINS, update=False))


def test_available_on_a_pnl_wrapping_one_aggregate(priced):
    assert 'structure' in available_charts(priced)


def test_never_a_portfolio():
    """Portfolio is out of scope: its units cede on different stages."""
    port = build('port CS.P agg A 5 claims 100 xs 0 sev lognorm 10 cv .75 '
                 'occurrence net of 15 xs 5 poisson', update=False)
    assert 'structure' not in available_charts(port)
    with pytest.raises(NotImplementedError, match='Portfolio'):
        chart_structure(port)


def test_not_any_objects_own_picture(declared, priced):
    """The program is a view of a book, not the book's own portrait."""
    assert primary_chart(declared) != 'structure'
    assert primary_chart(priced) != 'structure'


# ------------------------------------------------------------- the declared tier

def test_geometry_only_before_update(declared):
    """An un-updated object has the shape and no statistics at all."""
    doc = build_chart_doc(declared, 'structure')
    assert doc.meta['tiers'] == ['declared']
    assert not doc.series
    assert layers_on(doc, 'occ')[0].label_lines == ('50.0% po 30 xs 40',)
    assert layers_on(doc, 'agg')[0].label_lines == ('100 xs 200',)


def test_lee_refuses_an_un_updated_object(declared):
    with pytest.raises(ValueError, match='update'):
        build_chart_doc(declared, 'structure', lee=True)


def test_the_resolved_as_name_is_the_headline(declared):
    doc = build_chart_doc(declared, 'structure')
    assert layers_on(doc, 'occ')[0].label == 'ClashLayer'
    assert layers_on(doc, 'agg')[0].label == 'AggCover'


def test_an_unnamed_layer_falls_back_to_its_terms(net_of):
    doc = build_chart_doc(net_of, 'structure')
    assert layers_on(doc, 'occ')[0].label == '15 xs 5'


def test_the_fallback_headline_does_not_repeat_as_an_annotation(net_of):
    """One fact belongs on a block once."""
    doc = build_chart_doc(net_of, 'structure', annotate=('geometry', 'el'))
    block = layers_on(doc, 'occ')[0]
    assert block.label == '15 xs 5'
    assert '15 xs 5' not in block.label_lines
    assert len(block.label_lines) == 1 and block.label_lines[0].startswith('el')


def test_a_named_layer_keeps_its_terms_as_an_annotation(declared):
    """The name and the terms are two facts, so both are drawn."""
    block = layers_on(build_chart_doc(declared, 'structure'), 'occ')[0]
    assert block.label == 'ClashLayer'
    assert block.label_lines == ('50.0% po 30 xs 40',)


# ----------------------------------------------------------------- geometry

def test_share_is_width_and_the_remainder_is_co_participation(declared):
    doc = build_chart_doc(declared, 'structure')
    layer, = layers_on(doc, 'occ')
    assert (layer.x0, layer.x1) == (0.0, 0.5)
    co, = [b for b in blocks_on(doc, 'occ')
           if b.role == 'co_participation']
    assert (co.x0, co.x1) == (0.5, 1.0)
    assert (co.y0, co.y1) == (layer.y0, layer.y1)


def test_a_full_line_has_no_co_participation(net_of):
    doc = build_chart_doc(net_of, 'structure')
    assert 'co_participation' not in roles_on(doc, 'occ')


def test_the_band_below_and_above_the_tower_is_retained(declared):
    doc = build_chart_doc(declared, 'structure')
    assert roles_on(doc, 'occ') == ['retention', 'layer',
                                    'co_participation', 'retention']
    bottom = blocks_on(doc, 'occ')[0]
    assert (bottom.y0, bottom.y1) == (0.0, 40.0)


def test_a_zero_share_layer_draws_as_a_gap():
    doc = build_chart_doc(build(_GAPPED, update=False), 'structure')
    gap, = [b for b in blocks_on(doc, 'occ') if b.role == 'gap']
    assert (gap.y0, gap.y1) == (40.0, 60.0)
    assert (gap.x0, gap.x1) == (0.0, 1.0)


def test_an_unlimited_top_layer_is_open_and_the_window_contains_it():
    doc = build_chart_doc(build(_GAPPED, update=False), 'structure')
    top = layers_on(doc, 'occ')[-1]
    assert top.open_top
    assert top.y0 == 60.0
    loss = {a.id: a for a in doc.axes}['occ_loss']
    assert loss.suggested_range[1] >= top.y1


def test_the_window_always_contains_the_whole_tower(net_of):
    """A cropped tower would draw a program the object does not have."""
    doc = build_chart_doc(net_of, 'structure')
    axes = {a.id: a for a in doc.axes}
    for panel in ('occ', 'agg'):
        lo, hi = axes[f'{panel}_loss'].suggested_range
        for block in blocks_on(doc, panel):
            assert lo <= block.y0 and block.y1 <= hi


def test_quota_share_is_an_aggregate_stage_word():
    """The same clause on the occurrence stage keeps its literal wording."""
    doc = build_chart_doc(build(_QUOTA, update=False), 'structure')
    layer, = layers_on(doc, 'agg')
    assert layer.label == '25.0% quota share'
    assert layer.open_top
    assert _terms(0.25, np.inf, 0.0, 'agg') == '25.0% quota share'
    assert _terms(0.25, np.inf, 0.0, 'occ') == '25.0% po unlimited xs 0'


def test_the_gross_slab_is_the_policy_layer_it_carves_up(declared):
    doc = build_chart_doc(declared, 'structure')
    slab, = blocks_on(doc, 'gross')
    assert slab.role == 'gross'
    assert slab.label == 'CS.Declared'
    assert '100 xs 0' in lines_of(slab)


def test_the_gross_slab_declares_an_open_top_when_the_window_crops_it():
    """Saying nothing would assert a ceiling the book has not got."""
    doc = build_chart_doc(build(_QUOTA, update=False), 'structure')
    slab, = blocks_on(doc, 'gross')
    assert slab.open_top


# ---------------------------------------------------------------- the built tier

def test_statistics_arrive_with_the_grid(net_of):
    doc = build_chart_doc(net_of, 'structure')
    assert doc.meta['tiers'] == ['declared', 'built']
    assert 'el ' in lines_of(layers_on(doc, 'occ')[0])


def test_the_gross_slab_reports_the_matching_moments(net_of):
    """Per-claim moments beside a per-claim axis, and no other kind."""
    doc = build_chart_doc(net_of, 'structure')
    slab, = blocks_on(doc, 'gross')
    stats = net_of.reins_stats_df
    mean = stats.loc[('sev', 'mean'), ('occ', 'Gross')]
    assert f'mean {_money(mean)}' in slab.label_lines


def test_an_aggregate_only_program_reads_its_slab_off_the_aggregate():
    doc = build_chart_doc(build(_QUOTA), 'structure')
    slab, = blocks_on(doc, 'gross')
    stats = build(_QUOTA).reins_stats_df
    mean = stats.loc[('agg', 'mean'), ('occ', 'Gross')]
    assert f'mean {_money(mean)}' in slab.label_lines


@pytest.mark.parametrize('program', [_NET_OF, _CEDED_TO])
def test_the_aggregate_tower_reproduces_the_frames_pr_attach(program):
    """Both occurrence outputs, because the subject differs between them.

    The aggregate cover's subject is the aggregate of the *requested*
    occurrence output. Naming ``p_agg_net_occ`` or ``p_agg_ceded_occ``
    directly would be right for one of these two programs and silently
    wrong for the other, so both are checked.
    """
    agg = build(program)
    doc = build_chart_doc(agg, 'structure',
                          annotate=('geometry', 'pr_attach', 'pr_detach'))
    stats = agg.reins_stats_df
    for index, block in enumerate(layers_on(doc, 'agg'), start=1):
        column = ('agg', f'layer.{index}')
        expected = stats.loc[('meta', 'pr_attach'), column]
        assert f'pr attach {_pct(expected)}' in block.label_lines
        detach = stats.loc[('meta', 'pr_detach'), column]
        if np.isfinite(detach):
            assert f'pr detach {_pct(detach)}' in block.label_lines


@pytest.mark.parametrize('program', [_NET_OF, _CEDED_TO])
def test_the_aggregate_lee_curve_is_the_subject_not_the_gross(program):
    """The curve the aggregate boundaries are read against.

    This is where the basis actually lives: an annotation copied off the
    frame cannot be wrong, a reference distribution chosen by hand can.
    """
    agg = build(program)
    doc = build_chart_doc(agg, 'structure', lee=True)
    curve, = [s for s in doc.series if s.panel_id == 'agg_lee']
    subject = agg.reins_density_df['p_agg_subject'].to_numpy(dtype=float)
    # The curve is a cumulative distribution over the subject's support, so
    # its last probability is the subject's total mass, and its top outcome
    # is the largest the subject reaches.
    assert curve.x[-1] == pytest.approx(float(subject.sum()), rel=1e-9)
    loss = agg.reins_density_df['loss'].to_numpy(dtype=float)
    top = float(loss[subject > 0][-1])
    outcome = curve.y if curve.y else None
    if outcome is None:
        start, step, count = curve.y_lattice
        outcome = (start + step * (count - 1),)
    assert outcome[-1] == pytest.approx(top, rel=1e-9)


def test_lee_panels_share_the_towers_loss_axis(net_of):
    doc = build_chart_doc(net_of, 'structure', lee=True)
    panels = {p.id: p for p in doc.panels}
    assert panels['occ'].y_axis == panels['occ_lee'].y_axis == 'occ_loss'
    assert panels['agg'].y_axis == panels['agg_lee'].y_axis == 'agg_loss'
    # and the two stages do not share one, being different quantities
    assert panels['occ'].y_axis != panels['agg'].y_axis


def test_the_boundaries_are_marks_and_the_joining_rules_are_faint(net_of):
    doc = build_chart_doc(net_of, 'structure', lee=True)
    tower = [m for m in doc.marks if m.panel_id == 'occ']
    joins = [m for m in doc.marks if m.panel_id == 'occ_lee']
    assert [m.at for m in tower] == [5.0, 20.0]
    assert all(m.orient == 'h' for m in tower + joins)
    assert not any(m.faint for m in tower)
    assert all(m.faint for m in joins)
    assert [m.label for m in joins] == ['5', '20']


def test_no_marks_without_the_lee_panels(net_of):
    doc = build_chart_doc(net_of, 'structure')
    assert not any(m.faint for m in doc.marks)


# --------------------------------------------------------------- the priced tier

def test_economics_need_a_pnl():
    """A plain agg strips the premium clauses, so the fields cannot appear.

    The program below *declares* a ``rol`` and a ``cede``. The build warns
    that it is dropping them, since a loss-only object has no premium
    context to resolve them against, so the diagram has no economics to
    draw whatever ``annotate`` asks for.
    """
    with pytest.warns(IgnoredDecLClauseWarning):
        agg = build('agg CS.Stripped 10 claims 100 xs 0 sev lognorm 20 cv 1.5 '
                    'occurrence net of 30 xs 40 rol 0.1 cede 0.2 poisson')
    doc = build_chart_doc(agg, 'structure', annotate=ANNOTATE_FIELDS)
    assert 'priced' not in doc.meta['tiers']
    lines = lines_of(layers_on(doc, 'occ')[0])
    for absent in ('premium', 'lr ', 'rol', 'cede', 'reinstatement'):
        assert absent not in lines
    # the built tier still answers, so the tuple is not simply ignored
    assert 'el ' in lines and 'lol ' in lines


def test_a_pnl_with_no_premium_clause_stays_at_the_built_tier():
    doc = build_chart_doc(build(_UNPRICED_PNL), 'structure')
    assert doc.meta['tiers'] == ['declared', 'built']


def test_premium_is_shown_at_one_hundred_percent_terms(priced):
    """The stored figure is placed; the diagram quotes the layer."""
    doc = build_chart_doc(priced, 'structure')
    placed = priced.economics['pc_occ_by_layer']
    for index, block in enumerate(layers_on(doc, 'occ')):
        share = block.x1 - block.x0
        assert f'premium {_money(placed[index] / share)}' in block.label_lines
    # and the placement really is partial here, so the two differ
    assert placed[0] != pytest.approx(placed[0] / 0.5)


def test_the_ratios_are_share_invariant(priced):
    """``lr``, ``rol`` and ``lol`` read the same at either basis."""
    doc = build_chart_doc(priced, 'structure',
                          annotate=('premium', 'el', 'lr', 'rol'))
    block = layers_on(doc, 'occ')[0]
    lines = dict(line.split(' ', 1) for line in block.label_lines)
    premium = float(lines['premium'].replace(',', ''))
    el = float(lines['el'].replace(',', ''))
    assert lines['lr'] == _pct(el / premium)
    assert lines['rol'] == _pct(premium / 5000.0)


def test_the_ceding_commission_is_a_rate(priced):
    doc = build_chart_doc(priced, 'structure', annotate=('cede',))
    assert layers_on(doc, 'occ')[0].label_lines == ('cede 30.0%',)
    # the aggregate layers carry no cede clause, so nothing is claimed
    assert layers_on(doc, 'agg')[0].label_lines == ()


def test_the_reinstatement_schedule_annotates_the_layer_it_decorates():
    pnl = build('pnl CS.Reinst 10000 premium less agg CS.Reinst_e 10000 prem '
                'at 85% lr sev lognorm 50 cv 3 occurrence net of 95% po 100 '
                'xs 100 rol 18% reinstatements 1 free and 1 at 50% '
                'and 2 at 100% poisson')
    doc = build_chart_doc(pnl, 'structure', annotate=('reinstatements',))
    assert layers_on(doc, 'occ')[0].label_lines == (
        '4 reinstatements at 0%, 50.0%, 100.0%, 100.0%',)


# ------------------------------------------------------------------- annotate

def test_annotate_renders_in_canonical_order_whatever_order_it_is_passed(priced):
    forward = build_chart_doc(priced, 'structure',
                              annotate=('geometry', 'premium', 'el'))
    backward = build_chart_doc(priced, 'structure',
                               annotate=('el', 'premium', 'geometry'))
    assert canonical_json(forward) == canonical_json(backward)
    assert [line.split(' ')[0]
            for line in layers_on(forward, 'occ')[0].label_lines[1:]] \
        == ['premium', 'el']


def test_annotate_vocabulary_is_checked(priced):
    with pytest.raises(ValueError, match='unknown annotate field'):
        build_chart_doc(priced, 'structure', annotate=('geometry', 'margin'))


def test_the_empty_tuple_gives_bare_rectangles(priced):
    doc = build_chart_doc(priced, 'structure', annotate=())
    assert all(b.label_lines == () for b in layers_on(doc, 'occ'))
    # the headline survives, since it names the block rather than annotating it
    assert layers_on(doc, 'occ')[0].label == 'Occ1'


def test_the_default_is_what_a_reader_asks_first(priced):
    assert DEFAULT_ANNOTATE == ('geometry', 'premium', 'el', 'lr')
    assert set(DEFAULT_ANNOTATE) <= set(ANNOTATE_FIELDS)
    doc = build_chart_doc(priced, 'structure')
    assert len(layers_on(doc, 'occ')[0].label_lines) == 4


def test_every_field_resolves_on_a_priced_object(priced):
    """The whole vocabulary, so a field cannot rot unnoticed."""
    doc = build_chart_doc(priced, 'structure', annotate=ANNOTATE_FIELDS)
    lines = lines_of(layers_on(doc, 'occ')[0])
    for token in ('po ', 'premium ', 'el ', 'lr ', 'rol ', 'lol ', 'sd ',
                  'pr attach ', 'pr detach ', 'cede '):
        assert token in lines


# -------------------------------------------------------------- the document

def test_tex_is_total_over_the_block_strings(priced):
    doc = build_chart_doc(priced, 'structure', annotate=ANNOTATE_FIELDS)
    assert set(doc.tex) == set(human_strings(doc))
    assert 'Occ1' in doc.tex
    assert '50.0% po 5,000 xs 10,000' in doc.tex


def test_the_document_round_trips_hash_for_hash(priced):
    doc = build_chart_doc(priced, 'structure', lee=True)
    back = load_chart_doc(json.loads(canonical_json(doc)))
    assert back.blocks == doc.blocks
    assert all(isinstance(b, TowerBlock) for b in back.blocks)
    assert back.hash == doc.hash


def test_deterministic(priced):
    assert canonical_json(build_chart_doc(priced, 'structure')) == \
        canonical_json(build_chart_doc(priced, 'structure'))


def test_every_panel_is_a_tower_or_a_lee_curve(net_of):
    doc = build_chart_doc(net_of, 'structure', lee=True)
    kinds = {p.id: p.kind for p in doc.panels}
    assert kinds == {'gross': 'tower', 'occ': 'tower', 'occ_lee': 'xy',
                     'agg': 'tower', 'agg_lee': 'xy'}


def test_a_stage_with_no_program_contributes_no_panel():
    doc = build_chart_doc(build(_QUOTA, update=False), 'structure')
    assert [p.id for p in doc.panels] == ['gross', 'agg']


def test_the_placement_axis_is_a_unitless_share(declared):
    doc = build_chart_doc(declared, 'structure')
    axes = {a.id: a for a in doc.axes}
    for panel in doc.panels:
        if panel.kind != 'tower':
            continue
        assert axes[panel.x_axis].unit == 'ratio'
        assert axes[panel.x_axis].full_range == (0.0, 1.0)
        assert axes[panel.y_axis].unit == 'currency'
        # a tower is interrogated by loss, never by placement
        assert panel.read_axis == 'y'


# -------------------------------------------------------------- formatting

def test_money_never_reaches_for_scientific_notation():
    assert _money(10000.000000001) == '10,000'
    assert _money(20795.2) == '20,795'
    assert _money(15) == '15'
    assert _money(0.5) == '0.5'
    assert _money(1.5) == '1.5'
    assert _money(np.inf) == 'unlimited'


def test_a_unit_scaled_book_keeps_its_digits():
    """A layer of 1.5 xs 0.5 must not read as 2 xs 0."""
    assert _terms(1.0, 1.5, 0.5, 'occ') == '1.5 xs 0.5'


# -------------------------------------------------------------- rendering

def test_the_tower_renders_natively(net_of):
    """No degradation, no confession: matplotlib draws a tower as a tower."""
    import matplotlib
    matplotlib.use('Agg')
    from aggregate.plots import plot_chartdoc, plt
    from aggregate.plots._chartdoc import _NATIVE

    assert 'tower' in _NATIVE
    doc = build_chart_doc(net_of, 'structure', lee=True)
    fig = plot_chartdoc(doc, strict=True)
    assert len(fig.axes) == len(doc.panels)
    plt.close(fig)


def test_the_placement_axis_carries_no_ticks(net_of):
    """Width is share, read by comparison, never off a scale."""
    import matplotlib
    matplotlib.use('Agg')
    from aggregate.plots import plot_chartdoc, plt

    doc = build_chart_doc(net_of, 'structure')
    fig = plot_chartdoc(doc)
    tower = fig.axes[[p.id for p in doc.panels].index('occ')]
    assert list(tower.get_xticks()) == []
    assert tower.get_xlabel() == ''
    plt.close(fig)


def test_the_quantity_axis_is_ticked_at_the_boundaries(net_of):
    """A tower is read at its breaks, in currency, not on a scale."""
    import matplotlib
    matplotlib.use('Agg')
    from aggregate.plots import plot_chartdoc, plt

    doc = build_chart_doc(net_of, 'structure')
    fig = plot_chartdoc(doc)
    tower = fig.axes[[p.id for p in doc.panels].index('occ')]
    assert list(tower.get_yticks()) == [5.0, 20.0]
    assert [lab.get_text() for lab in tower.get_yticklabels()] == ['5', '20']
    plt.close(fig)


def test_a_tower_figure_is_wider_where_the_curves_are(net_of):
    """A tower is a strip; a curve beside it gets the room it needs."""
    import matplotlib
    matplotlib.use('Agg')
    from aggregate.plots import plot_chartdoc, plt

    doc = build_chart_doc(net_of, 'structure', lee=True)
    fig = plot_chartdoc(doc)
    ids = [p.id for p in doc.panels]
    tower = fig.axes[ids.index('occ')].get_position().width
    curve = fig.axes[ids.index('occ_lee')].get_position().width
    assert curve > 1.5 * tower
    plt.close(fig)


def test_a_one_stage_document_shares_its_loss_axis():
    """The tower and the curve beside it are one reading, so one axis."""
    import matplotlib
    matplotlib.use('Agg')
    from aggregate.plots import plot_chartdoc, plt

    one = build('agg CS.OneStage 5 claims 100 xs 0 sev lognorm 10 cv .75 '
                'occurrence net of 15 xs 5 poisson')
    doc = build_chart_doc(one, 'structure', lee=True)
    assert len({p.y_axis for p in doc.panels}) == 1
    fig = plot_chartdoc(doc)
    limits = {tuple(ax.get_ylim()) for ax in fig.axes}
    assert len(limits) == 1
    plt.close(fig)


def test_two_stages_never_share_one_window(net_of):
    """A per-claim loss and an annual aggregate are different quantities."""
    import matplotlib
    matplotlib.use('Agg')
    from aggregate.plots import plot_chartdoc, plt

    doc = build_chart_doc(net_of, 'structure', lee=True)
    fig = plot_chartdoc(doc)
    ids = [p.id for p in doc.panels]
    assert fig.axes[ids.index('occ')].get_ylim()         != fig.axes[ids.index('agg')].get_ylim()
    plt.close(fig)


def test_a_renderer_without_the_kind_says_so():
    """Honest degradation: a tower is refused by name, never drawn as xy."""
    import matplotlib
    matplotlib.use('Agg')
    from aggregate.charts.ir import ChartCapabilityError
    from aggregate.plots import _chartdoc

    doc = build_chart_doc(build(_NET_OF, update=False), 'structure')
    native = _chartdoc._NATIVE
    try:
        _chartdoc._NATIVE = {'xy', 'heatmap'}
        with pytest.raises(ChartCapabilityError, match='not yet realized'):
            _chartdoc.plot_chartdoc(doc)
    finally:
        _chartdoc._NATIVE = native
