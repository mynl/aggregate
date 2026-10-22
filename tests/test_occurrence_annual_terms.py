"""[Occurrence-Annual-Terms]: an annual aggregate deductible and an aggregate
limit on an occurrence layer.

``aad d`` / ``agglimit m`` decorate one occurrence layer, directly after the
layer, so the reinsurer pays ``min((R - d)+, m)`` on the layer's annual
recovery ``R``. Amounts are for 100% of the layer and the placement share
applies after. Step ``[Annual-Terms-DecL]`` covers the grammar, the spec keys,
the refusals and the plain ``agg``: on ``occurrence ceded to`` the terms are
honored as the aggregate stage, exactly; on ``net of`` they are ignored with an
:class:`~aggregate.constants.IgnoredDecLClauseWarning` and the stored program
keeps them for a ``pnl`` / ``xpnl``. See
``dev/done/plan-occurrence-annual-deductible-limit.md``.

These DecL programs are mirrored in ``src/aggregate/agg/decl-testers.agg``
under the ``OAT.*`` section.
"""

import numpy as np
import pytest

from aggregate import Underwriter
from aggregate.constants import IgnoredDecLClauseWarning

#: A small book; the layer clause is spliced in.
_BOOK = 'agg {name} 10 claims sev lognorm 100 cv 2 occurrence {kind} {layer} poisson {tail}'


def _prog(name, layer, kind='net of', tail=''):
    """One plain ``agg`` program on the small book."""
    return _BOOK.format(name=name, kind=kind, layer=layer, tail=tail).strip()


@pytest.fixture
def uw():
    """A clean underwriter (no databases) with eager update."""
    return Underwriter(databases=None, update=True)


def _occ_keys(uw, program):
    """The occurrence spec keys of one parsed program."""
    _, _, spec = uw.parser.parse(program)
    return {k: v for k, v in spec.items() if k.startswith('occ_')}


# ----------------------------------------------------------------------
# grammar / transformer: the three forms, with and without a share
# ----------------------------------------------------------------------
@pytest.mark.parametrize('layer, share, aad, agglimit', [
    ('90% po 100 xs 100 aad 50', 0.9, 50.0, np.inf),
    ('750 xs 250 agglimit 500', 1.0, 0.0, 500.0),
    ('650 xs 350 aad 50 agglimit 500', 1.0, 50.0, 500.0),
    ('675 po 750 xs 250 aad 50 agglimit 500', 0.9, 50.0, 500.0),
])
def test_forms_parse_to_the_spec_keys(uw, layer, share, aad, agglimit):
    occ = _occ_keys(uw, _prog('T', layer))
    assert occ['occ_reins'][0][0] == pytest.approx(share)
    assert occ['occ_reins_aad'] == aad
    assert occ['occ_reins_agglimit'] == agglimit


def test_terms_sit_before_the_premium(uw):
    occ = _occ_keys(uw, _prog(
        'T', '750 xs 250 aad 50 agglimit 500 deposit 100 cede 10% as Layer'))
    assert occ['occ_reins_aad'] == 50.0
    assert occ['occ_reins_agglimit'] == 500.0
    assert occ['occ_reins_premium'] == [('deposit', 100.0)]
    assert occ['occ_reins_cede'] == [0.1]
    assert occ['occ_reins_label'] == ['Layer']


def test_undecorated_layer_has_no_annual_keys(uw):
    occ = _occ_keys(uw, _prog('T', '750 xs 250'))
    assert 'occ_reins_aad' not in occ
    assert 'occ_reins_agglimit' not in occ


# ----------------------------------------------------------------------
# refusals
# ----------------------------------------------------------------------
@pytest.mark.parametrize('program, match', [
    ('agg T 10 claims sev lognorm 100 cv 2 poisson aggregate net of 750 xs 250 aad 50',
     'aggregate layer is itself'),
    (_prog('T', '750 xs 250 aad 50 and 1000 xs 1000'),
     'single occurrence layer'),
    (_prog('T', '750 xs 250 aad 50 deposit 100 reinstatements 1 free'),
     "cannot combine with 'reinstatements'"),
    (_prog('T', '750 xs 250 aad -5'), "'aad' must be finite and >= 0"),
    (_prog('T', '750 xs 250 agglimit 0'), "'agglimit' must be finite and > 0"),
    (_prog('T', '750 xs 250 agglimit inf'), "'agglimit' must be finite and > 0"),
    (_prog('T', '750 xs 250 aad 50 aad 60'), 'parse error'),
])
def test_refusals(uw, program, match):
    with pytest.raises(ValueError, match=match):
        uw.parser.parse(program)


# ----------------------------------------------------------------------
# plain agg, ``ceded to``: honored as the aggregate stage, exactly
# ----------------------------------------------------------------------
def test_ceded_to_equals_the_hand_written_aggregate_stage(uw):
    a = uw(_prog('A', '90% po 750 xs 250 aad 50 agglimit 500', 'ceded to'))
    b = uw(_prog('B', '90% po 750 xs 250', 'ceded to',
                 'aggregate ceded to 450 xs 45'))
    np.testing.assert_array_equal(a.agg_density, b.agg_density)
    assert a.agg_reins == [(1.0, 450.0, 45.0)]
    assert a.agg_kind == 'ceded to'
    assert a.labels.agg_reins == {0: 'AAD 50, agglimit 500'}


def test_ceded_to_aad_alone_is_unlimited(uw):
    a = uw(_prog('A', '750 xs 250 aad 50', 'ceded to'))
    assert a.agg_reins == [(1.0, np.inf, 50.0)]
    assert a.labels.agg_reins == {0: 'AAD 50'}


def test_ceded_to_with_an_aggregate_clause_raises(uw):
    with pytest.raises(ValueError, match="cannot also carry an 'aggregate'"):
        uw(_prog('A', '750 xs 250 aad 50', 'ceded to',
                 'aggregate ceded to 300 xs 0'))


def test_ceded_to_keeps_the_declaration(uw):
    uw(_prog('A', '750 xs 250 aad 50 agglimit 500', 'ceded to'))
    stored = uw[('agg', 'A')].spec
    assert stored['occ_reins_aad'] == 50.0
    assert stored['occ_reins_agglimit'] == 500.0
    assert 'agg_reins' not in stored


# ----------------------------------------------------------------------
# plain agg, ``net of``: ignored with one warning, kept for a P&L
# ----------------------------------------------------------------------
def test_net_of_is_ignored_with_one_warning(uw):
    with pytest.warns(IgnoredDecLClauseWarning) as rec:
        a = uw(_prog('N', '750 xs 250 aad 50 agglimit 500'))
    ours = [w for w in rec
            if issubclass(w.category, IgnoredDecLClauseWarning)]
    assert len(ours) == 1
    assert 'aad / agglimit' in str(ours[0].message)
    plain = uw(_prog('P', '750 xs 250'))
    np.testing.assert_array_equal(a.agg_density, plain.agg_density)


def test_net_of_keeps_the_declaration(uw):
    with pytest.warns(IgnoredDecLClauseWarning):
        a = uw(_prog('N', '750 xs 250 aad 50 agglimit 500'))
    stored = uw[('agg', 'N')].spec
    assert stored['occ_reins_aad'] == 50.0
    assert stored['occ_reins_agglimit'] == 500.0
    assert 'aad 50 agglimit 500' in a.program


# ----------------------------------------------------------------------
# the writer renders the terms back as written
# ----------------------------------------------------------------------
@pytest.mark.parametrize('layer', [
    '90% po 100 xs 100 aad 50',
    '750 xs 250 agglimit 500',
    '750 xs 250 aad 50 agglimit 500 deposit 100 cede 0.1',
    '750 xs 250 aad 0',
])
def test_writer_round_trips(uw, layer):
    from aggregate.decl_writer import format_program
    program = _prog('W', layer)
    _, _, spec = uw.parser.parse(program)
    again = format_program(program, layout='terse')
    _, _, spec2 = uw.parser.parse(again)
    assert spec2 == spec
    assert layer.split(' xs ')[1].split(' ', 1)[1] in again


# ----------------------------------------------------------------------
# bvagg components follow the same two routes
# ----------------------------------------------------------------------
@pytest.fixture
def uwb():
    """A clean underwriter with lazy update: the routes act in ``__init__``."""
    return Underwriter(databases=None, update=False)


def test_bvagg_net_of_is_ignored_naming_the_unit(uwb):
    with pytest.warns(IgnoredDecLClauseWarning, match='V1: ignoring the aad'):
        b = uwb('grossceded agg V1 dfreq [3] dsev [2 4 6 8 10 12] '
                'occurrence net of 6 xs 6 aad 2')
    assert b._nc_agg.occ_reins is not None


def test_bvagg_ceded_to_becomes_the_aggregate_stage(uwb):
    b = uwb('bivariate V2 5 claims '
            'agg V2a dfreq [0 1] [.5 .5] dsev [2 4 6 8 10 12] '
            'occurrence ceded to 6 xs 6 aad 2 agglimit 3 '
            'agg V2b dfreq [0 1] [.5 .5] dsev [1 2 3 4] copula normal 0.5')
    assert b.units[0].agg_reins == [(1.0, 3.0, 2.0)]
    assert b.units[0].agg_kind == 'ceded to'


# ======================================================================
# step [Annual-Terms-PnL]: the cedant's net in pnl / xpnl
# ======================================================================
from aggregate.constants import (CoarseJointGridWarning,  # noqa: E402
                                 ZeroPremiumCessionWarning)
from aggregate.contract_terms import AnnualLayerTerms  # noqa: E402

#: The book of the occurrence-layer note (section 5): a 1,000 limit book with
#: a 750 xs 250 layer. ``{layer}`` is the occurrence cession clause.
_NOTE = ('{face} {name} 2000 premium less agg {name}_e 2000 premium at 70% lr '
         '1000 xs 0 sev 120 * lomax 2.5 occurrence net of {layer} '
         'mixed gamma .185 {tail}')


def _pnl(uw, name, layer, face='xpnl', tail=''):
    """Build one P&L on the note's book, warnings silenced."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return uw(_NOTE.format(face=face, name=name, layer=layer,
                               tail=tail).strip())


def test_terms_maps_and_validation():
    t = AnnualLayerTerms(limit=675.0, annual_deductible=45.0,
                         annual_limit=450.0, deposit=300.0, share=0.9)
    R = np.array([0.0, 30.0, 45.0, 200.0, 495.0, 900.0])
    np.testing.assert_array_equal(
        t.recovery(R), np.minimum(np.maximum(R - 45.0, 0.0), 450.0))
    np.testing.assert_array_equal(t.phi(R), t.recovery(R))
    np.testing.assert_array_equal(t.reinstatement_premium(R), np.zeros(6))
    np.testing.assert_array_equal(t.ceded_premium(R), np.full(6, 300.0))
    assert t.target_leg == 'ceded_loss'
    assert t.loss_basis == 'occurrence_recovery'
    assert t.decl_clause == 'aad 50 agglimit 500'
    assert AnnualLayerTerms(limit=750.0, annual_deductible=50.0).decl_clause \
        == 'aad 50'
    assert AnnualLayerTerms(limit=750.0, annual_limit=500.0).decl_clause \
        == 'agglimit 500'
    for bad in (dict(limit=0.0), dict(limit=750.0, annual_deductible=-1.0),
                dict(limit=750.0, annual_limit=0.0),
                dict(limit=750.0, deposit=-1.0),
                dict(limit=750.0, share=0.0)):
        with pytest.raises(ValueError, match='AnnualLayerTerms'):
            AnnualLayerTerms(**bad)


@pytest.mark.parametrize('annual, reinst', [
    ('agglimit 1500 deposit 300', 'deposit 300 reinstatements 1 free'),
    ('agglimit 750 deposit 300', 'deposit 300 no reinstatements'),
])
def test_matches_reinstatements_exactly(uw, annual, reinst):
    """Same annual map ``min(R, Y)`` and both premiums flat: every leg of
    both faces is identical, row for row."""
    for face in ('xpnl', 'pnl'):
        a = _pnl(uw, f'A{face}', f'750 xs 250 {annual}', face)
        b = _pnl(uw, f'B{face}', f'750 xs 250 {reinst}', face)
        assert type(a.engine.reinstatement_terms) is AnnualLayerTerms
        ea, eb = a.economic_df, b.economic_df
        assert ea.shape == eb.shape
        np.testing.assert_allclose(ea[['EX', 'SD']].to_numpy(),
                                   eb[['EX', 'SD']].to_numpy(),
                                   rtol=0, atol=1e-9)


def test_recovery_and_net_are_the_pushforward(uw):
    """90% share, terms at 100%: recovery ``min((R - 45)+, 450)`` off the
    builder's own joint, and the consolidated net loss ``E[L]`` less it."""
    layer = '90% po 750 xs 250 aad 50 agglimit 500 deposit 300 cede 10%'
    x = _pnl(uw, 'PFX', layer)
    t = x.engine.reinstatement_terms
    assert (t.annual_deductible, t.annual_limit) == pytest.approx((45.0, 450.0))
    assert t.limit == pytest.approx(675.0)
    bv = x._source
    hand = bv.transformed_moments(
        lambda l, r: np.minimum(np.maximum(r - 45.0, 0.0), 450.0))['mean']
    e_l = bv.transformed_moments(lambda l, r: l + 0.0 * r)['mean']
    s = x.economic_df
    step = 'occ 90% po 750 xs 250 aad 50 agglimit 500'
    assert s.loc[(step, 'Obligation', f'{step} recovery'), 'EX'] \
        == pytest.approx(hand, abs=1e-9)
    # the deposit is quoted for 100% of the layer and booked at the share,
    # like every other occurrence premium
    assert s.loc[(step, 'Consideration', f'{step} premium'), 'EX'] \
        == pytest.approx(-270.0)
    # cede: the commission leg is cede x the booked deposit
    assert s.loc[(step, 'Obligation', f'{step} commission'), 'EX'] \
        == pytest.approx(0.1 * 270.0)
    p = _pnl(uw, 'PFP', layer, 'pnl')
    assert p.economic_df.xs('Loss (net)', level='Label').iloc[0]['EX'] \
        == pytest.approx(-(e_l - hand), abs=1e-9)
    assert p.economic_df.xs('commission', level='Label').iloc[0]['EX'] \
        == pytest.approx(0.1 * 270.0)


def test_faces_agree(uw):
    x = _pnl(uw, 'FX', '750 xs 250 aad 50 agglimit 500 deposit 300')
    p = _pnl(uw, 'FP', '750 xs 250 aad 50 agglimit 500 deposit 300', 'pnl')
    assert p.est_m == pytest.approx(x.est_m, abs=1e-9)
    assert p.est_sd == pytest.approx(x.est_sd, abs=1e-9)


def test_declared_label_names_the_step(uw):
    x = _pnl(uw, 'LX', '750 xs 250 aad 50 deposit 300 as Layer')
    steps = list(dict.fromkeys(x.economic_df.index.get_level_values('Step')))
    assert steps == ['Gross', 'Layer', 'All']


def test_by_reference_equals_inline(uw):
    """The ``net of`` agg ignores the terms, its stored program keeps them,
    and an ``xpnl`` over it by reference rebuilds them."""
    import warnings
    layer = '750 xs 250 aad 50 agglimit 500 deposit 300'
    with pytest.warns(IgnoredDecLClauseWarning):
        uw('agg RefBook 2000 premium at 70% lr 1000 xs 0 sev 120 * lomax 2.5 '
           f'occurrence net of {layer} mixed gamma .185')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ref = uw('xpnl RefX 2000 premium less agg.RefBook')
    inline = _pnl(uw, 'RefI', layer)
    np.testing.assert_allclose(ref.economic_df[['EX', 'SD']].to_numpy(),
                               inline.economic_df[['EX', 'SD']].to_numpy(),
                               rtol=0, atol=1e-9)


def test_composes_with_an_aggregate_profit_commission(uw):
    """Ruling 7: the aggregate-tier feature rides the same joint and reads the
    net of everything subject, ``L - A(R)``, as it does after reinstatements."""
    tail = 'aggregate net of 500 xs 1200 deposit 100 pc 50% after 10%'
    x = _pnl(uw, 'PCX', '750 xs 250 aad 50 agglimit 500 deposit 300',
             tail=tail)
    t = x.engine.reinstatement_terms
    bv = x._source

    def g(l, r):
        return np.minimum(np.maximum(np.maximum(l - t.recovery(r), 0.0)
                                     - 1200.0, 0.0), 500.0)

    hand = bv.transformed_moments(
        lambda l, r: 0.5 * np.maximum(1.0 - g(l, r) / 100.0 - 0.1, 0.0)
        * 100.0)['mean']
    leg = x.economic_df.xs('profit commission', level='Label').iloc[0]
    assert leg['EX'] == pytest.approx(hand, abs=1e-9)
    assert leg['SD'] > 0
    p = _pnl(uw, 'PCP', '750 xs 250 aad 50 agglimit 500 deposit 300', 'pnl',
             tail=tail)
    assert p.est_m == pytest.approx(x.est_m, abs=1e-9)


def test_premium_is_optional(uw):
    """Ruling 9: no premium clause books zero, with the usual warning."""
    import warnings
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        x = uw(_NOTE.format(face='xpnl', name='ZP', layer='750 xs 250 aad 50',
                            tail='').strip())
    assert any(issubclass(w.category, ZeroPremiumCessionWarning) for w in rec)
    step = 'occ 750 xs 250 aad 50'
    assert x.economic_df.loc[(step, 'Consideration', f'{step} premium'),
                             'EX'] == 0.0


def test_grid_check_reads_the_limit_not_the_deductible(uw):
    """Review ruling 3: a finite aggregate limit narrower than 20 buckets
    warns; the deductible is not checked."""
    import warnings
    with pytest.warns(CoarseJointGridWarning, match='aggregate limit 100'):
        uw(_NOTE.format(face='xpnl', name='GW', tail='',
                        layer='750 xs 250 agglimit 100 deposit 300').strip())
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        uw(_NOTE.format(face='xpnl', name='GQ', tail='',
                        layer='750 xs 250 aad 50 deposit 300').strip())
    assert not [w for w in rec if issubclass(w.category, CoarseJointGridWarning)]


def test_acceptance_reproduces_the_note(uw):
    """Section 5 of the occurrence-layer note, built by hand there with a
    corridor over a bucket-5 joint: net margin 460.70, SD 474.20, layer
    recovery 187.08. The DecL face uses the default joint (bucket 20) and
    agrees to its grid accuracy; on the note's bucket of 5 it agrees to the
    displayed figures."""
    from aggregate._pnl_builders import build_reinstatement_pnl
    layer = '750 xs 250 aad 50 deposit 326.389307'
    x = _pnl(uw, 'Cedant', layer)
    step = 'occ 750 xs 250 aad 50'
    rec = x.economic_df.loc[(step, 'Obligation', f'{step} recovery'), 'EX']
    assert x.est_m == pytest.approx(460.70, abs=0.15)
    assert rec == pytest.approx(187.08, abs=0.15)
    eng = x.engine
    biv = eng.occ_bivariate(views=('gross', 'ceded'), bs=5, total_log2=21)
    fine = build_reinstatement_pnl(eng, source=biv.bivariate,
                                   terms=eng.reinstatement_terms,
                                   gross_premium=2000.0, walk=True)
    fine_rec = fine.economic_df.xs(f'{step} recovery',
                                   level='Label').iloc[0]['EX']
    assert fine.est_m == pytest.approx(460.70, abs=0.005)
    assert fine.est_sd == pytest.approx(474.20, abs=0.005)
    assert fine_rec == pytest.approx(187.08, abs=0.005)


def test_structure_chart_annotates_the_terms(uw):
    from aggregate.charts import build_chart_doc
    x = _pnl(uw, 'CH', '90% po 750 xs 250 aad 50 agglimit 500 deposit 300')
    doc = build_chart_doc(x, 'structure', annotate=('reinstatements',))
    occ = [b for b in doc.blocks if b.panel_id == 'occ' and b.role == 'layer']
    assert occ[0].label_lines == ('aad 50 agglimit 500',)
