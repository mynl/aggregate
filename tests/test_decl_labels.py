"""DecL human labels, quoted names, and expense grouping ([DecL-Labels]).

Covers the ``as`` display-label clause on objects (``agg`` / ``pnl`` / ``sev`` /
``port``), on the premium head, on reinsurance cessions, and the two-level
expense grammar (``and`` combines into one leg; juxtaposition makes separate
legs). No computed value changes -- labels land in dict keys, column headers, and
repr / exhibit titles. See ``dev/plan-decl-labels.md``.

The DecL programs here are mirrored in ``src/aggregate/agg/decl-testers.agg``
section Y (this module is the canonical source).
"""
import pytest

from aggregate import Underwriter, build
from aggregate.decl_writer import spec_to_decl

TOL = 1e-6

_PNL_BASE = ('pnl B 1000 premium less agg B_e 850 loss sev lognorm 100 cv 1 poisson')


@pytest.fixture(scope="module")
def uw():
    return Underwriter()


def _parse(uw, program):
    return uw.parser.parse(uw.lexer.tokenize(program))


# ----------------------------------------------------------------------
# Object display labels
# ----------------------------------------------------------------------
def test_no_label_label_falls_back_to_name():
    a = build('agg Plain 100 claims sev lognorm 100 cv 2 poisson')
    assert a._label is None                 # no explicit label stored
    assert a.label == a.name == 'Plain'     # resolved property falls back to name


def test_quoted_object_label():
    a = build('agg GrossBook as "Gross Book P&L" 100 claims sev lognorm 100 cv 2 poisson')
    assert a.name == 'GrossBook'                    # identity handle unchanged
    assert a.label == 'Gross Book P&L'      # spaces carried by quotes
    assert a.label == 'Gross Book P&L'
    assert 'Gross Book P&L' in repr(a)


def test_bareword_object_label_skips_quotes():
    a = build('agg Lae as lae 100 claims sev lognorm 100 cv 2 poisson')
    assert a.label == 'lae'
    assert a.name == 'Lae'


def test_sev_object_label():
    s = build('sev MySev as "My Severity" lognorm 100 cv 2')
    assert s.name == 'MySev'
    assert s.label == 'My Severity'
    assert s.label == 'My Severity'


def test_pnl_object_label():
    p = build(_PNL_BASE.replace('pnl B', 'pnl B as "My Book"'))
    assert p.name == 'B'
    assert p.label == 'My Book'
    assert 'My Book' in repr(p)


def test_port_object_label():
    p = build('port MyPort as "My Portfolio" '
              'agg A 50 claims sev lognorm 100 cv 2 poisson '
              'agg B 50 claims sev lognorm 100 cv 2 poisson')
    assert p.name == 'MyPort'
    assert p.label == 'My Portfolio'
    assert p.label == 'My Portfolio'


def test_title_name_shows_label_and_identity():
    a = build('agg GrossBook as "Gross Book" 100 claims sev lognorm 100 cv 2 poisson')
    assert a._title_name == 'Gross Book (GrossBook)'


# ----------------------------------------------------------------------
# `as` is a reserved word
# ----------------------------------------------------------------------
def test_as_is_reserved(uw):
    # a bare `as` where an identifier is expected must not lex as an ID
    with pytest.raises(Exception):
        _parse(uw, 'agg as 100 claims sev lognorm 100 cv 2 poisson')


# ----------------------------------------------------------------------
# Premium label -> consideration leg key
# ----------------------------------------------------------------------
def _lines(pnl):
    """The declared-leg Line labels of the stats sheet (leg-level labels
    live there; summary_df is the fixed card)."""
    return list(pnl.stats_df.index.get_level_values('Line'))


def _leg(pnl, label):
    return pnl.stats_df.xs(label, level='Line').iloc[0]


def test_premium_label_names_consideration_leg():
    p = build(_PNL_BASE.replace('1000 premium', '1000 premium as "GWP"'))
    assert 'GWP' in _lines(p)
    assert 'premium' not in _lines(p)
    assert _leg(p, 'GWP')['EX'] == pytest.approx(1000.0)


def test_premium_no_label_keeps_default_leg():
    # one canonical default across faces: 'premium' (the walk's gross step
    # uses the same name; 'consideration' retired as the DecL default)
    p = build(_PNL_BASE)
    assert 'premium' in _lines(p)


# ----------------------------------------------------------------------
# Expense grouping: `and` combines, juxtaposition separates
# ----------------------------------------------------------------------
def test_single_group_and_joined_is_one_leg():
    # backward compatible: and-joined terms sum into one leg named 'expense'
    p = build(_PNL_BASE + ' less 25% premium expense and 200 fixed expense')
    assert 'expense' in _lines(p)
    # ledger rows are signed: a sold expense books negative
    assert _leg(p, 'expense')['EX'] == pytest.approx(-(0.25 * 1000 + 200))


def test_juxtaposed_groups_are_separate_legs():
    p = build(_PNL_BASE + ' less 25% premium expense 200 fixed expense')
    lines = _lines(p)
    # two separate legs, default basis-derived names
    assert 'premium expense' in lines
    assert 'fixed expense' in lines
    assert 'expense' not in lines
    assert _leg(p, 'premium expense')['EX'] == pytest.approx(-250.0)
    assert _leg(p, 'fixed expense')['EX'] == pytest.approx(-200.0)


def test_labeled_expense_groups():
    p = build(_PNL_BASE + ' less 25% premium expense as acq 30% loss expense as lae')
    lines = _lines(p)
    assert 'acq' in lines and 'lae' in lines
    assert _leg(p, 'acq')['EX'] == pytest.approx(-250.0)


def test_combine_vs_separate_total_matches():
    # combined (and) vs separate (juxtaposition) must give the same total expense
    combined = build(_PNL_BASE + ' less 25% premium expense and 200 fixed expense')
    separate = build(_PNL_BASE + ' less 25% premium expense 200 fixed expense')
    tot_c = combined.stats_df.loc[('Obligation', 'Total'), 'EX']
    tot_s = separate.stats_df.loc[('Obligation', 'Total'), 'EX']
    assert tot_c == pytest.approx(tot_s, rel=TOL)


def test_labeled_group_combines_multiple_terms():
    p = build(_PNL_BASE + ' less 25% premium expense and 200 fixed expense as "acquisition"')
    assert 'acquisition' in _lines(p)
    assert _leg(p, 'acquisition')['EX'] == pytest.approx(-(0.25 * 1000 + 200))


# ----------------------------------------------------------------------
# Reins-clause label -> cession leg / group labels in the ledger
# ----------------------------------------------------------------------
def test_reins_label_names_cession_rows():
    # the consolidated pnl nets the cession out
    # ([Decision-PnL-Is-Consolidated]); the reins ``as`` label names the
    # cover's step and its legs on the xpnl walk
    p = build('xpnl RP 1000 premium less agg RP_e 850 loss sev lognorm 100 cv 1 '
              'occurrence net of 100 xs 200 deposit 50 as "Cat XL" poisson')
    assert 'Cat XL premium' in _lines(p)
    assert 'ceded occ premium' not in _lines(p)
    # the cover's label becomes the step; its result / running net live at
    # (step, 'Margin', ...)
    q = build('xpnl RQ 1000 premium less agg RQ_e 850 loss sev lognorm 100 cv 1 '
              'poisson aggregate net of 500 xs 1000 deposit 50 as "Stop Loss"')
    lines = _lines(q)
    for row in ('Stop Loss premium', 'Stop Loss recovery'):
        assert row in lines, row
    s = q.stats_df
    assert ('Stop Loss', 'Margin', 'Total') in s.index
    assert ('Stop Loss', 'Margin', 'Net') in s.index


def test_reins_no_label_keeps_structural_rows():
    p = build('xpnl RP 1000 premium less agg RP_e 850 loss sev lognorm 100 cv 1 '
              'occurrence net of 100 xs 200 deposit 50 poisson')
    assert 'occ 100 xs 200 premium' in _lines(p)


# ----------------------------------------------------------------------
# Round-trip: labels survive spec -> DecL -> spec
# ----------------------------------------------------------------------
@pytest.mark.parametrize("program", [
    'agg GrossBook as "Gross Book" 100 claims sev lognorm 100 cv 2 poisson',
    'sev MySev as lae lognorm 100 cv 2',
    'pnl Book as "My Book" 1000 premium as GWP less agg Book_e 850 loss '
    'sev lognorm 100 cv 1 poisson less 25% premium expense as acq 30% loss expense as lae',
    'pnl RP 1000 premium less agg RP_e 850 loss sev lognorm 100 cv 1 '
    'occurrence net of 100 xs 200 deposit 50 as "Cat XL" poisson',
    # interior labels ([DecL-Labels-Everywhere]: exposure / layer / severity)
    'agg E1 10000 premium as "GWP 2026" at 0.65 lr 1000 xs 0 sev lognorm 100 cv 2 poisson',
    'agg E2 100 claims as "Claim count" sev lognorm 100 cv 2 poisson',
    'agg L1 100 claims 1000 xs 500 as "Working Layer" sev lognorm 100 cv 2 poisson',
    'agg S1 100 claims sev lognorm 100 cv 2 as "ISO ME B" poisson',
    'agg ALL 10000 premium as GWP at 0.65 lr 1000 xs 500 as "Layer 1" '
    'sev lognorm 100 cv 2 as "ISO" poisson',
])
def test_label_roundtrip(uw, program):
    kind, name, spec = _parse(uw, program)
    rendered = spec_to_decl(spec, kind, name).replace('\n', ' ')
    _kind2, _name2, spec2 = _parse(uw, rendered)
    assert spec == spec2


# ----------------------------------------------------------------------
# Interior sub-object labels ([DecL-Labels-Everywhere]): exposure / layer /
# inline severity clause -> Aggregate.label_map + the labels namespace view
# ----------------------------------------------------------------------
def test_reins_labels_pool_into_label_map():
    """Per-layer cession labels live in label_map as sparse {index: label}
    dicts -- a.labels.occ_reins[i] -- with no parallel label attributes:
    the labels namespace is the complete interior-label surface."""
    a = build('agg RL 100 claims sev lognorm 100 cv 2 '
              'occurrence net of 50 xs 50 and 100 xs 100 as "Occ Layer B" '
              'poisson aggregate net of 2000 xs 8000 as "Agg 2 x 8"',
              update=False)
    assert a.labels.occ_reins == {1: 'Occ Layer B'}   # sparse: layer 0 unlabeled
    assert a.labels.occ_reins[1] == 'Occ Layer B'
    assert a.labels.agg_reins == {0: 'Agg 2 x 8'}
    assert a.label_map['occ_reins'] == {1: 'Occ Layer B'}
    assert not hasattr(a, 'occ_reins_label')          # one home, no parallels
    assert not hasattr(a, 'agg_reins_label')
    # unlabeled cessions leave the sites absent (None via the view)
    b = build('agg RU 100 claims sev lognorm 100 cv 2 '
              'occurrence net of 50 xs 50 poisson', update=False)
    assert b.labels.occ_reins is None
    assert 'occ_reins' not in b.label_map


def test_exposure_label_premium_mid_clause():
    a = build('agg E 10000 premium as "GWP 2026" at 0.65 lr 1000 xs 0 '
              'sev lognorm 100 cv 2 poisson')
    assert a.label_map['exposure'] == 'GWP 2026'
    assert a.labels.exposure == 'GWP 2026'


def test_exposure_label_claims_end_clause():
    a = build('agg E 100 claims as "Claim count" sev lognorm 100 cv 2 poisson')
    assert a.labels.exposure == 'Claim count'


def test_severity_clause_label():
    a = build('agg S 100 claims sev lognorm 100 cv 2 as "ISO ME B" poisson')
    assert a.label_map['severity'] == 'ISO ME B'
    assert a.labels.severity == 'ISO ME B'


def test_layer_label():
    a = build('agg L 100 claims 1000 xs 500 as "Working Layer" '
              'sev lognorm 100 cv 2 poisson')
    assert a.labels.layer == 'Working Layer'


def test_layer_label_does_not_corrupt_following_sev(uw):
    # S2 ambiguity guard: the layer's ``as "..."`` must attach to the layer,
    # never be eaten by the severity clause that follows.
    _kind, _name, spec = _parse(
        uw, 'agg L 100 claims 1000 xs 500 as "Working Layer" '
            'sev lognorm 100 cv 2 poisson')
    assert spec['exp_limit'] == 1000.0
    assert spec['exp_attachment'] == 500.0
    assert spec['sev_name'] == 'lognorm'
    assert spec['label_map'] == {'layer': 'Working Layer'}


def test_all_interior_labels_combine():
    a = build('agg ALL 10000 premium as GWP at 0.65 lr 1000 xs 500 as "Layer 1" '
              'sev lognorm 100 cv 2 as "ISO" poisson')
    assert a.label_map == {'exposure': 'GWP', 'layer': 'Layer 1',
                           'severity': 'ISO'}


def test_no_interior_labels_leaves_label_map_empty():
    a = build('agg P 100 claims sev lognorm 100 cv 2 poisson')
    assert a.label_map == {}
    # a missing interior site resolves to None (never raises)
    assert a.labels.exposure is None
    assert a.labels.severity is None


def test_bivariate_strips_shared_exposure_label():
    # bivariate reuses the shared exposures production but is out of scope for
    # labels; a label there must not leak into the bivariate spec.
    b = build('bivariate BV 25 claims as "shared count" '
              'agg A dfreq [0 1] [.3 .7] sev lognorm 40 cv 1.2 '
              'agg B dfreq [0 1] [.5 .5] sev lognorm 60 cv 1.5 '
              'copula normal 0.4 poisson')
    assert type(b).__name__ == 'BivariateAggregate'


# ----------------------------------------------------------------------
# Distortion label realignment (D6): name = kind handle, label resolves
# label -> auto-pretty derived default -> handle
# ----------------------------------------------------------------------
def test_distortion_name_is_kind_handle():
    from aggregate.spectral import Distortion
    d = Distortion.ph(0.5)
    assert d.name == 'ph'                 # handle is the kind
    assert d.label == 'PH(0.5)'    # auto-pretty derived default
    assert str(d) == 'PH(0.5)'


def test_distortion_explicit_label_wins():
    from aggregate.spectral import Distortion
    d = Distortion('ph', a=0.5, label='My PH')
    assert d.name == 'ph'
    assert d.label == 'My PH'


def test_distortion_direct_construction_gets_auto_pretty():
    from aggregate.spectral import PHDistortion
    d = PHDistortion(a=0.7)
    assert d.name == 'ph'
    assert d.label == 'PH(0.7)'


def test_use_labels_switch_invalidates_renamer():
    a = build('agg P 100 claims sev lognorm 100 cv 2 poisson')
    assert a.use_labels is True
    a.use_labels = False
    assert a.use_labels is False


# ----------------------------------------------------------------------
# Phase 4: Portfolio exhibits route the unit axis through the label renamer
# (labeled unit -> its display label; unlabeled -> handle; total -> total)
# ----------------------------------------------------------------------
_PORT_LABELED = ('port PP '
                 'agg A as "Line A" 100 claims sev lognorm 100 cv 2 poisson '
                 'agg B 50 claims sev lognorm 80 cv 1.5 poisson')


def test_portfolio_renamer_maps_labeled_units():
    p = build(_PORT_LABELED, update=False)
    assert p.renamer == {'A': 'Line A', 'B': 'B', 'total': 'total'}


def test_portfolio_unlabeled_renamer_is_identity():
    p = build('port QQ agg X 100 claims sev lognorm 100 cv 2 poisson '
              'agg Y 50 claims sev lognorm 80 cv 1.5 poisson', update=False)
    assert p.renamer == {'X': 'X', 'Y': 'Y', 'total': 'total'}


def test_portfolio_summary_df_relabels_unit_axis():
    p = build(_PORT_LABELED)
    units = p.summary_df.index.get_level_values('unit').unique().tolist()
    assert 'Line A' in units and 'B' in units and 'total' in units
    # use_labels=False serves the raw handle-keyed view
    p.use_labels = False
    assert 'A' in p.summary_df.index.get_level_values('unit')


def test_portfolio_pricing_df_relabels_unit_columns():
    p = build(_PORT_LABELED)
    p.calibrate_distortions(0.1, p=0.99)
    r = p.analyze_distortions(p=0.99)
    assert 'Line A' in r.pricing_df.columns
    assert 'total' in r.pricing_df.columns
