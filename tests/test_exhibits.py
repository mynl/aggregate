"""Tests for :mod:`aggregate.exhibits` ([Exhibits-Module] plan).

Two tiers, split on the greater_tables dependency:

* The frame stage (:func:`aggregate.exhibits.exhibit_frames`,
  :func:`aggregate.exhibits.available_exhibits`) is pure pandas and is
  tested without greater_tables.
* IR conversion (:func:`aggregate.exhibits.build_exhibit`,
  :meth:`aggregate.exhibits.Exhibit.to_payload`) runs under
  ``pytest.importorskip('greater_tables')``, including committed
  ``canonical_dict`` snapshots per (exhibit, perspective, kind) that guard
  both the business translation and IR drift. Snapshots are captured by
  ``tests/capture_exhibit_snapshots.py``; regenerate them with

      uv run python tests/capture_exhibit_snapshots.py

  whenever an exhibit's presentation deliberately changes.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from aggregate import build
from aggregate import exhibits
from aggregate.exhibits import (
    EXHIBITS, Exhibit, Perspective, available_exhibits, build_exhibit,
    exhibit_frames,
)

SNAPSHOT_PATH = Path(__file__).parent / 'data' / 'exhibit_snapshots.json'

# DecL programs for the exhibit fixtures, one per first class kind. Mirrored
# EXACTLY in ``tests/capture_exhibit_snapshots.py`` and in section EX of
# ``src/aggregate/agg/decl-testers.agg``.
PROGRAMS = {
    'Aggregate': 'agg EX.Dice dfreq [3] dsev [1:6]',
    'Portfolio': ('port EX.Port '
                  'agg EX.UnitA as "Unit Alpha" 1 claim dsev [1 2 3] fixed '
                  'agg EX.UnitB 2 claims dsev [2 4] fixed'),
    'BivariateAggregate': ('bv EX.BV dfreq [1 2] [.7 .3] dbvsev '
                           '[[10 1 .4] [10 2 .1] [20 1 .2] [20 2 .1] '
                           '[20 3 .2]]'),
    'PnL': ('pnl EX.B 1000 premium less agg EX.Be 850 loss '
            'sev lognorm 100 cv 1 poisson'),
    'Distortion': 'dist EX.PH ph 0.5',
    # ceding fixtures for the reins exhibit ([Exhibits-Reins-Insurer])
    'ReinsAggregate': ('agg EX.Re dfreq [1 2] dsev [10 20 30] '
                       'occurrence net of 10 xs 10'),
    'ReinsPortfolio': ('port EX.RePort '
                       'agg EX.ReA dfreq [1 2] dsev [10 20 30] '
                       'occurrence net of 10 xs 10 '
                       'agg EX.ReB 1 claim dsev [5 10] fixed'),
    # walks, for the economic exhibits ([Exhibits-Economic-Insurer],
    # [Exhibits-Waterfall]). Tower shares atoms and carries a kappa ladder;
    # Peel is stitched, so its ladder is marginal and the waterfall's
    # diversified column blanks.
    'Tower': ('xpnl EX.Tower 1000 prem less agg EX.TowerE 1000 prem at 70% lr '
              'sev lognorm 100 cv 2 '
              'occurrence ceded to 500 xs 500 deposit 100 poisson'),
    'Peel': ('xpnl EX.Peel 1000 premium less agg EX.PeelE 1000 premium at '
             '70% lr sev lognorm 100 cv 2 '
             'occurrence net of 100 xs 100 deposit 60 and 300 xs 200 '
             'deposit 40 poisson peel top-down'),
}
_WALK_EXHIBITS = ['summary', 'stats', 'validation', 'economic',
                  'economic_ratios', 'economic_waterfall']
AGG_PROGRAM = PROGRAMS['Aggregate']
PORT_PROGRAM = PROGRAMS['Portfolio']

# Exhibit names served per fixture after [Exhibits-Reins-Insurer].
_DIAG = ['bs_window', 'tail_behavior']
EXPECTED_EXHIBITS = {
    'Aggregate': ['summary', 'tail', 'stats', 'validation', *_DIAG],
    'Portfolio': ['summary', 'tail', 'stats', 'validation', *_DIAG],
    'BivariateAggregate': ['summary', 'stats', 'validation', 'dependency',
                           'bs_window'],
    'PnL': ['summary', 'stats', 'validation', 'economic', 'economic_ratios'],
    'Distortion': ['summary', 'stats', 'validation'],
    'ReinsAggregate': ['summary', 'tail', 'stats', 'validation', 'reins',
                       *_DIAG],
    'ReinsPortfolio': ['summary', 'tail', 'stats', 'validation', 'reins',
                       *_DIAG],
    'Tower': _WALK_EXHIBITS,
    'Peel': _WALK_EXHIBITS,
    # the pricing result fixtures ([Pricing-Exhibits]); a result serves the
    # pricing leaves and nothing else, because nothing else is registered for
    # its type
    'CalibrationPortfolio': ['pricing.calibrate', 'pricing.stand_alone',
                             'pricing.allocate'],
    'CalibrationReins': ['pricing.calibrate', 'pricing.stand_alone'],
    'CalibrationReinsGross': ['pricing.calibrate', 'pricing.stand_alone',
                              'pricing.allocate'],
    'CalibrationAggregate': ['pricing.calibrate', 'pricing.stand_alone'],
    'Evaluation': ['pricing.evaluate'],
}


def pricing_results(objects):
    """The pricing result fixtures. Mirrors capture_exhibit_snapshots.py EXACTLY.

    Built from the objects above rather than from new programs: a calibration
    is a calculation over an object that already has an exhibit story, and
    reusing them keeps the file talking about one book. The three calibrations
    are the three shapes ``pricing.stand_alone`` serves, units, views and
    neither; the fourth is the same cession struck on **gross**, which is the
    only basis with a premium to allocate across an occurrence program.
    """
    return {
        'CalibrationPortfolio':
            objects['Portfolio'].calibrate_distortions(0.15, p=0.99),
        'CalibrationReins':
            objects['ReinsAggregate'].calibrate_distortions(0.15, p=0.99),
        'CalibrationReinsGross':
            objects['ReinsAggregate'].calibrate_distortions(
                0.15, p=0.99, reins_view='gross'),
        'CalibrationAggregate':
            objects['Aggregate'].calibrate_distortions(0.15, p=0.99),
        'Evaluation': objects['Aggregate'].evaluate(12.0, p=0.99),
    }


@pytest.fixture(scope='module')
def objects():
    built = {kind: build(program) for kind, program in PROGRAMS.items()}
    built.update(pricing_results(built))
    return built


@pytest.fixture(scope='module')
def dice(objects):
    return objects['Aggregate']


@pytest.fixture(scope='module')
def port(objects):
    return objects['Portfolio']


# --- availability -----------------------------------------------------------

@pytest.mark.parametrize('kind', sorted(EXPECTED_EXHIBITS))
def test_available_exhibits_by_kind(kind, objects):
    got = available_exhibits(objects[kind])
    assert [name for name, _ in got] == EXPECTED_EXHIBITS[kind]
    for _, perspectives in got:
        assert perspectives == [Perspective.RAW, Perspective.INSURER]


# --- the RAW invariant ([Exhibit-Perspective-Contract]) ----------------------

def test_every_raw_block_is_a_public_frame(objects):
    """RAW is exactly the public frame, one block per frame, no inventions.

    The invariant this sweep installs: a RAW block names an attribute on the
    object and serves that frame, in the frame's own orientation, with no
    split and no dropped rows. It is a forcing function as much as a contract,
    because it says what a new exhibit owes: a block with no frame behind it
    means the frame is the thing that is missing, and the exhibit layer is not
    the place to invent one.
    """
    seen = 0
    for obj in objects.values():
        for name, _ in available_exhibits(obj):
            for block, served, _kw in exhibit_frames(obj, name):
                assert hasattr(obj, block), (
                    f'{type(obj).__name__}/{name}: RAW block {block!r} names '
                    'no public frame')
                source = getattr(obj, block)
                relabel = getattr(obj, '_relabel', None)
                if relabel is not None:
                    source = relabel(source)
                pd.testing.assert_frame_equal(served, source)
                seen += 1
    assert seen > 20                         # the sweep actually swept


def test_no_served_block_carries_an_unnamed_index_level(objects):
    """An unnamed index reaches a reader as a column headed ``level_0``.

    Which names nothing, and is what the app's bucket window pane showed
    until a254. The frames it caught then were ``bs_window_df`` on an
    Aggregate, ``tail_behavior_df`` on a Portfolio (whose Aggregate twin was
    named all along) and ``legs_df``.
    """
    for obj in objects.values():
        for name, _ in available_exhibits(obj):
            for perspective in (Perspective.RAW, Perspective.INSURER):
                for block, df, _kw in exhibit_frames(obj, name, perspective):
                    assert all(n is not None for n in df.index.names), (
                        f'{type(obj).__name__}/{name}/{perspective.value}: '
                        f'block {block!r} has an unnamed index level')


# --- the column vocabulary sweep ([Format-Sheet-Enforcement]) ---------------
# The format sheets are a registry of the column vocabulary as much as a
# format table: an entry asserts that a label means one thing across the
# package. This sweep is what makes that bite. It walks every served block
# under both perspectives and reports any **float data column** with no
# declared reading, which is a new word entering the vocabulary unannounced.
#
# Float only, and deliberately: an int, bool, string or date column is typed
# by the IR and reads correctly with no help, while a float is exactly the
# column whose digit count cannot be inferred honestly.

#: Labels the sweep does not ask about, with the reason each is exempt. These
#: are not vocabulary: they are an axis flattened into column headers, so the
#: label is a value and the reading belongs to the row.
VOCABULARY_EXEMPTIONS = (
    (r'^κ\d\d$',
     'kappa scenario columns: a state axis flattened into headers'),
    (r'^P\d\d$',
     'the percentile ladder: a probability axis flattened into headers'),
)

#: Whole blocks the sweep skips, because their columns are an axis and their
#: column axis is **not named**, so the frame cannot say so for itself. A
#: named column axis needs no entry here: the sweep reads the name and draws
#: the same conclusion. Naming these two would retire this table, which is the
#: a254 fix ([BS-Window-Diagnostics] gave four frames honest index names)
#: applied to the other axis, and is worth doing upstream rather than here.
AXIS_BLOCKS = {
    ('stats', 'stats_df'):
        'the canonical moment store: measures run down the rows, and across '
        'are computation views, unit names, or the two axes of a bivariate',
    ('reins', 'reins_stats_df'):
        'the layering store, in the same shape: views and units across',
}

#: Served today with no declared reading and no exemption: the open list this
#: sweep exists to produce, for the author to rule on one label at a time (a
#: sheet entry, an exemption with a reason, or a rename onto a word the sheet
#: already carries). It is a ratchet in both directions. A **new** undeclared
#: label fails the sweep, and a label that stops being served has to come out
#: of this set, so the list cannot rot into a blanket exemption.
#:
#: Reading it as a punch list, the groups are: the moment vocabulary that
#: drifted before the registry existed (``EX`` / ``SD`` / ``Sk`` beside the
#: declared ``CV`` and ``Skew``, and ``mean`` / ``sd`` / ``skew`` / ``cv``
#: again in lower case on the bivariate and tail behavior frames); the
#: composed validation headers (``Est EX``, ``Gross Sk``, ``Change CV``,
#: ``Subject EX``, which are a basis and a measure joined into one label);
#: the waterfall's composed readings (``M / SD``, ``M @ 1-in-100
#: diversified``); and a handful of one-off diagnostics (``Gate``, ``tau``,
#: ``cov``, ``corr``, the bivariate support bounds).
PENDING_VOCABULARY = frozenset({
    'Change CV', 'Change EX', 'D_g_inv', 'Est', 'Gross CV',
    'Gross EX', 'Gross Sk', 'M / SD', 'M / capital diversified',
    'M / capital standalone', 'M @ 1-in-100 diversified',
    'M @ 1-in-100 standalone',
    'Net CV', 'Net EX', 'Net Sk', 'Ref', 'Subject CV',
    'Subject EX', 'Subject Sk', 'closed_form', 'corr', 'cov',
    'cv', 'max', 'mean', 'min', 'sd', 'skew', 'support_max',
    'support_min', 'tau',
})


def _undeclared_float_columns(obj, name, perspective):
    """Float data labels on one exhibit with no reading and no exemption.

    Declared means ``FormatSheet.declares``, so a pattern match counts: a rule
    about a family is a declaration, and a stronger one than the same reading
    written out for each member.
    """
    from aggregate.exhibits import format_sheet
    sheet = format_sheet(perspective)
    out = {}
    for block, df, _kw in exhibit_frames(obj, name, perspective):
        if any(n is not None for n in df.columns.names):
            # the columns are values of that axis (units, views, layers,
            # probe steps), so they are data rather than vocabulary
            continue
        if (name, block) in AXIS_BLOCKS:
            continue
        for i, column in enumerate(df.columns):
            label = column[-1] if isinstance(column, tuple) else column
            if sheet.declares(label, name) \
                    or not pd.api.types.is_float_dtype(df.iloc[:, i]):
                continue
            if any(re.search(pattern, str(label))
                   for pattern, _reason in VOCABULARY_EXEMPTIONS):
                continue
            out.setdefault(label, set()).add(f'{name}/{perspective.value}'
                                             f'/{block}')
    return out


def test_every_served_column_has_a_declared_reading(objects, probed):
    """A new word in the column vocabulary has to be declared, or excused.

    The failure reads as "column ``foo`` is served with no declared reading",
    and the fix is one of three: add a sheet entry, add an exemption with a
    reason, or rename the column onto a word the sheet already carries. The
    third is the one worth wanting, because a vocabulary that grows a synonym
    for every frame is not a vocabulary.
    """
    served = {}
    for obj in [*objects.values(), probed]:
        for name, perspectives in available_exhibits(obj):
            for perspective in perspectives:
                for label, where in _undeclared_float_columns(
                        obj, name, perspective).items():
                    served.setdefault(label, set()).update(where)
    undeclared = {label: sorted(where) for label, where in served.items()
                  if label not in PENDING_VOCABULARY}
    assert not undeclared, (
        'served with no declared reading: '
        + '; '.join(f'{label!r} on {", ".join(where)}'
                    for label, where in sorted(undeclared.items(), key=str))
        + '. Add an entry to src/aggregate/formats/formats-raw.yaml, an '
          'exemption with a reason, or rename onto an existing word.')
    stale = sorted(PENDING_VOCABULARY - set(served), key=str)
    assert not stale, (
        f'no longer served, delete from PENDING_VOCABULARY: {stale}')


def test_the_sweep_actually_sweeps(objects, probed):
    """A sweep that silently stopped looking would pass every assertion."""
    seen = 0
    for obj in [*objects.values(), probed]:
        for name, perspectives in available_exhibits(obj):
            for perspective in perspectives:
                for _block, df, _kw in exhibit_frames(obj, name, perspective):
                    seen += sum(pd.api.types.is_float_dtype(df.iloc[:, i])
                                for i in range(df.shape[1]))
    assert seen > 400, f'only {seen} float columns walked'


def test_every_exemption_pattern_earns_its_place(objects):
    """An exemption nothing matches is a rule with no case, so it comes out."""
    labels = {column[-1] if isinstance(column, tuple) else column
              for obj in objects.values()
              for name, perspectives in available_exhibits(obj)
              for perspective in perspectives
              for _b, df, _kw in exhibit_frames(obj, name, perspective)
              for column in df.columns}
    for pattern, reason in VOCABULARY_EXEMPTIONS:
        assert any(re.search(pattern, str(label)) for label in labels), \
            f'exemption {pattern!r} ({reason}) matches nothing served'
    blocks = {(name, block)
              for obj in objects.values()
              for name, perspectives in available_exhibits(obj)
              for perspective in perspectives
              for block, _df, _kw in exhibit_frames(obj, name, perspective)}
    for key, reason in AXIS_BLOCKS.items():
        assert key in blocks, f'{key} ({reason}) is not served any more'


def test_a_perspective_may_restructure_the_block_list(tower):
    """[Perspective-May-Restructure]: block lists differ between perspectives.

    So ``meta['blocks']`` is a property of the (exhibit, perspective) pair,
    and a client must not assume parity across perspectives. Only INSURER may
    do this: it re-orients, splits, merges and re-captions to say what a frame
    *means*, over RAW's what it *is*.
    """
    raw = [b for b, _, _ in exhibit_frames(tower, 'economic_ratios')]
    insurer = [b for b, _, _ in
               exhibit_frames(tower, 'economic_ratios', Perspective.INSURER)]
    assert raw == ['economic_ratios_df', 'legs_df']
    assert insurer == ['amounts', 'ratios', 'legs']
    assert len(insurer) != len(raw)


def test_available_exhibits_pre_update():
    a = build(AGG_PROGRAM, update=False)
    names = [name for name, _ in available_exhibits(a)]
    assert 'summary' in names
    assert 'tail' not in names  # the ladder needs the realized grid


def test_available_exhibits_unregistered_type():
    assert available_exhibits(object()) == []
    assert available_exhibits(42) == []


def test_registry_shape():
    # every exhibit name maps to (generic_fn, perspectives_fn)
    for name, (fn, perspectives_fn) in EXHIBITS.items():
        assert fn.__name__ == name
        assert callable(perspectives_fn)
        assert isinstance(fn.title, str)


# --- the sharpen exhibit ([Sharpen-Exhibit]) ---------------------------------

@pytest.fixture(scope='module')
def probed():
    """A small object that has been through the grid probe.

    Deliberately not one of the ``PROGRAMS`` fixtures and deliberately not
    snapshotted: the audit carries a ``seconds`` column, so a captured
    document would differ on every run and on every machine.
    """
    import warnings
    a = build('agg EX.Probe 20 claims sev lognorm 100 cv 2 poisson')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')     # the probe visits bad cells
        a.sharpen()
    return a


def test_sharpen_is_absent_until_the_probe_has_run(dice, probed):
    """The audit is what the exhibit shows, so no audit is no exhibit."""
    assert dice.sharpen_df is None
    assert 'sharpen' not in [n for n, _ in available_exhibits(dice)]
    assert 'sharpen' in [n for n, _ in available_exhibits(probed)]
    with pytest.raises(ValueError, match='not available'):
        exhibit_frames(dice, 'sharpen')


def test_sharpen_raw_is_the_audit_frame(probed):
    blocks = exhibit_frames(probed, 'sharpen')
    assert [b for b, _, _ in blocks] == ['sharpen_df']
    pd.testing.assert_frame_equal(blocks[0][1], probed.sharpen_df)


def test_sharpen_insurer_leads_with_the_score_grid(probed):
    """[Sharpen-Grid-Is-A-Reading]: the grid is a reading, not a new frame.

    ``score`` is a column on the audit, so the grid unstacks it rather than
    coming from anywhere else. One block raw, two under insurer, which is the
    smallest exercise of [Perspective-May-Restructure].
    """
    blocks = exhibit_frames(probed, 'sharpen', Perspective.INSURER)
    assert [b for b, _, _ in blocks] == ['score_grid', 'sharpen_df']
    grid = blocks[0][1]
    pd.testing.assert_frame_equal(
        grid, probed.sharpen_df['score'].unstack('d_log2'))
    assert grid.index.name == 'd_bs'
    assert grid.columns.name == 'd_log2'


def test_sharpen_errors_do_not_read_as_zero(probed):
    """At a fixed .4f a good cell and a perfect cell both print 0.0000.

    The readings come from the format sheets since a287, so this asks the
    sheet rather than the block kwargs: RAW takes significant figures, which
    go scientific where they have to. The insurer sheet deliberately takes
    fixed decimals for this table (author, 2026-08-14), which is why the
    assertion is about the raw perspective.
    """
    from aggregate.exhibits import format_sheet
    columns = format_sheet('raw').columns
    assert columns['u_agg_mean'] == '.2e'
    assert columns['u_agg_cv'] == '.5g'
    assert format(1e-7, columns['u_agg_cv']) == '1e-07'


# --- frame stage ------------------------------------------------------------

def test_frames_raw_aggregate(dice):
    blocks = exhibit_frames(dice, 'summary')
    assert len(blocks) == 1
    block_name, df, kw = blocks[0]
    assert block_name == 'summary_df'
    assert list(df.index) == ['Freq', 'Sev', 'Agg']
    # RAW is the untouched frame, described but not interpreted (a226): a
    # caption saying what it is, no row emphasis, no dropped rows, no
    # rearrangement. Column readings come from the format sheets at build
    # time since a287, so a passthrough's kwargs carry the caption alone.
    assert set(kw) == {'caption'}
    assert 'count risk' in kw['caption']


def test_frames_insurer_aggregate(dice):
    block_name, df, kw = exhibit_frames(dice, 'summary', 'insurer')[0]
    assert 'blank by design' in kw['caption']
    assert kw['row_flags'] == {2: ('total',)}


def test_frames_insurer_portfolio_summary_flags(port):
    block_name, df, kw = exhibit_frames(port, 'summary', 'insurer')[0]
    flags = kw['row_flags']
    for i, (unit, x) in enumerate(df.index):
        if unit == 'total':
            assert flags[i] == ('total',)
        elif x == 'Agg':
            assert flags[i] == ('subtotal',)
        else:
            assert i not in flags


def test_frames_insurer_tail_emphasis(dice):
    block_name, df, kw = exhibit_frames(dice, 'tail', 'insurer')[0]
    rows = [i for i, f in kw['row_flags'].items() if 'emphasis' in f]
    emphasized = {int(df['T'].iloc[i]) for i in rows}
    assert emphasized == {200, 250}
    # the ladder is symmetric in P, so each anchor emphasizes both of its rungs
    assert len(rows) == 4


def test_frames_insurer_tail_portfolio_total(port):
    block_name, df, kw = exhibit_frames(port, 'tail', 'insurer')[0]
    flags = kw['row_flags']
    for i, (unit, _) in enumerate(df.index):
        if unit == 'total':
            assert 'total' in flags[i]
        if float(df['T'].iloc[i]) in (200.0, 250.0):
            assert 'emphasis' in flags[i]


def test_relabel_honored(port):
    _, df, _ = exhibit_frames(port, 'summary')[0]
    units = df.index.get_level_values('unit')
    assert 'Unit Alpha' in units
    assert 'EX.UnitA' not in units
    try:
        port.use_labels = False
        _, raw_df, _ = exhibit_frames(port, 'summary')[0]
        assert 'EX.UnitA' in raw_df.index.get_level_values('unit')
    finally:
        port.use_labels = True


# --- stats / validation / dependency ([Exhibits-Stats-Validation]) ----------

def test_stats_insurer_drops_raw_moments(dice, port):
    for obj in (dice, port):
        _, raw_df, raw_kw = exhibit_frames(obj, 'stats')[0]
        _, ins_df, ins_kw = exhibit_frames(obj, 'stats', 'insurer')[0]
        assert len(raw_df) == 26 and set(raw_kw) == {'caption'}
        assert len(ins_df) == 17
        measures = set(ins_df.index.get_level_values('measure'))
        assert measures.isdisjoint({'ex1', 'ex2', 'ex3'})
        # the meta block and the central measures survive
        assert {'mean', 'cv', 'skew'} <= measures
        assert 'dropped' in ins_kw['caption'].lower() \
            or 'raw' in ins_kw['caption'].lower()


def test_stats_identity_for_other_kinds(objects):
    for kind in ('BivariateAggregate', 'Distortion'):
        obj = objects[kind]
        _, raw_df, _ = exhibit_frames(obj, 'stats')[0]
        _, ins_df, _ = exhibit_frames(obj, 'stats', 'insurer')[0]
        pd.testing.assert_frame_equal(raw_df, ins_df)


def test_pnl_stats_is_the_engine_moment_store(objects):
    """[PnL-Economic-Frames]: a P&L's stats_df is its engine's, not the ledger."""
    pn = objects['PnL']
    _, raw_df, _ = exhibit_frames(pn, 'stats')[0]
    assert list(raw_df.index.names) == ['component', 'measure']
    pd.testing.assert_frame_equal(raw_df, pn.engine.stats_df)
    # and it takes the ordinary insurer treatment, the raw-moment drop
    _, ins_df, _ = exhibit_frames(pn, 'stats', 'insurer')[0]
    assert len(ins_df) < len(raw_df)
    assert set(ins_df.index.get_level_values('measure')).isdisjoint(
        {'ex1', 'ex2', 'ex3'})
    # the ledger is a different exhibit and a different document
    assert build_exhibit(pn, 'stats').hash \
        != build_exhibit(pn, 'economic').hash


def test_validation_insurer_pass_frames_no_emphasis(objects):
    # every fixture passes validation, so no emphasis flags anywhere;
    # the portfolio total block still carries its total flags
    _, df, kw = exhibit_frames(objects['Portfolio'], 'validation', 'insurer')[0]
    assert all(flags == ('total',) for flags in kw['row_flags'].values())
    _, df, kw = exhibit_frames(objects['Distortion'], 'validation', 'insurer')[0]
    assert kw['row_flags'] == {}
    _, df, kw = exhibit_frames(objects['BivariateAggregate'],
                               'validation', 'insurer')[0]
    assert kw['row_flags'] == {}


def test_validation_insurer_emphasis_on_failure():
    # a deliberately starved grid fails moment validation; the failing
    # Sev / Agg rows are emphasized
    bad = build('agg EX.Bad 10 claims sev lognorm 100 cv 3 poisson', log2=4)
    assert not bad.valid.passes
    _, df, kw = exhibit_frames(bad, 'validation', 'insurer')[0]
    emphasized = {df.index[i] for i, f in kw['row_flags'].items()
                  if 'emphasis' in f}
    assert emphasized, 'a failing object must emphasize at least one row'
    assert emphasized <= {'Sev', 'Agg'}


def test_check_table_emphasis_helper():
    from aggregate.exhibits._core import _check_table_emphasis
    df = pd.DataFrame({'Est': [1.0, 2.0], 'Pass': [True, False]})
    assert _check_table_emphasis(df) == {1: ('emphasis',)}


def test_dependency_two_blocks(objects):
    blocks = exhibit_frames(objects['BivariateAggregate'], 'dependency')
    assert [name for name, _, _ in blocks] == ['dependency_df',
                                               'axis_support_df']
    for _, df, kw in blocks:
        # no insurer override: raw and insurer agree, both described
        assert set(kw) == {'caption'}
    raw = exhibit_frames(objects['BivariateAggregate'], 'dependency',
                         'insurer')
    pd.testing.assert_frame_equal(raw[0][1], blocks[0][1])


# --- reins ([Exhibits-Reins-Insurer]) ---------------------------------------

def test_reins_raw_is_two_blocks_in_the_frames_own_orientation(objects):
    for kind in ('ReinsAggregate', 'ReinsPortfolio'):
        raw = exhibit_frames(objects[kind], 'reins')
        assert [name for name, _, _ in raw] == ['reins_stats_df',
                                                'reins_summary_df']
        assert set(raw[0][2]) == set(raw[1][2]) == {'caption'}
        # raw keeps every row, the noncentral moments among them
        measures = set(raw[0][1].index.get_level_values('measure'))
        assert {'ex1', 'ex2', 'ex3'} <= measures


def test_reins_portfolio_insurer_drops_the_noncentral_moments(objects):
    """The book's frame has no layer axis, so its insurer view is unchanged."""
    ins = exhibit_frames(objects['ReinsPortfolio'], 'reins', 'insurer')
    assert [name for name, _, _ in ins] == ['reins_stats_df',
                                            'reins_summary_df']
    measures = set(ins[0][1].index.get_level_values('measure'))
    assert measures.isdisjoint({'ex1', 'ex2', 'ex3'})
    assert 'caption' in ins[0][2] and 'caption' in ins[1][2]


def test_reins_aggregate_insurer_turns_the_layering_over(objects):
    """[Reins-Insurer-Orientation]: layers down the rows, in two blocks.

    Two different kinds of thing were reading as one table, so a column
    header changed meaning half way down: what the layer **is**, and what it
    **does** to the three distributions. Gross, ceded and net now read down a
    column, which is the comparison a reinsurance reader makes.
    """
    obj = objects['ReinsAggregate']
    ins = exhibit_frames(obj, 'reins', 'insurer')
    assert [name for name, _, _ in ins] == [
        'reins_layer_terms', 'reins_layer_moments', 'reins_summary_df']
    terms, moments = ins[0][1], ins[1][1]
    # layers down the rows in both, on the same rows
    assert terms.index.names == ['view', 'layer']
    assert list(terms.index) == list(moments.index)
    assert list(terms.columns) == ['share', 'limit', 'attach', 'pr_attach',
                                   'pr_detach', 'pr_loss', 'lol', 'output']
    assert list(moments.columns.get_level_values(0).unique()) == \
        ['cover', 'freq', 'sev', 'agg']
    assert list(moments.columns) == [
        ('cover', 'share'), ('cover', 'limit'), ('cover', 'attach'),
        ('freq', 'mean'),
        ('sev', 'mean'), ('sev', 'cv'), ('sev', 'skew'),
        ('agg', 'mean'), ('agg', 'cv'), ('agg', 'skew')]
    # [Reins-Insurer-Moments-Block]: the cover is deliberately repeated so the
    # block stands alone, and it is the only thing the two blocks share.
    shared = set(terms.columns) & set(moments.columns.get_level_values(1))
    assert shared == {'share', 'limit', 'attach'}
    # frequency keeps its mean only: cv and skew describe the thinning, not
    # the cover, so they stay in the frame and out of this view
    assert ('freq', 'cv') not in moments.columns
    assert ('freq', 'skew') not in moments.columns
    # still no noncentral moments in the insurer reading
    assert 'ex1' not in moments.columns.get_level_values(1)
    assert all('caption' in kw for _, _, kw in ins)


def test_reins_layer_frequency_is_the_thinned_count(objects):
    """[Reins-Insurer-Moments-Block]: the block foots, row by row.

    The layer columns thin the gross count by the probability the subject
    reaches the layer, so layer frequency is ground up frequency times
    P(attach) and freq mean times sev mean is agg mean on every row. That
    identity is what makes the three components readable side by side, and it
    is asserted here so it cannot regress silently: it holds at the frame
    level today and nothing in the view is allowed to break it.
    """
    obj = objects['ReinsAggregate']
    moments = exhibit_frames(obj, 'reins', 'insurer')[1][1]
    for row in moments.index:
        f = moments.loc[row, ('freq', 'mean')]
        s = moments.loc[row, ('sev', 'mean')]
        a = moments.loc[row, ('agg', 'mean')]
        if pd.isna(f) or pd.isna(s):
            continue
        assert f * s == pytest.approx(a, rel=1e-9), row
    # and the layer count really is the thinned gross count
    gross = moments.loc[('occ', 'Gross'), ('freq', 'mean')]
    terms = exhibit_frames(obj, 'reins', 'insurer')[0][1]
    for view, layer in moments.index:
        if not str(layer).startswith('layer'):
            continue
        pr = terms.loc[(view, layer), 'pr_attach']
        assert moments.loc[(view, layer), ('freq', 'mean')] == \
            pytest.approx(gross * pr, rel=1e-9)


def test_reins_aggregate_insurer_is_a_reading_of_the_raw_frame(objects):
    """Both new blocks come out of ``reins_stats_df``, nothing else."""
    obj = objects['ReinsAggregate']
    raw = exhibit_frames(obj, 'reins')[0][1]
    terms, moments = [b[1] for b in
                      exhibit_frames(obj, 'reins', 'insurer')[:2]]
    for view, layer in terms.index:
        for measure in terms.columns:
            got, want = terms.loc[(view, layer), measure], \
                raw.loc[('meta', measure), (view, layer)]
            assert (got == want) or (pd.isna(got) and pd.isna(want))
        for component, measure in moments.columns:
            # the cover group is the meta rows restated under a name the
            # reader needs, which is what a view is allowed to do
            source = 'meta' if component == 'cover' else component
            got = moments.loc[(view, layer), (component, measure)]
            want = raw.loc[(source, measure), (view, layer)]
            assert (got == want) or (pd.isna(got) and pd.isna(want))


def test_reins_portfolio_total_flags(objects):
    _, df, kw = exhibit_frames(objects['ReinsPortfolio'], 'reins',
                               'insurer')[1]
    flags = kw['row_flags']
    for i, key in enumerate(df.index):
        if key[0] == 'total':
            assert flags[i] == ('total',)
        else:
            assert i not in flags
    assert flags  # the total block exists


def test_reins_unavailable_without_cession(dice):
    assert 'reins' not in [n for n, _ in available_exhibits(dice)]
    with pytest.raises(ValueError, match='not available'):
        exhibit_frames(dice, 'reins')


# --- pnl ([Exhibits-PnL-Translation], raw stage) ----------------------------

def test_economic_raw(objects):
    pn = objects['PnL']
    blocks = exhibit_frames(pn, 'economic')
    assert [name for name, _, _ in blocks] == ['economic_df']
    _, df, kw = blocks[0]
    assert set(kw) == {'caption'}
    assert list(df.index.names) in (['Side', 'Label'],
                                    ['Step', 'Side', 'Label'])
    # RAW stays the untouched passthrough; INSURER translates the same frame
    # rather than reshaping it ([Exhibits-Economic-Insurer]), and since
    # [Ledger-Insurer-Abbreviated] it also narrows it. Same rows, same values,
    # a subset of the columns.
    _, ins_df, ins_kw = exhibit_frames(pn, 'economic', 'insurer')[0]
    pd.testing.assert_frame_equal(df[ins_df.columns], ins_df)
    assert len(ins_df.columns) < len(df.columns)
    assert set(ins_kw) == {'caption', 'row_flags'}


def test_economic_ratios_raw(objects):
    pn = objects['PnL']
    blocks = exhibit_frames(pn, 'economic_ratios')
    assert [name for name, _, _ in blocks] == ['economic_ratios_df', 'legs_df']
    ratio_frame = blocks[0][1]
    assert ratio_frame.index.name == 'Step'
    assert {'P', 'L', 'M', 'LR', 'ER', 'CR'} <= set(ratio_frame.columns)
    legs_frame = blocks[1][1]
    assert {'Step', 'Side', 'Label', 'kind', 'EX', 'SD'} \
        <= set(legs_frame.columns)


# --- economic insurer ([Exhibits-Economic-Insurer]) -------------------------

@pytest.fixture(scope='module')
def tower(objects):
    """A walk whose rows share atoms, so the ladder is a kappa ladder."""
    return objects['Tower']


def test_economic_insurer_ledger_flags(tower):
    """Flags follow the ledger plan: one total, results subtotal, nets muted."""
    _, df, kw = exhibit_frames(tower, 'economic', 'insurer')[0]
    flags = kw['row_flags']
    kinds = dict(enumerate(k for _, k, _ in tower._plan))
    totals = [i for i, f in flags.items() if 'total' in f]
    assert len(totals) == 1, 'exactly one bottom line'
    assert kinds[totals[0]] == 'grand_result'
    for i, f in flags.items():
        if 'subtotal' in f:
            assert kinds[i] in ('group_result', 'tier_result', 'grand_total')
        if 'muted' in f:
            assert kinds[i] == 'running_net'
    # legs are never flagged
    assert all(kinds[i] != 'leg' for i in flags)


def test_economic_single_group_has_a_total(objects):
    """A single group ledger's one result IS the bottom line, not a subtotal."""
    _, df, kw = exhibit_frames(objects['PnL'], 'economic', 'insurer')[0]
    assert any('total' in f for f in kw['row_flags'].values())


def test_economic_insurer_caption_states_the_ladder_regime(tower):
    _, df, kw = exhibit_frames(tower, 'economic', 'insurer')[0]
    scenario = any(str(c).startswith('κ') for c in df.columns)
    caption = kw['caption']
    assert scenario, 'this fixture shares atoms, so it should carry kappa'
    assert 'scenario states' in caption and 'foots exactly' in caption
    assert 'adverse state' in caption


# --- the abbreviated insurer ledger ([Ledger-Insurer-Abbreviated]) ----------

def test_economic_insurer_is_abbreviated(tower, peel):
    """Four columns: the two moments, the CV, and the **adverse** tail state.

    The tail rung is the bottom of the ladder, not the top. A P&L is in
    payoff sign convention, left tail bad, so ``κ01`` is the state a reader
    is scanning the sheet for and ``κ99`` is the benign one; an abbreviation
    ending at ``κ99`` would report the good news in the slot the eye reads as
    the bad. ``peel`` is stitched, so it takes the same rung under the plain
    ``P`` header of a marginal ladder.
    """
    _, tower_df, _ = exhibit_frames(tower, 'economic', 'insurer')[0]
    assert list(tower_df.columns) == ['EX', 'SD', 'CV', 'κ01']
    _, peel_df, _ = exhibit_frames(peel, 'economic', 'insurer')[0]
    assert list(peel_df.columns) == ['EX', 'SD', 'CV', 'P01']
    # the adverse state really is the adverse one: the bottom line, the row
    # the ledger plan flags ``total``, loses in it rather than making its
    # mean. Read off the flag rather than off ``iloc[-1]``: the last row of a
    # walk is ``Impact``, a difference between two positions, whose adverse
    # column is the cession paying and so is correctly *positive*.
    _, _, kw = exhibit_frames(tower, 'economic', 'insurer')[0]
    bottom = next(i for i, f in kw['row_flags'].items() if 'total' in f)
    assert tower_df['κ01'].iloc[bottom] < 0 < tower_df['EX'].iloc[bottom]


def test_economic_raw_keeps_the_whole_sheet(tower):
    """RAW is the escape hatch the abbreviation leans on: nothing is lost."""
    from aggregate._pnl import PERCENTILE_LADDER
    _, df, _ = exhibit_frames(tower, 'economic')[0]
    assert list(df.columns)[:4] == ['EX', 'SD', 'CV', 'Skew']
    assert len(df.columns) == 4 + len(PERCENTILE_LADDER)


def test_measure_formats_where_measures_are_columns(dice, tower):
    """CV and Skew take their declared readings on the card and the ledger.

    The declaration is the format sheet since a287, and it reaches the block
    at build time rather than through the frames builder, so the assertion
    is on the built document. Read it as: a measure that *is* a column is
    formatted; the canonical moment store, where measures run down a column,
    still cannot be, which is the asymmetry recorded in plan-exhibits.

    Each case names the measures its block still carries as columns: the card
    has both, and the ledger has ``CV`` alone since the insurer ledger
    dropped ``Skew`` at [Ledger-Insurer-Abbreviated]. Both stay on INSURER,
    whose sheet is the one declaring these fixed decimal readings (raw reads
    ``Skew`` as ``.3g``, significant figures, which is a different claim).
    """
    gt_ = pytest.importorskip('greater_tables')
    expected = {'CV': gt_.FormatSpec(kind='pct', digits=1),
                'Skew': gt_.FormatSpec(kind='dec', digits=3)}
    for obj, name, measures in ((dice, 'summary', ('CV', 'Skew')),
                                (tower, 'economic', ('CV',))):
        _, df, _kw = exhibit_frames(obj, name, 'insurer')[0]
        assert set(measures) <= set(df.columns)
        doc = build_exhibit(obj, name, 'insurer').ir_blocks[0]
        spec = {c.name[-1]: c.format for c in doc.columns}
        for measure in measures:
            assert spec[measure] == expected[measure]


def test_economic_ratios_insurer_splits_units(tower):
    """One unit per column: amounts, ratios, legs, per the reporting rule."""
    blocks = exhibit_frames(tower, 'economic_ratios', 'insurer')
    assert [b for b, _, _ in blocks] == ['amounts', 'ratios', 'legs']
    (_, amounts, amounts_kw), (_, ratios, ratios_kw), _ = blocks
    assert set(amounts.columns) == {'P', 'L', 'E', 'M'}
    assert set(ratios.columns).isdisjoint(amounts.columns)
    # every ratio column points at the `ratio` style in the format sheets,
    # which stamps greater_tables' own ratio tag, so the block declares no
    # ratio_cols of its own ([Format-Sheets], a287)
    assert 'ratio_cols' not in ratios_kw
    doc = build_exhibit(tower, 'economic_ratios', 'insurer').ir_blocks[1]
    assert {c.tag for c in doc.columns if c.role == 'data'} == {'ratio'}
    # M == P - L - E, the identity the caption claims
    import numpy as np
    np.testing.assert_allclose(
        amounts['M'],
        amounts['P'] - amounts['L'] - amounts['E'], atol=1e-9)


# --- economic_waterfall ([Exhibits-Waterfall]) ------------------------------

@pytest.fixture(scope='module')
def peel(objects):
    """A stitched walk: no shared atoms, so no kappa ladder."""
    return objects['Peel']


def test_waterfall_needs_a_tower(objects, tower):
    """A single group P&L has one margin row and no walk to draw."""
    single = objects['PnL']
    assert not single._tower
    assert 'economic_waterfall' not in [n for n, _ in available_exhibits(single)]
    with pytest.raises(ValueError, match='not available'):
        exhibit_frames(single, 'economic_waterfall')
    assert 'economic_waterfall' in [n for n, _ in available_exhibits(tower)]


def test_waterfall_two_blocks_pure_units(tower):
    blocks = exhibit_frames(tower, 'economic_waterfall')
    assert [b for b, _, _ in blocks] == ['walk_df', 'evaluation_df']
    walk, evaluation = blocks[0][1], blocks[1][1]
    assert walk.index.name == 'Step' and evaluation.index.name == 'Step'
    assert list(walk.index) == list(evaluation.index)
    # walk is currency, evaluation is dimensionless: no column in both
    assert set(walk.columns).isdisjoint(evaluation.columns)


def test_waterfall_serves_the_frames_the_pnl_publishes(tower):
    """[Waterfall-Frames-Are-Owed]: the blocks are frames, not inventions.

    Until a253 this exhibit computed its two tables in the exhibit layer and
    no public frame stood behind them, which is the one thing the RAW
    invariant forbids.
    """
    blocks = exhibit_frames(tower, 'economic_waterfall')
    for name, served, _kw in blocks:
        pd.testing.assert_frame_equal(served, getattr(tower, name))


def test_waterfall_diversified_foots_and_standalone_does_not(tower):
    """The thesis of the exhibit, stated as an assertion.

    Conditioning on the whole book's 1-in-100 makes the column a set of
    conditional means, which add; each step's own 1-in-100 is a quantile,
    and quantiles do not add. Showing both side by side is the point.
    """
    _, walk, _ = exhibit_frames(tower, 'economic_waterfall')[0]
    div = walk['M @ 1-in-100 diversified']
    sa = walk['M @ 1-in-100 standalone']
    # the steps before the closing row sum to the closing row, exactly
    assert div.iloc[:-1].sum() == pytest.approx(div.iloc[-1], rel=1e-9)
    assert sa.iloc[:-1].sum() != pytest.approx(sa.iloc[-1], rel=1e-6)
    # the expected margin foots too, by linearity
    assert walk['M'].iloc[:-1].sum() == pytest.approx(walk['M'].iloc[-1],
                                                      rel=1e-9)


def test_waterfall_capital_ratio_definition(tower):
    """``M / -M_100``: margin over the capital that outcome would call for."""
    _, walk, _ = exhibit_frames(tower, 'economic_waterfall')[0]
    _, ev, _ = exhibit_frames(tower, 'economic_waterfall')[1]
    for basis in ('standalone', 'diversified'):
        m100 = walk[f'M @ 1-in-100 {basis}']
        got = ev[f'M / capital {basis}']
        for step in walk.index:
            capital = -m100[step]
            if capital > 0:
                assert got[step] == pytest.approx(walk['M'][step] / capital)
            else:
                # a purchased layer releases capital in the adverse state, so
                # there is no capital to return on and the cell is blank
                assert pd.isna(got[step])


def test_waterfall_blanks_diversified_without_shared_atoms(peel):
    """No shared atoms means no conditioning happened; say so, do not guess."""
    _, walk, kw = exhibit_frames(peel, 'economic_waterfall')[0]
    assert walk['M @ 1-in-100 diversified'].isna().all()
    assert walk['M @ 1-in-100 standalone'].notna().any()
    assert 'blank here' in kw['caption']


def test_waterfall_includes_tier_subtotals(peel):
    """A two layer peel carries a tier subtotal step, and it is picked up."""
    _, walk, _ = exhibit_frames(peel, 'economic_waterfall')[0]
    kinds = {k for _, k, _ in peel._plan}
    assert 'tier_result' in kinds
    assert len(walk) > 3
    assert walk.index[-1] == 'All'


# --- the pricing leaves ([Pricing-Exhibits]) --------------------------------

def test_a_result_object_serves_the_pricing_leaves_and_nothing_else(objects):
    """[Pricing-Keyed-On-Result]: dispatch is on the result, not on the book.

    The corollary matters as much as the ruling: available_exhibits of the
    **built** object is untouched, so a client's capability payload does not
    move because the pricing leaves exist.
    """
    calibration = objects['CalibrationPortfolio']
    assert [n for n, _ in available_exhibits(calibration)] == [
        'pricing.calibrate', 'pricing.stand_alone', 'pricing.allocate']
    assert 'pricing.calibrate' not in [
        n for n, _ in available_exhibits(objects['Portfolio'])]


def test_pricing_calibrate_is_the_receipt_unchanged(objects):
    result = objects['CalibrationPortfolio']
    for perspective in (Perspective.RAW, Perspective.INSURER):
        blocks = exhibit_frames(result, 'pricing.calibrate', perspective)
        assert [b for b, _, _ in blocks] == ['distortion_df']
        # check_index_type: the serve step's relabel turns the ordered
        # categorical distortion axis into plain strings, as it does on every
        # relabeled exhibit. Row order survives, which is what the dtype was
        # carrying.
        pd.testing.assert_frame_equal(blocks[0][1], result.distortion_df,
                                      check_index_type=False, check_categorical=False)


def test_pricing_stand_alone_blocks_follow_the_source_shape(objects):
    """Three sources, three raw block lists, one registered builder."""
    raw = lambda key: [b for b, _, _ in
                       exhibit_frames(objects[key], 'pricing.stand_alone')]
    assert raw('CalibrationReins') == ['reins_price_df']
    assert raw('CalibrationAggregate') == ['calibration_df']
    assert raw('CalibrationPortfolio') == ['calibration_df', 'stand_alone_df']


def test_allocate_needs_parts_to_split_across(objects):
    """The one pricing leaf with a structural gate.

    A book always has units. An aggregate has the halves of an occurrence
    program, but only where one exists and only where the fit was struck on
    gross: a set calibrated on net has no gross premium to allocate. An
    aggregate with no cession has one distribution, which is not a degenerate
    allocation but the absence of one.
    """
    served = lambda key: [n for n, _ in available_exhibits(objects[key])]
    assert 'pricing.allocate' in served('CalibrationPortfolio')
    assert 'pricing.allocate' in served('CalibrationReinsGross')
    assert 'pricing.allocate' not in served('CalibrationAggregate')
    assert 'pricing.allocate' not in served('CalibrationReins')
    for key in ('CalibrationAggregate', 'CalibrationReins'):
        with pytest.raises(ValueError, match='not available'):
            exhibit_frames(objects[key], 'pricing.allocate')


def test_pricing_allocate_insurer_splits_the_book_into_stat_slices(objects):
    """One statistic at a time, units across: the comparison a reader makes."""
    result = objects['CalibrationPortfolio']
    blocks = exhibit_frames(result, 'pricing.allocate', Perspective.INSURER)
    assert [b for b, _, _ in blocks] == [
        'calibration_df', 'stat_LR', 'stat_P', 'stat_PQ', 'stat_ROE']
    for name, frame, kw in blocks[1:]:
        stat = name.removeprefix('stat_')
        pd.testing.assert_frame_equal(
            frame, result.pricing_df.xs(stat, level='stat'),
            check_index_type=False, check_categorical=False)
        assert kw['float_format']


def test_pricing_allocate_serves_a_mass_family_on_an_unbounded_book():
    """[Allocation-Default-Linear]: ccoc reaches every stat slice.

    Built here rather than added to the fixtures on purpose: every book in
    ``PROGRAMS`` is discrete and therefore bounded, and on a bounded book the
    lifted split could always allocate a mass family. The symptom needed an
    unbounded one, which is the ordinary case in the app. Mirrored in section
    EX of ``decl-testers.agg`` as ``EX.UnbPort``.
    """
    port = build('port EX.UnbPort '
                 'agg EX.UnbA 1 claim sev gamma 100 cv 0.5 fixed '
                 'agg EX.UnbB 1 claim sev gamma 50 cv 0.8 fixed')
    assert not port.bounded
    result = port.calibrate_distortions(0.10, p=0.99)
    blocks = exhibit_frames(result, 'pricing.allocate', Perspective.INSURER)
    slices = [(n, f) for n, f, _ in blocks if n.startswith('stat_')]
    assert [n for n, _ in slices] == ['stat_LR', 'stat_P', 'stat_PQ', 'stat_ROE']
    for name, frame in slices:
        assert 'ccoc' in frame.index, name
        assert frame.loc['ccoc'].notna().all(), name


def test_pricing_stand_alone_insurer_is_narrower_than_raw_on_a_cession(objects):
    """The first exhibit where RAW carries strictly more rows than INSURER.

    A ceded price is what the layer is worth to whoever writes it, which is a
    reinsurer's reading; this perspective is the cedent's, and the cedent's
    reading of the same cession is the difference between two of its own
    programs ([Difference-Is-A-Perspective]).
    """
    result = objects['CalibrationReins']
    _n, raw, _kw = exhibit_frames(result, 'pricing.stand_alone')[0]
    _n, insurer, kw = exhibit_frames(
        result, 'pricing.stand_alone', Perspective.INSURER)[0]
    assert 'ceded' in raw.index.get_level_values('view')
    views = set(insurer.index.get_level_values('view'))
    assert not any(v.startswith('ceded') for v in views)
    assert 'net*' in views                      # the calibrated basis, starred
    assert 'gross' in views
    assert 'net less gross' in views            # the difference, appended
    assert 'starred row is the calibrated one' in kw['caption']


def test_the_difference_row_recomputes_its_ratios(objects):
    """A loss ratio of a difference is not the difference of two loss ratios."""
    result = objects['CalibrationReins']
    _n, insurer, _kw = exhibit_frames(
        result, 'pricing.stand_alone', Perspective.INSURER)[0]
    for distortion in insurer.index.get_level_values('distortion').unique():
        block = insurer.xs(distortion, level='distortion')
        if 'net less gross' not in block.index:
            continue
        row = block.loc['net less gross']
        for stat in ('L', 'M', 'P', 'Q'):
            assert row[stat] == pytest.approx(
                block.loc['net*', stat] - block.loc['gross', stat])
        assert row['LR'] == pytest.approx(row['L'] / row['P'])
        assert row['ROE'] == pytest.approx(row['M'] / row['Q'])


def test_pricing_stand_alone_rows_read_view_then_difference(objects):
    """Each family reads as one small table, not as two distant ones."""
    result = objects['CalibrationReins']
    _n, insurer, _kw = exhibit_frames(
        result, 'pricing.stand_alone', Perspective.INSURER)[0]
    first = insurer.index.get_level_values('distortion')[0]
    head = [v for d, v in insurer.index if d == first]
    assert head[-1].endswith('less gross')


def test_stand_alone_on_a_book_sets_the_parts_against_the_whole(objects):
    """The frame the leaf exists for: separate prices, and what they add to."""
    result = objects['CalibrationPortfolio']
    _n, frame, kw = exhibit_frames(result, 'pricing.stand_alone')[1]
    pd.testing.assert_frame_equal(frame, result._relabel(result.stand_alone_df),
                                  check_index_type=False,
                                  check_categorical=False)
    assert 'what pooling is worth' in kw['caption']
    assert 'do not foot' in kw['caption']


def test_the_diversification_benefit_is_the_insurer_reading(objects):
    """Sum and total are facts and ride in RAW; their difference is a reading.

    ``[Difference-Is-A-Perspective]``, one book up from the cession case: the
    sum is arithmetic on measurements and the total is a measurement, so both
    are raw; what the gap between them means belongs to a perspective.
    """
    result = objects['CalibrationPortfolio']
    _n, raw, _kw = exhibit_frames(result, 'pricing.stand_alone')[1]
    _n, insurer, kw = exhibit_frames(
        result, 'pricing.stand_alone', Perspective.INSURER)[1]
    units = set(raw.index.get_level_values('unit'))
    assert 'sum of parts less total' not in units
    assert 'sum of parts less total' in set(
        insurer.index.get_level_values('unit'))
    assert len(insurer) > len(raw)
    assert 'diversification benefit' in kw['caption']


def test_the_benefit_row_recomputes_its_ratios(objects):
    """A loss ratio of a difference is not the difference of two loss ratios.

    ``ccoc`` on this book is a real zero rather than a missing number, and
    worth knowing: a mass at zero family charges the essential supremum, and
    on a bounded book both the supremum and the mean are additive across
    independent units, so its margin is exactly additive and it books no
    diversification benefit at all. The ratios of an all zero row are ``NaN``,
    which is what ``0 / 0`` should say.
    """
    result = objects['CalibrationPortfolio']
    _n, insurer, _kw = exhibit_frames(
        result, 'pricing.stand_alone', Perspective.INSURER)[1]
    seen = 0
    for family in insurer.index.get_level_values('distortion').unique():
        block = insurer.xs(family, level='distortion')
        row = block.loc['sum of parts less total']
        for stat in ('L', 'M', 'P', 'Q'):
            assert row[stat] == pytest.approx(
                block.loc['sum of parts', stat] - block.loc['total', stat])
        # the two rows stand behind the same assets, so the difference has none
        assert row['a'] == pytest.approx(0.0)
        assert row['P'] >= -1e-12                 # concave: pooling never costs
        if row['P'] == 0:
            assert pd.isna(row['PQ'])
            continue
        assert row['LR'] == pytest.approx(row['L'] / row['P'])
        assert row['ROE'] == pytest.approx(row['M'] / row['Q'])
        seen += 1
    assert seen                                   # at least one live benefit


def test_the_benefit_row_reads_beside_its_family(objects):
    """Each family reads as one small table, not as two distant ones."""
    result = objects['CalibrationPortfolio']
    _n, insurer, _kw = exhibit_frames(
        result, 'pricing.stand_alone', Perspective.INSURER)[1]
    first = insurer.index.get_level_values('distortion')[0]
    head = [u for d, u in insurer.index if d == first]
    assert head[-1] == 'sum of parts less total'


def test_the_occurrence_allocation_foots_and_holds_its_gross(objects):
    """One premium decomposed: the rows add up and the gross row is constant."""
    result = objects['CalibrationReinsGross']
    blocks = exhibit_frames(result, 'pricing.allocate')
    assert [b for b, _, _ in blocks] == ['natural_allocation_df']
    _n, frame, kw = blocks[0]
    for family in frame.index.get_level_values('distortion').unique():
        block = frame.xs(family, level='distortion')
        assert block.loc['ceded', 'P'] + block.loc['net', 'P'] == pytest.approx(
            block.loc['gross', 'P'], rel=1e-12)
    assert frame.xs('gross', level='view')['P'].nunique() == 1
    assert 'foot to gross exactly' in kw['caption']
    assert 'one price decomposed' in kw['caption']


def test_the_allocation_caption_carries_the_grid_it_was_priced_on(objects):
    """A priced exhibit should not be readable without its grid."""
    _n, _frame, kw = exhibit_frames(
        objects['CalibrationReinsGross'], 'pricing.allocate')[0]
    assert 'rho_gap' in kw['caption']
    assert 'deficit' in kw['caption']
    assert 'cells' in kw['caption']


def test_raw_equals_insurer_on_the_occurrence_allocation(objects):
    """The allocation is already the cedent's one basis reading.

    Nothing to drop, nothing to star, and no difference rows, because the
    whole table is a decomposition. The ceded row here is the cedent's
    allocated cost of the program rather than a reinsurer's quote for it,
    which is what makes this different from the stand-alone table's ceded row.
    """
    result = objects['CalibrationReinsGross']
    raw = exhibit_frames(result, 'pricing.allocate')
    insurer = exhibit_frames(result, 'pricing.allocate', Perspective.INSURER)
    assert [b for b, _, _ in raw] == [b for b, _, _ in insurer]
    pd.testing.assert_frame_equal(raw[0][1], insurer[0][1])


def test_the_two_leaves_answer_differently_on_one_cession(objects):
    """The reading the pane exists for: net alone is not net's share of gross.

    Stand-alone prices net as its own distribution; allocate gives net its
    share of the one gross premium. Two numbers, two questions, two tabs.

    Not every family separates on every program, and this fixture shows why.
    ``ccoc`` puts its mass on the essential supremum, and on a program whose
    worst gross year is also its worst ceded year (two claims of 30 against a
    10 xs 10 layer) the largest cession and the cession at the largest gross
    outcome are the same number, so the two readings coincide exactly. That is
    a fact about this cession rather than about the two questions.
    """
    result = objects['CalibrationReinsGross']
    _n, alone, _kw = exhibit_frames(result, 'pricing.stand_alone')[0]
    _n, split, _kw = exhibit_frames(result, 'pricing.allocate')[0]
    separated = 0
    for family in split.index.get_level_values('distortion').unique():
        priced = alone.loc[(family, 'net'), 'P']
        allocated = split.loc[(family, 'net'), 'P']
        if priced != pytest.approx(allocated, rel=1e-6):
            separated += 1
    assert separated >= 3
    # and the ceded rows are different animals: one is a reinsurer's quote for
    # the layer, the other the cedent's allocated cost of it
    assert alone.loc[('ph', 'ceded'), 'P'] != pytest.approx(
        split.loc[('ph', 'ceded'), 'P'], rel=1e-6)


def test_pricing_evaluate_serves_the_panel(objects):
    result = objects['Evaluation']
    for perspective in (Perspective.RAW, Perspective.INSURER):
        blocks = exhibit_frames(result, 'pricing.evaluate', perspective)
        assert [b for b, _, _ in blocks] == ['evaluation_df']
        pd.testing.assert_frame_equal(blocks[0][1], result.evaluation_df,
                                      check_index_type=False, check_categorical=False)
    _n, _f, kw = exhibit_frames(
        result, 'pricing.evaluate', Perspective.INSURER)[0]
    # the insurer caption states the anchor, the premium, and what a blank
    # row means, which is the thing readers get wrong
    assert f'{result.a:,.0f}' in kw['caption']
    assert 'cannot lose' in kw['caption']
    assert 'Cherny and Madan' in kw['caption']


def test_a_pricing_exhibit_is_titled_after_its_source(objects):
    """Calibrated distortions: EX.Port, not : CalibrationResult."""
    e = build_exhibit(objects['CalibrationPortfolio'], 'pricing.calibrate')
    assert e.title == 'Calibrated distortions: EX.Port'
    assert e.meta['kind'] == 'CalibrationResult'
    assert e.meta['object'] == 'EX.Port'


def test_a_pricing_exhibit_carries_the_source_labels(objects):
    """The unit axis is relabeled through the book, not left handle keyed."""
    _n, frame, _kw = exhibit_frames(
        objects['CalibrationPortfolio'], 'pricing.allocate',
        Perspective.INSURER)[1]
    assert 'Unit Alpha' in frame.columns


# --- register_simple_exhibit ([Exhibits-Package-Split]) ---------------------

def test_simple_exhibits_are_passthroughs(dice):
    """A manifest-declared exhibit serves one frame, raw and insurer alike.

    Both are caption-only since a287: ``bs_window``'s column readings (the
    window edges, ``W``, ``bs``, ``clipped``) moved into the format sheets
    with every other column's, so a passthrough's kwargs carry prose alone.
    """
    for name, attr, kw_keys in (
            ('bs_window', 'bs_window_df', {'caption'}),
            ('tail_behavior', 'tail_behavior_df', {'caption'})):
        blocks = exhibit_frames(dice, name)
        assert [b for b, _, _ in blocks] == [attr]
        raw_name, raw_df, raw_kw = blocks[0]
        ins_name, ins_df, ins_kw = exhibit_frames(dice, name, 'insurer')[0]
        # no override registered -> INSURER is RAW, by the default rule
        assert raw_kw == ins_kw
        assert set(raw_kw) == kw_keys
        pd.testing.assert_frame_equal(raw_df, ins_df)


def test_every_block_carries_a_caption(objects):
    """No exhibit block ships without prose ([Loss-Lab-Round-3] phase D).

    The gap this closes: a passthrough returned ``{}`` for its frame kwargs
    and captions are lifted from exactly those, so a raw frame arrived with
    nothing said about it, and the client that wanted prose wrote its own.
    That is how one frame's description came to have three possible sources
    that could disagree. Sweeping every (object, exhibit, perspective) keeps
    a new passthrough from reopening it.
    """
    missing = []
    for kind, obj in objects.items():
        for name, perspectives in available_exhibits(obj):
            for perspective in perspectives:
                for block, _, kw in exhibit_frames(obj, name, perspective):
                    if not kw.get('caption'):
                        missing.append(f'{name}/{perspective.value}/{kind}'
                                       f' block {block}')
    assert missing == []


def test_captions_are_per_class_where_the_frame_differs(objects):
    """One frame does not have one description across five classes.

    ``summary_df`` is count risk / severity / total loss on an Aggregate and
    the three ledger rows on a PnL. A caption true of both would say nothing.
    """
    caption_of = lambda kind: exhibit_frames(
        objects[kind], 'summary')[0][2]['caption']
    assert 'count risk' in caption_of('Aggregate')
    assert 'Consideration' in caption_of('PnL')
    assert 'distortion' in caption_of('Distortion')
    assert len({caption_of(k) for k in
                ('Aggregate', 'PnL', 'Distortion', 'BivariateAggregate')}) == 4


def test_insurer_caption_wins_over_the_manifest(dice):
    """RAW says what the frame is; INSURER says what it means, and replaces."""
    raw = exhibit_frames(dice, 'tail')[0][2]['caption']
    insurer = exhibit_frames(dice, 'tail', 'insurer')[0][2]['caption']
    assert raw != insurer
    assert 'anchors are emphasized' in insurer      # the business reading
    assert 'anchors are emphasized' not in raw


def test_register_simple_exhibit_is_open(dice):
    """User code can declare a passthrough exhibit over any frame."""
    from aggregate.exhibits import register_simple_exhibit
    from aggregate._aggregate import Aggregate
    try:
        fn = register_simple_exhibit('sev_density', 'Severity density',
                                     'sev_density_df', [Aggregate])
        assert 'sev_density' in EXHIBITS
        assert fn.title == 'Severity density'
        assert 'sev_density' in [n for n, _ in available_exhibits(dice)]
        name, df, kw = exhibit_frames(dice, 'sev_density')[0]
        assert name == 'sev_density_df' and kw == {}
        assert len(df) > 0
    finally:
        EXHIBITS.pop('sev_density', None)


# --- errors -----------------------------------------------------------------

def test_unknown_exhibit_name(dice):
    with pytest.raises(KeyError, match='unknown exhibit'):
        exhibit_frames(dice, 'nope')


def test_unimplemented_perspectives(dice):
    for p in ('insured', 'reinsurer'):
        with pytest.raises(NotImplementedError, match='not implemented'):
            exhibit_frames(dice, 'summary', p)


def test_unregistered_type_raises(dice):
    with pytest.raises(NotImplementedError, match='int'):
        exhibit_frames(7, 'summary')


def test_unavailable_predicate():
    a = build(AGG_PROGRAM, update=False)
    with pytest.raises(ValueError, match='not available'):
        exhibit_frames(a, 'tail')


def test_perspective_resolution(dice):
    for p in ('raw', 'RAW', Perspective.RAW):
        blocks = exhibit_frames(dice, 'summary', p)
        assert set(blocks[0][2]) == {'caption'}
    with pytest.raises(ValueError, match='unknown perspective'):
        exhibit_frames(dice, 'summary', 'bogus')


# --- IR conversion (greater_tables required) --------------------------------

gt = pytest.importorskip('greater_tables')


def test_build_exhibit_tabledocs(dice):
    e = build_exhibit(dice, 'summary', 'insurer')
    assert isinstance(e, Exhibit)
    assert all(isinstance(doc, gt.TableDoc) for doc in e.ir_blocks)
    assert re.fullmatch(r'[0-9a-f]{12}', e.hash)
    assert e.title == 'Summary: EX.Dice'
    assert e.perspective is Perspective.INSURER
    assert e.meta['kind'] == 'Aggregate'
    assert e.meta['blocks'] == ['summary_df']
    assert 'summary_df' in e.meta['captions']


def test_generic_function_returns_exhibit(dice):
    e = exhibits.summary(dice)
    assert isinstance(e, Exhibit)
    assert e.perspective is Perspective.RAW


def test_payload_shape_and_determinism(dice):
    p1 = build_exhibit(dice, 'tail', 'insurer').to_payload()
    p2 = build_exhibit(dice, 'tail', 'insurer').to_payload()
    assert sorted(p1) == ['blocks', 'hash', 'meta', 'name', 'perspective',
                          'title']
    assert p1['perspective'] == 'insurer'
    assert json.dumps(p1, sort_keys=True) == json.dumps(p2, sort_keys=True)


def test_every_served_block_reconstructs_hash_for_hash(objects):
    """The standing envelope contract, asserted on this side of the wire.

    A served block travels as its canonical_dict and is rebuilt by the
    consumer through gt.TableDoc.model_validate. If the rebuilt document
    hashed differently the ETag would be a lie and every cached table on the
    other end would be serving a document nobody can verify. Swept over every
    (object, exhibit, perspective), which is how the pricing leaves inherit it
    rather than being asserted about separately.
    """
    seen = 0
    for obj in objects.values():
        for name, perspectives in available_exhibits(obj):
            for perspective in perspectives:
                for doc in build_exhibit(obj, name, perspective).ir_blocks:
                    back = gt.TableDoc.model_validate(gt.canonical_dict(doc))
                    assert back.hash == doc.hash, f'{name}/{perspective}'
                    seen += 1
    assert seen > 50


# The committed snapshot file drives the case list, so a freshly captured
# exhibit (a new kind, a new phase) is guarded the moment the capture script
# runs; a case that disappears from the live registry fails its build below.
_SNAPSHOT_KEYS = sorted(json.loads(
    SNAPSHOT_PATH.read_text(encoding='utf-8'))) if SNAPSHOT_PATH.exists() else []


def test_snapshot_coverage(objects):
    """Every (exhibit, perspective, kind) the registry serves is snapshotted."""
    expected = {f'{name}/{p.value}/{kind}'
                for kind, obj in objects.items()
                for name, perspectives in available_exhibits(obj)
                for p in perspectives}
    assert expected == set(_SNAPSHOT_KEYS), \
        'snapshot file out of sync; run capture_exhibit_snapshots.py'


@pytest.mark.parametrize('key', _SNAPSHOT_KEYS)
def test_canonical_snapshot(key, objects):
    """Committed canonical_dict snapshots guard translation and IR drift."""
    snapshots = json.loads(SNAPSHOT_PATH.read_text(encoding='utf-8'))
    exhibit, perspective, obj_key = key.split('/')
    e = build_exhibit(objects[obj_key], exhibit, perspective)
    got = [gt.canonical_dict(doc, include_hash=False) for doc in e.ir_blocks]
    assert got == snapshots[key]
