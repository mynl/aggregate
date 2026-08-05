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
}
AGG_PROGRAM = PROGRAMS['Aggregate']
PORT_PROGRAM = PROGRAMS['Portfolio']

# Exhibit names served per fixture after [Exhibits-Reins-Insurer].
EXPECTED_EXHIBITS = {
    'Aggregate': ['summary', 'tail', 'stats', 'validation'],
    'Portfolio': ['summary', 'tail', 'stats', 'validation'],
    'BivariateAggregate': ['summary', 'stats', 'validation', 'dependency'],
    'PnL': ['summary', 'stats', 'validation', 'pnl_ledger', 'pnl_ratios'],
    'Distortion': ['summary', 'stats', 'validation'],
    'ReinsAggregate': ['summary', 'tail', 'stats', 'validation', 'reins'],
    'ReinsPortfolio': ['summary', 'tail', 'stats', 'validation', 'reins'],
}


@pytest.fixture(scope='module')
def objects():
    return {kind: build(program) for kind, program in PROGRAMS.items()}


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


# --- frame stage ------------------------------------------------------------

def test_frames_raw_aggregate(dice):
    blocks = exhibit_frames(dice, 'summary')
    assert len(blocks) == 1
    block_name, df, kw = blocks[0]
    assert block_name == 'summary_df'
    assert list(df.index) == ['Freq', 'Sev', 'Agg']
    assert kw == {}  # RAW is the untouched passthrough


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
    emphasized = {int(df.index[i]) for i, f in kw['row_flags'].items()
                  if 'emphasis' in f}
    assert emphasized == {200, 250}


def test_frames_insurer_tail_portfolio_total(port):
    block_name, df, kw = exhibit_frames(port, 'tail', 'insurer')[0]
    flags = kw['row_flags']
    for i, (unit, period) in enumerate(df.index):
        if unit == 'total':
            assert 'total' in flags[i]
        if float(period) in (200.0, 250.0):
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
        assert len(raw_df) == 26 and raw_kw == {}
        assert len(ins_df) == 17
        measures = set(ins_df.index.get_level_values('measure'))
        assert measures.isdisjoint({'ex1', 'ex2', 'ex3'})
        # the meta block and the central measures survive
        assert {'mean', 'cv', 'skew'} <= measures
        assert 'dropped' in ins_kw['caption'].lower() \
            or 'raw' in ins_kw['caption'].lower()


def test_stats_identity_for_other_kinds(objects):
    for kind in ('BivariateAggregate', 'PnL', 'Distortion'):
        obj = objects[kind]
        _, raw_df, _ = exhibit_frames(obj, 'stats')[0]
        _, ins_df, _ = exhibit_frames(obj, 'stats', 'insurer')[0]
        pd.testing.assert_frame_equal(raw_df, ins_df)


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
    from aggregate.exhibits import _check_table_emphasis
    df = pd.DataFrame({'Est': [1.0, 2.0], 'Pass': [True, False]})
    assert _check_table_emphasis(df) == {1: ('emphasis',)}


def test_dependency_two_blocks(objects):
    blocks = exhibit_frames(objects['BivariateAggregate'], 'dependency')
    assert [name for name, _, _ in blocks] == ['dependency_df',
                                               'axis_support_df']
    for _, df, kw in blocks:
        assert kw == {}  # no insurer override: raw and insurer agree
    raw = exhibit_frames(objects['BivariateAggregate'], 'dependency',
                         'insurer')
    pd.testing.assert_frame_equal(raw[0][1], blocks[0][1])


# --- reins ([Exhibits-Reins-Insurer]) ---------------------------------------

def test_reins_two_blocks_and_moment_drop(objects):
    for kind in ('ReinsAggregate', 'ReinsPortfolio'):
        obj = objects[kind]
        raw = exhibit_frames(obj, 'reins')
        ins = exhibit_frames(obj, 'reins', 'insurer')
        assert [name for name, _, _ in raw] == ['reins_stats_df',
                                                'reins_summary_df']
        # raw is untouched; insurer drops the raw noncentral moment rows
        assert raw[0][2] == {} and raw[1][2] == {}
        raw_measures = set(raw[0][1].index.get_level_values('measure'))
        ins_measures = set(ins[0][1].index.get_level_values('measure'))
        assert {'ex1', 'ex2', 'ex3'} <= raw_measures
        assert ins_measures.isdisjoint({'ex1', 'ex2', 'ex3'})
        # both insurer blocks are captioned
        assert 'caption' in ins[0][2] and 'caption' in ins[1][2]


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

def test_pnl_ledger_raw(objects):
    pn = objects['PnL']
    blocks = exhibit_frames(pn, 'pnl_ledger')
    assert [name for name, _, _ in blocks] == ['stats_df']
    _, df, kw = blocks[0]
    assert kw == {}
    assert list(df.index.names) in (['Side', 'Label'],
                                    ['Step', 'Side', 'Label'])
    # the insurer framing is author gated: INSURER equals RAW for now
    _, ins_df, ins_kw = exhibit_frames(pn, 'pnl_ledger', 'insurer')[0]
    pd.testing.assert_frame_equal(df, ins_df)
    assert ins_kw == {}


def test_pnl_ratios_raw(objects):
    pn = objects['PnL']
    blocks = exhibit_frames(pn, 'pnl_ratios')
    assert [name for name, _, _ in blocks] == ['ratio_df', 'legs_df']
    ratio_frame = blocks[0][1]
    assert ratio_frame.index.name == 'Step'
    assert {'P', 'L', 'M', 'LR', 'ER', 'CR'} <= set(ratio_frame.columns)
    legs_frame = blocks[1][1]
    assert {'Step', 'Side', 'Label', 'kind', 'EX', 'SD'} \
        <= set(legs_frame.columns)


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
        assert blocks[0][2] == {}
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
