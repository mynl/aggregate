"""Tests for [Derived-Programs]: the program that reproduces a derived object.

``program`` says what an object *is*, from the text it was declared with. These
three say what an object you *arrived at* would be declared with: ``sharpen()``
moves the grid, a P&L wraps an engine, a cession changes the loss structure, and
none of the three produced text before now.

Round trip is the whole contract, so that is what these assert: the derived text
parses, builds, and produces the object it claims to. The cases a naive
implementation breaks on get their own tests, namely a program that already
carries ``note{}`` / ``hints{}`` / both (a spec holds one of each, so the clause
is merged, never appended), an engine with no premium going through the P&L
wrap, and the two reinsurance slots, which sit on opposite sides of the
frequency clause.

Programs mirrored in ``src/aggregate/agg/decl-testers.agg`` (DP block), the
derived outputs included.
"""

import warnings

import pytest

from aggregate import build, Underwriter
from aggregate._program import _merge_hints, _merge_note


# A book whose auto-sized grid the probe improves on.
MOVES = 'agg DP.Moves 100 claims sev lognorm 100 cv 2 poisson'
# Thin enough that its grid holds everything and clears the probe gate.
CLEAN = 'agg DP.Clean 100 claims sev gamma 100 cv 1 poisson'
# Noted rides CLEAN, not MOVES: the note-merge case needs the probe to confirm,
# and MOVES clips a sliver of its tail, which the soundness gate never waves
# through however slack the score target is.
NOTED = CLEAN.replace('DP.Clean', 'DP.Noted') + ' note{a stored note}'
HINTED = MOVES.replace('DP.Moves', 'DP.Hinted') + ' hints{bs=1/32; padding=2}'
BOTH = (MOVES.replace('DP.Moves', 'DP.Both')
        + ' note{a stored note} hints{bs=1/32; padding=2}')
# An engine that states its own premium, so the P&L inherits rather than sizes.
PREMIUM = 'agg DP.Premium 1000 premium at 0.65 lr sev lognorm 100 cv 2 poisson'
# Already ceded on both tiers, to check a cession replaces its own tier only.
CEDED = ('agg DP.Ceded 100 claims sev lognorm 100 cv 2 '
         'occurrence net of 250 xs 250 poisson aggregate net of 9000 xs 1000')
DISCRETE = 'agg DP.Discrete dfreq [1 2 3] dsev [10 20 30]'
APPROX = ('agg DP.Approx 100 claims sev lognorm 100 cv 2 poisson '
          'approximate sgamma')
PORT = ('port DP.Port note{a stored note} '
        'agg DP.PortA 50 claims sev lognorm 100 cv 2 poisson '
        'agg DP.PortB 20 claims sev lognorm 200 cv 1 poisson')

OCC = 'occurrence net of 500 xs 500'
AGG = 'aggregate net of 9000 xs 1000'


@pytest.fixture(autouse=True)
def _quiet():
    """Grid probing deliberately visits bad cells; their warnings are expected."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield


def _moved(ob):
    """True when the last probe's winning cell is not the cell it started on."""
    win = ob.sharpen_df[ob.sharpen_df['selected']].iloc[0]
    return float(win['bs']) != float(ob.bs) or int(win['log2']) != int(ob.log2)


def _booked_premium(face):
    """The consideration a built :class:`PnL` actually books."""
    legs = face.legs_df
    return float(legs.loc[legs['kind'] == 'premium', 'EX'].iloc[0])


# ------------------------------------------------------- sharpen_program

def test_sharpen_program_is_empty_before_the_probe():
    """Empty before ``sharpen()`` has run, exactly as ``sharpen_df`` is."""
    a = build(MOVES)
    assert a.sharpen_df is None
    assert a.sharpen_program == ''


def test_sharpen_program_pins_the_moved_grid():
    """The grid moved, so the hints are the record, and they rebuild it."""
    a = build(MOVES)
    grid = (a.bs, a.log2)
    a.sharpen()
    assert (a.bs, a.log2) != grid
    program = a.sharpen_program
    assert 'hints{' in program
    b = build(program)
    assert (b.bs, b.log2) == (a.bs, a.log2)


def test_sharpen_program_notes_a_confirmed_grid():
    """Confirmed means a note and deliberately NO hints.

    Pinning a grid the automatic selector would have picked anyway adds noise to
    a program someone is going to read and share. The note is the record that
    the audit ran, which is what saves running it twice.
    """
    a = build(CLEAN)
    a.sharpen()
    program = a.sharpen_program
    assert 'note{sharpen: grid confirmed, no change}' in program
    assert 'hints{' not in program
    b = build(program)
    assert (b.bs, b.log2) == (a.bs, a.log2)


def test_sharpen_program_records_an_unexecuted_recommendation():
    """``execute=False`` leaves the grid alone, so nothing is pinned."""
    a = build(MOVES)
    grid = (a.bs, a.log2)
    a.sharpen(execute=False, good_enough=0)
    assert (a.bs, a.log2) == grid
    program = a.sharpen_program
    assert 'hints{' not in program
    if _moved(a):
        assert 'probe not executed' in program
    else:
        assert 'grid confirmed' in program


def test_sharpen_program_merges_an_existing_note():
    """One note per spec, so the record joins the note rather than opening a second."""
    a = build(NOTED)
    a.sharpen()                     # grid confirmed, so the outcome is the note
    program = a.sharpen_program
    assert program.count('note{') == 1
    assert 'a stored note' in program
    assert 'grid confirmed' in program
    assert build(program).note == 'a stored note; sharpen: grid confirmed, no change'


def test_sharpen_program_merges_into_existing_hints():
    """One hints per spec: the settings the probe owns are replaced, the rest survive."""
    a = build(HINTED)
    a.sharpen()
    program = a.sharpen_program
    assert program.count('hints{') == 1
    assert 'padding=2' in program
    b = build(program)
    assert (b.bs, b.log2) == (a.bs, a.log2)
    assert b.padding == 2


def test_sharpen_program_keeps_note_and_hints_together():
    """A program carrying both comes back carrying both, one of each."""
    a = build(BOTH)
    a.sharpen()
    program = a.sharpen_program
    assert program.count('note{') == 1 and program.count('hints{') == 1
    b = build(program)
    assert b.note == 'a stored note'
    assert (b.bs, b.log2) == (a.bs, a.log2)


def test_sharpen_program_on_a_portfolio():
    """The clauses land on the ``port`` statement, ahead of the units."""
    p = build(PORT)
    p.sharpen(good_enough=0)
    program = p.sharpen_program
    q = build(program)
    assert (q.bs, q.log2) == (p.bs, p.log2)
    assert q.note.startswith('a stored note')
    assert [u.name for u in q] == [u.name for u in p]


def test_sharpen_program_is_empty_without_a_program():
    """An object built in Python has no text to start from."""
    from aggregate.distributions import Aggregate
    a = Aggregate('DPBare', exp_en=10, sev_name='lognorm', sev_mean=100,
                  sev_cv=1, freq_name='poisson')
    assert a.sharpen_program == ''


# ----------------------------------------------------------- pnl_program

def test_pnl_program_inherits_an_engine_premium():
    """The engine states a premium, so the P&L copies it and loss_ratio is unused."""
    a = build(PREMIUM)
    program = a.pnl_program(loss_ratio=0.4)
    assert 'inherit premium' in program
    p = build(program)
    assert p.name == 'DP.Premium_PnL'
    assert _booked_premium(p) == pytest.approx(1000.0)


def test_pnl_program_sizes_the_premium_from_the_loss_ratio():
    """No premium to inherit, so it is expected loss over the stated ratio."""
    a = build(MOVES)
    p = build(a.pnl_program(loss_ratio=0.65, expense_ratio=0))
    assert _booked_premium(p) == pytest.approx(a.est_m / 0.65)


def test_pnl_program_zero_expense_ratio_omits_the_clause():
    """Zero expense writes no clause rather than a zero one."""
    a = build(MOVES)
    assert 'expenses' not in a.pnl_program(expense_ratio=0)
    assert '0.25 premium expenses' in a.pnl_program()


def test_pnl_program_engine_is_the_object_inlined():
    """The wrapped engine reproduces the aggregate it came from."""
    a = build(CEDED)
    p = build(a.pnl_program(expense_ratio=0))
    assert p.engine.est_m == pytest.approx(a.est_m)
    assert p.engine.occ_reins == a.occ_reins
    assert p.engine.agg_reins == a.agg_reins


def test_pnl_program_moves_the_trailer_to_the_wrapper():
    """The engine has no trailer slot, so the aggregate's metadata rides the pnl."""
    a = build(NOTED)
    program = a.pnl_program()
    assert program.count('note{') == 1
    assert build(program).note == 'a stored note'


def test_pnl_program_carries_the_label_to_the_loss_leg():
    """An aggregate's own ``as`` label names the P&L's loss leg."""
    a = build('agg DP.Labeled as "Motor book" 100 claims '
              'sev lognorm 100 cv 2 poisson')
    assert 'agg DP.Labeled as "Motor book"' in a.pnl_program()


def test_pnl_program_on_a_portfolio_references_it():
    """No inline portfolio engine in the grammar, so the text references the store."""
    p = build(PORT)
    program = p.pnl_program(loss_ratio=0.8, expense_ratio=0)
    assert 'port.DP.Port' in program
    face = build(program)
    assert _booked_premium(face) == pytest.approx(p.est_m / 0.8)


def test_pnl_program_rejects_a_pnl():
    """Wrapping a P&L in a P&L is a mistake, not a feature."""
    face = build('pnl DP.Wrapped 1000 premium less '
                 'agg DP.WrappedE 100 claims sev lognorm 100 cv 2 poisson')
    with pytest.raises(ValueError, match='already a P&L'):
        face.engine.pnl_program()


def test_pnl_program_needs_an_expected_loss():
    """Nothing to size a premium from until the object has been updated."""
    a = build(MOVES, update=False)
    with pytest.raises(ValueError, match='no expected loss'):
        a.pnl_program()


def test_pnl_program_needs_a_program():
    from aggregate.distributions import Aggregate
    a = Aggregate('DPBare2', exp_en=10, sev_name='lognorm', sev_mean=100,
                  sev_cv=1, freq_name='poisson')
    with pytest.raises(ValueError, match='no DecL program'):
        a.pnl_program()


# --------------------------------------------------------- reins_program

def test_reins_program_puts_an_occurrence_cession_before_the_frequency():
    """The slot is the point: occurrence cedes before the convolution."""
    a = build(MOVES)
    program = a.reins_program(OCC)
    assert program.index('occurrence net of') < program.index('poisson')
    b = build(program)
    assert b.occ_reins == [(1.0, 500.0, 500.0)]
    assert b.est_m < a.est_m


def test_reins_program_puts_an_aggregate_cession_after_the_frequency():
    """And aggregate cedes after it, which is why neither can be appended."""
    a = build(MOVES)
    program = a.reins_program(AGG)
    assert program.index('poisson') < program.index('aggregate net of')
    b = build(program)
    assert b.agg_reins == [(1.0, 9000.0, 1000.0)]


def test_reins_program_takes_both_tiers_at_once():
    a = build(MOVES)
    b = build(a.reins_program([OCC, AGG]))
    assert b.occ_reins == [(1.0, 500.0, 500.0)]
    assert b.agg_reins == [(1.0, 9000.0, 1000.0)]


def test_reins_program_replaces_only_its_own_tier():
    """A clause states its tier's whole cession program and leaves the other alone."""
    a = build(CEDED)
    b = build(a.reins_program(OCC))
    assert b.occ_reins == [(1.0, 500.0, 500.0)]
    assert b.agg_reins == a.agg_reins


def test_reins_program_works_on_a_discrete_body():
    """``dfreq`` has its own body rule, and the cession still lands in the slot."""
    a = build(DISCRETE)
    b = build(a.reins_program('occurrence net of 15 xs 5'))
    assert b.occ_reins == [(1.0, 15.0, 5.0)]
    assert b.est_m < a.est_m


def test_reins_program_is_self_contained():
    """It builds in a session that has never heard of the object.

    ``agg NEW agg.OLD occurrence net of ...`` is grammatical and is the obvious
    first idea, but it resolves only against a knowledge base that holds OLD.
    Text that carries its own body builds anywhere.
    """
    a = build(CEDED)
    program = a.reins_program(OCC)
    assert 'agg.' not in program
    fresh = Underwriter()
    b = fresh.build(program)
    assert b.occ_reins == [(1.0, 500.0, 500.0)]


def test_reins_program_rejects_a_clause_with_no_tier():
    a = build(MOVES)
    with pytest.raises(ValueError, match="opens with 'occurrence' or 'aggregate'"):
        a.reins_program('net of 500 xs 500')


def test_reins_program_rejects_two_clauses_on_one_tier():
    a = build(MOVES)
    with pytest.raises(ValueError, match='one clause per tier'):
        a.reins_program([OCC, 'occurrence net of 1 xs 1'])


def test_reins_program_rejects_an_empty_cession():
    a = build(MOVES)
    with pytest.raises(ValueError, match='no cession given'):
        a.reins_program([])


def test_reins_program_reports_an_unparseable_cession():
    a = build(MOVES)
    with pytest.raises(ValueError, match='could not parse the cession'):
        a.reins_program('occurrence net of oops')


def test_reins_program_rejects_occurrence_under_approximate():
    """The method-of-moments fit bypasses the per-occurrence convolution."""
    a = build(APPROX)
    with pytest.raises(ValueError, match='incompatible with occurrence'):
        a.reins_program(OCC)
    # the aggregate tier is always fine
    assert build(a.reins_program(AGG)).agg_reins == [(1.0, 9000.0, 1000.0)]


def test_reins_program_rejects_a_portfolio_program():
    """A portfolio has no cession clause; its units do."""
    p = build(PORT)
    assert not hasattr(p, 'reins_program')


# -------------------------------------------------------- trailer merging

@pytest.mark.parametrize('hints, updates, expected', [
    ('', {'log2': '18'}, 'log2=18'),
    ('bs=1/32', {'bs': '1/64'}, 'bs=1/64'),
    ('bs=1/32; padding=2', {'log2': '18', 'bs': '1/64'},
     'bs=1/64; padding=2; log2=18'),
    ('padding=2', {'log2': '18', 'bs': '8'}, 'padding=2; log2=18; bs=8'),
    # an unrecognized chunk is the author's and is carried through verbatim
    ('nonsense', {'log2': '18'}, 'nonsense; log2=18'),
    # a repeated key collapses onto the replacement
    ('bs=1/32; bs=1/16', {'bs': '2'}, 'bs=2'),
])
def test_merge_hints_replaces_in_place_and_appends_the_rest(hints, updates,
                                                            expected):
    assert _merge_hints(hints, updates) == expected


@pytest.mark.parametrize('note, addition, expected', [
    ('', 'added', 'added'),
    ('kept', '', 'kept'),
    ('kept', 'added', 'kept; added'),
])
def test_merge_note_joins_rather_than_duplicating(note, addition, expected):
    assert _merge_note(note, addition) == expected
