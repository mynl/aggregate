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
from aggregate._bucket_window import _fmt_bs
from aggregate._program import _merge_hints, _merge_note, _round_consideration
from aggregate.constants import ZeroPremiumCessionWarning
from aggregate.decl_writer import format_program


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
# An engine that states its own premium, so the P&L derives rather than sizes.
PREMIUM = 'agg DP.Premium 1000 premium at 0.65 lr sev lognorm 100 cv 2 poisson'
# The same, stated to the cent: a derived premium is never re-rounded.
ODD_PREMIUM = ('agg DP.OddPremium 1000.125 premium at 0.65 lr '
               'sev lognorm 100 cv 2 poisson')
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


# ------------------------------------------------------------- the pin
#
# sharpen() writes its outcome onto the object's own program and trailer, so
# `program` means the program that BUILDS this object rather than merely the one
# that built it, and the three records never disagree.

def test_sharpen_pins_the_grid_onto_the_object():
    a = build(MOVES)
    assert a.hints == ''
    a.sharpen()
    assert a.hints == f'log2={a.log2}; bs={_fmt_bs(a.bs)}'
    b = build(a.program)
    assert (b.bs, b.log2) == (a.bs, a.log2)


def test_sharpen_pin_completes_a_half_declared_grid():
    """The case that made this worth doing.

    A declaration pinning ``log2`` only, sharpened on the bucket axis, used to
    leave ``hints`` reading as a complete record of a grid it half described:
    ``build(a.program)`` came back on a different bucket.
    """
    a = build('agg DP.Half 100 claims sev lognorm 100 cv 2 poisson hints{log2=17}')
    a.sharpen()
    assert float(a.bs) != 1.0                  # the bucket moved
    assert 'bs=' in a.hints and 'log2=17' in a.hints
    assert (build(a.program).bs, build(a.program).log2) == (a.bs, a.log2)


def test_sharpen_pin_keeps_the_three_records_agreeing():
    a = build(BOTH)
    a.sharpen()
    _kind, _name, spec = build.parser.parse(a.program)
    assert spec.get('hints', '') == a.hints
    assert spec.get('note', '') == a.note


def test_sharpen_pin_replaces_its_own_verdict_rather_than_stacking():
    """Probe three times and there is one verdict, the latest."""
    a = build(NOTED)
    for _ in range(3):
        a.sharpen()
    assert a.note == 'a stored note; sharpen: grid confirmed, no change'
    assert a.program.count('note{') == 1


def test_sharpen_pin_drops_a_stale_verdict_when_the_grid_later_moves():
    """A confirmation left beside fresh hints would contradict them."""
    a = build('agg DP.Later 100 claims sev gamma 100 cv 1 poisson note{mine}')
    a.sharpen()
    assert 'grid confirmed' in a.note
    a.sharpen(bs=1 / 8, log2=14)               # a bad centre, so the probe moves
    assert a.note == 'mine'
    assert a.hints


def test_sharpen_pin_records_an_unexecuted_recommendation():
    a = build(MOVES)
    grid = (a.bs, a.log2)
    a.sharpen(execute=False, good_enough=0)
    assert (a.bs, a.log2) == grid
    assert a.hints == ''
    assert a.note.startswith('sharpen: ')


def test_sharpen_pin_is_a_noop_without_a_program():
    """A programmatic object has no text to merge into, and a probe still runs."""
    from aggregate.distributions import Aggregate
    a = Aggregate('DPBare3', exp_en=10, sev_name='lognorm', sev_mean=100,
                  sev_cv=1, freq_name='poisson')
    a.sharpen()
    assert a.program == ''
    assert a.sharpen_df is not None or a.sharpen_description


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

def test_pnl_program_derives_an_engine_premium():
    """The engine states a premium, so the P&L grosses it up and loss_ratio is unused.

    ``derive premium`` since 1.0.0a270: the engine's 1000 is technical, and the
    default 0.25 premium expense clause grosses it to 1000 / 0.75, so premium
    net of expenses returns the technical premium exactly.
    """
    a = build(PREMIUM)
    program = a.pnl_program(loss_ratio=0.4)
    assert 'derive premium' in program
    p = build(program)
    assert p.name == 'DP.Premium_PnL'
    assert _booked_premium(p) == pytest.approx(1000.0 / 0.75)


def test_pnl_program_sizes_the_premium_from_the_loss_ratio():
    """No premium to inherit, so it is expected loss over the stated ratio."""
    a = build(MOVES)
    p = build(a.pnl_program(loss_ratio=0.65, expense_ratio=0))
    assert _booked_premium(p) == pytest.approx(round(a.est_m / 0.65))
    # the sized number is within half a unit of the quotient it came from
    assert abs(_booked_premium(p) - a.est_m / 0.65) <= 0.5


def test_pnl_program_rounds_a_sized_premium_to_whole_units_above_100():
    """A program a reader keeps and edits does not carry sixteen digits."""
    a = build(MOVES)
    program = a.pnl_program(loss_ratio=0.65, expense_ratio=0)
    premium = program.split('premium')[0].split()[-1]
    assert premium == '15385'
    assert '.' not in premium


def test_pnl_program_keeps_two_decimals_at_or_below_100():
    """Below the threshold the cents are the number, not noise."""
    a = build(DISCRETE)
    program = a.pnl_program(loss_ratio=0.65, expense_ratio=0)
    premium = float(program.split('premium')[0].split()[-1])
    assert premium == pytest.approx(round(a.est_m / 0.65, 2))
    assert premium == round(premium, 2)


def test_consideration_rounding_is_idempotent():
    """Rounding an already round number changes nothing.

    The api rounded this downstream while the library did not, so the two had
    to agree on the day it landed here: applying the rule twice is applying it
    once.
    """
    for value in (1428.5840984231345, 100.0, 100.004, 99.999, 0.126, 1e6 + 0.5):
        once = _round_consideration(value)
        assert _round_consideration(once) == once


def test_pnl_program_does_not_round_a_derived_premium():
    """A derived premium starts from the number the program already stated.

    With no expense clause the derivation is the identity, so the booked
    premium is the stated 1000.125 to the cent; with the default 0.25 clause
    it is that number grossed up, still unrounded.
    """
    a = build(ODD_PREMIUM)
    assert 'derive premium' in a.pnl_program(loss_ratio=0.65)
    p = build(a.pnl_program(loss_ratio=0.65, expense_ratio=0))
    assert _booked_premium(p) == pytest.approx(1000.125)
    p = build(a.pnl_program(loss_ratio=0.65))
    assert _booked_premium(p) == pytest.approx(1000.125 / 0.75)


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


def test_pnl_program_on_a_portfolio_writes_the_units_out():
    """A portfolio engine is inlined too, never referenced.

    ``less port.NAME`` is grammatical and shorter, but resolves only against the
    underwriter holding NAME, so the text would build in the session that wrote
    it and nowhere else. A shared server would be writing every user's books
    into one knowledge base to make it resolve.
    """
    p = build(PORT)
    program = p.pnl_program(loss_ratio=0.8, expense_ratio=0)
    assert 'port.' not in program
    assert 'port DP.Port' in program
    assert [u.name for u in p] == ['DP.PortA', 'DP.PortB']
    for unit in p:
        assert unit.name in program
    fresh = Underwriter()                  # never heard of DP.Port
    face = fresh.build(program)
    assert _booked_premium(face) == pytest.approx(round(p.est_m / 0.8))


def test_pnl_program_on_a_portfolio_drops_the_book_trailer():
    """The engine slot has no trailer, and a book's note is not the P&L's."""
    p = build(PORT)
    assert p.note == 'a stored note'
    assert 'a stored note' not in p.pnl_program()


def test_inline_port_engine_round_trips():
    """The new engine form is a first-class parse, not just something we emit."""
    program = ('pnl DP.Inline 1500 premium less port DP.InlineE '
               'agg DP.IA 50 claims sev lognorm 100 cv 2 poisson '
               'agg DP.IB 20 claims sev lognorm 200 cv 1 poisson '
               'less 0.25 premium expenses note{book}')
    face = Underwriter().build(program)
    assert _booked_premium(face) == pytest.approx(1500.0)
    assert face.note == 'book'
    # the canonical render re-parses to the same spec
    _k, _n, one = build.parser.parse(program)
    _k, _n, two = build.parser.parse(
        format_program((_k, _n, one), layout='terse', trailer=True))
    assert one.keys() == two.keys()
    assert two['_engine_port_spec']['name'] == 'DP.InlineE'


def test_port_reference_engine_still_renders_as_a_reference():
    """``port.NAME`` is unchanged: it round-trips as the reference it was.

    Both source forms resolve to one spec now, so the underwriter never looks a
    portfolio up. Only the render differs, and it must not expand a reference
    the author deliberately wrote.
    """
    build(PORT)                            # register it in the default store
    face = build('pnl DP.Ref 1500 premium less port.DP.Port')
    assert 'port.DP.Port' in face.pprogram
    assert 'agg DP.PortA' not in face.pprogram


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


@pytest.mark.parametrize('note, addition, expected', [
    # nothing to replace: a plain append
    ('mine', 'gen: b', 'mine; gen: b'),
    # the previous generated chunk goes, the author's prose stays
    ('mine; gen: a', 'gen: b', 'mine; gen: b'),
    # several accumulated chunks all collapse onto the latest
    ('gen: a; mine; gen: b', 'gen: c', 'mine; gen: c'),
    # an empty addition is how a caller clears its own record
    ('mine; gen: a', '', 'mine'),
])
def test_merge_note_replaces_a_namespaced_record(note, addition, expected):
    """A generated verdict is replaceable; the author's own prose never matches."""
    assert _merge_note(note, addition, replace_prefix='gen: ') == expected


# ------------------------------------- pnl_program: the combined-ratio ladder
#
# [PnL-Reinsurance-Pricing], dev/done/plan-pnl-reinsurance-pricing.md. An unpriced
# cession books at zero and warns; ``net_combined_ratio`` prices the net book
# and every cover separately, adds them, and grosses the total up once, so the
# emitted program is complete enough to build silently. The contract these
# assert is the margin identity: the underwriting result is the *net* result,
# and it does not move when a reinsurance price moves.

# The plan's worked example, section 3. Both tiers, both placed 100 percent.
TOWER = ('agg DP.Tower 10 claims sev lognorm 100 cv 2 '
         'occurrence net of 100 xs 100 poisson '
         'aggregate net of 500 xs 1000')
# The same layer at half and at full placement, for the share invariance.
# Not ``DP.Half``: the sharpen pin test above already declares a different
# program under that name, and both would register into the shared knowledge
# base with the later build winning.
HALF = ('agg DP.HalfPlaced 10 claims sev lognorm 100 cv 2 '
        'occurrence net of 50% po 100 xs 100 poisson')
FULL = HALF.replace('DP.HalfPlaced', 'DP.FullPlaced').replace('50% po ', '')
# Two occurrence layers, for the per-layer sequence form.
TWO = ('agg DP.Two 10 claims sev lognorm 100 cv 2 '
       'occurrence net of 100 xs 100 and 1000 xs 1000 poisson')
# A layer the author already priced, on each of the three bases. ``rate`` is
# the one the ladder refuses: it resolves against the premium being computed.
PRICED_DEPOSIT = TWO.replace('DP.Two', 'DP.Deposit').replace(
    '100 xs 100 and', '100 xs 100 deposit 200 and')
PRICED_ROL = TWO.replace('DP.Two', 'DP.Rol').replace(
    '100 xs 100 and', '100 xs 100 rol 1.5 and')
PRICED_RATE = TWO.replace('DP.Two', 'DP.Rate').replace(
    '100 xs 100 and', '100 xs 100 rate 0.13 and')

_EXPENSE_RATIO = 0.25


def _ceded_premium(face):
    """Total resolved ceded premium, and the per-layer figures behind it.

    Empty rather than absent when the engine carries no cession, so the
    identity below reads the same either way.
    """
    econ = face.engine._pnl_recipe.get('econ') or {}
    return (float(econ.get('ceded', 0.0)),
            list(econ.get('pc_occ_by_layer') or []),
            list(econ.get('pc_agg_by_layer') or []))


def _gross_premium(face):
    """The premium the P&L books, before the ceded premium leaves it.

    :func:`_booked_premium` reads the consideration leg, which is already
    **net** of the ceded premium, so the ladder's ``P`` is that plus the cost
    of the covers.
    """
    ceded, _, _ = _ceded_premium(face)
    return _booked_premium(face) + ceded


def _margin(face):
    """The P&L's expected underwriting result."""
    return float(face.economic_df.loc[('Margin', 'Total'), 'EX'])


def _booked_loss(face):
    """The net loss the P&L actually books, sign flipped to a positive cost.

    Not the same float as the engine's ``est_m``, which is what the ladder
    priced off: the P&L means its own grid distribution over the net density.
    They agree to about 2e-10 relative, which is grid arithmetic and nothing
    to do with the pricing, so the identity is asserted against the number the
    P&L booked and the two are compared separately.
    """
    legs = face.legs_df
    return -float(legs.loc[legs['kind'] == 'loss', 'EX'].iloc[0])


def _deposits(program):
    """Every ceded-premium amount the program states, in order."""
    return [float(line.split('deposit')[1].split()[0])
            for line in program.splitlines() if 'deposit' in line]


def _identity(face):
    """``P x (1 - ER) - ceded - E_net``, what the margin must equal exactly.

    The ladder's whole point: what is left after expenses and the cost of the
    covers is the net book's own result, and nothing else.
    """
    ceded, _, _ = _ceded_premium(face)
    return (_gross_premium(face) * (1 - _EXPENSE_RATIO) - ceded
            - _booked_loss(face))


@pytest.mark.parametrize('program', [MOVES, PREMIUM, CEDED])
def test_pnl_ladder_default_is_inert(program):
    """``net_combined_ratio=None`` is the function exactly as it was.

    The tier arguments are read only through the net one, so passing them
    without it cannot move the text. That is what makes the ladder additive:
    no existing caller's output changes and nothing needs a deprecation.
    """
    a = build(program)
    today = a.pnl_program()
    assert today == a.pnl_program(occ_combined_ratio=0.5,
                                  agg_combined_ratio=0.5)
    assert 'deposit' not in today


def test_pnl_ladder_reproduces_the_worked_example():
    """The plan's section 3, text and numbers, to the digit.

    ``E_net`` 742.198761 over 0.9 is 824.665290; the layers cost 196.444937
    and 90.010738, which round to the deposits written; the total 1110.675290
    grosses to 1480.900387 and books at 1481.
    """
    a = build(TOWER)
    assert a.pnl_program(net_combined_ratio=0.9) == (
        'pnl DP.Tower_PnL\n'
        '  1481 premium\n'
        '  less\n'
        '    agg DP.Tower\n'
        '      10 claims\n'
        '      sev lognorm 100 cv 2\n'
        '      occurrence net of\n'
        '        100 xs 100 deposit 196\n'
        '      poisson\n'
        '      aggregate net of\n'
        '        500 xs 1000 deposit 90.01\n'
        '  less\n'
        '    0.25 premium expenses')


def test_pnl_ladder_rounds_every_deposit_like_a_consideration():
    """A deposit is currency, so it rounds by the one currency rule.

    Whole units above 100, cents at or below it. The two layers land either
    side of that joint, which is why both appear here.
    """
    a = build(TOWER)
    occ, agg = _deposits(a.pnl_program(net_combined_ratio=0.9))
    assert occ == _round_consideration(176.800443 / 0.9) == 196.0
    assert agg == _round_consideration(81.009664 / 0.9) == 90.01


@pytest.mark.parametrize('program, stated', [(TOWER, None), (PREMIUM, 1000.0)])
def test_pnl_ladder_margin_is_the_net_result(program, stated):
    """The identity, exactly, for a sized premium and for a stated one.

    Rounding ``P`` is the only inexactness left, because every deposit is
    rounded before the technical premium is formed, so the residual is not
    merely small: it is the closed form the identity predicts.

    ``stated`` is the engine's own technical premium where it declares one. It
    wins over the convention, so ``net_combined_ratio`` sizes nothing and the
    result is the engine's own risk load.
    """
    a = build(program)
    face = build(a.pnl_program(net_combined_ratio=0.9))
    assert _margin(face) == pytest.approx(_identity(face), rel=1e-12)
    # the loss the P&L books is the loss the ladder priced off, up to the grid
    net_loss = float(face.engine.est_m)
    assert _booked_loss(face) == pytest.approx(net_loss, rel=1e-9)
    # and the result lands where the ideal, unrounded, ladder says it should
    net_technical = net_loss / 0.9 if stated is None else stated
    assert _margin(face) == pytest.approx(
        net_technical - net_loss, abs=0.5 * (1 - _EXPENSE_RATIO))


def test_pnl_ladder_margin_does_not_move_with_a_reinsurance_price():
    """Raising a layer's price raises the booked premium by exactly its cost.

    This is the property the plan's first construction got wrong and the
    reason the premium is built upward. A reader turning the reinsurance knob
    should see the cost pass through, not eat the book's own result.
    """
    a = build(TOWER)
    cheap = build(a.pnl_program(net_combined_ratio=0.9,
                                occ_combined_ratio=0.9))
    dear = build(a.pnl_program(net_combined_ratio=0.9,
                               occ_combined_ratio=0.6))
    cheap_ceded, _, _ = _ceded_premium(cheap)
    dear_ceded, _, _ = _ceded_premium(dear)
    assert dear_ceded > cheap_ceded
    # the whole of the extra cost is passed through, grossed up for expenses
    assert (_gross_premium(dear) - _gross_premium(cheap)) == pytest.approx(
        (dear_ceded - cheap_ceded) / (1 - _EXPENSE_RATIO), abs=1.0)
    # so the underwriting result is untouched, up to the rounding of P
    assert _margin(dear) == pytest.approx(
        _margin(cheap), abs=0.5 * (1 - _EXPENSE_RATIO))


def test_pnl_ladder_prices_each_layer_at_its_own_ratio():
    """A sequence is one ratio per layer, in declaration order.

    The sequence form is the point of the exercise: it is how market rates
    arrive, read off a quote sheet rather than off a convention.
    """
    a = build(TWO)
    stats = a.reins_stats_df
    means = [float(stats.loc[('agg', 'mean'), ('occ', f'layer.{i}')])
             for i in (1, 2)]
    written = _deposits(a.pnl_program(net_combined_ratio=0.9,
                                      occ_combined_ratio=[0.8, 0.6]))
    assert written == [_round_consideration(means[0] / 0.8),
                       _round_consideration(means[1] / 0.6)]
    # and the scalar form is the same thing said once
    assert _deposits(a.pnl_program(net_combined_ratio=0.9,
                                   occ_combined_ratio=0.8))[0] == written[0]


@pytest.mark.parametrize('ratios', [[0.8], [0.8, 0.7, 0.6]])
def test_pnl_ladder_refuses_a_ratio_sequence_of_the_wrong_length(ratios):
    """Recycling or truncating would misprice a layer without saying so."""
    a = build(TWO)
    with pytest.raises(ValueError, match='occurrence combined ratios given'):
        a.pnl_program(net_combined_ratio=0.9, occ_combined_ratio=ratios)


@pytest.mark.parametrize('ratio', [0, -0.5, float('nan')])
def test_pnl_ladder_refuses_an_unusable_net_ratio(ratio):
    """It is the divisor for the net technical premium, so it is checked first.

    An engine with no cession at all reaches that division without passing
    through the per-layer resolution, so the guard cannot live there.
    """
    a = build(MOVES)
    with pytest.raises(ValueError, match='net_combined_ratio'):
        a.pnl_program(net_combined_ratio=ratio)


def test_pnl_ladder_deposit_is_quoted_at_full_placement():
    """A half placed layer writes the same deposit as the whole line.

    The frame's ceded loss is share adjusted and a DecL premium clause is
    quoted at 100 percent placement, so the deposit is the quotient of the two
    and is share invariant. This is the one place an implementation is likely
    to go wrong: writing the placed premium would charge the share twice.
    """
    half, full = build(HALF), build(FULL)
    written = _deposits(half.pnl_program(net_combined_ratio=0.9))
    assert written == _deposits(full.pnl_program(net_combined_ratio=0.9))
    # and it resolves back to exactly the share of what is written
    face = build(half.pnl_program(net_combined_ratio=0.9))
    _, occ_layers, _ = _ceded_premium(face)
    assert occ_layers == [pytest.approx(0.5 * written[0])]
    assert _margin(face) == pytest.approx(_identity(face), rel=1e-12)


@pytest.mark.parametrize('program, resolved', [
    (PRICED_DEPOSIT, 200.0),            # share 1 x deposit 200
    (PRICED_ROL, 150.0),                # share 1 x rol 1.5 x limit 100
])
def test_pnl_ladder_keeps_a_layer_the_author_already_priced(program, resolved):
    """An existing deposit or rol clause survives, and still enters the total.

    The ladder never overwrites terms the author wrote. Both bases resolve
    without reference to the premium being computed, so the cost is known in
    time to be paid for.
    """
    a = build(program)
    face = build(a.pnl_program(net_combined_ratio=0.9))
    _, occ_layers, _ = _ceded_premium(face)
    assert occ_layers[0] == pytest.approx(resolved)
    # the second layer was the ladder's to price, and it did
    assert occ_layers[1] == pytest.approx(
        _round_consideration(
            float(a.reins_stats_df.loc[('agg', 'mean'), ('occ', 'layer.2')])
            / 0.9))
    # the kept layer is paid for, so the identity still holds
    assert _margin(face) == pytest.approx(_identity(face), rel=1e-12)


def test_pnl_ladder_refuses_a_layer_already_priced_as_a_rate():
    """A rate resolves against the premium the ladder is computing.

    The loop does have a closed-form solution, but adopting it would put back
    the self-reference the deposit form was chosen to remove. The message
    names the layer and the two bases that are known in time.
    """
    a = build(PRICED_RATE)
    with pytest.raises(ValueError, match='rate against the P&L premium'):
        a.pnl_program(net_combined_ratio=0.9)
    # without the ladder it is none of the function's business, as before
    assert 'rate 0.13' in a.pnl_program()


def test_pnl_ladder_refuses_a_portfolio_engine():
    """A unit's cession would be quoted against the whole book's premium.

    That reads oddly and wants its own ruling, and the library does not warn
    about an unpriced unit cession today, so nothing is left half done.
    """
    p = build(PORT)
    with pytest.raises(ValueError, match='ladder prices the cessions'):
        p.pnl_program(net_combined_ratio=0.9)
    assert 'deposit' not in p.pnl_program()


def test_pnl_ladder_silences_the_zero_premium_warning():
    """The origin of the whole exercise: a priced program builds quietly.

    An unpriced cession books at zero and says "price the cover to silence
    this". This is what pricing it looks like.
    """
    a = build(TOWER)
    for program, expected in ((a.pnl_program(), 1),
                              (a.pnl_program(net_combined_ratio=0.9), 0)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            build(program)
        assert len([w for w in caught
                    if w.category is ZeroPremiumCessionWarning]) == expected


def test_pnl_ladder_keeps_derive_premium_with_no_cover_to_pay_for():
    """A stated premium and nothing ceded is what the sentinel already resolves to.

    So the program keeps saying ``derive premium`` and the linkage to the
    engine's own exposure clause survives. Ceded premium is a cost the
    sentinel cannot know, so once there is any, the number is written out.
    """
    assert 'derive premium' in build(PREMIUM).pnl_program(
        net_combined_ratio=0.9)
    ceded_engine = build(
        PREMIUM.replace('DP.Premium', 'DP.PremiumCeded')
        .replace('poisson', 'occurrence net of 250 xs 250 poisson'))
    priced = ceded_engine.pnl_program(net_combined_ratio=0.9)
    assert 'derive premium' not in priced
    assert 'deposit' in priced
