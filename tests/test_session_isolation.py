"""[Session-Isolation]: one recipe base, many users (1.0.0a302).

The library half of ``dev/plan-session-isolation.md``: :meth:`Underwriter.fork`
(phase L1), :meth:`Underwriter.preview` (L2) and
:class:`~aggregate.underwriter.RecipeNotFound` (L3).

A fork is a copy of an underwriter whose recipe base is a fresh dict over the
same parsed :class:`~aggregate.recipe.Recipe` objects. It costs microseconds
against the seconds a fresh load costs, and declarations built in it are
invisible everywhere else. The preview reports what a program would resolve to
and where each referent came from, which is what lets a caller tell a program
that means the same thing to everyone from one that leans on something private.

The tests here are the library's own contract. The application's cache rule,
its session registry and its route order live in ``aggregate_api``.
"""

import logging
import threading
import time

import pytest

from aggregate import build as global_build
from aggregate.underwriter import (Underwriter, ProgramPreview,
                                   RecipeNotFound, ResolvedReference)

logging.disable(logging.CRITICAL)

HINTS = 'hints{log2=16; bs=1/32}'

#: A one-line declaration every fork test can make without paying for an FFT.
CHEAP = 'dfreq [1] dsev [1]'


@pytest.fixture
def base():
    """A loaded underwriter holding one file-sourced entry.

    ``source`` is a file name rather than the ``'session'`` sentinel, which is
    how a shipped library entry reads and what the clobber test turns on.
    """
    uw = Underwriter(name='SI.base')
    uw.load()      # nothing configured: this just flips the loaded flag
    uw.add_recipe('agg', 'SI.Lib', {'name': 'SI.Lib'}, 'agg SI.Lib ...',
                  source='si-library.agg')
    return uw


# ---------------------------------------------------------------------------
# fork(): the copy itself
# ---------------------------------------------------------------------------

def test_fork_shares_the_parsed_entries():
    f = global_build.fork()
    assert len(f._recipes) == len(global_build._recipes)
    # the same Recipe objects, not copies: that is what makes it cheap
    key = next(iter(global_build._recipes))
    assert f._recipes[key] is global_build._recipes[key]


def test_fork_costs_microseconds():
    """The whole design rests on this. Measured ~7 us; pinned two orders up."""
    global_build.load()
    t0 = time.perf_counter()
    for _ in range(100):
        global_build.fork()
    per_fork = (time.perf_counter() - t0) / 100
    assert per_fork < 1e-3, f'{per_fork * 1e6:.0f} us per fork'


def test_fork_isolates_in_both_directions():
    a = global_build.fork('SI.a')
    b = global_build.fork('SI.b')
    a.build(f'agg SI.OnlyMine {CHEAP}', update=False)
    assert ('agg', 'SI.OnlyMine') in a._recipes
    assert ('agg', 'SI.OnlyMine') not in b._recipes
    assert ('agg', 'SI.OnlyMine') not in global_build._recipes
    b.build(f'agg SI.OnlyYours {CHEAP}', update=False)
    assert ('agg', 'SI.OnlyYours') not in a._recipes


def test_fork_contains_a_library_clobber(base):
    """The bug this contains: one session overwriting a shipped name."""
    f = base.fork('SI.clobber')
    f.add_recipe('agg', 'SI.Lib', {'name': 'SI.Lib', 'mine': True},
                 'agg SI.Lib mine')
    assert f['SI.Lib'].source == 'session'
    assert base['SI.Lib'].source == 'si-library.agg'
    assert 'mine' not in base['SI.Lib'].spec


def test_fork_resets_the_parser_to_its_own_base():
    """The substance of fork(): a shared parser reads the wrong recipe base."""
    f = global_build.fork('SI.parser')
    assert f._parser is None and f._lexer is None
    assert f.parser is not global_build.parser
    # the callback the parser holds is bound to the fork, not to the parent
    assert f.parser.safe_lookup.__self__ is f


def test_fork_parses_against_its_own_declarations():
    """The split-brain symptom, end to end: the parse must see the fork."""
    f = global_build.fork('SI.split')
    f.build(f'agg SI.Inner {CHEAP} {HINTS}', update=False)
    # resolves in the fork ...
    f.build(f'agg SI.Outer dfreq [1] sev agg.SI.Inner {HINTS}', update=False)
    # ... and not in the parent
    with pytest.raises((LookupError, ValueError)):
        global_build.build('agg SI.Outer2 dfreq [1] sev agg.SI.Inner',
                           update=False)


def test_fork_resets_the_cycle_guard_and_copies_the_lists():
    f = global_build.fork()
    assert f._sev_ref_stack == []
    assert f._sev_ref_stack is not global_build._sev_ref_stack
    assert f.databases == global_build.databases
    assert f.databases is not global_build.databases


def test_fork_loads_the_parent_first():
    """A fork of an unloaded underwriter would re-read the databases per fork."""
    uw = Underwriter(name='SI.lazy', databases=None)
    assert uw._loaded is False
    f = uw.fork()
    assert uw._loaded is True and f._loaded is True


def test_fork_names_itself():
    assert global_build.fork().name == f'{global_build.name}-fork'
    assert global_build.fork('SI.named').name == 'SI.named'


def test_fork_is_a_snapshot_not_a_subscription():
    f = global_build.fork('SI.snap')
    global_build.add_recipe('agg', 'SI.Later', {}, 'agg SI.Later')
    try:
        assert ('agg', 'SI.Later') not in f._recipes
    finally:
        del global_build._recipes[('agg', 'SI.Later')]


def test_interpret_file_is_written_on_fork():
    """The hand-rolled scratch copy interpret_file used to carry."""
    before = set(global_build._recipes)
    df = global_build.interpret_file()
    assert df.error.sum() == 0
    # validating a file must not pollute the base that validated it
    assert set(global_build._recipes) == before


# ---------------------------------------------------------------------------
# preview(): what a program resolves to
# ---------------------------------------------------------------------------

def test_preview_of_a_self_contained_program_resolves_nothing():
    pv = global_build.fork().preview(f'agg SI.Free {CHEAP}')
    assert isinstance(pv, ProgramPreview)
    assert pv.route == 'program'
    assert pv.resolved == ()
    assert [s.name for s in pv.statements] == ['SI.Free']


def test_preview_reports_an_inlined_reference_with_its_source():
    f = global_build.fork('SI.inline')
    pv = f.preview('agg SI.X 10 claims sev sev.ParetoSev poisson')
    assert [(r.kind, r.name) for r in pv.resolved] == [('sev', 'ParetoSev')]
    assert pv.resolved[0].source == global_build['ParetoSev'].source


def test_preview_reports_a_session_source_only_in_the_fork_that_made_it():
    """The cache rule in one test: one text, two forks, two answers."""
    program = 'agg SI.Y 10 claims sev sev.ParetoSev poisson'
    mine, yours = global_build.fork('SI.mine'), global_build.fork('SI.yours')
    mine.build('sev ParetoSev 100 * pareto 2.5', update=False)
    assert [r.source for r in mine.preview(program).resolved] == ['session']
    assert [r.source for r in yours.preview(program).resolved] != ['session']


def test_preview_follows_the_deferred_chain_to_the_bottom():
    """Depth two: the program names B, but it stands on A as well."""
    f = global_build.fork('SI.chain')
    f.build(f'agg SI.A {CHEAP} {HINTS}', update=False)
    f.build(f'agg SI.B dfreq [1] sev agg.SI.A {HINTS}', update=False)
    pv = f.preview('agg SI.C dfreq [1] sev agg.SI.B')
    assert [r.name for r in pv.resolved] == ['SI.B', 'SI.A']
    assert {r.source for r in pv.resolved} == {'session'}


def test_preview_of_a_bare_name_takes_the_name_route():
    f = global_build.fork('SI.bare')
    pv = f.preview('ParetoSev')
    assert pv.route == 'name'
    assert [s.name for s in pv.statements] == ['ParetoSev']
    # a bare name IS a reference
    assert [(r.kind, r.name) for r in pv.resolved] == [('sev', 'ParetoSev')]


def test_preview_agrees_with_build_on_what_a_one_word_program_means():
    f = global_build.fork('SI.agree')
    f.build(f'agg SI.Named {CHEAP}', update=False)
    assert f.preview('SI.Named').route == 'name'
    # not a name, so the same text is parsed by both
    assert f.preview(f'agg SI.NotAName {CHEAP}').route == 'program'


def test_preview_registers_nothing():
    f = global_build.fork('SI.clean')
    before = set(f._recipes)
    f.preview(f'agg SI.Ghost {CHEAP}')
    assert set(f._recipes) == before


def test_preview_resolves_a_programs_own_declarations_without_reporting_them():
    """A name the program brings is not a dependency of the program."""
    f = global_build.fork('SI.multi')
    pv = f.preview(f'agg SI.P {CHEAP} {HINTS}\n\n'
                   f'agg SI.Q dfreq [1] sev agg.SI.P')
    assert [s.name for s in pv.statements] == ['SI.P', 'SI.Q']
    assert pv.resolved == ()


def test_preview_keeps_an_expr_statement_and_resolves_nothing():
    pv = global_build.fork().preview('(1 + 2)')
    assert [s.kind for s in pv.statements] == ['expr']
    assert pv.resolved == ()


def test_preview_raises_the_ordinary_parse_error():
    with pytest.raises(ValueError) as e:
        global_build.fork().preview('agg SI.Bad this is not decl')
    assert getattr(e.value, 'report', None) is not None


def test_preview_raises_recipe_not_found_for_an_unknown_reference():
    with pytest.raises(RecipeNotFound) as e:
        global_build.fork().preview('agg SI.M 1 claim sev sev.SI.NoSuch fixed')
    assert e.value.kind == 'sev' and e.value.name == 'SI.NoSuch'


def test_preview_holds_no_instance_state_under_threads():
    """The host runs previews outside its one build slot, so this must hold."""
    f = global_build.fork('SI.threads')
    programs = [f'agg SI.T{i} {i + 1} claims sev sev.ParetoSev poisson'
                for i in range(8)]
    serial = [f.preview(p) for p in programs]
    results = [None] * len(programs)

    def work(i):
        results[i] = f.preview(programs[i])

    threads = [threading.Thread(target=work, args=(i,))
               for i in range(len(programs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert [r.resolved for r in results] == [s.resolved for s in serial]
    assert ([[st.name for st in r.statements] for r in results]
            == [[st.name for st in s.statements] for s in serial])
    assert f._sev_ref_stack == []


def test_resolved_reference_is_a_readable_triple():
    r = ResolvedReference('agg', 'X', 'session')
    assert (r.kind, r.name, r.source) == ('agg', 'X', 'session')
    assert tuple(r) == ('agg', 'X', 'session')


# ---------------------------------------------------------------------------
# RecipeNotFound
# ---------------------------------------------------------------------------

def test_recipe_not_found_is_a_key_error():
    uw = Underwriter(name='SI.rnf')
    with pytest.raises(RecipeNotFound):
        uw['SI.NoSuchThing']
    with pytest.raises(KeyError):
        uw['SI.NoSuchThing']
    with pytest.raises(LookupError):
        uw['SI.NoSuchThing']


def test_recipe_not_found_carries_kind_and_name():
    uw = Underwriter(name='SI.rnf2')
    with pytest.raises(RecipeNotFound) as e:
        uw.recipe('SI.Nope', kind='agg')
    assert e.value.kind == 'agg' and e.value.name == 'SI.Nope'
    with pytest.raises(RecipeNotFound) as e:
        uw.recipe('SI.Nope')
    assert e.value.kind is None and e.value.name == 'SI.Nope'


def test_recipe_not_found_renders_its_message_plainly():
    """KeyError.__str__ is repr(args[0]), which quotes a whole diagnosis."""
    uw = Underwriter(name='SI.rnf3')
    with pytest.raises(RecipeNotFound) as e:
        uw['SI.Nope']
    assert str(e.value) == "no recipe named 'SI.Nope'"


def test_an_ambiguous_name_stays_a_plain_key_error():
    """The entry is there; the question was not answerable."""
    uw = Underwriter(name='SI.ambig')
    uw.add_recipe('agg', 'SI.Twin', {}, 'agg SI.Twin')
    uw.add_recipe('sev', 'SI.Twin', {}, 'sev SI.Twin')
    with pytest.raises(KeyError, match='ambiguous') as e:
        uw['SI.Twin']
    assert not isinstance(e.value, RecipeNotFound)


def test_recipe_not_found_from_a_vanished_deferred_referent():
    """It parsed, so it was there when the program was read. The expiry path."""
    uw = Underwriter(name='SI.vanish')
    uw.build(f'agg SI.V.Inner {CHEAP} {HINTS}', update=False)
    uw._interpret_program(
        f'agg SI.V.Outer dfreq [1] sev agg.SI.V.Inner {HINTS}')
    del uw._recipes[('agg', 'SI.V.Inner')]
    with pytest.raises(RecipeNotFound) as e:
        uw.build('SI.V.Outer', update=False)
    assert e.value.kind == 'agg' and e.value.name == 'SI.V.Inner'
    assert 'removed or renamed since' in str(e.value)


def test_the_reference_cycle_guard_is_unchanged():
    """A cycle is a bad program, not a missing name: still a ValueError."""
    uw = Underwriter(name='SI.cycle')
    uw.build(f'agg SI.CY.A {CHEAP} {HINTS}', update=False)
    uw.build(f'agg SI.CY.B dfreq [1] sev agg.SI.CY.A {HINTS}', update=False)
    with pytest.raises(ValueError, match='severity reference cycle') as e:
        uw.build(f'agg SI.CY.A dfreq [1] sev agg.SI.CY.B {HINTS}')
    assert not isinstance(e.value, RecipeNotFound)
