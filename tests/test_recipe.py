"""Tests for :class:`aggregate.recipe.Recipe`, the library entry record.

A recipe is one DecL entry: identity, parsed spec, source program, provenance,
and the built object once the factory has run. Its description surface --
``note`` / ``tags`` / ``hints`` / ``decl`` -- is **derived from** the spec
rather than stored beside it, so there is exactly one copy of every fact, and
these tests mostly pin that.

The class carried a second half until 1.0.0a301: a ``doc{{{...}}}`` body parsed
into Problem / Solution / Discussion / Check, executed by a pytest harness and
rendered to a Quarto cookbook. All of it went with the clause
(``dev/done/plan-decommission-docs.md``). The invariants those docs asserted are
now ordinary asserts in ``tests/test_library_entries.py``.

See ``aggregate.recipe`` and ``dev/plan-meta-data.md`` [Recipe-Library].
"""
import copy
import dataclasses

import pytest

from aggregate import Underwriter
from aggregate.recipe import Recipe


@pytest.fixture(scope='module')
def uw():
    """An Underwriter over decl-testers.agg, which carries the AF fixtures."""
    u = Underwriter(databases='decl-testers')
    u.load()
    return u


@pytest.fixture(scope='module')
def lib():
    """A private Underwriter over ``library.agg``.

    Deliberately NOT the global ``build`` singleton: it is mutable shared
    state, and a test that reads it can be perturbed by any other test in the
    session that loads a database into it. Own your recipe base.
    """
    u = Underwriter(databases='library')
    u.load()
    return u


# ----------------------------------------------------------------------
# The record
# ----------------------------------------------------------------------
def test_underwriter_recipe_resolves_by_name_alone(uw):
    r = uw.recipe('AF.Tags.Spaces')
    assert isinstance(r, Recipe)
    assert r.kind == 'agg'
    assert r.name == 'AF.Tags.Spaces'
    assert r.tags == ('severity', 'frequency', 'aggregate')
    # `program` is the source line, verbatim; `decl` the canonical
    # re-rendering. Both name the entry.
    assert r.program.startswith('agg AF.Tags.Spaces')
    assert r.decl.startswith('agg AF.Tags.Spaces')


def test_a_recipe_is_the_entry_not_a_second_view_of_it(uw):
    """``recipe(name)`` and ``uw[name]`` return the same thing.

    Before 1.0.0a164 these were different classes -- a ``ParsedProgram``
    holding the entry and a ``Recipe`` holding its parsed doc -- so "which do
    I use?" had no good answer. One class now carries the entry, and since
    1.0.0a301 there is no second half for it to carry.
    """
    by_verb = uw.recipe('AF.Tags.Spaces')
    by_subscript = uw[('agg', 'AF.Tags.Spaces')]
    assert type(by_verb) is type(by_subscript) is Recipe
    assert by_verb.spec == by_subscript.spec
    assert by_verb.program == by_subscript.program
    assert by_verb.source == by_subscript.source


def test_lookup_hands_back_a_copy_so_the_store_cannot_be_mutated(uw):
    """``.object`` is set by the factory on the caller's copy, never the store."""
    r = uw.recipe('AF.Tags.Spaces')
    r.object = 'not really an Aggregate'
    assert uw.recipe('AF.Tags.Spaces').object is None
    assert uw[('agg', 'AF.Tags.Spaces')].object is None


def test_underwriter_recipe_unknown_name_raises(uw):
    with pytest.raises(KeyError, match='no recipe named'):
        uw.recipe('DefinitelyNotAnEntry')


def test_repr_is_cheap_and_names_the_entry(uw):
    r = uw.recipe('AF.Tags.Spaces')
    assert repr(r) == "Recipe('agg', 'AF.Tags.Spaces')"
    r.object = object()
    assert 'object=object' in repr(r)


# ----------------------------------------------------------------------
# The trailer, derived from the spec
# ----------------------------------------------------------------------
def test_metadata_is_derived_from_the_spec_not_duplicated(uw):
    """One copy of every fact: note/tags/hints read straight off ``spec``."""
    r = uw.recipe('AF.Tags.Spaces')
    assert r.note == r.spec['note']
    assert r.tags == tuple(r.spec['tags'])
    assert r.hints == r.spec.get('hints', '')


def test_an_entry_without_a_trailer_answers_with_empties():
    """Absence is a fact to report, not an exception."""
    r = Recipe(kind='agg', name='Bare')
    assert r.note == ''
    assert r.tags == ()
    assert r.hints == ''


def test_the_trailer_properties_tolerate_a_missing_spec():
    """A recipe built with no spec at all must not raise on description."""
    r = Recipe()
    assert (r.note, r.tags, r.hints, r.decl) == ('', (), '', '')


# ----------------------------------------------------------------------
# decl, the canonical re-rendering
# ----------------------------------------------------------------------
def test_decl_carries_hints_and_nothing_else():
    """``hints`` is part of the program; the rest of the trailer is about it."""
    uw = Underwriter()
    uw._interpret_program(
        'agg HintedRecipe 5 claims sev lognorm 10 cv 2 poisson '
        'note{n} tags{topic:aggregate} hints{log2=12}')
    r = uw.recipe('HintedRecipe')
    assert 'hints{log2=12}' in r.decl
    assert 'note{' not in r.decl and 'tags{' not in r.decl


def test_decl_rebuilds_the_entry(lib):
    """The rendered declaration is real DecL that reproduces the object."""
    from aggregate import build
    r = lib.recipe('DiceThreeEvenDice')
    rebuilt = build(r.decl)
    assert abs(rebuilt.actual_m - build('DiceThreeEvenDice').actual_m) < 1e-12


def test_decl_is_empty_when_the_entry_cannot_be_unparsed(lib):
    """A construct the writer cannot invert gets no rendering, not an error.

    ``MinimumDistortion`` references its two children by name and the spec does
    not retain those names, so there is nothing to render from. Failing here
    would make the whole library fail to load.
    """
    assert lib.recipe('MinimumDistortion').decl == ''


def test_decl_is_cached_and_the_cache_is_dropped_by_replace(uw):
    """``dataclasses.replace`` must re-derive: a replaced spec is a new entry."""
    r = uw.recipe('AF.Tags.Spaces')
    assert r._decl is None                 # untouched by construction
    first = r.decl
    assert r._decl is not None             # ... and cached after the first ask
    assert r.decl is first                 # same object, not re-rendered

    other = dataclasses.replace(r, name='Renamed')
    assert other._decl is None
    assert other.decl.startswith('agg Renamed')


# ----------------------------------------------------------------------
# The recipes frame
# ----------------------------------------------------------------------
def test_recipes_frame_shape(uw):
    df = uw.recipes
    assert df.index.names == ['kind', 'name']
    row = df.loc[('agg', 'AF.Tags.Spaces')]
    assert row['note']
    assert row['tags'] == ('severity', 'frequency', 'aggregate')


def test_recipes_frame_carries_the_entry_as_well_as_the_directory(uw):
    """One frame, not two: `program` / `spec` sit alongside the description.

    Ordered so the wide payload lands last -- a frame you cannot read at a
    terminal does not get used.
    """
    df = uw.recipes
    assert list(df.columns) == ['seq', 'note', 'tags', 'source', 'program',
                                'spec', 'as_read']
    row = df.loc[('agg', 'AF.Tags.Spaces')]
    assert row['program'].startswith('agg AF.Tags.Spaces')
    assert isinstance(row['spec'], dict) and row['spec']['note']


def test_recipes_frame_answers_what_is_undescribed(uw):
    """The question the frame exists for, now that there is only one tier."""
    df = uw.recipes
    assert len(df.query('not note')) > 0


def test_empty_recipe_base_still_gives_a_well_formed_frame():
    """A bare Underwriter loads nothing; the frame must still have its shape."""
    df = Underwriter().recipes
    assert df.empty
    assert df.index.names == ['kind', 'name']
    assert 'note' in df.columns and 'spec' in df.columns


# ----------------------------------------------------------------------
# Reading order and source text [Recipe-Seq-As-Read]
# ----------------------------------------------------------------------
# `library.agg` is written as a reading order and its sections build on one
# another, but the recipes frame sorts alphabetically and `program` is the
# flattened one-liner. `seq` carries the order, `as_read` carries the layout.
def test_seq_is_a_permutation_of_the_positions(lib):
    """Every entry has a distinct place, and the places are contiguous."""
    seqs = sorted(r.seq for r in lib._recipes.values())
    assert seqs == list(range(len(seqs)))


def test_sorting_by_seq_reproduces_the_file_order(lib):
    """The point of the field: `seq` order is `library.agg` order.

    Compared on names against a fresh split of the file, so this fails if the
    counter drifts from the read loop rather than merely being self-consistent.
    """
    from aggregate.parser import UnderwritingLexer
    path = next(p for p in lib.databases if p.name == 'library.agg')
    statements = UnderwritingLexer.preprocess(path.read_text(encoding='utf-8'))
    from_file = [lib.parser.parse(lib.lexer.tokenize(s))[1] for s in statements]
    by_seq = [r.name for r in sorted(lib._recipes.values(), key=lambda r: r.seq)]
    assert by_seq == from_file


def test_as_read_reparses_to_the_same_spec(lib):
    """The real contract: `as_read` is DecL, not a comment.

    It is deliberately NOT compared to `program` as text. `preprocess`
    reformats around brackets on its non-nested path, so a source `dfreq[1]`
    flattens to `dfreq [1]`, and which path it takes depends on whether the
    whole file holds a nested `[[...]]`. Spec equality is the invariant that
    survives that.
    """
    checked = 0
    for r in lib._recipes.values():
        assert r.as_read, f'{r.name}: a library entry must carry its source'
        kind, name, spec = lib.parser.parse(lib.lexer.tokenize(
            lib.lexer.preprocess(r.as_read)[0]))
        assert (kind, name) == (r.kind, r.name)
        assert repr(spec) == repr(r.spec), f'{r.name}: as_read changed the spec'
        checked += 1
    assert checked > 100


def test_as_read_keeps_the_layout_and_the_unexpanded_sugar(lib):
    """Multi-line, indented, and still showing what the author typed."""
    r = lib._recipes[('agg', 'DiceThreeEvenDice')]
    assert '\n' in r.as_read
    assert r.as_read.startswith('agg DiceThreeEvenDice')
    # as_read and the flattened program keep the sugar as written; only the
    # canonical rendering expands it. (The old form of this check asserted
    # 'dsev [2:12:2]' not in r.program, which held only because the
    # whole-file bracket-padding of preprocess step 3 respaced it to
    # 'dsev  [2:12:2]'; the library's nested dbvsev entries now route the
    # file down the depth-aware path, which keeps the spacing as typed.)
    assert 'dsev [2:12:2]' in r.as_read
    assert '[2:12:2]' in r.program
    assert '[2:12:2]' not in r.decl
    assert '[2 4 6 8 10 12]' in r.decl
    # comments and the terminating semicolon are not part of the statement
    assert '#' not in r.as_read
    assert not r.as_read.rstrip().endswith(';')


def test_a_session_build_has_no_source_text_and_sorts_last(lib):
    """No file to come from, and a place after everything already read."""
    u = lib.fork()
    top = max(r.seq for r in u._recipes.values())
    u('agg SeqSession.Mine 10 claims sev lognorm 100 cv 2 poisson')
    r = u._recipes[('agg', 'SeqSession.Mine')]
    assert r.as_read == ''
    assert r.seq > top


def test_rebuilding_an_existing_name_keeps_its_place(lib):
    """Reading order is stable: an overwrite is not a reordering."""
    u = lib.fork()
    before = u._recipes[('agg', 'DiceThreeEvenDice')].seq
    u('agg DiceThreeEvenDice dfreq [3] dsev [2 4 6 8 10 12]')
    assert u._recipes[('agg', 'DiceThreeEvenDice')].seq == before


def test_the_frame_carries_both_and_sorts_alphabetically_still(lib):
    """Additive: the default presentation is unchanged, `seq` is the opt-in."""
    df = lib.recipes
    assert list(df.index) == sorted(df.index)
    assert df['seq'].nunique() == len(df)
    by_seq = df.sort_values('seq')
    assert by_seq.index[0] != df.index[0] or len(df) == 1
    assert (by_seq['as_read'].str.len() > 0).all()


# ----------------------------------------------------------------------
# Detachment: nothing handed out of the recipe base shares mutable state
# ----------------------------------------------------------------------

def test_lookup_hands_back_a_detached_spec(uw):
    """The handed-back ``spec`` is a copy, not the stored dict.

    The sibling test above pins ``.object``, the shallow half of the same
    contract. This is the deep half, and it is the one that bit: building a
    ``pnl`` / ``xpnl`` **destructures** its spec, popping every P&L clause off
    so the remainder is a valid ``Aggregate(**spec)`` call, and while a
    ``dataclasses.replace`` copy shared the stored dict that popping stripped
    the library entry itself.
    """
    r = uw.recipe('AF.Tags.Spaces')
    stored_keys = set(uw.recipe('AF.Tags.Spaces').spec)
    assert r.spec is not uw.recipe('AF.Tags.Spaces').spec
    r.spec.pop('sev_name', None)
    r.spec['injected'] = 'not really a clause'
    assert set(uw.recipe('AF.Tags.Spaces').spec) == stored_keys


def test_detached_deep_copies_nested_spec_structure(lib):
    """A nested unit spec is copied too, not just the top-level dict.

    A ``port`` carries one spec per unit, so a shallow copy of the outer dict
    would still expose the inner ones.
    """
    r = lib.recipe('Capstone.PnL')
    other = lib.recipe('Capstone.PnL')
    assert r.spec is not other.spec
    for key, value in r.spec.items():
        if isinstance(value, (dict, list)):
            assert value is not other.spec[key], key


def test_premium_sentinels_survive_a_deep_copy():
    """``INHERIT_PREMIUM`` / ``DERIVE_PREMIUM`` copy to themselves.

    They are compared by identity in the consideration resolver and they ride
    on a parsed spec, which is now deep-copied on its way out of the recipe
    base. A cloning ``deepcopy`` broke the identity test and the resolver fell
    through to ``float(sentinel)``.
    """
    from aggregate.parser import DERIVE_PREMIUM, INHERIT_PREMIUM
    for sentinel in (INHERIT_PREMIUM, DERIVE_PREMIUM):
        assert copy.copy(sentinel) is sentinel
        assert copy.deepcopy(sentinel) is sentinel
        assert copy.deepcopy({'consideration': sentinel})['consideration'] is sentinel


# One entry per kind, plus both premium sentinels, which are the forms whose
# consideration is resolved after the engine is built.
TWICE_BUILT = [
    ('sev', 'Capstone.Sev'),
    ('agg', 'Capstone.XOL'),
    ('port', 'TwoLineBook'),
    ('pnl', 'PnLClaims'),           # plain numeric consideration
    ('pnl', 'PnLSimple'),           # inherit premium
    ('xpnl', 'PnLInherit'),         # inherit premium, tower
    ('xpnl', 'PnLDerive'),          # derive premium, tower
    ('xpnl', 'Capstone.PnL'),       # the entry the bug was found on
    ('distortion', 'PHDistortion'),
]


@pytest.mark.parametrize('kind,name', TWICE_BUILT,
                         ids=[f'{k}-{n}' for k, n in TWICE_BUILT])
def test_building_the_same_entry_twice_in_one_session(lib, kind, name):
    """A second build of a stored entry must behave like the first.

    The regression guard for the detachment contract, end to end. Until
    ``1.0.0a396`` a second build of any ``pnl`` / ``xpnl`` raised
    ``KeyError: 'consideration'``: the first build popped the P&L clauses off
    the shared spec, and ``consideration`` is the only mandatory one, so it is
    the only one popped without a default and the only one that announced
    itself. Every optional sibling (expenses, ledger labels, reinstatements,
    retro, the variable features) came back silently absent, which is why the
    crash was worth keeping rather than defaulting away.

    A live Jupyter kernel is where this bites: Quarto holds one across renders,
    so the first render passed and the second failed.
    """
    assert lib.recipe(name).kind == kind
    # The key SET, not the whole dict: a spec can carry numpy arrays (picks),
    # which make `==` ambiguous, and it is the key set that popping destroys.
    before = set(lib.recipe(name).spec)
    first = lib.build(name)
    second = lib.build(name)
    assert type(first) is type(second)
    # and the stored entry survived both builds unstripped
    assert set(lib.recipe(name).spec) == before
    if kind in ('pnl', 'xpnl'):
        assert 'consideration' in before
