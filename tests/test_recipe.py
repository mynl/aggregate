"""Tests for the recipe runtime -- parsing, running and auditing ``doc{{{...}}}``.

A recipe is a library entry's ``doc`` body parsed into Problem / Solution /
Discussion / Check. The Solution and Check share one namespace, so a recipe is
self-testing: the Check asserts against whatever the Solution built.

See ``aggregate.recipe`` and ``dev/plan-meta-data.md`` [Recipe-Library].
"""

import pytest

from aggregate import Underwriter
from aggregate.recipe import SECTIONS, Recipe, parse_doc

FULL = '''\
## Problem

You want a compound Poisson mean.

## Solution

```python
a = build('agg RecipeDemo 10 claims sev lognorm 50 cv 1 poisson')
```

## Discussion

E[A] = E[N] x E[X].

## Check

```python
assert abs(a.actual_m - 500) < 1e-9
assert abs(a.est_m - a.actual_m) < 1e-5
```
'''


# ----------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------
def test_parses_the_four_canonical_sections():
    r = parse_doc(FULL)
    assert r.sections == SECTIONS
    assert r.problem.startswith('You want')
    assert r.discussion.startswith('E[A]')
    assert 'build(' in r.solution_code
    assert r.n_asserts == 2


def test_headings_are_case_and_punctuation_tolerant():
    r = parse_doc('## solution:\n\n```python\nx = 1\n```\n\n## CHECK\n\n'
                  '```python\nassert x == 1\n```')
    assert r.sections == ('solution', 'check')
    assert r.n_asserts == 1


def test_empty_doc_gives_an_empty_recipe():
    r = parse_doc('')
    assert r.sections == ()
    assert not r.is_runnable
    assert r.n_asserts == 0


def test_unrecognised_content_is_preserved_not_dropped():
    """Preamble and non-canonical sections land in ``extra``, not in a section.

    Splitting on every level-2 heading (not just the canonical four) is what
    keeps a ``## References`` section from being glued onto ``## Problem``.
    """
    r = parse_doc('Some preamble.\n\n## Problem\n\nP.\n\n## References\n\n@Foo2020')
    assert r.problem == 'P.'
    assert 'Some preamble.' in r.extra
    assert '## References' in r.extra and '@Foo2020' in r.extra


def test_hash_hash_inside_code_is_not_a_heading():
    """A ``## note`` comment at line start inside a fence is code, not a section."""
    r = parse_doc('## Solution\n\n```python\n## a banner comment\nx = 1\n```')
    assert r.sections == ('solution',)
    assert '## a banner comment' in r.solution_code
    assert r.extra == ''


def test_h3_is_not_a_section_heading():
    r = parse_doc('## Problem\n\nP.\n\n### A subheading\n\nmore')
    assert r.problem.endswith('more')
    assert r.extra == ''


def test_repeated_heading_concatenates_rather_than_overwrites():
    r = parse_doc('## Check\n\n```python\nassert 1\n```\n\n'
                  '## Check\n\n```python\nassert 2\n```')
    assert r.n_asserts == 2


def test_multiple_fences_in_one_section_run_as_one_script():
    r = parse_doc('## Solution\n\n```python\nx = 1\n```\n\nthen\n\n```python\ny = x + 1\n```')
    assert r.solution_code.count('\n') >= 1
    ns = r.run()
    assert ns['y'] == 2


def test_non_python_fences_are_prose_not_code():
    """A ```text fence documents output; it must not be executed."""
    r = parse_doc('## Solution\n\n```text\nthis is not python\n```')
    assert r.solution_code == ''
    assert not r.is_runnable


# ----------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------
def test_check_runs_in_the_solutions_namespace():
    """This is what makes a recipe self-testing rather than self-describing."""
    ns = parse_doc(FULL).run()
    assert abs(ns['a'].actual_m - 500) < 1e-9


def test_a_false_check_fails_loudly():
    r = parse_doc('## Solution\n\n```python\nx = 1\n```\n\n'
                  '## Check\n\n```python\nassert x == 2\n```')
    with pytest.raises(AssertionError):
        r.run()


def test_failure_names_the_recipe_and_the_block():
    r = parse_doc('## Check\n\n```python\nraise ValueError("boom")\n```',
                  name='Demo')
    with pytest.raises(ValueError, match=r'Demo: check block failed'):
        r.run()


def test_namespace_is_seeded_with_the_house_verbs():
    ns = parse_doc('## Solution\n\n```python\npass\n```').namespace()
    assert {'build', 'qd', 'np', 'pd', 'aggregate'} <= set(ns)


def test_run_accepts_a_preseeded_namespace():
    ns = parse_doc('## Solution\n\n```python\ny = seed * 2\n```').namespace()
    ns['seed'] = 21
    assert parse_doc('## Solution\n\n```python\ny = seed * 2\n```').run(ns)['y'] == 42


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------
def test_markdown_reassembles_in_canonical_order():
    r = parse_doc('## Check\n\nC.\n\n## Problem\n\nP.')
    md = r.markdown()
    assert md.index('## Problem') < md.index('## Check')


def test_markdown_omits_absent_sections():
    assert parse_doc('## Problem\n\nP.').markdown() == '## Problem\n\nP.'


# ----------------------------------------------------------------------
# Underwriter surface
# ----------------------------------------------------------------------
@pytest.fixture(scope='module')
def uw():
    """An Underwriter over decl-testers.agg, which carries the AF fixtures."""
    u = Underwriter(databases='decl-testers')
    u.load()
    return u


def test_underwriter_recipe_resolves_by_name_alone(uw):
    r = uw.recipe('AF.Doc.Full')
    assert isinstance(r, Recipe)
    assert r.kind == 'agg'
    assert r.sections == SECTIONS
    assert r.tags == ('aggregate', 'intro', 'check:independent-oracle')
    # `program` is the source line, verbatim; `decl` the canonical doc-free
    # re-rendering. Both name the entry; only the first carries the doc.
    assert r.program.startswith('agg AF.Doc.Full')
    assert r.decl.startswith('agg AF.Doc.Full')


def test_underwriter_recipe_runs(uw):
    """The shipped fixture's own stated invariant holds."""
    ns = uw.recipe('AF.Doc.Full').run()
    assert 'a' in ns


# ----------------------------------------------------------------------
# One entry, one object ([Recipe-Is-The-Entry])
# ----------------------------------------------------------------------
def test_a_recipe_is_the_entry_not_a_second_view_of_it(uw):
    """``recipe(name)`` and ``uw[name]`` return the same thing.

    Before 1.0.0a164 these were different classes -- a ``ParsedProgram``
    holding the entry and a ``Recipe`` holding its parsed doc -- so "which do
    I use?" had no good answer. One class now carries both.
    """
    by_verb = uw.recipe('AF.Doc.Full')
    by_subscript = uw[('agg', 'AF.Doc.Full')]
    assert type(by_verb) is type(by_subscript) is Recipe
    assert by_verb.spec == by_subscript.spec
    assert by_verb.program == by_subscript.program
    # the entry half and the doc half are both present on each
    assert by_subscript.sections == SECTIONS
    assert by_verb.source == by_subscript.source


def test_lookup_hands_back_a_copy_so_the_store_cannot_be_mutated(uw):
    """``.object`` is set by the factory on the caller's copy, never the store."""
    r = uw.recipe('AF.Doc.Full')
    r.object = 'not really an Aggregate'
    assert uw.recipe('AF.Doc.Full').object is None
    assert uw[('agg', 'AF.Doc.Full')].object is None


def test_doc_metadata_is_derived_from_the_spec_not_duplicated(uw):
    """One copy of every fact: note/tags/hints/doc read straight off ``spec``."""
    r = uw.recipe('AF.Doc.Full')
    assert r.note == r.spec['note']
    assert r.tags == tuple(r.spec['tags'])
    assert r.doc == r.spec['doc']


def test_an_undocumented_recipe_costs_nothing_to_hold(uw):
    """Docs parse lazily -- most entries have none, and load() is on build's path."""
    r = uw.recipe('AF.Tags.Spaces')
    assert r._sections is None            # untouched by construction
    assert r.sections == ()
    assert r._sections is not None        # ... and cached after the first ask


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


#: A synthetic documented entry. Written here rather than read out of
#: ``library.agg``, which carries no ``doc{{{...}}}`` since 1.0.0a300: the
#: shipped library documents itself through notes in the presentations
#: repository, so the clause's own machinery needs its own fixture. Exact
#: discrete so ``run`` can assert on the moments to floating point.
_DOCUMENTED = (
    'agg DocDemo dfreq [3] dsev [1 2 3 4 5 6] '
    'note{three dice} tags{topic:aggregate} hints{log2=12}'
    "\n    doc{{{\n## Solution\n\n```python\na = build('''<<decl>>''')\n```\n"
    '\n## Check\n\n```python\nassert abs(a.actual_m - 10.5) < 1e-12\n```\n}}}')


@pytest.fixture
def documented():
    """An Underwriter carrying one synthetic documented entry."""
    uw = Underwriter()
    uw._interpret_program(_DOCUMENTED)
    return uw


def test_decl_placeholder_expands_to_the_entrys_own_program(documented):
    """``<<decl>>`` saves a recipe from retyping the program it documents.

    Two copies of the same declaration inside one statement would drift the
    first time either was edited -- the maintenance trap this removes.
    """
    r = documented.recipe('DocDemo')
    assert '<<decl>>' not in r.solution_code
    assert 'agg DocDemo' in r.solution_code
    assert "build('''" in r.solution_code


def test_expanded_decl_carries_hints_and_nothing_else(documented):
    """``Recipe.decl`` is the program, not the metadata around it.

    ``hints`` survives because it changes how the object *builds*: a
    copy-pasteable program without it would not reproduce the recipe. The note
    is the recipe's Problem, the tags are the page it sits on, and the doc is
    the page itself, so repeating any of them inside the program is redundant
    -- and dropping the doc is also the recursion guard, since a doc must not
    quote itself.

    Contrast ``.program``, the verbatim source line, which carries the lot.
    """
    r = documented.recipe('DocDemo')
    assert 'doc{{{' in r.program          # the source line has one ...
    for clause in ('doc{{{', 'note{', 'tags{'):
        assert clause not in r.decl       # ... the expansion has none of it
    # and the substitution inherits that, so none of it leaks into the code
    for clause in ('doc{{{', 'note{', 'tags{'):
        assert clause not in r.solution_code


def test_expanded_decl_keeps_hints_because_they_change_the_build(lib):
    """The one trailer clause that is part of the program, not about it."""
    from aggregate import Underwriter
    uw = Underwriter()
    uw._interpret_program(
        'agg HintedRecipe 5 claims sev lognorm 10 cv 2 poisson '
        'note{n} tags{topic:aggregate} hints{log2=12}'
        '\n    doc{{{\n## Solution\n\n```python\na = build(\'\'\'<<decl>>\'\'\')\n```\n}}}')
    r = uw.recipe('HintedRecipe')
    assert 'hints{log2=12}' in r.decl
    assert 'hints{log2=12}' in r.solution_code
    assert 'note{' not in r.decl and 'tags{' not in r.decl


def test_expanded_decl_actually_rebuilds_the_entry(documented):
    """The substituted program is real DecL that reproduces the object."""
    r = documented.recipe('DocDemo')
    ns = r.run()
    assert abs(ns['a'].actual_m - 10.5) < 1e-12


def test_placeholder_is_left_alone_without_a_program():
    """``parse_doc`` with no program leaves the placeholder visible.

    Better a visible ``<<decl>>`` than a silently empty Solution.
    """
    r = parse_doc('## Solution\n\n```python\nx = "<<decl>>"\n```')
    assert '<<decl>>' in r.solution_code


def test_underwriter_recipe_unknown_name_raises(uw):
    with pytest.raises(KeyError, match='no recipe named'):
        uw.recipe('DefinitelyNotAnEntry')


def test_undocumented_entry_yields_an_empty_recipe_not_an_error(uw):
    """Absence of a doc is a fact to audit, not an exception."""
    r = uw.recipe('AF.Tags.Spaces')
    assert r.sections == ()
    assert r.tags == ('severity', 'frequency', 'aggregate')


def test_recipes_frame_shape_and_flags(uw):
    df = uw.recipes
    assert df.index.names == ['kind', 'name']
    row = df.loc[('agg', 'AF.Doc.Full')]
    assert row['doc'] and row['note']
    assert row['problem'] and row['solution'] and row['discussion'] and row['check']
    assert row['n_asserts'] == 2
    # the tags-only fixtures are tagged but undocumented
    assert not df.loc[('agg', 'AF.Tags.Spaces'), 'doc']


def test_recipes_frame_carries_the_entry_as_well_as_the_audit(uw):
    """One frame, not two: `program` / `spec` sit alongside the doc flags.

    Ordered so the wide payload lands last -- an audit frame you cannot read
    at a terminal does not get used.
    """
    df = uw.recipes
    assert list(df.columns) == [
        'note', 'tags', 'doc', 'problem', 'solution', 'discussion', 'check',
        'n_asserts', 'source', 'program', 'spec']
    row = df.loc[('agg', 'AF.Doc.Full')]
    assert row['program'].startswith('agg AF.Doc.Full')
    assert isinstance(row['spec'], dict) and row['spec']['note']


def test_recipes_frame_supports_the_two_questions_it_exists_for(uw):
    """'What is undocumented?' and 'what is documented but unchecked?'"""
    df = uw.recipes
    assert len(df.query('not doc')) > 0
    unchecked = df.query('doc and n_asserts == 0')
    assert ('agg', 'AF.Doc.Full') not in set(unchecked.index)


def test_empty_recipe_base_still_gives_a_well_formed_frame():
    """A bare Underwriter loads nothing; the frame must still have its shape."""
    df = Underwriter().recipes
    assert df.empty
    assert df.index.names == ['kind', 'name']
    assert 'n_asserts' in df.columns and 'spec' in df.columns
