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
    assert r.program.startswith('agg AF.Doc.Full')


def test_underwriter_recipe_runs(uw):
    """The shipped fixture's own stated invariant holds."""
    ns = uw.recipe('AF.Doc.Full').run()
    assert 'a' in ns


@pytest.fixture(scope='module')
def lib():
    """A private Underwriter over ``library.agg``.

    Deliberately NOT the global ``build`` singleton: it is mutable shared
    state, and a test that reads it can be perturbed by any other test in the
    session that loads a database into it. Own your knowledge base.
    """
    u = Underwriter(databases='library')
    u.load()
    return u


def test_decl_placeholder_expands_to_the_entrys_own_program(lib):
    """``<<decl>>`` saves a recipe from retyping the program it documents.

    Two copies of the same declaration inside one statement would drift the
    first time either was edited -- the maintenance trap this removes.
    """
    r = lib.recipe('LimitProfile')
    assert '<<decl>>' not in r.solution_code
    assert 'agg LimitProfile' in r.solution_code
    assert "build('''" in r.solution_code


def test_expanded_decl_excludes_the_doc_but_keeps_note_tags_hints(lib):
    """The recursion guard: a doc must not quote itself.

    ``hints`` is kept deliberately -- it changes how the object builds, so a
    copy-pasteable program without it would not reproduce the recipe.
    """
    r = lib.recipe('LimitProfile')
    assert 'doc{{{' not in r.program
    assert 'note{' in r.program and 'tags{' in r.program
    # ... and the expansion inherits that, so no doc leaks into the code
    assert 'doc{{{' not in r.solution_code


def test_expanded_decl_actually_rebuilds_the_entry(lib):
    """The substituted program is real DecL that reproduces the object."""
    from aggregate import build
    r = lib.recipe('ThreeDice')
    ns = r.run()
    assert abs(ns['a'].actual_m - build('ThreeDice').actual_m) < 1e-12


def test_placeholder_is_left_alone_without_a_program():
    """``parse_doc`` with no program leaves the placeholder visible.

    Better a visible ``<<decl>>`` than a silently empty Solution.
    """
    r = parse_doc('## Solution\n\n```python\nx = "<<decl>>"\n```')
    assert '<<decl>>' in r.solution_code


def test_underwriter_recipe_unknown_name_raises(uw):
    with pytest.raises(KeyError, match='no knowledge entry'):
        uw.recipe('DefinitelyNotAnEntry')


def test_undocumented_entry_yields_an_empty_recipe_not_an_error(uw):
    """Absence of a doc is a fact to audit, not an exception."""
    r = uw.recipe('AF.Tags.Spaces')
    assert r.sections == ()
    assert r.tags == ('severity', 'frequency', 'aggregate')


def test_recipes_audit_frame_shape_and_flags(uw):
    df = uw.recipes
    assert df.index.names == ['kind', 'name']
    for col in ('tags', 'note', 'doc', 'problem', 'solution', 'discussion',
                'check', 'n_asserts', 'source'):
        assert col in df.columns
    row = df.loc[('agg', 'AF.Doc.Full')]
    assert row['doc'] and row['note']
    assert row['problem'] and row['solution'] and row['discussion'] and row['check']
    assert row['n_asserts'] == 2
    # the tags-only fixtures are tagged but undocumented
    assert not df.loc[('agg', 'AF.Tags.Spaces'), 'doc']


def test_recipes_audit_supports_the_two_questions_it_exists_for(uw):
    """'What is undocumented?' and 'what is documented but unchecked?'"""
    df = uw.recipes
    assert len(df.query('not doc')) > 0
    unchecked = df.query('doc and n_asserts == 0')
    assert ('agg', 'AF.Doc.Full') not in set(unchecked.index)
